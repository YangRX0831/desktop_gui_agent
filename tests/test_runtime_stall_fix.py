"""GENERIC RUNTIME STALL FIX 回归测试(纯非 GUI)。

FIX A:verify_completion 空 verdicts → 合法 UNKNOWN,StopIteration
不再逃逸。FIX B:任务级意外 Exception 终态化为 FAIL(不静默卡死),
BaseException 类控制流不被吞。
"""

import asyncio

import pytest
from agentscope.message import Msg

import agent.gui_agent as gui_agent_module
from agent.task_expectation import (
    TaskExpectation,
    extract_task_expectation,
    verify_completion,
)
from agent.task_manager import TaskManager
from tests.agent_test_support import make_semantic_agent

_CLICK = "Action: click(x=1, y=1)"


# ======================================================================
# FIX A — 空 verdicts 安全 UNKNOWN
# ======================================================================


def test_a1_empty_expectation_keeps_original_behavior() -> None:
    """完全空期望按现有语义不构成可验证任务(不进入完成判定)。"""
    expectation = extract_task_expectation("随便看看桌面")
    assert expectation.is_empty() is True


def test_a2_delivery_only_expectation_unknown_no_exception() -> None:
    """delivery-only 期望:五类传统判定器都不适用 → UNKNOWN,零异常。"""
    expectation = TaskExpectation(
        delivery_intent=True,
        expected_delivery_payload="会议已结束",
    )
    result = verify_completion(expectation, {"foreground_process": "x.exe"})
    assert result.status == "UNKNOWN"
    assert result.reason == "no_applicable_verdicts"


def test_a3_any_empty_verdicts_unknown() -> None:
    """空期望走既有早退分支(原行为保持),任何输入都不抛 StopIteration。"""
    result = verify_completion(TaskExpectation(), {})
    assert result.status == "UNKNOWN"
    assert result.reason == "no structured expectation available for this task"
    # 修复的直接目标路径:非空期望但无适用判定器(delivery-only 等)。
    future_like = TaskExpectation(delivery_intent=True, expected_delivery_payload="x")
    assert verify_completion(future_like, {}).status == "UNKNOWN"


def test_a4_verified_verdict_unchanged() -> None:
    result = verify_completion(
        TaskExpectation(expected_volume=20),
        {"current_volume": 22},
    )
    assert result.status == "VERIFIED"


def test_a5_not_verified_verdict_unchanged() -> None:
    result = verify_completion(
        TaskExpectation(expected_volume=20),
        {"current_volume": 80},
    )
    assert result.status == "NOT_VERIFIED"


def test_a6_unknown_precedence_unchanged() -> None:
    result = verify_completion(
        TaskExpectation(expected_volume=20),
        {"current_volume": "not-an-int"},
    )
    assert result.status == "UNKNOWN"
    assert result.reason != "no_applicable_verdicts"


def _delivery_boxes_call3(state):
    """按 detailed-perceive 调用序给出:composer→composer+发送→content。

    几何位于测试 fake 截图(1000x500)范围内。
    """

    def fake(recognizer, image, focus_point=None):
        idx = state["i"]
        state["i"] += 1
        if idx == 0:
            return (), (
                {"text": "hello", "bbox": (400, 400, 500, 430), "confidence": 0.99},
            )
        if idx == 1:
            return (
                (),
                (
                    {"text": "hello", "bbox": (400, 400, 500, 430), "confidence": 0.99},
                    {"text": "发送", "bbox": (900, 300, 950, 320), "confidence": 0.99},
                ),
            )
        return (), (
            {"text": "hello", "bbox": (380, 100, 480, 130), "confidence": 0.99},
        )

    return fake


def _delivery_agent(monkeypatch, state):
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "perceive_ocr_elements_detailed",
        _delivery_boxes_call3(state),
        raising=True,
    )
    # 前台身份:首调(run 起点捕获 agent_ui)返回 777,其后业务调用
    # 返回 999(目标应用),避免 protected-foreground 误拒 type 动作。
    fg = {"first": True}

    def fake_foreground():
        if fg["first"]:
            fg["first"] = False
            return 777
        return 999

    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        fake_foreground,
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "is_window_available",
        lambda hwnd: True,
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "focus_control_state",
        lambda: "text_input",
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "perceive_windows",
        lambda size, offset: (),
        raising=True,
    )
    agent = make_semantic_agent(
        [
            'Action: type(text="hello")',
            "Action: click(x=925, y=310)",
            _CLICK,
        ],
        max_steps=3,
    )
    agent._run_expectation = TaskExpectation(
        delivery_intent=True,
        expected_delivery_payload="hello",
    )
    return agent


def test_a7_delivery_only_strong_evidence_pipeline_verified(monkeypatch) -> None:
    """standard UNKNOWN + delivery 强证据 → 完整 proactive 管线 VERIFIED。"""
    captured: dict = {}
    original_emit = gui_agent_module.GuiAgent._emit_completion_decision

    def spy_emit(self, trigger, step_number, status, reason):
        captured["trigger"] = trigger
        captured["status"] = status
        captured["reason"] = reason
        captured["evidence"] = dict(self._last_delivery_evidence or {})
        return original_emit(self, trigger, step_number, status, reason)

    monkeypatch.setattr(
        gui_agent_module.GuiAgent, "_emit_completion_decision", spy_emit
    )
    agent = _delivery_agent(monkeypatch, {"i": 0})
    result = asyncio.run(agent(Msg("u", "给 user1 发送消息“hello”", "user")))
    assert "任务完成" in result.content
    assert agent._dependencies.model_client.calls == 2
    assert captured["trigger"] == "post_submit_delivery"
    assert captured["status"] == "VERIFIED"
    assert captured["reason"] == "payload_moved_from_composer_to_new_region"
    evidence = captured["evidence"]
    assert evidence.get("composer_cleared") is True
    assert evidence.get("delivered_bbox") == (380, 100, 480, 130)


def test_a8_delivery_only_insufficient_no_finish(monkeypatch) -> None:
    """standard UNKNOWN + delivery 证据不足 → 不完成,按 max_steps 失败。"""

    def fake(recognizer, image, focus_point=None):
        # 两次有感知的点都保持 payload 在 composer(未投递)。
        return (), (
            {"text": "hello", "bbox": (400, 400, 500, 430), "confidence": 0.99},
            {"text": "发送", "bbox": (900, 300, 950, 320), "confidence": 0.99},
        )

    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "perceive_ocr_elements_detailed",
        fake,
        raising=True,
    )
    fg = {"first": True}

    def fake_foreground():
        if fg["first"]:
            fg["first"] = False
            return 777
        return 999

    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        fake_foreground,
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "is_window_available",
        lambda hwnd: True,
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "focus_control_state",
        lambda: "text_input",
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "perceive_windows",
        lambda size, offset: (),
        raising=True,
    )
    agent = make_semantic_agent(
        [
            'Action: type(text="hello")',
            "Action: click(x=925, y=310)",
            _CLICK,
        ],
        max_steps=3,
    )
    agent._run_expectation = TaskExpectation(
        delivery_intent=True,
        expected_delivery_payload="hello",
    )
    result = asyncio.run(agent(Msg("u", "给 user1 发送消息“hello”", "user")))
    assert "任务执行失败" in result.content
    # 关键回归:全程无 StopIteration 逃逸(修复前此处会楔死)。


# ======================================================================
# FIX B — 任务级意外异常终态化
# ======================================================================


def _run_with_raising_helper(monkeypatch, exc_factory):
    def bomb(*args, **kwargs):
        raise exc_factory()

    monkeypatch.setattr(
        gui_agent_module.GuiAgent,
        "_proactive_completion_check",
        bomb,
        raising=True,
    )
    agent = make_semantic_agent([_CLICK], max_steps=2)
    agent._run_expectation = TaskExpectation(expected_volume=20)
    return asyncio.run(agent(Msg("u", "把音量调整到20%", "user")))


def test_b1_value_error_terminal_fail(monkeypatch, caplog) -> None:
    result = _run_with_raising_helper(monkeypatch, lambda: ValueError("boom"))
    assert "任务执行失败" in result.content
    assert "ValueError" in result.content
    assert any(
        "gui_agent_task_unexpected_exception" in r.getMessage() for r in caplog.records
    )


def test_b2_runtime_error_terminal_fail(monkeypatch) -> None:
    result = _run_with_raising_helper(
        monkeypatch,
        lambda: RuntimeError("generator raised StopIteration"),
    )
    assert "任务执行失败" in result.content
    assert "RuntimeError" in result.content


def test_b3_stop_iteration_terminal_fail(monkeypatch) -> None:
    """StopIteration(及其 async/generator 转换为 RuntimeError 的形态)
    都被安全网终态化,不再静默卡死。"""
    result = _run_with_raising_helper(monkeypatch, StopIteration)
    assert "任务执行失败" in result.content
    assert "内部异常" in result.content
    # 同步上下文直接验证 StopIteration 本体也能终态化。
    agent = make_semantic_agent([_CLICK], max_steps=1)
    manager = TaskManager("任务")
    manager.start()
    message = agent._terminalize_unexpected_exception(StopIteration(), manager)
    assert "StopIteration" in message.content


def test_b4_normal_execution_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "system_volume_state",
        lambda: 20,
        raising=True,
    )
    agent = make_semantic_agent([_CLICK], max_steps=2)
    agent._run_expectation = TaskExpectation(expected_volume=20)
    result = asyncio.run(agent(Msg("u", "把音量调整到20%", "user")))
    assert "任务完成" in result.content


def test_b5_terminal_failure_logged_once(monkeypatch, caplog) -> None:
    _run_with_raising_helper(monkeypatch, lambda: ValueError("x"))
    failed = [
        r.getMessage()
        for r in caplog.records
        if "gui_agent_task_failed" in r.getMessage()
    ]
    assert len(failed) == 1


def test_b6_cleanup_finally_executed(monkeypatch) -> None:
    agent_holder = {}

    original = gui_agent_module.GuiAgent._run_task_inner

    async def spy(self, task, manager):
        agent_holder["agent"] = self
        return await original(self, task, manager)

    monkeypatch.setattr(gui_agent_module.GuiAgent, "_run_task_inner", spy, raising=True)
    result = _run_with_raising_helper(monkeypatch, lambda: ValueError("x"))
    assert "任务执行失败" in result.content
    agent = agent_holder["agent"]
    # reply 的 finally 清理在异常终态化后照常执行。
    assert agent._run_expectation is None
    assert agent._current_step == 0
    assert agent._current_run_id == ""


def test_b7_exception_stack_recorded(monkeypatch, caplog) -> None:
    _run_with_raising_helper(monkeypatch, lambda: ValueError("x"))
    records = [
        r.getMessage()
        for r in caplog.records
        if "gui_agent_task_unexpected_exception" in r.getMessage()
    ]
    assert records and "stack=" in records[0] and "test_runtime_stall_fix" in records[0]


def test_b8_base_exception_not_swallowed(monkeypatch) -> None:
    """KeyboardInterrupt/SystemExit 不被转成普通 task FAIL。"""
    with pytest.raises((KeyboardInterrupt, SystemExit)):
        _run_with_raising_helper(monkeypatch, KeyboardInterrupt)
