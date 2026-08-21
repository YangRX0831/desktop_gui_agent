"""PHASE 2C PROACTIVE COMPLETION DETECTION 测试。

核心合同:动作后(UI 稳定)主动三态验证;仅 VERIFIED 提前成功且不增加
模型调用;数值/文本类的 post-action 感知缓存复用,不重复 OCR;max_steps
兜底终检;窗口关闭类由已验证 last_effect 提供 capability 证据。
"""

import asyncio
from dataclasses import replace

from agentscope.message import Msg

from agent import gui_agent as gui_agent_module
from agent.action_parser import ActionPromptState
from agent.task_expectation import (
    ANY_TRACKED_WINDOW,
    TaskExpectation,
    extract_task_expectation,
    verify_completion,
)
from agent.task_manager import TaskManager
from tests.agent_test_support import (
    MemoryControls,
    SequenceBackend,
    make_agent,
    make_semantic_agent,
)

_CLICK = "Action: click(x=1, y=1)"
_FINISH = 'Action: finish(result="完成")'


def _patch_facts(monkeypatch, volume=None, ocr_lines=(), foreground="explorer.exe"):
    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.system_volume_state",
        lambda: volume,
        raising=True,
    )
    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_ocr_elements_detailed",
        lambda recognizer, image, focus_point=None: (tuple(ocr_lines), ()),
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        lambda: 777,
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "get_window_process_name",
        lambda hwnd: foreground,
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "is_window_available",
        lambda hwnd: True,
        raising=True,
    )


class _NullRecognizer:
    """最小 OCR fake:满足构造期 callable 校验,识别结果为空。"""

    def recognize(self, image):
        return []


def test_post_action_verified_ends_without_more_model_calls(monkeypatch) -> None:
    """音量入容量 → 动作后立即成功,不再调用模型。"""
    agent = make_semantic_agent([_CLICK])
    agent._run_expectation = TaskExpectation(expected_volume=20)
    _patch_facts(monkeypatch, volume=22)
    result = asyncio.run(agent(Msg("u", "将音量调整到大约20%", "user")))
    assert "任务完成(程序化验证)" in result.content
    assert agent._dependencies.model_client.calls == 1


def test_post_action_not_verified_continues_loop(monkeypatch) -> None:
    """音量未达标 → proactive NOT_VERIFIED → 循环继续(第二次调用发生)。

    模型随后提出的 finish 也因同一程序证据被 Phase 2A 验证拒绝,
    任务最终按 max_steps 失败——继续执行而非提前误判成功。
    """
    agent = make_semantic_agent([_CLICK, _FINISH], max_steps=2)
    agent._run_expectation = TaskExpectation(expected_volume=20)
    _patch_facts(monkeypatch, volume=50)
    result = asyncio.run(agent(Msg("u", "将音量调整到大约20%", "user")))
    assert agent._dependencies.model_client.calls == 2
    assert "任务执行失败" in result.content


def test_post_action_unknown_continues_loop(monkeypatch) -> None:
    """无结构化期望 → UNKNOWN → 保持常规循环。"""
    agent = make_semantic_agent([_CLICK, _FINISH], max_steps=2)
    agent._run_expectation = TaskExpectation()
    _patch_facts(monkeypatch, volume=None)
    result = asyncio.run(agent(Msg("u", "随便一个任务", "user")))
    assert result.content == "完成"
    assert agent._dependencies.model_client.calls == 2


def test_numeric_result_proactive_success(monkeypatch) -> None:
    """post-action OCR 含期望数值 → 提前成功。"""
    agent = make_semantic_agent([_CLICK])
    agent._run_expectation = TaskExpectation(expected_numeric_result=2)
    _patch_facts(
        monkeypatch,
        ocr_lines=('text="2" bbox=(100, 100, 130, 130) confidence=1.00',),
        foreground="CalculatorApp.exe",
    )
    monkeypatch.setattr(agent, "_ocr_target_window_text", lambda *args: "1 + 1 = 2")
    result = asyncio.run(agent(Msg("u", "打开计算器，计算1+1", "user")))
    assert "任务完成(程序化验证)" in result.content
    assert agent._dependencies.model_client.calls == 1


def test_numeric_result_requires_calculator_foreground(monkeypatch) -> None:
    """无关应用中的相同数字不得触发计算任务提前完成。"""
    agent = make_semantic_agent([_CLICK, _FINISH])
    _patch_facts(
        monkeypatch,
        ocr_lines=('text="2" bbox=(100, 100, 130, 130) confidence=1.00',),
        foreground="WindowsTerminal.exe",
    )
    result = asyncio.run(agent(Msg("u", "打开计算器，计算1+1", "user")))
    assert "任务完成(程序化验证)" not in result.content
    assert agent._dependencies.model_client.calls == 3


def test_volume_proactive_success(monkeypatch) -> None:
    """音量期望的 proactive 路径(60±5)。"""
    agent = make_semantic_agent([_CLICK])
    agent._run_expectation = TaskExpectation(expected_volume=60)
    _patch_facts(monkeypatch, volume=64)
    result = asyncio.run(agent(Msg("u", "将音量调整到大约60%", "user")))
    assert "任务完成(程序化验证)" in result.content


def test_window_closed_proactive_success() -> None:
    """窗口关闭类:已验证 last_effect=window_closed → VERIFIED。"""
    agent = make_semantic_agent([_CLICK])
    expectation = extract_task_expectation(
        "关闭桌面上那个测试专用的空白记事本窗口，不要关闭其他窗口。",
    )
    assert expectation.expected_window_closed == ANY_TRACKED_WINDOW
    agent._run_expectation = expectation
    state = replace(
        ActionPromptState(step_number=1, max_steps=10),
        last_effect="window_closed",
    )
    facts = agent._completion_facts(state, "")
    assert facts["tracked_window_exists"] is False
    assert verify_completion(expectation, facts).status == "VERIFIED"
    # 关闭尚未发生时不得误判成功。
    state_pending = replace(
        ActionPromptState(step_number=1, max_steps=10),
        last_effect="none",
    )
    facts_pending = agent._completion_facts(state_pending, "")
    assert verify_completion(expectation, facts_pending).status == "NOT_VERIFIED"


def test_window_completion_uses_exact_bound_target(monkeypatch) -> None:
    """其他窗口关闭不能替代 metadata 绑定的目标窗口关闭。"""
    agent = make_semantic_agent([_CLICK])
    agent._run_expectation = TaskExpectation(expected_window_closed=123)
    agent._task_target = {"hwnd": 123, "process": "notepad.exe"}
    monkeypatch.setattr(gui_agent_module, "is_window_existing", lambda hwnd: True)
    state = replace(
        ActionPromptState(step_number=1, max_steps=10),
        last_effect="window_closed",
    )
    facts = agent._completion_facts(state, "")
    assert facts["tracked_window_exists"] is True
    assert verify_completion(agent._run_expectation, facts).status == "NOT_VERIFIED"


def test_calculator_completion_uses_result_band_not_keypad(monkeypatch) -> None:
    """数字键按钮中的目标字符不得替代计算器结果带证据。"""
    agent = make_semantic_agent([_CLICK])
    agent._run_expectation = extract_task_expectation(
        "打开系统计算器，计算1+1，并让最终计算结果保留在计算器界面。",
    )
    monkeypatch.setattr(gui_agent_module, "get_foreground_app_hwnd", lambda: 777)
    monkeypatch.setattr(
        gui_agent_module,
        "get_window_process_name",
        lambda hwnd: "ApplicationFrameHost.exe",
    )
    monkeypatch.setattr(agent, "_window_title_matches", lambda hwnd, words: True)
    monkeypatch.setattr(agent, "_ocr_target_window_text", lambda *args: "")
    assert (
        agent._verify_completion_proposal(
            ActionPromptState(step_number=1, max_steps=10),
        ).status
        == "UNKNOWN"
    )
    monkeypatch.setattr(agent, "_ocr_target_window_text", lambda *args: "1 + 1 = 2")
    assert (
        agent._verify_completion_proposal(
            ActionPromptState(step_number=1, max_steps=10),
        ).status
        == "VERIFIED"
    )


def test_numeric_finish_unknown_is_rejected(monkeypatch) -> None:
    """数值计算缺少最终结果证据时，模型自述完成也不得通过。"""
    agent = make_semantic_agent([_CLICK, _FINISH], max_steps=2)
    _patch_facts(monkeypatch, foreground="CalculatorApp.exe")
    monkeypatch.setattr(agent, "_ocr_target_window_text", lambda *args: "1 + 2")
    result = asyncio.run(agent(Msg("u", "打开计算器，计算1+1", "user")))
    assert "任务执行失败" in result.content


def test_text_proactive_success() -> None:
    """文本期望的 proactive 事实路径。"""
    agent = make_semantic_agent([_CLICK])
    agent._run_expectation = TaskExpectation(expected_text="Hello World")
    state = ActionPromptState(step_number=1, max_steps=10)
    facts = agent._completion_facts(
        state,
        "Hello World bbox=(1, 2, 3, 4)",
    )
    assert verify_completion(agent._run_expectation, facts).status == "VERIFIED"


def test_pending_perception_reused_no_double_ocr(monkeypatch) -> None:
    """proactive OCR 后未完成 → 同一份感知供下一次模型调用,OCR 仅一次。"""
    _patch_facts(monkeypatch, volume=None, ocr_lines=())
    ocr_calls = []

    def fake_ocr(recognizer, image, focus_point=None):
        ocr_calls.append(1)
        return ((), ())

    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_ocr_elements_detailed",
        fake_ocr,
        raising=True,
    )
    agent = make_semantic_agent(
        ["Action: click(x=1, y=1)", "Action: click(x=2, y=2)"],
        max_steps=2,
    )
    agent._run_expectation = TaskExpectation(expected_numeric_result=99)
    result = asyncio.run(agent(Msg("u", "计算88+11", "user")))
    # 每个感知点恰好一次 OCR:初始 1 次 + 两次 post-action 各 1 次 = 3;
    # 第二次模型调用复用 proactive 缓存(若重复感知会是 4 次)。
    assert agent._dependencies.model_client.calls == 2
    assert len(ocr_calls) == 3
    assert "任务执行失败" in result.content


def test_no_extra_llm_calls_from_proactive(monkeypatch) -> None:
    """proactive 机制本身零额外模型调用(VERIFIED 提前结束)。"""
    agent = make_semantic_agent([_CLICK])
    agent._run_expectation = TaskExpectation(expected_volume=20)
    _patch_facts(monkeypatch, volume=20)
    asyncio.run(agent(Msg("u", "将音量调整到大约20%", "user")))
    assert agent._dependencies.model_client.calls == 1


def test_max_steps_final_check_saves_completed_task(monkeypatch) -> None:
    """max_steps 兜底:proactive 均未证实但终态验证完成 → 成功收尾。"""
    _patch_facts(monkeypatch, volume=None, ocr_lines=())

    # proactive 感知点走 detailed 版本(P1 后主循环保留 box 通道),
    # 这里保持无证据;终态 _verify_completion_proposal 仍用
    # perceive_ocr_elements,由下方 fake 提供数值证据。
    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_ocr_elements_detailed",
        lambda recognizer, image, focus_point=None: ((), ()),
        raising=True,
    )

    def fake_ocr(recognizer, image, focus_point=None):
        return ('text="99" bbox=(100, 100, 130, 130) confidence=1.00',)

    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_ocr_elements",
        fake_ocr,
        raising=True,
    )
    agent = make_semantic_agent(
        ["Action: click(x=1, y=1)", "Action: click(x=2, y=2)"],
        max_steps=2,
    )
    # 终态验证的 OCR 依赖真实 recognizer 存在;注入最小 fake 使
    # _verify_completion_proposal 走到被 patch 的感知函数。
    agent._dependencies = replace(
        agent._dependencies,
        ocr_recognizer=_NullRecognizer(),
    )
    agent._run_expectation = TaskExpectation(expected_numeric_result=99)
    result = asyncio.run(agent(Msg("u", "计算88+11", "user")))
    assert "任务完成(程序化验证)" in result.content
    assert agent._dependencies.model_client.calls == 2


def test_proactive_decision_trace_record(monkeypatch) -> None:
    """proactive 提前结束写入 completion_decision(trigger)记录。"""
    recorded = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    backend = SequenceBackend([_CLICK])
    agent = make_agent(
        backend,
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
        trace_writer=_TraceWriter(),
    )
    agent._run_expectation = TaskExpectation(expected_volume=20)
    _patch_facts(monkeypatch, volume=20)
    asyncio.run(agent(Msg("u", "将音量调整到大约20%", "user")))
    decision = next(
        r for r in recorded if r.get("record_type") == "completion_decision"
    )
    assert decision["completion_trigger"] == "proactive_post_action"
    assert decision["completion_verification"] == "VERIFIED"


def test_finish_proposal_verifier_still_works(monkeypatch) -> None:
    """Phase 2A finish 提案验证保持工作并携带 trigger 标记。"""
    recorded = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    agent = make_agent(
        SequenceBackend([_FINISH, _FINISH]),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
        trace_writer=_TraceWriter(),
    )
    agent._run_expectation = TaskExpectation(expected_volume=20)
    _patch_facts(monkeypatch, volume=50)
    result = asyncio.run(agent(Msg("u", "将音量调整到大约20%", "user")))
    assert "任务执行失败" in result.content
    decision = next(r for r in recorded if r.get("record_type") == "finish_decision")
    assert decision["completion_trigger"] == "model_finish"
    assert decision["completion_verification"] == "NOT_VERIFIED"
