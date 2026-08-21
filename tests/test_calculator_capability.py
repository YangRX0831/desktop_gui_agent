"""Calculator keyboard-expression capability 的语法、安全与接线测试。"""

import asyncio

import pytest
from agentscope.message import Msg

from agent.semantic_routes import (
    build_calculator_expression_route,
    calculator_foreground_is_reliable,
    extract_calculator_expression,
    normalize_calculator_expression,
)
from agent.task_manager import TaskManager
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1+1", "1+1"),
        ("12.5 × (3 - 1)", "12.5*(3-1)"),
        ("(8+2)÷5", "(8+2)/5"),
        ("1+2*3-4/2", "1+2*3-4/2"),
    ],
)
def test_calculator_expression_normalization(
    source: str,
    expected: str,
) -> None:
    """白名单表达式只做空格与键盘运算符表示归一化。"""
    assert normalize_calculator_expression(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "1",
        "1+",
        "+1",
        "1++2",
        "1(2+3)",
        "(1+2",
        "1+2)",
        "1..2+3",
        "1e3+2",
        "1+__import__('os')",
        "1+2;dir",
        "(((((((((1+2)))))))))",
        "1+" + "2" * 64,
    ],
)
def test_calculator_expression_rejects_unsafe_or_ambiguous_syntax(
    source: str,
) -> None:
    """代码、命令、残缺语法、隐式乘法和超限输入全部拒绝。"""
    assert normalize_calculator_expression(source) is None


def test_extract_calculator_expression_is_task_generic() -> None:
    """抽取依赖计算语义而非 task ID、固定算式或 benchmark marker。"""
    assert (
        extract_calculator_expression("请打开计算器，计算12.5×(3-1)，并保留结果。")
        == "12.5*(3-1)"
    )
    assert extract_calculator_expression("calculate 7/2.") == "7/2"
    assert extract_calculator_expression("打开计算器查看历史记录。") is None
    assert extract_calculator_expression("计算1+1;启动shell") is None


def test_calculator_route_uses_only_type_and_hotkey() -> None:
    """能力路线不求值，只依次键入表达式并提交。"""
    route = build_calculator_expression_route("23+(4*5)")

    first = route.next_step()
    second = route.next_step()

    assert first is not None
    assert first.action == {
        "action_type": "type",
        "params": {"text": "23+(4*5)"},
    }
    assert second is not None
    assert second.action == {
        "action_type": "hotkey",
        "params": {"keys": ("enter",)},
    }
    assert route.is_exhausted


def test_calculator_foreground_identity_fails_closed() -> None:
    """直接进程可确认；通用 UWP 宿主必须同时有 Calculator 标题证据。"""
    assert calculator_foreground_is_reliable("CalculatorApp.exe", False)
    assert calculator_foreground_is_reliable("ApplicationFrameHost.exe", True)
    assert not calculator_foreground_is_reliable("ApplicationFrameHost.exe", False)
    assert not calculator_foreground_is_reliable("SearchHost.exe", True)


def test_agent_activates_calculator_capability_once(monkeypatch) -> None:
    """可靠 Calculator 前台触发一次 type→Enter，不调用模型或重复提交。"""
    monkeypatch.setattr(
        "agent.gui_agent.get_foreground_app_hwnd",
        lambda: 101,
        raising=True,
    )
    monkeypatch.setattr(
        "agent.gui_agent.process_name_of_hwnd",
        lambda hwnd: "CalculatorApp.exe" if hwnd == 101 else "",
        raising=True,
    )
    monkeypatch.setattr(
        "agent.gui_agent.is_window_available",
        lambda hwnd: hwnd == 999,
        raising=True,
    )
    controls = MemoryControls()
    backend = SequenceBackend(['Action: finish(result="unused")'])
    agent = make_agent(
        backend,
        controls,
        TaskManager("计算器任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
    )

    asyncio.run(
        agent(
            Msg(
                "u",
                "打开系统计算器，计算23+(4*5)，并让结果保留。",
                "user",
                metadata={"agent_ui_window_hwnd": 999},
            ),
        ),
    )

    assert controls.calls == [
        ("type", "23+(4*5)"),
        ("hotkey", "enter"),
    ]
    assert backend.calls == 0
