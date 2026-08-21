"""LOCAL Compact Prompt(键盘优先)与 param_wrap 修复测试。

合同:compact 仅在 model_mode=local 且 V3 协议时使用;API Clean V3
路径与静态 Prompt 完全不变;语法行来自 action_parser 真实 grammar;
param_wrap 仅在签名完全一致时执行,值原样保留。
"""

import asyncio

from agentscope.message import Msg

from agent.action_parser import (
    ACTION_SYSTEM_PROMPT,
    CANONICAL_V3_ACTIONS,
    action_grammar_lines,
    action_param_signature,
    parse_action,
)
from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3
from agent.local_compact_prompt import (
    LOCAL_ACTION_SYSTEM_PROMPT,
    compose_local_compact_prompt,
)
from agent.local_output_repair import repair_local_output
from agent.task_manager import TaskManager
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent


def _state(**overrides):
    from agent.action_parser import ActionPromptState

    base = dict(
        step_number=2,
        max_steps=10,
        foreground_after="explorer.exe",
        keyboard_input_ready="unknown",
        focused_control="other",
        last_action="none",
        last_effect="none",
        platform="windows",
        steps_remaining=8,
    )
    base.update(overrides)
    return ActionPromptState(**base)


def test_local_uses_compact_prompt_and_api_keeps_v3() -> None:
    """local+V3 → compact;api+V3 → Clean V3 动态块。"""
    local_agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager("打开记事本"),
        max_steps=1,
        retry_count=0,
        model_mode="local",
        reject_initial_finish=False,
        decision_protocol_v3=True,
    )
    asyncio.run(local_agent(Msg("u", "打开记事本", "user")))
    local_prompt = local_agent._dependencies.model_client.prompts[0]
    assert local_prompt.startswith(LOCAL_ACTION_SYSTEM_PROMPT[:40])
    assert "键盘优先策略" in local_prompt
    assert "Interactive elements" not in local_prompt
    assert "Current perception" not in local_prompt

    api_agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager("打开记事本"),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
    )
    asyncio.run(api_agent(Msg("u", "打开记事本", "user")))
    api_prompt = api_agent._dependencies.model_client.prompts[0]
    assert "Current execution state:" in api_prompt
    assert "键盘优先策略" not in api_prompt


def test_optimized_v3_prompt_stays_concise_relative_to_v1() -> None:
    """V3 保留规划信息但不以脆弱的固定字符数作为合同。"""
    assert 1000 < len(ACTION_SYSTEM_PROMPT_V3) < len(ACTION_SYSTEM_PROMPT) * 0.75


def test_compact_contains_policy_not_task_specific() -> None:
    """包含 keyboard-first policy;不含任务特判。"""
    text = compose_local_compact_prompt("任意任务", _state(), "normalized_1000")
    for rule in (
        "优先使用键盘",
        "启动应用优先",
        "标准窗口关闭快捷键",
        "keyboard_input_ready=false",
        "不要猜屏幕坐标",
    ):
        assert rule in text
    for banned in (
        "S04",
        "S06",
        "S02",
        "Chrome",
        "chrome",
        "记事本路径",
        "Alt+F4",
        "alt+f4",
        "notepad",
        "Win+R",
    ):
        assert banned not in text, banned


def test_compact_dynamic_context_shorter_than_full_v3() -> None:
    """compact 动态部分明显短于完整 V3 local payload。"""
    state = _state(
        ocr_elements=tuple(
            f'text="条目{i}" bbox=({i}, {i}, {i + 5}, {i + 5}) confidence=0.99'
            for i in range(20)
        ),
        windows=("id:1 fg=false process=app1.exe bbox=(0,0,100,100)",),
    )
    compact = compose_local_compact_prompt("任务", state, "normalized_1000")
    from agent.action_parser import compose_action_dynamic_prompt

    full_dynamic = compose_action_dynamic_prompt("任务", state, "normalized_1000")
    full_local = f"{ACTION_SYSTEM_PROMPT_V3}\n\n{full_dynamic}"
    assert len(compact) < len(full_local) * 0.75
    assert compact.count("条目") <= 5  # OCR 截断到 5 条
    assert "windows" not in compact


def test_grammar_lines_derived_from_parser() -> None:
    """语法行来自 parser 真实分组;签名与 grammar 一致。"""
    lines = action_grammar_lines()
    assert len(lines) == 8
    names = tuple(line.split(". ", 1)[1].split("(", 1)[0] for line in lines)
    assert names == CANONICAL_V3_ACTIONS
    for line in lines:
        assert line in LOCAL_ACTION_SYSTEM_PROMPT
    assert lines[0] == "1. click(x=<整数>, y=<整数>) - 单击可见控件"
    assert "hotkey(" in lines[6]
    assert action_param_signature("click") == ("x", "y")
    assert action_param_signature("drag") == ("x1", "y1", "x2", "y2")
    assert action_param_signature("unknown") is None


def test_param_wrap_full_named_click_repaired() -> None:
    """完整 named 参数 → 包装修复成功。"""
    repaired, repairs = repair_local_output("Action: click, x=100, y=200")
    assert repaired == "Action: click(x=100, y=200)"
    assert "param_wrap" in repairs
    assert parse_action(repaired) is not None


def test_param_wrap_missing_param_rejected() -> None:
    """缺 y → 不修,保持失败。"""
    repaired, _ = repair_local_output("Action: click, x=100")
    assert repaired == "Action: click, x=100"
    assert parse_action(repaired) is None


def test_param_wrap_unknown_param_rejected() -> None:
    """未知参数 → 不修。"""
    repaired, _ = repair_local_output("Action: click, x=100, y=200, z=1")
    assert repaired == "Action: click, x=100, y=200, z=1"
    assert parse_action(repaired) is None


def test_param_wrap_coordinates_untouched() -> None:
    """坐标数值前后完全一致。"""
    repaired, _ = repair_local_output(
        "Action: drag, x1=10, y1=20, x2=370, y2=999",
    )
    parsed = parse_action(repaired)
    assert parsed["params"] == {
        "x1": 10,
        "y1": 20,
        "x2": 370,
        "y2": 999,
    }


def test_param_wrap_multiple_actions_rejected() -> None:
    """多 Action → 不修。"""
    text = "Action: click, x=1, y=2\nAction: click, x=3, y=4"
    repaired, _ = repair_local_output(text)
    assert repaired == text
    assert parse_action(repaired) is None


def test_param_wrap_string_comma_preserved() -> None:
    """字符串参数内部逗号不被误处理。"""
    repaired, _ = repair_local_output('Action: type, text="你好，世界"')
    assert repaired == 'Action: type(text="你好，世界")'
    assert parse_action(repaired)["params"]["text"] == "你好，世界"


def test_param_wrap_never_in_api_mode(monkeypatch) -> None:
    """API 模式永远不经过 local repair(含 param_wrap)。"""
    import agent.gui_agent as ga

    calls = []
    monkeypatch.setattr(ga, "repair_local_output", lambda r: calls.append(1) or (r, []))
    agent = make_agent(
        SequenceBackend(
            ["Action: click, x=1, y=1", 'Action: finish(result="ok")'],
        ),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
    )
    asyncio.run(agent(Msg("u", "任务", "user")))
    assert calls == []
