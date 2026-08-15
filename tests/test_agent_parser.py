"""Action parser 确定性测试。"""

import asyncio

import pytest
from agentscope.message import Msg

from agent.action_parser import parse_action
from agent.task_manager import TaskManager
from main import _format_action
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent


@pytest.mark.parametrize(
    "response",
    [
        "Action: click(x=100, y=200)",
        'Action: type(text="Hello World")',
        'Action: scroll(direction="up", steps=3)',
        'Action: scroll(direction="down", steps=1)',
        'Action: hotkey(key1="ctrl", key2="c")',
        'Action: hotkey(key1="enter")',
        'Action: finish(result="done")',
        "Action: right_click(x=0, y=1000)",
        "Action: double_click(x=1000, y=0)",
        "Action: drag(x1=0, y1=0, x2=1000, y2=1000)",
    ],
)
def test_approved_prompt_actions_match_strict_parser(response: str) -> None:
    """批准 Prompt 中的合法动作示例全部通过严格 parser。"""
    assert parse_action(response) is not None


def test_type_parser_preserves_json_string_escapes() -> None:
    """type 使用 JSON 字符串解析，引号、反斜杠和换行转义不被切坏。"""
    action = parse_action(
        r'Action: type(text="line 1\n\"quoted\"\\tail")',
    )
    assert action == {
        "action_type": "type",
        "params": {"text": 'line 1\n"quoted"\\tail'},
    }


@pytest.mark.parametrize(
    "response",
    [
        "",
        "好的，我来操作。\nAction: click(x=100, y=200)",
        "Action: click(x=100, y=200)\n操作完成。",
        "```text\nAction: click(x=100, y=200)\n```",
        'Action: click(x=100, y=200)\nAction: type(text="abc")',
        'Action: press(key="enter")',
        "Action: move_to(x=100, y=200)",
        "Action: click(x=123)",
        "Action: click(y=456)",
        'Action: click(x="123", y="456")',
        "Action: click(x=123.5, y=456)",
        'Action: scroll(direction="left", steps=3)',
        'Action: scroll(direction="down", steps=0)',
        'Action: scroll(direction="down", steps=-1)',
        'Action: hotkey(key2="c")',
        'Action: hotkey(key1="ctrl", key3="s")',
        'Action: hotkey(key1="ctrl+c")',
        'Action: hotkey(key1="not_a_real_key")',
        "Action: finish()",
        "```Action: finish()```",
    ],
)
def test_parser_rejects_extra_text_or_invalid_actions(response: str) -> None:
    assert parse_action(response) is None


def test_cli_action_format_matches_parser_contract() -> None:
    """CLI 动作显示可被同一严格 Parser 重新解析。

    覆盖全部八种动作，确保 _format_action 对每种动作类型都输出可被
    parse_action 重新解析的格式；历史版本曾漏掉 right_click/double_click/
    drag，在显示回调中对这些动作访问 params["result"] 触发 KeyError。
    """
    for action in [
        parse_action("Action: click(x=10, y=20)"),
        parse_action("Action: right_click(x=30, y=40)"),
        parse_action("Action: double_click(x=50, y=60)"),
        parse_action("Action: drag(x1=10, y1=20, x2=30, y2=40)"),
        parse_action('Action: type(text="Hello World")'),
        parse_action('Action: scroll(direction="up", steps=3)'),
        parse_action('Action: hotkey(key1="ctrl", key2="c")'),
        parse_action('Action: finish(result="完成")'),
    ]:
        assert action is not None
        assert parse_action(f"Action: {_format_action(action)}") == action


def test_no_same_parsed_action_replay() -> None:
    """fresh retry 产生新动作,不重放同一 ParsedAction。"""
    backend = SequenceBackend(
        ["Action: click(x=10, y=20)", "Action: click(x=30, y=40)"],
    )
    controls = MemoryControls(fail_first=1)
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    # 两次不同的 click 动作,不是同一动作重放。
    assert controls.calls == [("click", 10, 20, "left"), ("click", 30, 40, "left")]


def test_parse_failure_retries_with_fresh_model() -> None:
    """parse 失败在 retry 预算内继续 fresh model 决策。"""
    backend = SequenceBackend(["bad", 'Action: finish(result="ok")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    assert result.content == "ok"
    assert backend.calls == 2
    assert "模型动作解析失败。 (invalid_syntax)" in backend.prompts[1]


def test_parse_fail_all_retries_step_record_exists() -> None:
    """parse fail × initial+3 retries → step record 存在,retry_count=3。"""
    # model 返回无效文本 → parse 失败。
    outcomes = ["bad_text"] * 4 + ['Action: finish(result="done")']
    backend = SequenceBackend(outcomes)
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3, max_steps=2)(
            Msg("u", "任务", "user"),
        ),
    )
    assert result.content == "done"
    step1 = manager.state.steps[0]
    assert step1.result is False
    assert step1.retry_count == 3
    assert len(step1.attempts) == 4
    for a in step1.attempts:
        assert a.action is None
        assert a.stage == "parse"
        assert a.failure_reason is not None


def test_main_argument_parser_defaults() -> None:
    """CLI 参数解析器默认值正确。"""
    from main import create_argument_parser

    parser = create_argument_parser()
    args = parser.parse_args([])
    assert args.model_mode == "local"
    assert args.max_steps == 10
    assert args.retry_count == 3


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            "Action: right_click(x=100, y=200)",
            {"action_type": "right_click", "params": {"x": 100, "y": 200}},
        ),
        (
            "Action: double_click(x=0, y=1000)",
            {"action_type": "double_click", "params": {"x": 0, "y": 1000}},
        ),
        (
            "Action: drag(x1=10, y1=20, x2=30, y2=40)",
            {
                "action_type": "drag",
                "params": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
            },
        ),
        (
            'Action: hotkey(key1="win", key2="s")',
            {"action_type": "hotkey", "params": {"keys": ("win", "s")}},
        ),
    ],
)
def test_extended_actions_parse_success(response: str, expected: dict) -> None:
    """新增鼠标动作与 win 键解析为结构化动作。"""
    assert parse_action(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "Action: right_click(x=100,y=200)",
        "Action: double_click(x=100)",
        "Action: drag(x1=10, y1=20, x2=30)",
        "Action: drag(x1=10, y1=20, x2=30, y2=40) extra",
        "Action: move_to(x=1, y=2)",
        'Action: press(key1="ctrl")',
    ],
)
def test_extended_actions_reject_invalid(response: str) -> None:
    """参数缺失、逗号无空格、白名单外动作仍然拒绝。"""
    assert parse_action(response) is None


def test_normalize_model_output_fixes_prefix_variants() -> None:
    """小模型常见前缀偏差经归一化后可被严格 parser 接受。"""
    from agent.action_parser import normalize_model_output, parse_action

    cases = [
        ("1. click(x=500, y=1000)", "Action: click(x=500, y=1000)"),
        ('2. type(text="hello")', 'Action: type(text="hello")'),
        ('3. finish(result="done")', 'Action: finish(result="done")'),
        ("click(x=100, y=200)", "Action: click(x=100, y=200)"),
        ("Action: click(x=1, y=2)", "Action: click(x=1, y=2)"),
        ("random text", "random text"),
        ('8. hotkey(key1="win")', 'Action: hotkey(key1="win")'),
    ]
    for raw, expected in cases:
        assert normalize_model_output(raw) == expected, raw

    assert parse_action("1. click(x=500, y=1000)") == {
        "action_type": "click",
        "params": {"x": 500, "y": 1000},
    }
    assert parse_action('2. type(text="hello")') == {
        "action_type": "type",
        "params": {"text": "hello"},
    }
    assert parse_action('5. finish(result="done")') is not None
    # 非法内容归一化后仍被 parser 拒绝
    assert parse_action("1. click(x=abc, y=def)") is None
    assert parse_action("1. unknown_action(x=1)") is None
