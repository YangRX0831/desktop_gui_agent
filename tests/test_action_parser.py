"""测试动作语言提示词、严格语法和日志隐私。"""

import ast
import io
import logging
from pathlib import Path

import pytest

from agent.action_parser import ACTION_SYSTEM_PROMPT, parse_action


@pytest.mark.parametrize(
    "fragment",
    [
        "Action: click(x=<整数>, y=<整数>)",
        'Action: type(text="<JSON 字符串>")',
        'Action: scroll(direction="<up 或 down>", steps=<正整数>)',
        'Action: hotkey(key1="<按键>", key2="<按键>", ...)',
        'Action: finish(result="<JSON 字符串>")',
        "每次响应只输出一个动作",
        "桌面 GUI 操作智能体",
        "当前屏幕截图和用户指令",
        "生成下一步动作",
        "任务完成时必须使用 finish",
        "禁止 Markdown",
        "禁止解释",
        "JSON 双引号",
        "严格连续",
    ],
)
def test_system_prompt_contains_frozen_contract(fragment: str) -> None:
    """提示词完整说明动作语言和禁止事项。"""
    assert isinstance(ACTION_SYSTEM_PROMPT, str)
    assert ACTION_SYSTEM_PROMPT
    assert fragment in ACTION_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "forbidden",
    ["API Key", "Qwen", "Transformers", "用户任务："],
)
def test_system_prompt_excludes_runtime_configuration(forbidden: str) -> None:
    """提示词不包含任务、秘密或模型配置。"""
    assert forbidden not in ACTION_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            "Action: click(x=12, y=34)",
            {"action_type": "click", "params": {"x": 12, "y": 34}},
        ),
        (
            "Action: click(x=0, y=0)",
            {"action_type": "click", "params": {"x": 0, "y": 0}},
        ),
        (
            "Action: click(x=-12, y=-1)",
            {"action_type": "click", "params": {"x": -12, "y": -1}},
        ),
        (
            'Action: type(text="中文，括号()")',
            {"action_type": "type", "params": {"text": "中文，括号()"}},
        ),
        (
            r'Action: type(text="quote: \" and slash: \\")',
            {
                "action_type": "type",
                "params": {"text": 'quote: " and slash: \\'},
            },
        ),
        (
            r'Action: type(text="line1\nline2")',
            {"action_type": "type", "params": {"text": "line1\nline2"}},
        ),
        (
            'Action: type(text="")',
            {"action_type": "type", "params": {"text": ""}},
        ),
        (
            'Action: scroll(direction="up", steps=1)',
            {
                "action_type": "scroll",
                "params": {"direction": "up", "steps": 1},
            },
        ),
        (
            'Action: scroll(direction="down", steps=25)',
            {
                "action_type": "scroll",
                "params": {"direction": "down", "steps": 25},
            },
        ),
        (
            'Action: hotkey(key1="enter")',
            {"action_type": "hotkey", "params": {"keys": ("enter",)}},
        ),
        (
            'Action: hotkey(key1="ctrl", key2="shift", key3="s")',
            {
                "action_type": "hotkey",
                "params": {"keys": ("ctrl", "shift", "s")},
            },
        ),
        (
            'Action: finish(result="完成")',
            {"action_type": "finish", "params": {"result": "完成"}},
        ),
        (
            'Action: finish(result="")',
            {"action_type": "finish", "params": {"result": ""}},
        ),
        (
            " \tAction: click(x=1, y=2)\r\n",
            {"action_type": "click", "params": {"x": 1, "y": 2}},
        ),
    ],
)
def test_parse_valid_actions(
    response: str,
    expected: dict[str, object],
) -> None:
    """五种动作及 JSON 转义按冻结结构返回。"""
    assert parse_action(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "",
        " \t\r\n",
        'Action: type(text="a")\nextra',
        'Action: type(text="a")\rextra',
        "```text\nAction: click(x=1, y=2)\n```",
        "说明：Action: click(x=1, y=2)",
        "Action: click(x=1, y=2) trailing",
        'Action: click(x=1, y=2) Action: finish(result="done")',
        "action: click(x=1, y=2)",
        "Action: Click(x=1, y=2)",
        "Action: unknown(x=1)",
        "Action click(x=1, y=2)",
        "Action:  click(x=1, y=2)",
        "Action: click(x=1,y=2)",
        "Action: click(x=1,  y=2)",
        "Action: click(y=2, x=1)",
        "Action: click(x=1)",
        "Action: click(x=1, y=2, z=3)",
        "Action: click(x=1, x=2, y=3)",
        "Action: click(1, 2)",
        "Action: click(x=1+1, y=2)",
        "Action: click(x=int(1), y=2)",
        "Action: click(x=[1], y=2)",
        'Action: click(x={"a": 1}, y=2)',
        "Action: click(x=(1), y=2)",
        "Action: click(x=NaN, y=2)",
        "Action: click(x=Infinity, y=2)",
        "Action: click(x=1.0, y=2)",
        "Action: click(x=1e2, y=2)",
        "Action: click(x=+1, y=2)",
        "Action: click(x=01, y=2)",
        "Action: click(x=1_000, y=2)",
    ],
)
def test_parse_rejects_invalid_general_syntax(response: str) -> None:
    """额外文本、宽松格式及 Python 表达式均失败。"""
    assert parse_action(response) is None


@pytest.mark.parametrize(
    "response",
    [
        "Action: click(x=a, y=2)",
        "Action: click(x=1, y=a)",
        "Action: click(x=1)",
        "Action: click(y=2, x=1)",
        "Action: click(x=1, y=2, x=3)",
        "Action: click(x=-0, y=2)",
    ],
)
def test_click_rejects_invalid_parameters(response: str) -> None:
    """click 仅接受固定顺序的两个十进制整数。"""
    assert parse_action(response) is None


@pytest.mark.parametrize(
    "response",
    [
        "Action: type(text='single')",
        "Action: type(text=plain)",
        r'Action: type(text="\q")',
        'Action: type(text="open)',
        'Action: type(text="a", other="b")',
        "Action: type(text=123)",
        'Action: type(text=["a"])',
    ],
)
def test_type_rejects_non_json_string(response: str) -> None:
    """type 仅接受一个完整 JSON 字符串。"""
    assert parse_action(response) is None


@pytest.mark.parametrize(
    "response",
    [
        'Action: scroll(direction="left", steps=1)',
        "Action: scroll(direction='up', steps=1)",
        'Action: scroll(direction="up", steps=0)',
        'Action: scroll(direction="up", steps=-1)',
        'Action: scroll(direction="up", steps=1.0)',
        'Action: scroll(steps=1, direction="up")',
        'Action: scroll(direction="up")',
    ],
)
def test_scroll_rejects_invalid_parameters(response: str) -> None:
    """scroll 方向和正整数步数必须满足固定合同。"""
    assert parse_action(response) is None


@pytest.mark.parametrize(
    "response",
    [
        "Action: hotkey()",
        'Action: hotkey(key0="ctrl")',
        'Action: hotkey(key2="ctrl")',
        'Action: hotkey(key1="ctrl", key3="c")',
        'Action: hotkey(key1="ctrl", key1="c")',
        'Action: hotkey(key1="")',
        "Action: hotkey(key1='ctrl')",
        "Action: hotkey(key1=1)",
        'Action: hotkey(keys="ctrl+c")',
        'Action: hotkey(key="ctrl")',
        'Action: hotkey(key1="ctrl",key2="c")',
        'Action: hotkey(key1="ctrl",  key2="c")',
        'Action: hotkey(key1="ctrl", )',
    ],
)
def test_hotkey_rejects_invalid_parameters(response: str) -> None:
    """hotkey 只接受从 key1 起连续编号的 JSON 字符串。"""
    assert parse_action(response) is None


@pytest.mark.parametrize(
    "response",
    [
        "Action: finish()",
        'Action: finish(value="done")',
        "Action: finish(result='done')",
        r'Action: finish(result="\q")',
        'Action: finish(result="done", extra="x")',
    ],
)
def test_finish_rejects_invalid_parameters(response: str) -> None:
    """finish 必须且只能包含 JSON result。"""
    assert parse_action(response) is None


def test_overlong_integer_fails_without_execution() -> None:
    """超过 Python 安全转换限制的整数安全失败。"""
    huge_integer = "9" * 5000
    assert parse_action(f"Action: click(x={huge_integer}, y=1)") is None


@pytest.mark.parametrize("value", [None, 1, [], object()])
def test_non_string_raises_without_parse_log(
    value: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """类型错误不进入模型文本解析失败日志。"""
    with caplog.at_level(logging.WARNING, logger="agent.action_parser"):
        with pytest.raises(TypeError):
            parse_action(value)  # type: ignore[arg-type]
    assert caplog.records == []


def test_malicious_expression_is_not_executed(tmp_path: Path) -> None:
    """恶意表达式不会创建文件或被当作参数执行。"""
    target = tmp_path / "must_not_exist"
    response = (
        'Action: click(x=__import__("pathlib").Path(' f'"{target}").touch(), y=1)'
    )
    assert parse_action(response) is None
    assert not target.exists()


def test_parser_source_has_no_dynamic_evaluation() -> None:
    """解析器源码不调用 eval、exec 或 ast.literal_eval。"""
    source_path = Path(__file__).parents[1] / "agent" / "action_parser.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec"}:
                forbidden_calls.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "literal_eval":
                forbidden_calls.append(node.func.attr)
    assert forbidden_calls == []


def test_parser_source_uses_regular_expressions() -> None:
    """动作类型和参数结构确实通过 re 正则匹配。"""
    source_path = Path(__file__).parents[1] / "agent" / "action_parser.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports_re = any(
        isinstance(node, ast.Import) and any(alias.name == "re" for alias in node.names)
        for node in tree.body
    )
    compile_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
        and node.func.attr == "compile"
    ]
    assert imports_re
    assert compile_calls


def test_parse_failure_log_is_single_and_private() -> None:
    """最终日志只包含固定类别和长度，不包含原始响应。"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    target_logger = logging.getLogger("agent.action_parser")
    old_level = target_logger.level
    target_logger.setLevel(logging.WARNING)
    target_logger.addHandler(handler)
    marker = "MODEL_RESPONSE_SECRET_MARKER"
    response = f'Action: unknown(value="{marker}")'
    try:
        assert parse_action(response) is None
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(old_level)

    output = stream.getvalue()
    assert output.count("action_parse_failed") == 1
    assert "unsupported_action" in output
    assert f"响应长度={len(response)}" in output
    assert marker not in output
