"""定义模型动作语言、系统提示词和严格解析器。"""

import json
import logging
import re
from typing import Literal, TypedDict

logger = logging.getLogger(__name__)

ActionType = Literal["click", "type", "scroll", "hotkey", "finish"]


class ClickParams(TypedDict):
    """点击动作参数。"""

    x: int
    y: int


class TypeParams(TypedDict):
    """文本输入动作参数。"""

    text: str


class ScrollParams(TypedDict):
    """滚动动作参数。"""

    direction: Literal["up", "down"]
    steps: int


class HotkeyParams(TypedDict):
    """组合键动作参数。"""

    keys: tuple[str, ...]


class FinishParams(TypedDict):
    """任务结束动作参数。"""

    result: str


ParsedParams = ClickParams | TypeParams | ScrollParams | HotkeyParams | FinishParams


class ParsedAction(TypedDict):
    """已通过语法校验的单个动作。"""

    action_type: ActionType
    params: ParsedParams


ACTION_SYSTEM_PROMPT = """你是桌面 GUI 操作智能体。
请根据当前屏幕截图和用户指令生成下一步动作。
每次响应只输出一个动作，并且必须严格以“Action: ”开头。
动作名称大小写敏感，只允许 click、type、scroll、hotkey、finish。
禁止 Markdown 代码块，禁止解释、分析、前言、后记和任何额外内容。
参数名、参数顺序、逗号和空格必须严格遵循下列唯一格式：
Action: click(x=<整数>, y=<整数>)
Action: type(text="<JSON 字符串>")
Action: scroll(direction="<up 或 down>", steps=<正整数>)
Action: hotkey(key1="<按键>", key2="<按键>", ...)
Action: finish(result="<JSON 字符串>")
type 和 finish 必须使用 JSON 双引号字符串；引号和反斜杠使用 JSON 转义，
需要换行时使用 \\n，禁止输出真实内部换行。
hotkey 至少包含 key1，参数编号必须从 key1 开始严格连续。
任务完成时必须使用 finish 动作。
一条响应不得包含多个 Action。
禁止使用 Python 表达式或函数调用作为参数值。"""

# 模型输出视为不可信文本：仅接受白名单动作语法并解析为结构化数据，
# 不使用 eval、exec 或其他动态执行方式处理原始模型文本。
_INTEGER_TOKEN = r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)"
_JSON_STRING_TOKEN = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
_CLICK_PATTERN = re.compile(
    rf"Action: click\(x=(?P<x>{_INTEGER_TOKEN}), " rf"y=(?P<y>{_INTEGER_TOKEN})\)",
)
_TYPE_PATTERN = re.compile(
    rf"Action: type\(text=(?P<text>{_JSON_STRING_TOKEN})\)",
)
_SCROLL_PATTERN = re.compile(
    rf'Action: scroll\(direction="(?P<direction>up|down)", '
    rf"steps=(?P<steps>{_INTEGER_TOKEN})\)",
)
_FINISH_PATTERN = re.compile(
    rf"Action: finish\(result=(?P<result>{_JSON_STRING_TOKEN})\)",
)
_ACTION_NAME_PATTERN = re.compile(r"Action: ([A-Za-z_][A-Za-z0-9_]*)\(")
_HOTKEY_ITEM_PATTERN = re.compile(
    rf"key(?P<number>[0-9]+)=(?P<value>{_JSON_STRING_TOKEN})(?P<end>, |$)",
)
_HOTKEY_PREFIX = "Action: hotkey("
_ALLOWED_ACTIONS = {"click", "type", "scroll", "hotkey", "finish"}


def _log_parse_failure(category: str, response_length: int) -> None:
    """记录不包含模型原始响应的固定失败摘要。"""
    logger.warning(
        "action_parse_failed：类别=%s，响应长度=%d",
        category,
        response_length,
    )


def _parse_integer(token: str) -> int | None:
    """安全转换已经通过词法约束的十进制整数。"""
    try:
        return int(token, 10)
    except ValueError:
        return None


def _parse_json_string(token: str) -> str | None:
    """解析单个 JSON 字符串字面量。"""
    try:
        value = json.loads(token)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _parse_click(response: str) -> ParsedAction | None:
    """解析点击动作。"""
    match = _CLICK_PATTERN.fullmatch(response)
    if match is None:
        return None
    x = _parse_integer(match.group("x"))
    y = _parse_integer(match.group("y"))
    if x is None or y is None:
        return None
    return {"action_type": "click", "params": {"x": x, "y": y}}


def _parse_type(response: str) -> ParsedAction | None:
    """解析文本输入动作。"""
    match = _TYPE_PATTERN.fullmatch(response)
    if match is None:
        return None
    text = _parse_json_string(match.group("text"))
    if text is None:
        return None
    return {"action_type": "type", "params": {"text": text}}


def _parse_scroll(response: str) -> ParsedAction | None:
    """解析滚动动作。"""
    match = _SCROLL_PATTERN.fullmatch(response)
    if match is None:
        return None
    steps = _parse_integer(match.group("steps"))
    if steps is None or steps <= 0:
        return None
    direction_value = match.group("direction")
    direction: Literal["up", "down"]
    if direction_value == "up":
        direction = "up"
    else:
        direction = "down"
    return {
        "action_type": "scroll",
        "params": {"direction": direction, "steps": steps},
    }


def _parse_hotkey(response: str) -> ParsedAction | None:
    """解析参数编号连续的组合键动作。"""
    if not response.startswith(_HOTKEY_PREFIX) or not response.endswith(")"):
        return None
    body = response[len(_HOTKEY_PREFIX) : -1]
    if not body:
        return None

    keys: list[str] = []
    position = 0
    expected_number = 1
    while position < len(body):
        match = _HOTKEY_ITEM_PATTERN.match(body, position)
        if match is None or int(match.group("number")) != expected_number:
            return None
        key = _parse_json_string(match.group("value"))
        if key is None or not key:
            return None
        keys.append(key)
        position = match.end()
        if match.group("end") and position == len(body):
            return None
        expected_number += 1

    return {"action_type": "hotkey", "params": {"keys": tuple(keys)}}


def _parse_finish(response: str) -> ParsedAction | None:
    """解析带结果文本的结束动作。"""
    match = _FINISH_PATTERN.fullmatch(response)
    if match is None:
        return None
    result = _parse_json_string(match.group("result"))
    if result is None:
        return None
    return {"action_type": "finish", "params": {"result": result}}


def parse_action(response: str) -> ParsedAction | None:
    """严格解析单条模型动作响应。

    Args:
        response: 模型返回的原始文本。

    Returns:
        合法动作字典；文本不符合动作语言时返回 None。

    Raises:
        TypeError: response 不是字符串。
    """
    if not isinstance(response, str):
        raise TypeError("response 必须是 str。")

    response_length = len(response)
    stripped = response.strip()
    if not stripped:
        _log_parse_failure("empty_response", response_length)
        return None
    if "\r" in stripped or "\n" in stripped:
        _log_parse_failure("multiline_response", response_length)
        return None

    action_name_match = _ACTION_NAME_PATTERN.match(stripped)
    if action_name_match is None:
        _log_parse_failure("invalid_syntax", response_length)
        return None
    action_name = action_name_match.group(1)
    if action_name not in _ALLOWED_ACTIONS:
        _log_parse_failure("unsupported_action", response_length)
        return None

    parsers = {
        "click": _parse_click,
        "type": _parse_type,
        "scroll": _parse_scroll,
        "hotkey": _parse_hotkey,
        "finish": _parse_finish,
    }
    parsed = parsers[action_name](stripped)
    if parsed is None:
        _log_parse_failure("invalid_parameters", response_length)
    return parsed
