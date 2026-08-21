"""把模型动作响应的确定性表示差异转换为 PRD canonical 文本。

本模块只处理无需截图、任务或语义推断的机械语法差异。每次响应至多应用
一条规则；输出仍必须交给 ``agent.action_parser.parse_action`` 严格验证。
无法唯一确定动作、参数意义或参数边界时保持原文，使 parser 明确拒绝。
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from agent.action_parser import CANONICAL_V3_ACTIONS

NormalizationReason = Literal[
    "action_canon_trim_outer_horizontal_whitespace",
    "action_canon_action_prefix_spacing",
    "action_canon_missing_action_prefix",
    "action_canon_001_click_missing_y_name",
    "action_canon_002_click_positional_xy",
    "action_canon_002_finish_unquoted_plain_text",
    "action_canon_type_single_quoted_text",
    "action_canon_scroll_single_quoted_direction",
    "action_canon_hotkey_key_list_representation",
    "action_adapt_001_symbolic_grounding_click",
]

_INTEGER_TOKEN = r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)"
_PRD_ACTION_NAMES = frozenset(CANONICAL_V3_ACTIONS)
_CANONICAL_ACTION_PATTERN = "|".join(re.escape(name) for name in CANONICAL_V3_ACTIONS)
_MISSING_PREFIX_PATTERN = re.compile(
    rf"(?P<action>{_CANONICAL_ACTION_PATTERN})\(.*\)",
)
_ACTION_PREFIX_SPACING_PATTERN = re.compile(
    rf"Action:(?P<body>(?:{_CANONICAL_ACTION_PATTERN})\(.*\))",
)
_CLICK_MISSING_Y_PATTERN = re.compile(
    rf"Action: click\(x=(?P<x>{_INTEGER_TOKEN}), (?P<y>{_INTEGER_TOKEN})\)",
)
_CLICK_POSITIONAL_XY_PATTERN = re.compile(
    rf"(?:Action: )?click\((?P<x>{_INTEGER_TOKEN}), ?" rf"(?P<y>{_INTEGER_TOKEN})\)",
)
_SYMBOLIC_CLICK_PATTERN = re.compile(
    r"(?:Action: )?click\((?P<symbol>E[1-9][0-9]*)\)",
)
_FINISH_UNQUOTED_PATTERN = re.compile(
    r"Action: finish\(result=(?P<value>[^,=()\"'\\\r\n]+)\)",
)
_TYPE_SINGLE_QUOTE_PATTERN = re.compile(
    r"Action: type\(text='(?P<value>[^'\"\\\r\n]*)'\)",
)
_SCROLL_SINGLE_QUOTE_PATTERN = re.compile(
    rf"Action: scroll\(direction='(?P<direction>up|down)', "
    rf"steps=(?P<steps>{_INTEGER_TOKEN})\)",
)
_HOTKEY_KEY_ALIAS_PATTERN = re.compile(
    r"Action: hotkey\(key=(?P<value>\"[^\"\\\r\n]+\"|'[^'\"\\\r\n]+')\)",
)
_HOTKEY_SINGLE_QUOTED_KEYN_PATTERN = re.compile(
    r"Action: hotkey\((?P<body>key1='[^'\"\\\r\n]+'"
    r"(?:, key[2-5]='[^'\"\\\r\n]+')*)\)",
)
_HOTKEY_POSITIONAL_QUOTED_PATTERN = re.compile(
    r"Action: hotkey\((?P<body>'[^'\"\\\r\n]+'" r"(?:, '[^'\"\\\r\n]+')*)\)",
)
_HOTKEY_POSITIONAL_BARE_PATTERN = re.compile(
    r"Action: hotkey\((?P<body>[A-Za-z0-9_]+(?:, [A-Za-z0-9_]+)*)\)",
)


@dataclass(frozen=True)
class AdaptedActionResponse:
    """保存 raw response、适配结果和唯一规则原因。"""

    raw_response: str
    normalized_response: str
    normalization_reason: NormalizationReason | None


def _quoted(value: str) -> str:
    """把已确定边界的字符串编码为 canonical JSON string literal。"""
    return json.dumps(value, ensure_ascii=False)


def _adapt_hotkey(response: str) -> str | None:
    """把 corpus 中参数顺序唯一的 hotkey 表示转换为 keyN 形式。"""
    alias = _HOTKEY_KEY_ALIAS_PATTERN.fullmatch(response)
    if alias is not None:
        token = alias.group("value")
        value = token[1:-1]
        return f"Action: hotkey(key1={_quoted(value)})"

    named = _HOTKEY_SINGLE_QUOTED_KEYN_PATTERN.fullmatch(response)
    if named is not None:
        pairs = re.findall(r"key([1-5])='([^']+)'", named.group("body"))
        if [int(number) for number, _ in pairs] != list(
            range(1, len(pairs) + 1),
        ):
            return None
        values = [value for _, value in pairs]
    else:
        positional = _HOTKEY_POSITIONAL_QUOTED_PATTERN.fullmatch(response)
        if positional is not None:
            values = re.findall(r"'([^']+)'", positional.group("body"))
        else:
            bare = _HOTKEY_POSITIONAL_BARE_PATTERN.fullmatch(response)
            if bare is None:
                return None
            values = bare.group("body").split(", ")
    arguments = ", ".join(
        f"key{index}={_quoted(value)}" for index, value in enumerate(values, 1)
    )
    return f"Action: hotkey({arguments})"


def _resolve_grounding_center(
    symbol: str,
    candidates: Sequence[Mapping[str, object]],
    turn_token: str | None,
) -> tuple[int, int] | None:
    """把当轮唯一符号解析为合法 bbox 中心；任何歧义均拒绝。"""
    if turn_token is None:
        return None
    matches = [
        candidate
        for candidate in candidates
        if candidate.get("symbol") == symbol
        and candidate.get("turn_token") == turn_token
    ]
    if len(matches) != 1:
        return None
    bbox = matches[0].get("bbox")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
    ):
        return None
    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        return None
    return (left + right) // 2, (top + bottom) // 2


def adapt_action_response(
    response: str,
    grounding_candidates: Sequence[Mapping[str, object]] = (),
    *,
    grounding_turn_token: str | None = None,
) -> AdaptedActionResponse:
    """应用至多一条确定性语法规则，不判断或补造动作语义。

    Args:
        response: 模型返回的原始字符串。
        grounding_candidates: 当前模型输入中已显式呈现的候选。
        grounding_turn_token: 当前感知轮次令牌；只解析同轮候选。

    Returns:
        包含原文、canonicalization 候选和固定原因码的不可变结果。

    Raises:
        TypeError: response 不是字符串。

    适配结果不代表动作合法；调用方必须继续使用严格 ActionParser 验证。
    """
    if not isinstance(response, str):
        raise TypeError("response 必须是 str。")

    symbolic_click = _SYMBOLIC_CLICK_PATTERN.fullmatch(response)
    if symbolic_click is not None:
        center = _resolve_grounding_center(
            symbolic_click.group("symbol"),
            grounding_candidates,
            grounding_turn_token,
        )
        if center is not None:
            return AdaptedActionResponse(
                response,
                f"Action: click(x={center[0]}, y={center[1]})",
                "action_adapt_001_symbolic_grounding_click",
            )
        return AdaptedActionResponse(response, response, None)

    if "\r" not in response and "\n" not in response:
        stripped = response.strip(" \t")
        if stripped != response:
            return AdaptedActionResponse(
                response,
                stripped,
                "action_canon_trim_outer_horizontal_whitespace",
            )

    prefix_spacing = _ACTION_PREFIX_SPACING_PATTERN.fullmatch(response)
    if prefix_spacing is not None:
        return AdaptedActionResponse(
            response,
            f"Action: {prefix_spacing.group('body')}",
            "action_canon_action_prefix_spacing",
        )

    positional_click = _CLICK_POSITIONAL_XY_PATTERN.fullmatch(response)
    if positional_click is not None:
        return AdaptedActionResponse(
            response,
            "Action: click("
            f"x={positional_click.group('x')}, "
            f"y={positional_click.group('y')})",
            "action_canon_002_click_positional_xy",
        )

    missing_prefix = _MISSING_PREFIX_PATTERN.fullmatch(response)
    if (
        missing_prefix is not None
        and missing_prefix.group("action") in _PRD_ACTION_NAMES
    ):
        return AdaptedActionResponse(
            response,
            f"Action: {response}",
            "action_canon_missing_action_prefix",
        )

    click = _CLICK_MISSING_Y_PATTERN.fullmatch(response)
    if click is not None:
        return AdaptedActionResponse(
            response,
            f"Action: click(x={click.group('x')}, y={click.group('y')})",
            "action_canon_001_click_missing_y_name",
        )

    finish = _FINISH_UNQUOTED_PATTERN.fullmatch(response)
    if finish is not None and finish.group("value").strip():
        value = finish.group("value")
        return AdaptedActionResponse(
            response,
            f"Action: finish(result={_quoted(value)})",
            "action_canon_002_finish_unquoted_plain_text",
        )

    typed = _TYPE_SINGLE_QUOTE_PATTERN.fullmatch(response)
    if typed is not None:
        return AdaptedActionResponse(
            response,
            f"Action: type(text={_quoted(typed.group('value'))})",
            "action_canon_type_single_quoted_text",
        )

    scroll = _SCROLL_SINGLE_QUOTE_PATTERN.fullmatch(response)
    if scroll is not None:
        return AdaptedActionResponse(
            response,
            "Action: scroll("
            f'direction="{scroll.group("direction")}", '
            f'steps={scroll.group("steps")})',
            "action_canon_scroll_single_quoted_direction",
        )

    hotkey = _adapt_hotkey(response)
    if hotkey is not None:
        return AdaptedActionResponse(
            response,
            hotkey,
            "action_canon_hotkey_key_list_representation",
        )

    return AdaptedActionResponse(response, response, None)
