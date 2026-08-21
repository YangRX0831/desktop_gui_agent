"""对本地模型输出执行确定性的表面格式修复。

处理链路为：本地模型原始输出 → 本模块 → Action Response Adapter → 严格
动作解析器。API 模式由调用方直接跳过本模块。

本模块只处理无歧义且不改变动作语义的表示差异：
1. 删除纯 Markdown fence 标记行；
2. 全文恰好存在一个 Action 行时提取该行；
3. 仅在字符串之外把中文语法标点转换为半角；
4. 将已知动作名的大小写规范为小写；
5. 规范首尾空白；
6. 在参数名及顺序与动作签名完全一致时补齐括号形式。

不得在多个动作之间选择，不修改坐标，不猜测缺失参数，不根据任务内容
硬编码结果，也不放宽公共解析器。无法确定修复结果时保持原文，由严格解析器
拒绝。
"""

import re

from agent.action_parser import action_param_signature

_KNOWN_ACTION_NAMES = frozenset(
    {
        "click",
        "right_click",
        "double_click",
        "drag",
        "type",
        "scroll",
        "hotkey",
        "finish",
        "observe",
    },
)
_FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})[^`~]*(`{3,}|~{3,})?\s*$")
_ACTION_LINE = re.compile(r"^\s*(?:\d+[.、)]\s*)?action\s*[:：]", re.IGNORECASE)
_ACTION_VERB = re.compile(
    r"^(?P<prefix>\s*(?:\d+[.、)]\s*)?[Aa][Cc][Tt][Ii][Oo][Nn]\s*[:：]\s*)"
    r"(?P<verb>[A-Za-z_]+)(?P<rest>\()",
)
# 仅在引号外替换 Action 语法中的全角冒号、逗号和圆括号。
_CJK_PUNCT_MAP = {
    "：": ":",
    "，": ",",
    "（": "(",
    "）": ")",
}


def _strip_fence_lines(text: str) -> str:
    """删除纯 Markdown fence 标记行，内部内容按行原样保留。"""
    lines = [line for line in text.split("\n") if not _FENCE_LINE.match(line)]
    return "\n".join(lines)


def _extract_unique_action_line(text: str) -> str:
    """全文恰有一个 Action 行时返回该行，否则原样返回。"""
    action_lines = [line for line in text.split("\n") if _ACTION_LINE.match(line)]
    if len(action_lines) == 1:
        return action_lines[0].strip()
    return text


def _fix_punct_outside_strings(text: str) -> str:
    """把中文语法标点替换为半角，同时保留双引号字符串内部内容。

    以半角双引号为界交替划分字符串外与字符串内片段。Action 语法关键字符
    只在字符串外规范化，因此文本参数中的中文标点不会被改写。
    """
    parts = text.split('"')
    rebuilt = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            for cjk, ascii_punct in _CJK_PUNCT_MAP.items():
                part = part.replace(cjk, ascii_punct)
            # 协议要求参数逗号后使用一个空格；只处理字符串外片段。
            part = re.sub(r",\s*", ", ", part)
        rebuilt.append(part)
    result = '"'.join(rebuilt)
    # 只移除完整动作右括号后的中文句号，不处理参数字符串中的句号。
    if result.endswith("。") and result[:-1].rstrip().endswith(")"):
        result = result[:-1].rstrip()
    return result


def _fix_action_verb_case(text: str) -> str:
    """把已知动作名规范为小写，参数和值保持不变。"""
    match = _ACTION_VERB.match(text)
    if match is None:
        return text
    verb = match.group("verb")
    if verb.lower() in _KNOWN_ACTION_NAMES:
        return (
            match.group("prefix")
            + verb.lower()
            + match.group("rest")
            + text[match.end() :]
        )
    return text


_COMMA_ACTION = re.compile(
    r"^(?:\d+[.、)]\s*)?[Aa][Cc][Tt][Ii][Oo][Nn]\s*[:：]\s*"
    r"(?P<verb>[A-Za-z_]+)\s*,\s*(?P<body>.+)$",
)
_PARAM_PART = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")


def _wrap_comma_named_params(text: str) -> str:
    """将逗号参数形式转换为标准括号形式。

    例如 ``Action: click, x=100, y=200`` 可转换为
    ``Action: click(x=100, y=200)``。只有在动作已知、参数名序列与合法签名
    完全一致且没有多余内容时才转换；否则原样交给严格解析器。
    """
    match = _COMMA_ACTION.match(text)
    if match is None:
        return text
    verb = match.group("verb").lower()
    signature = action_param_signature(verb)
    if signature is None:
        return text
    body = match.group("body")
    # 按引号状态切分顶层逗号，避免把字符串参数内部逗号当作参数分隔符。
    parts = []
    current = []
    in_quote = False
    for char in body:
        if char == '"':
            in_quote = not in_quote
            current.append(char)
        elif char == "," and not in_quote:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    parsed = []
    for part in parts:
        part_match = _PARAM_PART.match(part)
        if part_match is None:
            return text
        parsed.append((part_match.group(1), part_match.group(2)))
    names = tuple(name for name, _value in parsed)
    if names != signature:
        return text
    args = ", ".join(f"{name}={value}" for name, value in parsed)
    return f"Action: {verb}({args})"


def repair_local_output(response: str) -> tuple[str, list[str]]:
    """对本地模型原始输出执行确定性表面修复。

    Args:
        response: 模型返回的原始文本。

    Returns:
        修复后文本与已应用修复类型列表。各规则按固定顺序执行；不能确定的
        表示保持原样，由严格解析器决定是否接受。

    Raises:
        TypeError: response 不是 str。
    """
    if not isinstance(response, str):
        raise TypeError("response 必须是 str。")
    text = response.strip()
    repairs: list[str] = []
    fenced = _strip_fence_lines(text)
    if fenced != text:
        text = fenced.strip()
        repairs.append("markdown_fence")
    extracted = _extract_unique_action_line(text)
    if extracted != text:
        text = extracted
        repairs.append("unique_action_line")
    cased = _fix_action_verb_case(text)
    if cased != text:
        text = cased
        repairs.append("action_case")
    punct = _fix_punct_outside_strings(text)
    if punct != text:
        text = punct
        repairs.append("cjk_punct")
    wrapped = _wrap_comma_named_params(text)
    if wrapped != text:
        text = wrapped
        repairs.append("param_wrap")
    return text.strip(), repairs
