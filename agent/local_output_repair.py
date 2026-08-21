"""LOCAL-ONLY 输出表面修复:仅服务 model_mode == local 的 2B 模型。

流程合同:LOCAL RAW → local_output_repair → Action Response Adapter
→ 现有 strict parser。API 模式完全跳过本模块(调用方按 model_mode 门控)。

安全边界:只做确定性、无歧义、不改变动作语义的修复——
1. 剥 Markdown fence 行(``` / ~~~ 纯标记行);
2. 恰好一个 Action 行时提取该行(≥2 个 Action 候选保持原样交严格 parser 拒绝);
3. 中文标点→半角(仅作用于引号外区域的 Action 语法部分,字符串参数内
   的中文标点原样保留);
4. 已知动作名大小写标准化(仅动词 token);
5. 首尾空白 strip。
严禁:多 Action 二选一、改坐标、猜参数、按任务 hardcode、放宽 common
parser。任何无法无歧义修复的输入保持 parse failure。
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
# 中文标点→半角:仅引号外片段执行;全角冒号/逗号/圆括号。
_CJK_PUNCT_MAP = {
    "：": ":",
    "，": ",",
    "（": "(",
    "）": ")",
}


def _strip_fence_lines(text: str) -> str:
    """删除纯 Markdown fence 标记行;内部内容逐行原样保留。"""
    lines = [line for line in text.split("\n") if not _FENCE_LINE.match(line)]
    return "\n".join(lines)


def _extract_unique_action_line(text: str) -> str:
    """全文恰有一个 Action 行时返回该行;否则原样返回。"""
    action_lines = [line for line in text.split("\n") if _ACTION_LINE.match(line)]
    if len(action_lines) == 1:
        return action_lines[0].strip()
    return text


def _fix_punct_outside_strings(text: str) -> str:
    """把中文语法标点替换为半角;双引号字符串内部不动。

    以半角双引号为界交替划分引号外/内片段;Action 语法的关键字符
    (冒号/逗号/括号)只出现在引号外,参数字符串内的中文标点由此
    得到完整保护。
    """
    parts = text.split('"')
    rebuilt = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            for cjk, ascii_punct in _CJK_PUNCT_MAP.items():
                part = part.replace(cjk, ascii_punct)
            # 中文逗号替换后无空格;协议要求"逗号后一个空格",把引号外
            # 的逗号后空白统一收敛为恰好一个空格(确定性,不动字符串内)。
            part = re.sub(r",\s*", ", ", part)
        rebuilt.append(part)
    result = '"'.join(rebuilt)
    # Action 末尾的中文句号:只剥"最后一个引号外字符且其前是右括号"的形态。
    if result.endswith("。") and result[:-1].rstrip().endswith(")"):
        result = result[:-1].rstrip()
    return result


def _fix_action_verb_case(text: str) -> str:
    """把已知动作名 token 标准化为小写;坐标与参数不动。"""
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
    """``Action: click, x=100, y=200`` → ``Action: click(x=100, y=200)``。

    仅当:单行唯一 Action;动词已知;参数名序列与该动作合法 signature
    完全一致;无多余 token;所有参数值原样保留。否则原样返回交严格
    parser 拒绝(缺参数/未知参数/改语义一律不修)。
    """
    match = _COMMA_ACTION.match(text)
    if match is None:
        return text
    verb = match.group("verb").lower()
    signature = action_param_signature(verb)
    if signature is None:
        return text
    body = match.group("body")
    # 引号感知的顶层逗号切分:字符串参数内部的逗号属于参数值。
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
    """对 local 模型原始输出做确定性表面修复。

    Args:
        response: 模型返回的原始文本。

    Returns:
        (修复后文本, 使用的修复类型列表);修复按固定顺序执行,
        任何一步不确定即保持原样交由严格 parser 裁决。

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
