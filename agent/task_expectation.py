"""Phase 2A 程序化任务期望与完成验证(纯函数,无 GUI/模型副作用)。

设计约束:
    finish 只能视为 FINISH PROPOSAL,是否接受由程序化 CompletionVerifier
    裁决;verifier 只使用运行时已可靠拥有的结构化事实(音量、OCR、前台
    进程、窗口存在性),证据不足时返回 UNKNOWN,绝不把 UNKNOWN 当成功。
    期望字段来自任务文本的窄模式抽取或调用方注入,不做通用 NLP。
"""

import re
from dataclasses import dataclass, field
from typing import Literal

VerificationStatus = Literal["VERIFIED", "NOT_VERIFIED", "UNKNOWN"]
ProgressStatus = Literal["UNKNOWN", "PROGRESSED", "NO_PROGRESS", "INSUFFICIENT_RATE"]

# 前台停留在这类 Shell 叠层进程时,屏幕证据明确不支持"目标应用已完成"。
SHELL_OVERLAY_PROCESSES = frozenset(
    {
        "searchhost.exe",
        "startmenuexperiencehost.exe",
        "shellexperiencehost.exe",
    },
)

_VOLUME_PATTERN = re.compile(r"音量[^0-9%]{0,8}(\d{1,3})%")
_CALC_PATTERN = re.compile(r"计算(\d{1,4}\s*[+\-×÷]\s*\d{1,4})")
# 输入内容标识的特殊 token 抽取(word+3位数字)已按 PRD-BC-001 人工
# 裁决移除:其形状与 benchmark 生成标识精确重合,无独立产品依据;
# 不以新的猜测性标识 regex 替代。expected_text 仅保留通用注入口。
# 窗口关闭类任务:运行时无法从文本知道具体 HWND,用哨兵表示
# "程序验证到窗口已关闭即满足";证据来自 _verify_action_effect 的
# window_closed 效果,不按 task_id 分支。
ANY_TRACKED_WINDOW = -1
_CLOSE_WINDOW_PATTERN = re.compile(r"关闭.{0,16}窗口")

# 文本投递意图(P1 POST_SUBMIT_DELIVERY_CONFIRMATION):发送/回复类动词
# 携带引号或冒号引导的 payload。只认投递动词,复制/搜索/写到/删除等
# 动词不触发,负例由测试锁定。
_DELIVERY_QUOTED_PATTERNS = (
    # 给 user1 发送"X" / 向 user1 发送消息"X" / 回复对方"X"
    re.compile(r'(?:发送|发消息|回复)[^"“”]{0,12}?["“]([^"”]{1,80})["”]'),
    # 发送"X"给 user1 / 把"X"发给 user1
    re.compile(r'["“]([^"”]{1,80})["”][^。]{0,8}(?:发给|发送给)'),
    # 给 user1 发一条消息：X / 发送消息：X(冒号后到句读/句尾为止,
    # payload 允许内部空格,如"任务编号 ZX-418 完成")
    re.compile(r"(?:发一条消息|发消息|发送消息)[:：]\s*([^。,;；]{1,80})"),
)

# P2 MULTI_STEP_DOWNLOAD_SAVE_COMPLETION:图片类保存/下载意图。
# 仅当保存/下载动词与图片类宾语共现时成立;"保存网页/文档"等其它
# 保存对象不触发(第一版只覆盖 IMAGE 内容,保守不误触发)。
_SAVE_IMAGE_INTENT_PATTERNS = (
    # 保存这张图片 / 另存为...图片 / 下载图片 / save this picture(动词在
    # 前,允许少量间隔;中英动词均认)
    re.compile(
        r"(?:保存|另存|下载|save|download)[^,。;]{0,10}?"
        r"(图片|图像|照片|photo|image|picture)",
        re.IGNORECASE,
    ),
    # 把这张图保存下来 / 图片另存到...(宾语在前)
    re.compile(
        r"(图片|图像|照片)[^,。;]{0,6}(?:保存|另存|下载|存下)",
    ),
    # 这张图保存成 X / 把图片存为(短形"图"字)
    re.compile(r"这[张张幅]图[^,。;]{0,6}(?:保存|另存|存)"),
)
_SAVE_FILENAME_PATTERNS = (
    # 保存/另存/命名/文件名赋值措辞 + 扩展名约束。"文件名"分支带
    # 查询/复制/保持类动词的负向后视与"的文件"后缀拒绝,避免吞掉
    # "搜索文件名为X的文件"这类非赋值语境。
    re.compile(
        r"(?<!搜索)(?<!查找)(?<!找到)(?<!复制)(?<!查看)(?<!显示)(?<!保持)"
        r"(?<!修改)(?<!使用)(?<!保留)"
        r"(?:另存为|保存成|保存为|命名为|文件名(?:设为|设置为|用|为)?)"
        r"[\s\"“]*"
        r"([A-Za-z0-9_\-一-龥][\w\-一-龥]*\.(?:png|jpg|jpeg|webp|gif|bmp))"
        r"(?!\s*的文件)(?!\s*的图片)",
        re.IGNORECASE,
    ),
)
_SAVE_FOLDER_REQUIRED_PATTERN = re.compile(
    r"(?:保存|另存|下载)[^,。;]{0,12}(?:到|至)[^,。;]{0,20}"
    r"(?:目录|文件夹|路径|folder)",
    re.IGNORECASE,
)
# P2 folder navigation:已知 shell 文件夹或指令明确给出的绝对路径。
# 仅作为"要键入 Save 对话框的文本"来源,不做任何文件系统操作。
_SAVE_FOLDER_KNOWN_PATTERN = re.compile(
    r"(?:保存|另存|下载|存)[^,。;]{0,12}(?:到|至)\s*"
    r"(桌面|Desktop|下载(?:文件夹)?|Downloads|文档|Documents|图片|Pictures)",
    re.IGNORECASE,
)
# 命名子文件夹:"桌面的"X"文件夹中"/"下载文件夹中的"X"文件夹"——
# 已知基座 + 引号内子目录名(子目录名完全来自用户可见 instruction,
# 不做任何 benchmark 特判)。
_SAVE_FOLDER_NAMED_PATTERN = re.compile(
    r"(桌面|Desktop|下载(?:文件夹)?|Downloads|文档|Documents|图片|Pictures)"
    r'(?:中的|的)?\s*["“]([^"”]{1,60})["”]\s*(?:文件夹|目录)',
    re.IGNORECASE,
)
_SAVE_FOLDER_PATH_PATTERN = re.compile(
    r"(?:保存|另存|下载|存)[^,。;]{0,12}(?:到|至)\s*"
    r"([A-Za-z]:\\[^\s。,;\"”]{1,120})",
    re.IGNORECASE,
)
# 已知文件夹名 -> shell folder 标识(resolve 层换算为要键入的路径文本)。
_KNOWN_SAVE_FOLDERS = {
    "桌面": "Desktop",
    "desktop": "Desktop",
    "下载": "Downloads",
    "下载文件夹": "Downloads",
    "downloads": "Downloads",
    "文档": "Documents",
    "documents": "Documents",
    "图片": "Pictures",
    "pictures": "Pictures",
}
# P3 CROSS_APP_CONTENT_TRANSFER:复制类动词 + 目标方向词 + 目标应用
# 别名(来自 APP_LAUNCH_MAPPINGS 的通用身份)共现才成立;"只复制/从X复制/
# 复制文件到桌面"等单动词或非文本方向不触发。
_TRANSFER_COPY_VERBS = r"(?:复制|拷贝|粘贴|paste|copy)"
_TRANSFER_DIRECTION = r"(?:到|至|进|入|到新建|到新的|然后粘贴到|并放进|并粘贴到|再粘贴)"
_TRANSFER_TARGET_PREFIX = r"(?:第一行写|第一行输入标题|先输入标题|标题写|标题为|先写)"


@dataclass(frozen=True)
class TaskExpectation:
    """保存可程序验证的任务期望;全部字段缺省 None 表示不可验证。"""

    expected_app: str | None = None
    expected_app_processes: tuple[str, ...] = ()
    expected_app_title_keywords: tuple[str, ...] = ()
    expected_text: str | None = None
    expected_numeric_result: int | None = None
    expected_volume: int | None = None
    expected_volume_tolerance: int = 5
    expected_window_closed: int | None = None
    delivery_intent: bool = False
    expected_delivery_payload: str | None = None
    save_image_intent: bool = False
    expected_save_filename: str | None = None
    expected_save_folder: str | None = None
    save_folder_required: bool = False
    cross_app_transfer_intent: bool = False
    transfer_target_app: str | None = None
    transfer_target_app_processes: tuple[str, ...] = ()
    transfer_target_prefix_text: str | None = None
    transfer_source_text_hint: str | None = None

    def is_empty(self) -> bool:
        """没有任何可验证期望时返回 True。"""
        return not any(
            (
                self.expected_app,
                self.expected_text,
                self.expected_numeric_result is not None,
                self.expected_volume is not None,
                self.expected_window_closed is not None,
                self.delivery_intent,
                self.save_image_intent,
                self.cross_app_transfer_intent,
            ),
        )


def _eval_expression(expression: str) -> int | None:
    """计算窄格式二元算术表达式;无法整除或任何意外返回 None。

    表达式由 ``_CALC_PATTERN`` 上游约束为两个 1-4 位整数与单个
    ``+ - × ÷`` 运算符;``÷`` 为真实除法,结果非整数时按无可验证
    期望处理。以显式分支替代 eval,消除编码规范 9.2 的 eval 使用。
    """
    compact = expression.replace(" ", "")
    match = re.fullmatch(r"(\d{1,4})([+\-×÷])(\d{1,4})", compact)
    if match is None:
        return None
    left = int(match.group(1))
    right = int(match.group(3))
    operator = match.group(2)
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "×":
        return left * right
    if right == 0:
        return None
    quotient = left / right
    return int(quotient) if quotient.is_integer() else None


def extract_task_expectation(task_text: str) -> TaskExpectation:
    """从任务文本抽取窄模式期望;匹配不到的字段保持 None。

    只覆盖已知指令模板中高度可靠的窄模式:音量目标百分比、计算器
    算式,以及 PART B-E 的应用启动/投递/保存/跨应用意图;输入内容
    标识类抽取已按 PRD-BC-001 裁决移除(见模块常量注释)。
    不做通用自然语言理解。
    """
    if not isinstance(task_text, str) or not task_text.strip():
        return TaskExpectation()
    volume_match = _VOLUME_PATTERN.search(task_text)
    expected_volume = int(volume_match.group(1)) if volume_match else None
    calc_match = _CALC_PATTERN.search(task_text)
    expected_numeric = _eval_expression(calc_match.group(1)) if calc_match else None
    expected_text = None
    expected_window_closed = (
        ANY_TRACKED_WINDOW if _CLOSE_WINDOW_PATTERN.search(task_text) else None
    )
    # PART B:纯应用启动 intent 抽取 expected_app + 进程标识。
    expected_app = None
    expected_app_processes: tuple[str, ...] = ()
    expected_app_title_keywords: tuple[str, ...] = ()
    from agent.semantic_routes import extract_app_launch_info

    app_info = extract_app_launch_info(task_text)
    if app_info is not None:
        expected_app = app_info["canonical_name"]
        expected_app_processes = tuple(app_info["process_names"])
        expected_app_title_keywords = tuple(app_info["aliases"])
    elif expected_numeric is not None and "计算器" in task_text:
        # 复合计算任务不属于“纯启动”路线，但完成证据仍必须绑定到
        # Calculator 前台，不能由其他窗口中的相同数字触发。
        expected_app = "Calculator"
        expected_app_processes = (
            "calculatorapp.exe",
            "calculator.exe",
        )
        expected_app_title_keywords = ("Calculator", "计算器")
    # PART C(P1):文本投递意图与 payload;仅当指令语义是"发送/回复
    # 一段文本"时置位,复制/搜索/写到/删除等动词不触发。
    delivery_payload = _extract_delivery_payload(task_text)
    # PART D(P2):图片保存/下载意图、期望文件名、目录期望与要求。
    save_image = any(p.search(task_text) for p in _SAVE_IMAGE_INTENT_PATTERNS)
    save_filename = _extract_save_filename(task_text)
    save_folder = _extract_save_folder(task_text)
    save_folder_required = bool(_SAVE_FOLDER_REQUIRED_PATTERN.search(task_text)) or (
        save_folder is not None
    )
    # PART E(P3):跨应用文本搬运意图(复制动词 + 方向 + 目标应用别名
    # 共现)、目标应用身份与可选前置标题;source 文本提示仅当指令
    # 明确引用具体文本时记录,绝不来自 benchmark expected body。
    transfer_target = _extract_transfer_target(task_text)
    transfer_prefix = _extract_transfer_prefix(task_text)
    return TaskExpectation(
        expected_app=expected_app,
        expected_app_processes=expected_app_processes,
        expected_app_title_keywords=expected_app_title_keywords,
        expected_numeric_result=expected_numeric,
        expected_volume=expected_volume,
        expected_text=expected_text,
        expected_window_closed=expected_window_closed,
        delivery_intent=delivery_payload is not None,
        expected_delivery_payload=delivery_payload,
        save_image_intent=save_image,
        expected_save_filename=save_filename,
        expected_save_folder=save_folder,
        save_folder_required=save_folder_required,
        cross_app_transfer_intent=transfer_target is not None,
        transfer_target_app=(
            transfer_target["canonical_name"] if transfer_target else None
        ),
        transfer_target_app_processes=(
            tuple(transfer_target["process_names"]) if transfer_target else ()
        ),
        transfer_target_prefix_text=transfer_prefix,
    )


def _extract_transfer_target(task_text: str) -> dict | None:
    """识别"复制/粘贴 … 到 …目标应用"的跨应用搬运目标。

    目标应用候选来自 APP_LAUNCH_MAPPINGS 的通用别名表;要求复制类
    动词与目标应用别名都在句中出现,且存在方向词(到/进/入等)或
    "然后粘贴"衔接,避免"打开记事本/只复制"误判。
    """
    from agent.semantic_routes import APP_LAUNCH_MAPPINGS

    # 搜索/查找带引号短语(如 搜索“复制到记事本”)是查询语义,不是搬运。
    if re.search(r'(?:搜索|查找)\s*["“]', task_text):
        return None
    if not re.search(_TRANSFER_COPY_VERBS, task_text, re.IGNORECASE):
        return None
    if not re.search(_TRANSFER_DIRECTION, task_text):
        return None
    for mapping in APP_LAUNCH_MAPPINGS.values():
        for alias in mapping["aliases"]:
            if alias.lower() in task_text.lower():
                return {
                    "canonical_name": mapping["search_text"],
                    "process_names": mapping["process_names"],
                }
    return None


def _extract_transfer_prefix(task_text: str) -> str | None:
    """抽取粘贴前要输入的目标前置标题(第一行/标题)。

    支持"第一行写X/先输入标题X/标题写X/标题为X/先写X再粘贴",
    X 为引号或空格界定的短文本;未要求返回 None(不发明内容)。
    """
    patterns = (
        # 引号形式;后随"的那段/正文/段落"是源段落描述(如
        # 标题为"发布时间"的那段正文),不是要输入的标题,显式拒绝。
        rf'{_TRANSFER_TARGET_PREFIX}\s*["“]([^"”]{{1,40}})["”]'
        r"(?!\s*的(?:那段|正文|段落|部分|内容|一行))",
        rf"{_TRANSFER_TARGET_PREFIX}\s*[:：]\s*([一-龥A-Za-z0-9_\-]{{1,40}})",
        rf"{_TRANSFER_TARGET_PREFIX}\s*([一-龥A-Za-z0-9_\-]{{1,40}})",
    )
    for pattern in patterns:
        match = re.search(pattern, task_text)
        if match:
            text = match.group(1).strip()
            if text:
                return text
    return None


def _extract_save_folder(task_text: str) -> str | None:
    """抽取保存目标目录:命名子文件夹、已知 shell 文件夹或绝对路径。

    命名子文件夹归一为 ``<基座>\\<子目录>`` 复合标识(如
    ``Desktop\\Reports``);已知文件夹归一为 Desktop/Downloads 等
    标识;绝对路径原样保留。全部仅作为 resolve 层的键入文本来源。
    """
    match = _SAVE_FOLDER_PATH_PATTERN.search(task_text)
    if match:
        return match.group(1).rstrip("\\")
    match = _SAVE_FOLDER_NAMED_PATTERN.search(task_text)
    if match:
        subfolder = match.group(2).strip().strip("\\/")
        base = _KNOWN_SAVE_FOLDERS.get(match.group(1).lower())
        if subfolder and base:
            return f"{base}\\{subfolder}"
    match = _SAVE_FOLDER_KNOWN_PATTERN.search(task_text)
    if match:
        return _KNOWN_SAVE_FOLDERS[match.group(1).lower()]
    return None


def _extract_save_filename(task_text: str) -> str | None:
    """从指令抽取明确指定的保存文件名;未指定返回 None(保留默认名)。"""
    for pattern in _SAVE_FILENAME_PATTERNS:
        match = pattern.search(task_text)
        if match:
            return match.group(1).strip().strip('"“”')
    return None


def _extract_delivery_payload(task_text: str) -> str | None:
    """从指令文本抽取投递 payload;非投递语义返回 None。

    依次尝试引号形式(发送"X"/把"X"发给)与冒号形式(发一条消息：X);
    命中即返回去空白后的 payload,否则 None(交付意图不成立)。
    """
    for pattern in _DELIVERY_QUOTED_PATTERNS:
        match = pattern.search(task_text)
        if match:
            payload = match.group(1).strip()
            if payload:
                return payload
    return None


@dataclass(frozen=True)
class VerificationResult:
    """一次完成验证的三态结论与安全证据。"""

    status: VerificationStatus
    reason: str
    evidence: dict[str, object] = field(default_factory=dict)


def _volume_verdict(
    expectation: TaskExpectation,
    facts: dict[str, object],
) -> VerificationResult | None:
    if expectation.expected_volume is None:
        return None
    current = facts.get("current_volume")
    target = expectation.expected_volume
    tolerance = expectation.expected_volume_tolerance
    if not isinstance(current, int):
        return VerificationResult(
            "UNKNOWN",
            f"current_volume unavailable, target_volume={target}",
        )
    ok = abs(current - target) <= tolerance
    return VerificationResult(
        "VERIFIED" if ok else "NOT_VERIFIED",
        f"current_volume={current}, target_volume={target}",
        {"current_volume": current, "target_volume": target},
    )


def _overlay_or_ocr_verdict(
    expectation: TaskExpectation,
    facts: dict[str, object],
    needle: str | None,
    label: str,
) -> VerificationResult | None:
    """数值/文本证据共用:前台是 Shell 叠层→NOT_VERIFIED,OCR 命中→VERIFIED。"""
    if needle is None:
        return None
    foreground = str(facts.get("foreground_process") or "").lower()
    evidence: dict[str, object] = {
        "foreground_process": foreground or "unknown",
        "needle": needle,
    }
    if foreground in SHELL_OVERLAY_PROCESSES:
        return VerificationResult(
            "NOT_VERIFIED",
            f"foreground is shell overlay '{foreground}'; expected {label} "
            f"'{needle}' not evidenced on current screen",
            evidence,
        )
    ocr_text = str(facts.get("ocr_text") or "")
    if needle in ocr_text:
        return VerificationResult(
            "VERIFIED",
            f"expected {label} '{needle}' found in current OCR evidence",
            evidence,
        )
    return VerificationResult(
        "UNKNOWN",
        f"expected {label} '{needle}' not found in current OCR; "
        "no decisive counter-evidence (OCR may have missed it)",
        evidence,
    )


def _window_closed_verdict(
    expectation: TaskExpectation,
    facts: dict[str, object],
) -> VerificationResult | None:
    if expectation.expected_window_closed is None:
        return None
    exists = facts.get("tracked_window_exists")
    if not isinstance(exists, bool):
        return VerificationResult(
            "UNKNOWN",
            "tracked window existence unavailable",
        )
    return VerificationResult(
        "VERIFIED" if not exists else "NOT_VERIFIED",
        f"tracked window exists={exists}",
        {"tracked_window_exists": exists},
    )


def _app_foreground_verdict(
    expectation: TaskExpectation,
    facts: dict[str, object],
) -> VerificationResult | None:
    """expected_app:需本 task 产生的真实状态变化,非仅进程存在。

    CASE A(pre 无窗口):本 run 观察到目标应用 visible/foreground → VERIFIED。
    CASE B(pre 已有窗口):需出现新 HWND(post - pre 非空)才 VERIFIED;
    仅 foreground 匹配(可能是 pre-existing)不充分。
    """
    if expectation.expected_app is None:
        return None
    foreground = str(facts.get("foreground_process") or "").lower()
    expected_processes = expectation.expected_app_processes
    title_matches = facts.get("foreground_title_matches_expected") is True
    pre_hwnds = facts.get("pre_target_hwnds")
    post_hwnds = facts.get("post_target_hwnds")
    evidence: dict[str, object] = {
        "foreground_process": foreground or "unknown",
        "expected_app": expectation.expected_app,
        "expected_processes": list(expected_processes),
        "foreground_title_matches_expected": title_matches,
    }
    if isinstance(pre_hwnds, (set, frozenset)) and isinstance(
        post_hwnds,
        (set, frozenset),
    ):
        evidence["pre_target_hwnds"] = sorted(pre_hwnds)
        evidence["post_target_hwnds"] = sorted(post_hwnds)
        new_hwnds = post_hwnds - pre_hwnds
        evidence["new_target_hwnds"] = sorted(new_hwnds)
        if new_hwnds:
            return VerificationResult(
                "VERIFIED",
                f"new {expectation.expected_app} window(s) appeared: "
                f"{sorted(new_hwnds)}",
                evidence,
            )
        if not pre_hwnds and post_hwnds:
            return VerificationResult(
                "VERIFIED",
                f"{expectation.expected_app} appeared (was absent pre-run)",
                evidence,
            )
        if pre_hwnds:
            return VerificationResult(
                "NOT_VERIFIED",
                f"{expectation.expected_app} pre-existing, no new window "
                f"(pre={len(pre_hwnds)}, post={len(post_hwnds)})",
                evidence,
            )
    # 无快照时退化为前台进程匹配(旧逻辑,仅为兼容)。
    if not foreground or foreground == "unknown":
        return VerificationResult(
            "UNKNOWN",
            f"foreground process unavailable, expected {expectation.expected_app}",
            evidence,
        )
    if expected_processes:
        for proc in expected_processes:
            if proc.lower() in foreground:
                return VerificationResult(
                    "VERIFIED",
                    f"foreground '{foreground}' matches expected "
                    f"{expectation.expected_app} (no snapshot)",
                    evidence,
                )
        if title_matches:
            return VerificationResult(
                "VERIFIED",
                f"foreground title matches expected {expectation.expected_app}",
                evidence,
            )
        return VerificationResult(
            "NOT_VERIFIED",
            f"foreground '{foreground}' does not match expected "
            f"{expectation.expected_app}",
            evidence,
        )
    return VerificationResult(
        "UNKNOWN",
        f"no process mapping for expected {expectation.expected_app}",
        evidence,
    )


def verify_completion(
    expectation: TaskExpectation,
    facts: dict[str, object],
) -> VerificationResult:
    """按结构化期望与当前事实给出三态结论。

    Args:
        expectation: 任务期望;空期望直接 UNKNOWN。
        facts: current_volume / ocr_text / foreground_process /
            tracked_window_exists 等程序事实,缺失按不可判定处理。

    Returns:
        聚合结论:任一维度 NOT_VERIFIED 即 NOT_VERIFIED;
        全部 VERIFIED 才 VERIFIED;否则 UNKNOWN。
    """
    if expectation.is_empty():
        return VerificationResult(
            "UNKNOWN",
            "no structured expectation available for this task",
        )
    verdicts = [
        v
        for v in (
            _volume_verdict(expectation, facts),
            _app_foreground_verdict(expectation, facts),
            _overlay_or_ocr_verdict(
                expectation,
                facts,
                (
                    str(expectation.expected_numeric_result)
                    if expectation.expected_numeric_result is not None
                    else None
                ),
                "numeric result",
            ),
            _overlay_or_ocr_verdict(
                expectation,
                facts,
                expectation.expected_text,
                "text",
            ),
            _window_closed_verdict(expectation, facts),
        )
        if v is not None
    ]
    # 空 verdicts 是合法状态:期望存在(如 delivery-only)但当前没有
    # 任何适用判定器时,必须返回 UNKNOWN,绝不触发未设防 next() 的
    # StopIteration 逃逸(该逃逸曾使任务静默卡死,见 P1 诊断报告)。
    if not verdicts:
        return VerificationResult("UNKNOWN", "no_applicable_verdicts")
    if any(v.status == "NOT_VERIFIED" for v in verdicts):
        blocker = next(v for v in verdicts if v.status == "NOT_VERIFIED")
        return VerificationResult("NOT_VERIFIED", blocker.reason, blocker.evidence)
    if all(v.status == "VERIFIED" for v in verdicts):
        return verdicts[0]
    unknown = next(v for v in verdicts if v.status == "UNKNOWN")
    return VerificationResult("UNKNOWN", unknown.reason, unknown.evidence)


@dataclass(frozen=True)
class StrategyObservation:
    """一条带坐标动作对可量化事实的效果记录。"""

    action_type: str
    fact_before: float
    fact_after: float
    distance_before: float
    distance_after: float


class ProgressTracker:
    """跟踪单一可量化事实向目标推进的速率,最多保留最近三条观察。"""

    MAX_OBSERVATIONS = 3

    def __init__(self, target: float | None) -> None:
        self._target = target
        self._observations: list[StrategyObservation] = []

    @property
    def target(self) -> float | None:
        """本统计策略期望命中的目标值(如目标音量百分比)。"""
        return self._target

    def record(
        self,
        action_type: str,
        fact_before: float | None,
        fact_after: float | None,
    ) -> None:
        """记录一步前后的事实值;任一端缺失或无目标时忽略。"""
        if self._target is None or fact_before is None or fact_after is None:
            return
        self._observations.append(
            StrategyObservation(
                action_type=action_type,
                fact_before=fact_before,
                fact_after=fact_after,
                distance_before=abs(fact_before - self._target),
                distance_after=abs(fact_after - self._target),
            ),
        )
        self._observations = self._observations[-self.MAX_OBSERVATIONS :]

    def evaluate(
        self,
        current: float | None,
        steps_remaining: int,
    ) -> tuple[ProgressStatus, str]:
        """按最近观察估算速率并判定推进状态。

        判定顺序:无量化事实→UNKNOWN;最新动作未缩短目标距离→
        NO_PROGRESS;最新动作虽推进但剩余预算不够→INSUFFICIENT_RATE;
        否则 PROGRESSED。历史振荡不得覆盖最新动作已经产生的真实推进。
        Controller 只报告事实与预算约束,不指定具体 GUI 策略。
        """
        if self._target is None or current is None:
            return "UNKNOWN", "no quantitative fact tracked for this task"
        remaining = abs(current - self._target)
        if remaining == 0:
            return "PROGRESSED", f"current={current} reached target"
        if not self._observations:
            return "UNKNOWN", "no observed action effect on the tracked fact yet"
        latest = self._observations[-1]
        per_action = latest.distance_before - latest.distance_after
        if per_action <= 0:
            return (
                "NO_PROGRESS",
                f"latest action is not moving toward the target "
                f"(current={current}, target={self._target})",
            )
        if per_action * max(steps_remaining, 0) < remaining:
            return (
                "INSUFFICIENT_RATE",
                f"current={current:g}, target={self._target:g}, "
                f"recent_delta_per_action≈{per_action:.1f}, "
                f"steps_remaining={steps_remaining}, "
                f"required_delta={remaining:g}; current rate cannot reach "
                "the target within the remaining step budget",
            )
        return (
            "PROGRESSED",
            f"current={current:g} moving toward target={self._target:g} "
            f"at ≈{per_action:.1f}/action with {steps_remaining} steps left",
        )

    def observations_summary(self) -> tuple[str, ...]:
        """输出最多三条策略观察的单行摘要,供动态状态注入。"""
        return tuple(
            f"{obs.action_type}: {obs.fact_before:g}->{obs.fact_after:g} "
            f"(distance {obs.distance_before:g}->{obs.distance_after:g})"
            for obs in self._observations
        )
