"""P3 CROSS_APP_CONTENT_TRANSFER:跨应用文本内容搬运的通用状态机。

SOURCE_CONTENT_SELECTED → COPY_DISPATCHED → COPY_STATE_CONFIRMED →
TARGET_WINDOW_ACQUIRED → TARGET_PREPARATION_COMPLETED → PASTE_DISPATCHED
→ TARGET_CONTENT_TRANSITION_VERIFIED。

第一版只处理 TEXT 内容(浏览器/编辑器 → 记事本/文本编辑器族)。
全部执行经 ActionDispatcher 真实键盘;焦点获取只用有界 Alt+Tab 扫描,
不用 SetForegroundWindow;clipboard 只观察序号变化,绝不读取内容。
文本匹配基于字符 n-gram 包含率,阈值通用且集中定义。
"""

import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 源选择框与 OCR box 的相交判定:box 落入 drag 矩形的面积占比阈值。
SELECTION_BOX_CONTAINMENT = 0.3
# post-paste 强证据:source 字符 bigram 在 target 文本中的包含率阈值。
SOURCE_OVERLAP_STRONG = 0.6
# pre-paste 假阳性防线:pre 中已有 source 重叠超过此值视为已存在。
SOURCE_OVERLAP_PRE_EXISTS = 0.6
# 有界 Alt+Tab 扫描的候选窗口数余量。
FOCUS_SCAN_MARGIN = 2
# 每次 Alt+Tab 后的本地等待(秒)。
FOCUS_SETTLE_SECONDS = 0.6
# bounded Alt+Tab 搜索的全局上限(通用常量,与任何具体任务无关)。
TARGET_ACQUISITION_GLOBAL_CAP = 8


def normalize_text(text: str) -> str:
    """归一化文本:去空白、统一中英标点为无分隔,小写化。"""
    lowered = text.lower()
    table = str.maketrans(
        {
            "，": "",
            "。": "",
            "；": "",
            "：": "",
            "“": "",
            "”": "",
            "、": "",
            "！": "",
            "？": "",
            "（": "",
            "）": "",
            ",": "",
            ".": "",
            ";": "",
            ":": "",
            '"': "",
            "'": "",
            "!": "",
            "?": "",
            "(": "",
            ")": "",
            " ": "",
            "\t": "",
            "\n": "",
            "\r": "",
        }
    )
    return lowered.translate(table)


def char_bigrams(text: str) -> set[str]:
    """字符 bigram 集合(中文鲁棒的最小语义单元)。"""
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


@dataclass(frozen=True)
class TextSignature:
    """一段可见文本的结构化签名(不保存 benchmark 期望内容)。"""

    normalized: str
    bigrams: frozenset[str]
    length: int
    digest: str

    @classmethod
    def from_text(cls, text: str) -> "TextSignature":
        """由原文构建文本签名(归一化正文、bigram 集、长度与摘要)。"""
        normalized = normalize_text(text)
        return cls(
            normalized=normalized,
            bigrams=frozenset(char_bigrams(text)),
            length=len(normalized),
            digest=hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8],
        )


def source_overlap_in_text(signature: TextSignature, text: str) -> float:
    """source 签名 bigram 在目标文本中的包含率(0..1)。"""
    if not signature.bigrams:
        return 0.0
    target = char_bigrams(text)
    hits = sum(1 for gram in signature.bigrams if gram in target)
    return hits / len(signature.bigrams)


def boxes_intersect(
    box: tuple[int, int, int, int],
    rect: tuple[int, int, int, int],
) -> float:
    """box 被 rect 覆盖的面积占比(0..1);坐标同为 0..1000 归一化。"""
    x1 = max(box[0], rect[0])
    y1 = max(box[1], rect[1])
    x2 = min(box[2], rect[2])
    y2 = min(box[3], rect[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return ((x2 - x1) * (y2 - y1)) / box_area


def extract_selection_text(
    drag_rect: tuple[int, int, int, int],
    boxes: list[dict],
) -> tuple[str, list[tuple[int, int, int, int]]]:
    """从 drag 矩形与结构化 OCR box 提取被选可见文本。

    仅返回与矩形显著相交(包含率达标)的 box 文本,按 y 再 x 排序
    保持阅读顺序;文本为空即为 SOURCE_SELECTION_EVIDENCE INSUFFICIENT。
    """
    hits: list[tuple[int, int, str, tuple[int, int, int, int]]] = []
    for item in boxes:
        bbox = item.get("bbox")
        text = str(item.get("text", "")).strip()
        if not text or not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        if boxes_intersect(bbox, drag_rect) >= SELECTION_BOX_CONTAINMENT:
            hits.append((bbox[1], bbox[0], text, bbox))
    hits.sort(key=lambda h: (h[0], h[1]))
    selected = [h[2] for h in hits]
    return "".join(selected), [h[3] for h in hits]


@dataclass
class PendingCrossAppTransfer:
    """task 局部跨应用搬运状态;新任务与 reply 结束时整体清空。"""

    source_window_hwnd: int | None = None
    source_process: str = ""
    source_step: int | None = None

    selection_bbox: tuple[int, int, int, int] | None = None
    source_signature: TextSignature | None = None

    clipboard_sequence_before: int | None = None
    clipboard_sequence_after: int | None = None
    pre_copy_foreground_hwnd: int | None = None
    pre_copy_foreground_process: str = ""

    copy_step: int | None = None
    copy_confirmed: bool = False
    copy_state: str = "NONE"  # NONE|DISPATCHED|CONFIRMED|PROVISIONAL

    target_app: str = ""
    target_window_hwnd: int | None = None
    target_process: str = ""

    target_prefix_text: str | None = None
    prefix_completed: bool = False

    target_pre_paste_text: str | None = None

    paste_step: int | None = None
    paste_dispatched: bool = False

    transfer_verified: bool = False
    focus_scan_dispatched: int = 0
    last_failure: str = ""
    history: list[str] = field(default_factory=list)

    def record_selection(
        self,
        step: int,
        drag_rect: tuple[int, int, int, int],
        selected_text: str,
        source_hwnd: int | None,
        source_process: str,
    ) -> None:
        """记录源选择证据(drag + OCR 相交文本签名)。"""
        self.source_step = step
        self.selection_bbox = drag_rect
        self.source_window_hwnd = source_hwnd
        self.source_process = source_process
        if selected_text:
            self.source_signature = TextSignature.from_text(selected_text)

    @property
    def selection_evidence_sufficient(self) -> bool:
        """源选择证据是否足以支撑强验证(签名非空且长度合理)。"""
        return self.source_signature is not None and self.source_signature.length >= 4


def get_clipboard_sequence() -> int | None:
    """读取 Windows clipboard 序号(只观察变化,不读内容);失败 None。"""
    import ctypes

    try:
        user32 = ctypes.windll.user32
        return int(user32.GetClipboardSequenceNumber())
    except Exception as exception:
        # 探测失败按证据不足降级;debug 留痕防止系统性失效被掩盖。
        logger.debug(
            "clipboard_sequence_probe_failed：exception_type=%s",
            type(exception).__name__,
        )
        return None


@dataclass(frozen=True)
class TransferVerdict:
    """一次 post-paste 转移判定结论与证据。"""

    status: str  # VERIFIED | NOT_VERIFIED
    reason: str
    evidence: dict[str, object] = field(default_factory=dict)


def evaluate_transfer_transition(
    pending: PendingCrossAppTransfer,
    post_target_text: str,
    target_hwnd_now: int | None,
    foreground_process: str,
) -> TransferVerdict:
    """post-paste 强验证:粘贴真实分发 + 同窗 + 新增 source 重叠。

    弱证据(粘贴已发/序号变化/文本变长/整窗常见词)一律 NOT_VERIFIED。
    """
    signature = pending.source_signature
    evidence: dict[str, object] = {
        "paste_step": pending.paste_step,
        "target_hwnd": pending.target_window_hwnd,
        "target_hwnd_now": target_hwnd_now,
        "foreground_process": foreground_process,
        "prefix_required": pending.target_prefix_text is not None,
        "prefix_completed": pending.prefix_completed,
    }
    if not pending.paste_dispatched or signature is None:
        return TransferVerdict(
            "NOT_VERIFIED", "paste_not_dispatched_or_no_source", evidence
        )
    if target_hwnd_now is not None and pending.target_window_hwnd is not None:
        if target_hwnd_now != pending.target_window_hwnd:
            return TransferVerdict("NOT_VERIFIED", "target_window_changed", evidence)
    pre_overlap = (
        source_overlap_in_text(signature, pending.target_pre_paste_text or "")
        if pending.target_pre_paste_text is not None
        else 0.0
    )
    post_overlap = source_overlap_in_text(signature, post_target_text)
    evidence["pre_overlap"] = round(pre_overlap, 2)
    evidence["post_overlap"] = round(post_overlap, 2)
    if pre_overlap >= SOURCE_OVERLAP_PRE_EXISTS:
        return TransferVerdict(
            "NOT_VERIFIED",
            "source_already_present_before_paste",
            evidence,
        )
    if post_overlap < SOURCE_OVERLAP_STRONG:
        return TransferVerdict(
            "NOT_VERIFIED",
            "source_overlap_below_threshold",
            evidence,
        )
    if pending.target_prefix_text is not None and not pending.prefix_completed:
        return TransferVerdict("NOT_VERIFIED", "prefix_not_completed", evidence)
    evidence["source_digest"] = signature.digest
    evidence["source_length"] = signature.length
    return TransferVerdict(
        "VERIFIED",
        "source_content_newly_present_in_target",
        evidence,
    )
