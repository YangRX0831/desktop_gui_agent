"""P2 MULTI_STEP_DOWNLOAD_SAVE_COMPLETION:保存对话框状态转移验证。

通用链路:SAVE INTENT → save-like 触发 → 系统对话框到达(状态转移)
→ 既有 Save-As 完成路线 → 对话框关闭/前台回归 → VERIFIED。

全部判定基于 before/after 状态转移与结构化 OCR box(0..1000 归一化),
无绝对屏幕坐标、无 benchmark 特判;文件系统事实仅归 benchmark
verifier,不作为 Agent 完成证据。第一版只覆盖 IMAGE 内容保存
(intent_type=SAVE_CONTENT),不扩展其它对话框类型。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

# STRONG:完整图片另存类标签,可直接作为 save-like 触发证据。
STRONG_SAVE_LABELS: tuple[str, ...] = (
    "图片另存为",
    "图像另存为",
    "另存图片",
    "保存图片",
    "Save image as",
    "Save picture as",
)
# SUPPORTING:部分词,只有叠加图片意图+弹层上下文等事实才可考虑。
SUPPORTING_SAVE_LABELS: tuple[str, ...] = ("另存为", "Save as")
# 明确非图片保存对象:出现即判 NOT_IMAGE_SAVE,不得触发。
NON_IMAGE_SAVE_LABELS: tuple[str, ...] = (
    "保存网页",
    "网页另存为",
    "Save page as",
)

# click 命中标签的空间阈值(0..1000 归一化坐标)。
SAVE_LABEL_PROXIMITY = 80.0
# 对话框到达轮询(本地、有界;不增加模型调用)。
DIALOG_POLL_INTERVAL_S = 0.12
DIALOG_POLL_TOTAL_S = 2.5
# 保存动作后的关闭确认轮询。
CLOSE_POLL_INTERVAL_S = 0.15
CLOSE_POLL_TOTAL_S = 1.5
# 同一触发签名的重复判定步窗口。
DUPLICATE_TRIGGER_STEP_WINDOW = 3
# 触发签名的 bbox 量化粒度(归一化坐标)。
TRIGGER_BBOX_QUANTUM = 50

_DIALOG_CLASS = "#32770"
_SAVE_TITLE_KEYWORDS = ("Save As", "另存为", "保存", "Save")


@dataclass(frozen=True)
class MenuCandidate:
    """一个 save-like 上下文菜单候选及其等级。"""

    level: str  # STRONG | SUPPORTING | NOT_IMAGE_SAVE
    label: str
    bbox: tuple[int, int, int, int]


def classify_save_menu_candidate(
    boxes: list[dict],
    click_point: tuple[int, int] | None = None,
) -> MenuCandidate | None:
    """在当前感知 OCR box 中识别 save-like 菜单候选。

    STRONG:完整图片另存标签(可选要求与 click 点邻近);
    NOT_IMAGE_SAVE:命中"保存网页/Save page as"类标签;
    SUPPORTING:仅有"另存为/Save as"部分词;无命中返回 None。
    """
    for label in NON_IMAGE_SAVE_LABELS:
        hit = _find_label(label, boxes)
        if hit is not None and _within(click_point, hit):
            return MenuCandidate("NOT_IMAGE_SAVE", label, hit)
    for label in STRONG_SAVE_LABELS:
        hit = _find_label(label, boxes)
        if hit is not None and _within(click_point, hit):
            return MenuCandidate("STRONG", label, hit)
    for label in SUPPORTING_SAVE_LABELS:
        hit = _find_label(label, boxes)
        if hit is not None and _within(click_point, hit):
            return MenuCandidate("SUPPORTING", label, hit)
    return None


def _find_label(
    label: str,
    boxes: list[dict],
) -> tuple[int, int, int, int] | None:
    needle = "".join(label.lower().split())
    for item in boxes:
        text = "".join(str(item.get("text", "")).lower().split())
        if not text or needle not in text:
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, tuple) and len(bbox) == 4:
            return bbox
    return None


def _within(
    click_point: tuple[int, int] | None,
    bbox: tuple[int, int, int, int],
) -> bool:
    if click_point is None:
        return True
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    distance = (
        (click_point[0] - center[0]) ** 2 + (click_point[1] - center[1]) ** 2
    ) ** 0.5
    return distance <= SAVE_LABEL_PROXIMITY


def trigger_signature(label: str, bbox: tuple[int, int, int, int]) -> str:
    """触发签名:标签 + 量化 bbox,用于重复触发事实记录。"""
    quantized = tuple((v // TRIGGER_BBOX_QUANTUM) * TRIGGER_BBOX_QUANTUM for v in bbox)
    return f"{label}@{quantized}"


@dataclass
class PendingDialogTransition:
    """task 局部保存对话框转移状态;新任务开始时必须整体清空。"""

    intent_type: str = "SAVE_CONTENT"
    expected_filename: str | None = None
    trigger_step: int | None = None
    trigger_action: str | None = None
    trigger_label: str | None = None
    trigger_bbox: tuple[int, int, int, int] | None = None
    trigger_process: str = ""
    pre_dialog_hwnds: frozenset[int] = frozenset()
    other_dialog_hwnds_at_save: frozenset[int] = frozenset()
    armed_timestamp: float = 0.0
    dialog_hwnd: int | None = None
    dialog_process: str = ""
    dialog_title: str = ""
    dialog_arrived_step: int | None = None
    save_route_dispatched: bool = False
    last_trigger_signature: str = ""
    last_trigger_step: int | None = None
    last_trigger_result: str = ""
    trigger_attempts: int = 0

    def arm_trigger(
        self,
        step: int,
        action_type: str,
        candidate: MenuCandidate,
        trigger_process: str,
        pre_dialog_hwnds: frozenset[int],
    ) -> None:
        """记录一次 save-like 触发与触发前对话框快照。"""
        self.trigger_step = step
        self.trigger_action = action_type
        self.trigger_label = candidate.label
        self.trigger_bbox = candidate.bbox
        self.trigger_process = trigger_process
        self.pre_dialog_hwnds = pre_dialog_hwnds
        self.armed_timestamp = time.monotonic()
        self.dialog_hwnd = None
        self.dialog_arrived_step = None
        self.trigger_attempts += 1
        self.last_trigger_signature = trigger_signature(candidate.label, candidate.bbox)
        self.last_trigger_step = step
        self.last_trigger_result = "PENDING"

    def mark_trigger_failed(self, step: int) -> None:
        """有界等待后对话框未到达:记录失败供 policy 避免盲循环。"""
        self.last_trigger_result = "NO_DIALOG"
        self.trigger_step = None

    def is_duplicate_trigger(
        self,
        candidate: MenuCandidate,
        step: int,
    ) -> bool:
        """同一签名在短步窗内重复且上次无对话框 → 重复触发。"""
        if self.last_trigger_step is None or self.last_trigger_result != "NO_DIALOG":
            return False
        if step - self.last_trigger_step > DUPLICATE_TRIGGER_STEP_WINDOW:
            return False
        return (
            trigger_signature(candidate.label, candidate.bbox)
            == self.last_trigger_signature
        )


@dataclass(frozen=True)
class ArrivalResult:
    """对话框到达判定结论与证据。"""

    status: str  # ARRIVED | NOT_ARRIVED
    hwnd: int | None = None
    process: str = ""
    title: str = ""
    reason: str = ""


def evaluate_dialog_arrival(
    pre_dialog_hwnds: frozenset[int],
    post_windows: list[dict],
    trigger_process: str = "",
) -> ArrivalResult:
    """按 pre/post 状态转移判定保存对话框是否到达。

    CASE A pre 无、post 新 #32770 → ARRIVED;CASE B pre 有无关 A、
    post A+B → B 为新候选;CASE C 无新增 → NOT_ARRIVED;CASE D 新
    #32770 属其它进程 → NOT_ARRIVED(process_mismatch)。标题仅作
    supporting 记录,不单独构成到达。
    """
    for window in post_windows:
        hwnd = window.get("hwnd")
        if hwnd in pre_dialog_hwnds:
            continue
        if window.get("class") != _DIALOG_CLASS:
            continue
        process = str(window.get("process", ""))
        if trigger_process and process.lower() != trigger_process.lower():
            continue
        return ArrivalResult(
            "ARRIVED",
            hwnd=hwnd,
            process=process,
            title=str(window.get("title", "")),
            reason="new_32770_state_transition",
        )
    return ArrivalResult("NOT_ARRIVED", reason="no_new_matching_dialog")


@dataclass(frozen=True)
class DialogPollTiming:
    """对话框到达轮询的时序配置;sleep/时钟为可注入测试接缝。"""

    interval_s: float = DIALOG_POLL_INTERVAL_S
    total_s: float = DIALOG_POLL_TOTAL_S
    sleep_fn: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic


def poll_for_dialog(
    poll_fn: Callable[[], list[dict]],
    pre_dialog_hwnds: frozenset[int],
    trigger_process: str = "",
    timing: DialogPollTiming | None = None,
) -> ArrivalResult:
    """有界轮询对话框到达;poll_fn 返回当前窗口列表。超时 NOT_ARRIVED。"""
    if timing is None:
        timing = DialogPollTiming()
    deadline = timing.monotonic() + timing.total_s
    while True:
        result = evaluate_dialog_arrival(
            pre_dialog_hwnds,
            poll_fn(),
            trigger_process,
        )
        if result.status == "ARRIVED":
            return result
        if timing.monotonic() >= deadline:
            return result
        timing.sleep_fn(timing.interval_s)


@dataclass(frozen=True)
class PostSaveResult:
    """保存动作后的对话框关闭/前台回归判定。"""

    status: str  # VERIFIED | NOT_VERIFIED | SECONDARY_DIALOG_PRESENT
    reason: str
    evidence: dict[str, object] = field(default_factory=dict)


def evaluate_post_save(
    dialog_hwnd: int,
    post_windows: list[dict],
    foreground_process: str,
    trigger_process: str = "",
    other_dialog_hwnds_at_save: frozenset[int] = frozenset(),
) -> PostSaveResult:
    """保存后强证据:原对话框消失 + 无新二级对话框 + 前台回归触发应用。"""
    evidence: dict[str, object] = {
        "dialog_hwnd": dialog_hwnd,
        "trigger_process": trigger_process,
        "foreground_process": foreground_process,
    }
    still_open = any(w.get("hwnd") == dialog_hwnd for w in post_windows)
    evidence["dialog_closed"] = not still_open
    if still_open:
        return PostSaveResult("NOT_VERIFIED", "dialog_still_open", evidence)
    secondary = [
        w
        for w in post_windows
        if w.get("class") == _DIALOG_CLASS
        and w.get("hwnd") not in other_dialog_hwnds_at_save
        and w.get("hwnd") != dialog_hwnd
    ]
    evidence["secondary_dialog_count"] = len(secondary)
    if secondary:
        evidence["secondary_dialog_hwnd"] = secondary[0].get("hwnd")
        evidence["secondary_dialog_title"] = secondary[0].get("title", "")
        return PostSaveResult(
            "SECONDARY_DIALOG_PRESENT",
            "secondary_dialog_after_save",
            evidence,
        )
    if trigger_process and foreground_process.lower() != trigger_process.lower():
        return PostSaveResult(
            "NOT_VERIFIED",
            "foreground_not_returned_to_trigger_app",
            evidence,
        )
    return PostSaveResult("VERIFIED", "dialog_closed_and_foreground_returned", evidence)


def dialog_title_supports_save(title: str) -> bool:
    """对话框标题是否含保存类关键词(supporting 证据,非独立判据)。"""
    return any(keyword in title for keyword in _SAVE_TITLE_KEYWORDS)


# 触发证据分层:OCR_STRONG/DIALOG_CONFIRMED 可进入强完成,其余不可。
TRIGGER_OCR_STRONG = "OCR_STRONG_TRIGGER"
TRIGGER_DIALOG_CONFIRMED = "DIALOG_CONFIRMED_TRIGGER"
TRIGGER_SUPPORTING = "SUPPORTING_TRIGGER"
TRIGGER_NONE = "NONE"

# retroactive arm 的近期触发类动作回看步窗。
RETRO_TRIGGER_STEP_WINDOW = 3
_TRIGGER_LIKE_ACTIONS = frozenset({"click", "right_click", "hotkey", "type"})


@dataclass
class DialogWindowHistory:
    """task 局部对话框窗口历史:每步真实快照,供转移判定。

    不伪造 pre 集合:所有 pre 数据都来自先前真实枚举快照。
    """

    previous_dialogs: list[dict] = field(default_factory=list)
    current_dialogs: list[dict] = field(default_factory=list)
    last_snapshot_step: int | None = None
    last_foreground_process: str = ""

    def snapshot(
        self,
        step: int,
        windows: list[dict],
        foreground_process: str,
    ) -> None:
        """记录本步快照;上一次 current 成为 previous。"""
        self.previous_dialogs = self.current_dialogs
        self.current_dialogs = list(windows)
        self.last_snapshot_step = step
        self.last_foreground_process = foreground_process


def retroactive_save_dialog_arm(
    history: DialogWindowHistory,
    current_windows: list[dict],
    recent_action_types: list[str],
    save_intent: bool,
) -> tuple[str, dict] | None:
    """按真实窗口转移 retroactively 建立保存对话框证据(强条件)。

    返回 (TRIGGER_DIALOG_CONFIRMED, dialog_dict) 或 None。条件全部
    基于真实快照差集:新可见 #32770 ∧ 前台 ∧ 进程匹配最近前台应用
    ∧ 标题支持保存语义 ∧ 近 N 步存在触发类动作 ∧ 任务为保存意图。
    """
    if not save_intent:
        return None
    pre_hwnds = {w.get("hwnd") for w in history.previous_dialogs}
    foreground_process = history.last_foreground_process
    for window in current_windows:
        if window.get("hwnd") in pre_hwnds:
            continue
        if window.get("class") != _DIALOG_CLASS:
            continue
        if not window.get("foreground"):
            continue
        process = str(window.get("process", ""))
        if foreground_process and process.lower() != foreground_process.lower():
            continue
        if not dialog_title_supports_save(str(window.get("title", ""))):
            continue
        recent = recent_action_types[-RETRO_TRIGGER_STEP_WINDOW:]
        if not any(a in _TRIGGER_LIKE_ACTIONS for a in recent):
            continue
        return TRIGGER_DIALOG_CONFIRMED, window
    return None
