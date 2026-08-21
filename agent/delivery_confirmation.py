"""P1 POST_SUBMIT_DELIVERY_CONFIRMATION:文本投递的强状态转移验证。

适用 type → submit → 投递状态转移 的通用任务族(聊天消息、评论、
短文本发布)。全部逻辑为本地证据计算:复用既有感知环的结构化 OCR
box(0..1000 归一化坐标),不新增任何模型调用或第二次完整 OCR。

强 VERIFIED 必须同时成立:
1. 近期真实 dispatch 过 type(payload);
2. 随后真实 dispatch 过 submit-like action;
3. 提交前 payload 位于 composer 区域(有 payload 的 OCR occurrence);
4. 提交后 composer 区域 payload 消失,且 composer 区域之外出现新的
   payload occurrence(空间上与提交前的历史 occurrence 不同);
5. 前台应用身份未切换。

任何弱证据(输入框空/整窗含 payload/出现次数等)单独出现一律
INSUFFICIENT,绝不提前 finish。
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

# submit-like 可见文本词表:generic、集中定义、带正负例测试。
# 第一版刻意不含 确定/OK/保存 等过宽词,避免把普通对话框操作误认
# 成文本投递;后续可在测试保护下扩展。
SUBMIT_LIKE_WORDS: tuple[str, ...] = (
    "发送",
    "提交",
    "发布",
    "回复",
    "Send",
    "Submit",
    "Post",
    "Reply",
)

# click 命中 submit-like 词的空间阈值(0..1000 归一化坐标下,click
# 中心与词框中心的最大距离)。
SUBMIT_CLICK_PROXIMITY = 80.0
# 两个 occurrence 视为"同一处"的最大中心距离(归一化坐标)。
OCCURRENCE_SAME_PLACE = 30.0
# composer 判定:occurrence 与 pre_payload_bbox(膨胀此比例后)
# 相交即视为仍在 composer。
COMPOSER_EXPAND_RATIO = 0.5


@dataclass(frozen=True)
class Occurrence:
    """一个 payload 的 OCR 出现位置;bbox 为 0..1000 归一化坐标。"""

    text: str
    bbox: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[float, float]:
        """bbox 的几何中心点坐标。"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class PendingSubmission:
    """task 局部的待验证投递状态;新任务开始时必须整体清空。

    payload 优先来自最近一次成功 dispatch 的 type 文本(实际输入),
    instruction 解析出的 expected payload 仅作一致性参考;两者都不
    来自 benchmark 私有元数据。
    """

    payload: str
    typed_step: int
    foreground_app: int | None
    typed_timestamp: float = 0.0
    submit_step: int | None = None
    submit_action_type: str | None = None
    submit_control_word: str | None = None
    submit_control_bbox: tuple[int, int, int, int] | None = None
    pre_payload_bbox: tuple[int, int, int, int] | None = None
    pre_occurrences: tuple[tuple[int, int, int, int], ...] = ()
    post_submit_checked: bool = False

    def arm_submit(
        self,
        step: int,
        action_type: str,
        control_word: str,
        control_bbox: tuple[int, int, int, int] | None,
    ) -> None:
        """记录一次 submit-like dispatch,并保存提交前感知快照。"""
        self.submit_step = step
        self.submit_action_type = action_type
        self.submit_control_word = control_word
        self.submit_control_bbox = control_bbox

    def snapshot_pre(self, occurrences: list[Occurrence]) -> None:
        """提交前快照:全部 payload occurrence 位置 + composer 候选。

        composer 候选取 y 最大(最靠下)的 occurrence——type-and-submit
        界面的输入区通常位于内容区下方。
        """
        self.pre_occurrences = tuple(item.bbox for item in occurrences)
        if occurrences:
            lowest = max(occurrences, key=lambda item: item.center[1])
            self.pre_payload_bbox = lowest.bbox

    def disarm_submit(self) -> None:
        """INSUFFICIENT 后解除本次 submit 记录,等待下一次真实提交。"""
        self.submit_step = None
        self.submit_action_type = None
        self.submit_control_word = None
        self.submit_control_bbox = None
        self.post_submit_checked = True


@dataclass(frozen=True)
class SubmitCandidate:
    """一次被识别为 submit-like 的动作及其控件证据。"""

    action_type: str
    control_word: str
    control_bbox: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class DeliveryVerdict:
    """投递状态转移判定结论与证据。"""

    status: str  # "VERIFIED" | "INSUFFICIENT"
    reason: str
    evidence: dict[str, object] = field(default_factory=dict)


def find_payload_occurrences(
    payload: str,
    boxes: list[dict],
) -> list[Occurrence]:
    """在结构化 OCR 结果中定位包含完整 payload 的 occurrence。

    仅接受无 bbox 的条目会被忽略(无位置证据不构成强判据);文本侧
    要求 occurrence 文本包含完整 payload(前缀粘连如"回复:payload"
    可容忍,截断命中不构成完整投递)。
    """
    occurrences: list[Occurrence] = []
    needle = "".join(payload.split())
    if not needle:
        return occurrences
    for item in boxes:
        text = "".join(str(item.get("text", "")).split())
        bbox = item.get("bbox")
        if not text or not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        if needle in text:
            occurrences.append(Occurrence(text=text, bbox=bbox))
    return occurrences


def _center_distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _expand(
    bbox: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    dx, dy = int(width * ratio / 2), int(height * ratio / 2)
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def _overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def classify_submit_candidate(
    action_type: str,
    action_params: Mapping[str, object],
    boxes: list[dict],
    has_pending_payload: bool,
    delivery_intent: bool,
) -> SubmitCandidate | None:
    """判断一个将 dispatch 的动作是否为文本投递的 submit-like 动作。

    click:click 中心附近存在 submit-like 可见词才候选(普通联系人/
    页面点击不是);Enter/Ctrl+Enter:仅当存在待投递 payload 且任务
    属投递意图时候选,普通 Enter 不触发。
    """
    if action_type == "click":
        x = action_params.get("x")
        y = action_params.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        for item in boxes:
            word = "".join(str(item.get("text", "")).split())
            for candidate_word in SUBMIT_LIKE_WORDS:
                if candidate_word in word:
                    bbox = item.get("bbox")
                    if not isinstance(bbox, tuple) or len(bbox) != 4:
                        continue
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    if _center_distance((x, y), center) <= SUBMIT_CLICK_PROXIMITY:
                        return SubmitCandidate("click", candidate_word, bbox)
        return None
    if action_type == "hotkey":
        keys = {str(action_params.get("key1", "")), str(action_params.get("key2", ""))}
        if "enter" in keys and has_pending_payload and delivery_intent:
            return SubmitCandidate("hotkey", "enter", None)
    return None


def evaluate_delivery_transition(
    pending: PendingSubmission,
    post_boxes: list[dict],
    post_foreground_app: int | None,
) -> DeliveryVerdict:
    """对提交后感知做投递状态转移判定。

    VERIFIED 需同时满足:composer payload 消失、composer 外出现新的
    payload occurrence(与提交前任何 occurrence 都不在同一位置)、前台
    应用身份未变。其余一律 INSUFFICIENT(弱证据不 finish)。
    """
    evidence: dict[str, object] = {
        "payload": pending.payload,
        "typed_step": pending.typed_step,
        "submit_step": pending.submit_step,
        "submit_action_type": pending.submit_action_type,
        "submit_control_word": pending.submit_control_word,
        "pre_payload_bbox": pending.pre_payload_bbox,
        "pre_occurrence_count": len(pending.pre_occurrences),
        "foreground_same": pending.foreground_app == post_foreground_app,
    }
    if pending.foreground_app != post_foreground_app:
        return DeliveryVerdict(
            "INSUFFICIENT",
            "foreground_changed_after_submit",
            evidence,
        )
    if pending.pre_payload_bbox is None:
        return DeliveryVerdict(
            "INSUFFICIENT",
            "pre_composer_payload_not_observed",
            evidence,
        )
    occurrences = find_payload_occurrences(pending.payload, post_boxes)
    evidence["post_occurrence_bboxes"] = [item.bbox for item in occurrences]
    if not occurrences:
        return DeliveryVerdict(
            "INSUFFICIENT",
            "post_payload_not_found",
            evidence,
        )
    composer_region = _expand(pending.pre_payload_bbox, COMPOSER_EXPAND_RATIO)
    in_composer = [item for item in occurrences if _overlap(item.bbox, composer_region)]
    if in_composer:
        return DeliveryVerdict(
            "INSUFFICIENT",
            "payload_still_in_composer",
            evidence,
        )
    pre_bboxes = [bbox for bbox in pending.pre_occurrences]
    delivered = [
        item
        for item in occurrences
        if not any(
            _center_distance(item.center, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            <= OCCURRENCE_SAME_PLACE
            for b in pre_bboxes
        )
    ]
    evidence["delivered_bbox"] = delivered[0].bbox if delivered else None
    if not delivered:
        return DeliveryVerdict(
            "INSUFFICIENT",
            "no_new_payload_occurrence_outside_composer",
            evidence,
        )
    evidence["composer_cleared"] = True
    return DeliveryVerdict(
        "VERIFIED",
        "payload_moved_from_composer_to_new_region",
        evidence,
    )
