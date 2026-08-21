"""P1 POST_SUBMIT_DELIVERY_CONFIRMATION 泛化单元测试(纯非 GUI)。

覆盖:投递意图解析正负例、submit 候选识别、投递状态转移 10 场景、
弱证据拒绝。全部使用随机/多样 payload,不依赖 benchmark 原始 marker。
"""

import pytest

from agent.delivery_confirmation import (
    Occurrence,
    PendingSubmission,
    classify_submit_candidate,
    evaluate_delivery_transition,
    find_payload_occurrences,
)
from agent.task_expectation import TaskExpectation, extract_task_expectation

# ======================================================================
# A. 投递意图解析
# ======================================================================

POSITIVES = [
    ("给 user1 发送“会议已结束”", "会议已结束"),
    ("给 user1 发一条消息：任务编号 ZX-418 完成", "任务编号 ZX-418 完成"),
    ("向 user1 发送消息“收到，谢谢”", "收到，谢谢"),
    ("发送“今晚7点见”给 user1", "今晚7点见"),
    ("把“done-42”发给 user1", "done-42"),
    ("回复对方“测试完成”", "测试完成"),
]

NEGATIVES = [
    "打开聊天页面",
    "点击 user1",
    "复制“测试完成”",
    "搜索“测试完成”",
    "把“测试完成”写到记事本",
    "删除消息“测试完成”",
    "关闭聊天窗口",
    "阅读 user1 发送的消息",
]


@pytest.mark.parametrize("instruction,payload", POSITIVES)
def test_intent_positive(instruction, payload) -> None:
    expectation = extract_task_expectation(instruction)
    assert expectation.delivery_intent is True
    assert expectation.expected_delivery_payload == payload


@pytest.mark.parametrize("instruction", NEGATIVES)
def test_intent_negative(instruction) -> None:
    expectation = extract_task_expectation(instruction)
    assert expectation.delivery_intent is False
    assert expectation.expected_delivery_payload is None


def test_delivery_intent_counts_as_nonempty_expectation() -> None:
    assert TaskExpectation(delivery_intent=True).is_empty() is False
    assert TaskExpectation().is_empty() is True


# ======================================================================
# B. Submit 候选识别
# ======================================================================


def _boxes(*items):
    return [{"text": t, "bbox": b, "confidence": c} for t, b, c in items]


def test_submit_word_click_is_candidate() -> None:
    boxes = _boxes(
        ("发送", (900, 940, 950, 960), 0.99),
        ("user1", (50, 100, 120, 120), 0.98),
    )
    candidate = classify_submit_candidate(
        "click", {"x": 925, "y": 950}, boxes, True, True
    )
    assert candidate is not None and candidate.control_word == "发送"


def test_contact_click_is_not_candidate() -> None:
    boxes = _boxes(("user1", (50, 100, 120, 120), 0.98))
    assert (
        classify_submit_candidate("click", {"x": 85, "y": 110}, boxes, True, True)
        is None
    )


def test_plain_page_click_is_not_candidate() -> None:
    boxes = _boxes(("聊天记录", (400, 300, 500, 320), 0.9))
    assert (
        classify_submit_candidate("click", {"x": 450, "y": 310}, boxes, True, True)
        is None
    )


def test_enter_with_delivery_context_is_candidate() -> None:
    candidate = classify_submit_candidate("hotkey", {"key1": "enter"}, [], True, True)
    assert candidate is not None and candidate.control_word == "enter"


def test_ctrl_enter_with_delivery_context_is_candidate() -> None:
    candidate = classify_submit_candidate(
        "hotkey", {"key1": "ctrl", "key2": "enter"}, [], True, True
    )
    assert candidate is not None


def test_enter_without_pending_payload_is_not_candidate() -> None:
    assert (
        classify_submit_candidate("hotkey", {"key1": "enter"}, [], False, True) is None
    )


def test_enter_without_delivery_intent_is_not_candidate() -> None:
    assert (
        classify_submit_candidate("hotkey", {"key1": "enter"}, [], True, False) is None
    )


def test_submit_click_proximity_boundary() -> None:
    far = 120  # 明显超出 SUBMIT_CLICK_PROXIMITY(80)
    boxes = _boxes(("Send", (500, 500, 520, 510), 0.99))
    assert (
        classify_submit_candidate(
            "click", {"x": 500 + far, "y": 505}, boxes, True, True
        )
        is None
    )


# ======================================================================
# C. 状态转移场景
# ======================================================================


def _pending_with_pre(pre_boxes, payload="hello", foreground=777):
    pending = PendingSubmission(
        payload=payload,
        typed_step=3,
        foreground_app=foreground,
        typed_timestamp=1.0,
    )
    pending.arm_submit(4, "click", "发送", (900, 940, 950, 960))
    occurrences = find_payload_occurrences(payload, pre_boxes)
    if occurrences:
        pending.snapshot_pre(occurrences)
    return pending


def _box(text, bbox, confidence=0.99):
    return {"text": text, "bbox": bbox, "confidence": confidence}


def test_transition_1_verified() -> None:
    """pre composer 有 payload;post composer 清空且 content 区新出现 → VERIFIED。"""
    pre = [_box("hello", (400, 900, 500, 930))]
    post = [_box("hello", (380, 300, 480, 330))]
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post, 777)
    assert verdict.status == "VERIFIED"
    assert verdict.evidence["composer_cleared"] is True
    assert verdict.evidence["delivered_bbox"] == (380, 300, 480, 330)


def test_transition_2_still_in_composer_not_verified() -> None:
    pre = [_box("hello", (400, 900, 500, 930))]
    post = [_box("hello", (410, 905, 510, 935))]
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post, 777)
    assert verdict.status == "INSUFFICIENT"
    assert verdict.reason == "payload_still_in_composer"


def test_transition_3_composer_clear_no_payload_not_verified() -> None:
    pre = [_box("hello", (400, 900, 500, 930))]
    post: list[str] = []
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post, 777)
    assert verdict.status == "INSUFFICIENT"
    assert verdict.reason == "post_payload_not_found"


def test_transition_4_preexisting_history_not_verified() -> None:
    """pre 历史区已有同 payload;post 无新空间 occurrence → NOT VERIFIED。"""
    pre = [
        _box("hello", (400, 900, 500, 930)),
        _box("hello", (380, 300, 480, 330)),
    ]
    post = [_box("hello", (380, 300, 480, 330))]
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post, 777)
    assert verdict.status == "INSUFFICIENT"
    assert verdict.reason == "no_new_payload_occurrence_outside_composer"


def test_transition_5_bbox_unchanged_not_verified() -> None:
    pre = [_box("hello", (400, 900, 500, 930))]
    post = [_box("hello", (400, 900, 500, 930))]
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post, 777)
    assert verdict.status == "INSUFFICIENT"
    assert verdict.reason == "payload_still_in_composer"


def test_transition_6_no_pre_composer_observed_not_verified() -> None:
    pending = _pending_with_pre([])
    verdict = evaluate_delivery_transition(
        pending, [_box("hello", (380, 300, 480, 330))], 777
    )
    assert verdict.status == "INSUFFICIENT"
    assert verdict.reason == "pre_composer_payload_not_observed"


def test_transition_7_foreground_changed_not_verified() -> None:
    pre = [_box("hello", (400, 900, 500, 930))]
    post = [_box("hello", (380, 300, 480, 330))]
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post, 999)
    assert verdict.status == "INSUFFICIENT"
    assert verdict.reason == "foreground_changed_after_submit"


def test_transition_8_ocr_without_bbox_not_verified() -> None:
    pre = [_box("hello", (400, 900, 500, 930))]
    post_nobox = [{"text": "hello", "confidence": 0.9}]
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post_nobox, 777)
    assert verdict.status == "INSUFFICIENT"
    assert verdict.reason == "post_payload_not_found"


def test_transition_9_history_plus_new_delivered_verified() -> None:
    """多个 occurrence:一个是旧历史,另一个新 delivered → VERIFIED。"""
    pre = [
        _box("hello", (400, 900, 500, 930)),
        _box("hello", (380, 200, 480, 230)),
    ]
    post = [
        _box("hello", (380, 200, 480, 230)),
        _box("hello", (380, 420, 480, 450)),
    ]
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post, 777)
    assert verdict.status == "VERIFIED"
    assert verdict.evidence["delivered_bbox"] == (380, 420, 480, 450)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_transition_10_resolution_invariance(scale) -> None:
    """窗口尺寸/DPI 变化(坐标等比缩放)不改变判定结论。"""

    def scaled(bbox):
        return tuple(int(v * scale) for v in bbox)

    pre = [_box("hello", scaled((400, 900, 500, 930)))]
    post = [_box("hello", scaled((380, 300, 480, 330)))]
    verdict = evaluate_delivery_transition(_pending_with_pre(pre), post, 777)
    assert verdict.status == "VERIFIED"


# ======================================================================
# 弱证据拒绝(设计文档第三节)
# ======================================================================


def test_occurrence_requires_complete_payload() -> None:
    boxes = [
        _box("hel", (100, 100, 140, 120)),
        _box("say hello world", (200, 200, 320, 220)),
    ]
    found = find_payload_occurrences("hello", boxes)
    assert [item.bbox for item in found] == [(200, 200, 320, 220)]


def test_whitespace_insensitive_payload_match() -> None:
    boxes = [_box("hel lo", (100, 100, 160, 120))]
    found = find_payload_occurrences("hello", boxes)
    assert len(found) == 1


def test_case_and_punctuation_differences_do_not_match() -> None:
    """POL-001 冻结裁决(KEEP POLICY_A):仅空白归一,大小写/标点敏感。

    大小写或标点不同的 occurrence 不构成命中,不得据此 VERIFIED。
    """
    case_box = [_box("Task Done", (100, 100, 200, 130))]
    assert find_payload_occurrences("task done", case_box) == []
    assert find_payload_occurrences("TASK DONE", case_box) == []
    punct_box = [_box("任务,完成", (100, 100, 200, 130))]
    assert find_payload_occurrences("任务。完成", punct_box) == []
    assert find_payload_occurrences("任务完成", punct_box) == []


def test_disarm_submit_keeps_payload_for_resubmit() -> None:
    pending = _pending_with_pre([_box("hello", (400, 900, 500, 930))])
    pending.disarm_submit()
    assert pending.submit_step is None
    assert pending.post_submit_checked is True
    assert pending.payload == "hello"


def test_occurrence_center_math() -> None:
    item = Occurrence("x", (100, 200, 300, 400))
    assert item.center == (200.0, 300.0)
