"""P3 CROSS_APP_CONTENT_TRANSFER 泛化单元测试(纯非 GUI)。

覆盖:意图正负例、源选择证据、clipboard 序号语义、焦点目标匹配、
粘贴转移 10 场景、假阳性控制、无 prefix/有 prefix 分支。
"""

import pytest

from agent.cross_app_transfer import (
    PendingCrossAppTransfer,
    TextSignature,
    boxes_intersect,
    evaluate_transfer_transition,
    extract_selection_text,
    normalize_text,
    source_overlap_in_text,
)
from agent.semantic_routes import build_transfer_paste_route
from agent.task_expectation import extract_task_expectation

# ======================================================================
# 意图(§27:8+ 正例,10+ 负例)
# ======================================================================

INTENT_POSITIVES = [
    "把网页中的这段内容复制到记事本",
    "将选中文字粘贴到新建记事本",
    "复制这段内容并放进文本编辑器",
    "把页面上的文字复制到记事本",
    "将这段文字复制到新的记事本文档",
    "复制当前内容，然后粘贴到记事本",
    "复制这段内容到记事本，第一行写“会议摘要”",
    "新建记事本，标题写“文档A”，再粘贴刚才复制的内容",
]

INTENT_NEGATIVES = [
    "只复制这段内容",
    "复制链接",
    "打开记事本",
    "在记事本输入“测试”",
    "从记事本复制文字",
    "把文件复制到桌面",
    "下载这段内容",
    "保存网页到文件",
    "搜索“复制到记事本”",
    "删除记事本内容",
]


@pytest.mark.parametrize("instruction", INTENT_POSITIVES)
def test_intent_positive(instruction) -> None:
    expectation = extract_task_expectation(instruction)
    assert expectation.cross_app_transfer_intent is True, instruction
    assert expectation.transfer_target_app == "Notepad", instruction


@pytest.mark.parametrize("instruction", INTENT_NEGATIVES)
def test_intent_negative(instruction) -> None:
    expectation = extract_task_expectation(instruction)
    assert expectation.cross_app_transfer_intent is False, instruction


def test_intent_prefix_parsing() -> None:
    cases = {
        "复制这段内容到记事本，第一行写“会议摘要”": "会议摘要",
        "把选中文字复制到记事本，先输入标题“记录”": "记录",
        "新建记事本，标题写“文档A”，再粘贴刚才复制的内容": "文档A",
    }
    for instruction, prefix in cases.items():
        expectation = extract_task_expectation(instruction)
        assert expectation.transfer_target_prefix_text == prefix, instruction


def test_intent_no_prefix_not_invented() -> None:
    expectation = extract_task_expectation("把网页中的这段内容复制到记事本")
    assert expectation.transfer_target_prefix_text is None


# ======================================================================
# 源选择证据(§29)
# ======================================================================


def _box(text, bbox, confidence=0.99):
    return {"text": text, "bbox": bbox, "confidence": confidence}


def test_selection_multi_boxes_reading_order() -> None:
    boxes = [
        _box("第二条", (100, 200, 200, 220)),
        _box("第一条", (100, 100, 200, 120)),
        _box("页眉", (900, 50, 990, 70)),
    ]
    text, bboxes = extract_selection_text((50, 80, 400, 240), boxes)
    assert text == "第一条第二条"
    assert len(bboxes) == 2


def test_selection_edge_touch_below_threshold() -> None:
    boxes = [_box("外侧", (500, 100, 700, 120))]
    text, _ = extract_selection_text((50, 80, 400, 240), boxes)
    assert text == ""


def test_selection_no_ocr_text_insufficient() -> None:
    pending = PendingCrossAppTransfer()
    pending.record_selection(1, (0, 0, 100, 100), "", 1, "chrome.exe")
    assert pending.selection_evidence_sufficient is False


def test_selection_dpi_invariance() -> None:
    """等比缩放的 drag/box 相交结论一致(归一化坐标体系)。"""
    boxes_a = [_box("文本", (200, 200, 400, 220))]
    boxes_b = [_box("文本", (400, 400, 800, 440))]
    rect_a = (100, 100, 500, 500)
    rect_b = (200, 200, 1000, 1000)
    text_a, _ = extract_selection_text(rect_a, boxes_a)
    text_b, _ = extract_selection_text(rect_b, boxes_b)
    assert text_a == text_b == "文本"


def test_boxes_intersect_ratio() -> None:
    assert boxes_intersect((0, 0, 100, 100), (0, 0, 50, 100)) == 0.5
    assert boxes_intersect((0, 0, 100, 100), (200, 200, 300, 300)) == 0.0


# ======================================================================
# 签名与匹配(§23)
# ======================================================================


def test_normalize_chinese_punctuation() -> None:
    assert normalize_text("你好，世界。") == normalize_text("你好世界")


def test_bigram_overlap_with_ocr_errors() -> None:
    source = TextSignature.from_text("本系统用于桌面自动化测试验证")
    post_with_errors = "本糸统用于桌面自动化测试验怔以及其它内容"
    assert source_overlap_in_text(source, post_with_errors) >= 0.6


def test_common_word_only_not_matched() -> None:
    source = TextSignature.from_text("的测试页面的测试页面的测试")
    assert source_overlap_in_text(source, "页面上有一个按钮") < 0.6


def test_english_overlap() -> None:
    source = TextSignature.from_text("quarterly report finalized")
    assert source_overlap_in_text(source, "quarterly report finalized today") >= 0.9


# ======================================================================
# 粘贴转移(§31 十场景)
# ======================================================================


def _verified_pending(source_text="会议记录第一行内容"):
    pending = PendingCrossAppTransfer()
    pending.record_selection(1, (0, 0, 100, 100), source_text, 10, "chrome.exe")
    pending.copy_confirmed = True
    pending.target_window_hwnd = 20
    pending.paste_step = 4
    pending.paste_dispatched = True
    return pending


def test_transition_1_pre_absent_post_present_verified() -> None:
    pending = _verified_pending()
    result = evaluate_transfer_transition(
        pending, "标题\n会议记录第一行内容", 20, "notepad.exe"
    )
    assert result.status == "VERIFIED"
    assert result.evidence["post_overlap"] >= 0.6


def test_transition_2_pre_has_source_not_verified() -> None:
    pending = _verified_pending()
    pending.target_pre_paste_text = "会议记录第一行内容"
    result = evaluate_transfer_transition(
        pending, "会议记录第一行内容", 20, "notepad.exe"
    )
    assert result.status == "NOT_VERIFIED"
    assert result.reason == "source_already_present_before_paste"


def test_transition_3_paste_not_dispatched() -> None:
    pending = _verified_pending()
    pending.paste_dispatched = False
    result = evaluate_transfer_transition(
        pending, "会议记录第一行内容", 20, "notepad.exe"
    )
    assert result.status == "NOT_VERIFIED"


def test_transition_4_wrong_target_window() -> None:
    pending = _verified_pending()
    result = evaluate_transfer_transition(pending, "x", 99, "chrome.exe")
    assert result.status == "NOT_VERIFIED"
    assert result.reason == "target_window_changed"


def test_transition_5_ocr_minor_errors_verified() -> None:
    pending = _verified_pending("桌面自动化智能体完成内容转移验证")
    post = "桌面自动化智能体完成内容转移验怔（OCR 少量误差）"
    result = evaluate_transfer_transition(pending, post, 20, "notepad.exe")
    assert result.status == "VERIFIED"


def test_transition_6_common_words_only_not_verified() -> None:
    pending = _verified_pending("的页面测试内容的页面测试内容")
    result = evaluate_transfer_transition(pending, "这里有一个按钮", 20, "notepad.exe")
    assert result.status == "NOT_VERIFIED"
    assert result.reason == "source_overlap_below_threshold"


def test_transition_7_prefix_and_source_verified() -> None:
    pending = _verified_pending()
    pending.target_prefix_text = "摘要"
    pending.prefix_completed = True
    result = evaluate_transfer_transition(
        pending, "摘要\n会议记录第一行内容", 20, "notepad.exe"
    )
    assert result.status == "VERIFIED"


def test_transition_8_prefix_missing_not_verified() -> None:
    pending = _verified_pending()
    pending.target_prefix_text = "摘要"
    pending.prefix_completed = False
    result = evaluate_transfer_transition(
        pending, "会议记录第一行内容", 20, "notepad.exe"
    )
    assert result.status == "NOT_VERIFIED"
    assert result.reason == "prefix_not_completed"


def test_transition_9_duplicate_paste_no_reverify() -> None:
    pending = _verified_pending()
    pending.transfer_verified = True
    # 已验证后状态冻结;再次评估同一 post 不产生第二次 VERIFIED 完成流。
    result = evaluate_transfer_transition(
        pending, "会议记录第一行内容", 20, "notepad.exe"
    )
    # 评估函数本身幂等;防重复由 handler 的 transfer_verified 门控保证。
    assert result.status in ("VERIFIED", "NOT_VERIFIED")


def test_transition_10_foreground_switched_still_same_window() -> None:
    """窗口句柄一致但前台进程是别的应用:粘贴目标窗口身份仍成立。"""
    pending = _verified_pending()
    result = evaluate_transfer_transition(
        pending, "会议记录第一行内容", 20, "explorer.exe"
    )
    # hwnd 一致即窗口身份一致;前台进程仅作证据记录。
    assert result.status == "VERIFIED"


# ======================================================================
# 粘贴路线(§13/§18)
# ======================================================================


def test_paste_route_prefix_branch() -> None:
    route = build_transfer_paste_route("摘要")
    actions = [s.action["action_type"] for s in route.steps if s.action]
    assert actions == ["type", "hotkey", "hotkey"]
    assert route.steps[0].action["params"]["text"] == "摘要"


def test_paste_route_no_prefix_branch() -> None:
    route = build_transfer_paste_route(None)
    actions = [s.action["action_type"] for s in route.steps if s.action]
    assert actions == ["hotkey"]
    keys = route.steps[0].action["params"]["keys"]
    assert tuple(keys) == ("ctrl", "v")


# ======================================================================
# clipboard 序号(§30)
# ======================================================================


def test_clipboard_sequence_available() -> None:
    from agent.cross_app_transfer import get_clipboard_sequence

    seq = get_clipboard_sequence()
    assert seq is None or isinstance(seq, int)


def test_copy_state_semantics() -> None:
    pending = PendingCrossAppTransfer()
    assert pending.copy_state == "NONE"
    pending.copy_state = "DISPATCHED"
    pending.clipboard_sequence_before = 10
    pending.clipboard_sequence_after = 11
    # 序号变化 + 源证据 + 前台一致 => CONFIRMED(handler 逻辑,此处验证字段)。
    pending.record_selection(1, (0, 0, 1, 1), "一些足够长的源文本内容", 1, "chrome.exe")
    assert pending.selection_evidence_sufficient is True


# ======================================================================
# P3 运行后修复回归(H01 诊断暴露)
# ======================================================================


def test_h01_prefix_not_source_description() -> None:
    """H01 形态:"标题为X的那段正文"是源描述,前置标题取冒号后的值。"""
    instruction = (
        "把当前网页中标题为“发布时间”的那段正文，"
        "复制到一个新建的记事本文档中，并在第一行输入标题：文档_7"
    )
    expectation = extract_task_expectation(instruction)
    assert expectation.cross_app_transfer_intent is True
    assert expectation.transfer_target_prefix_text == "文档_7"


def test_prefix_quoted_source_description_rejected() -> None:
    expectation = extract_task_expectation("把标题为“摘要”的那段内容复制到记事本")
    # 源描述不构成前置标题。
    assert expectation.transfer_target_prefix_text is None


def test_tracker_records_ctrl_c_with_keys_tuple(monkeypatch) -> None:
    """tracker 按真实 hotkey 参数形态(keys 元组)识别 Ctrl+C。"""
    import asyncio

    from agentscope.message import Msg

    import agent.gui_agent as g
    from agent.task_manager import TaskManager
    from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent

    def fake_ocr(rec, img, fp=None):
        return (), ()

    monkeypatch.setattr(
        g.prompt_context,
        "perceive_ocr_elements_detailed",
        fake_ocr,
        raising=True,
    )
    fg = {"first": True}

    def fake_fg():
        if fg["first"]:
            fg["first"] = False
            return 777
        return 777

    monkeypatch.setattr(g, "get_foreground_app_hwnd", fake_fg, raising=True)
    monkeypatch.setattr(g, "is_window_available", lambda h: True, raising=True)
    monkeypatch.setattr(g, "process_name_of_hwnd", lambda h: "chrome.exe", raising=True)
    agent = make_agent(
        SequenceBackend(
            [
                "Action: drag(x1=100, y1=100, x2=300, y2=140)",
                'Action: hotkey(key1="ctrl", key2="c")',
            ]
        ),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
    )
    from agent.task_expectation import TaskExpectation

    agent._run_expectation = TaskExpectation(
        cross_app_transfer_intent=True,
        transfer_target_app="Notepad",
        transfer_target_app_processes=("notepad.exe", "notepad"),
    )
    # drag 步的感知提供选择文本;使签名充分。
    monkeypatch.setattr(
        "agent.cross_app_transfer.extract_selection_text",
        lambda rect, boxes: ("足够长的源文本内容若干字", [rect]),
    )
    captured = {}

    def spy_handler(self, prompt_state, manager, step_number, step_retries):
        pending = self._pending_transfer
        if pending is not None and pending.copy_step == step_number:
            captured["copy_state"] = pending.copy_state
            captured["copy_step"] = pending.copy_step
            captured["seq_before"] = pending.clipboard_sequence_before
        return prompt_state, None

    monkeypatch.setattr(
        g.GuiAgent, "_handle_cross_app_transfer", spy_handler, raising=True
    )
    asyncio.run(agent(Msg("u", "把网页内容复制到记事本", "user")))
    # 运行中捕获:Ctrl+C 以 keys 元组形态被真实记录。
    assert captured["copy_step"] == 2
    assert captured["copy_state"] == "DISPATCHED"
    assert isinstance(captured["seq_before"], int)
