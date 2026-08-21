"""P2 MULTI_STEP_DOWNLOAD_SAVE_COMPLETION 泛化单元测试(纯非 GUI)。

覆盖:保存意图解析正负例、save-like 菜单分类矩阵与不变性、
对话框到达/关闭状态转移 12 场景、保存路线回归、完整管线集成
(arm→arrive→route 注入→post-save VERIFIED;no-dialog 事实与
重复触发防护)。
"""

import asyncio
import ctypes

import pytest
from agentscope.message import Msg

import agent.gui_agent as gui_agent_module
from agent.dialog_transition import (
    DialogPollTiming,
    PendingDialogTransition,
    classify_save_menu_candidate,
    dialog_title_supports_save,
    evaluate_dialog_arrival,
    evaluate_post_save,
    poll_for_dialog,
    trigger_signature,
)
from agent.semantic_routes import build_save_dialog_route
from agent.task_expectation import TaskExpectation, extract_task_expectation
from agent.task_manager import TaskManager
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent

# ======================================================================
# 保存意图解析(§19)
# ======================================================================

INTENT_POSITIVES = [
    "保存这张图片",
    "把图片保存下来",
    "下载这张图片",
    "将图片另存为 photo.png",
    "把这张图保存成 cat-picture.jpg",
    "保存图片，文件名用 截图01.png",
    "请把页面里的图片下载到本地",
    "save this picture to local",
]

INTENT_NEGATIVES = [
    "打开图片",
    "复制图片",
    "搜索图片",
    "查看下载内容",
    "打开Downloads文件夹",
    "保存网页",
    "保存文档",
    "复制图片链接",
    "删除图片",
    "重命名图片",
]


@pytest.mark.parametrize("instruction", INTENT_POSITIVES)
def test_intent_positive(instruction) -> None:
    expectation = extract_task_expectation(instruction)
    assert expectation.save_image_intent is True, instruction


@pytest.mark.parametrize("instruction", INTENT_NEGATIVES)
def test_intent_negative(instruction) -> None:
    expectation = extract_task_expectation(instruction)
    assert expectation.save_image_intent is False, instruction


def test_intent_filename_extraction() -> None:
    cases = {
        "将图片另存为 photo.png": "photo.png",
        "把这张图保存成 cat-picture.jpg": "cat-picture.jpg",
        "保存图片，文件名用 截图01.png": "截图01.png",
        "另存为 capture_731.webp": "capture_731.webp",
    }
    for instruction, filename in cases.items():
        expectation = extract_task_expectation(instruction)
        assert expectation.expected_save_filename == filename, instruction


def test_intent_filename_none_when_unspecified() -> None:
    for instruction in ("保存这张图片", "把图片保存下来", "下载这张图片"):
        expectation = extract_task_expectation(instruction)
        assert expectation.expected_save_filename is None, instruction


def test_intent_folder_required_flag() -> None:
    expectation = extract_task_expectation("把图片另存到指定目录")
    assert expectation.save_image_intent is True
    assert expectation.save_folder_required is True
    plain = extract_task_expectation("保存这张图片")
    assert plain.save_folder_required is False


def test_save_page_vs_image_conservative() -> None:
    """save page 与 save image 无法安全区分时保守不触发。"""
    assert extract_task_expectation("保存网页").save_image_intent is False
    assert extract_task_expectation("Save page as").save_image_intent is False


# ======================================================================
# 菜单分类(§20)
# ======================================================================


def _box(text, bbox, confidence=0.99):
    return {"text": text, "bbox": bbox, "confidence": confidence}


def test_menu_strong_labels() -> None:
    for label, bbox in [
        ("图片另存为", (500, 300, 600, 320)),
        ("Save image as", (500, 300, 600, 320)),
        ("Save picture as", (480, 310, 610, 330)),
    ]:
        candidate = classify_save_menu_candidate([_box(label, bbox)])
        assert candidate is not None and candidate.level == "STRONG", label


def test_menu_page_save_not_image_save() -> None:
    for label in ("保存网页", "Save page as"):
        candidate = classify_save_menu_candidate([_box(label, (500, 300, 600, 320))])
        assert candidate is not None and candidate.level == "NOT_IMAGE_SAVE", label


def test_menu_copy_labels_no_candidate() -> None:
    for label in ("复制图片", "Copy image"):
        assert classify_save_menu_candidate([_box(label, (500, 300, 600, 320))]) is None


def test_menu_partial_label_supporting() -> None:
    candidate = classify_save_menu_candidate([_box("另存为", (500, 300, 560, 320))])
    assert candidate is not None and candidate.level == "SUPPORTING"


def test_menu_proximity_filters_far_click() -> None:
    boxes = [_box("图片另存为", (500, 300, 600, 320))]
    assert classify_save_menu_candidate(boxes, (950, 900)) is None
    assert classify_save_menu_candidate(boxes, (550, 310)) is not None


def test_menu_arrangement_scale_invariance() -> None:
    """bbox 平移/等比缩放不改变分类结论(HWND/相对坐标体系)。"""
    base = [_box("图片另存为", (500, 300, 600, 320))]
    moved = [_box("图片另存为", (700, 600, 800, 620))]
    scaled = [_box("图片另存为", (250, 150, 300, 160))]
    for boxes in (base, moved, scaled):
        candidate = classify_save_menu_candidate(boxes)
        assert candidate is not None and candidate.level == "STRONG"


# ======================================================================
# 对话框到达转移(§21 1-8)
# ======================================================================


def _dialog(hwnd, process="chrome.exe", title="另存为", foreground=True):
    return {
        "hwnd": hwnd,
        "class": "#32770",
        "process": process,
        "title": title,
        "foreground": foreground,
    }


def test_arrival_case_a() -> None:
    result = evaluate_dialog_arrival(frozenset(), [_dialog(11)])
    assert result.status == "ARRIVED" and result.hwnd == 11


def test_arrival_case_b() -> None:
    result = evaluate_dialog_arrival(frozenset({10}), [_dialog(10), _dialog(11)])
    assert result.status == "ARRIVED" and result.hwnd == 11


def test_arrival_case_c() -> None:
    result = evaluate_dialog_arrival(frozenset({10}), [_dialog(10)])
    assert result.status == "NOT_ARRIVED"


def test_arrival_wrong_process() -> None:
    result = evaluate_dialog_arrival(
        frozenset(), [_dialog(11, process="notepad.exe")], "chrome.exe"
    )
    assert result.status == "NOT_ARRIVED"


def test_arrival_non_dialog_window_ignored() -> None:
    window = {"hwnd": 12, "class": "Chrome_WidgetWin_1", "process": "chrome.exe"}
    assert evaluate_dialog_arrival(frozenset(), [window]).status == "NOT_ARRIVED"


def test_arrival_delayed_poll() -> None:
    polls = iter(
        [
            [],
            [],
            [_dialog(11)],
        ]
    )
    sleeps = []
    result = poll_for_dialog(
        lambda: next(polls),
        frozenset(),
        "chrome.exe",
        timing=DialogPollTiming(
            interval_s=0.01,
            total_s=1.0,
            sleep_fn=sleeps.append,
            monotonic=_fake_clock(),
        ),
    )
    assert result.status == "ARRIVED" and result.hwnd == 11
    assert len(sleeps) == 2


def test_arrival_bounded_wait_exhausted() -> None:
    sleeps = []

    result = poll_for_dialog(
        lambda: [],
        frozenset(),
        timing=DialogPollTiming(
            interval_s=0.01,
            total_s=0.05,
            sleep_fn=sleeps.append,
            monotonic=_fake_clock(step=0.02),
        ),
    )
    assert result.status == "NOT_ARRIVED"
    assert len(sleeps) <= 5  # 有界,不无限等待


def _fake_clock(step=0.0):
    state = {"t": 0.0}

    def clock():
        state["t"] += step if step else 0.0
        if not step:
            state["t"] = 0.0
        return state["t"]

    return clock


def test_arrival_preexisting_dialog_cannot_satisfy() -> None:
    """pre 已有无关 #32770:post 仍是它 → NOT ARRIVED(CASE C)。"""
    result = evaluate_dialog_arrival(frozenset({30}), [_dialog(30, title="打印")])
    assert result.status == "NOT_ARRIVED"


def test_arrival_hwnd_based_dpi_invariant() -> None:
    """判定基于 HWND 集合,与窗口坐标/DPI 无关。"""
    assert (
        evaluate_dialog_arrival(frozenset({10}), [_dialog(10), _dialog(11)]).status
        == "ARRIVED"
    )
    assert (
        evaluate_dialog_arrival(frozenset({10}), [_dialog(10)]).status == "NOT_ARRIVED"
    )


# ======================================================================
# post-save 判定(§21 9-11)
# ======================================================================


def test_post_save_dialog_gone_verified() -> None:
    result = evaluate_post_save(
        11,
        [],
        foreground_process="chrome.exe",
        trigger_process="chrome.exe",
    )
    assert result.status == "VERIFIED"
    assert result.evidence["dialog_closed"] is True


def test_post_save_dialog_remains_not_verified() -> None:
    result = evaluate_post_save(
        11,
        [_dialog(11)],
        foreground_process="chrome.exe",
        trigger_process="chrome.exe",
    )
    assert result.status == "NOT_VERIFIED"
    assert result.reason == "dialog_still_open"


def test_post_save_secondary_dialog_not_verified() -> None:
    result = evaluate_post_save(
        11,
        [_dialog(12, title="确认替换")],
        foreground_process="chrome.exe",
        trigger_process="chrome.exe",
    )
    assert result.status == "SECONDARY_DIALOG_PRESENT"


def test_post_save_secondary_excludes_preexisting_others() -> None:
    """save 时已存在的无关 #32770 不算二级对话框。"""
    result = evaluate_post_save(
        11,
        [_dialog(30, title="打印")],
        foreground_process="chrome.exe",
        trigger_process="chrome.exe",
        other_dialog_hwnds_at_save=frozenset({30}),
    )
    assert result.status == "VERIFIED"


def test_post_save_foreground_not_returned() -> None:
    result = evaluate_post_save(
        11,
        [],
        foreground_process="explorer.exe",
        trigger_process="chrome.exe",
    )
    assert result.status == "NOT_VERIFIED"
    assert result.reason == "foreground_not_returned_to_trigger_app"


def test_post_save_title_supporting_only() -> None:
    assert dialog_title_supports_save("另存为") is True
    assert dialog_title_supports_save("打印") is False


# ======================================================================
# 保存路线回归(§22)
# ======================================================================


def test_route_default_filename_only_enter() -> None:
    route = build_save_dialog_route(None)
    actions = [step.action for step in route.steps if step.action is not None]
    serialized = [str(a["action_type"]) for a in actions]
    assert serialized == ["hotkey"]  # 仅 Enter,保留默认文件名


def test_route_explicit_filename_sequence() -> None:
    route = build_save_dialog_route("photo_QX731.png")
    actions = [step.action for step in route.steps if step.action is not None]
    assert [a["action_type"] for a in actions] == ["hotkey", "type", "hotkey"]
    assert actions[1]["params"]["text"] == "photo_QX731.png"


# ======================================================================
# 集成管线(fake 窗口枚举/前台;驱动完整 arm→arrive→route→verified)
# ======================================================================


def _save_agent(monkeypatch, windows_state, responses=None):
    responses = responses or ["Action: click(x=550, y=310)", "Action: click(x=1, y=1)"]
    agent = make_agent(
        SequenceBackend(responses),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=4,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
    )
    agent._run_expectation = TaskExpectation(
        save_image_intent=True,
        expected_save_filename=None,
    )
    fg = {"first": True}

    def fake_foreground():
        if fg["first"]:
            fg["first"] = False
            return 777
        return 999

    monkeypatch.setattr(
        gui_agent_module, "get_foreground_app_hwnd", fake_foreground, raising=True
    )
    monkeypatch.setattr(
        gui_agent_module, "is_window_available", lambda hwnd: True, raising=True
    )
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "focus_control_state",
        lambda: "text_input",
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "perceive_windows",
        lambda size, offset: (),
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "process_name_of_hwnd",
        lambda hwnd: "chrome.exe",
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "enumerate_save_dialog_windows",
        lambda: list(windows_state["windows"]),
        raising=True,
    )
    return agent


def _menu_boxes():
    return ({"text": "图片另存为", "bbox": (500, 300, 600, 320), "confidence": 0.99},)


def test_pipeline_save_verified(monkeypatch, caplog) -> None:
    """arm → 有界轮询到达 → 注入路线 → 路线耗尽 → post-save VERIFIED。"""
    windows_state = {"windows": [], "armed": False, "saved": False}
    agent = _save_agent(monkeypatch, windows_state)
    # 本用例只验证 P2 fake 窗口管线，隔离真实前台残留的 Save-As。
    monkeypatch.setattr(agent, "_check_save_dialog_route", lambda task: None)
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "perceive_ocr_elements_detailed",
        lambda rec, img, fp=None: ((), _menu_boxes()),
        raising=True,
    )

    def advance_windows():
        # 触发轮询第一拍后出现对话框;post-save 轮询时已消失。
        if windows_state["armed"]:
            windows_state["windows"] = [_dialog(11)]
            windows_state["armed"] = False
            windows_state["saved"] = True
        elif windows_state.get("saved"):
            windows_state["windows"] = []
        return list(windows_state["windows"])

    original_arm = PendingDialogTransition.arm_trigger

    def spy_arm(self, *args, **kwargs):
        original_arm(self, *args, **kwargs)
        windows_state["armed"] = True

    monkeypatch.setattr(PendingDialogTransition, "arm_trigger", spy_arm)
    monkeypatch.setattr(
        gui_agent_module,
        "enumerate_save_dialog_windows",
        lambda: advance_windows(),
        raising=True,
    )
    result = asyncio.run(agent(Msg("u", "保存这张图片", "user")))
    assert "任务完成" in result.content, result.content
    assert "post_dialog_save_completion" in result.content
    # 1 次模型调用(菜单点击)后全程路线驱动,零额外模型调用。
    assert agent._dependencies.model_client.calls == 1


def test_pipeline_no_dialog_sets_policy_fact(monkeypatch) -> None:
    """触发后无对话框:NO_DIALOG 事实写入 prompt,不盲循环重触发。"""
    windows_state = {"windows": []}
    agent = _save_agent(
        monkeypatch,
        windows_state,
        responses=["Action: click(x=550, y=310)"] * 4,
    )
    # 本用例只验证无对话框事实，隔离真实前台残留的 Save-As。
    monkeypatch.setattr(agent, "_check_save_dialog_route", lambda task: None)
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "perceive_ocr_elements_detailed",
        lambda rec, img, fp=None: ((), _menu_boxes()),
        raising=True,
    )
    result = asyncio.run(agent(Msg("u", "保存这张图片", "user")))
    assert "任务执行失败" in result.content  # 无对话框 → 未完成
    pending = agent._pending_dialog
    assert pending is None or pending.last_trigger_result == "NO_DIALOG"


def test_duplicate_trigger_signature_logic() -> None:
    pending = PendingDialogTransition()
    candidate = classify_save_menu_candidate([_box("图片另存为", (500, 300, 600, 320))])
    pending.arm_trigger(2, "click", candidate, "chrome.exe", frozenset())
    pending.mark_trigger_failed(2)
    same = classify_save_menu_candidate(
        [_box("图片另存为", (505, 305, 605, 325))]
    )  # 同位置微移 → 同量化签名
    assert pending.is_duplicate_trigger(same, 3) is True
    assert pending.is_duplicate_trigger(same, 9) is False  # 超出步窗
    other = classify_save_menu_candidate([_box("图片另存为", (900, 800, 990, 820))])
    assert pending.is_duplicate_trigger(other, 3) is False
    assert "图片另存为" in trigger_signature("图片另存为", (500, 300, 600, 320))


# ======================================================================
# P2 RECOVERY — retroactive arm(§24)
# ======================================================================


from agent.dialog_transition import (  # noqa: E402
    DialogWindowHistory,
    retroactive_save_dialog_arm,
)
from agent.semantic_routes import (  # noqa: E402
    build_save_dialog_route_v2,
    resolve_save_folder_location,
)


def _history_with(previous, foreground_process="chrome.exe"):
    history = DialogWindowHistory()
    history.snapshot(1, previous, foreground_process)
    history.snapshot(2, [], foreground_process)
    return history


def test_retro_1_confirmed_by_dialog() -> None:
    """保存意图 + 真实差集新对话框 + 近期触发动作 → DIALOG_CONFIRMED。"""
    history = _history_with(previous=[])
    hit = retroactive_save_dialog_arm(
        history, [_dialog(11)], ["right_click", "click"], True
    )
    assert hit is not None
    assert hit[0] == "DIALOG_CONFIRMED_TRIGGER"
    assert hit[1]["hwnd"] == 11


def test_retro_2_no_save_intent_no_arm() -> None:
    history = _history_with(previous=[])
    assert (
        retroactive_save_dialog_arm(
            history, [_dialog(11)], ["right_click", "click"], False
        )
        is None
    )


def test_retro_3_wrong_process_no_arm() -> None:
    history = _history_with(previous=[], foreground_process="chrome.exe")
    assert (
        retroactive_save_dialog_arm(
            history,
            [_dialog(11, process="notepad.exe")],
            ["click"],
            True,
        )
        is None
    )


def test_retro_4_same_dialog_no_transition() -> None:
    history = _history_with(previous=[_dialog(10)])
    assert retroactive_save_dialog_arm(history, [_dialog(10)], ["click"], True) is None


def test_retro_5_no_recent_trigger_insufficient() -> None:
    history = _history_with(previous=[])
    assert retroactive_save_dialog_arm(history, [_dialog(11)], [], True) is None
    assert (
        retroactive_save_dialog_arm(history, [_dialog(11)], ["scroll", "observe"], True)
        is None
    )


def test_retro_6_ocr_strong_plus_arrival_level() -> None:
    """OCR 触发武装 + 到达 → OCR_STRONG 路径(管线既有)。"""
    pending = PendingDialogTransition()
    candidate = classify_save_menu_candidate(
        [_box("Save image as", (500, 300, 600, 320))]
    )
    pending.arm_trigger(2, "click", candidate, "chrome.exe", frozenset())
    result = evaluate_dialog_arrival(frozenset(), [_dialog(11)], "chrome.exe")
    assert result.status == "ARRIVED"
    assert pending.dialog_hwnd is None  # 到达前未定


def test_retro_7_ocr_trigger_no_dialog_no_completion() -> None:
    history = _history_with(previous=[])
    assert retroactive_save_dialog_arm(history, [], ["click"], True) is None


def test_retro_8_preexisting_plus_new_b() -> None:
    history = _history_with(previous=[_dialog(10, title="打印")])
    hit = retroactive_save_dialog_arm(
        history,
        [_dialog(10, title="打印"), _dialog(11)],
        ["click"],
        True,
    )
    assert hit is not None and hit[1]["hwnd"] == 11


# ======================================================================
# P2 RECOVERY — folder 解析(§25)
# ======================================================================


def test_folder_parsing_known_and_path() -> None:
    cases = {
        "把图片保存到桌面": "Desktop",
        "保存到 Desktop": "Desktop",
        "下载到下载文件夹": "Downloads",
        "保存到 Downloads": "Downloads",
        "保存到 Documents": "Documents",
        r"把图片保存到 C:\Users\Test\Pictures": r"C:\Users\Test\Pictures",
    }
    for instruction, folder in cases.items():
        expectation = extract_task_expectation(instruction)
        assert expectation.expected_save_folder == folder, instruction
    # 含图片词的样例同时成立保存意图(§25 只要求 folder 值的样例不要求)。
    for instruction in ("把图片保存到桌面", r"把图片保存到 C:\Users\Test\Pictures"):
        assert extract_task_expectation(instruction).save_image_intent is True


def test_folder_parsing_negatives() -> None:
    assert extract_task_expectation("打开桌面").expected_save_folder is None
    assert extract_task_expectation("从桌面复制图片").expected_save_folder is None


def test_folder_resolution_via_env(monkeypatch) -> None:
    monkeypatch.setenv("USERPROFILE", r"C:\Users\T")
    assert resolve_save_folder_location("Desktop") == r"C:\Users\T\Desktop"
    assert resolve_save_folder_location("Downloads") == r"C:\Users\T\Downloads"
    assert resolve_save_folder_location(r"C:\Data\Saved") == r"C:\Data\Saved"
    assert resolve_save_folder_location(None) is None
    assert resolve_save_folder_location("不存在的文件夹") is None


# ======================================================================
# P2 RECOVERY — 四象限路线(§26)
# ======================================================================


def _route_signature(route):
    return [
        (
            step.action["action_type"],
            tuple(sorted(step.action["params"].items())),
        )
        for step in route.steps
        if step.action is not None
    ]


def test_quadrant_a_no_folder_no_filename() -> None:
    route = build_save_dialog_route_v2(None, None)
    assert _route_signature(route) == [("hotkey", (("keys", ("enter",)),))]


def test_quadrant_b_filename_only() -> None:
    route = build_save_dialog_route_v2(None, "foo.png")
    signature = _route_signature(route)
    assert signature[0] == ("hotkey", (("keys", ("ctrl", "a")),))
    assert signature[1] == ("type", (("text", "foo.png"),))
    assert signature[2] == ("hotkey", (("keys", ("enter",)),))


def test_quadrant_c_folder_only_preserves_default_filename() -> None:
    route = build_save_dialog_route_v2(r"C:\Users\T\Desktop", None)
    signature = _route_signature(route)
    assert signature[0] == ("type", (("text", r"C:\Users\T\Desktop"),))
    assert signature[1] == ("hotkey", (("keys", ("enter",)),))
    # 不 Ctrl+A、不发明文件名;提交用 Alt+S 而非假设 Enter。
    assert signature[2] == ("hotkey", (("keys", ("alt", "s")),))
    assert not any(
        s[0] == "type" and s[1][0][1] != r"C:\Users\T\Desktop" for s in signature
    )


def test_quadrant_d_folder_and_filename() -> None:
    route = build_save_dialog_route_v2(r"C:\Users\T\Downloads", "photo_QX731.png")
    signature = _route_signature(route)
    assert signature[0] == ("type", (("text", r"C:\Users\T\Downloads"),))
    assert signature[1] == ("hotkey", (("keys", ("enter",)),))
    assert signature[2] == ("hotkey", (("keys", ("ctrl", "a")),))
    assert signature[3] == ("type", (("text", "photo_QX731.png"),))
    assert signature[4] == ("hotkey", (("keys", ("alt", "s")),))


def test_route_has_wait_step_after_navigation(monkeypatch) -> None:
    monkeypatch.setenv("USERPROFILE", r"C:\Users\T")
    route = build_save_dialog_route_v2(r"C:\Users\T\Desktop", None)
    waits = [s for s in route.steps if s.action is None]
    assert len(waits) == 1 and waits[0].wait_seconds == 0.8


def test_folder_named_subfolder_parsing_and_resolution(monkeypatch) -> None:
    """命名子文件夹:"桌面的X文件夹" -> Desktop\\X 复合标识与绝对路径。"""
    monkeypatch.setenv("USERPROFILE", r"C:\Users\T")
    cases = {
        '把图片保存到桌面的"Reports"文件夹中': "Desktop\\Reports",
        "把网页图片保存到桌面的\u201c相册\u201d文件夹中": "Desktop\\相册",
    }
    for instruction, folder in cases.items():
        expectation = extract_task_expectation(instruction)
        assert expectation.expected_save_folder == folder, instruction
        assert resolve_save_folder_location(folder) == (
            "C:\\Users\\T\\Desktop\\" + folder.split("\\", 1)[1]
        )


def test_legacy_report_sets_save_route_dispatched(monkeypatch) -> None:
    """legacy 前台检出置 save_route_dispatched,post-save 验证可触发。"""
    agent = _save_agent(monkeypatch, {"windows": []})
    agent._run_expectation = TaskExpectation(
        save_image_intent=True,
        expected_save_folder="Desktop",
    )
    agent._pending_dialog = PendingDialogTransition()
    monkeypatch.setattr(
        gui_agent_module,
        "process_name_of_hwnd",
        lambda hwnd: "chrome.exe",
        raising=True,
    )
    agent._report_save_dialog_observed(4242, "另存为")
    pending = agent._pending_dialog
    assert isinstance(pending, PendingDialogTransition)
    assert pending.dialog_hwnd == 4242
    assert pending.save_route_dispatched is True


def test_folder_named_subfolder_downloads_base(monkeypatch) -> None:
    """下载基座命名子文件夹:"下载文件夹中的X文件夹" -> Downloads\\X。"""
    monkeypatch.setenv("USERPROFILE", r"C:\Users\T")
    cases = {
        "请把图片保存到下载文件夹中的“P2Variant”文件夹": "Downloads\\P2Variant",
        "把图片保存到下载文件夹中的\u201c归档\u201d文件夹": "Downloads\\归档",
    }
    for instruction, folder in cases.items():
        expectation = extract_task_expectation(instruction)
        assert expectation.expected_save_folder == folder, instruction
        assert resolve_save_folder_location(folder) == (
            "C:\\Users\\T\\Downloads\\" + folder.split("\\", 1)[1]
        )


def test_filename_setwei_variant_parsing() -> None:
    """variant 措辞回归:"文件名设为 photo_XX.png" 必须解析出文件名。"""
    instruction = (
        "请把页面里标题为“海滩”的图片保存到下载文件夹中的"
        "“P2Variant_QK28”文件夹，文件名设为 photo_QK28.png。"
    )
    expectation = extract_task_expectation(instruction)
    assert expectation.save_image_intent is True
    assert expectation.expected_save_filename == "photo_QK28.png"
    assert expectation.expected_save_folder == "Downloads\\P2Variant_QK28"


_FILENAME_POSITIVES = [
    "文件名设为 photo.png",
    "文件名设置为 cat.jpg",
    "文件名用 capture.webp",
    "文件名为 test.png",
    "保存成 screenshot.png",
    "命名为 result.jpg",
]

_FILENAME_NEGATIVES = [
    "文件名是什么？",
    "查看文件名",
    "显示文件名",
    "保持文件名不变",
    "不要修改文件名",
    "使用默认文件名",
    "原文件名即可",
    "文件名如果存在就保留",
    "复制文件名",
    "搜索文件名为 photo.png 的文件",
    "查找文件名为 a.png 的文件",
    "复制文件名为 b.jpg 的文件",
]


@pytest.mark.parametrize("phrase", _FILENAME_POSITIVES)
def test_filename_assignment_positives(phrase) -> None:
    expectation = extract_task_expectation("请保存图片，" + phrase)
    assert expectation.expected_save_filename is not None, phrase


@pytest.mark.parametrize("phrase", _FILENAME_NEGATIVES)
def test_filename_assignment_negatives(phrase) -> None:
    expectation = extract_task_expectation("请保存图片，" + phrase)
    assert expectation.expected_save_filename is None, phrase


def test_original_m03_expectation_regression() -> None:
    """原 M03 指令解析回归:quadrant C 语义不被新措辞分支改变。"""
    instruction = (
        "把当前网页中标题为“海滩”的那张图片，"
        "保存到桌面的“GUIAgentBenchmark_Paired”文件夹中。"
    )
    expectation = extract_task_expectation(instruction)
    assert expectation.save_image_intent is True
    assert expectation.expected_save_folder == "Desktop\\GUIAgentBenchmark_Paired"
    assert expectation.expected_save_filename is None


# ======================================================================
# B1 characterization:legacy 前台检测的标题知识(锁 observable 行为)
# ======================================================================


class _FakeSaveDialogUser32:
    """最小 user32 桩:仅覆盖 _check_save_dialog_route 路径的三个查询。"""

    def __init__(self, hwnd: int, window_class: str, title: str) -> None:
        self._hwnd = hwnd
        self._window_class = window_class
        self._title = title

    def GetForegroundWindow(self) -> int:
        return self._hwnd

    def GetClassNameW(self, hwnd, buf, size) -> int:
        buf.value = self._window_class
        return 1

    def GetWindowTextW(self, hwnd, buf, size) -> int:
        buf.value = self._title
        return 1


def _check_route_with_fake_dialog(monkeypatch, title: str):
    """以 #32770 前台 + 给定标题驱动 legacy 检测;保存任务恒为允许。"""
    monkeypatch.setattr(
        ctypes.windll,
        "user32",
        _FakeSaveDialogUser32(4242, "#32770", title),
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "is_save_download_task",
        lambda task: True,
        raising=True,
    )
    agent = _save_agent(monkeypatch, {"windows": []})
    agent._pending_dialog = PendingDialogTransition()
    return agent._check_save_dialog_route("保存这张图片")


@pytest.mark.parametrize("title", ["Save As", "Save", "另存为", "保存"])
def test_check_save_dialog_route_positive_titles(monkeypatch, title) -> None:
    """#32770 前台 + 保存类标题 + 保存任务 → 构建保存路线。"""
    route = _check_route_with_fake_dialog(monkeypatch, title)
    assert route is not None
    assert route.steps


def test_check_save_dialog_route_negative_title(monkeypatch) -> None:
    """#32770 前台但标题不含保存关键词 → 不构建路线。"""
    route = _check_route_with_fake_dialog(monkeypatch, "打印")
    assert route is None
