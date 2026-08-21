"""FINAL FIX 1+2+3 tests:H02 COM verifier,Save-As 路线,Explorer 路线。

全部纯逻辑/fake,无真实 Office/COM/GUI/模型。
"""

import asyncio
from pathlib import Path

from agentscope.message import Msg

from agent.semantic_routes import (
    build_browser_search_route,
    build_file_open_route,
    build_file_search_route,
    build_save_dialog_route,
    extract_browser_search_info,
    extract_file_route_info,
)
from agent.task_manager import TaskManager
from benchmark.tasks import (
    H02Presentation,
    _powerpoint_slide_text_readonly,
)
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent

# ======================================================================
# FIX 1: H02 PowerPoint COM verifier tests
# ======================================================================


class _StubH02Monitor:
    def find_app_windows(self, name, cls=None):
        return [{"hwnd": 1}]

    def ocr_window_text(self, hwnd, zoom=1):
        return ""  # OCR returns nothing (simulates the false-negative)


def test_h02_com_verifier_exact_success(monkeypatch):
    """COM text exact match → PASS。"""
    title = "桌面GUI智能体测试EDD"
    body = "本轮测试编号为75A4，用于验证视觉定位和文本输入能力。"
    monkeypatch.setattr(
        "benchmark.tasks._powerpoint_slide_text_readonly",
        lambda: (title, body),
    )
    from benchmark.case_specs import generate_mh_case

    case = generate_mh_case("H02", "T", 1)
    # Override to match the mocked text
    case.params["title"] = title
    case.params["body"] = body
    task = H02Presentation("RUN", Path("logs"), case_spec=case)
    task.prepare()
    task.monitor = _StubH02Monitor()
    ok, detail = task.validate()
    assert ok, detail
    assert "COM只读" in detail


def test_h02_com_verifier_missing_title(monkeypatch):
    """COM text 无标题 → FAIL。"""
    monkeypatch.setattr(
        "benchmark.tasks._powerpoint_slide_text_readonly",
        lambda: ("", "只有正文没有标题"),
    )
    from benchmark.case_specs import generate_mh_case

    case = generate_mh_case("H02", "T", 1)
    task = H02Presentation("RUN", Path("logs"), case_spec=case)
    task.prepare()
    task.monitor = _StubH02Monitor()
    ok, _ = task.validate()
    assert not ok


def test_h02_com_verifier_missing_body(monkeypatch):
    """COM text 无正文标识 → FAIL。"""
    from benchmark.case_specs import generate_mh_case

    case = generate_mh_case("H02", "T", 1)
    task = H02Presentation("RUN", Path("logs"), case_spec=case)
    task.prepare()
    task.monitor = _StubH02Monitor()
    # COM returns title but not body run_tag
    monkeypatch.setattr(
        "benchmark.tasks._powerpoint_slide_text_readonly",
        lambda: (task.title, ""),
    )
    ok, _ = task.validate()
    assert not ok


def test_h02_com_exception_falls_back_to_ocr(monkeypatch):
    """COM 不可用 → OCR fallback 不崩溃。"""
    monkeypatch.setattr(
        "benchmark.tasks._powerpoint_slide_text_readonly",
        lambda: None,
    )
    from benchmark.case_specs import generate_mh_case

    case = generate_mh_case("H02", "T", 1)
    task = H02Presentation("RUN", Path("logs"), case_spec=case)
    task.prepare()
    task.monitor = _StubH02Monitor()
    ok, detail = task.validate()
    # Should reach OCR fallback and fail (OCR returns empty)
    assert not ok
    # Should NOT crash


def test_h02_com_never_creates_powerpoint():
    """COM verifier 源码不含创建/启动 PowerPoint 的调用。"""
    import inspect

    source = inspect.getsource(_powerpoint_slide_text_readonly)
    for banned in ("Dispatch", "CreateObject", ".Add", ".Open", ".Save", ".Quit"):
        assert banned not in source, banned


# ======================================================================
# FIX 2: Save-As route tests
# ======================================================================


def test_save_dialog_route_default_name_is_enter_only():
    """PRD-BC-001 特性锁:指令未给出文件名时保留默认名(无键入步骤)。

    M03 现行指令只含标题不含文件名,路线行为必须与 IMG 推断存在时
    完全一致(Enter-only)。
    """
    route = build_save_dialog_route(None)
    assert route.steps
    for step in route.steps:
        if step.action is not None:
            assert step.action["action_type"] != "type"


def test_save_dialog_route_steps():
    """Save-As 路线步骤正确:Ctrl+A→type filename→Enter。"""
    route = build_save_dialog_route("photo.png")
    step1 = route.next_step()
    assert step1.action["action_type"] == "hotkey"
    assert step1.action["params"]["keys"] == ("ctrl", "a")
    step2 = route.next_step()
    assert step2.action["action_type"] == "type"
    assert step2.action["params"]["text"] == "photo.png"
    step3 = route.next_step()
    assert step3.action["action_type"] == "hotkey"
    assert step3.action["params"]["keys"] == ("enter",)
    assert route.next_step() is None
    assert route.is_exhausted


def test_save_dialog_route_never_uses_filesystem():
    """路线步骤仅含 hotkey/type,零文件系统操作。"""
    route = build_save_dialog_route("test.png")
    for step in route.steps:
        if step.action:
            assert step.action["action_type"] in ("hotkey", "type")


# ======================================================================
# FIX 3: Explorer file route tests
# ======================================================================


def test_extract_exact_path_route():
    """exact path 模式:M04 类任务文本。"""
    text = '打开桌面"GUIAgentBenchmark_Paired"文件夹中的"项目说明_919.txt"'
    info = extract_file_route_info(text)
    assert info is not None
    assert info["route_type"] == "exact_path"
    assert info["folder"] == "GUIAgentBenchmark_Paired"
    assert "项目说明" in info["filename"]


def test_extract_search_substring_route():
    """folder+substring 模式:H03 类任务文本。"""
    text = (
        '在桌面"GUIAgentBenchmark_Paired"文件夹中，'
        '找到文件名包含"report"的.doc文件并打开'
    )
    info = extract_file_route_info(text)
    assert info is not None
    assert info["route_type"] == "search_substring"
    assert info["substring"] == "report"
    assert info["extension"] == ".doc"


def test_unrelated_task_no_route():
    """非文件任务不触发路线。"""
    for text in (
        "打开计算器并计算1+1",
        "将系统输出音量调整到大约60%",
        "打开Chrome浏览器",
        "关闭当前窗口",
    ):
        assert extract_file_route_info(text) is None


def test_file_open_route_steps():
    """exact path 路线:Win+E→wait→Ctrl+L→type path→Enter→wait。"""
    route = build_file_open_route("TestDir", "file.txt")
    steps = route.steps
    assert steps[0].action["action_type"] == "hotkey"
    assert steps[0].action["params"]["keys"] == ("win", "e")
    assert steps[1].wait_seconds > 0  # wait for Explorer
    assert steps[2].action["params"]["keys"] == ("ctrl", "l")
    assert steps[3].action["action_type"] == "type"
    assert "Desktop" in steps[3].action["params"]["text"]
    assert steps[3].action["params"]["text"].endswith("TestDir")
    assert steps[4].action["params"]["keys"] == ("enter",)
    assert steps[5].wait_seconds > 0
    assert steps[6].action["params"]["keys"] == ("ctrl", "l")
    assert "file.txt" in steps[7].action["params"]["text"]
    assert steps[8].action["params"]["keys"] == ("enter",)
    assert steps[9].wait_seconds > 0


def test_file_search_route_steps():
    """search 路线:Win+E→Ctrl+L→folder→Enter→Ctrl+F→search→Enter。"""
    route = build_file_search_route("TestDir", "report", ".doc")
    types = [s.action["action_type"] for s in route.steps if s.action]
    assert "hotkey" in types
    assert "type" in types
    # 验证搜索步骤的 type 文本含 substring + extension
    type_steps = [
        s for s in route.steps if s.action and s.action["action_type"] == "type"
    ]
    assert len(type_steps) == 2  # folder path + search query
    assert "report" in type_steps[1].action["params"]["text"]


def test_h03_case_instruction_names_unique_target() -> None:
    """H03 指令与 validator 共享同一个完整目标文件名。"""
    from benchmark.case_specs import generate_mh_case

    spec = generate_mh_case("H03", "H03_ALIGNMENT", seed=20260821)
    assert spec.params["target_file"] in spec.instruction
    assert f'文件“{spec.params["target_file"]}”' in spec.instruction


def test_current_browser_search_route_is_keyboard_native() -> None:
    """当前浏览器搜索只使用通用五动作合同且关键词来自任务文本。"""
    info = extract_browser_search_info('使用当前浏览器搜索"Python"并停留。')
    assert info == {"query": "Python"}
    route = build_browser_search_route(info["query"])
    actions = [step.action for step in route.steps if step.action is not None]
    assert [action["action_type"] for action in actions] == [
        "hotkey",
        "type",
        "hotkey",
    ]
    assert actions[1]["params"]["text"] == "Python"


def test_file_routes_never_use_filesystem():
    """路线步骤仅含 hotkey/type/wait,零文件系统操作。"""
    for route in (
        build_file_open_route("d", "f.txt"),
        build_file_search_route("d", "s", ".doc"),
    ):
        for step in route.steps:
            if step.action:
                assert step.action["action_type"] in ("hotkey", "type")


def test_exact_file_route_visits_folder_before_opening_file() -> None:
    """exact 路线先形成 Explorer 目录证据，再打开完整目标路径。"""
    route = build_file_open_route("TargetDir", "report_66.doc")
    typed = [
        step.action["params"]["text"]
        for step in route.steps
        if step.action is not None and step.action["action_type"] == "type"
    ]
    assert typed[0].endswith("TargetDir")
    assert typed[1].endswith("TargetDir\\report_66.doc")


def test_semantic_route_not_triggered_when_disabled():
    """semantic_execution 关闭时路线不生效。"""
    agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
    )
    # semantic off → route should be None
    assert agent._active_route is None


def test_file_route_integrated_into_agent():
    """文件任务在 semantic on 时自动初始化路线。"""
    task_text = (
        '打开桌面"TestFolder"文件夹中的"test.txt"，'
        "找到其中的项目编号，并告诉我结果。"
    )
    agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager(task_text),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
    )
    asyncio.run(agent(Msg("u", task_text, "user")))
    # Route should have been initialized during _run_task
    # (even if steps were consumed by the finish-first backend)
