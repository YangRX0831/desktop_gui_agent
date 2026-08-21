"""FIX A + FIX B tests:resolved instruction 数据流 + M03 functional verifier。"""

from pathlib import Path

from agent.semantic_routes import (
    build_save_dialog_route,
    is_save_download_task,
)
from benchmark.case_specs import acceptance_cases, generate_mh_case
from benchmark.tasks import H03FileSearch, M03Download, M04Document, S01Calculator

PROJECT = Path(__file__).resolve().parent.parent


# ======================================================================
# FIX A: paired_runner resolved instruction
# ======================================================================


def test_fix_a_spec_instruction_contains_placeholder():
    """原始 CaseSpec instruction 含占位符(语义保持不变)。"""
    for task_id in ("M03", "M04", "H03"):
        case = generate_mh_case(task_id, f"PRD_{task_id}", 20260819)
        assert "GUIAgentBenchmark_CASE" in case.instruction, task_id


def test_fix_a_task_instruction_resolved():
    """task.instruction() 不含占位符,含实际目录名。"""
    desktop_dir = Path("fake_desktop_dir")
    for cls, task_id in (
        (M03Download, "M03"),
        (M04Document, "M04"),
        (H03FileSearch, "H03"),
    ):
        case = generate_mh_case(task_id, f"PRD_{task_id}", 20260819)
        task = cls("RUN", desktop_dir, case_spec=case)
        resolved = task.instruction()
        assert "GUIAgentBenchmark_CASE" not in resolved, task_id
        assert "fake_desktop_dir" in resolved, task_id


def test_fix_a_s01_instruction_unchanged():
    """无占位符的 S01 instruction 语义完全不变。"""
    case = next(c for c in acceptance_cases() if c.task_id == "S01")
    task = S01Calculator("RUN", Path("any"), case_spec=case)
    assert task.instruction() == case.instruction
    assert "GUIAgentBenchmark" not in task.instruction()


def test_fix_a_all_acceptance_no_placeholder_leak():
    """全部 15 项 task.instruction() 不泄漏 GUIAgentBenchmark_CASE。"""
    classes = {
        "S01": S01Calculator,
        "M03": M03Download,
        "M04": M04Document,
        "H03": H03FileSearch,
    }
    for case in acceptance_cases():
        cls = classes.get(case.task_id)
        if cls is None:
            continue
        task = cls("RUN", Path("real_dir"), case_spec=case)
        resolved = task.instruction()
        assert "GUIAgentBenchmark_CASE" not in resolved, case.task_id


# ======================================================================
# FIX B: Save-As route eligibility relaxation
# ======================================================================


def test_fix_b_save_download_task_detection():
    """保存/下载任务语义识别。"""
    assert is_save_download_task("把图片保存到桌面的文件夹中")
    assert is_save_download_task("下载图片到桌面")
    assert is_save_download_task("另存为文件")
    assert not is_save_download_task("打开计算器并计算1+1")
    assert not is_save_download_task("关闭当前窗口")
    assert not is_save_download_task("打开Chrome浏览器")


def test_fix_b_route_with_filename():
    """有文件名:Ctrl+A→type→Enter。"""
    route = build_save_dialog_route("photo.png")
    actions = [s.action for s in route.steps if s.action]
    assert len(actions) == 3
    assert actions[0]["params"]["keys"] == ("ctrl", "a")
    assert actions[1]["params"]["text"] == "photo.png"
    assert actions[2]["params"]["keys"] == ("enter",)


def test_fix_b_route_without_filename():
    """无文件名:直接 Enter(保留默认名,不 Ctrl+A 改名)。"""
    route = build_save_dialog_route(None)
    actions = [s.action for s in route.steps if s.action]
    assert len(actions) == 1
    assert actions[0]["params"]["keys"] == ("enter",)


def test_fix_b_route_never_filesystem():
    """路线步骤仅 hotkey/type,零文件系统操作。"""
    for route in (build_save_dialog_route("x.png"), build_save_dialog_route(None)):
        for step in route.steps:
            if step.action:
                assert step.action["action_type"] in ("hotkey", "type")


# ======================================================================
# FIX B: M03 functional verifier
# ======================================================================


class _StubMonitor:
    def find_app_windows(self, name, cls=None):
        return []


def _m03_task(tmp_path, with_preexisting=False):
    case = generate_mh_case("M03", "T", 1)
    save_dir = tmp_path / "bench"
    save_dir.mkdir()
    if with_preexisting:
        (save_dir / "old_photo.png").write_bytes(b"old")
    task = M03Download("RUN", save_dir, case_spec=case)
    task.monitor = _StubMonitor()
    # Simulate prepare without fixture page
    task.save_dir = save_dir
    task.target = next(
        img
        for img in __import__(
            "benchmark.tasks", fromlist=["FIXTURE_GALLERY"]
        ).FIXTURE_GALLERY
        if img["filename"] == case.params["filename"]
    )
    task.target_path = save_dir / task.target["filename"]
    task._target_before = None
    task.result.params = {
        "image_title": task.target["title"],
        "filename": task.target["filename"],
    }
    return task, save_dir


def test_fix_b_m03_new_image_pass(tmp_path):
    """run 后目标图片首次出现且非空 → PASS。"""
    task, _ = _m03_task(tmp_path)
    task.target_path.write_bytes(b"\x89PNG fake data")
    ok, detail = task.validate()
    assert ok, detail
    assert "已新增或更新" in detail


def test_fix_b_m03_no_new_file_fail(tmp_path):
    """run 后无新文件 → FAIL。"""
    task, _ = _m03_task(tmp_path)
    ok, detail = task.validate()
    assert not ok
    assert "未保存" in detail


def test_fix_b_m03_preexisting_not_counted(tmp_path):
    """run 前已有图片不算新下载。"""
    task, save_dir = _m03_task(tmp_path, with_preexisting=True)
    ok, detail = task.validate()
    assert not ok  # "old_photo.png" is preexisting, not new


def test_fix_b_m03_exact_filename_still_pass(tmp_path):
    """任务开始后新建确切目标文件名 → PASS。"""
    task, save_dir = _m03_task(tmp_path)
    task.target_path.write_bytes(b"\x89PNG data")
    ok, detail = task.validate()
    assert ok


def test_fix_b_m03_non_image_new_file_not_pass(tmp_path):
    """新增非图片文件不算 → FAIL。"""
    task, save_dir = _m03_task(tmp_path)
    (save_dir / "notes.txt").write_text("text")
    ok, _ = task.validate()
    assert not ok


def test_fix_b_m03_empty_image_not_pass(tmp_path):
    """新增空图片文件 → FAIL。"""
    task, save_dir = _m03_task(tmp_path)
    (save_dir / "empty.png").write_bytes(b"")
    ok, detail = task.validate()
    assert not ok


def test_fix_b_m03_verifier_never_writes():
    """M03 verifier 源码零文件系统写操作。"""
    import inspect

    source = inspect.source = inspect.getsource(M03Download.validate)
    for banned in (".write", ".unlink", ".mkdir", "shutil", "os.rename"):
        assert banned not in source, banned
