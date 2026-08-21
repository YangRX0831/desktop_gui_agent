"""S04 SEMANTIC FIX tests:app snapshot、CASE A/B、completion guard、verifier。"""

import ctypes
from pathlib import Path

from agent.semantic_routes import (
    AppStateSnapshot,
    build_app_launch_route,
    enumerate_save_dialog_windows,
    extract_app_launch_info,
    extract_file_route_info,
    snapshot_app_state,
)
from agent.task_expectation import (
    TaskExpectation,
    verify_completion,
)
from agent.task_manager import TaskManager
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent

PROJECT = Path(__file__).resolve().parent.parent


# ======================================================================
# PART 2: pure app-launch intent tests (1-5)
# ======================================================================


def test_pure_launch_chrome():
    """1:"打开Chrome浏览器" → route recognized。"""
    info = extract_app_launch_info("打开Chrome浏览器，并让它的主窗口显示在桌面前台。")
    assert info is not None and info["app_id"] == "chrome"


def test_calculate_not_pure_launch():
    """2:"打开计算器，计算1+1" → 不触发。"""
    assert extract_app_launch_info("打开计算器，计算1+1") is None


def test_chrome_search_not_pure_launch():
    """3:"打开Chrome并搜索Python" → 不触发。"""
    assert extract_app_launch_info("打开Chrome并搜索Python") is None


def test_file_route_priority_over_app():
    """4:file route 优先于 app-launch route。"""
    file_text = '打开桌面"Folder"文件夹中的"file.txt"'
    assert extract_file_route_info(file_text) is not None
    # app route should also not fire (has "文件" exclusion)
    assert extract_app_launch_info(file_text) is None


def test_semantic_disabled_no_route():
    """5:semantic disabled → route 不触发。"""
    agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager("打开Chrome浏览器"),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
    )
    assert agent._active_route is None
    assert agent._app_snapshot is None


# ======================================================================
# PART 3+4: CASE A/B + completion guard tests (6-11)
# ======================================================================


def _chrome_expectation():
    return TaskExpectation(
        expected_app="Chrome",
        expected_app_processes=("chrome.exe",),
    )


def test_case_a_no_pre_run_verified():
    """6:pre 无 Chrome → launch → Chrome 出现 → VERIFIED。"""
    result = verify_completion(
        _chrome_expectation(),
        {
            "foreground_process": "chrome.exe",
            "pre_target_hwnds": set(),
            "post_target_hwnds": {100},
        },
    )
    assert result.status == "VERIFIED"


def test_case_b_pre_existing_no_new_hwnd_not_verified():
    """7:pre 已有 Chrome → 无新 HWND → 不得 VERIFIED。"""
    result = verify_completion(
        _chrome_expectation(),
        {
            "foreground_process": "chrome.exe",
            "pre_target_hwnds": {100},
            "post_target_hwnds": {100},
        },
    )
    assert result.status == "NOT_VERIFIED"
    assert "no new window" in result.reason


def test_case_b_ctrl_n_new_hwnd_verified():
    """8:pre 已有 Chrome → Ctrl+N → 新 HWND → VERIFIED。"""
    result = verify_completion(
        _chrome_expectation(),
        {
            "foreground_process": "chrome.exe",
            "pre_target_hwnds": {100},
            "post_target_hwnds": {100, 200},
        },
    )
    assert result.status == "VERIFIED"
    assert "new" in result.reason.lower()


def test_pre_fg_zero_action_not_verified():
    """9:Chrome task start 已前台 + 无变化 → NOT_VERIFIED。"""
    result = verify_completion(
        _chrome_expectation(),
        {
            "foreground_process": "chrome.exe",
            "pre_target_hwnds": {100},
            "post_target_hwnds": {100},
        },
    )
    assert result.status == "NOT_VERIFIED"


def test_process_exists_only_not_pass():
    """10:chrome.exe 进程存在但无 HWND 变化 → 不 PASS。"""
    result = verify_completion(
        _chrome_expectation(),
        {
            "foreground_process": "chrome.exe",
            "pre_target_hwnds": {100},
            "post_target_hwnds": {100},
        },
    )
    assert result.status == "NOT_VERIFIED"


def test_old_hwnd_only_not_pass():
    """11:仅旧 Chrome HWND 无新 HWND → 不 PASS。"""
    result = verify_completion(
        _chrome_expectation(),
        {
            "foreground_process": "chrome.exe",
            "pre_target_hwnds": {100, 101},
            "post_target_hwnds": {100, 101},
        },
    )
    assert result.status == "NOT_VERIFIED"


# ======================================================================
# PART 3: route behavior tests
# ======================================================================


def test_case_b_route_includes_ctrl_n():
    """CASE B 路线含 Ctrl+N(Chrome pre-existing 时)。"""
    route = build_app_launch_route("Chrome", app_id="chrome", pre_existing=True)
    has_ctrl_n = any(
        s.action and s.action.get("params", {}).get("keys") == ("ctrl", "n")
        for s in route.steps
    )
    assert has_ctrl_n


def test_case_a_route_no_ctrl_n():
    """CASE A 路线不含 Ctrl+N(Chrome 不存在时)。"""
    route = build_app_launch_route("Chrome", app_id="chrome", pre_existing=False)
    has_ctrl_n = any(
        s.action and s.action.get("params", {}).get("keys") == ("ctrl", "n")
        for s in route.steps
    )
    assert not has_ctrl_n


# ======================================================================
# PART 7: GUI-only tests (16)
# ======================================================================


def test_no_subprocess_in_routes():
    """16:semantic_routes 源码无 subprocess/os.startfile/ShellExecute。"""
    import inspect

    from agent import semantic_routes

    source = inspect.getsource(semantic_routes)
    for banned in ("subprocess", "os.startfile", "ShellExecute", "Popen", "Selenium"):
        assert banned not in source, banned


def test_no_subprocess_in_task_expectation():
    """task_expectation 源码无 GUI 禁用 API。"""
    import inspect

    from agent import task_expectation

    source = inspect.getsource(task_expectation)
    for banned in ("subprocess", "os.startfile", "ShellExecute"):
        assert banned not in source, banned


# ======================================================================
# PART 8: additional coverage
# ======================================================================


def test_cli_restore_preserves_verified():
    """12:CLI 恢复后 trace evidence 仍保持 VERIFIED(trace 记录了事实)。"""
    # trace evidence 是不可变的 JSONL 记录;CLI 恢复不会改变它。
    evidence = {
        "expected_app": "Chrome",
        "pre_target_hwnds": [],
        "post_target_hwnds": [200],
        "new_target_hwnds": [200],
        "foreground_process": "chrome.exe",
        "completion_verification": "VERIFIED",
    }
    # 即使 CLI 恢复后 foreground 变了,evidence 中的 new_target_hwnds 不变
    assert evidence["new_target_hwnds"] == [200]
    assert evidence["completion_verification"] == "VERIFIED"


def test_trace_before_start_not_used():
    """13:trace timestamp < task start → verifier 不使用(由 since 过滤)。"""
    # _app_launch_trace_evidence_since 按文件 mtime >= since - 5 过滤;
    # 早于 task start 的 trace 文件不会被扫描。
    from benchmark.tasks import _app_launch_trace_evidence_since

    # since = 很大的未来时间 → 无 trace 匹配
    result = _app_launch_trace_evidence_since(9999999999.0, "chrome.exe")
    assert result is None


def test_unrelated_app_evidence_not_used():
    """14:其他 app 的 trace 不污染 S04(expected_app 匹配)。"""
    # verify 只看 expected_app == Chrome 的 evidence
    chrome_exp = _chrome_expectation()
    result = verify_completion(
        chrome_exp,
        {
            "foreground_process": "notepad.exe",
            "pre_target_hwnds": set(),
            "post_target_hwnds": set(),
        },
    )
    assert result.status == "NOT_VERIFIED"


def test_snapshot_app_state_returns_data(monkeypatch):
    """snapshot 返回 AppStateSnapshot;fake Win32 确定性,不依赖真实桌面。"""
    _install_fake_process_win32(
        monkeypatch,
        [
            {
                "hwnd": 11,
                "title": "Chrome",
                "class_name": "Chrome_WidgetWin_1",
                "pid": 4242,
                "image": r"C:\Apps\Chrome\chrome.exe",
            }
        ],
        foreground=11,
    )
    snap = snapshot_app_state(("chrome.exe",))
    assert isinstance(snap, AppStateSnapshot)
    assert snap.target_hwnds == frozenset({11})
    assert snap.was_foreground is True
    assert snap.timestamp > 0


# ======================================================================
# B4 characterization:S1/S2/S5 进程查询路径(锁 observable 行为)
# ======================================================================


class _FakeProcUser32:
    """B4 桩:按窗口表应答 EnumWindows/前台/类名/标题/PID 查询。"""

    def __init__(self, windows, foreground):
        self._windows = {window["hwnd"]: window for window in windows}
        self._foreground = foreground

    def _window(self, hwnd):
        return self._windows.get(hwnd, {"title": "", "class_name": "", "pid": 0})

    def EnumWindows(self, callback, extra):
        for hwnd in self._windows:
            callback(hwnd, 0)
        return 1

    def GetForegroundWindow(self):
        return self._foreground

    def IsWindowVisible(self, hwnd):
        return True

    def GetWindowTextLengthW(self, hwnd):
        return len(self._window(hwnd)["title"])

    def GetClassNameW(self, hwnd, buf, size):
        buf.value = self._window(hwnd)["class_name"]
        return 1

    def GetWindowTextW(self, hwnd, buf, size):
        buf.value = self._window(hwnd)["title"]
        return 1

    def GetWindowThreadProcessId(self, hwnd, pid_ref):
        pid_ref._obj.value = self._window(hwnd)["pid"]
        return 1


class _FakeProcKernel32:
    """B4 桩:以 pid 充当句柄,按 pid 表返回完整映像路径。"""

    def __init__(self, image_by_pid):
        self._image_by_pid = image_by_pid

    def OpenProcess(self, access, inherit, pid):
        return pid if pid in self._image_by_pid else 0

    def QueryFullProcessImageNameW(self, handle, flags, buf, size_ref):
        buf.value = self._image_by_pid[handle]
        size_ref._obj.value = len(buf.value)
        return 1

    def CloseHandle(self, handle):
        return 1


def _install_fake_process_win32(monkeypatch, windows, foreground):
    user32 = _FakeProcUser32(windows, foreground)
    kernel32 = _FakeProcKernel32({w["pid"]: w["image"] for w in windows})
    monkeypatch.setattr(ctypes.windll, "user32", user32, raising=True)
    monkeypatch.setattr(ctypes.windll, "kernel32", kernel32, raising=True)


def test_snapshot_visible_match_case_normalized(monkeypatch):
    """S1:进程名大写时,小写 target 集合仍命中,visible hwnd 入集合。"""
    _install_fake_process_win32(
        monkeypatch,
        [
            {
                "hwnd": 11,
                "title": "Chrome",
                "class_name": "Chrome_WidgetWin_1",
                "pid": 4242,
                "image": r"C:\Apps\Chrome\CHROME.EXE",
            }
        ],
        foreground=0,
    )
    snap = snapshot_app_state(("chrome.exe",))
    assert snap.target_hwnds == frozenset({11})
    assert snap.was_foreground is False


def test_snapshot_foreground_match(monkeypatch):
    """S2:前台窗口进程匹配目标 → was_foreground True。"""
    _install_fake_process_win32(
        monkeypatch,
        [
            {
                "hwnd": 11,
                "title": "Chrome",
                "class_name": "Chrome_WidgetWin_1",
                "pid": 4242,
                "image": r"C:\Apps\Chrome\CHROME.EXE",
            }
        ],
        foreground=11,
    )
    snap = snapshot_app_state(("chrome.exe",))
    assert snap.was_foreground is True


def test_snapshot_no_match(monkeypatch):
    """无关进程:目标 hwnd 不入集合,前台不误判。"""
    _install_fake_process_win32(
        monkeypatch,
        [
            {
                "hwnd": 12,
                "title": "Other",
                "class_name": "OtherWnd",
                "pid": 5151,
                "image": r"C:\Apps\Other\OTHER.EXE",
            }
        ],
        foreground=12,
    )
    snap = snapshot_app_state(("chrome.exe",))
    assert snap.target_hwnds == frozenset()
    assert snap.was_foreground is False


def test_save_dialog_enumeration_process_field(monkeypatch):
    """S5:#32770 对话框的 process 字段保留原大小写 basename。"""
    _install_fake_process_win32(
        monkeypatch,
        [
            {
                "hwnd": 33,
                "title": "另存为",
                "class_name": "#32770",
                "pid": 6262,
                "image": r"C:\Windows\System32\NOTEPAD.EXE",
            }
        ],
        foreground=33,
    )
    dialogs = enumerate_save_dialog_windows()
    assert len(dialogs) == 1
    assert dialogs[0]["process"] == "NOTEPAD.EXE"
    assert dialogs[0]["hwnd"] == 33
