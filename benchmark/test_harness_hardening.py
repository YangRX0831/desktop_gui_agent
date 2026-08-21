"""HARNESS HARDENING 回归测试(纯非 GUI;tmp_path/fake 日志与监视器)。

覆盖 PHASE 1 五项修复:
H1 跨日志轮转的当前 run 终态检测;H2 WebFixture 失败 → ENV_ERROR;
H3 S01 显示区 token 验证;H4 S04 独立状态转移验证;H5 超时常量同源。
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from benchmark.case_specs import CaseSpec
from benchmark.core import (
    Monitor,
    Status,
    find_marker_line_since,
    log_candidate_paths,
    window_band_region,
)
from benchmark.runner import TASK_TIMEOUT, TASK_TIMEOUT_MEDIUM_HIGH
from benchmark.tasks import (
    ALL_TASKS,
    FIXTURE_TASK_IDS,
    S01Calculator,
    S04OpenApp,
    display_tokens_match,
)

PROJECT = Path(__file__).resolve().parent.parent
TERMINAL_MARKERS = ["gui_agent_task_succeeded", "gui_agent_task_failed"]


def log_line(delta_seconds: float, message: str) -> str:
    """构造带时间戳的日志行;delta 相对当前时间。"""
    ts = datetime.now() + timedelta(seconds=delta_seconds)
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    return f"{stamp} | INFO | gui_agent | {message}"


# ======================================================================
# H1 — 跨轮转终态检测
# ======================================================================


def test_h1_current_run_success_detected(tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_text(
        log_line(-600, "旧任务 gui_agent_task_failed：reason=旧")
        + "\n"
        + log_line(-1, "gui_agent_task_succeeded")
        + "\n",
        encoding="utf-8",
    )
    line = find_marker_line_since(log, TERMINAL_MARKERS, time.time() - 5)
    assert line is not None and "succeeded" in line


def test_h1_current_run_failed_detected(tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_text(log_line(-1, "gui_agent_task_failed：reason=x") + "\n")
    line = find_marker_line_since(log, TERMINAL_MARKERS, time.time() - 5)
    assert line is not None and "failed" in line


def test_h1_previous_run_terminal_not_satisfying(tmp_path: Path) -> None:
    """历史 run 的终态行(时间早于本次等待)不得满足当前 run。"""
    log = tmp_path / "app.log"
    log.write_text(log_line(-600, "gui_agent_task_succeeded") + "\n")
    assert find_marker_line_since(log, TERMINAL_MARKERS, time.time() - 5) is None


def test_h1_terminal_in_rotated_log_detected(tmp_path: Path) -> None:
    """终态行写入后发生轮转:内容进入副本,当前文件已重建 → 仍可检测。"""
    rotated = tmp_path / "app.log.2026-08-20"
    rotated.write_text(log_line(-2, "gui_agent_task_succeeded") + "\n")
    current = tmp_path / "app.log"
    current.write_text(log_line(-1, "cli_ready") + "\n")
    line = find_marker_line_since(current, TERMINAL_MARKERS, time.time() - 5)
    assert line is not None and "succeeded" in line


def test_h1_rotation_mid_task_current_terminal(tmp_path: Path) -> None:
    """轮转发生在任务中途:旧 run 终态在副本,当前终态在新文件 →
    只接受当前 run 的那条。"""
    rotated = tmp_path / "app.log.2026-08-20"
    rotated.write_text(
        log_line(-600, "gui_agent_task_failed：旧") + "\n",
        encoding="utf-8",
    )
    current = tmp_path / "app.log"
    current.write_text(
        log_line(-1, "gui_agent_task_failed：当前") + "\n",
        encoding="utf-8",
    )
    line = find_marker_line_since(current, TERMINAL_MARKERS, time.time() - 5)
    assert line is not None and "当前" in line


def test_h1_multiple_history_runs_only_current(tmp_path: Path) -> None:
    rotated = tmp_path / "app.log.2026-08-19"
    rotated.write_text(
        "\n".join(log_line(-d, "gui_agent_task_succeeded") for d in (900, 800, 700))
        + "\n",
        encoding="utf-8",
    )
    current = tmp_path / "app.log"
    current.write_text(
        "\n".join(log_line(-d, "gui_agent_task_failed") for d in (500, 400)) + "\n",
        encoding="utf-8",
    )
    since = time.time() - 5
    assert find_marker_line_since(current, TERMINAL_MARKERS, since) is None
    current.write_text(
        current.read_text(encoding="utf-8")
        + log_line(-1, "gui_agent_task_succeeded")
        + "\n",
        encoding="utf-8",
    )
    line = find_marker_line_since(current, TERMINAL_MARKERS, since)
    assert line is not None and "succeeded" in line


def test_h1_no_current_terminal_returns_none(tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_text(log_line(-3, "普通步骤日志,无终态") + "\n")
    assert find_marker_line_since(log, TERMINAL_MARKERS, time.time() - 5) is None


def test_h1_candidate_paths_current_plus_newest_rotated(tmp_path: Path) -> None:
    (tmp_path / "app.log.2026-08-18").write_text("a", encoding="utf-8")
    (tmp_path / "app.log.2026-08-19").write_text("b", encoding="utf-8")
    current = tmp_path / "app.log"
    current.write_text("c", encoding="utf-8")
    assert log_candidate_paths(current) == [
        current,
        tmp_path / "app.log.2026-08-19",
    ]


def test_h1_monitor_wait_for_log_session_scoped(tmp_path: Path) -> None:
    """wait_for_log 用时间戳窗口关联当前会话:历史 cli_ready 不误报。"""
    from benchmark.core import Monitor

    log = tmp_path / "app.log"
    log.write_text(log_line(-600, "cli_ready") + "\n")
    monitor = Monitor()
    assert (
        monitor.wait_for_log(log, ["cli_ready"], timeout=0.3, poll_interval=0.1) is None
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(log_line(0, "cli_ready") + "\n")
    found = monitor.wait_for_log(log, ["cli_ready"], timeout=3, poll_interval=0.1)
    assert found is not None and "cli_ready" in found


# ======================================================================
# H2 — WebFixture 失败 → ENV_ERROR
# ======================================================================


def test_h2_fixture_task_ids_definition() -> None:
    assert FIXTURE_TASK_IDS == frozenset({"M02", "M03", "M05", "H01"})


def _m02_spec() -> CaseSpec:
    return CaseSpec(
        task_id="M02",
        pair_id="M02_T01",
        case_seed=1,
        instruction="在当前浏览器打开的聊天页面中发送消息。",
        params={},
    )


def test_h2_paired_fixture_failure_env_error(tmp_path, monkeypatch) -> None:
    """fixture 未启动:依赖任务直接 ENV_ERROR,且不创建任何 Agent 会话。"""
    from benchmark import paired_runner

    def _bomb(*args, **kwargs):
        raise AssertionError("fixture 失败时不得创建 AgentSession")

    monkeypatch.setattr(paired_runner, "AgentSession", _bomb)
    entry = paired_runner.run_arm(
        _m02_spec(),
        "v3",
        tmp_path,
        tmp_path,
        semantic=True,
        fixture_ok=False,
    )
    assert entry["status"] == "ENV_ERROR"
    assert entry["failure_reason"] == "fixture_start_failed"


def test_h2_paired_non_fixture_task_not_gated(tmp_path, monkeypatch) -> None:
    """非 fixture 任务不受 fixture_ok 影响:通过门控并进入会话启动。"""
    from benchmark import paired_runner

    class _StubSession:
        def __init__(self, env_overrides=None):
            self.monitor = None

        def start(self, cli_args=None):
            return False  # 让 run_arm 走 ENV_ERROR 短路,证明已过门控

        def stop(self):
            return None

    reached = {"session": False}

    def _factory(env_overrides=None):
        reached["session"] = True
        return _StubSession()

    monkeypatch.setattr(paired_runner, "AgentSession", _factory)
    monkeypatch.setattr(
        paired_runner, "restore_benchmark_desktop_state", lambda monitor: True
    )
    spec = CaseSpec(
        task_id="S01",
        pair_id="S01_T01",
        case_seed=None,
        instruction="打开系统计算器,计算1+1。",
        params={},
    )
    entry = paired_runner.run_arm(
        spec, "v3", tmp_path, tmp_path, semantic=False, fixture_ok=False
    )
    assert reached["session"] is True
    assert entry["status"] == "ENV_ERROR"


# ======================================================================
# H3 — S01 显示区 token 验证
# ======================================================================


def test_h3_display_token_matcher() -> None:
    assert display_tokens_match("71", "71") is True
    assert display_tokens_match("标准 2", "2") is True
    assert display_tokens_match("2.0", "2") is True
    assert display_tokens_match("12", "2") is False
    assert display_tokens_match("20", "2") is False
    assert display_tokens_match("2,000", "2000") is True
    assert display_tokens_match("", "2") is False
    assert display_tokens_match("abc", "2") is False


def _s01_task(expected: int = 2) -> S01Calculator:
    task = S01Calculator(
        run_id="T",
        desktop_dir=PROJECT / "logs",
        case_spec=CaseSpec(
            task_id="S01",
            pair_id="S01_T01",
            case_seed=None,
            instruction="打开系统计算器,计算1+1。",
            params={"expression": "1+1", "expected": expected},
        ),
    )
    task.prepare()
    return task


class _S01StubMonitor:
    def __init__(self, band_text: str) -> None:
        self._band_text = band_text

    def visible_windows(self):
        return [{"hwnd": 31, "process": "ApplicationFrameHost.exe"}]

    def get_window_title(self, hwnd):
        return "计算器"

    def ocr_window_band_text(self, hwnd, top_fraction, height_fraction, zoom=1):
        return self._band_text


def test_h3_s01_display_result_pass() -> None:
    task = _s01_task(expected=2)
    task.monitor = cast(Monitor, _S01StubMonitor("2"))
    ok, actual = task.validate()
    assert ok, actual
    assert "独立结果 2" in actual


def test_h3_s01_keyboard_substring_not_pass() -> None:
    """显示区不含 2(只有 12)时,即使键盘区有 2 也不得通过。"""
    task = _s01_task(expected=2)
    task.monitor = cast(Monitor, _S01StubMonitor("12"))
    ok, _ = task.validate()
    assert ok is False


def test_h3_s01_float_normalization() -> None:
    task = _s01_task(expected=2)
    task.monitor = cast(Monitor, _S01StubMonitor("2.0"))
    ok, _ = task.validate()
    assert ok is True


def test_h3_s01_no_evidence_fails() -> None:
    """显示区 OCR 无证据(如矩形不可用)→ 证据不足,不做弱通过。"""
    task = _s01_task(expected=2)
    task.monitor = cast(Monitor, _S01StubMonitor(""))
    ok, actual = task.validate()
    assert ok is False
    assert "证据不足" in actual


def test_h3_band_region_math() -> None:
    region = window_band_region((0, 0, 100, 200), 0.0, 0.32)
    assert region is not None
    assert region == {"left": 0, "top": 0, "width": 100, "height": 64}
    scaled = window_band_region((10, 20, 500, 1000), 0.0, 0.32)
    assert scaled is not None
    assert scaled["height"] == 320 and scaled["width"] == 500
    clipped = window_band_region((-8, -8, 100, 200), 0.0, 0.32)
    assert clipped is not None
    assert clipped["left"] == 0 and clipped["top"] == 0


@pytest.mark.parametrize(
    "top,height",
    [(1.0, 0.3), (-0.1, 0.3), (0.0, 0.0), (0.0, 1.5), (0.5, 0.6)],
)
def test_h3_band_region_invalid_fractions(top, height) -> None:
    assert window_band_region((0, 0, 100, 200), top, height) is None


def test_h3_band_region_zero_size() -> None:
    assert window_band_region((0, 0, 0, 0), 0.0, 0.3) is None


# ======================================================================
# H4 — S04 独立状态转移验证
# ======================================================================


class _S04StubMonitor:
    def __init__(self, windows: list[dict]) -> None:
        self._windows = windows

    def find_app_windows(self, process, window_class=None):
        return [w for w in self._windows if w["process"] == process]

    def find_windows_by_title(self, keyword):
        return [w for w in self._windows if keyword in w.get("title", "")]


def _s04_task() -> S04OpenApp:
    return S04OpenApp(
        run_id="T",
        desktop_dir=PROJECT / "logs",
        case_spec=CaseSpec(
            task_id="S04",
            pair_id="S04_T01",
            case_seed=None,
            instruction="打开Chrome浏览器,并让它的主窗口显示在桌面前台。",
            params={
                "application": "Chrome浏览器",
                "process": "chrome.exe",
                "title_keyword": None,
                "window_class": "Chrome_WidgetWin_1",
            },
        ),
    )


def _run_s04(monkeypatch, pre: list[dict], post: list[dict], trace=None):
    # 先注入 stub 监视器再 prepare,保证 pre 快照完全来自假数据,
    # 不触碰真实系统窗口枚举。
    task = _s04_task()
    task.monitor = cast(Monitor, _S04StubMonitor(pre))
    task.prepare()
    task.monitor = cast(Monitor, _S04StubMonitor(post))
    monkeypatch.setattr(
        "benchmark.tasks._app_launch_trace_evidence_since",
        lambda since, process: trace,
    )
    return task.validate()


def test_h04_forged_verified_without_transition_fails(monkeypatch) -> None:
    ok, actual = _run_s04(
        monkeypatch,
        pre=[{"hwnd": 1, "process": "chrome.exe"}],
        post=[{"hwnd": 1, "process": "chrome.exe"}],
        trace={"completion_verification": "VERIFIED"},
    )
    assert ok is False
    assert "无状态转移" in actual and "VERIFIED" in actual


def test_h04_absent_to_present_passes(monkeypatch) -> None:
    ok, actual = _run_s04(
        monkeypatch,
        pre=[],
        post=[{"hwnd": 5, "process": "chrome.exe"}],
        trace=None,
    )
    assert ok, actual
    assert "BENCH_STATE_TRANSITION" in actual


def test_h04_new_window_case_b_passes(monkeypatch) -> None:
    ok, _ = _run_s04(
        monkeypatch,
        pre=[{"hwnd": 1, "process": "chrome.exe"}],
        post=[
            {"hwnd": 1, "process": "chrome.exe"},
            {"hwnd": 7, "process": "chrome.exe"},
        ],
        trace=None,
    )
    assert ok is True


def test_h04_preexisting_only_fails(monkeypatch) -> None:
    ok, _ = _run_s04(
        monkeypatch,
        pre=[{"hwnd": 1, "process": "chrome.exe"}],
        post=[{"hwnd": 1, "process": "chrome.exe", "foreground": True}],
        trace=None,
    )
    assert ok is False


def test_h04_new_hwnd_of_other_process_fails(monkeypatch) -> None:
    """新 HWND 属于其他进程:目标进程过滤后无新窗口 → FAIL。"""
    ok, _ = _run_s04(
        monkeypatch,
        pre=[{"hwnd": 1, "process": "chrome.exe"}],
        post=[
            {"hwnd": 1, "process": "chrome.exe"},
            {"hwnd": 9, "process": "notepad.exe"},
        ],
        trace=None,
    )
    assert ok is False


def test_h04_stale_trace_verified_not_sufficient(monkeypatch) -> None:
    """旧 run 遗留的 VERIFIED + 当前无转移 → FAIL(同伪造场景语义)。"""
    ok, _ = _run_s04(
        monkeypatch,
        pre=[{"hwnd": 1, "process": "chrome.exe"}],
        post=[{"hwnd": 1, "process": "chrome.exe"}],
        trace={"completion_verification": "VERIFIED", "stale": True},
    )
    assert ok is False


def test_h04_transition_survives_cli_restore(monkeypatch) -> None:
    """CLI 前台恢复后窗口枚举仍见新 HWND → PASS(枚举与前台无关)。"""
    ok, actual = _run_s04(
        monkeypatch,
        pre=[{"hwnd": 1, "process": "chrome.exe"}],
        post=[
            {"hwnd": 1, "process": "chrome.exe"},
            {"hwnd": 8, "process": "chrome.exe"},
        ],
        trace={"completion_verification": "VERIFIED"},
    )
    assert ok, actual


def test_h04_absent_and_still_absent_fails(monkeypatch) -> None:
    ok, _ = _run_s04(monkeypatch, pre=[], post=[], trace=None)
    assert ok is False


# ======================================================================
# H5 — 超时常量同源 + 结果分类
# ======================================================================


def test_h5_timeout_constants_single_source() -> None:
    import benchmark.paired_runner as paired

    assert TASK_TIMEOUT == 480
    assert TASK_TIMEOUT_MEDIUM_HIGH == 1000
    assert paired.TASK_TIMEOUT is TASK_TIMEOUT
    assert paired.TASK_TIMEOUT_MEDIUM_HIGH is TASK_TIMEOUT_MEDIUM_HIGH


def test_h5_difficulty_split_matches_timeout_branch() -> None:
    """简单任务走 480s 分支,中等/复杂走 1000s 分支的判定输入正确。"""
    for cls in ALL_TASKS:
        is_simple = cls.difficulty == "简单"
        assert is_simple == cls.task_id.startswith("S"), cls.task_id


def test_h5_status_taxonomy() -> None:
    members = {s.name for s in Status}
    assert members == {
        "PASS",
        "FAIL",
        "SKIP",
        "ENV_ERROR",
        "TIMEOUT",
        "SAFETY_SKIP",
    }
