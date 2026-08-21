"""被动运行诊断(OBSERVABILITY_ONLY)单元测试:纯非 GUI。

覆盖:开关门控、BEGIN/END+duration、异常路径、敏感字段丢弃、
watchdog 阈值(30s 警告/60s/120s 栈快照,不重复)、栈快照助手、
阶段登记/清除生命周期。
"""

import logging

import pytest

from utils import run_diagnostics as rd


@pytest.fixture(autouse=True)
def _reset_watchdog():
    rd._WATCHDOG.clear()
    yield
    rd._WATCHDOG.clear()


def _records(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage()]


def test_gate_off_emits_nothing(caplog, monkeypatch) -> None:
    monkeypatch.delenv("GUI_AGENT_RUN_DIAGNOSTICS", raising=False)
    with caplog.at_level(logging.INFO, logger="gui_agent"):
        with rd.diag_phase("diag_model", step=1, model="m"):
            pass
        rd.diag_log("diag_x", step=2)
    assert _records(caplog) == []


def test_phase_begin_end_with_duration(caplog, monkeypatch) -> None:
    monkeypatch.setenv("GUI_AGENT_RUN_DIAGNOSTICS", "1")
    with caplog.at_level(logging.INFO, logger="gui_agent"):
        with rd.diag_phase("diag_model", step=3, model="qwen"):
            pass
    msgs = _records(caplog)
    assert any(m.startswith("diag_model_begin") and "step=3" in m for m in msgs)
    ends = [m for m in msgs if m.startswith("diag_model_end")]
    assert len(ends) == 1 and "duration_ms=" in ends[0]


def test_phase_exception_path_records_and_reraises(caplog, monkeypatch) -> None:
    monkeypatch.setenv("GUI_AGENT_RUN_DIAGNOSTICS", "1")
    with caplog.at_level(logging.INFO, logger="gui_agent"):
        with pytest.raises(ValueError):
            with rd.diag_phase("diag_ocr", step=1):
                raise ValueError("boom")
    msgs = _records(caplog)
    ends = [m for m in msgs if m.startswith("diag_ocr_end")]
    assert len(ends) == 1 and "exception_type=ValueError" in ends[0]
    # 异常路径必须清除 watchdog 登记,否则残留阶段会误报。
    assert rd._WATCHDOG._phase is None


def test_sensitive_fields_dropped(caplog, monkeypatch) -> None:
    monkeypatch.setenv("GUI_AGENT_RUN_DIAGNOSTICS", "1")
    with caplog.at_level(logging.INFO, logger="gui_agent"):
        rd.diag_log(
            "diag_probe",
            step=1,
            api_key="sk-SECRET",
            token="t",
            text="用户全文",
            action_type="type",
        )
    msgs = _records(caplog)
    assert len(msgs) == 1
    assert "sk-SECRET" not in msgs[0] and "用户全文" not in msgs[0]
    assert "action_type=type" in msgs[0]


def test_field_value_truncated(caplog, monkeypatch) -> None:
    monkeypatch.setenv("GUI_AGENT_RUN_DIAGNOSTICS", "1")
    with caplog.at_level(logging.INFO, logger="gui_agent"):
        rd.diag_log("diag_probe", note="x" * 300)
    msg = _records(caplog)[0]
    assert len(msg) < 200


def test_text_digest_stable_short() -> None:
    assert rd.text_digest("hello") == rd.text_digest("hello")
    assert rd.text_digest("hello") != rd.text_digest("hellx")
    assert len(rd.text_digest("hello")) == 8


def test_watchdog_warning_once_then_stacks(monkeypatch, caplog) -> None:
    monkeypatch.setenv("GUI_AGENT_RUN_DIAGNOSTICS", "1")
    rd._WATCHDOG.register("diag_model", begin=0.0, step=7)
    with caplog.at_level(logging.INFO, logger="gui_agent"):
        rd._WATCHDOG.check(now=31.0)
        rd._WATCHDOG.check(now=35.0)  # 不重复告警
        rd._WATCHDOG.check(now=61.0)  # 第一次栈快照
        rd._WATCHDOG.check(now=70.0)  # 不重复
        rd._WATCHDOG.check(now=121.0)  # 第二次栈快照
    msgs = _records(caplog)
    warnings = [m for m in msgs if m.startswith("DIAG_STALL_WARNING")]
    stacks = [m for m in msgs if m.startswith("DIAG_STALL_STACK")]
    assert len(warnings) == 1 and "phase=diag_model" in warnings[0]
    assert "step=7" in warnings[0]
    assert len(stacks) == 2
    assert "frames_top=" in stacks[0]


def test_watchdog_below_threshold_silent(monkeypatch, caplog) -> None:
    monkeypatch.setenv("GUI_AGENT_RUN_DIAGNOSTICS", "1")
    rd._WATCHDOG.register("diag_ocr", begin=0.0, step=1)
    with caplog.at_level(logging.INFO, logger="gui_agent"):
        rd._WATCHDOG.check(now=10.0)
    assert _records(caplog) == []


def test_watchdog_clear_stops_reporting(monkeypatch, caplog) -> None:
    monkeypatch.setenv("GUI_AGENT_RUN_DIAGNOSTICS", "1")
    rd._WATCHDOG.register("diag_ocr", begin=0.0, step=1)
    rd._WATCHDOG.clear()
    with caplog.at_level(logging.INFO, logger="gui_agent"):
        rd._WATCHDOG.check(now=100.0)
    assert _records(caplog) == []


def test_thread_stack_snapshot_contains_current_thread() -> None:
    snapshot = rd.thread_stack_snapshot()
    assert "test_thread_stack_snapshot" in snapshot
