"""隐私安全且 PRD 忠实的 acceptance harness 回归测试。"""

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path
from typing import cast

import pytest

from benchmark.case_specs import generate_mh_case
from benchmark.core import Monitor, WebFixture
from benchmark.tasks import (
    H01WebToDoc,
    H02Presentation,
    H03FileSearch,
    M02Email,
    M03Download,
    _explorer_location_matches,
    _file_evidence,
    _m01_data_matches,
    _sent_email_matches,
    search_title_matches,
)


def _m02_task() -> M02Email:
    case = generate_mh_case("M02", "PRD_M02", 20260819)
    task = M02Email("RUN", Path("logs"), case_spec=case)
    task.recipient = case.params["recipient"]
    task.subject = case.params["subject"]
    task.body = case.params["body"]
    return task


def test_m02_chat_store_cannot_pass_email_validator(monkeypatch) -> None:
    task = _m02_task()
    monkeypatch.setattr(
        "benchmark.tasks.fetch_fixture_status",
        lambda: {
            "chat_messages": [
                {
                    "contact": task.recipient,
                    "message": f"{task.subject} - {task.body}",
                },
            ],
            "sent_emails": [],
        },
    )

    ok, _ = task.validate()

    assert not ok


@pytest.mark.parametrize("missing", ["recipient", "subject", "body", "state"])
def test_m02_missing_required_email_field_fails(monkeypatch, missing: str) -> None:
    task = _m02_task()
    email = {
        "recipient": task.recipient,
        "subject": task.subject,
        "body": task.body,
        "state": "sent",
    }
    email[missing] = ""
    monkeypatch.setattr(
        "benchmark.tasks.fetch_fixture_status",
        lambda: {"sent_emails": [email]},
    )

    ok, _ = task.validate()

    assert not ok


def test_m02_complete_sent_email_passes(monkeypatch) -> None:
    task = _m02_task()
    email = {
        "recipient": task.recipient,
        "subject": task.subject,
        "body": task.body,
        "state": "sent",
    }
    monkeypatch.setattr(
        "benchmark.tasks.fetch_fixture_status",
        lambda: {"sent_emails": [email]},
    )

    ok, detail = task.validate()

    assert ok, detail


def test_sent_email_matcher_rejects_chat_shape() -> None:
    assert not _sent_email_matches(
        [{"contact": "user1@gui.test", "message": "hello"}],
        "user1@gui.test",
        "subject",
        "body",
    )


def test_s05_generic_search_title_without_query_fails() -> None:
    assert not search_title_matches("搜索结果 - Chrome", "Python")
    assert search_title_matches("Python - 搜索结果 - Chrome", "Python")


def test_m01_missing_row_or_field_fails() -> None:
    headers = ["姓名", "部门", "分数"]
    rows = [["陈晨", "研发", "86"], ["林宇", "产品", "91"], ["周宁", "测试", "78"]]
    complete = [headers, *rows]
    assert _m01_data_matches(complete, headers, rows)[0]
    assert not _m01_data_matches(complete[:-1], headers, rows)[0]
    missing_field = [row[:] for row in complete]
    missing_field[2][1] = ""
    assert not _m01_data_matches(missing_field, headers, rows)[0]


def test_m01_rejects_flattened_or_shifted_grid() -> None:
    """所有文本出现也不能替代从 A1 开始的独立 row/column cell 结构。"""
    headers = ["姓名", "部门", "分数"]
    rows = [["陈晨", "研发", "86"], ["林宇", "产品", "91"], ["周宁", "测试", "78"]]
    flattened = [["姓名部门分数陈晨研发86林宇产品91周宁测试78"]]
    shifted = [["", "姓名", "部门", "分数"], *[["", *row] for row in rows]]
    extra_leading_row = [["说明"], headers, *rows]
    assert not _m01_data_matches(flattened, headers, rows)[0]
    assert not _m01_data_matches(shifted, headers, rows)[0]
    assert not _m01_data_matches(extra_leading_row, headers, rows)[0]


def _m03_task(path: Path, before: tuple[int, int, str] | None) -> M03Download:
    case = generate_mh_case("M03", "PRD_M03", 20260819)
    task = M03Download("RUN", path, case_spec=case)
    task.target = {
        "title": case.params["image_title"],
        "filename": case.params["filename"],
    }
    task.target_path = path / case.params["filename"]
    task._target_before = before
    return task


def test_m03_preexisting_unchanged_target_does_not_pass(tmp_path) -> None:
    target = tmp_path / generate_mh_case("M03", "PRD_M03", 20260819).params["filename"]
    target.write_bytes(b"preexisting")
    before = _file_evidence(target)
    task = _m03_task(tmp_path, before)

    ok, _ = task.validate()

    assert not ok


def test_m03_changed_target_proves_current_save(tmp_path) -> None:
    task = _m03_task(tmp_path, None)
    task.target_path.write_bytes(b"new image bytes")

    ok, detail = task.validate()

    assert ok, detail


def _h02_task() -> H02Presentation:
    case = generate_mh_case("H02", "PRD_H02", 20260819)
    task = H02Presentation("RUN", Path("logs"), case_spec=case)
    task.prepare()
    return task


@pytest.mark.parametrize("part", ["title", "body"])
def test_h02_requires_complete_title_and_body(monkeypatch, part: str) -> None:
    task = _h02_task()
    slide = (task.title, "") if part == "title" else ("", task.body)
    monkeypatch.setattr(
        "benchmark.tasks._powerpoint_slide_text_readonly",
        lambda: slide,
    )

    ok, _ = task.validate()

    assert not ok


class _H03Monitor:
    def __init__(self, title: str) -> None:
        self.title = title

    def visible_windows(self) -> list[dict]:
        return [{"hwnd": 9, "process": "WINWORD.EXE"}]

    def get_window_title(self, hwnd: int) -> str:
        return self.title


def _h03_task(tmp_path: Path, title: str) -> H03FileSearch:
    task = H03FileSearch.__new__(H03FileSearch)
    task.search_dir = tmp_path / "FileSearch_ABC"
    task.search_dir.mkdir()
    task.target_file = "report_66.doc"
    task._preexisting_explorer_locations = set()
    task.monitor = cast(Monitor, _H03Monitor(title))
    return task


def test_h03_wrong_report_like_window_does_not_pass(tmp_path, monkeypatch) -> None:
    task = _h03_task(tmp_path, "report_other.doc - Word")
    monkeypatch.setattr(
        "benchmark.tasks._file_explorer_locations_readonly",
        lambda: [(5, task.search_dir.resolve().as_uri())],
    )

    ok, _ = task.validate()

    assert not ok


def test_h03_requires_explorer_transition_and_exact_target(
    tmp_path, monkeypatch
) -> None:
    task = _h03_task(tmp_path, "report_66.doc - Word")
    location = task.search_dir.resolve().as_uri()
    assert _explorer_location_matches(location, task.search_dir, "report")
    monkeypatch.setattr(
        "benchmark.tasks._file_explorer_locations_readonly",
        lambda: [(5, location)],
    )

    ok, detail = task.validate()

    assert ok, detail


def test_h01_word_unavailable_is_explicit_environment_blocker(monkeypatch) -> None:
    case = generate_mh_case("H01", "PRD_H01", 20260819)
    task = H01WebToDoc("RUN", Path("logs"), case_spec=case)
    monkeypatch.setattr("benchmark.tasks._word_available", lambda: False)

    assert not task.prepare()
    assert "environment_blocker" in task.result.failure_reason


def test_webmail_fixture_has_email_fields_and_sent_store() -> None:
    fixture = WebFixture(port=0)
    fixture.configure("RUN", {"email_recipients": ["user1@gui.test"]})
    assert fixture.start()
    try:
        base = f"http://127.0.0.1:{fixture.port}"
        with urllib.request.urlopen(f"{base}/webmail", timeout=2) as response:
            html = response.read().decode("utf-8")
        for field in ('id="recipient"', 'id="subject"', 'id="body"', 'id="send"'):
            assert field in html
        payload = json.dumps(
            {
                "recipient": "user1@gui.test",
                "subject": "Synthetic subject",
                "body": "Synthetic body",
            },
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/api/email/send",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert result == {"ok": True, "state": "sent"}
        with urllib.request.urlopen(f"{base}/api/status", timeout=2) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert status["sent_emails"][0]["state"] == "sent"
        assert status["chat_messages"] == []
    finally:
        fixture.stop()


def test_webfixture_partial_request_shutdown_is_bounded_and_leak_free() -> None:
    fixture = WebFixture(port=0)
    fixture.configure("RUN", {})
    assert fixture.start()
    thread_name = f"benchmark-web-fixture-{fixture.port}"
    client = socket.create_connection(("127.0.0.1", fixture.port), timeout=1)
    client.sendall(
        b"POST /api/email/send HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Length: 100000\r\n\r\n{",
    )
    time.sleep(0.05)

    started = time.monotonic()
    fixture.stop()
    elapsed = time.monotonic() - started
    client.close()

    assert elapsed < 2.0
    assert all(thread.name != thread_name for thread in threading.enumerate())
