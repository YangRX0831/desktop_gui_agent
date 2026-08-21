"""H01 Word verifier 回归测试；全部使用 fake COM/monitor。"""

from benchmark.tasks import H01WebToDoc, _normalized_contains

GOOD_TITLE = "文档_76A"
GOOD_BODY = "本文档发布于2026年8月。"


class _FakeMonitor:
    def __init__(self, text: str = "", has_word: bool = True) -> None:
        self.text = text
        self.has_word = has_word

    def find_app_windows(self, process: str) -> list[dict]:
        if process == "WINWORD" and self.has_word:
            return [{"hwnd": 1, "process": "WINWORD.EXE"}]
        return []

    def ocr_window_client_text(self, hwnd: int, zoom: int = 1) -> str:
        return self.text


def _make_task(preexisting: set[tuple[str, str]] | None = None) -> H01WebToDoc:
    task = H01WebToDoc.__new__(H01WebToDoc)
    task.doc_title = GOOD_TITLE
    task.target_section = {"title": "发布时间", "content": GOOD_BODY}
    task._preexisting_word_documents = preexisting or set()
    task.monitor = _FakeMonitor()
    return task


def test_word_com_new_document_with_full_title_and_body_passes(monkeypatch) -> None:
    task = _make_task()
    monkeypatch.setattr(
        "benchmark.tasks._word_documents_readonly",
        lambda: [("Document1", f"{GOOD_TITLE}\r{GOOD_BODY}")],
    )

    ok, detail = task.validate()

    assert ok, detail
    assert "Word" in detail and "COM只读" in detail


def test_word_com_preexisting_matching_document_does_not_pass(monkeypatch) -> None:
    document = ("Document1", f"{GOOD_TITLE}\r{GOOD_BODY}")
    task = _make_task({document})
    monkeypatch.setattr(
        "benchmark.tasks._word_documents_readonly",
        lambda: [document],
    )

    ok, _ = task.validate()

    assert not ok


def test_word_com_missing_title_fails(monkeypatch) -> None:
    task = _make_task()
    monkeypatch.setattr(
        "benchmark.tasks._word_documents_readonly",
        lambda: [("Document1", GOOD_BODY)],
    )

    ok, _ = task.validate()

    assert not ok


def test_word_com_missing_body_fails(monkeypatch) -> None:
    task = _make_task()
    monkeypatch.setattr(
        "benchmark.tasks._word_documents_readonly",
        lambda: [("Document1", GOOD_TITLE)],
    )

    ok, _ = task.validate()

    assert not ok


def test_word_ocr_fallback_requires_full_title_and_body(monkeypatch) -> None:
    task = _make_task()
    task.monitor = _FakeMonitor(f"{GOOD_TITLE}\n{GOOD_BODY}")
    monkeypatch.setattr("benchmark.tasks._word_documents_readonly", lambda: None)

    ok, detail = task.validate()

    assert ok, detail
    assert "OCR" in detail


def test_word_ocr_title_only_fails(monkeypatch) -> None:
    task = _make_task()
    task.monitor = _FakeMonitor(GOOD_TITLE)
    monkeypatch.setattr("benchmark.tasks._word_documents_readonly", lambda: None)

    ok, _ = task.validate()

    assert not ok


def test_word_ocr_no_word_window_fails(monkeypatch) -> None:
    task = _make_task()
    task.monitor = _FakeMonitor(has_word=False)
    monkeypatch.setattr("benchmark.tasks._word_documents_readonly", lambda: None)

    ok, detail = task.validate()

    assert not ok
    assert "Word窗口不存在" in detail


def test_normalized_contains_accepts_punctuation_and_whitespace_variation() -> None:
    assert _normalized_contains("本文档发布于 2026年8月,", GOOD_BODY)


def test_normalized_contains_rejects_short_prefix() -> None:
    assert not _normalized_contains("本文档", GOOD_BODY)


def test_normalized_contains_rejects_empty_expected() -> None:
    assert not _normalized_contains("some text", "")
