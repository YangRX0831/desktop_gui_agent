"""测试键盘文本、控制字符、Unicode 和异常契约。

Windows 与非 Windows 分支均由内存 fake 接收；非 BMP 仅有模拟测试证据。
"""

import logging
from collections import defaultdict

import pytest

from control import keyboard_controller
from utils.exceptions import KeyboardOperationError

from tests.keyboard_test_support import FakeEnvironment
from tests.keyboard_test_support import FakeKey
from tests.keyboard_test_support import FakeTextBackend
from tests.keyboard_test_support import _controller
from tests.keyboard_test_support import fake_environment


class FakeKeyboardBackend:
    """记录键盘操作并支持按调用位置注入异常。"""

    def __init__(self, timeline: list[tuple[object, ...]] | None = None) -> None:
        self.events: list[tuple[str, object]] = []
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.failures: dict[tuple[str, int], Exception] = {}
        self.timeline = timeline

    def fail_on(self, operation: str, call_number: int, exc: Exception) -> None:
        self.failures[(operation, call_number)] = exc

    def press(self, key: object) -> None:
        self._record("press", key)

    def release(self, key: object) -> None:
        self._record("release", key)

    def _record(self, operation: str, key: object) -> None:
        self.counts[operation] += 1
        self.events.append((operation, key))
        if self.timeline is not None:
            self.timeline.append(("keyboard", operation, key))
        failure = self.failures.get((operation, self.counts[operation]))
        if failure is not None:
            raise failure


class FakeScrollBackend:
    """记录滚动操作并支持注入异常。"""

    def __init__(self) -> None:
        self.events: list[tuple[int, int]] = []
        self.failure: Exception | None = None

    def scroll(self, dx: int, dy: int) -> None:
        self.events.append((dx, dy))
        if self.failure is not None:
            raise self.failure


@pytest.mark.parametrize(
    ("text", "keys"),
    [
        ("Hello", list("Hello")),
        ("中文", list("中文")),
        ("中A文", list("中A文")),
        (" ", [" "]),
        ("\t", [FakeKey.tab]),
        ("\n", [FakeKey.enter]),
        ("\r", [FakeKey.enter]),
        ("\r\n", [FakeKey.enter, FakeKey.enter]),
        ("😀", ["😀"]),
        ("𠮷", ["𠮷"]),
    ],
)
def test_type_passes_characters_in_order(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    keys: list[object],
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "linux")
    controller = _controller(fake_environment)

    result = controller.type(text)

    expected = [
        event
        for key in keys
        for event in (("press", key), ("release", key))
    ]
    assert result is None
    assert fake_environment.keyboard.events == expected
    assert fake_environment.sleeps == [0.05] * max(len(text) - 1, 0) + [0.1]


def test_type_empty_string_is_complete_noop(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    result = controller.type("")

    assert result is None
    assert fake_environment.keyboard.events == []
    assert fake_environment.text_factory_calls == []
    assert fake_environment.sleeps == []


def test_type_non_string_fails_without_backend(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(TypeError):
        controller.type(1)  # type: ignore[arg-type]

    assert fake_environment.keyboard.events == []
    assert fake_environment.text_factory_calls == []
    assert fake_environment.sleeps == []


@pytest.mark.parametrize(
    ("operation", "call_number", "expected_events"),
    [
        ("press", 1, [("press", "a")]),
        ("release", 1, [("press", "a"), ("release", "a")]),
        (
            "press",
            2,
            [("press", "a"), ("release", "a"), ("press", "b")],
        ),
    ],
)
def test_type_backend_failure_stops_immediately(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    call_number: int,
    expected_events: list[tuple[str, object]],
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "linux")
    original = RuntimeError("backend failed")
    fake_environment.keyboard.fail_on(operation, call_number, original)
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.type("abc")

    assert caught.value.__cause__ is original
    assert fake_environment.keyboard.events == expected_events
    assert 0.1 not in fake_environment.sleeps


@pytest.mark.parametrize(
    ("failure_at", "expected_events"),
    [
        (1, [("press", "a"), ("release", "a")]),
        (
            2,
            [
                ("press", "a"),
                ("release", "a"),
                ("press", "b"),
                ("release", "b"),
            ],
        ),
    ],
)
def test_type_interval_failure_stops(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: int,
    expected_events: list[tuple[str, object]],
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "linux")
    calls = 0

    def fail_selected(duration: float) -> None:
        nonlocal calls
        calls += 1
        fake_environment.sleeps.append(duration)
        if calls == failure_at:
            raise RuntimeError("sleep failed")

    monkeypatch.setattr(keyboard_controller, "_sleep", fail_selected)
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError):
        controller.type("abc")

    assert fake_environment.keyboard.events == expected_events
    assert 0.1 not in fake_environment.sleeps


def test_type_action_delay_failure_is_wrapped(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "linux")
    original = RuntimeError("sleep failed")
    fake_environment.sleep_failure[0] = original
    controller = _controller(fake_environment, typing_interval=0)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.type("x")

    assert caught.value.__cause__ is original
    assert fake_environment.keyboard.events == [
        ("press", "x"),
        ("release", "x"),
    ]
    assert fake_environment.sleeps == [0.1]


def test_sensitive_type_data_is_not_logged_or_exposed(
    fake_environment: FakeEnvironment,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "linux")
    sensitive = "密_TOKEN_password-123"
    original = RuntimeError("backend failed")
    fake_environment.keyboard.fail_on("press", 1, original)
    controller = _controller(fake_environment)

    with caplog.at_level(logging.ERROR), pytest.raises(
        KeyboardOperationError
    ) as caught:
        controller.type(sensitive)

    combined = " ".join(record.message for record in caplog.records)
    combined += str(caught.value)
    assert sensitive not in combined
    assert "密" not in combined
    assert "TOKEN" not in combined
    assert "password" not in combined
    assert "123" not in combined


@pytest.mark.parametrize(
    ("text", "expected_batches"),
    [
        ("中文", [(0x4E2D,), (0x6587,)]),
        ("A 中 b", [(0x41,), (0x20,), (0x4E2D,), (0x20,), (0x62,)]),
        ("😀", [(0xD83D, 0xDE00)]),
        ("𠮷", [(0xD842, 0xDFB7)]),
    ],
)
def test_windows_type_builds_strict_utf16_batches(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    expected_batches: list[tuple[int, ...]],
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    controller = _controller(fake_environment)

    controller.type(text)

    assert fake_environment.text.batches == expected_batches
    assert fake_environment.sleeps == [0.05] * (len(text) - 1) + [0.1]


@pytest.mark.parametrize("surrogate", [0xD800, 0xDFFF])
def test_windows_type_rejects_isolated_surrogate_before_side_effects(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    surrogate: int,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    controller = _controller(fake_environment)

    with pytest.raises(ValueError, match="孤立 UTF-16 代理项"):
        controller.type(chr(surrogate))

    assert fake_environment.text_factory_calls == []
    assert fake_environment.text.batches == []
    assert fake_environment.keyboard.events == []
    assert fake_environment.sleeps == []


def test_windows_type_validates_full_text_before_input(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    controller = _controller(fake_environment)

    with pytest.raises(ValueError, match="孤立 UTF-16 代理项"):
        controller.type("valid" + chr(0xD800))

    assert fake_environment.text_factory_calls == []
    assert fake_environment.text.batches == []
    assert fake_environment.keyboard.events == []
    assert fake_environment.sleeps == []


def test_windows_type_keeps_control_characters_on_keyboard_backend(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    controller = _controller(fake_environment)

    controller.type("\t\n\r\r\n")

    assert fake_environment.text_factory_calls == []
    assert fake_environment.keyboard.events == [
        event
        for key in (
            FakeKey.tab,
            FakeKey.enter,
            FakeKey.enter,
            FakeKey.enter,
            FakeKey.enter,
        )
        for event in (("press", key), ("release", key))
    ]


def test_windows_type_preserves_mixed_text_and_control_order(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    controller = _controller(fake_environment)

    controller.type("A\t中\n")

    assert fake_environment.timeline == [
        ("text", (0x41,)),
        ("keyboard", "press", FakeKey.tab),
        ("keyboard", "release", FakeKey.tab),
        ("text", (0x4E2D,)),
        ("keyboard", "press", FakeKey.enter),
        ("keyboard", "release", FakeKey.enter),
    ]
    assert fake_environment.sleeps == [0.05, 0.05, 0.05, 0.1]


def test_windows_text_backend_is_lazy_and_reused(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    controller = _controller(fake_environment)

    assert fake_environment.text_factory_calls == []

    controller.type("\t")
    assert fake_environment.text_factory_calls == []

    controller.type("A")
    controller.type("b")

    assert fake_environment.text_factory_calls == ["text"]
    assert fake_environment.text.batches == [(0x41,), (0x62,)]
    assert fake_environment.factory_calls == ["keyboard", "scroll"]


def test_windows_text_failure_stops_without_action_delay(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    original = RuntimeError("send failed")
    fake_environment.text.fail_on(2, original)
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.type("abc")

    assert caught.value.__cause__ is original
    assert fake_environment.text.batches == [(0x61,), (0x62,)]
    assert fake_environment.sleeps == [0.05]
    assert 0.1 not in fake_environment.sleeps


def test_windows_text_factory_failure_preserves_cause(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    original = RuntimeError("factory failed")

    def fail_factory() -> FakeTextBackend:
        raise original

    monkeypatch.setattr(
        keyboard_controller,
        "_create_windows_text_backend",
        fail_factory,
    )
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.type("A")

    assert caught.value.__cause__ is original
    assert fake_environment.text.batches == []
    assert fake_environment.keyboard.events == []
    assert fake_environment.sleeps == []
