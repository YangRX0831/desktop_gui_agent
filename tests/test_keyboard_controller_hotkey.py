"""测试 hotkey 预校验、按下与逆序释放及多异常清理。

所有按键事件由内存 fake 接收，不产生真实输入。
"""

import logging
from collections import defaultdict

import pytest

from tests.keyboard_test_support import (
    FakeEnvironment,
    FakeKey,
    _controller,
    fake_environment,
)
from utils.exceptions import KeyboardOperationError


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


class FakeTextBackend:
    """记录 UTF-16 文本批次并支持按调用位置注入异常。"""

    def __init__(self, timeline: list[tuple[object, ...]] | None = None) -> None:
        self.batches: list[tuple[int, ...]] = []
        self.failures: dict[int, Exception] = {}
        self.timeline = timeline

    def fail_on(self, call_number: int, exc: Exception) -> None:
        self.failures[call_number] = exc

    def send(self, code_units: tuple[int, ...]) -> None:
        self.batches.append(code_units)
        if self.timeline is not None:
            self.timeline.append(("text", code_units))
        failure = self.failures.get(len(self.batches))
        if failure is not None:
            raise failure


def test_hotkey_success_order_and_delay(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    result = controller.hotkey("ctrl", "shift", "a")

    assert result is None
    assert fake_environment.keyboard.events == [
        ("press", FakeKey.ctrl),
        ("press", FakeKey.shift),
        ("press", "a"),
        ("release", "a"),
        ("release", FakeKey.shift),
        ("release", FakeKey.ctrl),
    ]
    assert fake_environment.sleeps == [0.1]


def test_hotkey_single_key_success_order_and_delay(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    result = controller.hotkey("enter")

    assert result is None
    assert fake_environment.keyboard.events == [
        ("press", FakeKey.enter),
        ("release", FakeKey.enter),
    ]
    assert fake_environment.sleeps == [0.1]


def test_hotkey_two_key_order_regression(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    controller.hotkey("ctrl", "c")

    assert fake_environment.keyboard.events == [
        ("press", FakeKey.ctrl),
        ("press", "c"),
        ("release", "c"),
        ("release", FakeKey.ctrl),
    ]


def test_hotkey_rejects_zero_keys_without_backend(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(ValueError):
        controller.hotkey()

    assert fake_environment.keyboard.events == []
    assert fake_environment.sleeps == []


def test_hotkey_rejects_invalid_single_key_without_backend(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(ValueError):
        controller.hotkey("invalid")

    assert fake_environment.keyboard.events == []
    assert fake_environment.sleeps == []


@pytest.mark.parametrize(
    "keys",
    [
        ("ctrl", "ctrl"),
        ("CTRL", " ctrl "),
    ],
)
def test_hotkey_rejects_normalized_duplicate_keys(
    fake_environment: FakeEnvironment,
    keys: tuple[str, str],
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(ValueError):
        controller.hotkey(*keys)

    assert fake_environment.keyboard.events == []


def test_hotkey_rejects_backend_alias(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FakeKey, "ctrl_l", FakeKey.ctrl)
    controller = _controller(fake_environment)

    with pytest.raises(ValueError):
        controller.hotkey("ctrl", "ctrl_l")

    assert fake_environment.keyboard.events == []


def test_hotkey_validates_all_keys_before_backend(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(ValueError):
        controller.hotkey("ctrl", "invalid")

    assert fake_environment.keyboard.events == []


def test_hotkey_first_press_failure_has_no_cleanup(
    fake_environment: FakeEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = RuntimeError("first press failed")
    fake_environment.keyboard.fail_on("press", 1, original)
    controller = _controller(fake_environment)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(KeyboardOperationError) as caught,
    ):
        controller.hotkey("enter")

    assert caught.value.__cause__ is original
    assert fake_environment.keyboard.events == [("press", FakeKey.enter)]
    assert "enter" not in caplog.text


def test_hotkey_single_release_failure_preserves_cause_and_redacts_key(
    fake_environment: FakeEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = RuntimeError("SENSITIVE_RELEASE_FAILURE")
    fake_environment.keyboard.fail_on("release", 1, original)
    controller = _controller(fake_environment)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(KeyboardOperationError) as caught,
    ):
        controller.hotkey("密")

    assert caught.value.__cause__ is original
    assert fake_environment.keyboard.events == [
        ("press", "密"),
        ("release", "密"),
    ]
    assert fake_environment.sleeps == []
    assert "密" not in caplog.text
    assert "SENSITIVE_RELEASE_FAILURE" not in caplog.text


def test_hotkey_main_failure_is_logged_once(
    fake_environment: FakeEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_environment.keyboard.fail_on(
        "press",
        2,
        RuntimeError("press failed"),
    )
    controller = _controller(fake_environment)

    with caplog.at_level(logging.ERROR), pytest.raises(KeyboardOperationError):
        controller.hotkey("ctrl", "a")

    messages = [record.message for record in caplog.records]
    assert sum(message.startswith("快捷键按下失败") for message in messages) == 1


def test_hotkey_press_failure_cleans_all_pressed_keys(
    fake_environment: FakeEnvironment,
) -> None:
    original = RuntimeError("third press failed")
    fake_environment.keyboard.fail_on("press", 3, original)
    fake_environment.keyboard.fail_on(
        "release",
        1,
        RuntimeError("cleanup failed"),
    )
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.hotkey("ctrl", "shift", "a")

    assert caught.value.__cause__ is original
    assert fake_environment.keyboard.events == [
        ("press", FakeKey.ctrl),
        ("press", FakeKey.shift),
        ("press", "a"),
        ("release", FakeKey.shift),
        ("release", FakeKey.ctrl),
    ]
    assert fake_environment.sleeps == []


def test_hotkey_press_failure_continues_after_multiple_cleanup_failures(
    fake_environment: FakeEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    main_failure = RuntimeError("press failed")
    first_cleanup = RuntimeError("first cleanup failed")
    second_cleanup = RuntimeError("second cleanup failed")
    fake_environment.keyboard.fail_on("press", 4, main_failure)
    fake_environment.keyboard.fail_on("release", 1, first_cleanup)
    fake_environment.keyboard.fail_on("release", 2, second_cleanup)
    controller = _controller(fake_environment)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(KeyboardOperationError) as caught,
    ):
        controller.hotkey("ctrl", "shift", "alt", "a")

    assert caught.value.__cause__ is main_failure
    assert caught.value.__cause__ is not first_cleanup
    assert caught.value.__cause__ is not second_cleanup
    expected_releases = [
        ("release", FakeKey.alt),
        ("release", FakeKey.shift),
        ("release", FakeKey.ctrl),
    ]
    assert fake_environment.keyboard.events == [
        ("press", FakeKey.ctrl),
        ("press", FakeKey.shift),
        ("press", FakeKey.alt),
        ("press", "a"),
        *expected_releases,
    ]
    assert fake_environment.keyboard.events[4:] == expected_releases
    assert fake_environment.keyboard.counts["release"] == 3
    assert (
        sum(record.message.startswith("快捷键按下失败") for record in caplog.records)
        == 1
    )
    assert (
        sum(
            record.message.startswith("快捷键清理释放失败") for record in caplog.records
        )
        == 2
    )
    assert fake_environment.sleeps == []


def test_hotkey_release_failures_keep_first_cause_and_continue(
    fake_environment: FakeEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = RuntimeError("first release failed")
    second = RuntimeError("second release failed")
    fake_environment.keyboard.fail_on("release", 1, first)
    fake_environment.keyboard.fail_on("release", 2, second)
    controller = _controller(fake_environment)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(KeyboardOperationError) as caught,
    ):
        controller.hotkey("ctrl", "shift", "a")

    assert caught.value.__cause__ is first
    assert caught.value.__cause__ is not second
    expected_releases = [
        ("release", "a"),
        ("release", FakeKey.shift),
        ("release", FakeKey.ctrl),
    ]
    assert fake_environment.keyboard.events == [
        ("press", FakeKey.ctrl),
        ("press", FakeKey.shift),
        ("press", "a"),
        *expected_releases,
    ]
    assert fake_environment.keyboard.events[3:] == expected_releases
    assert fake_environment.keyboard.counts["release"] == 3
    assert (
        sum(record.message.startswith("快捷键释放失败") for record in caplog.records)
        == 1
    )
    assert (
        sum(
            record.message.startswith("快捷键后续释放失败") for record in caplog.records
        )
        == 1
    )
    assert fake_environment.sleeps == []


def test_hotkey_action_delay_failure_is_wrapped(
    fake_environment: FakeEnvironment,
) -> None:
    original = RuntimeError("sleep failed")
    fake_environment.sleep_failure[0] = original
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.hotkey("ctrl", "a")

    assert caught.value.__cause__ is original
    assert fake_environment.keyboard.events == [
        ("press", FakeKey.ctrl),
        ("press", "a"),
        ("release", "a"),
        ("release", FakeKey.ctrl),
    ]
    assert fake_environment.sleeps == [0.1]
