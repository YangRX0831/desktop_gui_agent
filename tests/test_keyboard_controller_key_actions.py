"""测试键名解析、单键 press/release、延迟和异常契约。

键盘与滚动 backend 均由 fake 隔离。
"""

import logging
import math

import pytest

from control import keyboard_controller
from tests.keyboard_test_support import (
    APPROVED_KEYS,
    REJECTED_KEYS,
    FakeEnvironment,
    FakeKey,
    FakeKeyboardBackend,
    FakeScrollBackend,
    _controller,
    fake_environment,
)
from utils.exceptions import KeyboardOperationError


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


@pytest.mark.parametrize(
    ("typing_interval", "action_delay"),
    [(0, 0), (1, 2), (0.25, 0.5)],
)
def test_valid_delay_values(
    fake_environment: FakeEnvironment,
    typing_interval: float,
    action_delay: float,
) -> None:
    controller = _controller(
        fake_environment,
        typing_interval,
        action_delay,
    )

    assert controller._typing_interval == float(typing_interval)
    assert controller._action_delay == float(action_delay)


@pytest.mark.parametrize(
    ("name", "value", "error_type"),
    [
        ("typing_interval", True, TypeError),
        ("action_delay", False, TypeError),
        ("typing_interval", "0.1", TypeError),
        ("action_delay", None, TypeError),
        ("typing_interval", -0.1, ValueError),
        ("action_delay", -1, ValueError),
        ("typing_interval", math.nan, ValueError),
        ("action_delay", math.inf, ValueError),
        ("typing_interval", -math.inf, ValueError),
    ],
)
def test_invalid_delays_fail_before_factories(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        keyboard_controller,
        "_create_keyboard_backend",
        lambda: calls.append("keyboard"),
    )
    monkeypatch.setattr(
        keyboard_controller,
        "_create_scroll_backend",
        lambda: calls.append("scroll"),
    )
    kwargs = {name: value}

    with pytest.raises(error_type):
        keyboard_controller.KeyboardController(**kwargs)

    assert calls == []


@pytest.mark.parametrize(
    ("factory_name", "message"),
    [
        ("_create_keyboard_backend", "键盘后端初始化失败"),
        ("_create_scroll_backend", "滚动后端初始化失败"),
    ],
)
def test_factory_failure_is_wrapped_and_logged_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    factory_name: str,
    message: str,
) -> None:
    original = RuntimeError("factory failed")
    calls: list[str] = []

    def create_keyboard() -> tuple[FakeKeyboardBackend, object]:
        calls.append("keyboard")
        if factory_name == "_create_keyboard_backend":
            raise original
        return FakeKeyboardBackend(), FakeKey

    def create_scroll() -> FakeScrollBackend:
        calls.append("scroll")
        if factory_name == "_create_scroll_backend":
            raise original
        return FakeScrollBackend()

    monkeypatch.setattr(
        keyboard_controller,
        "_create_keyboard_backend",
        create_keyboard,
    )
    monkeypatch.setattr(
        keyboard_controller,
        "_create_scroll_backend",
        create_scroll,
    )

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(KeyboardOperationError) as caught,
    ):
        keyboard_controller.KeyboardController()

    assert caught.value.__cause__ is original
    expected_calls = (
        ["keyboard"]
        if factory_name == "_create_keyboard_backend"
        else ["keyboard", "scroll"]
    )
    assert calls == expected_calls
    assert sum(record.message.startswith(message) for record in caplog.records) == 1


@pytest.mark.parametrize("key_name", APPROVED_KEYS)
def test_all_approved_named_keys_are_resolved(
    fake_environment: FakeEnvironment,
    key_name: str,
) -> None:
    controller = _controller(fake_environment)

    controller.press(key_name)

    assert fake_environment.keyboard.events == [("press", getattr(FakeKey, key_name))]


def test_approved_named_keys_match_production_whitelist_exactly() -> None:
    assert keyboard_controller._NAMED_KEYS == frozenset(APPROVED_KEYS)


@pytest.mark.parametrize("key_name", REJECTED_KEYS)
def test_rejected_named_keys_fail_without_backend(
    fake_environment: FakeEnvironment,
    key_name: str,
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(ValueError):
        controller.press(key_name)

    assert fake_environment.keyboard.events == []
    assert fake_environment.sleeps == []


@pytest.mark.parametrize(
    ("raw", "resolved"),
    [
        ("CTRL", FakeKey.ctrl),
        (" ctrl ", FakeKey.ctrl),
        ("PAGE_DOWN", FakeKey.page_down),
        (" page_up ", FakeKey.page_up),
        ("A", "A"),
        ("a", "a"),
        ("中", "中"),
        (" ", " "),
        ("space", FakeKey.space),
    ],
)
def test_key_normalization(
    fake_environment: FakeEnvironment,
    raw: str,
    resolved: object,
) -> None:
    controller = _controller(fake_environment)

    controller.press(raw)

    assert fake_environment.keyboard.events == [("press", resolved)]


@pytest.mark.parametrize("key", ["", "   ", "ordinary text"])
def test_invalid_key_values_fail_without_backend(
    fake_environment: FakeEnvironment,
    key: str,
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(ValueError):
        controller.press(key)

    assert fake_environment.keyboard.events == []


def test_non_string_key_fails_without_backend(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(TypeError):
        controller.press(1)  # type: ignore[arg-type]

    assert fake_environment.keyboard.events == []


def test_missing_approved_key_is_backend_error(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(fake_environment)
    monkeypatch.delattr(FakeKey, "ctrl")

    with pytest.raises(KeyboardOperationError) as caught:
        controller.press("ctrl")

    assert isinstance(caught.value.__cause__, AttributeError)
    assert fake_environment.keyboard.events == []


@pytest.mark.parametrize("method_name", ["press", "release"])
def test_single_key_method_only_calls_matching_backend_operation(
    fake_environment: FakeEnvironment,
    method_name: str,
) -> None:
    controller = _controller(fake_environment)

    result = getattr(controller, method_name)("中")

    assert result is None
    assert fake_environment.keyboard.events == [(method_name, "中")]
    assert fake_environment.sleeps == [0.1]


@pytest.mark.parametrize("method_name", ["press", "release"])
def test_single_key_backend_failure_is_wrapped_without_delay(
    fake_environment: FakeEnvironment,
    method_name: str,
) -> None:
    original = RuntimeError("backend failed")
    fake_environment.keyboard.fail_on(method_name, 1, original)
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        getattr(controller, method_name)("x")

    assert caught.value.__cause__ is original
    assert fake_environment.sleeps == []


@pytest.mark.parametrize("method_name", ["press", "release"])
def test_single_key_action_delay_failure_is_wrapped(
    fake_environment: FakeEnvironment,
    method_name: str,
) -> None:
    original = RuntimeError("sleep failed")
    fake_environment.sleep_failure[0] = original
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        getattr(controller, method_name)("x")

    assert caught.value.__cause__ is original
    assert fake_environment.keyboard.events == [(method_name, "x")]
    assert fake_environment.sleeps == [0.1]


@pytest.mark.parametrize("method_name", ["press", "release"])
def test_character_key_is_redacted_from_failure(
    fake_environment: FakeEnvironment,
    caplog: pytest.LogCaptureFixture,
    method_name: str,
) -> None:
    sensitive_character = "密"
    fake_environment.keyboard.fail_on(
        method_name,
        1,
        RuntimeError("backend failed"),
    )
    controller = _controller(fake_environment)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(KeyboardOperationError) as caught,
    ):
        getattr(controller, method_name)(sensitive_character)

    output = " ".join(record.message for record in caplog.records)
    assert sensitive_character not in output
    assert sensitive_character not in str(caught.value)
