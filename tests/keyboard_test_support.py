"""提供键盘测试共享 fake backend、事件记录和隔离辅助。

本模块只服务测试，不创建真实 pynput Controller、Listener 或 user32 输入。
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

from control import keyboard_controller

APPROVED_KEYS = (
    "alt",
    "alt_l",
    "alt_r",
    "alt_gr",
    "backspace",
    "caps_lock",
    "cmd",
    "cmd_l",
    "cmd_r",
    "ctrl",
    "ctrl_l",
    "ctrl_r",
    "delete",
    "down",
    "end",
    "enter",
    "esc",
    *(f"f{number}" for number in range(1, 21)),
    "home",
    "left",
    "page_down",
    "page_up",
    "right",
    "shift",
    "shift_l",
    "shift_r",
    "space",
    "tab",
    "up",
)

REJECTED_KEYS = (
    "escape",
    "control",
    "command",
    "win",
    "media_play_pause",
    "f21",
    "f22",
    "f23",
    "f24",
    "insert",
    "menu",
    "num_lock",
    "pause",
    "print_screen",
    "scroll_lock",
)


class FakeKey:
    """提供批准命名键的内存值。"""


for _key_name in APPROVED_KEYS:
    setattr(FakeKey, _key_name, object())


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


@dataclass
class FakeEnvironment:
    """保存每个测试独立使用的完全模拟组件。"""

    keyboard: FakeKeyboardBackend
    scroll: FakeScrollBackend
    text: FakeTextBackend
    sleeps: list[float]
    factory_calls: list[str]
    text_factory_calls: list[str]
    sleep_failure: list[Exception | None]
    timeline: list[tuple[object, ...]]


@pytest.fixture
def fake_environment(monkeypatch: pytest.MonkeyPatch) -> FakeEnvironment:
    """为每个测试提供全新的键盘、滚动、文本和等待记录。"""
    timeline: list[tuple[object, ...]] = []
    keyboard = FakeKeyboardBackend(timeline)
    scroll = FakeScrollBackend()
    text = FakeTextBackend(timeline)
    sleeps: list[float] = []
    factory_calls: list[str] = []
    text_factory_calls: list[str] = []
    sleep_failure: list[Exception | None] = [None]

    def create_keyboard() -> tuple[FakeKeyboardBackend, object]:
        factory_calls.append("keyboard")
        return keyboard, FakeKey

    def create_scroll() -> FakeScrollBackend:
        factory_calls.append("scroll")
        return scroll

    def create_text() -> FakeTextBackend:
        text_factory_calls.append("text")
        return text

    def sleep(duration: float) -> None:
        sleeps.append(duration)
        if sleep_failure[0] is not None:
            raise sleep_failure[0]

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
    monkeypatch.setattr(
        keyboard_controller,
        "_create_windows_text_backend",
        create_text,
    )
    monkeypatch.setattr(keyboard_controller, "_sleep", sleep)
    return FakeEnvironment(
        keyboard=keyboard,
        scroll=scroll,
        text=text,
        sleeps=sleeps,
        factory_calls=factory_calls,
        text_factory_calls=text_factory_calls,
        sleep_failure=sleep_failure,
        timeline=timeline,
    )


def _controller(
    fake_environment: FakeEnvironment,
    typing_interval: float = 0.05,
    action_delay: float = 0.1,
) -> keyboard_controller.KeyboardController:
    return keyboard_controller.KeyboardController(
        typing_interval=typing_interval,
        action_delay=action_delay,
    )


def _source() -> str:
    return Path(keyboard_controller.__file__).read_text(encoding="utf-8")


class FakeSendInput:
    """在内存中模拟 SendInput ABI 调用，不接触系统输入。"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, object, int]] = []
        self.result: int | None = None
        self.failure: Exception | None = None
        self.argtypes: tuple[object, ...] | None = None
        self.restype: object | None = None

    def __call__(
        self,
        input_count: int,
        inputs: object,
        structure_size: int,
    ) -> int:
        self.calls.append((input_count, inputs, structure_size))
        if self.failure is not None:
            raise self.failure
        return input_count if self.result is None else self.result


def _keyboard_events(inputs: object, count: int) -> list[tuple[int, int, int]]:
    """从内存 ctypes 数组提取键盘事件字段供 ABI 断言。"""
    return [
        (
            inputs[index].value.keyboard.virtual_key,
            inputs[index].value.keyboard.scan_code,
            inputs[index].value.keyboard.flags,
        )
        for index in range(count)
    ]
