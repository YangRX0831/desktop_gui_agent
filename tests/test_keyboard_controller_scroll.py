"""测试键盘控制模块 scroll 方法的方向、步数和异常契约。

滚动 backend 由 fake 隔离，不产生真实滚轮事件。
"""

from collections import defaultdict

import pytest

from tests.keyboard_test_support import FakeEnvironment, _controller, fake_environment
from utils.exceptions import KeyboardOperationError


class FakeKey:
    """提供批准命名键的内存值。"""


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


@pytest.mark.parametrize(
    ("direction", "steps", "expected"),
    [
        ("up", 3, (0, 3)),
        ("down", 4, (0, -4)),
        ("UP", 2, (0, 2)),
        (" down ", 5, (0, -5)),
    ],
)
def test_scroll_normalizes_and_maps_once(
    fake_environment: FakeEnvironment,
    direction: str,
    steps: int,
    expected: tuple[int, int],
) -> None:
    controller = _controller(fake_environment)

    result = controller.scroll(direction, steps)

    assert result is None
    assert fake_environment.scroll.events == [expected]
    assert fake_environment.keyboard.events == []
    assert fake_environment.sleeps == [0.1]


@pytest.mark.parametrize(
    ("direction", "steps", "error_type"),
    [
        (1, 1, TypeError),
        ("", 1, ValueError),
        ("left", 1, ValueError),
        ("up", 1.0, TypeError),
        ("up", True, TypeError),
        ("up", 0, ValueError),
        ("up", -1, ValueError),
    ],
)
def test_scroll_rejects_invalid_parameters_without_backend(
    fake_environment: FakeEnvironment,
    direction: object,
    steps: object,
    error_type: type[Exception],
) -> None:
    controller = _controller(fake_environment)

    with pytest.raises(error_type):
        controller.scroll(direction, steps)  # type: ignore[arg-type]

    assert fake_environment.scroll.events == []
    assert fake_environment.sleeps == []


def test_scroll_backend_failure_is_wrapped_without_delay(
    fake_environment: FakeEnvironment,
) -> None:
    original = RuntimeError("scroll failed")
    fake_environment.scroll.failure = original
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.scroll("up", 3)

    assert caught.value.__cause__ is original
    assert fake_environment.sleeps == []


def test_scroll_action_delay_failure_is_wrapped(
    fake_environment: FakeEnvironment,
) -> None:
    original = RuntimeError("sleep failed")
    fake_environment.sleep_failure[0] = original
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.scroll("down", 2)

    assert caught.value.__cause__ is original
    assert fake_environment.scroll.events == [(0, -2)]
    assert fake_environment.keyboard.events == []
    assert fake_environment.sleeps == [0.1]
