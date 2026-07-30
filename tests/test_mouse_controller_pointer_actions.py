"""测试鼠标移动、点击、桌面边界和失败契约。

后端、桌面边界和等待均由 fake 提供，不产生真实鼠标事件。
"""

import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import Any
from typing import cast

import numpy as np
import pytest

from control import mouse_controller
from utils.exceptions import MouseOperationError

from tests.mouse_test_support import SafeEnvironment
from tests.mouse_test_support import _controller
from tests.mouse_test_support import safe_environment


class FakeBackend:
    """记录鼠标调用并支持在指定调用处模拟失败。"""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.failures: dict[tuple[str, int], Exception] = {}

    def fail_on(self, operation: str, call_number: int, exc: Exception) -> None:
        self.failures[(operation, call_number)] = exc

    def move_to(self, x: int, y: int) -> None:
        self._record("move_to", x, y)

    def click(self, button: str, count: int) -> None:
        self._record("click", button, count)

    def press(self, button: str) -> None:
        self._record("press", button)

    def release(self, button: str) -> None:
        self._record("release", button)

    def _record(self, operation: str, *values: object) -> None:
        self.counts[operation] += 1
        self.events.append((operation, *values))
        failure = self.failures.get((operation, self.counts[operation]))
        if failure is not None:
            raise failure


def test_virtual_bounds_closes_mss_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exited: list[bool] = []

    class FakeMSS:
        monitors = [{"left": 0, "top": 0, "width": 10, "height": 10}]

        def __enter__(self) -> "FakeMSS":
            return self

        def __exit__(self, *args: object) -> None:
            exited.append(True)

    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: SimpleNamespace(MSS=FakeMSS),
    )

    mouse_controller._get_virtual_screen_bounds()

    assert exited == [True]


def test_virtual_bounds_allows_negative_left_and_top(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMSS:
        monitors = [{"left": -300, "top": -200, "width": 600, "height": 400}]

        def __enter__(self) -> "FakeMSS":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: SimpleNamespace(MSS=FakeMSS),
    )

    assert mouse_controller._get_virtual_screen_bounds() == (
        -300,
        -200,
        600,
        400,
    )


@pytest.mark.parametrize(
    "monitor",
    [
        {"left": True, "top": 0, "width": 10, "height": 10},
        {"left": 0, "top": False, "width": 10, "height": 10},
        {"left": 0, "top": 0, "width": 10.0, "height": 10},
        {"left": 0, "top": 0, "width": 0, "height": 10},
        {"left": 0, "top": 0, "width": 10, "height": -1},
    ],
)
def test_invalid_virtual_bounds_raise_mouse_operation_error(
    monkeypatch: pytest.MonkeyPatch,
    monitor: dict[str, object],
) -> None:
    class FakeMSS:
        monitors = [monitor]

        def __enter__(self) -> "FakeMSS":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: SimpleNamespace(MSS=FakeMSS),
    )

    with pytest.raises(MouseOperationError):
        mouse_controller._get_virtual_screen_bounds()


def test_virtual_bounds_failure_is_converted_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("mss failed")
    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(MouseOperationError) as exc_info:
        mouse_controller._get_virtual_screen_bounds()

    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize("value", [1.0, "1", None, np.int64(1)])
def test_non_python_int_coordinate_is_rejected(
    safe_environment: SafeEnvironment,
    value: object,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(TypeError):
        controller.move_to(cast(Any, value), 1)


@pytest.mark.parametrize("value", [True, False])
def test_bool_coordinate_is_rejected(
    safe_environment: SafeEnvironment,
    value: bool,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(TypeError):
        controller.move_to(value, 1)


@pytest.mark.parametrize("coordinates", [(-1, 0), (0, -1)])
def test_negative_coordinate_is_rejected(
    safe_environment: SafeEnvironment,
    coordinates: tuple[int, int],
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(ValueError):
        controller.move_to(*coordinates)


@pytest.mark.parametrize("coordinates", [(800, 0), (0, 600)])
def test_coordinate_at_upper_bound_is_rejected(
    safe_environment: SafeEnvironment,
    coordinates: tuple[int, int],
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(ValueError):
        controller.move_to(*coordinates)


def test_last_virtual_pixel_is_valid(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.move_to(799, 599)

    assert safe_environment.backend.events == [("move_to", 699, 549)]


def test_negative_virtual_offset_is_added_to_coordinates(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.move_to(10, 20)

    assert safe_environment.backend.events == [("move_to", -90, -30)]


def test_invalid_coordinate_does_not_query_bounds_or_backend(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(TypeError):
        controller.click(True, 1)

    assert safe_environment.bounds_calls == []
    assert safe_environment.backend.events == []


def test_move_to_sets_absolute_position_and_returns_none(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    result = controller.move_to(100, 100)

    assert result is None
    assert safe_environment.backend.events == [("move_to", 0, 50)]


def test_move_to_queries_bounds_once_and_delays_once(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.move_to(1, 2)

    assert safe_environment.bounds_calls == ["bounds"]
    assert safe_environment.sleeps == [0.1]


def test_move_failure_is_converted_and_does_not_delay(
    safe_environment: SafeEnvironment,
) -> None:
    failure = OSError("move failed")
    safe_environment.backend.fail_on("move_to", 1, failure)
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError) as exc_info:
        controller.move_to(1, 2)

    assert exc_info.value.__cause__ is failure
    assert safe_environment.sleeps == []


def test_click_without_coordinates_does_not_query_bounds(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.click()

    assert safe_environment.bounds_calls == []


def test_click_without_coordinates_clicks_current_position(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.click()

    assert safe_environment.backend.events == [("click", "left", 1)]


def test_click_with_coordinates_moves_before_clicking(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.click(10, 20)

    assert safe_environment.backend.events == [
        ("move_to", -90, -30),
        ("click", "left", 1),
    ]


def test_click_defaults_to_left_count_one(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.click()

    assert safe_environment.backend.events == [("click", "left", 1)]


def test_click_supports_right_count_one(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.click(button="right")

    assert safe_environment.backend.events == [("click", "right", 1)]


@pytest.mark.parametrize("button", [1, None, True])
def test_non_string_button_is_rejected(
    safe_environment: SafeEnvironment,
    button: object,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(TypeError):
        controller.click(button=cast(Any, button))


@pytest.mark.parametrize("button", ["middle", "x1", "LEFT", ""])
def test_unknown_button_is_rejected(
    safe_environment: SafeEnvironment,
    button: str,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(ValueError):
        controller.click(button=button)


def test_only_x_is_rejected_before_bounds_query(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(ValueError):
        controller.click(x=1)

    assert safe_environment.bounds_calls == []


def test_only_y_is_rejected_before_bounds_query(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(ValueError):
        controller.click(y=1)

    assert safe_environment.bounds_calls == []


def test_click_move_failure_prevents_click(
    safe_environment: SafeEnvironment,
) -> None:
    safe_environment.backend.fail_on("move_to", 1, OSError("move failed"))
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError):
        controller.click(1, 2)

    assert safe_environment.backend.counts["click"] == 0


def test_click_failure_does_not_delay(
    safe_environment: SafeEnvironment,
) -> None:
    safe_environment.backend.fail_on("click", 1, OSError("click failed"))
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError):
        controller.click()

    assert safe_environment.sleeps == []


def test_right_click_uses_right_count_one(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.right_click()

    assert safe_environment.backend.events == [("click", "right", 1)]


def test_right_click_with_coordinates_delays_only_once(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.right_click(1, 2)

    assert safe_environment.sleeps == [0.1]
    assert safe_environment.bounds_calls == ["bounds"]


def test_double_click_uses_left_count_two(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.double_click()

    assert safe_environment.backend.events == [("click", "left", 2)]


def test_double_click_with_coordinates_delays_only_once(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.double_click(1, 2)

    assert safe_environment.sleeps == [0.1]
    assert safe_environment.bounds_calls == ["bounds"]


@pytest.mark.parametrize("method_name", ["right_click", "double_click"])
def test_special_click_does_not_call_public_click(
    safe_environment: SafeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    controller = _controller(safe_environment)
    monkeypatch.setattr(
        controller,
        "click",
        lambda *args, **kwargs: pytest.fail("public click must not be called"),
    )

    getattr(controller, method_name)()

    assert safe_environment.sleeps == [0.1]


def test_backend_failure_records_error_log(
    safe_environment: SafeEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    safe_environment.backend.fail_on("click", 1, OSError("failed"))
    controller = _controller(safe_environment)

    with caplog.at_level(logging.ERROR), pytest.raises(MouseOperationError):
        controller.click()

    assert "鼠标后端操作失败" in caplog.text
