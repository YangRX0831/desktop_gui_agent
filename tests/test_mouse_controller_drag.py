"""测试拖拽插值、动作顺序及主异常与清理异常优先级。

后端和等待均为内存 fake，不产生真实鼠标事件。
"""

import logging
import math
from collections import defaultdict
from typing import Any
from typing import cast

import pytest

from control import mouse_controller
from utils.exceptions import MouseOperationError

from tests.mouse_test_support import SENSITIVE_LOG_PARTS
from tests.mouse_test_support import SafeEnvironment
from tests.mouse_test_support import _controller
from tests.mouse_test_support import formatted_log_output
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


@pytest.mark.parametrize("value", ["0.5", None, object()])
def test_non_numeric_duration_is_rejected(
    safe_environment: SafeEnvironment,
    value: object,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(TypeError):
        controller.drag_from_to(0, 0, 1, 1, cast(Any, value))


def test_bool_duration_is_rejected(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(TypeError):
        controller.drag_from_to(0, 0, 1, 1, True)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_duration_is_rejected(
    safe_environment: SafeEnvironment,
    value: float,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(ValueError):
        controller.drag_from_to(0, 0, 1, 1, value)


def test_negative_duration_is_rejected(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    with pytest.raises(ValueError):
        controller.drag_from_to(0, 0, 1, 1, -0.1)


def test_zero_duration_drag_has_required_order(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 10, 10, 0)

    assert safe_environment.backend.events == [
        ("move_to", -100, -50),
        ("press", "left"),
        ("move_to", -90, -40),
        ("release", "left"),
    ]
    assert safe_environment.sleeps == [0.1]


def test_zero_duration_drag_has_no_process_sleep(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 10, 10, 0)

    assert safe_environment.sleeps == [0.1]


def test_positive_duration_uses_ceiling_step_count(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 30, 30, 0.025)

    assert safe_environment.backend.counts["move_to"] == 4
    assert len(safe_environment.sleeps[:-1]) == 3


def test_drag_intervals_do_not_exceed_point_zero_one(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 30, 30, 0.025)

    assert all(0 < interval <= 0.01 for interval in safe_environment.sleeps[:-1])


def test_drag_interpolation_is_linear_and_ordered(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 30, 60, 0.03)

    assert safe_environment.backend.events == [
        ("move_to", -100, -50),
        ("press", "left"),
        ("move_to", -90, -30),
        ("move_to", -80, -10),
        ("move_to", -70, 10),
        ("release", "left"),
    ]


def test_drag_final_position_is_exact_endpoint(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(1, 1, 7, 8, 0.025)

    move_events = [
        event for event in safe_environment.backend.events if event[0] == "move_to"
    ]
    assert move_events[-1] == ("move_to", -93, -42)


def test_drag_process_sleep_total_matches_duration(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 10, 10, 0.025)

    assert sum(safe_environment.sleeps[:-1]) == pytest.approx(0.025)


def test_drag_always_uses_left_button(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 1, 1, 0)

    assert ("press", "left") in safe_environment.backend.events
    assert ("release", "left") in safe_environment.backend.events


def test_successful_drag_delays_once_after_release(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 1, 1, 0)

    assert safe_environment.backend.events[-1] == ("release", "left")
    assert safe_environment.sleeps == [0.1]


def test_drag_start_failure_does_not_press_or_release(
    safe_environment: SafeEnvironment,
) -> None:
    safe_environment.backend.fail_on("move_to", 1, OSError("start failed"))
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError):
        controller.drag_from_to(0, 0, 1, 1, 0)

    assert safe_environment.backend.counts["press"] == 0
    assert safe_environment.backend.counts["release"] == 0


def test_drag_press_failure_does_not_release(
    safe_environment: SafeEnvironment,
) -> None:
    safe_environment.backend.fail_on("press", 1, OSError("press failed"))
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError):
        controller.drag_from_to(0, 0, 1, 1, 0)

    assert safe_environment.backend.counts["release"] == 0


def test_drag_move_failure_after_press_still_releases(
    safe_environment: SafeEnvironment,
) -> None:
    safe_environment.backend.fail_on("move_to", 2, OSError("drag move failed"))
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError):
        controller.drag_from_to(0, 0, 10, 10, 0)

    assert safe_environment.backend.counts["release"] == 1


def test_drag_sleep_failure_after_press_still_releases(
    safe_environment: SafeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("sleep failed")
    monkeypatch.setattr(
        mouse_controller,
        "_sleep",
        lambda interval: (_ for _ in ()).throw(failure),
    )
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError) as exc_info:
        controller.drag_from_to(0, 0, 10, 10, 0.01)

    assert exc_info.value.__cause__ is failure
    assert safe_environment.backend.counts["release"] == 1


def test_release_failure_is_converted_and_chained(
    safe_environment: SafeEnvironment,
) -> None:
    failure = OSError("release failed")
    safe_environment.backend.fail_on("release", 1, failure)
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError) as exc_info:
        controller.drag_from_to(0, 0, 1, 1, 0)

    assert exc_info.value.__cause__ is failure


def test_primary_drag_failure_wins_over_release_failure(
    safe_environment: SafeEnvironment,
) -> None:
    primary = OSError("primary failed")
    release = OSError("release failed")
    safe_environment.backend.fail_on("move_to", 2, primary)
    safe_environment.backend.fail_on("release", 1, release)
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError) as exc_info:
        controller.drag_from_to(0, 0, 1, 1, 0)

    assert exc_info.value.__cause__ is primary


@pytest.mark.parametrize(
    "failure_operation",
    ["move_to", "press", "release"],
)
def test_failed_drag_does_not_run_action_delay(
    safe_environment: SafeEnvironment,
    failure_operation: str,
) -> None:
    call_number = 2 if failure_operation == "move_to" else 1
    safe_environment.backend.fail_on(
        failure_operation,
        call_number,
        OSError("failed"),
    )
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError):
        controller.drag_from_to(0, 0, 1, 1, 0)

    assert safe_environment.sleeps == []


def test_drag_uses_one_bounds_snapshot_for_both_endpoints(
    safe_environment: SafeEnvironment,
) -> None:
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 799, 599, 0)

    assert safe_environment.bounds_calls == ["bounds"]


def test_drag_full_call_order_includes_final_delay(
    safe_environment: SafeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordered: list[tuple[Any, ...]] = []
    backend = safe_environment.backend

    def record_sleep(interval: float) -> None:
        ordered.append(("sleep", interval))

    original_record = backend._record

    def record_backend(operation: str, *values: object) -> None:
        ordered.append((operation, *values))
        original_record(operation, *values)

    setattr(backend, "_record", record_backend)
    monkeypatch.setattr(mouse_controller, "_sleep", record_sleep)
    controller = _controller(safe_environment)

    controller.drag_from_to(0, 0, 1, 1, 0)

    assert ordered == [
        ("move_to", -100, -50),
        ("press", "left"),
        ("move_to", -99, -49),
        ("release", "left"),
        ("sleep", 0.1),
    ]


def test_action_delay_failure_is_converted_and_chained(
    safe_environment: SafeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("delay failed")
    monkeypatch.setattr(
        mouse_controller,
        "_sleep",
        lambda interval: (_ for _ in ()).throw(failure),
    )
    controller = _controller(safe_environment)

    with pytest.raises(MouseOperationError) as exc_info:
        controller.click()

    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize("operation", ["move_to", "click", "press", "release"])
def test_backend_failure_preserves_cause(
    safe_environment: SafeEnvironment,
    operation: str,
) -> None:
    failure = OSError(f"{operation} failed")
    safe_environment.backend.fail_on(operation, 1, failure)
    controller = _controller(safe_environment)

    if operation == "move_to":
        call = lambda: controller.move_to(1, 2)
    elif operation == "click":
        call = controller.click
    else:
        call = lambda: controller.drag_from_to(0, 0, 1, 1, 0)

    with pytest.raises(MouseOperationError) as exc_info:
        call()

    assert exc_info.value.__cause__ is failure


def test_failure_log_contains_no_screen_or_user_content(
    safe_environment: SafeEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    safe_environment.backend.fail_on("click", 1, OSError("failed"))
    controller = _controller(safe_environment)

    with caplog.at_level(logging.ERROR), pytest.raises(MouseOperationError):
        controller.click()

    forbidden = ("截图", "窗口标题", "OCR", "用户输入")
    assert all(value not in caplog.text for value in forbidden)


def test_backend_failure_final_log_excludes_sensitive_exception_data(
    safe_environment: SafeEnvironment,
) -> None:
    message = (
        r"SENSITIVE_EXCEPTION_MESSAGE C:\private\model "
        "region=(10,20,30,40) user_text_marker"
    )
    original_error = OSError(message)
    safe_environment.backend.fail_on("click", 1, original_error)
    controller = _controller(safe_environment)

    with formatted_log_output("control.mouse_controller") as stream:
        with pytest.raises(MouseOperationError) as error_info:
            controller.click(10, 20, "right")

    output = stream.getvalue()
    assert error_info.value.__cause__ is original_error
    assert "鼠标后端操作失败" in output
    assert output.count("鼠标后端操作失败") == 1
    assert "button" not in output
    assert all(marker not in output for marker in SENSITIVE_LOG_PARTS)


def test_drag_primary_and_cleanup_failures_are_each_logged_once(
    safe_environment: SafeEnvironment,
) -> None:
    primary = OSError("SENSITIVE_EXCEPTION_MESSAGE user_text_marker")
    cleanup = OSError(r"C:\private\model region=(10,20,30,40)")
    safe_environment.backend.fail_on("move_to", 2, primary)
    safe_environment.backend.fail_on("release", 1, cleanup)
    controller = _controller(safe_environment)

    with formatted_log_output("control.mouse_controller") as stream:
        with pytest.raises(MouseOperationError) as error_info:
            controller.drag_from_to(0, 0, 1, 1, 0)

    output = stream.getvalue()
    assert error_info.value.__cause__ is primary
    assert safe_environment.backend.counts["release"] == 1
    assert output.count("鼠标后端操作失败") == 2
    assert all(marker not in output for marker in SENSITIVE_LOG_PARTS)
