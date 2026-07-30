"""提供基于虚拟桌面坐标的鼠标控制。"""

import importlib
import logging
import math
import time
from collections.abc import Callable
from typing import Protocol

from utils.exceptions import MouseOperationError
from utils.safe_logging import log_safe_exception

logger = logging.getLogger(__name__)

_DRAG_INTERVAL = 0.01
_sleep = time.sleep


class _PynputController(Protocol):
    @property
    def position(self) -> tuple[int, int]:
        ...

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        ...

    def click(self, button: object, count: int = 1) -> None:
        ...

    def press(self, button: object) -> None:
        ...

    def release(self, button: object) -> None:
        ...


class _MouseBackend(Protocol):
    def move_to(self, x: int, y: int) -> None:
        ...

    def click(self, button: str, count: int) -> None:
        ...

    def press(self, button: str) -> None:
        ...

    def release(self, button: str) -> None:
        ...


class _PynputMouseBackend:
    def __init__(
        self,
        controller: _PynputController,
        buttons: dict[str, object],
    ) -> None:
        self._controller = controller
        self._buttons = buttons

    def move_to(self, x: int, y: int) -> None:
        self._controller.position = (x, y)

    def click(self, button: str, count: int) -> None:
        self._controller.click(self._buttons[button], count)

    def press(self, button: str) -> None:
        self._controller.press(self._buttons[button])

    def release(self, button: str) -> None:
        self._controller.release(self._buttons[button])


def _create_backend() -> _MouseBackend:
    # 真实 Controller 延迟到实例构造路径创建，避免导入模块即获得输入能力，
    # 同时允许模拟测试替换工厂而不接触桌面。
    try:
        mouse_module = importlib.import_module("pynput.mouse")
        controller = mouse_module.Controller()
        buttons = {
            "left": mouse_module.Button.left,
            "right": mouse_module.Button.right,
        }
    except Exception as exc:
        log_safe_exception(logger, "鼠标后端初始化失败", exc)
        raise MouseOperationError("无法初始化鼠标后端") from exc

    return _PynputMouseBackend(controller, buttons)


def _get_virtual_screen_bounds() -> tuple[int, int, int, int]:
    try:
        mss_module = importlib.import_module("mss")
        try:
            factory = mss_module.MSS
        except AttributeError:
            factory = mss_module.mss
        with factory() as screen_capture:
            monitor = screen_capture.monitors[0]
        values = tuple(monitor[key] for key in ("left", "top", "width", "height"))
    except Exception as exc:
        log_safe_exception(logger, "获取虚拟桌面边界失败", exc)
        raise MouseOperationError("无法获取虚拟桌面边界") from exc

    if (
        len(values) != 4
        or any(type(value) is not int for value in values)
        or values[2] <= 0
        or values[3] <= 0
    ):
        logger.error("虚拟桌面边界数据无效：%r", values)
        raise MouseOperationError("虚拟桌面边界数据无效")

    left, top, width, height = values
    return left, top, width, height


def _validate_non_negative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是 int 或 float")

    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} 必须是有限数值")
    if normalized < 0:
        raise ValueError(f"{name} 不能小于 0")
    return normalized


def _validate_coordinate(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} 必须是 Python int")
    return value


def _validate_optional_coordinates(
    x: int | None,
    y: int | None,
) -> tuple[int, int] | None:
    if (x is None) != (y is None):
        raise ValueError("x 和 y 必须同时提供或同时为 None")
    if x is None or y is None:
        return None
    return _validate_coordinate(x, "x"), _validate_coordinate(y, "y")


def _validate_button(button: object) -> str:
    if not isinstance(button, str):
        raise TypeError("button 必须是 str")
    if button not in {"left", "right"}:
        raise ValueError("button 只支持 left 或 right")
    return button


def _to_absolute_coordinates(
    x: int,
    y: int,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, width, height = bounds
    if x < 0 or x >= width:
        raise ValueError(f"x 超出虚拟桌面范围：x={x}, width={width}")
    if y < 0 or y >= height:
        raise ValueError(f"y 超出虚拟桌面范围：y={y}, height={height}")
    return left + x, top + y


class MouseController:
    """执行经过边界校验的鼠标移动、点击和拖拽操作。"""

    def __init__(self, action_delay: float = 0.1) -> None:
        """初始化鼠标控制器。

        Args:
            action_delay: 每个完整公共操作成功后的延迟秒数。

        Raises:
            TypeError: action_delay 类型无效。
            ValueError: action_delay 不是有限的非负数。
            MouseOperationError: 鼠标后端初始化失败。
        """
        self._action_delay = _validate_non_negative_number(
            action_delay,
            "action_delay",
        )
        try:
            self._backend = _create_backend()
        except MouseOperationError:
            raise
        except Exception as exc:
            log_safe_exception(logger, "鼠标后端初始化失败", exc)
            raise MouseOperationError("无法初始化鼠标后端") from exc

    def move_to(self, x: int, y: int) -> None:
        """移动鼠标到虚拟桌面归一化坐标。

        Args:
            x: 相对虚拟桌面左上角的水平坐标。
            y: 相对虚拟桌面左上角的垂直坐标。

        Raises:
            TypeError: 坐标不是 Python int。
            ValueError: 坐标超出虚拟桌面范围。
            MouseOperationError: 获取边界、移动或延迟失败。
        """
        coordinates = (
            _validate_coordinate(x, "x"),
            _validate_coordinate(y, "y"),
        )
        bounds = _get_virtual_screen_bounds()
        self._move_without_delay(*coordinates, bounds)
        self._delay_after_action("move_to")

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
    ) -> None:
        """在当前位置或指定归一化坐标单击鼠标。

        Args:
            x: 可选的虚拟桌面水平坐标。
            y: 可选的虚拟桌面垂直坐标。
            button: ``left`` 或 ``right``。

        Raises:
            TypeError: 坐标或按钮类型无效。
            ValueError: 坐标组合、范围或按钮值无效。
            MouseOperationError: 获取边界、移动、点击或延迟失败。
        """
        validated_button = _validate_button(button)
        coordinates = _validate_optional_coordinates(x, y)
        self._click_without_delay(coordinates, validated_button, 1)
        self._delay_after_action("click")

    def right_click(
        self,
        x: int | None = None,
        y: int | None = None,
    ) -> None:
        """在当前位置或指定归一化坐标单击鼠标右键。

        Args:
            x: 可选的虚拟桌面水平坐标。
            y: 可选的虚拟桌面垂直坐标。

        Raises:
            TypeError: 坐标类型无效。
            ValueError: 坐标组合或范围无效。
            MouseOperationError: 获取边界、移动、点击或延迟失败。
        """
        coordinates = _validate_optional_coordinates(x, y)
        self._click_without_delay(coordinates, "right", 1)
        self._delay_after_action("right_click")

    def double_click(
        self,
        x: int | None = None,
        y: int | None = None,
    ) -> None:
        """在当前位置或指定归一化坐标双击鼠标左键。

        Args:
            x: 可选的虚拟桌面水平坐标。
            y: 可选的虚拟桌面垂直坐标。

        Raises:
            TypeError: 坐标类型无效。
            ValueError: 坐标组合或范围无效。
            MouseOperationError: 获取边界、移动、点击或延迟失败。
        """
        coordinates = _validate_optional_coordinates(x, y)
        self._click_without_delay(coordinates, "left", 2)
        self._delay_after_action("double_click")

    def drag_from_to(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 0.5,
    ) -> None:
        """按住左键从起点线性拖拽到终点。

        Args:
            x1: 起点水平坐标。
            y1: 起点垂直坐标。
            x2: 终点水平坐标。
            y2: 终点垂直坐标。
            duration: 拖拽过程持续秒数。

        Raises:
            TypeError: 坐标或 duration 类型无效。
            ValueError: 坐标越界或 duration 值无效。
            MouseOperationError: 获取边界、移动、按键、休眠或释放失败。
        """
        start = (
            _validate_coordinate(x1, "x1"),
            _validate_coordinate(y1, "y1"),
        )
        end = (
            _validate_coordinate(x2, "x2"),
            _validate_coordinate(y2, "y2"),
        )
        normalized_duration = _validate_non_negative_number(duration, "duration")
        bounds = _get_virtual_screen_bounds()
        absolute_start = _to_absolute_coordinates(*start, bounds)
        absolute_end = _to_absolute_coordinates(*end, bounds)

        self._run_backend(
            "移动到拖拽起点",
            lambda: self._backend.move_to(*absolute_start),
        )
        self._run_backend("按下拖拽左键", lambda: self._backend.press("left"))

        # 按下成功后无论拖拽是否失败都尝试释放；释放失败仅在没有更早
        # 主异常时成为最终异常，从而保留调用方最需要的初始失败原因。
        primary_error: MouseOperationError | None = None
        try:
            self._perform_drag(
                start,
                end,
                bounds,
                normalized_duration,
            )
        except MouseOperationError as exc:
            primary_error = exc
        finally:
            try:
                self._run_backend(
                    "释放拖拽左键",
                    lambda: self._backend.release("left"),
                )
            except MouseOperationError:
                if primary_error is None:
                    raise

        if primary_error is not None:
            raise primary_error
        self._delay_after_action("drag_from_to")

    def _click_without_delay(
        self,
        coordinates: tuple[int, int] | None,
        button: str,
        count: int,
    ) -> None:
        if coordinates is not None:
            bounds = _get_virtual_screen_bounds()
            self._move_without_delay(*coordinates, bounds)
        self._run_backend(
            f"点击鼠标：button={button}, count={count}",
            lambda: self._backend.click(button, count),
        )

    def _move_without_delay(
        self,
        x: int,
        y: int,
        bounds: tuple[int, int, int, int],
    ) -> None:
        absolute_x, absolute_y = _to_absolute_coordinates(x, y, bounds)
        self._run_backend(
            f"移动鼠标：x={absolute_x}, y={absolute_y}",
            lambda: self._backend.move_to(absolute_x, absolute_y),
        )

    def _perform_drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        bounds: tuple[int, int, int, int],
        duration: float,
    ) -> None:
        if duration == 0:
            self._move_drag_point(end, bounds)
            return

        steps = math.ceil(duration / _DRAG_INTERVAL)
        interval = duration / steps
        for step in range(1, steps + 1):
            # 最后一步直接写入目标点，避免中间插值的舍入误差累积到终点。
            if step == steps:
                point = end
            else:
                ratio = step / steps
                point = (
                    round(start[0] + (end[0] - start[0]) * ratio),
                    round(start[1] + (end[1] - start[1]) * ratio),
                )
            self._move_drag_point(point, bounds)
            self._sleep_during_drag(interval)

    def _move_drag_point(
        self,
        point: tuple[int, int],
        bounds: tuple[int, int, int, int],
    ) -> None:
        absolute_x, absolute_y = _to_absolute_coordinates(*point, bounds)
        self._run_backend(
            f"移动拖拽位置：x={absolute_x}, y={absolute_y}",
            lambda: self._backend.move_to(absolute_x, absolute_y),
        )

    def _sleep_during_drag(self, interval: float) -> None:
        try:
            _sleep(interval)
        except Exception as exc:
            log_safe_exception(logger, "拖拽过程休眠失败", exc)
            raise MouseOperationError("拖拽过程休眠失败") from exc

    def _delay_after_action(self, operation: str) -> None:
        try:
            _sleep(self._action_delay)
        except Exception as exc:
            log_safe_exception(logger, "鼠标操作成功后延迟失败", exc)
            raise MouseOperationError(f"{operation} 操作后延迟失败") from exc

    @staticmethod
    def _run_backend(operation: str, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as exc:
            log_safe_exception(logger, "鼠标后端操作失败", exc)
            raise MouseOperationError(f"鼠标后端操作失败：{operation}") from exc
