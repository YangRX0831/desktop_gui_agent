"""提供基于虚拟桌面坐标的鼠标控制。

职责：
    包装 pynput 鼠标后端，提供移动、点击、右击、双击和线性拖拽公共接口。
    公共坐标相对虚拟桌面左上角，允许底层全局坐标具有负偏移。

校验约束：
    坐标必须是 Python int，按钮限于 left/right，延迟和拖拽时间必须是有限
    非负数。所有校验在调用真实后端前完成，非法参数不产生桌面动作。

拖拽约束：
    拖拽按固定时间间隔插值，并保证终点恰好执行一次。中途失败仍尽力释放
    已按下按钮；清理异常不得覆盖最初的移动或后端异常。

安全边界：
    pynput 和 mss 在构造或调用路径动态加载，模块导入不创建控制器。日志只
    记录固定动作类别、参数范围和安全异常位置，不记录窗口或用户内容。
"""

import ctypes
import importlib
import logging
import math
import time
from collections.abc import Callable
from typing import Protocol

from utils.exceptions import MouseOperationError
from utils.logger import log_safe_exception

logger = logging.getLogger(__name__)

_DRAG_INTERVAL = 0.01
_sleep = time.sleep


def _enable_windows_dpi_awareness() -> None:
    """在创建真实鼠标后端前统一 Windows 截图与控制坐标空间。"""
    loader = getattr(ctypes, "windll", None)
    if loader is None:
        return
    try:
        if loader.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except (AttributeError, OSError):
        pass
    try:
        loader.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


class _PynputController(Protocol):
    """描述 pynput Controller 被鼠标适配器使用的最小接口。

    Attributes:
        position: 当前全局鼠标坐标，可读取或写入。

    典型用法是由 ``_PynputMouseBackend`` 包装真实 Controller；测试可提供
    内存对象，而不需要创建桌面输入能力。
    """

    @property
    def position(self) -> tuple[int, int]:
        """读取当前全局像素坐标。"""
        ...

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        """把光标移动到给定全局像素坐标。"""
        ...

    def click(self, button: object, count: int = 1) -> None:
        """点击平台按钮指定次数。"""
        ...

    def press(self, button: object) -> None:
        """按下平台按钮。"""
        ...

    def release(self, button: object) -> None:
        """释放平台按钮。"""
        ...


class _MouseBackend(Protocol):
    """定义 MouseController 与平台鼠标实现之间的内部边界。

    Attributes:
        后端不保存公共配置；动作延迟和坐标校验属于 MouseController。

    典型实现是 ``_PynputMouseBackend``，测试实现只需记录这些调用。
    """

    def move_to(self, x: int, y: int) -> None:
        """移动到全局像素坐标。"""
        ...

    def click(self, button: str, count: int) -> None:
        """点击已映射按钮。"""
        ...

    def press(self, button: str) -> None:
        """按下已映射按钮。"""
        ...

    def release(self, button: str) -> None:
        """释放已映射按钮。"""
        ...


class _PynputMouseBackend:
    """把项目按钮名称和像素坐标转换为 pynput 调用。

    Attributes:
        controller: 延迟创建的 pynput Controller。
        buttons: left/right 名称到 pynput Button 的固定映射。

    本类假设上层已完成参数校验，不自行记录用户动作或增加重试。
    """

    def __init__(
        self,
        controller: _PynputController,
        buttons: dict[str, object],
    ) -> None:
        """保存平台 Controller 与固定按钮映射。"""
        self._controller = controller
        self._buttons = buttons

    def move_to(self, x: int, y: int) -> None:
        """把像素坐标写入 pynput position。"""
        self._controller.position = (x, y)

    def click(self, button: str, count: int) -> None:
        """把项目按钮名映射为 pynput Button 后点击。"""
        self._controller.click(self._buttons[button], count)

    def press(self, button: str) -> None:
        """按下映射后的 pynput Button。"""
        self._controller.press(self._buttons[button])

    def release(self, button: str) -> None:
        """释放映射后的 pynput Button。"""
        self._controller.release(self._buttons[button])


def _create_backend() -> _MouseBackend:
    """动态导入 pynput 并创建项目内部鼠标后端。

    延迟创建避免模块导入即获得桌面控制能力。导入、Controller 或按钮
    映射失败都转换为 ``MouseOperationError`` 并保留原因。
    """
    # 真实 Controller 延迟到实例构造路径创建，避免导入模块即获得输入能力，
    # 同时允许模拟测试替换工厂而不接触桌面。
    _enable_windows_dpi_awareness()
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
    """读取包含所有显示器的虚拟桌面边界。

    mss monitor 0 是批准的汇总坐标来源。返回 ``left, top, width, height``；
    不合法数据作为领域异常处理，不自行退回主显示器。
    """
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
    """验证延迟和时长是有限非负数并规范化为 float。

    NaN、无穷和 bool 会使动作时间不可预测，因此在控制调用前拒绝。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是 int 或 float")

    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} 必须是有限数值")
    if normalized < 0:
        raise ValueError(f"{name} 不能小于 0")
    return normalized


def _validate_coordinate(value: object, name: str) -> int:
    """验证公共像素坐标是严格 Python int。

    不在这里限制正负；相对坐标值域需要结合虚拟桌面尺寸判断。
    """
    if type(value) is not int:
        raise TypeError(f"{name} 必须是 Python int")
    return value


def _validate_optional_coordinates(
    x: int | None,
    y: int | None,
) -> tuple[int, int] | None:
    """要求可选 x/y 同时提供或同时省略。

    省略表示使用当前光标位置，部分坐标会产生含糊动作，故明确拒绝。
    """
    if (x is None) != (y is None):
        raise ValueError("x 和 y 必须同时提供或同时为 None")
    if x is None or y is None:
        return None
    return _validate_coordinate(x, "x"), _validate_coordinate(y, "y")


def _validate_button(button: object) -> str:
    """把鼠标按钮限制为公开支持的 left 或 right 名称。"""
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
    """把虚拟桌面相对坐标转换为 pynput 全局坐标。

    Args:
        x: 相对虚拟桌面左上角的水平像素。
        y: 相对虚拟桌面左上角的垂直像素。
        bounds: mss 提供的全局左上角与尺寸。

    Returns:
        可直接交给 pynput 的全局坐标。
    """
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
        """移动鼠标到相对虚拟桌面左上角的整数像素坐标。

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
        """在当前位置或相对虚拟桌面左上角的像素坐标单击鼠标。

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
        """在当前位置或相对虚拟桌面左上角的像素坐标单击鼠标右键。

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
        """在当前位置或相对虚拟桌面左上角的像素坐标双击鼠标左键。

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
        # 先验证终点，确保拖拽按下前失败。
        _to_absolute_coordinates(*end, bounds)

        self._run_backend(
            "移动到拖拽起点",
            lambda: self._backend.move_to(*absolute_start),
        )
        self._run_backend("按下拖拽左键", lambda: self._backend.press("left"))

        # 按下成功后无论拖拽是否失败都尝试释放；释放失败仅在没有原始
        # 拖拽异常时向上抛出，避免覆盖最先发生的错误。
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
        """执行可选移动和点击，但不应用公共动作后延迟。

        right_click 与 double_click 复用该原子路径，避免重复延迟或边界校验。
        """
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
        """校验虚拟桌面范围并执行一次后端移动。

        后端异常转换为领域异常；动作后延迟由公共调用者单独管理。
        """
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
        """按固定插值点完成拖拽并保证按钮尽力释放。

        主动作异常优先；释放失败仅在没有主异常时成为最终失败。
        """
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
        """移动到单个拖拽插值点并转换后端异常。"""
        absolute_x, absolute_y = _to_absolute_coordinates(*point, bounds)
        self._run_backend(
            f"移动拖拽位置：x={absolute_x}, y={absolute_y}",
            lambda: self._backend.move_to(absolute_x, absolute_y),
        )

    def _sleep_during_drag(self, interval: float) -> None:
        """执行拖拽点间延迟，并用领域异常标记计时失败。"""
        try:
            _sleep(interval)
        except Exception as exc:
            log_safe_exception(logger, "拖拽过程休眠失败", exc)
            raise MouseOperationError("拖拽过程休眠失败") from exc

    def _delay_after_action(self, operation: str) -> None:
        """在完整公共鼠标动作成功后应用一次配置延迟。"""
        try:
            _sleep(self._action_delay)
        except Exception as exc:
            log_safe_exception(logger, "鼠标操作成功后延迟失败", exc)
            raise MouseOperationError(f"{operation} 操作后延迟失败") from exc

    @staticmethod
    def _run_backend(operation: str, action: Callable[[], None]) -> None:
        """执行单个鼠标后端调用并安全包装普通异常。

        ``operation`` 是内部固定动作名，只用于脱敏日志，不包含坐标。
        """
        try:
            action()
        except Exception as exc:
            log_safe_exception(logger, "鼠标后端操作失败", exc)
            raise MouseOperationError(f"鼠标后端操作失败：{operation}") from exc
