"""将已解析动作安全分发到桌面控制层。

即使动作已经通过解析器，本模块仍会在执行前再次校验动作名称、参数字段、
类型、值域、截图尺寸和坐标范围，避免其他 Python 调用方绕过解析边界。

坐标模式由运行配置显式注入。图像像素坐标直接叠加 ``region_offset``；
0..1000 相对坐标先映射到当前截图像素，再叠加偏移。最终虚拟桌面边界仍由
``MouseController`` 校验。

动作到控制能力之间使用固定白名单分支，不接受模型提供的函数名、模块名或任意
可调用对象。产生桌面副作用的动作还必须通过当前任务的临时权限作用域；任务
结束后权限立即清空，不能跨任务继承。
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from agent.action_parser import ParsedAction
from control.keyboard_controller import is_supported_key
from control.operation_executor import execute_operation

logger = logging.getLogger(__name__)

IMAGE_PIXEL_COORDINATE_MODE = "image_pixel"
NORMALIZED_COORDINATE_MODE = "normalized_1000"
COORDINATE_MODES = frozenset(
    {IMAGE_PIXEL_COORDINATE_MODE, NORMALIZED_COORDINATE_MODE},
)

# 这些动作会产生真实桌面副作用，因此执行前必须通过当前任务的权限检查。
# finish 只是编排层的完成信号，不直接操作桌面。
_SIDE_EFFECT_ACTIONS = frozenset(
    {
        "click",
        "right_click",
        "double_click",
        "drag",
        "type",
        "scroll",
        "hotkey",
    },
)

# 区域截图左上角相对虚拟桌面的像素偏移；全屏截图使用 (0, 0)。
_DEFAULT_REGION_OFFSET = (0, 0)


@dataclass(frozen=True)
class PermissionScope:
    """表示单个任务运行期间的临时动作权限作用域。

    每个任务创建独立实例。分发器只在作用域激活且动作类型位于
    ``allowed_actions`` 中时执行有副作用的操作；任务结束后作用域失效。

    Attributes:
        allowed_actions: 当前任务允许执行的动作类型集合。
        token: 当前任务作用域的唯一标识。
    """

    allowed_actions: frozenset[str]
    token: int


class _MouseController(Protocol):
    """定义分发器所需的最小鼠标控制接口。"""

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
    ) -> None:
        """在指定坐标点击鼠标。"""

    def right_click(self, x: int | None = None, y: int | None = None) -> None:
        """在指定坐标单击鼠标右键。"""

    def double_click(self, x: int | None = None, y: int | None = None) -> None:
        """在指定坐标双击鼠标左键。"""

    def drag_from_to(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 0.5,
    ) -> None:
        """按住左键从起点拖拽到终点。"""


class _KeyboardController(Protocol):
    """定义分发器所需的最小键盘与滚动接口。"""

    def type(self, text: str) -> None:
        """输入文本。"""

    def scroll(self, direction: str, steps: int) -> None:
        """按方向滚动。"""

    def hotkey(self, *keys: str) -> None:
        """执行组合键。"""


class ActionDispatcher:
    """通过固定白名单把已解析动作分发到桌面控制层。"""

    def __init__(
        self,
        mouse_controller: _MouseController,
        keyboard_controller: _KeyboardController,
        *,
        coordinate_mode: str = IMAGE_PIXEL_COORDINATE_MODE,
        operation_executor: Callable[..., bool] = execute_operation,
    ) -> None:
        """初始化动作分发器。

        Args:
            mouse_controller: 鼠标控制器。
            keyboard_controller: 键盘控制器。
            coordinate_mode: 图像像素或 0..1000 相对坐标模式。
            operation_executor: 将控制异常转换为布尔结果的执行包装器。

        Raises:
            TypeError: 注入对象未提供所需调用能力。
            ValueError: 坐标模式不受支持。
        """
        if not callable(getattr(mouse_controller, "click", None)):
            raise TypeError("mouse_controller 必须提供可调用的 click。")
        for method_name in ("right_click", "double_click", "drag_from_to"):
            if not callable(getattr(mouse_controller, method_name, None)):
                raise TypeError(
                    "mouse_controller 必须提供可调用的 "
                    "right_click、double_click 和 drag_from_to。",
                )
        for method_name in ("type", "scroll", "hotkey"):
            if not callable(getattr(keyboard_controller, method_name, None)):
                raise TypeError(
                    "keyboard_controller 必须提供可调用的 type、scroll 和 hotkey。",
                )
        if not callable(operation_executor):
            raise TypeError("operation_executor 必须可调用。")
        if not isinstance(coordinate_mode, str):
            raise TypeError("coordinate_mode 必须是 str。")
        if coordinate_mode not in COORDINATE_MODES:
            raise ValueError("coordinate_mode 不是受支持的坐标模式。")

        self._mouse_controller = mouse_controller
        self._keyboard_controller = keyboard_controller
        self._coordinate_mode = coordinate_mode
        self._operation_executor = operation_executor
        self._run_scope: PermissionScope | None = None

    def activate_run_scope(self, scope: PermissionScope) -> None:
        """激活当前任务新建的临时权限作用域。"""
        if not isinstance(scope, PermissionScope):
            raise TypeError("scope 必须是 PermissionScope。")
        self._run_scope = scope

    def clear_run_scope(self) -> None:
        """任务结束后清空权限作用域，防止权限跨任务继承。"""
        self._run_scope = None

    def _is_authorized(self, action: ParsedAction) -> bool:
        """判断有副作用动作是否得到当前任务授权。"""
        scope = self._run_scope
        if scope is None:
            return False
        action_type = action.get("action_type")
        return action_type in scope.allowed_actions

    def dispatch(
        self,
        action: ParsedAction,
        screenshot_size: tuple[int, int],
        region_offset: tuple[int, int] = _DEFAULT_REGION_OFFSET,
    ) -> bool:
        """校验并执行一个白名单动作。

        Args:
            action: 解析器产生的结构化动作。
            screenshot_size: 产生动作的截图尺寸 ``(width, height)``。
            region_offset: 截图左上角相对虚拟桌面的 ``(left, top)`` 偏移。

        Returns:
            动作通过校验且控制调用完成时返回 True，否则返回 False。

        ``finish`` 与兼容协议中的 ``observe`` 不产生控制副作用。有副作用动作在
        调用控制器前必须通过当前任务的权限检查。
        """
        action_object: object = action
        if not isinstance(action_object, dict):
            return self._validation_failure("invalid_action")

        if not (
            isinstance(region_offset, tuple)
            and len(region_offset) == 2
            and all(type(value) is int and value >= 0 for value in region_offset)
        ):
            return self._validation_failure("region_offset")

        action_type = action_object.get("action_type")
        params = action_object.get("params")
        if not isinstance(params, dict):
            return self._validation_failure("invalid_params")

        if action_type in _SIDE_EFFECT_ACTIONS:
            if not self._is_authorized(action):
                return self._validation_failure("unauthorized_action")

        if action_type == "click":
            return self._dispatch_click(params, screenshot_size, region_offset)
        if action_type == "right_click":
            return self._dispatch_click_like(
                params,
                screenshot_size,
                region_offset,
                "right_click",
                self._mouse_controller.right_click,
            )
        if action_type == "double_click":
            return self._dispatch_click_like(
                params,
                screenshot_size,
                region_offset,
                "double_click",
                self._mouse_controller.double_click,
            )
        if action_type == "drag":
            return self._dispatch_drag(params, screenshot_size, region_offset)
        if action_type == "type":
            return self._dispatch_type(params)
        if action_type == "scroll":
            return self._dispatch_scroll(params)
        if action_type == "hotkey":
            return self._dispatch_hotkey(params)
        if action_type == "finish":
            if set(params) != {"result"} or not isinstance(params["result"], str):
                return self._validation_failure("invalid_finish")
            return True
        if action_type == "observe":
            # observe 只触发上层等待与重新观察，不调用桌面控制器。
            if params != {}:
                return self._validation_failure("invalid_observe")
            return True
        return self._validation_failure("unsupported_action")

    def _dispatch_click(
        self,
        params: dict[object, object],
        screenshot_size: tuple[int, int],
        region_offset: tuple[int, int],
    ) -> bool:
        """按当前坐标模式把 click 映射为全局桌面像素。"""
        return self._dispatch_click_like(
            params,
            screenshot_size,
            region_offset,
            "click",
            self._mouse_controller.click,
        )

    def _dispatch_click_like(
        self,
        params: dict[object, object],
        screenshot_size: tuple[int, int],
        region_offset: tuple[int, int],
        category: str,
        executor_method: Callable[..., object],
    ) -> bool:
        """把单击类动作坐标映射为全局桌面像素并分发。"""
        if set(params) != {"x", "y"}:
            return self._validation_failure(f"{category}_params")

        raw_x = params["x"]
        raw_y = params["y"]
        if type(raw_x) is not int or type(raw_y) is not int:
            return self._validation_failure(f"{category}_coordinate_type")

        pixels = self._resolve_point_pixels(
            raw_x,
            raw_y,
            screenshot_size,
            category,
        )
        if pixels is None:
            return False
        global_x = region_offset[0] + pixels[0]
        global_y = region_offset[1] + pixels[1]
        return self._operation_executor(executor_method, global_x, global_y)

    def _dispatch_drag(
        self,
        params: dict[object, object],
        screenshot_size: tuple[int, int],
        region_offset: tuple[int, int],
    ) -> bool:
        """把 drag 起点和终点映射为全局桌面像素后执行拖拽。"""
        if set(params) != {"x1", "y1", "x2", "y2"}:
            return self._validation_failure("drag_params")
        raw_x1 = params["x1"]
        raw_y1 = params["y1"]
        raw_x2 = params["x2"]
        raw_y2 = params["y2"]
        if (
            type(raw_x1) is not int
            or type(raw_y1) is not int
            or type(raw_x2) is not int
            or type(raw_y2) is not int
        ):
            return self._validation_failure("drag_coordinate_type")

        start = self._resolve_point_pixels(
            raw_x1,
            raw_y1,
            screenshot_size,
            "drag_start",
        )
        if start is None:
            return False
        end = self._resolve_point_pixels(
            raw_x2,
            raw_y2,
            screenshot_size,
            "drag_end",
        )
        if end is None:
            return False
        return self._operation_executor(
            self._mouse_controller.drag_from_to,
            region_offset[0] + start[0],
            region_offset[1] + start[1],
            region_offset[0] + end[0],
            region_offset[1] + end[1],
        )

    def _resolve_point_pixels(
        self,
        raw_x: int,
        raw_y: int,
        screenshot_size: tuple[int, int],
        category: str,
    ) -> tuple[int, int] | None:
        """按坐标模式把一对模型坐标映射为截图内像素。

        校验失败时记录固定拒绝类别并返回 None，与布尔失败合同一致。
        """
        size = self._validate_screenshot_size(screenshot_size)
        if size is None:
            return None
        width, height = size
        if self._coordinate_mode == NORMALIZED_COORDINATE_MODE:
            if not 0 <= raw_x <= 1000:
                self._validation_failure(f"{category}_x_range")
                return None
            if not 0 <= raw_y <= 1000:
                self._validation_failure(f"{category}_y_range")
                return None
            return (
                round(raw_x * (width - 1) / 1000),
                round(raw_y * (height - 1) / 1000),
            )
        if not 0 <= raw_x < width:
            self._validation_failure(f"{category}_x_range")
            return None
        if not 0 <= raw_y < height:
            self._validation_failure(f"{category}_y_range")
            return None
        return raw_x, raw_y

    def _dispatch_type(self, params: dict[object, object]) -> bool:
        """验证唯一 text 字段并交给键盘控制器。"""
        if set(params) != {"text"} or not isinstance(params["text"], str):
            return self._validation_failure("type_params")
        return self._operation_executor(
            self._keyboard_controller.type,
            params["text"],
        )

    def _dispatch_scroll(self, params: dict[object, object]) -> bool:
        """验证方向和正整数步数后分发滚动。"""
        if set(params) != {"direction", "steps"}:
            return self._validation_failure("scroll_params")
        direction = params["direction"]
        steps = params["steps"]
        if not isinstance(direction, str) or direction not in {"up", "down"}:
            return self._validation_failure("scroll_direction")
        if type(steps) is not int or steps <= 0:
            return self._validation_failure("scroll_steps")
        return self._operation_executor(
            self._keyboard_controller.scroll,
            direction,
            steps,
        )

    def _dispatch_hotkey(self, params: dict[object, object]) -> bool:
        """验证非空按键元组并分发组合键。"""
        if set(params) != {"keys"}:
            return self._validation_failure("hotkey_params")
        keys = params["keys"]
        if (
            not isinstance(keys, tuple)
            or not keys
            or any(not is_supported_key(key) for key in keys)
        ):
            return self._validation_failure("hotkey_keys")
        return self._operation_executor(self._keyboard_controller.hotkey, *keys)

    @staticmethod
    def _validate_screenshot_size(
        screenshot_size: object,
    ) -> tuple[int, int] | None:
        """把截图尺寸收窄为两个正 Python int。"""
        if not isinstance(screenshot_size, tuple) or len(screenshot_size) != 2:
            ActionDispatcher._validation_failure("screenshot_size_shape")
            return None
        width, height = screenshot_size
        if type(width) is not int or type(height) is not int:
            ActionDispatcher._validation_failure("screenshot_size_type")
            return None
        if width <= 0 or height <= 0:
            ActionDispatcher._validation_failure("screenshot_size_range")
            return None
        return width, height

    @staticmethod
    def _validation_failure(category: str) -> bool:
        """记录固定拒绝类别并返回统一失败值。"""
        logger.warning("action_dispatch_rejected：类别=%s", category)
        return False
