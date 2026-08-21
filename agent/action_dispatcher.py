"""将已解析动作安全分发到桌面控制层。

设计约束：
    Parser 的成功结果仍按不可信运行时对象处理。分发前再次校验动作名称、
    参数字段、类型、值域和截图尺寸，避免其他 Python 调用方绕过 Parser。

    模型坐标模式由 production 配置显式注入。图像像素直接加
    ``region_offset``；0..1000 相对坐标先映射到当前截图像素，再加偏移。
    实际虚拟桌面边界仍由 ``MouseController`` 校验。

安全边界：
    固定分支是模型动作到控制能力的唯一映射，不接受函数名、模块名或任意
    可调用对象。所有控制异常都由 ``execute_operation`` 转换为布尔结果，
    供上层执行有限单步重试，不把模型输出变成任意代码执行。

    在控制调用前额外执行最小 permission / side-effect 授权：production wiring
    必须为每个 run 注入新建的授权作用域;未注入授权函数(None)或授权返回 False
    的产生副作用动作一律在控制调用前拒绝(deny-before-side-effect)。授权不依
    赖任何任务 recipe,只判断"当前已解析动作是否获当前 run 执行权限"。

典型流程：
    ``GuiAgent`` 提供一个 ``ParsedAction`` 和产生该动作的截图尺寸；成功
    返回 True，校验或控制失败返回 False，且不向上层暴露敏感异常正文。
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

# 产生真实桌面副作用的白名单动作；这些动作在控制调用前必须通过当前 run
# 的授权。finish 仅是模型对编排层的语义信号，不产生副作用，不需要授权。
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

# region 截图模式下截图左上角相对全屏虚拟桌面像素的偏移;全屏模式为 (0, 0)。
# Dispatcher 把 crop-local 图像像素坐标加该偏移得到全局桌面像素;全屏时为 0。
_DEFAULT_REGION_OFFSET = (0, 0)


@dataclass(frozen=True)
class PermissionScope:
    """表示一次 run 的临时动作权限作用域。

    每个用户 task/run 创建一个新的实例(不同 identity);Dispatcher 只在当前
    scope 处于激活状态时放行副作用动作。run 结束后 scope 失效,不得跨 run
    继承。scope 只回答"某 action_type 是否被当前 run 授权",不理解任何具体
    应用、不规划任务。

    Attributes:
        allowed_actions: 当前 run 授权的 action_type 集合。
        token: 每次 run 唯一的标识,确保 task1 scope != task2 scope。
    """

    allowed_actions: frozenset[str]
    token: int


class _MouseController(Protocol):
    """定义分发器需要的最小鼠标控制合同。

    Attributes:
        实现自行保存平台后端；分发器只调用公开动作方法。

    production 使用 ``MouseController``，测试可注入内存控制器。
    """

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
    """定义分发器需要的键盘与滚动控制合同。

    Attributes:
        实现自行保存输入状态；分发器不访问任何私有后端。

    production 使用 ``KeyboardController``，调用限于 type、scroll、hotkey。
    """

    def type(self, text: str) -> None:
        """输入文本。"""

    def scroll(self, direction: str, steps: int) -> None:
        """按方向滚动。"""

    def hotkey(self, *keys: str) -> None:
        """执行组合键。"""


class ActionDispatcher:
    """通过固定白名单把已解析动作分发到桌面控制层。

    Attributes:
        控制器、坐标模式和异常包装器均为私有依赖，不向模型暴露。

    典型用法是由 production wiring 注入鼠标与键盘控制器，然后由
    ``GuiAgent`` 对每个 ``ParsedAction`` 调用 ``dispatch``。
    """

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
            mouse_controller: 已由调用方创建的鼠标控制器。
            keyboard_controller: 已由调用方创建的键盘控制器。
            coordinate_mode: 已批准的 W3 坐标模式。
            operation_executor: 控制操作的布尔结果包装器。

        Raises:
            TypeError: 注入对象未提供所需调用能力。
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
        # 当前 run 的授权作用域;默认 None 即 deny-before-side-effect。
        # 每个 run 由 GuiAgent.reply 经 activate_run_scope 新建并注入,
        # run 结束 clear_run_scope 清空,不跨 run 继承。
        self._run_scope: PermissionScope | None = None

    def activate_run_scope(self, scope: PermissionScope) -> None:
        """为当前 run 注入新建的临时授权作用域。"""
        if not isinstance(scope, PermissionScope):
            raise TypeError("scope 必须是 PermissionScope。")
        self._run_scope = scope

    def clear_run_scope(self) -> None:
        """run 结束后清空当前授权作用域,防止跨 run 继承。"""
        self._run_scope = None

    def _is_authorized(self, action: ParsedAction) -> bool:
        """对产生副作用的动作执行当前 run 授权;无 scope 或 scope 未授权即拒绝。"""
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
            action: ActionParser 已解析的结构化动作。
            screenshot_size: 产生该动作的截图 ``(width, height)``;全屏截图时
                即虚拟桌面尺寸,region 截图时为裁剪后的窗口尺寸。
            region_offset: 截图左上角相对全屏虚拟桌面的 ``(left, top)`` 偏移;
                全屏模式传 ``(0, 0)``。click 的归一化坐标先映射到截图内像素,
                再加该偏移得到全局桌面坐标。

        Returns:
            动作通过校验且控制调用完成时返回 True；否则返回 False。

        ``finish`` 不产生控制副作用，仅返回 True 供编排层识别。产生副作用的
        click/type/scroll/hotkey 在控制调用前通过当前 run 的授权函数（production
        必须注入）；授权返回 False 时直接拒绝，不产生任何控制副作用。
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
            # observe 不产生任何控制副作用;编排层负责等待与重新观察。
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
        """按已配置坐标模式把 click 映射为全局桌面像素。"""
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
        """把单击类动作映射为全局桌面像素并分发。

        image_pixel 校验截图内像素范围；normalized_1000 接受 0..1000，按截图
        尺寸映射到 0..width-1 / 0..height-1。两者最后都加 region_offset。
        """
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
        """把 drag 起终点映射为全局桌面像素后交控制器拖拽。"""
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
        """仅接受唯一 text 字段并转交键盘控制器。

        文本正文不在本层记录或重写，避免日志和编码策略发生漂移。
        """
        if set(params) != {"text"} or not isinstance(params["text"], str):
            return self._validation_failure("type_params")
        return self._operation_executor(
            self._keyboard_controller.type,
            params["text"],
        )

    def _dispatch_scroll(self, params: dict[object, object]) -> bool:
        """验证方向和正整数步数后分发滚动。

        bool 不作为 steps 接受，避免 True 被隐式解释为一次滚动。
        """
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
        """验证不可为空的按键元组并分发组合键。

        每个键必须满足控制器公开白名单，模型不能引入后端私有键对象。
        """
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
        """把截图尺寸收窄为两个正 Python int。

        校验失败返回 None，与 dispatcher 的布尔失败合同保持一致。
        """
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
        """记录固定拒绝类别并返回统一失败值。

        category 由模块内常量分支产生，不包含模型参数或用户正文。
        """
        logger.warning("action_dispatch_rejected：类别=%s", category)
        return False
