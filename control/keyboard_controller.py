"""提供键盘输入、快捷键和滚动控制。

职责：
    提供 Unicode 文本输入、单键按下/释放、组合键和滚动公共接口。命名键
    由固定白名单映射到 pynput，Windows Unicode 文本通过 SendInput 发送。

文本约束：
    可由键盘后端安全表示的字符走普通按键路径，其他字符按 UTF-16 code
    unit 发送。孤立代理项明确拒绝，避免向 Windows 输入 API 传递畸形文本。

组合键约束：
    hotkey 按给定顺序按下、逆序释放；重复的解析后按键被拒绝。任一步失败
    都尽力释放已经按下的键，且清理失败不能覆盖主异常。

滚动与延迟：
    direction 只接受 up/down，steps 是正整数。字符间隔和动作后延迟必须是
    有限非负数，延迟异常转换为领域异常，不在控制器内部自动重试。

安全边界：
    后端只在实例构造时创建，模块导入无桌面副作用。日志不记录输入正文或
    当前字符，只记录索引、长度、动作类别、键名和安全异常位置。
"""

import ctypes
import importlib
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from utils.exceptions import KeyboardOperationError

logger = logging.getLogger(__name__)

_sleep = time.sleep
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
SUPPORTED_NAMED_KEYS = frozenset(
    {
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
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        "f13",
        "f14",
        "f15",
        "f16",
        "f17",
        "f18",
        "f19",
        "f20",
        "home",
        "left",
        "media_next",
        "media_play_pause",
        "media_previous",
        "media_stop",
        "media_volume_down",
        "media_volume_mute",
        "media_volume_up",
        "page_down",
        "page_up",
        "right",
        "shift",
        "shift_l",
        "shift_r",
        "space",
        "tab",
        "up",
    }
)
# 公共键名到 pynput Key 属性名的别名表;详见 _resolve_key_alias。
_NAMED_KEY_ALIASES = {"win": "cmd"}


def is_supported_key(key: object) -> bool:
    """判断按键名称是否可由键盘控制器执行。

    Args:
        key: 待验证的按键值。

    Returns:
        单字符键或控制器支持的命名键（含别名）返回 True，否则返回 False。
    """
    if not isinstance(key, str):
        return False
    if len(key) == 1:
        return True
    normalized = key.strip().lower()
    return normalized != "" and _resolve_key_alias(normalized) is not None


def _resolve_key_alias(normalized: str) -> str | None:
    """把公共键名映射到后端属性名；不支持时返回 None。

    pynput 把 Windows 徽标键与 macOS Command 键统一命名为 ``cmd``；
    为让 Windows 提示词使用更直观的 ``win`` 名称，这里接受 ``win``
    作为 ``cmd`` 的别名，两者映射到同一个后端键对象。
    """
    if normalized in _NAMED_KEY_ALIASES:
        return _NAMED_KEY_ALIASES[normalized]
    return normalized if normalized in SUPPORTED_NAMED_KEYS else None


class _KeyboardBackend(Protocol):
    """定义键盘按下与释放的最小平台接口。

    Attributes:
        后端状态由平台 Controller 管理，本协议不暴露其私有属性。

    ``KeyboardController`` 串行调用该接口；测试可注入内存记录器。
    """

    def press(self, key: object) -> None:
        """按下平台键对象。"""
        ...

    def release(self, key: object) -> None:
        """释放平台键对象。"""
        ...


class _ScrollBackend(Protocol):
    """定义二维滚轮增量的最小平台接口。

    典型实现来自 pynput mouse Controller，项目公共接口只暴露 up/down。
    """

    def scroll(self, dx: int, dy: int) -> None:
        """按二维增量滚动。"""
        ...


class _TextBackend(Protocol):
    """定义发送一组 UTF-16 code unit 的 Windows 文本接口。

    实现必须把整组单元视为一个字符动作，失败时抛出异常供上层包装。
    """

    def send(self, code_units: tuple[int, ...]) -> None:
        """发送一个字符对应的 UTF-16 单元。"""
        ...


class _SendInputCallable(Protocol):
    """描述 ctypes 绑定的 Windows SendInput 函数签名。

    该窄协议避免 production 代码依赖整个 user32 对象，也便于内存测试。
    """

    def __call__(
        self,
        input_count: int,
        inputs: object,
        structure_size: int,
    ) -> int:
        """调用 Win32 SendInput 并返回已插入事件数。"""
        ...


# 下列结构的字段顺序和宽度对应 Windows INPUT ABI；布局变化会导致
# SendInput 按错误偏移读取内存。
class _MouseInput(ctypes.Structure):
    """映射 Windows INPUT 联合体中的 MOUSEINPUT 布局。

    Attributes:
        ``_fields_`` 顺序和 ctypes 宽度必须与 Windows ABI 保持一致。

    本项目不直接构造该分支，但完整联合体布局是 SendInput 的必要约束。
    """

    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouse_data", ctypes.c_ulong),
        ("flags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("extra_info", ctypes.c_void_p),
    ]


class _KeyboardInput(ctypes.Structure):
    """映射 Windows KEYBDINPUT 的固定二进制布局。

    Attributes:
        字段保存虚拟键、Unicode 扫描码、标志、时间和额外指针。

    ``_WindowsUnicodeTextBackend`` 用该结构构造按下与释放事件。
    """

    _fields_ = [
        ("virtual_key", ctypes.c_ushort),
        ("scan_code", ctypes.c_ushort),
        ("flags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("extra_info", ctypes.c_void_p),
    ]


class _HardwareInput(ctypes.Structure):
    """映射 Windows INPUT 联合体中的 HARDWAREINPUT 布局。

    Attributes:
        字段仅为保持联合体 ABI 完整，不由项目业务逻辑直接读取。

    典型使用发生在 ``_InputValue`` 类型定义中。
    """

    _fields_ = [
        ("message", ctypes.c_ulong),
        ("parameter_low", ctypes.c_ushort),
        ("parameter_high", ctypes.c_ushort),
    ]


class _InputValue(ctypes.Union):
    """保存 Windows INPUT 的鼠标、键盘和硬件三种联合分支。

    Attributes:
        ``keyboard`` 是本项目发送 Unicode 文本时使用的分支。

    联合体尺寸由 ctypes 计算并传给 SendInput。
    """

    _fields_ = [
        ("mouse", _MouseInput),
        ("keyboard", _KeyboardInput),
        ("hardware", _HardwareInput),
    ]


class _Input(ctypes.Structure):
    """映射 Windows INPUT 的类型标记和联合值。

    Attributes:
        type: ``_INPUT_KEYBOARD`` 等 Windows 输入类别。
        value: 与 type 对应的 ``_InputValue``。

    典型用法是组成两个元素数组，分别发送 Unicode keydown/keyup。
    """

    _fields_ = [
        ("input_type", ctypes.c_ulong),
        ("value", _InputValue),
    ]


class _WindowsUnicodeTextBackend:
    """通过 Windows SendInput 发送不可由普通按键表示的文本。

    Attributes:
        send_input: 已绑定参数类型的 Win32 调用函数。

    每个 UTF-16 单元发送按下和释放事件；返回数量不完整视为失败。
    """

    def __init__(self, send_input: _SendInputCallable) -> None:
        """保存已经绑定签名的 SendInput 函数。"""
        self._send_input = send_input

    def send(self, code_units: tuple[int, ...]) -> None:
        """为每个 UTF-16 单元发送按下和释放事件。

        SendInput 返回数量不足意味着平台拒绝了部分事件，必须整体失败。
        """
        # KEYEVENTF_UNICODE 按 UTF-16 code unit 发送，绕过键盘布局和
        # Caps Lock；每个 code unit 都需要成对的按下与释放事件。
        # 非 BMP 字符会编码为一对 UTF-16 代理项，因此生成四个输入事件。
        events = tuple(
            _Input(
                input_type=_INPUT_KEYBOARD,
                value=_InputValue(
                    keyboard=_KeyboardInput(
                        virtual_key=0,
                        scan_code=code_unit,
                        flags=_KEYEVENTF_UNICODE | key_up_flag,
                        time=0,
                        extra_info=None,
                    )
                ),
            )
            for code_unit in code_units
            for key_up_flag in (0, _KEYEVENTF_KEYUP)
        )
        input_array = (_Input * len(events))(*events)
        requested_count = len(input_array)
        inserted_count = self._send_input(
            requested_count,
            input_array,
            ctypes.sizeof(_Input),
        )
        if inserted_count != requested_count:
            error_code = ctypes.get_last_error()
            raise OSError(error_code, "SendInput 未完整插入文本事件")


@dataclass(frozen=True)
class _TypeAction:
    """描述一个字符应走普通键盘还是 Unicode 后端。

    Attributes:
        keyboard_key: 普通键盘路径的字符；Unicode 路径为 None。
        code_units: Unicode 路径的 UTF-16 单元；普通路径为 None。

    ``_build_type_actions`` 保证两个字段恰有一个可用。
    """

    keyboard_key: object | None = None
    code_units: tuple[int, ...] | None = None


def _create_keyboard_backend() -> tuple[_KeyboardBackend, object]:
    """动态创建 pynput 键盘 Controller 和命名键容器。

    动态导入避免模块 import 产生桌面控制能力；错误由构造器统一包装。
    """
    keyboard_module = importlib.import_module("pynput.keyboard")
    return keyboard_module.Controller(), keyboard_module.Key


def _create_scroll_backend() -> _ScrollBackend:
    """动态创建 pynput 鼠标 Controller 作为滚动后端。

    滚动与键盘共享公共控制器但保持独立后端，便于分别报告初始化失败。
    """
    mouse_module = importlib.import_module("pynput.mouse")
    return mouse_module.Controller()


def _create_windows_text_backend() -> _TextBackend:
    """绑定 Windows SendInput 并创建 Unicode 文本后端。

    仅在首次需要非普通字符时调用，普通英文任务不会触发 Win32 绑定。
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    send_input = user32.SendInput
    send_input.argtypes = (
        ctypes.c_uint,
        ctypes.POINTER(_Input),
        ctypes.c_int,
    )
    send_input.restype = ctypes.c_uint
    return _WindowsUnicodeTextBackend(send_input)


def _log_safe_exception(
    message: str,
    exc: BaseException,
    metadata: dict[str, str | int] | None = None,
) -> None:
    """记录键盘异常的固定元数据和安全代码位置。

    context 只允许动作、索引、长度和键名等审计字段，不得包含输入字符
    或文本正文。异常对象本身不会交给日志 Formatter。
    """
    traceback = exc.__traceback__
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next

    if traceback is None:
        filename = "<unknown>"
        function_name = "<unknown>"
        line_number = 0
    else:
        code = traceback.tb_frame.f_code
        filename = Path(code.co_filename).name
        function_name = code.co_name
        line_number = traceback.tb_lineno

    safe_metadata = (
        ""
        if not metadata
        else ", ".join(f"{name}={value}" for name, value in metadata.items())
    )
    # 只记录筛选后的操作元数据、异常类型、基础文件名、函数名和行号；
    # 不记录异常正文、绝对路径、完整堆栈或用户输入内容。
    logger.error(
        "%s：%s异常类型=%s，文件=%s，函数=%s，行号=%d",
        message,
        f"{safe_metadata}，" if safe_metadata else "",
        type(exc).__name__,
        filename,
        function_name,
        line_number,
    )


def _validate_delay(value: object, name: str) -> float:
    """验证键盘延迟是有限非负数并规范化为 float。

    bool、NaN 和无穷会破坏动作时序，因此在创建后端前拒绝。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        logger.error("%s 参数类型无效", name)
        raise TypeError(f"{name} 必须是 int 或 float")

    normalized = float(value)
    if not math.isfinite(normalized):
        logger.error("%s 参数不是有限值", name)
        raise ValueError(f"{name} 必须是有限值")
    if normalized < 0:
        logger.error("%s 参数小于 0", name)
        raise ValueError(f"{name} 不能小于 0")
    return normalized


class KeyboardController:
    """执行键盘输入、按键、快捷键和滚动操作。

    该类不保证线程安全，应由调用方串行使用。``press`` 只负责按下按键，
    ``release`` 只负责释放按键；二者独立调用时，调用方需要保证正确
    配对。
    """

    def __init__(
        self,
        typing_interval: float = 0.05,
        action_delay: float = 0.1,
    ) -> None:
        """初始化键盘控制器。

        Args:
            typing_interval: 相邻输入字符之间的固定延迟秒数。
            action_delay: 完整公共操作成功后的固定延迟秒数。

        Raises:
            TypeError: 延迟参数类型无效。
            ValueError: 延迟参数不是有限的非负数。
            KeyboardOperationError: 键盘或滚动后端初始化失败。
        """
        self._typing_interval = _validate_delay(
            typing_interval,
            "typing_interval",
        )
        self._action_delay = _validate_delay(action_delay, "action_delay")

        try:
            self._keyboard, self._keys = _create_keyboard_backend()
        except Exception as exc:
            _log_safe_exception(
                "键盘后端初始化失败",
                exc,
                {"backend": "keyboard"},
            )
            raise KeyboardOperationError("无法初始化键盘后端") from exc

        try:
            self._scroll_backend = _create_scroll_backend()
        except Exception as exc:
            _log_safe_exception(
                "滚动后端初始化失败",
                exc,
                {"backend": "scroll"},
            )
            raise KeyboardOperationError("无法初始化滚动后端") from exc
        self._text_backend: _TextBackend | None = None

    def type(self, text: str) -> None:
        """按字符顺序输入文本。

        Args:
            text: 待输入的 Unicode 文本。

        Raises:
            TypeError: text 不是字符串。
            ValueError: text 包含孤立的 UTF-16 代理项。
            KeyboardOperationError: 按键、释放或延迟失败。
        """
        if not isinstance(text, str):
            logger.error("type 参数类型无效")
            raise TypeError("text 必须是 str")
        if not text:
            return

        actions = self._build_type_actions(text)
        text_length = len(actions)
        for index, action in enumerate(actions):
            if action.code_units is None:
                self._execute_keyboard_type_action(
                    action.keyboard_key,
                    index,
                    text_length,
                )
            else:
                self._execute_windows_text_action(
                    action.code_units,
                    index,
                    text_length,
                )

            if index < text_length - 1:
                self._delay(
                    self._typing_interval,
                    "type",
                    "字符间延迟失败",
                )

        self._delay(self._action_delay, "type", "动作延迟失败")

    def press(self, key: str) -> None:
        """按下一个字符键或命名键。

        Args:
            key: 单个 Unicode 字符或批准的命名键。

        Raises:
            TypeError: key 不是字符串。
            ValueError: key 为空或不是批准的按键。
            KeyboardOperationError: 后端按键、能力检查或延迟失败。
        """
        resolved, description = self._parse_key(key, "press")
        try:
            self._keyboard.press(resolved)
        except Exception as exc:
            _log_safe_exception(
                "按键失败",
                exc,
                {"action": "press", "key": description},
            )
            raise KeyboardOperationError("按键失败") from exc

        self._delay(self._action_delay, "press", "动作延迟失败")

    def release(self, key: str) -> None:
        """释放一个字符键或命名键。

        Args:
            key: 单个 Unicode 字符或批准的命名键。

        Raises:
            TypeError: key 不是字符串。
            ValueError: key 为空或不是批准的按键。
            KeyboardOperationError: 后端释放、能力检查或延迟失败。
        """
        resolved, description = self._parse_key(key, "release")
        try:
            self._keyboard.release(resolved)
        except Exception as exc:
            _log_safe_exception(
                "释放按键失败",
                exc,
                {"action": "release", "key": description},
            )
            raise KeyboardOperationError("释放按键失败") from exc

        self._delay(self._action_delay, "release", "动作延迟失败")

    def hotkey(self, *keys: str) -> None:
        """按顺序按下并按相反顺序释放一个或多个按键。

        Args:
            *keys: 至少一个互不重复的字符键或命名键。

        Raises:
            TypeError: 任一按键不是字符串。
            ValueError: 按键数量、名称或重复状态无效。
            KeyboardOperationError: 后端按键、释放、清理或延迟失败。
        """
        # Agent 没有独立 press 动作，因此单键 hotkey 表示完整按下和释放。
        if not keys:
            logger.error("hotkey 按键数量不足")
            raise ValueError("hotkey 至少需要一个键")

        # 全部键先完成解析和去重，保证无效组合在首次真实按键前失败。
        resolved_keys = [self._parse_key(key, "hotkey")[0] for key in keys]
        self._validate_unique_hotkey(resolved_keys)

        pressed: list[object] = []
        for index, resolved in enumerate(resolved_keys):
            try:
                self._keyboard.press(resolved)
            except Exception as exc:
                _log_safe_exception(
                    "快捷键按下失败",
                    exc,
                    {
                        "action": "hotkey",
                        "index": index,
                        "key_count": len(keys),
                    },
                )
                self._cleanup_pressed(pressed)
                raise KeyboardOperationError("快捷键按下失败") from exc
            pressed.append(resolved)

        # 逆序释放维持组合键的栈式配对；后续释放继续执行，但首个失败
        # 始终保留为最终异常的 cause。
        first_release_error: Exception | None = None
        for index, resolved in enumerate(reversed(pressed)):
            try:
                self._keyboard.release(resolved)
            except Exception as exc:
                if first_release_error is None:
                    first_release_error = exc
                    _log_safe_exception(
                        "快捷键释放失败",
                        exc,
                        {
                            "action": "hotkey",
                            "index": index,
                            "key_count": len(keys),
                        },
                    )
                else:
                    _log_safe_exception(
                        "快捷键后续释放失败",
                        exc,
                        {
                            "action": "hotkey",
                            "index": index,
                            "key_count": len(keys),
                        },
                    )

        if first_release_error is not None:
            raise KeyboardOperationError("快捷键释放失败") from first_release_error

        self._delay(self._action_delay, "hotkey", "动作延迟失败")

    def scroll(self, direction: str, steps: int) -> None:
        """按指定方向滚动一次。

        Args:
            direction: ``up`` 或 ``down``，忽略大小写和首尾空白。
            steps: 严格为正的 Python int。

        Raises:
            TypeError: direction 或 steps 类型无效。
            ValueError: direction 或 steps 值无效。
            KeyboardOperationError: 滚动后端或延迟失败。
        """
        normalized_direction = self._validate_direction(direction)
        validated_steps = self._validate_steps(steps)
        delta = validated_steps if normalized_direction == "up" else -validated_steps

        try:
            self._scroll_backend.scroll(0, delta)
        except Exception as exc:
            _log_safe_exception(
                "滚动失败",
                exc,
                {
                    "action": "scroll",
                    "direction": normalized_direction,
                    "steps": validated_steps,
                },
            )
            raise KeyboardOperationError("滚动失败") from exc

        self._delay(self._action_delay, "scroll", "动作延迟失败")

    def _parse_key(self, key: object, action: str) -> tuple[object, str]:
        """把公共键名解析为后端键对象和安全显示名称。

        单字符原样使用；多字符名称必须位于固定白名单且真实后端存在对应
        属性。错误日志不能包含任意调用方对象表示。
        """
        if not isinstance(key, str):
            logger.error("%s 按键参数类型无效", action)
            raise TypeError("key 必须是 str")
        if len(key) == 1:
            return key, "字符键"

        normalized = key.strip().lower()
        if not normalized:
            logger.error("%s 按键参数为空", action)
            raise ValueError("key 不能为空")
        attribute_name = _resolve_key_alias(normalized)
        if attribute_name is None:
            logger.error("%s 按键名称无效", action)
            raise ValueError("key 不是支持的命名键")

        try:
            return getattr(self._keys, attribute_name), normalized
        except AttributeError as exc:
            _log_safe_exception(
                "后端缺少命名键能力",
                exc,
                {"action": action, "key": normalized},
            )
            raise KeyboardOperationError("当前后端不支持批准的命名键") from exc

    def _build_type_actions(self, text: str) -> list[_TypeAction]:
        """为每个 Unicode 字符选择普通键盘或 Windows 文本路径。

        控制字符和 pynput 可表示字符走普通路径，其余字符编码为 UTF-16。
        该选择在任何输入副作用前完成，以便畸形文本整体失败。
        """
        # 先检查整段文本是否包含孤立的 UTF-16 代理项，避免输入到一半
        # 才发现错误，造成部分文本已经写入且无法回滚。
        for character in text:
            code_point = ord(character)
            if 0xD800 <= code_point <= 0xDFFF:
                logger.error("type 包含孤立 UTF-16 代理项")
                raise ValueError("text 包含孤立 UTF-16 代理项")

        # Tab 使用 Tab 键，CR 和 LF 分别使用 Enter 键处理；因此 CRLF
        # 会产生两次 Enter，其他 Windows 文本通过 Unicode 后端发送。
        actions: list[_TypeAction] = []
        for character in text:
            if character == "\t":
                key = self._resolve_named_key("tab", "type")
                actions.append(_TypeAction(keyboard_key=key))
            elif character in {"\n", "\r"}:
                key = self._resolve_named_key("enter", "type")
                actions.append(_TypeAction(keyboard_key=key))
            elif sys.platform == "win32":
                actions.append(
                    _TypeAction(
                        code_units=self._encode_utf16_units(character),
                    )
                )
            else:
                actions.append(_TypeAction(keyboard_key=character))
        return actions

    @staticmethod
    def _encode_utf16_units(character: str) -> tuple[int, ...]:
        """把一个已验证 Unicode 字符编码为一到两个 UTF-16 单元。

        ``surrogatepass`` 不被使用，孤立代理项必须由调用路径提前拒绝。
        """
        encoded = character.encode("utf-16-le", errors="strict")
        return tuple(
            int.from_bytes(encoded[index : index + 2], "little")
            for index in range(0, len(encoded), 2)
        )

    def _execute_keyboard_type_action(
        self,
        key: object,
        index: int,
        text_length: int,
    ) -> None:
        """按下并释放一个普通字符键，失败时尽力清理。

        日志仅记录字符索引和总长度，不记录字符本身。
        """
        try:
            self._keyboard.press(key)
            self._keyboard.release(key)
        except Exception as exc:
            _log_safe_exception(
                "文本输入失败",
                exc,
                {
                    "action": "type",
                    "index": index,
                    "text_length": text_length,
                },
            )
            raise KeyboardOperationError("文本输入失败") from exc

    def _execute_windows_text_action(
        self,
        code_units: tuple[int, ...],
        index: int,
        text_length: int,
    ) -> None:
        """通过延迟创建的 Windows 后端发送一个字符动作。

        code unit 数量可用于诊断但不还原用户正文。
        """
        backend = self._get_text_backend()
        try:
            backend.send(code_units)
        except Exception as exc:
            _log_safe_exception(
                "Windows 文本输入失败",
                exc,
                {
                    "action": "type",
                    "index": index,
                    "text_length": text_length,
                    "unit_count": len(code_units),
                },
            )
            raise KeyboardOperationError("Windows 文本输入失败") from exc

    def _get_text_backend(self) -> _TextBackend:
        """读取或首次创建进程内控制器专属的 Unicode 后端。

        初始化失败转换为键盘领域异常，下一次调用仍可重新尝试创建。
        """
        # 仅在 Windows 首次输入普通文本时创建该后端；模块导入、控制器
        # 初始化和控制字符输入都不会加载 user32。
        if self._text_backend is None:
            try:
                self._text_backend = _create_windows_text_backend()
            except Exception as exc:
                _log_safe_exception(
                    "Windows 文本后端初始化失败",
                    exc,
                    {"action": "type", "backend": "windows_unicode"},
                )
                raise KeyboardOperationError("无法初始化 Windows 文本后端") from exc
        return self._text_backend

    def _resolve_named_key(self, name: str, action: str) -> object:
        """从后端命名键容器读取批准名称对应的对象。

        白名单存在但平台后端缺少属性时明确失败，不降级成字符序列。
        """
        try:
            return getattr(self._keys, name)
        except AttributeError as exc:
            _log_safe_exception(
                "后端缺少命名键能力",
                exc,
                {"action": action, "key": name},
            )
            raise KeyboardOperationError("当前后端不支持批准的命名键") from exc

    def _validate_unique_hotkey(self, keys: list[object]) -> None:
        """拒绝解析后重复的组合键成员。

        重复按下同一键会使释放状态含糊，不能依赖后端自行纠正。
        """
        for index, key in enumerate(keys):
            if any(key == previous for previous in keys[:index]):
                logger.error("hotkey 包含重复键")
                raise ValueError("hotkey 不允许重复键")

    def _cleanup_pressed(self, pressed: list[object]) -> None:
        """逆序尽力释放 hotkey 已按下的全部键。

        单个释放失败后继续其余清理，避免键长期保持按下状态；清理异常只
        记录安全元数据，不覆盖触发清理的主异常。
        """
        # 尽力逆序释放已按下键；释放失败不能覆盖原始按键异常。
        for index, resolved in enumerate(reversed(pressed)):
            try:
                self._keyboard.release(resolved)
            except Exception as exc:
                _log_safe_exception(
                    "快捷键清理释放失败",
                    exc,
                    {
                        "action": "hotkey_cleanup",
                        "cleanup_index": index,
                    },
                )

    def _delay(self, duration: float, action: str, message: str) -> None:
        """执行已验证延迟并转换时钟后端异常。

        action 和 message 都来自模块内固定字符串，不包含用户数据。
        """
        try:
            _sleep(duration)
        except Exception as exc:
            _log_safe_exception(
                message,
                exc,
                {"action": action},
            )
            raise KeyboardOperationError(message) from exc

    @staticmethod
    def _validate_direction(direction: object) -> str:
        """把滚动方向限制为 up 或 down 两个公开值。"""
        if not isinstance(direction, str):
            logger.error("scroll direction 参数类型无效")
            raise TypeError("direction 必须是 str")
        normalized = direction.strip().lower()
        if normalized not in {"up", "down"}:
            logger.error("scroll direction 参数值无效")
            raise ValueError("direction 只支持 up 或 down")
        return normalized

    @staticmethod
    def _validate_steps(steps: object) -> int:
        """验证滚动步数是严格正整数，拒绝 bool 和零。"""
        if type(steps) is not int:
            logger.error("scroll steps 参数类型无效")
            raise TypeError("steps 必须是 Python int")
        if steps <= 0:
            logger.error("scroll steps 参数必须大于 0")
            raise ValueError("steps 必须大于 0")
        return steps
