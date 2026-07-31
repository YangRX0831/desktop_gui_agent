"""提供键盘输入、快捷键和滚动控制。"""

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
_NAMED_KEYS = frozenset(
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


class _KeyboardBackend(Protocol):
    def press(self, key: object) -> None:
        ...

    def release(self, key: object) -> None:
        ...


class _ScrollBackend(Protocol):
    def scroll(self, dx: int, dy: int) -> None:
        ...


class _TextBackend(Protocol):
    def send(self, code_units: tuple[int, ...]) -> None:
        ...


class _SendInputCallable(Protocol):
    def __call__(
        self,
        input_count: int,
        inputs: object,
        structure_size: int,
    ) -> int:
        ...


# 下列结构的字段顺序和宽度对应 Windows INPUT ABI；布局变化会导致
# SendInput 按错误偏移读取内存。
class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouse_data", ctypes.c_ulong),
        ("flags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("extra_info", ctypes.c_void_p),
    ]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("virtual_key", ctypes.c_ushort),
        ("scan_code", ctypes.c_ushort),
        ("flags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("extra_info", ctypes.c_void_p),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ("message", ctypes.c_ulong),
        ("parameter_low", ctypes.c_ushort),
        ("parameter_high", ctypes.c_ushort),
    ]


class _InputValue(ctypes.Union):
    _fields_ = [
        ("mouse", _MouseInput),
        ("keyboard", _KeyboardInput),
        ("hardware", _HardwareInput),
    ]


class _Input(ctypes.Structure):
    _fields_ = [
        ("input_type", ctypes.c_ulong),
        ("value", _InputValue),
    ]


class _WindowsUnicodeTextBackend:
    def __init__(self, send_input: _SendInputCallable) -> None:
        self._send_input = send_input

    def send(self, code_units: tuple[int, ...]) -> None:
        # KEYEVENTF_UNICODE 按 UTF-16 code unit 发送，绕过键盘布局和
        # Caps Lock；每个 code unit 都需要成对的按下与释放事件。
        # 非 BMP 字符会形成代理对和四个事件，目前仅有模拟测试证据。
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
    keyboard_key: object | None = None
    code_units: tuple[int, ...] | None = None


def _create_keyboard_backend() -> tuple[_KeyboardBackend, object]:
    keyboard_module = importlib.import_module("pynput.keyboard")
    return keyboard_module.Controller(), keyboard_module.Key


def _create_scroll_backend() -> _ScrollBackend:
    mouse_module = importlib.import_module("pynput.mouse")
    return mouse_module.Controller()


def _create_windows_text_backend() -> _TextBackend:
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

    safe_metadata = "" if not metadata else ", ".join(
        f"{name}={value}" for name, value in metadata.items()
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
        """按顺序按下并按相反顺序释放组合键。

        Args:
            *keys: 至少两个互不重复的字符键或命名键。

        Raises:
            TypeError: 任一按键不是字符串。
            ValueError: 按键数量、名称或重复状态无效。
            KeyboardOperationError: 后端按键、释放、清理或延迟失败。
        """
        if len(keys) < 2:
            logger.error("hotkey 按键数量不足")
            raise ValueError("hotkey 至少需要两个键")

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
            raise KeyboardOperationError(
                "快捷键释放失败"
            ) from first_release_error

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
        if not isinstance(key, str):
            logger.error("%s 按键参数类型无效", action)
            raise TypeError("key 必须是 str")
        if len(key) == 1:
            return key, "字符键"

        normalized = key.strip().lower()
        if not normalized:
            logger.error("%s 按键参数为空", action)
            raise ValueError("key 不能为空")
        if normalized not in _NAMED_KEYS:
            logger.error("%s 按键名称无效", action)
            raise ValueError("key 不是支持的命名键")

        try:
            return getattr(self._keys, normalized), normalized
        except AttributeError as exc:
            _log_safe_exception(
                "后端缺少命名键能力",
                exc,
                {"action": action, "key": normalized},
            )
            raise KeyboardOperationError("当前后端不支持批准的命名键") from exc

    def _build_type_actions(self, text: str) -> list[_TypeAction]:
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
                raise KeyboardOperationError(
                    "无法初始化 Windows 文本后端"
                ) from exc
        return self._text_backend

    def _resolve_named_key(self, name: str, action: str) -> object:
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
        for index, key in enumerate(keys):
            if any(key == previous for previous in keys[:index]):
                logger.error("hotkey 包含重复键")
                raise ValueError("hotkey 不允许重复键")

    def _cleanup_pressed(self, pressed: list[object]) -> None:
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
        if type(steps) is not int:
            logger.error("scroll steps 参数类型无效")
            raise TypeError("steps 必须是 Python int")
        if steps <= 0:
            logger.error("scroll steps 参数必须大于 0")
            raise ValueError("steps 必须大于 0")
        return steps
