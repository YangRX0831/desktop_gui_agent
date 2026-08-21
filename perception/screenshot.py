"""提供桌面屏幕截图功能。

职责：
    使用 PRD 指定的 mss 截取虚拟桌面、单个显示器或显示器内矩形区域，并
    将 BGRX 缓冲区转换为 Pillow RGB 图像。

坐标约束：
    ``region`` 相对所选 monitor 左上角，必须完全落在该 monitor 内。
    ``screen_id=0`` 使用 mss 的虚拟桌面汇总区域，保留负全局坐标偏移。

资源约束：
    MSS 实例创建成本较高，因此进程内复用并用可重入锁串行访问。grab 失败
    时必须关闭并清空实例，使下一次调用可以重新构造，而非复用损坏句柄。

安全边界：
    参数在 grab 前验证；日志不记录截图内容。底层异常转换为项目异常并保留
    ``__cause__``，供上层在不输出异常正文的情况下判断失败类别。
"""

import atexit
import ctypes
import ctypes.wintypes
import logging
import os
import threading
from collections.abc import Callable
from typing import Literal

import mss
from mss.exception import ScreenShotError as MssScreenShotError
from PIL import Image

from utils.exceptions import ScreenCaptureError
from utils.logger import log_safe_exception

logger = logging.getLogger(__name__)

# 模块级复用单个 MSS 实例：每次新建实例有明显开销，且进程退出时应释放
# 底层句柄；共享实例必须用可重入锁串行化并发截图。
_mss_instance: mss.MSS | None = None
_mss_lock = threading.RLock()
_SHELL_WINDOW_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd"})
_MIN_APP_WINDOW_SIDE = 300
# 可确认接收键盘文本输入的焦点控件窗口类;仅收录 Win32 标准编辑类与浏览器
# 地址栏,未收录的平台控件一律按 unknown 处理,不猜测输入就绪。
_FOCUS_TEXT_INPUT_CLASSES = frozenset(
    {
        "Edit",
        "RichEdit20A",
        "RichEdit20W",
        "RICHEDIT50W",
        "Chrome_AutocompleteEditView",
    },
)


class _Rect(ctypes.Structure):
    """描述 Windows 窗口矩形。"""

    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _GuiThreadInfo(ctypes.Structure):
    """映射 Windows GUITHREADINFO 的焦点相关字段。"""

    _fields_ = [
        ("cb_size", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("hwnd_active", ctypes.wintypes.HWND),
        ("hwnd_focus", ctypes.wintypes.HWND),
        ("hwnd_capture", ctypes.wintypes.HWND),
        ("hwnd_active_owner", ctypes.wintypes.HWND),
        ("hwnd_update", ctypes.wintypes.HWND),
        ("rc_update", _Rect),
        ("c_captures", ctypes.c_uint),
    ]


def minimize_window(hwnd: int) -> bool:
    """最小化指定窗口;失败返回 False,不产生其他副作用。"""
    user32 = _load_user32()
    if not hwnd or user32 is None:
        return False
    try:
        return bool(user32.ShowWindowAsync(hwnd, 6))
    except Exception:
        return False


def activate_window(hwnd: int) -> bool:
    """尝试把指定窗口恢复并带到前台;失败返回 False。

    前台权限受系统限制,失败时调用方应容忍并继续。
    """
    user32 = _load_user32()
    if not hwnd or user32 is None:
        return False
    try:
        user32.ShowWindowAsync(hwnd, 9)
        return bool(user32.SetForegroundWindow(hwnd))
    except Exception:
        return False


FocusControlKindLiteral = Literal["text_input", "other", "none", "unknown"]


def get_focus_control_kind() -> FocusControlKindLiteral:
    """返回前台线程焦点控件类别：text_input/other/none/unknown。

    依据 GetGUIThreadInfo 的 hwndFocus 与窗口类名判断；任何查询失败都返回
    unknown，不以窗口标题或进程名猜测。
    """
    user32 = _load_user32()
    foreground = get_foreground_hwnd()
    if user32 is None or not foreground:
        return "unknown"
    try:
        thread_id = user32.GetWindowThreadProcessId(foreground, None)
        if not thread_id:
            return "unknown"
        info = _GuiThreadInfo()
        info.cb_size = ctypes.sizeof(_GuiThreadInfo)
        if not user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
            return "unknown"
        focus = info.hwnd_focus or 0
        if not focus:
            return "none"
        window_class = ctypes.create_unicode_buffer(256)
        if not user32.GetClassNameW(focus, window_class, 256):
            return "unknown"
        class_name = window_class.value
        if class_name in _FOCUS_TEXT_INPUT_CLASSES:
            return "text_input"
        return "other"
    except Exception:
        return "unknown"


def _load_user32() -> "ctypes.WinDLL | None":
    """返回 Windows user32；其他平台返回 None。

    WinDLL 属性经 typeshed 的 __getattr__ 得到可调用 _FuncPtr,是
    ctypes 动态 API 的诚实边界类型。
    """
    loader = getattr(ctypes, "windll", None)
    return None if loader is None else loader.user32


def get_window_screen_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """返回窗口裁剪到虚拟桌面内的物理矩形;供点击目标命中判定。

    与 ``_window_region`` 不同,本函数不做最小边长过滤,命令行等小窗口
    也能返回可见矩形;无法取得可靠几何时返回 None。负原点(最大化边框
    溢出)窗口按既有截图行为拒绝并返回 None。
    """
    user32 = _load_user32()
    if not hwnd or user32 is None:
        return None
    rect = _Rect()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return _normalize_window_region(
        (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top),
        _dpi_scale_for_hwnd(user32, hwnd),
        _virtual_desktop_bounds(),
        min_side=None,
    )


def get_window_screen_rect_clipped(
    hwnd: int,
) -> tuple[int, int, int, int] | None:
    """返回窗口的物理矩形,负原点按虚拟桌面可见范围截断。

    最大化窗口的原始矩形带有超出屏幕的负边框偏移(如 -7px),
    ``get_window_screen_rect`` 按截图路径的回退语义对其返回 None;
    窗口内容 OCR 等只读验证场景没有全屏回退,需要的是"窗口可见部分"
    的区域,因此本函数把负原点截断到 0 并裁剪到虚拟桌面内。
    窗口完全不在桌面内时返回 None。
    """
    user32 = _load_user32()
    if not hwnd or user32 is None:
        return None
    rect = _Rect()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    scale = _dpi_scale_for_hwnd(user32, hwnd)
    left = round(rect.left * scale)
    top = round(rect.top * scale)
    width = round((rect.right - rect.left) * scale)
    height = round((rect.bottom - rect.top) * scale)
    if left < 0:
        width += left
        left = 0
    if top < 0:
        height += top
        top = 0
    bounds = _virtual_desktop_bounds()
    if bounds is not None:
        bounds_left, bounds_top, bounds_width, bounds_height = bounds
        width = min(width, bounds_left + bounds_width - left)
        height = min(height, bounds_top + bounds_height - top)
    if width <= 0 or height <= 0:
        return None
    return left, top, width, height


def list_visible_windows_zorder() -> list[dict[str, object]]:
    """按 Z 序自顶向下返回可见顶层窗口的物理矩形与前台标记。

    供感知注入使用:调用方据此让模型区分层叠窗口的真实归属,避免点击
    落到视觉误判的窗口。每进程只保留名称,不读取窗口标题。任何失败都
    返回空列表,由调用方按无信息处理。
    """
    user32 = _load_user32()
    if user32 is None:
        return []
    foreground = get_foreground_hwnd()
    results: list[dict[str, object]] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        """收集一个可见顶层窗口的进程与几何信息。"""
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.GetWindowTextLengthW(hwnd) == 0:
            return True
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)
        process_name = ""
        if handle:
            try:
                name = ctypes.create_unicode_buffer(260)
                size = ctypes.wintypes.DWORD(260)
                if ctypes.windll.kernel32.QueryFullProcessImageNameW(
                    handle,
                    0,
                    name,
                    ctypes.byref(size),
                ):
                    process_name = os.path.basename(name.value)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        if not process_name:
            return True
        rect = _Rect()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        if rect.right - rect.left <= 0 or rect.bottom - rect.top <= 0:
            return True
        results.append(
            {
                "hwnd": hwnd,
                "pid": pid.value,
                "process": process_name,
                "rect": (
                    rect.left,
                    rect.top,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                ),
                "foreground": hwnd == foreground,
            },
        )
        return True

    try:
        wnd_enum_proc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.wintypes.HWND,
            ctypes.wintypes.LPARAM,
        )
        user32.EnumWindows(wnd_enum_proc(_callback), 0)
    except Exception:
        return []
    return results


def get_foreground_hwnd() -> int:
    """返回当前可见前台窗口句柄；不可用时返回 0。"""
    user32 = _load_user32()
    if user32 is None:
        return 0
    hwnd = user32.GetForegroundWindow()
    if not hwnd or not user32.IsWindowVisible(hwnd):
        return 0
    return hwnd


def is_window_available(hwnd: int) -> bool:
    """返回指定 HWND 是否仍表示可见窗口。"""
    if type(hwnd) is not int or hwnd <= 0:
        return False
    user32 = _load_user32()
    return bool(
        user32 is not None and user32.IsWindow(hwnd) and user32.IsWindowVisible(hwnd)
    )


def is_window_existing(hwnd: int) -> bool:
    """返回指定 HWND 是否仍属于一个 Windows 窗口。"""
    if type(hwnd) is not int or hwnd <= 0:
        return False
    user32 = _load_user32()
    return bool(user32 is not None and user32.IsWindow(hwnd))


def get_window_process_name(hwnd: int) -> str:
    """返回窗口所属进程的可执行文件名；不可验证时返回空字符串。"""
    if not is_window_existing(hwnd):
        return ""
    loader = getattr(ctypes, "windll", None)
    if loader is None:
        return ""
    pid = ctypes.wintypes.DWORD()
    loader.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = loader.kernel32.OpenProcess(0x1000, False, pid.value)
    if not handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(260)
        size = ctypes.wintypes.DWORD(len(buffer))
        if not loader.kernel32.QueryFullProcessImageNameW(
            handle,
            0,
            buffer,
            ctypes.byref(size),
        ):
            return ""
        return os.path.basename(buffer.value)
    finally:
        loader.kernel32.CloseHandle(handle)


def get_foreground_app_hwnd() -> int:
    """返回可用于区域截图的前台应用窗口句柄。"""
    hwnd = get_foreground_hwnd()
    user32 = _load_user32()
    if not hwnd or user32 is None:
        return 0
    window_class = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, window_class, 256)
    if window_class.value in _SHELL_WINDOW_CLASSES:
        return 0
    return 0 if user32.GetWindowTextLengthW(hwnd) == 0 else hwnd


def _dpi_scale_for_hwnd(user32: "ctypes.WinDLL", hwnd: int) -> float:
    """返回 Windows 逻辑坐标到物理像素的缩放系数。"""
    try:
        if user32.IsProcessDPIAware():
            return 1.0
        dpi = user32.GetDpiForWindow(hwnd)
    except Exception:
        return 1.0
    return dpi / 96.0 if isinstance(dpi, int) and dpi > 0 else 1.0


def _window_region(hwnd: int) -> tuple[int, int, int, int] | None:
    """返回应用窗口的全局物理截图区域。"""
    user32 = _load_user32()
    if not hwnd or user32 is None:
        return None
    rect = _Rect()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return _normalize_window_region(
        (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top),
        _dpi_scale_for_hwnd(user32, hwnd),
        _virtual_desktop_bounds(),
    )


def _normalize_window_region(
    rect: tuple[int, int, int, int],
    scale: float,
    bounds: tuple[int, int, int, int] | None,
    min_side: int | None = _MIN_APP_WINDOW_SIDE,
) -> tuple[int, int, int, int] | None:
    """把窗口逻辑矩形转换为裁剪到虚拟桌面内的物理区域。

    最大化或贴近屏幕边缘的窗口,其物理矩形可能超出虚拟桌面右/下边界
    (含 DPI 缩放后的边框溢出);越界区域交给 ``capture_screen`` 只会被
    整体拒绝并回退全屏,因此这里先按可见范围裁剪。负原点窗口保持拒绝
    回退全屏,维持最大化窗口的既有截图行为。
    """
    left, top, width, height = rect
    left = round(left * scale)
    top = round(top * scale)
    width = round(width * scale)
    height = round(height * scale)
    if left < 0 or top < 0:
        return None
    if bounds is not None:
        bounds_left, bounds_top, bounds_width, bounds_height = bounds
        width = min(width, bounds_left + bounds_width - left)
        height = min(height, bounds_top + bounds_height - top)
    if min_side is not None and (width < min_side or height < min_side):
        return None
    return left, top, width, height


def _virtual_desktop_bounds() -> tuple[int, int, int, int] | None:
    """读取 mss 虚拟桌面汇总边界;失败时返回 None,由调用方跳过裁剪。"""
    try:
        with _mss_lock:
            screen_capture = _get_mss_instance_unlocked()
            monitor = screen_capture.monitors[0]
        left = int(monitor["left"])
        top = int(monitor["top"])
        width = int(monitor["width"])
        height = int(monitor["height"])
    except (
        AttributeError,
        KeyError,
        MssScreenShotError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None
    if width <= 0 or height <= 0:
        return None
    return left, top, width, height


def virtual_desktop_origin() -> tuple[int, int] | None:
    """返回 mss 虚拟桌面汇总(monitors[0])的左上角原点;失败返回 None。

    供把虚拟桌面绝对坐标(如窗口物理矩形)换算为 ``capture_screen``
    的 monitors[0] 相对 region;原点不可读时调用方应放弃本次换算。
    """
    bounds = _virtual_desktop_bounds()
    if bounds is None:
        return None
    return bounds[0], bounds[1]


def select_capture_region(
    task_start_foreground: int,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int]]:
    """新前台应用有可靠区域时返回该区域，否则使用完整当前屏幕。"""
    foreground = get_foreground_app_hwnd()
    if foreground and foreground != task_start_foreground:
        region = _window_region(foreground)
        if region is not None:
            return region, (region[0], region[1])
    return None, (0, 0)


def _create_mss_instance() -> mss.MSS:
    """创建一个未共享的 MSS 句柄。

    优先使用新版本公开的 ``MSS`` 构造器，仅在属性不存在时兼容旧入口。
    构造错误由 ``capture_screen`` 的能力边界统一转换。
    """
    try:
        factory: type[mss.MSS] | Callable[[], mss.MSS] = mss.MSS
    except AttributeError:
        factory = mss.mss
    return factory()


def _get_mss_instance_unlocked() -> mss.MSS:
    """在调用方持锁时读取或延迟创建共享 MSS 实例。

    函数名显式标注 unlocked，防止未来调用者误以为内部已处理并发。
    """
    global _mss_instance

    if _mss_instance is None:
        _mss_instance = _create_mss_instance()
    return _mss_instance


def _close_mss_instance_unlocked() -> None:
    """在调用方持锁时清空引用并尽力关闭共享实例。

    先清空引用确保 close 失败也不会让下一次截图复用未知状态的句柄。
    清理失败只记录安全摘要，不覆盖原始 grab 异常。
    """
    global _mss_instance

    # 无论正常清理还是 grab 失败后的恢复路径，都先清空引用再 close，
    # 保证下一次调用能够重新创建实例。
    instance = _mss_instance
    _mss_instance = None
    if instance is None:
        return
    try:
        instance.close()
    except (AttributeError, MssScreenShotError, OSError) as exc:
        log_safe_exception(logger, "屏幕截图资源清理失败", exc)


def _cleanup_mss_instance() -> None:
    """释放当前进程中延迟创建的 MSS 资源。"""
    with _mss_lock:
        _close_mss_instance_unlocked()


atexit.register(_cleanup_mss_instance)


def _validate_screen_id(screen_id: int) -> None:
    """在接触 MSS monitor 列表前校验非负整数索引。

    bool 是 int 子类但不表达屏幕索引，因此必须显式拒绝。
    """
    if isinstance(screen_id, bool) or not isinstance(screen_id, int):
        logger.error("屏幕索引必须是 int，当前类型为 %s", type(screen_id).__name__)
        raise TypeError("screen_id 必须是 int")
    if screen_id < 0:
        logger.error("屏幕索引不能小于 0，当前值为 %d", screen_id)
        raise ValueError("screen_id 不能小于 0")


def _validate_region(
    region: tuple[int, int, int, int] | None,
    screen_width: int,
    screen_height: int,
) -> None:
    """校验相对 monitor 的截图矩形完全位于屏幕内。

    Args:
        region: 相对坐标和尺寸，None 表示完整 monitor。
        screen_width: 所选 monitor 的正整数宽度。
        screen_height: 所选 monitor 的正整数高度。

    Raises:
        TypeError: region 形状中的值不是严格整数。
        ValueError: 矩形尺寸、起点或边界不合法。
    """
    if region is None:
        return
    if not isinstance(region, tuple):
        logger.error("截图区域必须是 tuple，当前类型为 %s", type(region).__name__)
        raise TypeError("region 必须是包含四项的 tuple 或 None")
    if len(region) != 4:
        logger.error("截图区域必须包含四项，当前项数为 %d", len(region))
        raise ValueError("region 必须包含四项")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in region):
        logger.error("截图区域的四项必须都是 int，当前值为 %r", region)
        raise TypeError("region 的四项必须都是 int")

    left, top, width, height = region
    if left < 0 or top < 0:
        logger.error("截图区域左上角坐标不能为负数，当前值为 %r", region)
        raise ValueError("region 的 left 和 top 必须大于等于 0")
    if width <= 0 or height <= 0:
        logger.error("截图区域宽高必须为正数，当前值为 %r", region)
        raise ValueError("region 的 width 和 height 必须大于 0")
    if left + width > screen_width:
        logger.error(
            "截图区域超出屏幕右边界：区域=%r，屏幕宽度=%d",
            region,
            screen_width,
        )
        raise ValueError("region 超出所选屏幕右边界")
    if top + height > screen_height:
        logger.error(
            "截图区域超出屏幕下边界：区域=%r，屏幕高度=%d",
            region,
            screen_height,
        )
        raise ValueError("region 超出所选屏幕下边界")


def capture_screen(
    screen_id: int = 0,
    region: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """截取指定屏幕或屏幕区域。

    Args:
        screen_id: ``mss.monitors`` 的屏幕索引，0 表示虚拟桌面汇总区域。
        region: 相对所选屏幕左上角的 ``(left, top, width, height)`` 区域。

    Returns:
        RGB 模式的屏幕图像。

    Raises:
        TypeError: 屏幕索引或截图区域类型不正确。
        ValueError: 屏幕索引、截图区域数值或边界不合法。
        ScreenCaptureError: 底层截图操作失败。
    """
    _validate_screen_id(screen_id)

    try:
        with _mss_lock:
            screen_capture = _get_mss_instance_unlocked()
            monitors = screen_capture.monitors
            if screen_id >= len(monitors):
                logger.error(
                    "屏幕索引不存在：索引=%d，有效最大索引=%d",
                    screen_id,
                    len(monitors) - 1,
                )
                raise ValueError("screen_id 超出 mss.monitors 的有效索引")

            monitor = monitors[screen_id]
            screen_width = int(monitor["width"])
            screen_height = int(monitor["height"])
            _validate_region(region, screen_width, screen_height)

            if region is None:
                capture_area = {
                    "left": int(monitor["left"]),
                    "top": int(monitor["top"]),
                    "width": screen_width,
                    "height": screen_height,
                }
            else:
                left, top, width, height = region
                capture_area = {
                    "left": int(monitor["left"]) + left,
                    "top": int(monitor["top"]) + top,
                    "width": width,
                    "height": height,
                }
            screenshot = screen_capture.grab(capture_area)
    except (AttributeError, MssScreenShotError, OSError) as exc:
        # grab 失败后关闭并清空共享实例，使下一次调用可以重新初始化。
        with _mss_lock:
            _close_mss_instance_unlocked()
        log_safe_exception(logger, "屏幕截图失败", exc)
        raise ScreenCaptureError(
            f"无法截取屏幕：screen_id={screen_id}, region={region!r}"
        ) from exc

    return Image.frombytes(
        "RGB",
        screenshot.size,
        screenshot.raw,
        "raw",
        "BGRX",
    )
