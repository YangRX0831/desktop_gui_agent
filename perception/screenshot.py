"""提供桌面屏幕截图功能。"""

import atexit
import logging
import threading
from collections.abc import Callable

import mss
from mss.exception import ScreenShotError as MssScreenShotError
from PIL import Image

from utils.exceptions import ScreenCaptureError
from utils.safe_logging import log_safe_exception

logger = logging.getLogger(__name__)

# 模块级复用单个 MSS 实例：每次新建实例有明显开销，且进程退出时应释放
# 底层句柄；共享实例必须用可重入锁串行化并发截图。
_mss_instance: mss.MSS | None = None
_mss_lock = threading.RLock()


def _create_mss_instance() -> mss.MSS:
    try:
        factory: type[mss.MSS] | Callable[[], mss.MSS] = mss.MSS
    except AttributeError:
        factory = mss.mss
    return factory()


def _get_mss_instance_unlocked() -> mss.MSS:
    global _mss_instance

    if _mss_instance is None:
        _mss_instance = _create_mss_instance()
    return _mss_instance


def _close_mss_instance_unlocked() -> None:
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
