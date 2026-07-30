"""测试截图参数、区域转换、异常链和安全日志。

所有 mss 调用均由 fake 隔离，不读取真实桌面。
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from mss.exception import ScreenShotError as MssScreenShotError
from PIL import Image

from perception.screenshot import capture_screen
from utils.exceptions import ScreenCaptureError

MONITORS = [
    {"left": -100, "top": 0, "width": 300, "height": 200},
    {"left": 0, "top": 0, "width": 200, "height": 200},
    {"left": -100, "top": 20, "width": 100, "height": 100},
]

SENSITIVE_LOG_PARTS = (
    "SENSITIVE_EXCEPTION_MESSAGE",
    "private",
    "model",
    "region=(10,20,30,40)",
    "user_text_marker",
)


@contextmanager
def formatted_log_output(logger_name: str) -> Iterator[StringIO]:
    """捕获最终 Formatter 输出，并在退出时恢复 logger 状态。"""
    target_logger = logging.getLogger(logger_name)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s|%(name)s|%(message)s"))
    previous_level = target_logger.level
    previous_propagate = target_logger.propagate
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.ERROR)
    target_logger.propagate = False
    try:
        yield stream
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)
        target_logger.propagate = previous_propagate


class FakeScreenshot:
    """提供可供 Pillow 转换的模拟截图。"""

    size = (2, 1)
    bgra = bytes(
        (
            0,
            0,
            255,
            0,
            0,
            255,
            0,
            0,
        )
    )


class FakeMss:
    """模拟 mss 上下文管理器及截图调用。"""

    def __init__(
        self,
        monitors: list[dict[str, int]] | None = None,
        grab_error: BaseException | None = None,
    ) -> None:
        self.monitors = MONITORS if monitors is None else monitors
        self.grab_error = grab_error
        self.grabbed_area: dict[str, int] | None = None
        self.entered = False
        self.exited = False

    def __enter__(self) -> "FakeMss":
        self.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.exited = True
        return None

    def grab(self, area: dict[str, int]) -> FakeScreenshot:
        self.grabbed_area = area
        if self.grab_error is not None:
            raise self.grab_error
        return FakeScreenshot()


def test_capture_screen_prefers_modern_mss_entry() -> None:
    fake_mss = FakeMss()
    calls: list[str] = []

    def modern_factory() -> FakeMss:
        calls.append("MSS")
        return fake_mss

    def legacy_factory() -> FakeMss:
        pytest.fail("mss must not be called when MSS exists")

    fake_module = SimpleNamespace(MSS=modern_factory, mss=legacy_factory)

    with patch("perception.screenshot.mss", fake_module):
        capture_screen()

    assert calls == ["MSS"]
    assert fake_mss.grabbed_area == MONITORS[0]
    assert fake_mss.entered is True
    assert fake_mss.exited is True


def test_capture_screen_falls_back_to_legacy_mss_entry() -> None:
    fake_mss = FakeMss()
    calls: list[str] = []

    def legacy_factory() -> FakeMss:
        calls.append("mss")
        return fake_mss

    fake_module = SimpleNamespace(mss=legacy_factory)

    with patch("perception.screenshot.mss", fake_module):
        capture_screen()

    assert calls == ["mss"]
    assert fake_mss.grabbed_area == MONITORS[0]
    assert fake_mss.entered is True
    assert fake_mss.exited is True


def test_capture_screen_does_not_fallback_when_modern_constructor_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_error = OSError("modern constructor failed")

    def modern_factory() -> FakeMss:
        raise original_error

    def legacy_factory() -> FakeMss:
        pytest.fail("mss must not be called when MSS construction fails")

    fake_module = SimpleNamespace(MSS=modern_factory, mss=legacy_factory)

    with caplog.at_level(logging.ERROR, logger="perception.screenshot"):
        with patch("perception.screenshot.mss", fake_module):
            with pytest.raises(ScreenCaptureError) as error_info:
                capture_screen()

    assert error_info.value.__cause__ is original_error
    assert "屏幕截图失败" in caplog.text


def test_capture_screen_converts_missing_mss_entries() -> None:
    fake_module = SimpleNamespace()

    with patch("perception.screenshot.mss", fake_module):
        with pytest.raises(ScreenCaptureError) as error_info:
            capture_screen()

    assert isinstance(error_info.value.__cause__, AttributeError)


def test_capture_screen_uses_virtual_desktop_by_default() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        capture_screen()

    assert fake_mss.grabbed_area == MONITORS[0]


def test_capture_screen_uses_selected_physical_monitor() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        capture_screen(screen_id=1)

    assert fake_mss.grabbed_area == MONITORS[1]


def test_capture_screen_uses_relative_region() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        capture_screen(screen_id=1, region=(10, 20, 30, 40))

    assert fake_mss.grabbed_area == {
        "left": 10,
        "top": 20,
        "width": 30,
        "height": 40,
    }


def test_capture_screen_offsets_region_from_negative_monitor_coordinates() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        capture_screen(screen_id=2, region=(10, 5, 20, 30))

    assert fake_mss.grabbed_area == {
        "left": -90,
        "top": 25,
        "width": 20,
        "height": 30,
    }


def test_capture_screen_returns_pillow_image() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        result = capture_screen()

    assert isinstance(result, Image.Image)


def test_capture_screen_returns_rgb_image() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        result = capture_screen()

    assert result.mode == "RGB"


def test_capture_screen_converts_bgra_channels_to_rgb() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        result = capture_screen()

    assert result.getpixel((0, 0)) == (255, 0, 0)
    assert result.getpixel((1, 0)) == (0, 255, 0)


def test_capture_screen_rejects_non_integer_screen_id() -> None:
    with pytest.raises(TypeError):
        capture_screen(screen_id="0")  # type: ignore[arg-type] - 验证运行时类型校验


def test_capture_screen_rejects_boolean_screen_id() -> None:
    with pytest.raises(TypeError):
        capture_screen(screen_id=True)


def test_capture_screen_rejects_negative_screen_id() -> None:
    with pytest.raises(ValueError):
        capture_screen(screen_id=-1)


def test_capture_screen_rejects_screen_id_out_of_range() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ValueError):
            capture_screen(screen_id=3)


def test_capture_screen_rejects_non_tuple_region() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(TypeError):
            capture_screen(
                region=[0, 0, 10, 10]  # type: ignore[arg-type] - 验证运行时类型校验
            )


def test_capture_screen_rejects_region_with_wrong_length() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ValueError):
            capture_screen(
                region=(0, 0, 10)  # type: ignore[arg-type] - 验证运行时长度校验
            )


def test_capture_screen_rejects_non_integer_region_value() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(TypeError):
            capture_screen(
                region=(0, 0, 10, "10")  # type: ignore[arg-type] - 验证运行时类型校验
            )


def test_capture_screen_rejects_boolean_region_value() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(TypeError):
            capture_screen(region=(0, 0, 10, True))


def test_capture_screen_rejects_negative_region_left() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ValueError):
            capture_screen(region=(-1, 0, 10, 10))


def test_capture_screen_rejects_negative_region_top() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ValueError):
            capture_screen(region=(0, -1, 10, 10))


def test_capture_screen_rejects_non_positive_region_width() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ValueError):
            capture_screen(region=(0, 0, 0, 10))


def test_capture_screen_rejects_non_positive_region_height() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ValueError):
            capture_screen(region=(0, 0, 10, 0))


def test_capture_screen_rejects_region_past_right_boundary() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ValueError):
            capture_screen(screen_id=1, region=(190, 0, 11, 10))


def test_capture_screen_rejects_region_past_bottom_boundary() -> None:
    fake_mss = FakeMss()

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ValueError):
            capture_screen(screen_id=1, region=(0, 190, 10, 11))


def test_capture_screen_converts_mss_failure() -> None:
    fake_mss = FakeMss(grab_error=MssScreenShotError("capture failed"))

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ScreenCaptureError):
            capture_screen()


def test_capture_screen_preserves_mss_failure_as_cause() -> None:
    original_error = MssScreenShotError("capture failed")
    fake_mss = FakeMss(grab_error=original_error)

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ScreenCaptureError) as error_info:
            capture_screen()

    assert error_info.value.__cause__ is original_error


def test_capture_screen_converts_os_error_and_preserves_cause() -> None:
    original_error = OSError("capture failed")
    fake_mss = FakeMss(grab_error=original_error)

    with patch(
        "perception.screenshot._create_mss_instance",
        return_value=fake_mss,
    ):
        with pytest.raises(ScreenCaptureError) as error_info:
            capture_screen()

    assert error_info.value.__cause__ is original_error


def test_capture_screen_logs_mss_failure(caplog: pytest.LogCaptureFixture) -> None:
    fake_mss = FakeMss(grab_error=MssScreenShotError("capture failed"))

    with caplog.at_level(logging.ERROR, logger="perception.screenshot"):
        with patch(
            "perception.screenshot._create_mss_instance",
            return_value=fake_mss,
        ):
            with pytest.raises(ScreenCaptureError):
                capture_screen()

    assert "屏幕截图失败" in caplog.text


def test_capture_failure_final_log_excludes_sensitive_exception_data() -> None:
    message = (
        r"SENSITIVE_EXCEPTION_MESSAGE C:\private\model "
        "region=(10,20,30,40) user_text_marker"
    )
    original_error = MssScreenShotError(message)
    fake_mss = FakeMss(grab_error=original_error)

    with formatted_log_output("perception.screenshot") as stream:
        with patch(
            "perception.screenshot._create_mss_instance",
            return_value=fake_mss,
        ):
            with pytest.raises(ScreenCaptureError) as error_info:
                capture_screen(screen_id=0, region=(10, 20, 30, 40))

    output = stream.getvalue()
    assert error_info.value.__cause__ is original_error
    assert "屏幕截图失败" in output
    assert "ScreenShotError" in output
    assert output.count("屏幕截图失败") == 1
    assert all(marker not in output for marker in SENSITIVE_LOG_PARTS)
