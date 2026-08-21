"""验证截图、OCR、UI 标注和日志的 PRD 公共行为。"""

import logging
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import perception.screenshot as screenshot_module
import perception.ui_locator as locator_module
from perception.ocr_recognizer import OCRRecognizer
from perception.screenshot import capture_screen, is_window_available
from perception.ui_locator import annotate_ui_elements
from utils.exceptions import OCRRecognitionError, ScreenCaptureError


class FakeScreenshot:
    """提供 Pillow 可转换的两像素 BGRX 截图。"""

    size = (2, 1)
    raw = bytes((0, 0, 255, 0, 0, 255, 0, 0))


class FakeMss:
    """模拟可复用的 MSS 实例。"""

    def __init__(self, error: Exception | None = None) -> None:
        """保存可选 grab 异常。"""
        self.monitors = [
            {"left": -10, "top": 0, "width": 200, "height": 100},
            {"left": 0, "top": 0, "width": 100, "height": 100},
        ]
        self.error = error
        self.areas: list[dict[str, int]] = []
        self.close_calls = 0

    def grab(self, area: dict[str, int]) -> FakeScreenshot:
        """记录截图区域并返回图像或异常。"""
        self.areas.append(area)
        if self.error is not None:
            raise self.error
        return FakeScreenshot()

    def close(self) -> None:
        """记录资源释放。"""
        self.close_calls += 1


class FakeOCR:
    """记录 BGR 输入并返回预设 OCR 页面。"""

    def __init__(self, pages: object) -> None:
        """保存结果或异常。"""
        self.pages = pages
        self.images: list[np.ndarray] = []

    def predict(self, image: np.ndarray) -> object:
        """模拟 PaddleOCR predict。"""
        self.images.append(image)
        if isinstance(self.pages, Exception):
            raise self.pages
        return self.pages


@pytest.fixture(autouse=True)
def clean_screenshot_backend() -> object:
    """每个测试前后释放进程级 fake MSS 状态。"""
    screenshot_module._cleanup_mss_instance()
    yield
    screenshot_module._cleanup_mss_instance()


def test_capture_screen_uses_monitor_and_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """截图按所选屏幕坐标转换区域并返回 RGB 图像。"""
    backend = FakeMss()
    monkeypatch.setattr(screenshot_module, "mss", SimpleNamespace(MSS=lambda: backend))

    image = capture_screen(1, (10, 20, 2, 1))

    assert image.mode == "RGB"
    assert image.size == (2, 1)
    assert backend.areas == [{"left": 10, "top": 20, "width": 2, "height": 1}]
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((1, 0)) == (0, 255, 0)


@pytest.mark.parametrize(
    ("screen_id", "region", "error"),
    [(-1, None, ValueError), (True, None, TypeError), (1, (0, 0, 0, 1), ValueError)],
)
def test_capture_validates_before_grab(
    monkeypatch: pytest.MonkeyPatch,
    screen_id: object,
    region: object,
    error: type[Exception],
) -> None:
    """非法索引和区域不会调用截图后端。"""
    backend = FakeMss()
    monkeypatch.setattr(screenshot_module, "mss", SimpleNamespace(MSS=lambda: backend))

    with pytest.raises(error):
        capture_screen(screen_id, region)  # type: ignore[arg-type]  # 故意验证运行时参数保护。

    assert backend.areas == []


def test_capture_failure_closes_backend_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """grab 失败清理共享句柄并转换为项目异常。"""
    original = OSError("simulated")
    backend = FakeMss(original)
    monkeypatch.setattr(screenshot_module, "mss", SimpleNamespace(MSS=lambda: backend))

    with pytest.raises(ScreenCaptureError) as error_info:
        capture_screen()

    assert error_info.value.__cause__ is original
    assert backend.close_calls == 1


def test_window_availability_uses_visible_native_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HWND 必须同时存在且可见才可继续作为动作目标。"""
    user32 = SimpleNamespace(
        IsWindow=lambda hwnd: hwnd in {10, 20},
        IsWindowVisible=lambda hwnd: hwnd == 10,
    )
    monkeypatch.setattr(screenshot_module, "_load_user32", lambda: user32)

    assert is_window_available(10) is True
    assert is_window_available(20) is False
    assert is_window_available(30) is False
    assert is_window_available(0) is False
    assert is_window_available(True) is False


def test_ocr_converts_rgb_to_bgr_and_parses_public_result() -> None:
    """OCR 接收 BGR uint8 数组并输出稳定字段。"""
    engine = FakeOCR(
        [
            {
                "rec_texts": ["文本"],
                "rec_scores": [0.9],
                "rec_boxes": [[1, 2, 30, 40]],
            },
        ],
    )
    image = Image.new("RGB", (1, 1), (10, 20, 30))

    results = OCRRecognizer(engine).recognize(  # type: ignore[arg-type]  # fake 协议。
        image,
    )

    assert engine.images[0].tolist() == [[[30, 20, 10]]]
    assert results == [{"text": "文本", "bbox": (1, 2, 30, 40), "confidence": 0.9}]


def test_ocr_failure_preserves_cause() -> None:
    """OCR 推理异常被转换且不吞掉根因。"""
    original = RuntimeError("simulated")
    recognizer = OCRRecognizer(  # type: ignore[arg-type]  # fake OCR 协议。
        FakeOCR(original),
    )

    with pytest.raises(OCRRecognitionError) as error_info:
        recognizer.recognize(Image.new("RGB", (1, 1)))

    assert error_info.value.__cause__ is original


def test_ui_annotation_returns_copy_and_draws_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非空标注不修改原图，并在结果图绘制边框。"""
    image = Image.new("RGB", (40, 30), "white")
    monkeypatch.setattr(
        locator_module,
        "_load_font",
        lambda: locator_module.ImageFont.load_default(),
    )

    result = annotate_ui_elements(
        image,
        [{"text": "A", "bbox": (5, 10, 20, 20), "element_type": "button"}],
    )

    assert result is not image
    assert image.getpixel((5, 10)) == (255, 255, 255)
    assert result.getpixel((5, 10)) != (255, 255, 255)


def test_ui_annotation_validates_elements_before_drawing() -> None:
    """非法元素字段和越界坐标在绘图副作用前失败。"""
    image = Image.new("RGB", (20, 20))
    with pytest.raises(ValueError):
        annotate_ui_elements(
            image,
            [{"text": "A", "bbox": (0, 0, 20, 10), "element_type": "button"}],
        )


def test_empty_ui_elements_return_original_image() -> None:
    """空标注无需复制或加载字体。"""
    image = Image.new("RGB", (20, 20))
    assert annotate_ui_elements(image, []) is image


# ---------------------------------------------------------------------------
# PRD 4.5.2: UI 元素信息缓存
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def clean_annotation_cache():
    """每个 cache 测试前后清空 element-info 缓存。"""
    locator_module._clear_element_cache()
    yield
    locator_module._clear_element_cache()


def _sample_elements() -> list:
    """提供一组可复用的 UI 元素。"""
    return [{"text": "OK", "bbox": (5, 5, 50, 30), "element_type": "button"}]


def _mock_font(monkeypatch) -> None:
    """替换字体加载为默认字体。"""
    monkeypatch.setattr(
        locator_module,
        "_load_font",
        lambda: locator_module.ImageFont.load_default(),
    )


def test_cache_stale_image_regression(
    clean_annotation_cache,
    monkeypatch,
) -> None:
    """相同 elements + 相同尺寸 + 不同像素 → 第二次结果基于第二张图。"""
    _mock_font(monkeypatch)
    elements = _sample_elements()
    image_a = Image.new("RGB", (100, 80), "white")
    image_b = Image.new("RGB", (100, 80), "black")

    result_a = annotate_ui_elements(image_a, elements)
    result_b = annotate_ui_elements(image_b, elements)

    # result_a 基于白底,result_b 基于黑底;不返回旧帧缓存像素。
    # 标注区域外的背景像素应反映各自底图。
    assert result_a.getpixel((99, 79)) == (255, 255, 255)
    assert result_b.getpixel((99, 79)) == (0, 0, 0)


def test_cache_element_info_hit(clean_annotation_cache, monkeypatch) -> None:
    """相同 elements 重复调用 → element-info 缓存命中。"""
    _mock_font(monkeypatch)
    image = Image.new("RGB", (100, 80), "white")
    annotate_ui_elements(image, _sample_elements())
    assert len(locator_module._ELEMENT_CACHE) == 1
    # 第二次相同输入 → 命中(缓存大小不变)。
    annotate_ui_elements(image, _sample_elements())
    assert len(locator_module._ELEMENT_CACHE) == 1


def test_cache_changed_elements_miss(
    clean_annotation_cache,
    monkeypatch,
) -> None:
    """不同元素 → 缓存 miss,产生新条目。"""
    _mock_font(monkeypatch)
    image = Image.new("RGB", (100, 80), "white")
    annotate_ui_elements(image, _sample_elements())
    changed = [{"text": "X", "bbox": (5, 5, 50, 30), "element_type": "text"}]
    annotate_ui_elements(image, changed)
    assert len(locator_module._ELEMENT_CACHE) == 2


def test_cache_clear_invalidates(clean_annotation_cache, monkeypatch) -> None:
    """清空缓存后条目归零。"""
    _mock_font(monkeypatch)
    image = Image.new("RGB", (100, 80), "white")
    annotate_ui_elements(image, _sample_elements())
    assert len(locator_module._ELEMENT_CACHE) > 0
    locator_module._clear_element_cache()
    assert len(locator_module._ELEMENT_CACHE) == 0


def test_cache_mutation_does_not_pollute(
    clean_annotation_cache,
    monkeypatch,
) -> None:
    """调用方修改返回图像不污染 element-info 缓存。"""
    _mock_font(monkeypatch)
    image = Image.new("RGB", (100, 80), "white")
    result1 = annotate_ui_elements(image, _sample_elements())
    original_bg = result1.getpixel((99, 79))

    result1.putpixel((99, 79), (123, 123, 123))

    result2 = annotate_ui_elements(image, _sample_elements())
    assert result2.getpixel((99, 79)) == original_bg


def test_cache_empty_elements_not_cached(clean_annotation_cache) -> None:
    """空元素序列不进入缓存,直接返回原图。"""
    image = Image.new("RGB", (20, 20))
    result = annotate_ui_elements(image, [])
    assert result is image
    assert len(locator_module._ELEMENT_CACHE) == 0


def test_cache_bounded_clears_when_full(
    clean_annotation_cache,
    monkeypatch,
) -> None:
    """缓存达到上限后清空,防止无限增长。"""
    _mock_font(monkeypatch)
    limit = locator_module._ELEMENT_CACHE_LIMIT
    image = Image.new("RGB", (100, 80), "white")
    for i in range(limit):
        annotate_ui_elements(
            image,
            [{"text": str(i), "bbox": (i, i, i + 10, i + 10), "element_type": "text"}],
        )
    assert len(locator_module._ELEMENT_CACHE) == limit
    annotate_ui_elements(
        image,
        [{"text": "overflow", "bbox": (1, 1, 20, 20), "element_type": "text"}],
    )
    assert len(locator_module._ELEMENT_CACHE) == 1


# ---------------------------------------------------------------------------
# PRD 4.4.3: 日志记录功能
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_log_dir():
    """创建隔离的临时日志目录,测试后删除。"""
    d = Path(tempfile.mkdtemp(prefix="dga_test_logs_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_logging_records_all_levels(isolated_log_dir) -> None:
    """DEBUG/INFO/WARNING/ERROR 均可通过配置记录。"""
    from utils.logger import setup_logging

    logger = setup_logging(isolated_log_dir, "DEBUG")
    logger.debug("debug_msg")
    logger.info("info_msg")
    logger.warning("warning_msg")
    logger.error("error_msg")

    main_log = isolated_log_dir / "desktop_gui_agent.log"
    assert main_log.exists()
    content = main_log.read_text(encoding="utf-8")
    assert "debug_msg" in content
    assert "info_msg" in content
    assert "warning_msg" in content
    assert "error_msg" in content


def test_logging_error_separate_file(isolated_log_dir) -> None:
    """ERROR 进入 error-only 日志,INFO 不进入。"""
    from utils.logger import setup_logging

    logger = setup_logging(isolated_log_dir, "INFO")
    logger.info("normal_info")
    logger.error("real_error")

    error_log = isolated_log_dir / "desktop_gui_agent.error.log"
    assert error_log.exists()
    error_content = error_log.read_text(encoding="utf-8")
    assert "real_error" in error_content
    assert "normal_info" not in error_content


def test_logging_format_contains_required_fields(isolated_log_dir) -> None:
    """格式包含 timestamp/level/module/message。"""
    from utils.logger import setup_logging

    logger = setup_logging(isolated_log_dir, "DEBUG")
    logger.warning("format_test")

    main_log = isolated_log_dir / "desktop_gui_agent.log"
    line = main_log.read_text(encoding="utf-8").strip()
    # 格式: asctime | levelname | module | message
    assert "|" in line
    assert "WARNING" in line
    assert "format_test" in line
    # 时间戳格式 YYYY-MM-DD HH:MM:SS
    parts = line.split("|")
    assert len(parts[0].strip()) >= 10  # 至少有日期


def test_logging_repeated_setup_no_duplicate_handlers(isolated_log_dir) -> None:
    """重复 setup_logging 不累积 handler。"""
    from utils.logger import _HANDLER_MARKER, setup_logging

    target = logging.getLogger("test_repeat")
    setup_logging(isolated_log_dir, "INFO", logger=target)
    count1 = sum(1 for h in target.handlers if getattr(h, _HANDLER_MARKER, False))
    setup_logging(isolated_log_dir, "INFO", logger=target)
    count2 = sum(1 for h in target.handlers if getattr(h, _HANDLER_MARKER, False))
    assert count1 == count2  # 不增加
    assert count1 == 3  # console + main + error


def test_logging_no_secret_in_output(isolated_log_dir) -> None:
    """安全日志不泄露异常正文。"""
    from utils.logger import log_safe_exception, setup_logging

    logger = setup_logging(isolated_log_dir, "DEBUG")
    exc = ValueError("SENSITIVE_DATA")
    try:
        raise exc
    except ValueError as e:
        log_safe_exception(logger, "safe_event", e)

    main_log = isolated_log_dir / "desktop_gui_agent.log"
    content = main_log.read_text(encoding="utf-8")
    assert "SENSITIVE_DATA" not in content
    assert "ValueError" in content  # 异常类型名可记录


# ---------------------------------------------------------------------------
# OCR polygon/multi-page/structure deterministic tests
# ---------------------------------------------------------------------------


def test_ocr_polygon_parsing() -> None:
    """rec_polys 多点坐标正确转换为外接矩形。"""
    engine = FakeOCR(
        [
            {
                "rec_texts": ["test"],
                "rec_scores": [0.8],
                "rec_polys": [[[0, 0], [100, 0], [100, 50], [0, 50]]],
            },
        ],
    )
    image = Image.new("RGB", (1, 1))
    results = OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]
    assert results == [
        {"text": "test", "bbox": (0, 0, 100, 50), "confidence": 0.8},
    ]


def test_ocr_multi_page_results() -> None:
    """多页 OCR 结果按顺序合并。"""
    engine = FakeOCR(
        [
            {
                "rec_texts": ["a", "b"],
                "rec_scores": [0.9, 0.8],
                "rec_boxes": [[1, 2, 30, 40], [5, 6, 35, 45]],
            },
            {
                "rec_texts": ["c"],
                "rec_scores": [0.7],
                "rec_boxes": [[10, 20, 50, 60]],
            },
        ],
    )
    image = Image.new("RGB", (1, 1))
    results = OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]
    assert len(results) == 3
    assert results[2]["text"] == "c"


def test_ocr_box_invalid_order() -> None:
    """rec_boxes 左>=右或上>=下抛出结构错误。"""
    engine = FakeOCR(
        [{"rec_texts": ["x"], "rec_scores": [0.5], "rec_boxes": [[50, 50, 10, 10]]}],
    )
    image = Image.new("RGB", (1, 1))
    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]


def test_ocr_empty_result_valid() -> None:
    """空识别结果是合法的,返回空列表。"""
    engine = FakeOCR([{"rec_texts": [], "rec_scores": []}])
    image = Image.new("RGB", (1, 1))
    results = OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]
    assert results == []


def test_ocr_numpy_array_input() -> None:
    """OCR 引擎接受 NumPy 数组形式的 rec_boxes。"""
    engine = FakeOCR(
        [
            {
                "rec_texts": ["np"],
                "rec_scores": [0.9],
                "rec_boxes": np.array([[1, 2, 30, 40]]),
            },
        ],
    )
    image = Image.new("RGB", (1, 1))
    results = OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]
    assert results[0]["bbox"] == (1, 2, 30, 40)


def test_ocr_validate_image_input() -> None:
    """非 PIL.Image 输入抛出 TypeError。"""
    engine = FakeOCR([])
    recognizer = OCRRecognizer(engine)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        recognizer.recognize("not_an_image")  # type: ignore[arg-type]


def test_screenshot_region_exceeds_bounds(monkeypatch) -> None:
    """region 超出屏幕边界时抛出 ValueError。"""
    backend = FakeMss()
    monkeypatch.setattr(screenshot_module, "mss", SimpleNamespace(MSS=lambda: backend))
    with pytest.raises(ValueError, match="超出.*边界"):
        capture_screen(1, (90, 0, 20, 10))


def test_screenshot_region_negative_origin(monkeypatch) -> None:
    """region left/top 为负数时抛出 ValueError。"""
    backend = FakeMss()
    monkeypatch.setattr(screenshot_module, "mss", SimpleNamespace(MSS=lambda: backend))
    with pytest.raises(ValueError, match="大于等于 0"):
        capture_screen(1, (-1, 0, 10, 10))


def test_screenshot_invalid_screen_id_type(monkeypatch) -> None:
    """screen_id 为 bool 时抛出 TypeError。"""
    backend = FakeMss()
    monkeypatch.setattr(screenshot_module, "mss", SimpleNamespace(MSS=lambda: backend))
    with pytest.raises(TypeError):
        capture_screen(True)  # type: ignore[arg-type]


def test_screenshot_region_non_integer(monkeypatch) -> None:
    """region 包含非 int 时抛出 TypeError。"""
    backend = FakeMss()
    monkeypatch.setattr(screenshot_module, "mss", SimpleNamespace(MSS=lambda: backend))
    with pytest.raises(TypeError):
        capture_screen(1, (0, 0, 10.5, 10))  # type: ignore[arg-type]


def test_ocr_polygon_numpy_input() -> None:
    """rec_polys NumPy 数组输入正确解析为外接矩形。"""
    engine = FakeOCR(
        [
            {
                "rec_texts": ["poly"],
                "rec_scores": [0.9],
                "rec_polys": np.array([[[5, 5], [50, 5], [50, 40], [5, 40]]]),
            },
        ],
    )
    image = Image.new("RGB", (1, 1))
    results = OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]
    assert results[0]["bbox"] == (5, 5, 50, 40)


def test_ocr_confidence_non_finite_rejected() -> None:
    """非有限置信度抛出结构错误。"""
    engine = FakeOCR(
        [
            {
                "rec_texts": ["x"],
                "rec_scores": [float("inf")],
                "rec_boxes": [[1, 2, 3, 4]],
            }
        ],
    )
    image = Image.new("RGB", (1, 1))
    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]


def test_ocr_text_non_string_rejected() -> None:
    """rec_texts 非字符串项抛出结构错误。"""
    engine = FakeOCR(
        [{"rec_texts": [123], "rec_scores": [0.5], "rec_boxes": [[1, 2, 3, 4]]}],
    )
    image = Image.new("RGB", (1, 1))
    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]


def test_ocr_score_text_length_mismatch() -> None:
    """rec_texts 与 rec_scores 长度不一致抛出结构错误。"""
    engine = FakeOCR(
        [{"rec_texts": ["a", "b"], "rec_scores": [0.5], "rec_boxes": [[1, 2, 3, 4]]}],
    )
    image = Image.new("RGB", (1, 1))
    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(engine).recognize(image)  # type: ignore[arg-type]


# ======================================================================
# E2-A characterization:audio_state COM 生命周期(Release/CoUninitialize)
# ======================================================================


class _FakeComMachine:
    """脚本化 ole32/vtable/WINFUNCTYPE 全链;记录 Release 与卸载配对。"""

    def __init__(
        self,
        *,
        init_hr=0,
        create_hr=0,
        default_hr=0,
        activate_hr=0,
        level_hr=0,
        level=0.55,
        level_raises=False,
    ):
        self.init_calls: list[int] = []
        self.uninit_calls = 0
        self.released: list[int] = []
        self.script = {
            "init_hr": init_hr,
            "create_hr": create_hr,
            "default_hr": default_hr,
            "activate_hr": activate_hr,
            "level_hr": level_hr,
            "level": level,
            "level_raises": level_raises,
        }

    def CoInitializeEx(self, reserved, mode):
        self.init_calls.append(mode)
        return self.script["init_hr"]

    def CoUninitialize(self) -> None:
        self.uninit_calls += 1

    def CoCreateInstance(self, clsid, outer, context, iid, out):
        if self.script["create_hr"] == 0:
            out._obj.value = 0x1000
        return self.script["create_hr"]


def _fake_vtable(pointer):
    if pointer == 0x1000:
        return [0, 0, 0x1002, 0, 0x1004]
    if pointer == 0x2000:
        return [0, 0, 0x2002, 0x2003]
    return [0, 0, 0x3002] + [0] * 6 + [0x3009]


def _install_fake_com(monkeypatch, machine):
    from perception import audio_state

    def fake_winfunctype(*types):
        def binder(address):
            def call(*args):
                script = machine.script
                if address in (0x1002, 0x2002, 0x3002):
                    machine.released.append(address)
                    return 0
                if address == 0x1004:
                    if script["default_hr"] == 0:
                        args[3]._obj.value = 0x2000
                    return script["default_hr"]
                if address == 0x2003:
                    if script["activate_hr"] == 0:
                        args[4]._obj.value = 0x3000
                    return script["activate_hr"]
                if address == 0x3009:
                    if script["level_raises"]:
                        raise RuntimeError("level boom")
                    if script["level_hr"] == 0:
                        args[1]._obj.value = script["level"]
                    return script["level_hr"]
                raise AssertionError(f"未脚本化的 COM 地址: {address}")

            return call

        return binder

    monkeypatch.setattr(audio_state, "ole32", machine, raising=True)
    monkeypatch.setattr(audio_state, "_vtable", _fake_vtable, raising=True)
    monkeypatch.setattr(audio_state, "WINFUNCTYPE", fake_winfunctype, raising=True)


def test_com_volume_success_releases_and_uninitializes(monkeypatch) -> None:
    """成功路径:三接口逆序 Release 一次,CoUninitialize 恰好配对一次。"""
    from perception import audio_state

    machine = _FakeComMachine()
    _install_fake_com(monkeypatch, machine)
    assert audio_state.get_master_volume_percent() == 55
    assert machine.released == [0x3002, 0x2002, 0x1002]
    assert machine.uninit_calls == 1
    assert machine.init_calls == [4]


def test_com_acquire_failure_still_uninitializes(monkeypatch) -> None:
    """枚举器获取失败:无接口可释放,但 CoUninitialize 仍配对。"""
    from perception import audio_state

    machine = _FakeComMachine(create_hr=1)
    _install_fake_com(monkeypatch, machine)
    assert audio_state.get_master_volume_percent() is None
    assert machine.released == []
    assert machine.uninit_calls == 1


def test_com_default_failure_releases_enumerator_only(monkeypatch) -> None:
    """链中失败:只释放已获取接口(枚举器),再卸载。"""
    from perception import audio_state

    machine = _FakeComMachine(default_hr=1)
    _install_fake_com(monkeypatch, machine)
    assert audio_state.get_master_volume_percent() is None
    assert machine.released == [0x1002]
    assert machine.uninit_calls == 1


def test_com_level_exception_releases_all(monkeypatch) -> None:
    """取值异常路径:三接口全部释放,不外抛,卸载配对。"""
    from perception import audio_state

    machine = _FakeComMachine(level_raises=True)
    _install_fake_com(monkeypatch, machine)
    assert audio_state.get_master_volume_percent() is None
    assert machine.released == [0x3002, 0x2002, 0x1002]
    assert machine.uninit_calls == 1


def test_com_init_failure_skips_uninitialize(monkeypatch) -> None:
    """CoInitializeEx 失败 HRESULT:COM 未初始化,不得调用卸载。"""
    from perception import audio_state

    machine = _FakeComMachine(init_hr=-2147417850)
    _install_fake_com(monkeypatch, machine)
    assert audio_state.get_master_volume_percent() is None
    assert machine.uninit_calls == 0
    assert machine.released == []
