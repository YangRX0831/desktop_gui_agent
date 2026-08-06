"""测试 OCR 构造、图像预处理、后端异常链和安全日志。

PaddleOCR 构造与推理由内存 fake 隔离，不初始化真实模型。
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from perception.ocr_recognizer import OCRRecognizer
from utils.exceptions import OCRModelLoadError, OCRRecognitionError

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


class FakeOCREngine:
    """记录输入并返回预设分页结果的模拟 OCR 引擎。"""

    def __init__(
        self,
        pages: list[dict[str, object]] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.pages = [] if pages is None else pages
        self.error = error
        self.images: list[np.ndarray] = []

    def predict(self, image: np.ndarray) -> list[dict[str, object]]:
        self.images.append(image)
        if self.error is not None:
            raise self.error
        return self.pages


def make_ocr_page(
    texts: object = ("中文",),
    scores: object = (0.95,),
    boxes: object = ((1, 2, 30, 40),),
) -> dict[str, object]:
    """构造模拟的单页 OCR 结果。"""
    return {
        "rec_texts": texts,
        "rec_scores": scores,
        "rec_boxes": boxes,
    }


def test_ocr_default_engine_uses_pp_ocr_v4_and_chinese() -> None:
    constructor_calls: list[dict[str, object]] = []

    def fake_constructor(**kwargs: object) -> FakeOCREngine:
        constructor_calls.append(kwargs)
        return FakeOCREngine()

    module = SimpleNamespace(PaddleOCR=fake_constructor)
    with patch("perception.ocr_recognizer.import_module", return_value=module):
        OCRRecognizer()

    assert constructor_calls[0]["ocr_version"] == "PP-OCRv4"
    assert constructor_calls[0]["lang"] == "ch"


def test_ocr_default_engine_uses_cpu() -> None:
    constructor_calls: list[dict[str, object]] = []

    def fake_constructor(**kwargs: object) -> FakeOCREngine:
        constructor_calls.append(kwargs)
        return FakeOCREngine()

    module = SimpleNamespace(PaddleOCR=fake_constructor)
    with patch("perception.ocr_recognizer.import_module", return_value=module):
        OCRRecognizer()

    assert constructor_calls[0]["device"] == "cpu"
    assert constructor_calls[0]["enable_mkldnn"] is True
    assert constructor_calls[0]["cpu_threads"] == 8
    assert constructor_calls[0]["mkldnn_cache_capacity"] == 10
    assert constructor_calls[0]["text_det_limit_side_len"] == 736
    assert constructor_calls[0]["text_det_limit_type"] == "max"


def test_ocr_default_engine_disables_document_components() -> None:
    constructor_calls: list[dict[str, object]] = []

    def fake_constructor(**kwargs: object) -> FakeOCREngine:
        constructor_calls.append(kwargs)
        return FakeOCREngine()

    module = SimpleNamespace(PaddleOCR=fake_constructor)
    with patch("perception.ocr_recognizer.import_module", return_value=module):
        OCRRecognizer()

    assert constructor_calls[0]["use_doc_orientation_classify"] is False
    assert constructor_calls[0]["use_doc_unwarping"] is False
    assert constructor_calls[0]["use_textline_orientation"] is False
    assert len(constructor_calls[0]) == 11


def test_ocr_injected_engine_does_not_import_paddleocr() -> None:
    with patch(
        "perception.ocr_recognizer.import_module",
        side_effect=AssertionError("不得导入"),
    ):
        recognizer = OCRRecognizer(FakeOCREngine())

    assert isinstance(recognizer, OCRRecognizer)


def test_ocr_import_failure_is_converted_and_preserves_cause() -> None:
    original_error = ImportError("missing")

    with patch(
        "perception.ocr_recognizer.import_module",
        side_effect=original_error,
    ):
        with pytest.raises(OCRModelLoadError) as error_info:
            OCRRecognizer()

    assert error_info.value.__cause__ is original_error


def test_ocr_constructor_failure_is_converted_and_preserves_cause() -> None:
    original_error = RuntimeError("failed")

    def failing_constructor(**kwargs: object) -> FakeOCREngine:
        raise original_error

    module = SimpleNamespace(PaddleOCR=failing_constructor)

    with patch("perception.ocr_recognizer.import_module", return_value=module):
        with pytest.raises(OCRModelLoadError) as error_info:
            OCRRecognizer()

    assert error_info.value.__cause__ is original_error


def test_ocr_converts_rgb_input_to_bgr() -> None:
    engine = FakeOCREngine()
    image = Image.new("RGB", (1, 1), (10, 20, 30))

    OCRRecognizer(engine).recognize(image)

    assert engine.images[0].tolist() == [[[30, 20, 10]]]


def test_ocr_input_array_is_uint8_three_channel_and_contiguous() -> None:
    engine = FakeOCREngine()

    OCRRecognizer(engine).recognize(Image.new("RGB", (2, 3)))

    array = engine.images[0]
    assert array.dtype == np.uint8
    assert array.shape == (3, 2, 3)
    assert array.flags.c_contiguous


def test_ocr_rejects_non_pillow_input() -> None:
    with pytest.raises(TypeError):
        OCRRecognizer(FakeOCREngine()).recognize(
            object()  # type: ignore[arg-type]  # 验证运行时类型校验
        )


@pytest.mark.parametrize(
    "size",
    [
        pytest.param((0, 1), id="zero-width"),
        pytest.param((1, 0), id="zero-height"),
    ],
)
def test_ocr_rejects_zero_sized_image(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        OCRRecognizer(FakeOCREngine()).recognize(Image.new("RGB", size))


@pytest.mark.parametrize(
    ("mode", "color"),
    [
        pytest.param("RGBA", (10, 20, 30, 40), id="rgba"),
        pytest.param("L", 20, id="grayscale"),
        pytest.param("P", 0, id="palette"),
        pytest.param("CMYK", (0, 0, 0, 0), id="cmyk"),
    ],
)
def test_ocr_converts_supported_pillow_modes(
    mode: str,
    color: int | tuple[int, ...],
) -> None:
    engine = FakeOCREngine()

    OCRRecognizer(engine).recognize(Image.new(mode, (1, 1), color))

    assert engine.images[0].shape == (1, 1, 3)


def test_ocr_predict_failure_is_converted_and_preserves_cause() -> None:
    original_error = RuntimeError("failed")

    with pytest.raises(OCRRecognitionError) as error_info:
        OCRRecognizer(FakeOCREngine(error=original_error)).recognize(
            Image.new("RGB", (1, 1))
        )

    assert error_info.value.__cause__ is original_error


def test_ocr_generator_iteration_failure_is_converted_and_preserves_cause() -> None:
    original_error = RuntimeError("late failure")

    class DelayedFailureEngine:
        def predict(self, image: np.ndarray) -> Iterator[dict[str, object]]:
            yield from ()
            raise original_error

    with pytest.raises(OCRRecognitionError) as error_info:
        OCRRecognizer(DelayedFailureEngine()).recognize(Image.new("RGB", (1, 1)))

    assert error_info.value.__cause__ is original_error


def test_ocr_failures_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="perception.ocr_recognizer"):
        with pytest.raises(OCRRecognitionError):
            OCRRecognizer(FakeOCREngine(error=RuntimeError("failed"))).recognize(
                Image.new("RGB", (1, 1))
            )

    assert "OCR 推理失败" in caplog.text


def test_ocr_model_failure_final_log_excludes_sensitive_exception_data() -> None:
    message = (
        r"SENSITIVE_EXCEPTION_MESSAGE C:\private\model "
        "region=(10,20,30,40) user_text_marker"
    )
    original_error = RuntimeError(message)

    def failing_constructor(**kwargs: object) -> FakeOCREngine:
        raise original_error

    module = SimpleNamespace(PaddleOCR=failing_constructor)

    with formatted_log_output("perception.ocr_recognizer") as stream:
        with patch("perception.ocr_recognizer.import_module", return_value=module):
            with pytest.raises(OCRModelLoadError) as error_info:
                OCRRecognizer()

    output = stream.getvalue()
    assert error_info.value.__cause__ is original_error
    assert "OCR 模型初始化失败" in output
    assert output.count("OCR 模型初始化失败") == 1
    assert all(marker not in output for marker in SENSITIVE_LOG_PARTS)


def test_ocr_predict_failure_final_log_excludes_sensitive_exception_data() -> None:
    message = (
        r"SENSITIVE_EXCEPTION_MESSAGE C:\private\model "
        "region=(10,20,30,40) user_text_marker"
    )
    original_error = RuntimeError(message)
    recognizer = OCRRecognizer(FakeOCREngine(error=original_error))

    with formatted_log_output("perception.ocr_recognizer") as stream:
        with pytest.raises(OCRRecognitionError) as error_info:
            recognizer.recognize(Image.new("RGB", (1, 1)))

    output = stream.getvalue()
    assert error_info.value.__cause__ is original_error
    assert "OCR 推理失败" in output
    assert output.count("OCR 推理失败") == 1
    assert all(marker not in output for marker in SENSITIVE_LOG_PARTS)


def test_ocr_structure_error_does_not_log_recognized_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_text = "不得记录的完整文字"
    page = make_ocr_page(texts=(sensitive_text,), scores=())

    with caplog.at_level(logging.ERROR, logger="perception.ocr_recognizer"):
        with pytest.raises(OCRRecognitionError):
            OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))

    assert sensitive_text not in caplog.text
