"""测试 PaddleOCR 原始结果的兼容解析与输出标准化。

仅使用内存数据结构，区分合法空页与损坏结果。
"""

import numpy as np
import pytest
from PIL import Image

from perception.ocr_recognizer import OCRRecognizer
from utils.exceptions import OCRRecognitionError


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


def test_ocr_parses_single_result() -> None:
    result = OCRRecognizer(FakeOCREngine([make_ocr_page()])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result == [
        {"text": "中文", "bbox": (1, 2, 30, 40), "confidence": 0.95}
    ]


def test_ocr_preserves_multiple_bilingual_results_in_order() -> None:
    page = make_ocr_page(
        texts=("中文", "English"),
        scores=(0.9, 0.8),
        boxes=((1, 2, 3, 4), (5, 6, 7, 8)),
    )

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert [item["text"] for item in result] == ["中文", "English"]


def test_ocr_returns_python_integer_box_tuple() -> None:
    result = OCRRecognizer(FakeOCREngine([make_ocr_page()])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert isinstance(result[0]["bbox"], tuple)
    assert all(type(value) is int for value in result[0]["bbox"])


def test_ocr_accepts_numpy_integer_boxes() -> None:
    page = make_ocr_page(boxes=np.array([[1, 2, 3, 4]], dtype=np.int64))

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result[0]["bbox"] == (1, 2, 3, 4)


def test_ocr_accepts_huge_python_integer_coordinates() -> None:
    huge_coordinate = 10**1000
    page = make_ocr_page(
        texts=("大坐标",),
        scores=(0.9,),
        boxes=((huge_coordinate, 0, huge_coordinate + 1, 1),),
    )

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result[0]["bbox"] == (huge_coordinate, 0, huge_coordinate + 1, 1)
    assert all(type(value) is int for value in result[0]["bbox"])


def test_ocr_falls_back_to_polygons() -> None:
    page = make_ocr_page()
    page["rec_boxes"] = None
    page["rec_polys"] = (((1, 4), (5, 2), (3, 8)),)

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result[0]["bbox"] == (1, 2, 5, 8)


def test_ocr_does_not_fall_back_when_boxes_are_invalid() -> None:
    page = make_ocr_page(boxes="invalid")
    page["rec_polys"] = (((1, 2), (3, 4)),)

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_converts_real_confidence_to_float() -> None:
    page = make_ocr_page(scores=(np.float32(0.75),))

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result[0]["confidence"] == pytest.approx(0.75)
    assert type(result[0]["confidence"]) is float


def test_ocr_returns_empty_list_for_empty_prediction() -> None:
    result = OCRRecognizer(FakeOCREngine()).recognize(Image.new("RGB", (1, 1)))

    assert result == []


def test_ocr_returns_empty_list_for_page_without_text() -> None:
    page = make_ocr_page(texts=(), scores=(), boxes=())

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result == []


def test_ocr_returns_empty_list_when_both_position_fields_are_empty() -> None:
    page = make_ocr_page(texts=(), scores=(), boxes=())
    page["rec_polys"] = ()

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result == []


def test_ocr_rejects_empty_text_with_nonempty_secondary_positions() -> None:
    page = make_ocr_page(texts=(), scores=(), boxes=())
    page["rec_polys"] = (((1, 2), (3, 4)),)

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_returns_empty_list_for_flat_empty_numpy_boxes() -> None:
    page = make_ocr_page(
        texts=(),
        scores=(),
        boxes=np.empty((0,), dtype=np.int64),
    )

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result == []


def test_ocr_returns_empty_list_for_two_dimensional_empty_numpy_boxes() -> None:
    page = make_ocr_page(
        texts=(),
        scores=(),
        boxes=np.empty((0, 4), dtype=np.int64),
    )

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result == []


def test_ocr_returns_empty_list_for_flat_empty_numpy_polygons() -> None:
    page = make_ocr_page(texts=(), scores=(), boxes=None)
    page["rec_polys"] = np.empty((0,), dtype=np.int64)

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result == []


def test_ocr_rejects_empty_text_with_nonempty_numpy_boxes() -> None:
    page = make_ocr_page(
        texts=(),
        scores=(),
        boxes=np.array([[1, 2, 3, 4]], dtype=np.int64),
    )

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_rejects_empty_text_with_scalar_numpy_boxes() -> None:
    page = make_ocr_page(
        texts=(),
        scores=(),
        boxes=np.array(1),
    )

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_returns_empty_list_without_position_fields() -> None:
    page = {"rec_texts": (), "rec_scores": ()}

    result = OCRRecognizer(FakeOCREngine([page])).recognize(
        Image.new("RGB", (1, 1))
    )

    assert result == []


def test_ocr_rejects_empty_text_with_nonempty_positions() -> None:
    page = make_ocr_page(
        texts=(),
        scores=(),
        boxes=((1, 2, 3, 4),),
    )

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_flattens_multiple_pages() -> None:
    first_page = make_ocr_page(texts=("一",), boxes=((1, 1, 2, 2),))
    second_page = make_ocr_page(texts=("二",), boxes=((3, 3, 4, 4),))

    result = OCRRecognizer(
        FakeOCREngine([first_page, second_page])
    ).recognize(Image.new("RGB", (1, 1)))

    assert [item["text"] for item in result] == ["一", "二"]


def test_ocr_rejects_text_and_score_length_mismatch() -> None:
    page = make_ocr_page(texts=("一", "二"), scores=(0.9,))

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_rejects_text_without_position() -> None:
    page = {"rec_texts": ("一",), "rec_scores": (0.9,)}

    with pytest.raises(
        OCRRecognitionError,
        match="^OCR 结果项缺少文字位置字段$",
    ):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_rejects_non_mapping_page() -> None:
    engine = FakeOCREngine()
    engine.pages = [object()]  # type: ignore[list-item] - 验证运行时结构校验

    with pytest.raises(OCRRecognitionError, match="^OCR 结果项必须是映射$"):
        OCRRecognizer(engine).recognize(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        pytest.param(
            "rec_texts",
            "OCR 结果项缺少 rec_texts",
            id="missing-rec-texts",
        ),
        pytest.param(
            "rec_scores",
            "OCR 结果项缺少 rec_scores",
            id="missing-rec-scores",
        ),
    ],
)
def test_ocr_rejects_missing_required_field(
    field_name: str,
    message: str,
) -> None:
    page = make_ocr_page()
    del page[field_name]

    with pytest.raises(OCRRecognitionError, match=f"^{message}$"):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_rejects_non_string_text() -> None:
    page = make_ocr_page(texts=(123,))

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    "confidence",
    [
        pytest.param(True, id="boolean"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_ocr_rejects_invalid_confidence(confidence: object) -> None:
    page = make_ocr_page(scores=(confidence,))

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_rejects_box_with_wrong_length() -> None:
    page = make_ocr_page(boxes=((1, 2, 3),))

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    "coordinate",
    [
        pytest.param(True, id="boolean"),
        pytest.param(1.0, id="float"),
        pytest.param(float("inf"), id="non-finite"),
    ],
)
def test_ocr_rejects_invalid_box_coordinate(coordinate: object) -> None:
    page = make_ocr_page(boxes=((coordinate, 2, 3, 4),))

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    "box",
    [
        pytest.param((4, 2, 3, 5), id="reversed-horizontal"),
        pytest.param((1, 5, 3, 4), id="reversed-vertical"),
    ],
)
def test_ocr_rejects_reversed_box(box: tuple[int, int, int, int]) -> None:
    page = make_ocr_page(boxes=(box,))

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    "box",
    [
        pytest.param((1, 2, 1, 5), id="zero-width"),
        pytest.param((1, 2, 4, 2), id="zero-height"),
    ],
)
def test_ocr_rejects_zero_area_box(box: tuple[int, int, int, int]) -> None:
    page = make_ocr_page(boxes=(box,))

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    "polygon",
    [
        pytest.param((), id="empty"),
        pytest.param(((1, 2, 3),), id="invalid-point"),
    ],
)
def test_ocr_rejects_invalid_polygon(polygon: object) -> None:
    page = make_ocr_page(boxes=None)
    page["rec_polys"] = (polygon,)

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    "polygon",
    [
        pytest.param(((1, 2), (1, 5)), id="zero-width"),
        pytest.param(((1, 2), (4, 2)), id="zero-height"),
    ],
)
def test_ocr_rejects_zero_area_polygon(polygon: object) -> None:
    page = make_ocr_page(boxes=None)
    page["rec_polys"] = (polygon,)

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_rejects_float_polygon_coordinate() -> None:
    page = make_ocr_page(boxes=None)
    page["rec_polys"] = (((1.0, 2), (3, 4)),)

    with pytest.raises(OCRRecognitionError):
        OCRRecognizer(FakeOCREngine([page])).recognize(Image.new("RGB", (1, 1)))


def test_ocr_empty_result_does_not_raise() -> None:
    OCRRecognizer(FakeOCREngine()).recognize(Image.new("RGB", (1, 1)))
