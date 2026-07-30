"""测试 UI 元素输入校验、空输入与公共结构契约。

该文件不验证具体像素绘制结果。
"""

from collections.abc import Iterator

import pytest
from PIL import Image
from PIL import ImageFont

from perception import ui_locator
from perception.ui_locator import UIElement
from perception.ui_locator import annotate_ui_elements

BACKGROUND = (240, 240, 240)


@pytest.fixture
def memory_font() -> ImageFont.ImageFont:
    """在替换字体加载器前创建测试字体。"""
    return ImageFont.load_default()


@pytest.fixture(autouse=True)
def use_memory_font(
    monkeypatch: pytest.MonkeyPatch,
    memory_font: ImageFont.ImageFont,
) -> None:
    """使用内存字体隔离大部分测试的系统字体依赖。"""
    monkeypatch.setattr(
        ui_locator.ImageFont,
        "truetype",
        lambda _path, _size: memory_font,
    )


def make_image(
    mode: str = "RGB",
    size: tuple[int, int] = (160, 100),
) -> Image.Image:
    """创建内存测试图像。"""
    color: int | tuple[int, int, int] = 240 if mode == "L" else BACKGROUND
    return Image.new(mode, size, color)


def make_element(
    *,
    text: str = "设置",
    bbox: tuple[int, int, int, int] = (20, 30, 80, 70),
    element_type: str = "button",
) -> UIElement:
    """创建规范 UI 元素。"""
    return {
        "text": text,
        "bbox": bbox,
        "element_type": element_type,
    }  # type: ignore[return-value] - 测试需要构造无效类型值


def test_empty_elements_returns_same_image() -> None:
    image = make_image()

    result = annotate_ui_elements(image, [])

    assert result is image


def test_empty_elements_does_not_convert_non_rgb_image() -> None:
    image = make_image("L")

    result = annotate_ui_elements(image, [])

    assert result is image
    assert result.mode == "L"


@pytest.mark.parametrize("image", [None, object()])
def test_invalid_image_type_raises_type_error(image: object) -> None:
    with pytest.raises(TypeError):
        annotate_ui_elements(
            image,  # type: ignore[arg-type] - 验证运行时类型校验
            [],
        )


@pytest.mark.parametrize("size", [(0, 10), (10, 0)])
def test_zero_image_dimension_raises_value_error(
    size: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        annotate_ui_elements(make_image(size=size), [])


@pytest.mark.parametrize("elements", ["items", b"items", bytearray(b"items")])
def test_string_like_elements_raise_type_error(elements: object) -> None:
    with pytest.raises(TypeError):
        annotate_ui_elements(
            make_image(),
            elements,  # type: ignore[arg-type] - 验证运行时类型校验
        )


def test_generator_elements_raise_type_error() -> None:
    elements: Iterator[UIElement] = iter([make_element()])

    with pytest.raises(TypeError):
        annotate_ui_elements(
            make_image(),
            elements,  # type: ignore[arg-type] - 验证运行时类型校验
        )


def test_non_dict_element_raises_type_error() -> None:
    with pytest.raises(TypeError):
        annotate_ui_elements(
            make_image(),
            [object()],  # type: ignore[list-item] - 验证运行时类型校验
        )


def test_missing_element_field_raises_value_error() -> None:
    element = {"text": "A", "bbox": (10, 10, 20, 20)}

    with pytest.raises(ValueError):
        annotate_ui_elements(
            make_image(),
            [element],  # type: ignore[list-item] - 验证运行时字段校验
        )


def test_unknown_element_field_raises_value_error() -> None:
    element = {**make_element(), "confidence": 0.9}

    with pytest.raises(ValueError):
        annotate_ui_elements(
            make_image(),
            [element],  # type: ignore[list-item] - 验证运行时字段校验
        )


def test_non_string_text_raises_type_error() -> None:
    element = {**make_element(), "text": 1}

    with pytest.raises(TypeError):
        annotate_ui_elements(
            make_image(),
            [element],  # type: ignore[list-item] - 验证运行时字段校验
        )


def test_non_string_element_type_raises_type_error() -> None:
    element = {**make_element(), "element_type": 1}

    with pytest.raises(TypeError):
        annotate_ui_elements(
            make_image(),
            [element],  # type: ignore[list-item] - 验证运行时字段校验
        )


def test_unknown_element_type_raises_value_error() -> None:
    with pytest.raises(ValueError):
        annotate_ui_elements(
            make_image(),
            [make_element(element_type="unknown")],
        )


def test_non_tuple_bbox_raises_type_error() -> None:
    element = {**make_element(), "bbox": [10, 10, 20, 20]}

    with pytest.raises(TypeError):
        annotate_ui_elements(
            make_image(),
            [element],  # type: ignore[list-item] - 验证运行时字段校验
        )


def test_bbox_wrong_length_raises_value_error() -> None:
    element = {**make_element(), "bbox": (10, 10, 20)}

    with pytest.raises(ValueError):
        annotate_ui_elements(
            make_image(),
            [element],  # type: ignore[list-item] - 验证运行时字段校验
        )


@pytest.mark.parametrize("coordinate", [1.5, True])
def test_invalid_bbox_coordinate_type_raises_type_error(
    coordinate: object,
) -> None:
    element = {**make_element(), "bbox": (coordinate, 10, 20, 20)}

    with pytest.raises(TypeError):
        annotate_ui_elements(
            make_image(),
            [element],  # type: ignore[list-item] - 验证运行时字段校验
        )


@pytest.mark.parametrize(
    "bbox",
    [
        (30, 10, 20, 20),
        (10, 10, 10, 20),
        (10, 10, 20, 10),
    ],
)
def test_invalid_bbox_area_raises_value_error(
    bbox: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        annotate_ui_elements(make_image(), [make_element(bbox=bbox)])


@pytest.mark.parametrize(
    "bbox",
    [
        (-1, 10, 20, 20),
        (10, -1, 20, 20),
        (10, 10, 160, 20),
        (10, 10, 20, 100),
    ],
)
def test_out_of_bounds_bbox_raises_value_error(
    bbox: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        annotate_ui_elements(make_image(), [make_element(bbox=bbox)])


def test_empty_elements_do_not_load_font(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_path: str, _size: int) -> ImageFont.FreeTypeFont:
        raise AssertionError("字体加载器不应被调用")

    monkeypatch.setattr(ui_locator.ImageFont, "truetype", fail_if_called)
    image = make_image()

    result = annotate_ui_elements(image, [])

    assert result is image
