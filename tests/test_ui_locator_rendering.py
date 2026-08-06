"""测试 UI 边框、标签、字体与绘制结果。

测试只使用内存图像，与输入校验职责分离。
"""

from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from perception import ui_locator
from perception.ui_locator import UIElement, annotate_ui_elements

BACKGROUND = (240, 240, 240)

COLORS = {
    "text": (0, 102, 204),
    "button": (0, 153, 76),
    "input": (230, 126, 34),
    "icon": (128, 0, 128),
    "other": (204, 0, 0),
}


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
    }  # type: ignore[return-value]  # 测试需要构造无效类型值


def capture_text(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """记录 ImageDraw 收到的标签文本。"""
    labels: list[str] = []
    original_text = ImageDraw.ImageDraw.text

    def record_text(
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        labels.append(text)
        original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    return labels


def capture_rectangles(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int, int, int]]:
    """记录 ImageDraw 收到的矩形坐标。"""
    rectangles: list[tuple[int, int, int, int]] = []
    original_rectangle = ImageDraw.ImageDraw.rectangle

    def record_rectangle(
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int, int, int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        rectangles.append(tuple(xy))
        original_rectangle(draw, xy, *args, **kwargs)

    monkeypatch.setattr(
        ImageDraw.ImageDraw,
        "rectangle",
        record_rectangle,
    )
    return rectangles


def test_single_element_returns_new_pillow_image() -> None:
    image = make_image()

    result = annotate_ui_elements(image, [make_element()])

    assert isinstance(result, Image.Image)
    assert result is not image


def test_nonempty_rgb_input_remains_unchanged() -> None:
    image = make_image()
    original_bytes = image.tobytes()

    annotate_ui_elements(image, [make_element()])

    assert image.tobytes() == original_bytes


def test_non_rgb_input_returns_rgb_image() -> None:
    result = annotate_ui_elements(make_image("L"), [make_element()])

    assert result.mode == "RGB"


def test_single_element_draws_all_four_edges() -> None:
    result = annotate_ui_elements(make_image(), [make_element()])

    assert result.getpixel((20, 50)) == COLORS["button"]
    assert result.getpixel((80, 50)) == COLORS["button"]
    assert result.getpixel((50, 30)) == COLORS["button"]
    assert result.getpixel((50, 70)) == COLORS["button"]


def test_label_background_uses_element_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangles = capture_rectangles(monkeypatch)

    result = annotate_ui_elements(make_image(), [make_element()])

    label_left, label_top, _, _ = rectangles[1]
    assert result.getpixel((label_left, label_top)) == COLORS["button"]


def test_unrelated_region_keeps_original_background() -> None:
    result = annotate_ui_elements(make_image(), [make_element()])

    assert result.getpixel((150, 90)) == BACKGROUND


def test_multiple_elements_draws_every_element() -> None:
    elements = [
        make_element(bbox=(10, 40, 40, 70), element_type="text"),
        make_element(bbox=(100, 40, 140, 70), element_type="icon"),
    ]

    result = annotate_ui_elements(make_image(), elements)

    assert result.getpixel((10, 60)) == COLORS["text"]
    assert result.getpixel((140, 60)) == COLORS["icon"]


@pytest.mark.parametrize(
    ("element_type", "expected_color"),
    list(COLORS.items()),
)
def test_each_element_type_uses_fixed_color(
    element_type: str,
    expected_color: tuple[int, int, int],
) -> None:
    result = annotate_ui_elements(
        make_image(),
        [make_element(element_type=element_type)],
    )

    assert result.getpixel((20, 50)) == expected_color


def test_element_type_colors_are_distinct() -> None:
    assert len(set(COLORS.values())) == 5


def test_same_element_type_uses_same_color() -> None:
    elements = [
        make_element(bbox=(10, 40, 40, 70), element_type="input"),
        make_element(bbox=(100, 40, 140, 70), element_type="input"),
    ]

    result = annotate_ui_elements(make_image(), elements)

    assert result.getpixel((10, 60)) == COLORS["input"]
    assert result.getpixel((140, 60)) == COLORS["input"]


def test_first_label_starts_at_one(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = capture_text(monkeypatch)

    annotate_ui_elements(make_image(), [make_element(text="确定")])

    assert labels == ["1. 确定"]


def test_multiple_labels_increment_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = capture_text(monkeypatch)
    elements = [
        make_element(text="A", bbox=(10, 40, 40, 70)),
        make_element(text="B", bbox=(100, 40, 140, 70)),
    ]

    annotate_ui_elements(make_image(), elements)

    assert labels == ["1. A", "2. B"]


def test_nonempty_text_uses_approved_label_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = capture_text(monkeypatch)

    annotate_ui_elements(make_image(), [make_element(text="Settings")])

    assert labels == ["1. Settings"]


def test_empty_text_still_draws_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = capture_text(monkeypatch)

    annotate_ui_elements(make_image(), [make_element(text="")])

    assert labels == ["1."]


def test_mixed_chinese_and_english_label_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = capture_text(monkeypatch)

    result = annotate_ui_elements(
        make_image(),
        [make_element(text="设置 Settings")],
    )

    assert isinstance(result, Image.Image)
    assert labels == ["1. 设置 Settings"]


def test_label_uses_space_above_box_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangles = capture_rectangles(monkeypatch)

    annotate_ui_elements(make_image(), [make_element(bbox=(20, 50, 80, 80))])

    assert rectangles[1][1] < 50
    assert rectangles[1][3] < 50


def test_label_moves_inside_box_near_top_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangles = capture_rectangles(monkeypatch)

    annotate_ui_elements(make_image(), [make_element(bbox=(20, 1, 80, 40))])

    assert rectangles[1][1] == 1


def test_label_background_does_not_cross_left_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangles = capture_rectangles(monkeypatch)

    annotate_ui_elements(make_image(), [make_element(bbox=(0, 30, 50, 70))])

    assert rectangles[1][0] == 0


def test_label_background_does_not_cross_right_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangles = capture_rectangles(monkeypatch)

    annotate_ui_elements(
        make_image(),
        [make_element(text="Long label", bbox=(140, 30, 150, 70))],
    )

    assert rectangles[1][2] == 159


def test_label_wider_than_image_does_not_raise() -> None:
    result = annotate_ui_elements(
        make_image(size=(40, 60)),
        [make_element(text="A" * 100, bbox=(5, 30, 30, 50))],
    )

    assert isinstance(result, Image.Image)


def test_later_overlapping_element_covers_earlier_element() -> None:
    bbox = (20, 30, 80, 70)
    elements = [
        make_element(bbox=bbox, element_type="text"),
        make_element(bbox=bbox, element_type="other"),
    ]

    result = annotate_ui_elements(make_image(), elements)

    assert result.getpixel((20, 50)) == COLORS["other"]


def test_multiple_elements_load_font_once(
    monkeypatch: pytest.MonkeyPatch,
    memory_font: ImageFont.ImageFont,
) -> None:
    calls = 0

    def load_font(_path: str, _size: int) -> ImageFont.FreeTypeFont:
        nonlocal calls
        calls += 1
        return memory_font  # type: ignore[return-value]  # 测试字体满足绘制接口

    monkeypatch.setattr(ui_locator.ImageFont, "truetype", load_font)
    elements = [
        make_element(bbox=(10, 40, 40, 70)),
        make_element(bbox=(100, 40, 140, 70)),
    ]

    annotate_ui_elements(make_image(), elements)

    assert calls == 1


def test_font_candidates_are_tried_in_approved_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_font(path: str, _size: int) -> ImageFont.FreeTypeFont:
        calls.append(path)
        raise OSError("missing")

    monkeypatch.setattr(ui_locator.ImageFont, "truetype", fail_font)

    with pytest.raises(RuntimeError):
        ui_locator._load_font()

    assert calls == list(ui_locator._FONT_PATHS)


def test_font_loader_tries_next_candidate_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    memory_font: ImageFont.ImageFont,
) -> None:
    calls: list[str] = []

    def load_second(path: str, _size: int) -> ImageFont.FreeTypeFont:
        calls.append(path)
        if len(calls) == 1:
            raise OSError("missing")
        return memory_font  # type: ignore[return-value]  # 测试字体满足绘制接口

    monkeypatch.setattr(ui_locator.ImageFont, "truetype", load_second)

    result = ui_locator._load_font()

    assert result is memory_font
    assert calls == list(ui_locator._FONT_PATHS[:2])


def test_font_loader_stops_after_success(
    monkeypatch: pytest.MonkeyPatch,
    memory_font: ImageFont.ImageFont,
) -> None:
    calls: list[str] = []

    def load_first(path: str, _size: int) -> ImageFont.FreeTypeFont:
        calls.append(path)
        return memory_font  # type: ignore[return-value]  # 测试字体满足绘制接口

    monkeypatch.setattr(ui_locator.ImageFont, "truetype", load_first)

    result = ui_locator._load_font()

    assert result is memory_font
    assert calls == [ui_locator._FONT_PATHS[0]]


def test_all_font_failures_raise_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_font(_path: str, _size: int) -> ImageFont.FreeTypeFont:
        raise OSError("missing")

    monkeypatch.setattr(ui_locator.ImageFont, "truetype", fail_font)

    with pytest.raises(RuntimeError) as exc_info:
        ui_locator._load_font()

    assert isinstance(exc_info.value.__cause__, OSError)
