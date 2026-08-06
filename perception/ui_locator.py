"""在图像上标注调用方提供的 UI 元素。"""

import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, TypedDict, cast

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

UIElementType = Literal[
    "text",
    "button",
    "input",
    "icon",
    "other",
]


class UIElement(TypedDict):
    """描述一个待标注的 UI 元素。"""

    text: str
    bbox: tuple[int, int, int, int]
    element_type: UIElementType


_Color = tuple[int, int, int]
_ELEMENT_TYPES: tuple[UIElementType, ...] = (
    "text",
    "button",
    "input",
    "icon",
    "other",
)
_ELEMENT_COLORS: Mapping[UIElementType, _Color] = MappingProxyType(
    {
        "text": (0, 102, 204),
        "button": (0, 153, 76),
        "input": (230, 126, 34),
        "icon": (128, 0, 128),
        "other": (204, 0, 0),
    }
)
_FONT_PATHS = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
)
_FONT_SIZE = 16
_BORDER_WIDTH = 2
_LABEL_PADDING = 2
_LABEL_GAP = 2
_LABEL_TEXT_COLOR = (255, 255, 255)
_ELEMENT_FIELDS = {"text", "bbox", "element_type"}


def _validate_image(image: object) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image 必须是 Pillow Image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image 的宽高必须大于 0")
    return image


def _validate_bbox(
    bbox: object,
    image: Image.Image,
) -> tuple[int, int, int, int]:
    if not isinstance(bbox, tuple):
        raise TypeError("bbox 必须是 tuple")
    if len(bbox) != 4:
        raise ValueError("bbox 必须包含四个坐标")
    if any(type(coordinate) is not int for coordinate in bbox):
        raise TypeError("bbox 坐标必须是 Python int")

    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox 必须具有有效顺序和非零面积")
    if x1 < 0 or y1 < 0 or x2 >= image.width or y2 >= image.height:
        raise ValueError("bbox 必须完全位于图像范围内")
    return x1, y1, x2, y2


def _validate_element(element: object, image: Image.Image) -> UIElement:
    if not isinstance(element, dict):
        raise TypeError("elements 的每一项必须是 dict")
    if set(element) != _ELEMENT_FIELDS:
        raise ValueError("UI 元素字段必须严格为 text、bbox 和 element_type")

    text = element["text"]
    if not isinstance(text, str):
        raise TypeError("text 必须是 str")

    element_type = element["element_type"]
    if not isinstance(element_type, str):
        raise TypeError("element_type 必须是 str")
    if element_type not in _ELEMENT_TYPES:
        raise ValueError("element_type 不受支持")

    bbox = _validate_bbox(element["bbox"], image)
    return {
        "text": text,
        "bbox": bbox,
        "element_type": cast(UIElementType, element_type),
    }


def _validate_elements(
    elements: object,
    image: Image.Image,
) -> list[UIElement]:
    if isinstance(elements, (str, bytes, bytearray)) or not isinstance(
        elements,
        Sequence,
    ):
        raise TypeError("elements 必须是非字符串 Sequence")
    return [_validate_element(element, image) for element in elements]


def _load_font() -> ImageFont.FreeTypeFont:
    last_error: OSError | None = None
    for font_path in _FONT_PATHS:
        try:
            return ImageFont.truetype(font_path, _FONT_SIZE)
        except OSError as exc:
            last_error = exc
            logger.debug("候选字体加载失败：%s", font_path)
    logger.error("所有候选字体均加载失败")
    raise RuntimeError("无法加载 UI 元素标注字体") from last_error


def _label_position(
    draw: ImageDraw.ImageDraw,
    label: str,
    bbox: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont,
    image: Image.Image,
) -> tuple[float, float, float, float, float, float]:
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_width = text_width + 2 * _LABEL_PADDING
    label_height = text_height + 2 * _LABEL_PADDING

    # 标签优先绘制在元素上方；顶部空间不足时移入元素框内。同时限制
    # 标签背景矩形的范围，避免其超出图像边界。
    x1, y1, _, _ = bbox
    label_x = min(x1, max(image.width - label_width, 0))
    preferred_y = y1 - _LABEL_GAP - label_height
    label_y = preferred_y if preferred_y >= 0 else y1
    right = min(label_x + label_width - 1, image.width - 1)
    bottom = min(label_y + label_height - 1, image.height - 1)
    text_x = label_x + _LABEL_PADDING - text_bbox[0]
    text_y = label_y + _LABEL_PADDING - text_bbox[1]
    return label_x, label_y, right, bottom, text_x, text_y


def _draw_element(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    element: UIElement,
    index: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    color = _ELEMENT_COLORS[element["element_type"]]
    draw.rectangle(element["bbox"], outline=color, width=_BORDER_WIDTH)

    label = f"{index}." if not element["text"] else f"{index}. {element['text']}"
    left, top, right, bottom, text_x, text_y = _label_position(
        draw,
        label,
        element["bbox"],
        font,
        image,
    )
    draw.rectangle((left, top, right, bottom), fill=color)
    draw.text((text_x, text_y), label, fill=_LABEL_TEXT_COLOR, font=font)


def annotate_ui_elements(
    image: Image.Image,
    elements: Sequence[UIElement],
) -> Image.Image:
    """在图像上绘制调用方提供的 UI 元素。

    Args:
        image: 原始 Pillow 图像。
        elements: 按绘制顺序排列的 UI 元素。

    Returns:
        空元素序列返回原图；否则返回带标注的 RGB 图像副本。

    Raises:
        TypeError: 图像、元素、字段或坐标类型无效。
        ValueError: 图像尺寸、元素字段、类型或坐标值无效。
        RuntimeError: 所有候选字体均无法加载。
    """
    source_image = _validate_image(image)
    validated_elements = _validate_elements(elements, source_image)
    if not validated_elements:
        return source_image

    # 非空标注始终作用于副本，避免调用方后续复用原图时观察到隐式修改。
    result = (
        source_image.copy()
        if source_image.mode == "RGB"
        else source_image.convert("RGB")
    )
    font = _load_font()
    draw = ImageDraw.Draw(result)
    for index, element in enumerate(validated_elements, start=1):
        _draw_element(draw, result, element, index, font)
    return result
