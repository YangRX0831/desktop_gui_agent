"""在图像上标注调用方提供的 UI 元素。

本模块对 OCR 或其他感知组件已经提供的 UI 元素绘制编号、标签和类型颜色，
生成便于模型观察的 Pillow 图像；自身不负责目标检测或 OCR。

输入元素必须精确包含 ``text``、``bbox`` 和 ``element_type`` 三个字段。边界框
必须完全位于图像内且具有正面积。非空标注始终基于当前图像副本绘制，避免修改
调用方持有的原始截图。
"""

import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, TypedDict, cast

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

UIElementType = Literal[
    "text",
    "button",
    "input",
    "icon",
    "other",
]


class UIElement(TypedDict):
    """描述一个待标注的 UI 元素。

    Attributes:
        text: 可显示的元素文字，空字符串表示仅编号。
        bbox: 完全位于图像内的整数矩形。
        element_type: text、button、input、icon 或 other。
    """

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

# 缓存已经验证的 UI 元素结构，仅减少重复参数校验；渲染始终使用当前图像，
# 不缓存或复用旧截图的像素结果。
_ELEMENT_CACHE: dict[tuple, tuple[dict[str, object], ...]] = {}
_ELEMENT_CACHE_LIMIT = 64
_STABILITY_DIFF_RATIO = 0.005


def frames_stable(
    previous: Image.Image,
    current: Image.Image,
    threshold: float = _STABILITY_DIFF_RATIO,
) -> bool:
    """两帧显著差异像素占比小于阈值时返回 True。"""
    previous_gray = cv2.cvtColor(
        np.array(previous.convert("RGB")),
        cv2.COLOR_RGB2GRAY,
    )
    current_gray = cv2.cvtColor(
        np.array(current.convert("RGB")),
        cv2.COLOR_RGB2GRAY,
    )
    mask = cv2.threshold(
        cv2.absdiff(previous_gray, current_gray),
        30,
        255,
        cv2.THRESH_BINARY,
    )[1]
    return (int(np.count_nonzero(mask)) / mask.size) < threshold


def frame_change_ratio(previous: Image.Image, current: Image.Image) -> float:
    """返回两帧中差异超过 30 灰度级的像素比例。"""
    if previous.size != current.size:
        return 1.0
    difference = ImageChops.difference(
        previous.convert("RGB"),
        current.convert("RGB"),
    ).convert("L")
    histogram = difference.histogram()
    changed_pixels = sum(histogram[31:])
    return changed_pixels / (previous.width * previous.height)


def _validate_image(image: object) -> Image.Image:
    """校验输入为有正尺寸的 Pillow 图像并返回同一对象。"""
    if not isinstance(image, Image.Image):
        raise TypeError("image 必须是 Pillow Image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image 的宽高必须大于 0")
    return image


def _validate_bbox(
    bbox: object,
    image: Image.Image,
) -> tuple[int, int, int, int]:
    """校验绘制矩形的形状、顺序、面积和图像边界。"""
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
    """验证单个元素并返回类型收窄后的新字典。"""
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
    """验证元素容器并保持调用方给定的绘制顺序。"""
    if isinstance(elements, (str, bytes, bytearray)) or not isinstance(
        elements,
        Sequence,
    ):
        raise TypeError("elements 必须是非字符串 Sequence")
    return [_validate_element(element, image) for element in elements]


def _element_cache_key(
    image: Image.Image,
    elements: Sequence[UIElement],
) -> tuple | None:
    """构建包含图像尺寸和元素内容的确定性缓存键。"""
    try:
        fingerprint = tuple(
            (e["text"], tuple(e["bbox"]), e["element_type"]) for e in elements
        )
    except (KeyError, TypeError, ValueError):
        return None
    return (image.width, image.height, fingerprint)


def _clear_element_cache() -> None:
    """清空 UI 元素信息缓存。"""
    _ELEMENT_CACHE.clear()


def _load_font() -> ImageFont.FreeTypeFont:
    """按固定顺序加载首个可用中文字体。

    不使用网络或运行时字体安装；所有候选失败时保留最后异常为根因。
    """
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
    """计算受图像边界约束的标签背景与文字位置。"""
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_width = text_width + 2 * _LABEL_PADDING
    label_height = text_height + 2 * _LABEL_PADDING

    # 标签优先绘制在元素上方；顶部空间不足时移入元素框内，并限制背景矩形
    # 仍处于图像范围内。
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
    """用类型颜色绘制一个已验证元素及稳定编号标签。"""
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

    cache_key = _element_cache_key(source_image, elements)
    validated_elements: list[UIElement] | None = None
    if cache_key is not None and cache_key in _ELEMENT_CACHE:
        # 命中时复用已验证的元素结构副本，不复用历史图像。
        cached = cast(list[UIElement], _ELEMENT_CACHE[cache_key])
        validated_elements = cast(list[UIElement], [dict(e) for e in cached])

    if validated_elements is None:
        validated_elements = _validate_elements(elements, source_image)
        if cache_key is not None and validated_elements:
            if len(_ELEMENT_CACHE) >= _ELEMENT_CACHE_LIMIT:
                _ELEMENT_CACHE.clear()
            _ELEMENT_CACHE[cache_key] = tuple(dict(e) for e in validated_elements)

    if not validated_elements:
        return source_image

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
