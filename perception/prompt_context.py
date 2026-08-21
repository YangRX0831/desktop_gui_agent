"""为模型提示词组装动态状态与桌面感知上下文。

本模块从 OCR、窗口枚举、焦点控件和系统音量等只读感知源生成结构化信息，
供 ``GuiAgent`` 注入当前执行状态与当前感知。所有边界框统一换算为当前截图内
0..1000 相对坐标；窗口矩形会按截图视口裁剪，完全不可见的窗口不会输出。

本模块不产生桌面控制副作用，也不读取窗口标题。OCR 文本进入提示词前会压缩为
单行，感知失败时返回空结果，由调用方按信息不可用处理。
"""

import logging
import re
from typing import Literal, Protocol, cast

from PIL import Image

from perception.audio_state import get_master_volume_percent
from perception.ocr_recognizer import OCRResult
from perception.screenshot import (
    FocusControlKindLiteral,
    get_focus_control_kind,
    get_foreground_app_hwnd,
    get_window_screen_rect,
    is_window_existing,
    list_visible_windows_zorder,
)
from utils.run_diagnostics import diag_log, diag_phase

logger = logging.getLogger(__name__)

PromptReadinessLiteral = Literal["true", "false", "unknown"]

# 过滤低置信 OCR 噪声，并限制提示词中的元素数量。
OCR_MIN_CONFIDENCE = 0.8
OCR_MAX_ELEMENTS = 20
# 全图 OCR 对小字号控件可能漏检，因此允许围绕最近操作位置裁剪并放大后
# 再识别一次；焦点结果优先进入元素配额。
FOCUS_REGION_HALF = (450, 300)
FOCUS_ZOOM = 2
FOCUS_DEDUP_DISTANCE = 40
# 焦点区域已经具有足够多高置信结果时跳过二次放大识别，避免重复碎片占用配额。
FOCUS_REUSE_DENSE_MIN = 6
MAX_WINDOWS_IN_PROMPT = 8

# 交互候选只保留高置信短 OCR 标签和少量可见窗口，并排除 Agent 自身窗口。
GROUNDING_MAX_OCR_CANDIDATES = 15
GROUNDING_MAX_WINDOW_CANDIDATES = 5
GROUNDING_MIN_CONFIDENCE = 0.9
GROUNDING_MAX_TEXT_LENGTH = 12
_OCR_ELEMENT_PATTERN = re.compile(
    r'^text="(.*)" bbox=\((\d+), (\d+), (\d+), (\d+)\) confidence=([0-9.]+)$',
)
_WINDOW_LINE_PATTERN = re.compile(
    r"^id:(\d+) fg=(true|false) process=(\S+) bbox=\((\d+), (\d+), (\d+), (\d+)\)$",
)


def build_grounding_candidates(
    ocr_elements: tuple[str, ...],
    windows: tuple[str, ...],
    agent_ui_hwnd: int,
) -> list[dict[str, object]]:
    """把现有 OCR 与窗口感知行转换为结构化交互候选。

    复用已经归一化到 0..1000 坐标的感知结果，不重复执行窗口枚举或 OCR。
    窗口候选排除 Agent 自身控制窗口；OCR 候选按置信度筛选和排序。
    """
    candidates: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    for line in windows:
        match = _WINDOW_LINE_PATTERN.match(line)
        if match is None:
            continue
        hwnd = int(match.group(1))
        if hwnd == agent_ui_hwnd:
            continue
        window_rows.append(
            {
                "source": "window",
                "role": "window",
                "name": match.group(3),
                "bbox": (
                    int(match.group(4)),
                    int(match.group(5)),
                    int(match.group(6)),
                    int(match.group(7)),
                ),
                "fg": match.group(2) == "true",
            },
        )
    window_rows.sort(key=lambda row: not row["fg"])
    candidates.extend(window_rows[:GROUNDING_MAX_WINDOW_CANDIDATES])

    ocr_rows: list[tuple[float, int, dict[str, object]]] = []
    for index, line in enumerate(ocr_elements):
        match = _OCR_ELEMENT_PATTERN.match(line)
        if match is None:
            continue
        text = match.group(1)
        confidence = float(match.group(6))
        if confidence < GROUNDING_MIN_CONFIDENCE:
            continue
        if len(text) > GROUNDING_MAX_TEXT_LENGTH:
            continue
        ocr_rows.append(
            (
                confidence,
                index,
                {
                    "source": "ocr",
                    "role": "text",
                    "text": text,
                    "bbox": (
                        int(match.group(2)),
                        int(match.group(3)),
                        int(match.group(4)),
                        int(match.group(5)),
                    ),
                    "confidence": confidence,
                },
            ),
        )
    ocr_rows.sort(key=lambda item: (-item[0], item[1]))
    candidates.extend(row[2] for row in ocr_rows[:GROUNDING_MAX_OCR_CANDIDATES])
    return candidates


def render_interactive_elements(
    candidates: list[dict[str, object]],
) -> tuple[str, ...]:
    """把候选渲染为带 E 编号的紧凑机器可读元素行。"""
    lines = []
    for index, candidate in enumerate(candidates, 1):
        bbox = ",".join(
            str(value) for value in cast(tuple[int, ...], candidate["bbox"])
        )
        if candidate["source"] == "window":
            lines.append(
                "E{index} source=window role=window name={name} fg={fg} "
                "bbox=({bbox})".format(
                    index=index,
                    name=candidate["name"],
                    fg="true" if candidate["fg"] else "false",
                    bbox=bbox,
                ),
            )
        else:
            lines.append(
                'E{index} source=ocr role=text text="{text}" '
                "bbox=({bbox})".format(
                    index=index,
                    text=candidate["text"],
                    bbox=bbox,
                ),
            )
    return tuple(lines)


class OCRRecognizerProtocol(Protocol):
    """定义上下文组装所需的最小 OCR 接口。"""

    def recognize(self, image: Image.Image) -> list[OCRResult]:
        """识别图像文字并返回结构化结果。"""


def perceive_ocr_elements(
    recognizer: OCRRecognizerProtocol | None,
    image: Image.Image,
    focus_point: tuple[int, int] | None = None,
) -> tuple[str, ...]:
    """识别截图中可用于提示词定位的文字元素。

    失败或未启用时返回空元组。文本压缩为单行，边界框归一化为与点击动作一致
    的 0..1000 相对坐标。提供 ``focus_point`` 时可对附近区域执行一次放大识别，
    并与全图结果去重。
    """
    return perceive_ocr_elements_detailed(
        recognizer,
        image,
        focus_point,
    )[0]


def perceive_ocr_elements_detailed(
    recognizer: OCRRecognizerProtocol | None,
    image: Image.Image,
    focus_point: tuple[int, int] | None = None,
) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    """返回序列化 OCR 元素及其对应的结构化结果。

    结构化条目包含 ``text``、``bbox`` 与 ``confidence``；bbox 与提示词中的
    坐标保持一致。完成检测等调用方可以复用同一次 OCR 结果，不需要额外识别。
    """
    if recognizer is None:
        return (), ()
    with diag_phase("diag_ocr"):
        width, height = image.size
        full_results = _recognize_quietly(recognizer, image)
        focus_results: list[dict] = []
        if focus_point is not None:
            x, y = focus_point
            if 0 <= x < width and 0 <= y < height:
                if not _is_focus_region_dense(full_results, x, y):
                    crop, origin_left, origin_top = _focus_crop(image, x, y)
                    zoom_results = _recognize_quietly(recognizer, crop)
                    focus_results = _map_crop_results(
                        zoom_results,
                        origin_left,
                        origin_top,
                    )
        elements: list[str] = []
        detailed: list[dict] = []
        taken_centers: list[tuple[str, float, float]] = []
        for item in focus_results + list(full_results):
            confidence = item["confidence"]
            if confidence < OCR_MIN_CONFIDENCE:
                continue
            text = " ".join(item["text"].split())
            if not text:
                continue
            x1, y1, x2, y2 = item["bbox"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if _is_duplicate(text, center, taken_centers):
                continue
            taken_centers.append((text, center[0], center[1]))
            normalized = tuple(
                min(1000, max(0, round(value * 1000 / span)))
                for value, span in (
                    (x1, width),
                    (y1, height),
                    (x2, width),
                    (y2, height),
                )
            )
            elements.append(
                'text="{}" bbox=({}, {}, {}, {}) confidence={:.2f}'.format(
                    text,
                    *normalized,
                    confidence,
                ),
            )
            detailed.append(
                {
                    "text": text,
                    "bbox": normalized,
                    "confidence": confidence,
                },
            )
            if len(elements) >= OCR_MAX_ELEMENTS:
                break
        diag_log(
            "diag_ocr_stats",
            image_w=width,
            image_h=height,
            elements=len(elements),
        )
        return tuple(elements), tuple(detailed)


def _recognize_quietly(
    recognizer: OCRRecognizerProtocol,
    image: Image.Image,
) -> list[dict]:
    """执行一次 OCR；失败时按无信息处理并记录异常类型。"""
    try:
        return [dict(result) for result in recognizer.recognize(image)]
    except Exception as exception:
        logger.warning(
            "prompt_context_ocr_failed：exception_type=%s",
            type(exception).__name__,
        )
        return []


def _is_focus_region_dense(
    results: list[dict],
    x: int,
    y: int,
) -> bool:
    """判断全图 OCR 是否已经充分覆盖焦点附近区域。"""
    half_w, half_h = FOCUS_REGION_HALF
    left, top, right, bottom = x - half_w, y - half_h, x + half_w, y + half_h
    count = 0
    for item in results:
        if item["confidence"] < OCR_MIN_CONFIDENCE:
            continue
        x1, y1, x2, y2 = item["bbox"]
        if x2 > left and x1 < right and y2 > top and y1 < bottom:
            count += 1
            if count >= FOCUS_REUSE_DENSE_MIN:
                return True
    return False


def _focus_crop(
    image: Image.Image,
    x: int,
    y: int,
) -> tuple[Image.Image, int, int]:
    """以焦点为中心裁剪固定区域并放大，返回图像和裁剪原点。"""
    width, height = image.size
    half_w, half_h = FOCUS_REGION_HALF
    left = max(0, x - half_w)
    top = max(0, y - half_h)
    right = min(width, x + half_w)
    bottom = min(height, y + half_h)
    crop = image.crop((left, top, right, bottom))
    zoomed = crop.resize(
        (crop.width * FOCUS_ZOOM, crop.height * FOCUS_ZOOM),
        Image.Resampling.LANCZOS,
    )
    return zoomed, left, top


def _map_crop_results(
    zoom_results: list[dict],
    origin_left: int,
    origin_top: int,
) -> list[dict]:
    """把放大裁剪图中的 OCR 边界框换算回全图像素坐标。"""
    mapped: list[dict] = []
    for item in zoom_results:
        x1, y1, x2, y2 = item["bbox"]
        mapped.append(
            {
                "text": item["text"],
                "bbox": (
                    x1 / FOCUS_ZOOM + origin_left,
                    y1 / FOCUS_ZOOM + origin_top,
                    x2 / FOCUS_ZOOM + origin_left,
                    y2 / FOCUS_ZOOM + origin_top,
                ),
                "confidence": item["confidence"],
            },
        )
    return mapped


def _is_duplicate(
    text: str,
    center: tuple[float, float],
    taken: list[tuple[str, float, float]],
) -> bool:
    """同文本且中心距离过近时视为同一元素。"""
    return any(
        earlier_text == text
        and abs(earlier_x - center[0]) < FOCUS_DEDUP_DISTANCE
        and abs(earlier_y - center[1]) < FOCUS_DEDUP_DISTANCE
        for earlier_text, earlier_x, earlier_y in taken
    )


def perceive_windows(
    screenshot_size: tuple[int, int],
    region_offset: tuple[int, int],
) -> tuple[str, ...]:
    """把按 Z 序排列的可见窗口换算为当前截图中的相对矩形。

    每项包含窗口句柄、前台状态、进程名和边界框；与截图无交集的窗口会被跳过，
    并限制总数量以控制提示词长度。
    """
    entries: list[str] = []
    for window in list_visible_windows_zorder():
        if len(entries) >= MAX_WINDOWS_IN_PROMPT:
            break
        left, top, width, height = cast(
            tuple[int, int, int, int],
            window["rect"],
        )
        bbox = rect_to_viewport_bbox(
            (left, top, left + width, top + height),
            screenshot_size,
            region_offset,
        )
        if bbox is None:
            continue
        entries.append(
            "id:{} fg={} process={} bbox={}".format(
                window["hwnd"],
                "true" if window["foreground"] else "false",
                window["process"],
                bbox,
            ),
        )
    return tuple(entries)


def rect_to_viewport_bbox(
    rect: tuple[int, int, int, int],
    screenshot_size: tuple[int, int],
    region_offset: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """把全局物理矩形换算为截图内 0..1000 相对矩形。

    完全在截图之外返回 None；部分可见时裁剪到截图范围。
    """
    left, top, right, bottom = rect
    screen_w, screen_h = screenshot_size
    view_left = region_offset[0]
    view_top = region_offset[1]
    view_right = view_left + screen_w
    view_bottom = view_top + screen_h
    if (
        right <= view_left
        or bottom <= view_top
        or left >= view_right
        or top >= view_bottom
    ):
        return None
    clamped = (
        max(left, view_left),
        max(top, view_top),
        min(right, view_right),
        min(bottom, view_bottom),
    )

    def _scale(value: int, origin: int, span: int) -> int:
        return min(1000, max(0, round((value - origin) * 1000 / span)))

    return (
        _scale(clamped[0], view_left, screen_w),
        _scale(clamped[1], view_top, screen_h),
        _scale(clamped[2], view_left, screen_w),
        _scale(clamped[3], view_top, screen_h),
    )


def task_target_window_state(
    target: dict[str, object] | None,
    screenshot_size: tuple[int, int],
    region_offset: tuple[int, int],
) -> str:
    """序列化已绑定的任务目标窗口；未绑定时返回 ``none``。"""
    if target is None:
        return "none"
    return serialize_bound_window(
        int(cast(int, target["hwnd"])),
        str(target["process"]),
        screenshot_size,
        region_offset,
    )


def agent_ui_window_state(
    agent_ui_hwnd: int,
    screenshot_size: tuple[int, int],
    region_offset: tuple[int, int],
) -> str:
    """序列化 Agent 自身控制窗口；不可识别时返回 ``none``。"""
    if not agent_ui_hwnd:
        return "none"
    rect = get_window_screen_rect(agent_ui_hwnd)
    bbox_text = "unknown"
    if rect is not None:
        left, top, width, height = rect
        bbox = rect_to_viewport_bbox(
            (left, top, left + width, top + height),
            screenshot_size,
            region_offset,
        )
        if bbox is not None:
            bbox_text = str(bbox)
    return f"id:{agent_ui_hwnd}, bbox:{bbox_text}, protected:true"


def serialize_bound_window(
    hwnd: int,
    process: str,
    screenshot_size: tuple[int, int],
    region_offset: tuple[int, int],
) -> str:
    """把绑定窗口序列化为单行状态，包含存在性与相对矩形。"""
    exists = is_window_existing(hwnd)
    state = "exists" if exists else "closed"
    bbox_text = "unknown"
    if exists:
        rect = get_window_screen_rect(hwnd)
        if rect is not None:
            left, top, width, height = rect
            bbox = rect_to_viewport_bbox(
                (left, top, left + width, top + height),
                screenshot_size,
                region_offset,
            )
            if bbox is not None:
                bbox_text = str(bbox)
    return f"id:{hwnd}, process:{process}, bbox:{bbox_text}, state:{state}"


def keyboard_input_ready(
    focus_kind: FocusControlKindLiteral,
    protect_agent_ui: bool,
    agent_ui_hwnd: int,
    unlocked: bool,
) -> PromptReadinessLiteral:
    """根据焦点控件与 Agent 窗口保护状态判断是否可以直接输入文本。"""
    if focus_kind == "none":
        return "false"
    if focus_kind == "text_input":
        if (
            protect_agent_ui
            and agent_ui_hwnd
            and get_foreground_app_hwnd() == agent_ui_hwnd
            and not unlocked
        ):
            return "false"
        return "true"
    return "unknown"


def system_volume_state() -> int | None:
    """读取当前系统主音量百分比；不可用时返回 None。"""
    return get_master_volume_percent()


def focus_control_state() -> FocusControlKindLiteral:
    """读取前台焦点控件类别。"""
    return get_focus_control_kind()


def scale_image_for_model(
    image: Image.Image,
    max_dim_limit: int,
) -> Image.Image:
    """按比例缩放截图至模型输入上限以内，小图不放大。

    OCR 在原始图像上执行，模型图像则按配置限制长边，从而兼顾文字识别信息与
    模型视觉输入成本。
    """
    longest = max(image.width, image.height)
    if longest <= max_dim_limit:
        return image
    ratio = max_dim_limit / longest
    new_size = (round(image.width * ratio), round(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)
