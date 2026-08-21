"""为模型 Prompt 组装动态状态与感知上下文。

职责：
    从 OCR、窗口枚举、焦点控件与系统音量等感知源产出单行序列化结果，
    供 ``GuiAgent`` 注入 ``Current execution state`` 与 ``Current
    perception``。本模块只做读取与换算，不产生任何桌面副作用。

坐标约束：
    所有 bbox 均换算为当前截图内 0..1000 相对坐标；窗口矩形按截图
    视口裁剪，完全在视口外的窗口不输出。

安全边界：
    不读取窗口标题；OCR 文本压缩为单行后进入 Prompt，失败时返回空
    集合并由调用方按无信息处理。
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

# 键盘就绪三态;与 ActionPromptState.keyboard_input_ready 的取值合同一致。
PromptReadinessLiteral = Literal["true", "false", "unknown"]

# OCR 过滤:清晰屏幕文字的识别分数通常不低于 0.9,取 0.8 滤除阴影与
# 抗锯齿噪声;元素上限控制 Prompt 长度并保留识别顺序。
OCR_MIN_CONFIDENCE = 0.8
OCR_MAX_ELEMENTS = 20
# 焦点放大二遍识别:全图识别对界面内小字(表格单元格、表单字段)的
# 检测率不足;对上一次动作位置附近区域裁剪放大后再识别一遍,结果
# 优先进入元素配额,让模型能"看见"自己刚操作区域的文字反馈。
FOCUS_REGION_HALF = (450, 300)
FOCUS_ZOOM = 2
FOCUS_DEDUP_DISTANCE = 40
# 焦点区域复用阈值:全图结果在焦点裁剪区域内的高置信条目达到该数时
# 视为"已有结果充足",跳过放大二遍识别。P5B 代表性 corpus 实测:密集
# 场景下二遍只会把长行重切为碎片并挤占元素配额;稀疏场景保留二遍,
# 用于补救全图检测对小字的偶发漏检。
FOCUS_REUSE_DENSE_MIN = 6
# Prompt 窗口感知上限:覆盖常见层叠场景并控制长度。
MAX_WINDOWS_IN_PROMPT = 8

# STRUCTURED GROUNDING LITE(PHASE 2B)候选上限与过滤阈值:OCR 文本
# 候选只保留高置信短标签,窗口候选前台优先;Agent 自身窗口绝不入列。
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
    """把既有 OCR/窗口感知序列化行转成结构化 grounding 候选。

    复用感知层已经产出并归一化到截图 0..1000 坐标的字符串,不重复
    枚举或 OCR。窗口候选排除 Agent 自身控制窗口;OCR 候选过滤高置信
    短文本并按置信度排序。
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
    """把候选渲染为紧凑的机器可读元素行(E 编号 + 字段)。"""
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
    """定义上下文组装需要的最小 OCR 合同。

    ``OCRRecognizer`` 是 production 实现；测试可注入内存 fake，None 表示
    不启用 OCR 辅助感知。
    """

    def recognize(self, image: Image.Image) -> list[OCRResult]:
        """识别图像文字并返回结构化结果。"""


def perceive_ocr_elements(
    recognizer: OCRRecognizerProtocol | None,
    image: Image.Image,
    focus_point: tuple[int, int] | None = None,
) -> tuple[str, ...]:
    """识别截图中可用于 Prompt 定位的文字元素。

    失败或未启用时返回空元组；文本压缩为单行，bbox 归一化为与 click
    一致的 0..1000 相对坐标。提供 ``focus_point``(当前截图内的像素
    坐标,通常是上一次已分发动作的位置)时,先对该位置附近区域裁剪
    放大做第二遍识别,其结果优先占用 ``OCR_MAX_ELEMENTS`` 配额,
    其余名额按全图识别顺序补足;若全图结果已密集覆盖焦点区域,则
    跳过二遍识别,直接复用全图结果。
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
    """同 ``perceive_ocr_elements``,并同时返回结构化 OCR 条目。

    第二个返回值为通过置信度/去重/配额过滤的 ``{"text", "bbox",
    "confidence"}`` 字典元组,bbox 为与 Prompt 行一致的 0..1000 归一化
    坐标。供完成检测等本地证据计算复用同一次识别结果,不产生第二次
    OCR。recognizer 为 None 时返回 ((), ())。
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
    """执行一次识别;任何失败按无信息处理并记安全日志。"""
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
    """判断全图结果是否已密集覆盖焦点裁剪区域。

    区域内高置信条目数达到 ``FOCUS_REUSE_DENSE_MIN`` 时返回 True,
    此时放大二遍识别不再补充新信息,调用方应直接复用全图结果。
    """
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
    """以焦点为中心裁剪固定半宽高的区域并放大,返回图与裁剪原点。

    越界部分按图像边界收敛,原点用于把放大图上的 bbox 映射回全图。
    """
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
    """把放大裁剪图上的识别 bbox 换算回全图像素坐标。

    bbox 坐标除以放大倍数得到裁剪图坐标,再叠加裁剪原点。
    """
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
    """同文本且中心距离过近视为同一元素(全图与放大遍的去重)。"""
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
    """把 Z 序可见窗口换算为当前截图内相对矩形并序列化。

    每项形如 "id:123 fg=true process=explorer.exe bbox=(x1,y1,x2,y2)";
    与截图无交集的窗口跳过，最多保留前 ``MAX_WINDOWS_IN_PROMPT``
    个，不读取窗口标题。
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
    """把全局物理矩形(left,top,right,bottom)换算为截图内0..1000矩形。

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
    """序列化任务目标窗口；未绑定时为 none。

    存在性经 ``IsWindow`` 实时验证，bbox 随窗口移动更新；身份(hwnd)
    不随前台变化重新绑定。
    """
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
    """序列化 Agent 自身控制界面窗口；不可识别时为 none。"""
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
    """把绑定窗口序列化为 Prompt 单行，含实时存在性与相对矩形。"""
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
    """判断当前是否可以安全直接 type。

    依据 GetGUIThreadInfo 的焦点控件类别；none 表示无前台输入焦点，
    text_input 再叠加 Agent 界面只读保护；不确定的一律 unknown，不凭
    上一动作猜测。
    """
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
    """读取前台焦点控件类别供状态注入。"""
    return get_focus_control_kind()


def scale_image_for_model(
    image: Image.Image,
    max_dim_limit: int,
) -> Image.Image:
    """等比缩放截图至模型输入上限以内;小图不放大。

    使用 LANCZOS 保持文字可读性;OCR 在缩放前的原图上执行,模型同时
    收到缩放图与高分辨率 OCR 文字坐标,两类信息互补。上限由 config
    提供,适配不同模型的输入能力。
    """
    longest = max(image.width, image.height)
    if longest <= max_dim_limit:
        return image
    ratio = max_dim_limit / longest
    new_size = (round(image.width * ratio), round(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)
