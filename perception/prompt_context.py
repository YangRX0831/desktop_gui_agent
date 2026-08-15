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
from typing import Protocol

from PIL import Image

from perception.audio_state import get_master_volume_percent
from perception.ocr_recognizer import OCRResult
from perception.screenshot import (
    get_focus_control_kind,
    get_foreground_app_hwnd,
    get_window_screen_rect,
    is_window_existing,
    list_visible_windows_zorder,
)

logger = logging.getLogger(__name__)

# OCR 过滤:清晰屏幕文字的识别分数通常不低于 0.9,取 0.8 滤除阴影与
# 抗锯齿噪声;元素上限控制 Prompt 长度并保留识别顺序。
OCR_MIN_CONFIDENCE = 0.8
OCR_MAX_ELEMENTS = 20
# Prompt 窗口感知上限:覆盖常见层叠场景并控制长度。
MAX_WINDOWS_IN_PROMPT = 8


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
) -> tuple[str, ...]:
    """识别截图中可用于 Prompt 定位的文字元素。

    失败或未启用时返回空元组；文本压缩为单行，bbox 归一化为与 click
    一致的 0..1000 相对坐标，按识别顺序保留前 ``OCR_MAX_ELEMENTS``
    个高置信度元素。
    """
    if recognizer is None:
        return ()
    try:
        results = recognizer.recognize(image)
    except Exception as exception:
        logger.warning(
            "prompt_context_ocr_failed：exception_type=%s",
            type(exception).__name__,
        )
        return ()
    width, height = image.size
    elements: list[str] = []
    for item in results:
        confidence = item["confidence"]
        if confidence < OCR_MIN_CONFIDENCE:
            continue
        text = " ".join(item["text"].split())
        if not text:
            continue
        x1, y1, x2, y2 = item["bbox"]
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
        if len(elements) >= OCR_MAX_ELEMENTS:
            break
    return tuple(elements)


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
        left, top, width, height = window["rect"]
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
    return tuple(
        min(1000, max(0, round((value - origin) * 1000 / span)))
        for value, origin, span in (
            (clamped[0], view_left, screen_w),
            (clamped[1], view_top, screen_h),
            (clamped[2], view_left, screen_w),
            (clamped[3], view_top, screen_h),
        )
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
        int(target["hwnd"]),
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
    focus_kind: str,
    protect_agent_ui: bool,
    agent_ui_hwnd: int,
    unlocked: bool,
) -> str:
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


def focus_control_state() -> str:
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
    return image.resize(new_size, Image.LANCZOS)
