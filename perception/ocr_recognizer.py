"""提供基于 PP-OCRv4 的桌面图像文字识别适配。

职责：
    延迟创建 PRD 指定的 PP-OCRv4 中文 CPU 引擎，把 Pillow RGB 图像转换
    为后端需要的 BGR uint8 数组，并统一输出文字、边界框和置信度。

结构约束：
    PaddleOCR 不同结果形状只能通过显式字段和长度校验进入公共格式。文本、
    分数和框必须按索引一一对应；不完整结果作为识别失败，而非静默丢弃。

坐标约束：
    后端可能返回四坐标矩形或四点多边形。公共结果统一为包含所有点的整数
    外接矩形，不推断旋转、DPI 或跨模块坐标转换语义。

安全边界：
    OCR 原文和图像可能包含用户数据，不写入日志。后端异常转换为项目异常
    并保留根因；默认构造之外的测试可注入 predictor，避免加载真实模型。
"""

import logging
import math
from collections.abc import Iterable, Mapping
from importlib import import_module
from numbers import Integral, Real
from typing import NoReturn, Protocol, TypedDict, cast

import numpy as np
from PIL import Image

from utils.exceptions import OCRModelLoadError, OCRRecognitionError
from utils.logger import log_safe_exception

logger = logging.getLogger(__name__)


class OCRResult(TypedDict):
    """描述一项规范化后的 OCR 识别结果。

    Attributes:
        text: OCR 完整文字，只在内存感知链中传递。
        bbox: 识别区域的整数外接矩形。
        confidence: 后端返回的有限浮点置信度。

    ``OCRRecognizer.recognize`` 返回该结构给感知调用方。
    """

    text: str
    bbox: tuple[int, int, int, int]
    confidence: float


class OCRPredictor(Protocol):
    """定义 OCR 推理引擎所需的最小接口。

    Attributes:
        predictor 的模型和缓存状态由实现私有管理。

    默认实现是 PaddleOCR，测试实现不加载真实模型。
    """

    def predict(
        self,
        image: np.ndarray,
    ) -> Iterable[Mapping[str, object]]:
        """识别图像并返回可迭代的原始结果。

        Args:
            image: BGR 顺序的三通道图像数组。

        Returns:
            按推理引擎顺序产生的 OCR 原始结果。
        """


def _create_ocr_engine() -> OCRPredictor:
    """按冻结参数创建 PP-OCRv4 中文 CPU predictor。

    动态导入避免模块 import 即加载模型；任何导入或构造失败都转换为
    ``OCRModelLoadError`` 并保留原因，不自动下载或切换后端。
    """
    logger.info("开始初始化 PP-OCRv4 OCR 模型")
    try:
        module = import_module("paddleocr")
        paddle_ocr = module.PaddleOCR
        # 这些 CPU 推理与检测参数（MKL-DNN、8 线程、736/max）共同影响
        # OCR 的准确率和延迟；修改后应重新进行准确率与性能回归验证。
        engine = paddle_ocr(
            ocr_version="PP-OCRv4",
            lang="ch",
            device="cpu",
            enable_mkldnn=True,
            cpu_threads=8,
            mkldnn_cache_capacity=10,
            text_det_limit_side_len=736,
            text_det_limit_type="max",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:
        log_safe_exception(logger, "OCR 模型初始化失败", exc)
        raise OCRModelLoadError("无法初始化 PP-OCRv4 OCR 模型") from exc
    logger.info("PP-OCRv4 OCR 模型初始化成功")
    return cast(OCRPredictor, engine)


def _raise_structure_error(message: str) -> NoReturn:
    """以统一项目异常拒绝后端结构漂移。

    ``message`` 只描述固定字段合同，不能包含 OCR 原文或后端响应正文。
    """
    logger.error("OCR 结果结构无效：%s", message)
    raise OCRRecognitionError(message)


def _read_sequence(
    value: object,
    field_name: str,
    array_dimensions: int,
) -> list[object] | tuple[object, ...]:
    """读取结果序列并检查 NumPy 数组维度。

    字符串和任意 Iterable 不被接受，避免把 OCR 正文拆成字段项。
    """
    if isinstance(value, np.ndarray):
        if value.ndim != array_dimensions:
            _raise_structure_error(f"{field_name} 数组维度无效")
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return value
    _raise_structure_error(f"{field_name} 必须是序列")


def _read_coordinate(value: object, field_name: str) -> int:
    """把 Integral 坐标规范化为 Python int。

    bool 虽属于整数体系但没有像素坐标语义，因此显式拒绝。
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        _raise_structure_error(f"{field_name} 坐标必须是整数")
    coordinate = int(value)
    return coordinate


def _parse_box(value: object) -> tuple[int, int, int, int]:
    """解析具有正面积的四坐标矩形。

    这里只验证后端结构，不推断屏幕、裁剪区域或 DPI 坐标空间。
    """
    coordinates = _read_sequence(value, "rec_boxes", 1)
    if len(coordinates) != 4:
        _raise_structure_error("rec_boxes 的单项必须包含四个坐标")
    left = _read_coordinate(coordinates[0], "rec_boxes")
    top = _read_coordinate(coordinates[1], "rec_boxes")
    right = _read_coordinate(coordinates[2], "rec_boxes")
    bottom = _read_coordinate(coordinates[3], "rec_boxes")
    if left >= right or top >= bottom:
        _raise_structure_error("rec_boxes 必须具有有效顺序和非零面积")
    return left, top, right, bottom


def _parse_polygon(value: object) -> tuple[int, int, int, int]:
    """把非空点序列转换为覆盖全部点的外接矩形。

    每个点必须精确包含两个整数；退化多边形作为结构错误处理。
    """
    points = _read_sequence(value, "rec_polys", 2)
    if not points:
        _raise_structure_error("rec_polys 的单项不能为空")

    x_coordinates: list[int] = []
    y_coordinates: list[int] = []
    for point in points:
        pair = _read_sequence(point, "rec_polys 点", 1)
        if len(pair) != 2:
            _raise_structure_error("rec_polys 的点必须包含两个坐标")
        x_coordinates.append(_read_coordinate(pair[0], "rec_polys"))
        y_coordinates.append(_read_coordinate(pair[1], "rec_polys"))
    left = min(x_coordinates)
    top = min(y_coordinates)
    right = max(x_coordinates)
    bottom = max(y_coordinates)
    if left >= right or top >= bottom:
        _raise_structure_error("rec_polys 生成的边界框必须具有非零面积")
    return left, top, right, bottom


def _parse_confidence(value: object) -> float:
    """把有限实数置信度规范化为 float。

    不在此改变后端数值或自行建立阈值，保留 PRD 未定义的策略边界。
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        _raise_structure_error("rec_scores 的单项必须是实数")
    confidence = float(value)
    if not math.isfinite(confidence):
        _raise_structure_error("rec_scores 的单项必须是有限值")
    return confidence


def _validate_empty_positions(value: object, field_name: str) -> None:
    """确认空文本页面没有残留位置数据。

    这一区分合法空识别结果与并行字段错位，避免静默遗漏后端错误。
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            _raise_structure_error(f"{field_name} 数组维度无效")
        is_empty = value.shape[0] == 0
    elif isinstance(value, (list, tuple)):
        is_empty = not value
    else:
        _raise_structure_error(f"{field_name} 必须是序列")
    if not is_empty:
        _raise_structure_error("文字、置信度与位置字段长度不一致")


def _parse_page(page: object) -> list[OCRResult]:
    """把单页后端映射转换为索引严格对齐的公共结果。

    文本、置信度和位置字段长度必须一致；每项验证完成后才组装结果，
    防止部分结构有效时返回不完整 OCR 数据。
    """
    if not isinstance(page, Mapping):
        _raise_structure_error("OCR 结果项必须是映射")
    if "rec_texts" not in page:
        _raise_structure_error("OCR 结果项缺少 rec_texts")
    if "rec_scores" not in page:
        _raise_structure_error("OCR 结果项缺少 rec_scores")

    texts = _read_sequence(page["rec_texts"], "rec_texts", 1)
    scores = _read_sequence(page["rec_scores"], "rec_scores", 1)
    if len(texts) != len(scores):
        _raise_structure_error("rec_texts 与 rec_scores 长度不一致")

    # rec_boxes 和 rec_polys 同时存在时优先使用 rec_boxes，保证同一
    # 结果只采用一种边界框计算规则。
    boxes_value = page.get("rec_boxes")
    if not texts:
        # rec_texts 为空时位置字段也必须为空；空 NumPy 数组可以使用不同
        # 维度，但只要包含位置数据，就视为字段长度不一致。
        for field_name in ("rec_boxes", "rec_polys"):
            positions_value = page.get(field_name)
            if positions_value is not None:
                _validate_empty_positions(positions_value, field_name)
        return []

    use_polygons = boxes_value is None
    if use_polygons:
        if "rec_polys" not in page:
            _raise_structure_error("OCR 结果项缺少文字位置字段")
        positions = _read_sequence(page["rec_polys"], "rec_polys", 3)
    else:
        positions = _read_sequence(boxes_value, "rec_boxes", 2)
    if len(texts) != len(positions):
        _raise_structure_error("文字、置信度与位置字段长度不一致")

    # 输出统一为 Python 原生类型，隔离 PaddleOCR 和 NumPy 的容器细节，
    # 使上层只依赖稳定的文本、边界框和置信度契约。
    results: list[OCRResult] = []
    for index in range(len(texts)):
        text = texts[index]
        if not isinstance(text, str):
            _raise_structure_error("rec_texts 的单项必须是字符串")
        confidence = _parse_confidence(scores[index])
        bbox = (
            _parse_polygon(positions[index])
            if use_polygons
            else _parse_box(positions[index])
        )
        results.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
            }
        )
    return results


class OCRRecognizer:
    """将 PaddleOCR 结果转换为稳定的项目公共格式。"""

    def __init__(self, engine: OCRPredictor | None = None) -> None:
        """初始化识别器。

        Args:
            engine: 可选的 OCR 推理引擎；未提供时立即创建默认 OCR 引擎。

        Raises:
            OCRModelLoadError: 默认 OCR 模型初始化失败。
        """
        self._engine = _create_ocr_engine() if engine is None else engine

    def recognize(self, image: Image.Image) -> list[OCRResult]:
        """识别 Pillow 图像中的文字。

        Args:
            image: 待识别的 Pillow 图像。

        Returns:
            按推理引擎返回顺序整理后的 OCR 结果。

        Raises:
            TypeError: 输入不是 Pillow 图像。
            ValueError: 输入图像尺寸无效。
            OCRRecognitionError: OCR 推理失败或结果结构无效。
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 Pillow Image")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("image 的宽高必须大于 0")

        rgb_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr_array = rgb_array[:, :, ::-1].copy()
        try:
            pages = list(self._engine.predict(bgr_array))
        except Exception as exc:
            log_safe_exception(logger, "OCR 推理失败", exc)
            raise OCRRecognitionError("OCR 推理失败") from exc

        results: list[OCRResult] = []
        for page in pages:
            results.extend(_parse_page(page))
        if not results:
            logger.debug("OCR 未识别到文字")
        return results
