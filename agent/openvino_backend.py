"""提供 OpenVINO GenAI 的 Qwen2-VL 本地 CPU 推理后端。

职责：
    加载 OpenVINO INT4 导出模型，提供与 ``Qwen2VLLocalBackend`` 相同的
    ``generate(image, prompt)`` 合同，作为纯 CPU 环境下的本地推理路线；
    Transformers 路线是默认 canonical 实现，两者经
    ``GUI_AGENT_LOCAL_RUNTIME`` 配置显式选择。

安全边界：
    openvino_genai 属于可选后端，仅在本模块内动态导入；模块导入不加载
    模型。日志只记录固定事件与异常类型，不含 prompt 或图像内容。
"""

import importlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MAX_NEW_TOKENS = 256
_VISION_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>\n"


class OpenVINOModelLoadError(RuntimeError):
    """表示 OpenVINO 模型目录校验或 pipeline 加载失败。"""


class OpenVINOModelInferenceError(RuntimeError):
    """表示 OpenVINO 推理或输出提取失败。"""


class Qwen2VLOpenVINOBackend:
    """通过 OpenVINO GenAI VLMPipeline 执行 Qwen2-VL INT4 推理。

    Attributes:
        model_dir: OpenVINO INT4 导出模型目录(含 openvino_model.xml 等)。
        device: 推理设备，production 默认 CPU。
        max_new_tokens: 单次生成的输出 token 上限。

    pipeline 延迟加载：构造只校验目录，首次 generate 才加载权重。
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "CPU",
        max_new_tokens: int = 256,
    ) -> None:
        """校验并保存 OpenVINO 后端配置，不加载模型。

        Raises:
            TypeError: 参数类型不合法。
            ValueError: 目录不存在或缺少年份导出的模型文件。
        """
        if not isinstance(model_dir, (str, Path)):
            raise TypeError("model_dir 必须是 str 或 Path。")
        if isinstance(model_dir, str) and not model_dir.strip():
            raise ValueError("model_dir 不得为空。")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device 必须是非空 str。")
        if (
            type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= _MAX_NEW_TOKENS
        ):
            raise ValueError(f"max_new_tokens 必须在 1 到 {_MAX_NEW_TOKENS} 之间。")
        path = Path(model_dir)
        if not path.is_dir():
            raise ValueError("model_dir 不存在或不是目录。")
        if not (path / "openvino_language_model.xml").is_file():
            raise ValueError(
                "model_dir 缺少 openvino_language_model.xml，不是导出模型。",
            )

        self._model_dir = path
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._pipeline: object | None = None

    def _load_pipeline(self) -> object:
        """延迟加载 VLMPipeline；失败转换为项目异常并保留原因。"""
        if self._pipeline is not None:
            return self._pipeline
        try:
            genai = importlib.import_module("openvino_genai")
            pipeline_cls = getattr(genai, "VLMPipeline")
            self._pipeline = pipeline_cls(
                str(self._model_dir),
                device=self._device,
            )
        except Exception as exception:
            self._pipeline = None
            logger.warning(
                "openvino_backend_load_failed：exception_type=%s",
                type(exception).__name__,
            )
            raise OpenVINOModelLoadError("OpenVINO 模型加载失败。") from exception
        return self._pipeline

    def generate(self, image: Image.Image, prompt: str) -> str:
        """对单张截图和完整 Prompt 执行一次 CPU 推理。

        Args:
            image: RGB 截图。
            prompt: 完整动作 Prompt 文本。

        Returns:
            模型生成的非空文本。

        Raises:
            TypeError: 输入类型不合法。
            ValueError: prompt 为空。
            OpenVINOModelLoadError: 模型加载失败。
            OpenVINOModelInferenceError: 推理失败或输出为空。
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL.Image.Image。")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空 str。")

        pipeline = self._load_pipeline()
        try:
            openvino = importlib.import_module("openvino")
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            tensor = openvino.Tensor(array)
            config_cls = getattr(
                importlib.import_module("openvino_genai"),
                "GenerationConfig",
            )
            config = config_cls(
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
            result = pipeline.generate(  # type: ignore[attr-defined]  # 动态导入
                _VISION_PREFIX + prompt,
                image=tensor,
                generation_config=config,
            )
        except OpenVINOModelLoadError:
            raise
        except Exception as exception:
            logger.warning(
                "openvino_backend_inference_failed：exception_type=%s",
                type(exception).__name__,
            )
            raise OpenVINOModelInferenceError("OpenVINO 推理失败。") from exception

        text = str(result).strip() if result is not None else ""
        if not text:
            raise OpenVINOModelInferenceError("OpenVINO 模型输出为空。")
        return text
