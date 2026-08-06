"""提供本地与 API 多模态后端之间的统一调用边界。"""

import importlib
import json
import logging
import math
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

from agent.dashscope_api_backend import (
    DASHSCOPE_CONFIGURATION_MESSAGE,
    DashScopeAPIBackend,
    DashScopeAPIConfigurationError,
    DashScopeAPIRetryableError,
)
from utils.safe_logging import log_safe_exception

logger = logging.getLogger(__name__)

_LOCAL_FAILURE_MESSAGE = "本地模型调用失败。"
_API_FAILURE_MESSAGE = "API 模型调用失败。"
_LOCAL_FAILURE_EVENT = "local_model_call_failed"
_API_FAILURE_EVENT = "api_model_call_failed"
_QWEN_REPOSITORY = "Qwen/Qwen2-VL-2B-Instruct"
_CONVERSION_MANIFEST = "conversion-manifest.json"
_OPENVINO_CONFIG = "openvino_config.json"
_MODEL_CONFIG = "config.json"
_LOCAL_LOAD_MESSAGE = "本地 Qwen2-VL 模型加载失败。"
_LOCAL_INFERENCE_MESSAGE = "本地 Qwen2-VL 模型推理失败。"
_LOCAL_OUTPUT_MESSAGE = "本地 Qwen2-VL 模型输出无效。"
_MAX_NEW_TOKENS = 256
_MAX_VISUAL_TOKENS = 1280
_VISION_PATCH_SIZE = 28
_IMAGE_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>\n"


class ModelClientError(RuntimeError):
    """供具体模型后端在其责任边界表示运行失败。"""


class LocalModelLoadError(RuntimeError):
    """表示本地模型依赖、清单或 pipeline 加载失败。"""


class LocalModelInferenceError(RuntimeError):
    """表示本地模型图像处理或推理失败。"""


class LocalModelOutputError(RuntimeError):
    """表示本地模型没有返回可用的非空文本。"""


class ModelBackend(Protocol):
    """定义模型后端必须提供的最小同步接口。"""

    def generate(self, image: Image.Image, prompt: str) -> str:
        """根据图像和提示词生成文本。"""


class Qwen2VLLocalBackend:
    """通过延迟加载的 OpenVINO GenAI CPU pipeline 调用 Qwen2-VL。"""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_new_tokens: int = 64,
        min_visual_tokens: int = 256,
        max_visual_tokens: int = 512,
    ) -> None:
        """保存并验证本地模型调用配置，不加载模型。

        Args:
            model_dir: 本地 OpenVINO 模型目录。
            max_new_tokens: 单次生成的最大新 token 数。
            min_visual_tokens: 图像缩放后的最小视觉 token 数量。
            max_visual_tokens: 图像缩放后的最大视觉 token 数量。

        Raises:
            TypeError: 配置参数类型不合法。
            ValueError: 配置值越界、视觉 token 范围不合法，或模型目录
                不存在或不是目录。

        构造阶段只验证配置与路径，不加载模型 pipeline；模型保持延迟加载。
        """
        if not isinstance(model_dir, (str, Path)):
            raise TypeError("model_dir 必须是 str 或 Path。")
        if isinstance(model_dir, str) and not model_dir.strip():
            raise ValueError("model_dir 不得为空。")
        path = Path(model_dir)
        if not path.exists():
            raise ValueError("model_dir 不存在。")
        if not path.is_dir():
            raise ValueError("model_dir 必须是目录。")
        self._validate_integer(
            max_new_tokens,
            "max_new_tokens",
            _MAX_NEW_TOKENS,
        )
        self._validate_integer(
            min_visual_tokens,
            "min_visual_tokens",
            _MAX_VISUAL_TOKENS,
        )
        self._validate_integer(
            max_visual_tokens,
            "max_visual_tokens",
            _MAX_VISUAL_TOKENS,
        )
        if min_visual_tokens > max_visual_tokens:
            raise ValueError("min_visual_tokens 不得大于 max_visual_tokens。")

        self._model_dir = path
        self._max_new_tokens = max_new_tokens
        self._min_visual_tokens = min_visual_tokens
        self._max_visual_tokens = max_visual_tokens
        self._pipeline: object | None = None
        self._openvino_module: object | None = None
        self._numpy_module: object | None = None
        self._genai_module: object | None = None

    @staticmethod
    def _validate_integer(value: object, name: str, maximum: int) -> None:
        """严格验证正整数配置及其保守上限。"""
        if type(value) is not int:
            raise TypeError(f"{name} 必须是 int。")
        if not 1 <= value <= maximum:
            raise ValueError(f"{name} 必须在 1 到 {maximum} 之间。")

    @staticmethod
    def _read_json(path: Path) -> object:
        """读取 UTF-8 JSON；具体失败由模型加载边界统一包装。"""
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def _validate_model_files(self) -> None:
        """核对部署清单、revision、模型类型和 INT4 元数据。"""
        manifest = self._read_json(self._model_dir / _CONVERSION_MANIFEST)
        openvino_config = self._read_json(self._model_dir / _OPENVINO_CONFIG)
        model_config = self._read_json(self._model_dir / _MODEL_CONFIG)
        if not isinstance(manifest, dict):
            raise ValueError("invalid conversion manifest")
        if not isinstance(openvino_config, dict):
            raise ValueError("invalid OpenVINO config")
        if not isinstance(model_config, dict):
            raise ValueError("invalid model config")

        revision = manifest.get("revision")
        quantization = manifest.get("quantization")
        if (
            manifest.get("repository") != _QWEN_REPOSITORY
            or not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
            or not self._model_dir.name.startswith(f"{revision}-")
            or manifest.get("directory_name") != self._model_dir.name
            or manifest.get("format") != "openvino_int4"
            or manifest.get("device") != "CPU"
            or manifest.get("trust_remote_code") is not False
            or not isinstance(quantization, dict)
            or quantization.get("weight_format") != "int4"
            or quantization.get("bits") != 4
        ):
            raise ValueError("conversion manifest mismatch")

        ov_quantization = openvino_config.get("quantization_config")
        if (
            openvino_config.get("dtype") != "int4"
            or not isinstance(ov_quantization, dict)
            or ov_quantization.get("dtype") != "int4"
            or ov_quantization.get("bits") != 4
            or ov_quantization.get("trust_remote_code") is not False
            or model_config.get("model_type") != "qwen2_vl"
        ):
            raise ValueError("model metadata mismatch")

    def _load_pipeline(self) -> object:
        """验证本地文件并创建固定 CPU pipeline。"""
        if self._pipeline is not None:
            return self._pipeline
        try:
            self._validate_model_files()
            openvino_module = importlib.import_module("openvino")
            numpy_module = importlib.import_module("numpy")
            genai_module = importlib.import_module("openvino_genai")
            pipeline_factory = getattr(genai_module, "VLMPipeline")
            pipeline = pipeline_factory(self._model_dir, "CPU")
        except Exception as exception:
            self._pipeline = None
            self._openvino_module = None
            self._numpy_module = None
            self._genai_module = None
            raise LocalModelLoadError(_LOCAL_LOAD_MESSAGE) from exception

        self._openvino_module = openvino_module
        self._numpy_module = numpy_module
        self._genai_module = genai_module
        self._pipeline = pipeline
        return pipeline

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        """把图像缩放到配置的视觉 Token 范围和 28 像素有效网格。"""
        # 28 是 Qwen2-VL 视觉输入的有效网格对齐尺度（patch_size=14 ×
        # spatial_merge_size=2）；按该网格缩放图像，在控制视觉 token 数量的
        # 同时保持模型要求的尺寸对齐。
        current_tokens = width * height / (_VISION_PATCH_SIZE**2)
        target_tokens = min(
            max(current_tokens, self._min_visual_tokens),
            self._max_visual_tokens,
        )
        scale = math.sqrt(target_tokens * (_VISION_PATCH_SIZE**2) / (width * height))
        target_width = max(
            _VISION_PATCH_SIZE,
            round(width * scale / _VISION_PATCH_SIZE) * _VISION_PATCH_SIZE,
        )
        target_height = max(
            _VISION_PATCH_SIZE,
            round(height * scale / _VISION_PATCH_SIZE) * _VISION_PATCH_SIZE,
        )
        while (
            target_width * target_height / (_VISION_PATCH_SIZE**2)
            > self._max_visual_tokens
        ):
            if target_width >= target_height and target_width > _VISION_PATCH_SIZE:
                target_width -= _VISION_PATCH_SIZE
            elif target_height > _VISION_PATCH_SIZE:
                target_height -= _VISION_PATCH_SIZE
            else:
                break
        return target_width, target_height

    def _prepare_tensor(self, image: Image.Image) -> object:
        """从 PIL 图像副本创建 OpenVINO NHWC RGB Tensor。"""
        if self._openvino_module is None or self._numpy_module is None:
            raise RuntimeError("runtime modules are unavailable")
        rgb_image = image.convert("RGB").copy()
        target_size = self._target_size(*rgb_image.size)
        if rgb_image.size != target_size:
            rgb_image = rgb_image.resize(target_size, Image.Resampling.BICUBIC)
        asarray = getattr(self._numpy_module, "asarray")
        array = asarray(rgb_image).copy()
        tensor_factory = getattr(self._openvino_module, "Tensor")
        return tensor_factory(array)

    @staticmethod
    def _extract_text(result: object) -> str:
        """从官方结果对象提取首个非空文本。"""
        texts = getattr(result, "texts")
        if isinstance(texts, (str, bytes)) or len(texts) == 0:
            raise ValueError("missing result text")
        text = texts[0]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("invalid result text")
        return text

    @staticmethod
    def _validate_generate_args(image: Image.Image, prompt: str) -> None:
        """在延迟加载和图像处理前验证调用参数。"""
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL.Image.Image。")
        if not isinstance(prompt, str):
            raise TypeError("prompt 必须是 str。")
        if not prompt.strip():
            raise ValueError("prompt 不得为空。")

    def generate(self, image: Image.Image, prompt: str) -> str:
        """使用固定 CPU pipeline 生成非空文本。

        Args:
            image: 待理解的 PIL 图像。
            prompt: 非空提示词文本。

        Returns:
            模型生成的非空文本。

        Raises:
            TypeError: image 或 prompt 类型不合法。
            ValueError: prompt 为空。
            LocalModelLoadError: 本地模型加载失败。
            LocalModelInferenceError: 图像处理或推理失败。
            LocalModelOutputError: 模型未返回可用的非空文本。

        首次调用在需要时延迟加载本地模型 pipeline，后续调用复用。
        """
        self._validate_generate_args(image, prompt)
        pipeline = self._load_pipeline()
        try:
            tensor = self._prepare_tensor(image)
            if self._genai_module is None:
                raise RuntimeError("runtime module is unavailable")
            config_factory = getattr(self._genai_module, "GenerationConfig")
            config = config_factory(
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
            generate_method = getattr(pipeline, "generate")
            result = generate_method(
                _IMAGE_PREFIX + prompt,
                image=tensor,
                generation_config=config,
            )
        except Exception as exception:
            raise LocalModelInferenceError(
                _LOCAL_INFERENCE_MESSAGE,
            ) from exception
        try:
            return self._extract_text(result)
        except Exception as exception:
            raise LocalModelOutputError(_LOCAL_OUTPUT_MESSAGE) from exception


class ModelClient:
    """统一执行本地模型调用、API 回退和有限 API 重试。"""

    def __init__(
        self,
        local_backend: ModelBackend,
        api_backend: ModelBackend | None = None,
        *,
        fallback_enabled: bool = True,
        max_api_retries: int = 3,
    ) -> None:
        """初始化模型客户端。

        Args:
            local_backend: 本地模型后端。
            api_backend: 可选注入后端；省略时从环境创建通义千问后端。
            fallback_enabled: 本地失败时是否允许转向 API。
            max_api_retries: 首次 API 调用失败后允许的重试次数。

        Raises:
            TypeError: 参数类型不符合合同。
            ValueError: API 重试次数不在允许范围内。
        """
        self._validate_backend(local_backend, "local_backend")
        resolved_api_backend = api_backend
        if resolved_api_backend is None:
            try:
                resolved_api_backend = DashScopeAPIBackend.from_env()
            except DashScopeAPIConfigurationError:
                resolved_api_backend = DashScopeAPIBackend(None, None)
        self._validate_backend(resolved_api_backend, "api_backend")
        if type(fallback_enabled) is not bool:
            raise TypeError("fallback_enabled 必须是 bool。")
        if type(max_api_retries) is not int:
            raise TypeError("max_api_retries 必须是 int。")
        if not 0 <= max_api_retries <= 3:
            raise ValueError("max_api_retries 必须在 0 到 3 之间。")

        self._local_backend = local_backend
        self._api_backend = resolved_api_backend
        self._fallback_enabled = fallback_enabled
        self._max_api_retries = max_api_retries

    @staticmethod
    def _validate_backend(backend: object, name: str) -> None:
        """验证后端是否提供可调用的 generate。"""
        if not callable(getattr(backend, "generate", None)):
            raise TypeError(f"{name} 必须提供可调用的 generate。")

    @staticmethod
    def _validate_generate_args(
        image: Image.Image,
        prompt: str,
        mode: str,
    ) -> None:
        """在产生后端调用前验证公共调用参数。"""
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL.Image.Image。")
        if not isinstance(prompt, str):
            raise TypeError("prompt 必须是 str。")
        if not prompt.strip():
            raise ValueError("prompt 不得为空。")
        if not isinstance(mode, str):
            raise TypeError("mode 必须是 str。")
        if mode not in ("local", "api"):
            raise ValueError("mode 只能是 local 或 api。")

    @staticmethod
    def _require_text(response: object) -> str:
        """把后端响应约束为严格文本。"""
        if not isinstance(response, str):
            raise TypeError("模型后端必须返回 str。")
        return response

    def _generate_from_api(self, image: Image.Image, prompt: str) -> str:
        """执行首次 API 调用和次数有限的失败后重试。"""
        for attempt in range(1 + self._max_api_retries):
            try:
                response = self._api_backend.generate(image, prompt)
                return self._require_text(response)
            except DashScopeAPIConfigurationError as exception:
                log_safe_exception(logger, _API_FAILURE_EVENT, exception)
                return DASHSCOPE_CONFIGURATION_MESSAGE
            except DashScopeAPIRetryableError as exception:
                log_safe_exception(logger, _API_FAILURE_EVENT, exception)
                if attempt == self._max_api_retries:
                    break
            except Exception as exception:
                log_safe_exception(logger, _API_FAILURE_EVENT, exception)
                break

        return _API_FAILURE_MESSAGE

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        mode: Literal["local", "api"] = "local",
    ) -> str:
        """调用选定后端并返回严格文本。

        Args:
            image: 原样传给后端的 PIL 图像。
            prompt: 原样传给后端的非空提示词。
            mode: 首选调用模式。

        Returns:
            模型后端文本，或固定且脱敏的运行错误信息。

        Raises:
            TypeError: 公共参数类型错误。
            ValueError: 参数值不合法。
        """
        self._validate_generate_args(image, prompt, mode)
        if mode == "api":
            return self._generate_from_api(image, prompt)

        try:
            response = self._local_backend.generate(image, prompt)
            return self._require_text(response)
        except Exception as exception:
            log_safe_exception(logger, _LOCAL_FAILURE_EVENT, exception)
            if not self._fallback_enabled:
                return _LOCAL_FAILURE_MESSAGE
        return self._generate_from_api(image, prompt)
