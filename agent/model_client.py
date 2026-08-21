"""提供本地模型与 DashScope API 后端之间的统一调用边界。

``ModelClient`` 向智能体暴露统一的 ``generate`` 接口，并负责本地后端调用、
API 调用、有限的 API 传输重试以及本地模型加载失败时的 API 回退。

API 首次请求失败后最多重试三次。调用方不应在模型客户端外再次叠加网络重试，
以避免请求次数成倍放大。

本地 Transformers 后端使用 Qwen2-VL-2B-Instruct 与 4-bit 量化，模型权重
保存在仓库外并在首次生成时延迟加载。只有 ``LocalModelLoadError`` 可以触发
PRD 规定的本地模型加载失败回退；推理失败和输出失败不会自动发送到远程 API。

日志只记录固定事件、异常类型和安全代码位置，不记录原始图像、提示词、模型
响应、异常正文或模型目录内容。
"""

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

from agent.dashscope_api_backend import (
    DASHSCOPE_CONFIGURATION_MESSAGE,
    DashScopeAPIBackend,
    DashScopeAPIConfigurationError,
    DashScopeAPIRetryableError,
)
from utils.logger import log_safe_exception

logger = logging.getLogger(__name__)

_LOCAL_FAILURE_MESSAGE = "本地模型调用失败。"
_API_FAILURE_MESSAGE = "API 模型调用失败。"
_LOCAL_FAILURE_EVENT = "local_model_call_failed"
_API_FAILURE_EVENT = "api_model_call_failed"
_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
_LOCAL_LOAD_MESSAGE = "本地 Qwen2-VL 模型加载失败。"
_LOCAL_INFERENCE_MESSAGE = "本地 Qwen2-VL 模型推理失败。"
_LOCAL_OUTPUT_MESSAGE = "本地 Qwen2-VL 模型输出无效。"
_LOCAL_ARGS_MESSAGE = "本地 Qwen2-VL 模型参数无效。"
_MAX_NEW_TOKENS = 256


class LocalModelLoadError(RuntimeError):
    """表示本地模型权重、processor 或 pipeline 加载失败。

    该异常是本地模式允许切换到 API 后端的唯一错误类型；本地推理失败不属于
    模型加载失败。
    """


class LocalModelInferenceError(RuntimeError):
    """表示本地模型图像处理或推理失败。

    该错误不会触发本地模式到 API 模式的自动回退，本地后端也不会自行重试。
    """


class LocalModelOutputError(RuntimeError):
    """表示本地模型没有返回可用的非空文本。

    该错误不会通过放宽解析规则恢复，也不会触发 API 回退。
    """


class ModelBackend(Protocol):
    """定义模型后端必须提供的最小同步接口。"""

    def generate(self, image: Image.Image, prompt: str) -> str:
        """根据图像和提示词生成文本。"""


class Qwen2VLLocalBackend:
    """通过 Transformers 与 4-bit 量化加载 Qwen2-VL-2B-Instruct。

    模型和 processor 均采用延迟加载，构造阶段只保存并验证模型目录与生成参数。
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_new_tokens: int = 64,
    ) -> None:
        """保存并验证本地模型调用配置，不加载模型。

        Args:
            model_dir: 本地 Qwen2-VL-2B-Instruct 模型目录。
            max_new_tokens: 单次生成的最大新 token 数。

        Raises:
            TypeError: 配置参数类型不合法。
            ValueError: 配置值越界，或模型目录不存在或不是目录。
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
        if type(max_new_tokens) is not int:
            raise TypeError("max_new_tokens 必须是 int。")
        if not 1 <= max_new_tokens <= _MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens 必须在 1 到 {_MAX_NEW_TOKENS} 之间。")

        self._model_dir = path
        self._max_new_tokens = max_new_tokens
        self._model: object | None = None
        self._processor: object | None = None
        self._transformers_module: object | None = None

    @staticmethod
    def _read_config(path: Path) -> object:
        """读取 UTF-8 JSON 模型配置；失败由加载边界统一包装。"""
        import json

        return json.loads(path.read_text(encoding="utf-8-sig"))

    def _validate_model_dir(self) -> None:
        """核对模型目录包含 Qwen2-VL 所需的基本文件。"""
        config_path = self._model_dir / "config.json"
        if not config_path.is_file():
            raise ValueError("model_dir 缺少 config.json。")
        try:
            config = self._read_config(config_path)
        except Exception as exception:
            raise ValueError("model_dir config.json 无效。") from exception
        if not isinstance(config, dict):
            raise ValueError("model_dir config.json 结构无效。")
        if config.get("model_type") != "qwen2_vl":
            raise ValueError("model_dir 不是 qwen2_vl 模型。")
        if not (self._model_dir / "preprocessor_config.json").is_file():
            raise ValueError("model_dir 缺少 preprocessor_config.json。")
        if not (self._model_dir / "tokenizer.json").is_file():
            raise ValueError("model_dir 缺少 tokenizer.json。")

    def _load_pipeline(self) -> tuple[object, object, object]:
        """延迟加载 Transformers 模型、processor 与模块句柄。

        Raises:
            LocalModelLoadError: 模型、processor 或其依赖加载失败。
        """
        if self._model is not None and self._processor is not None:
            return self._model, self._processor, self._transformers_module
        try:
            self._validate_model_dir()
            transformers_module = importlib.import_module("transformers")
            auto_processor = getattr(transformers_module, "AutoProcessor")
            model_cls = getattr(transformers_module, "Qwen2VLForConditionalGeneration")
            bnb_config_cls = getattr(transformers_module, "BitsAndBytesConfig")
            # 使用 BitsAndBytesConfig(load_in_4bit=True) 配置 4-bit 权重量化。
            # local_files_only=True 防止模型文件缺失时隐式联网下载。
            quantization_config = bnb_config_cls(load_in_4bit=True)
            processor = auto_processor.from_pretrained(
                self._model_dir,
                local_files_only=True,
            )
            model = model_cls.from_pretrained(
                self._model_dir,
                quantization_config=quantization_config,
                local_files_only=True,
            )
            model.eval()
        except Exception as exception:
            self._model = None
            self._processor = None
            self._transformers_module = None
            raise LocalModelLoadError(_LOCAL_LOAD_MESSAGE) from exception

        self._model = model
        self._processor = processor
        self._transformers_module = transformers_module
        return model, processor, transformers_module

    @staticmethod
    def _validate_generate_args(image: Image.Image, prompt: str) -> None:
        """在延迟加载和图像处理前验证调用参数。"""
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL.Image.Image。")
        if not isinstance(prompt, str):
            raise TypeError("prompt 必须是 str。")
        if not prompt.strip():
            raise ValueError("prompt 不得为空。")

    def _build_inputs(
        self,
        processor: object,
        image: Image.Image,
        prompt: str,
    ) -> object:
        """把 PIL 图像与提示词转换为模型输入。"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            chat_template = getattr(processor, "apply_chat_template", None)
            if chat_template is None:
                raise ValueError("processor 缺少 apply_chat_template。")
            text = chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            rgb_image = image.convert("RGB")
            batch = processor(  # type: ignore[operator]  # Transformers 动态属性
                text=[text],
                images=[rgb_image],
                padding=True,
                return_tensors="pt",
            )
            return batch
        except Exception as exception:
            raise LocalModelInferenceError(_LOCAL_INFERENCE_MESSAGE) from exception

    def _generate_text(self, model: object, batch: object) -> str:
        """执行一次模型生成并提取首个非空文本。"""
        try:
            generate_kwargs = {
                "max_new_tokens": self._max_new_tokens,
                "do_sample": False,
            }
            output_ids = model.generate(  # type: ignore[attr-defined, arg-type]
                **batch, **generate_kwargs
            )
        except Exception as exception:
            raise LocalModelInferenceError(_LOCAL_INFERENCE_MESSAGE) from exception
        try:
            ids_len = batch["input_ids"].shape[1]  # type: ignore[index]  # Tensor 无存根
            generated = output_ids[:, ids_len:]
            processor = self._processor
            if processor is None:
                raise LocalModelOutputError(_LOCAL_OUTPUT_MESSAGE)
            decode = getattr(processor, "batch_decode", None)
            if decode is None:
                raise LocalModelOutputError(_LOCAL_OUTPUT_MESSAGE)
            texts = decode(generated, skip_special_tokens=True)
            if not texts or not isinstance(texts[0], str):
                raise LocalModelOutputError(_LOCAL_OUTPUT_MESSAGE)
            text = texts[0].strip()
            if not text:
                raise LocalModelOutputError(_LOCAL_OUTPUT_MESSAGE)
            return text
        except LocalModelOutputError:
            raise
        except Exception as exception:
            raise LocalModelOutputError(_LOCAL_OUTPUT_MESSAGE) from exception

    def generate(self, image: Image.Image, prompt: str) -> str:
        """使用 Transformers 4-bit 模型生成非空文本。

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
        """
        self._validate_generate_args(image, prompt)
        model, processor, _ = self._load_pipeline()
        batch = self._build_inputs(processor, image, prompt)
        return self._generate_text(model, batch)


@dataclass(frozen=True)
class ModelCallOptions:
    """保存一次模型调用可选的分层消息参数。

    ``None`` 表示不发送对应字段。
    """

    system_prompt: str | None = None
    temperature: float | None = None
    usage_out: dict[str, object] | None = None


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
            api_backend: 可选 API 后端；省略时根据环境配置 DashScope 后端。
            fallback_enabled: 是否允许本地模型加载失败时切换到 API。
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

    def _can_fallback(self, exception: Exception) -> bool:
        """判断异常是否属于允许切换到 API 的模型加载失败。"""
        if not isinstance(exception, LocalModelLoadError):
            return False
        return self._fallback_enabled

    def _generate_from_api(
        self,
        image: Image.Image,
        prompt: str,
        options: ModelCallOptions | None = None,
    ) -> str:
        """执行首次 API 调用和次数有限的失败后重试。"""
        options = options or ModelCallOptions()
        call_extra: dict[str, object] = {}
        if options.system_prompt is not None:
            call_extra["system_prompt"] = options.system_prompt
        if options.temperature is not None:
            call_extra["temperature"] = options.temperature
        if options.usage_out is not None:
            call_extra["usage_out"] = options.usage_out
        for attempt in range(1 + self._max_api_retries):
            try:
                response = self._api_backend.generate(image, prompt, **call_extra)
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
        options: ModelCallOptions | None = None,
    ) -> str:
        """调用选定后端并返回严格文本。

        Args:
            image: 原样传给后端的 PIL 图像。
            prompt: 原样传给后端的非空提示词。
            mode: 首选调用模式。
            options: 可选的 system prompt、temperature 和 usage 输出容器。

        Returns:
            模型后端文本，或固定且脱敏的运行错误信息。

        Raises:
            TypeError: 公共参数类型错误。
            ValueError: 参数值不合法。

        local 模式只在本地模型加载失败且 ``fallback_enabled`` 为 True 时把图像
        与提示词发送到 API；其它本地错误返回固定失败信息，不触发远程调用。
        """
        self._validate_generate_args(image, prompt, mode)
        if mode == "api":
            return self._generate_from_api(image, prompt, options)

        try:
            response = self._local_backend.generate(image, prompt)
            return self._require_text(response)
        except Exception as exception:
            log_safe_exception(logger, _LOCAL_FAILURE_EVENT, exception)
            if self._can_fallback(exception):
                return self._generate_from_api(image, prompt, options)
            return _LOCAL_FAILURE_MESSAGE
