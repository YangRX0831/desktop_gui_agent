"""提供本地与 API 多模态后端之间的统一调用边界。

职责:
    向 Agent 暴露统一 ``generate`` 接口，并把本地后端、DashScope 后端及
    有限 API transport retry 隔离在同一个责任边界内。

重试约束:
    API 失败允许首次请求后的最多三次重试。调用方不得围绕本客户端再做
    模型调用重试，否则实际网络请求会产生乘法放大。不可重试异常立即停止。

本地模型约束:
    production 采用 PRD 3.3/4.3.1 规定的 Transformers + Qwen2-VL-2B-Instruct
    + 4-bit 量化加载；模型权重位于仓库外，构造阶段只验证路径，权在首次
    generate 延迟加载。历史 OpenVINO 实现已不再作为 production baseline。

fallback 行为（PRD 4.3.1）:
    PRD 4.3.1 规定 local 模式下模型加载失败时自动切换到 API 模式;本模块
    据此在 ``LocalModelLoadError`` 时自动调用 API 后端,前提是构造时
    ``fallback_enabled`` 为 True。触发范围仅限 ``LocalModelLoadError``,
    不把任意本地推理异常扩大为 fallback trigger。

隐私与异常:
    图像和 prompt 仅传给显式选择的后端。日志只包含固定事件、异常类型和
    安全代码位置，不记录原始输入、响应、异常正文或模型目录内容。
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

    PRD 4.3.1 的 model-load-failure -> API fallback capability 仅以此类异常
    作为触发条件；本地推理失败不得归入此类。ModelClient 据此区分是否允许
    进入批准的 API fallback。
    """


class LocalModelInferenceError(RuntimeError):
    """表示本地模型图像处理或推理失败。

    该错误不属于 model-load 失败，因此不得触发 local->API fallback；
    本地后端不自行重试，避免隐藏推理成本和重复工作。
    """


class LocalModelOutputError(RuntimeError):
    """表示本地模型没有返回可用的非空文本。

    该错误不能通过宽松 Parser 或字符串 salvage 恢复，也不触发 fallback。
    """


class ModelBackend(Protocol):
    """定义模型后端必须提供的最小同步接口。

    production 实现为本地 Qwen 或 DashScope；fake 用于无外部副作用测试。
    """

    def generate(self, image: Image.Image, prompt: str) -> str:
        """根据图像和提示词生成文本。"""


class Qwen2VLLocalBackend:
    """通过 Transformers + 4-bit 量化加载 Qwen2-VL-2B-Instruct。

    Attributes:
        model_dir: 已验证的仓库外 Qwen2-VL-2B-Instruct 模型目录。
        max_new_tokens: 单次生成的输出 token 上限。
        model: 延迟加载的 Transformers 多模态生成模型。
        processor: 延迟加载的 Transformers 多模态 processor。

    production 通常构造一次再串行调用。模型保持延迟加载，构造阶段不读取
    权重，也不触发 ``from_pretrained``。
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
            ValueError: 配置值越界，或模型目录不存在/不是目录。
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
        """核对模型目录含 Qwen2-VL 必要文件，不加载权重。"""
        config_path = self._model_dir / "config.json"
        if not config_path.is_file():
            raise ValueError("model_dir 缺少 config.json。")
        try:
            config = self._read_config(config_path)
        except Exception as exception:
            raise ValueError("model_dir config.json 无效。") from exception
        if not isinstance(config, dict):
            raise ValueError("model_dir config.json 结构无效。")
        # 仅校验 model_type，不绑定特定 Transformers 版本字段集，保持最小必要。
        if config.get("model_type") != "qwen2_vl":
            raise ValueError("model_dir 不是 qwen2_vl 模型。")
        if not (self._model_dir / "preprocessor_config.json").is_file():
            raise ValueError("model_dir 缺少 preprocessor_config.json。")
        if not (self._model_dir / "tokenizer.json").is_file():
            raise ValueError("model_dir 缺少 tokenizer.json。")

    def _load_pipeline(self) -> tuple[object, object, object]:
        """延迟加载 Transformers 模型、processor 与模块句柄。

        Raises:
            LocalModelLoadError: 模型或 processor 加载失败。该异常类型被
                ModelClient 识别为 PRD model-load-failure，是触发 fallback 的
                唯一条件。
        """
        if self._model is not None and self._processor is not None:
            return self._model, self._processor, self._transformers_module
        try:
            self._validate_model_dir()
            transformers_module = importlib.import_module("transformers")
            auto_processor = getattr(transformers_module, "AutoProcessor")
            model_cls = getattr(transformers_module, "Qwen2VLForConditionalGeneration")
            bnb_config_cls = getattr(transformers_module, "BitsAndBytesConfig")
            # PRD 4.3.1 4-bit 量化加载合同：BitsAndBytesConfig(load_in_4bit=True)
            # 通过 bitsandbytes 把权重加载为 4-bit,降低显存占用。
            # transformers 5.x 不再接受 load_in_4bit 直接参数,必须通过
            # quantization_config 传递。local_files_only=True 防止缺文件时
            # 隐式联网下载。config.json model_type=qwen2_vl、无 auto_map,
            # 不需要 trust_remote_code。
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
        """把 PIL 图像与 prompt 转换为模型输入；失败归为推理错误。"""
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
            LocalModelLoadError: 本地模型加载失败（可触发批准的 API fallback）。
            LocalModelInferenceError: 图像处理或推理失败（不触发 fallback）。
            LocalModelOutputError: 模型未返回可用的非空文本（不触发 fallback）。

        首次调用在需要时延迟加载本地模型 pipeline，后续调用复用。
        """
        self._validate_generate_args(image, prompt)
        model, processor, _ = self._load_pipeline()
        batch = self._build_inputs(processor, image, prompt)
        return self._generate_text(model, batch)


@dataclass(frozen=True)
class ModelCallOptions:
    """一次模型调用的分层消息协议可选请求配置。

    对应旧 ``call_extra`` 事实承担的三个可选透传字段;None 表示
    不发送该字段(保持 V1 单消息合同)。
    """

    system_prompt: str | None = None
    temperature: float | None = None
    usage_out: dict[str, object] | None = None


class ModelClient:
    """统一执行本地模型调用、API 回退和有限 API 重试。

    Attributes:
        local_backend: 首选的本地模型实现，仅在 local 模式调用。
        api_backend: 显式 API 模式或 PRD 4.3.1 自动 fallback 时使用的实现。
        fallback_enabled: 是否启用 PRD model-load-failure -> API capability。
        max_api_retries: 首次 API 失败后的最多重试次数。

    fallback 按 PRD 4.3.1 自动触发:local 模式下 ``LocalModelLoadError``
    且 ``fallback_enabled`` 为 True 时,无需任何 run 级授权即切换到
    API 后端;其余本地异常不触发 fallback。
    """

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
            fallback_enabled: 是否保留 PRD model-load-failure -> API capability。
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
        """判断当前异常是否属于 PRD model-load-failure 且 capability 开启。"""
        if not isinstance(exception, LocalModelLoadError):
            return False
        return self._fallback_enabled

    def _generate_from_api(
        self,
        image: Image.Image,
        prompt: str,
        options: ModelCallOptions | None = None,
    ) -> str:
        """执行首次 API 调用和次数有限的失败后重试;透传消息协议参数。"""
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
            options: 分层消息协议可选配置(system/temperature/usage);None 保持
                V1 单消息合同。

        Returns:
            模型后端文本，或固定且脱敏的运行错误信息。

        Raises:
            TypeError: 公共参数类型错误。
            ValueError: 参数值不合法。

        local 模式下，仅当本地模型 *load* 失败且 ``fallback_enabled`` 为
        True 时，按 PRD 4.3.1 自动把图像与 prompt 切换到远程 API；其余本地
        失败一律返回固定本地失败信息，不发送任何远程数据。
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
