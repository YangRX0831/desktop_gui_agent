"""提供 OpenAI-compatible 多模态 API 后端。

职责：
    将一张 Pillow 截图和一个 prompt 编码为固定的多模态 HTTP 请求，并从
    响应中提取唯一文本。后端只做一次同步请求，重试属于 ``ModelClient``。

配置约束：
    endpoint、模型名和 API key 在实例构造时保存，但在真正 generate 前
    才验证完整性。模块导入和 ``from_env`` 不发起网络请求。

网络约束：
    重定向被禁止，响应体大小有上限，并区分可重试 transport/服务端失败与
    不可重试配置、认证和协议失败，防止无界请求或错误目标漂移。

隐私边界：
    截图会发送给用户显式配置的 API 服务，因此调用权限由运行任务
    管理。本模块不把 key、prompt、图片、响应正文或异常正文写入日志。
"""

import base64
import importlib
import io
import os
from typing import Protocol
from urllib.parse import urlparse

from PIL import Image

from utils.run_diagnostics import diag_log, diag_phase

DASHSCOPE_CONFIGURATION_MESSAGE = "API 配置缺失或无效。"

_API_KEY_ENV = "DASHSCOPE_API_KEY"
_MODEL_ENV = "DASHSCOPE_API_MODEL"
_ENDPOINT_ENV = "DASHSCOPE_API_ENDPOINT"
_TIMEOUT_ENV = "DASHSCOPE_TIMEOUT_SECONDS"
_DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_TIMEOUT_SECONDS = 120.0
_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class DashScopeAPIError(RuntimeError):
    """表示通义千问开放平台调用失败。

    Attributes:
        异常仅携带固定安全描述，不应包含响应正文或凭据。

    ``ModelClient`` 通过具体子类判断是否允许有限重试。
    """


class DashScopeAPIConfigurationError(DashScopeAPIError):
    """表示开放平台外部配置缺失或不安全。

    Attributes:
        配置值本身不会存入异常正文。

    该错误不可重试；调用方返回固定配置提示且不发送请求。
    """


class DashScopeAPIRetryableError(DashScopeAPIError):
    """表示可进行有限重试的临时 API 故障。

    Attributes:
        类别来自 transport 异常或明确服务端状态，不含响应正文。

    ``ModelClient`` 捕获后按 max_api_retries 再次调用后端。
    """


class DashScopeAPINonRetryableError(DashScopeAPIError):
    """表示不应重试的请求、鉴权或响应故障。

    Attributes:
        仅保存安全错误类别，不保存 API key、请求或响应数据。

    该错误立即停止，避免无效请求、成本放大或重复上传截图。
    """


class HTTPResponse(Protocol):
    """定义 API transport 返回值的最小合同。

    Attributes:
        status_code: HTTP 状态码。

    典型实现是 requests Response；后端只使用状态码和 ``json``。
    """

    status_code: int

    def json(self) -> object:
        """返回已解析的 JSON 响应。"""


class HTTPTransport(Protocol):
    """定义可注入的同步 HTTP transport。

    Attributes:
        transport 自行管理连接状态；后端不读取其私有属性。

    production 使用 requests session 兼容对象，测试使用内存 fake。
    """

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
        allow_redirects: bool,
    ) -> HTTPResponse:
        """发送一次 HTTP POST。"""


class DashScopeAPIBackend:
    """通过通义千问开放平台兼容接口执行多模态请求。

    Attributes:
        api_key: 只用于 Authorization header，不写入日志。
        model: 固定多模态模型名。
        endpoint: 经过 HTTPS 与主机约束的服务 URL。
        timeout_seconds: 受上限约束的同步请求超时。
        transport: 可注入的单请求 HTTP 客户端。

    典型用法是 ``from_env`` 构造，再由 ``ModelClient`` 管理有限重试。
    """

    def __init__(
        self,
        api_key: str | None,
        model: str | None,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        supports_thinking_options: bool = False,
        transport: HTTPTransport | None = None,
    ) -> None:
        """保存外部 API 配置并验证基础参数，不发送网络请求。

        Args:
            api_key: 通义千问 API 密钥，可省略。
            model: 模型名称，可省略。
            endpoint: 兼容接口端点地址。
            timeout_seconds: 单次请求超时秒数。
            enable_thinking: thinking 三态覆盖;None 表示不发送。
            thinking_budget: 可选正整数 token budget。
            supports_thinking_options: provider 是否支持 thinking 字段。
            transport: 可注入的同步 HTTP transport；省略时使用 requests。

        Raises:
            TypeError: 参数类型不合法。
            ValueError: 参数值不合法。

        构造阶段只保存并验证基础参数，不产生网络请求，也不持久化密钥。
        """
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("api_key 必须是 str 或 None。")
        if model is not None and not isinstance(model, str):
            raise TypeError("model 必须是 str 或 None。")
        if not isinstance(endpoint, str):
            raise TypeError("endpoint 必须是 str。")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(
            timeout_seconds,
            bool,
        ):
            raise TypeError("timeout_seconds 必须是数值。")
        if not 0 < float(timeout_seconds) <= _MAX_TIMEOUT_SECONDS:
            raise ValueError("timeout_seconds 必须在 0 到 120 之间。")
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise TypeError("enable_thinking 必须是 bool 或 None。")
        if thinking_budget is not None:
            if type(thinking_budget) is not int:
                raise TypeError("thinking_budget 必须是 int 或 None。")
            if thinking_budget <= 0:
                raise ValueError("thinking_budget 必须大于 0。")
        if type(supports_thinking_options) is not bool:
            raise TypeError("supports_thinking_options 必须是 bool。")
        if transport is not None and not callable(
            getattr(transport, "post", None),
        ):
            raise TypeError("transport 必须提供可调用的 post。")

        self._api_key = api_key.strip() if api_key is not None else None
        self._model = model.strip() if model is not None else None
        self._endpoint = endpoint.strip()
        self._timeout_seconds = float(timeout_seconds)
        self._enable_thinking = enable_thinking
        self._thinking_budget = thinking_budget
        self._supports_thinking_options = supports_thinking_options
        self._transport = transport

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        supports_thinking_options: bool = False,
        transport: HTTPTransport | None = None,
    ) -> "DashScopeAPIBackend":
        """从环境变量创建后端，不读取配置文件或持久化秘密。

        Args:
            model: 配置层提供的模型名;None 时兼容读取既有模型环境变量。
            enable_thinking: 配置层提供的 thinking 三态覆盖。
            thinking_budget: 配置层提供的可选 thinking budget。
            supports_thinking_options: provider capability 配置。
            transport: 可注入的同步 HTTP transport；省略时使用 requests。

        Returns:
            DashScopeAPIBackend 实例。

        Raises:
            DashScopeAPIConfigurationError: 环境变量值无法转换为合法配置。

        本方法只读取指定的环境变量，不读取配置文件，不持久化 secret，
        本身不发送网络请求。
        """
        timeout_text = os.environ.get(_TIMEOUT_ENV)
        timeout_seconds = _DEFAULT_TIMEOUT_SECONDS
        if timeout_text is not None:
            try:
                timeout_seconds = float(timeout_text)
            except ValueError as exception:
                raise DashScopeAPIConfigurationError(
                    DASHSCOPE_CONFIGURATION_MESSAGE,
                ) from exception
        try:
            return cls(
                os.environ.get(_API_KEY_ENV),
                model if model is not None else os.environ.get(_MODEL_ENV),
                endpoint=os.environ.get(_ENDPOINT_ENV, _DEFAULT_ENDPOINT),
                timeout_seconds=timeout_seconds,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                supports_thinking_options=supports_thinking_options,
                transport=transport,
            )
        except (TypeError, ValueError) as exception:
            raise DashScopeAPIConfigurationError(
                DASHSCOPE_CONFIGURATION_MESSAGE,
            ) from exception

    @staticmethod
    def _validate_generate_args(image: Image.Image, prompt: str) -> None:
        """在编码图像或发送网络请求前验证参数。"""
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL.Image.Image。")
        if not isinstance(prompt, str):
            raise TypeError("prompt 必须是 str。")
        if not prompt.strip():
            raise ValueError("prompt 不得为空。")

    def _validate_configuration(self) -> None:
        """验证凭据、模型和 OpenAI-compatible HTTPS endpoint。"""
        parsed = urlparse(self._endpoint)
        try:
            port = parsed.port
        except ValueError as exception:
            raise DashScopeAPIConfigurationError(
                DASHSCOPE_CONFIGURATION_MESSAGE,
            ) from exception
        if (
            not self._api_key
            or not self._model
            or parsed.scheme != "https"
            or not parsed.hostname
            or port not in {None, 443}
            or not parsed.path.endswith(_CHAT_COMPLETIONS_SUFFIX)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise DashScopeAPIConfigurationError(
                DASHSCOPE_CONFIGURATION_MESSAGE,
            )

    @staticmethod
    def _image_data_url(image: Image.Image) -> str:
        """把内存图像编码为兼容请求所需的 PNG data URL。"""
        buffer = io.BytesIO()
        try:
            image.convert("RGB").save(buffer, format="PNG")
        except Exception as exception:
            raise DashScopeAPINonRetryableError(
                "API 图像编码失败。",
            ) from exception
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _payload(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, object]:
        """构造单轮消息;分层协议使用 system/user 消息,文本先于图像。

        system_prompt 为 None 时保持 V1 合同:单 user 消息、图像在前、
        无采样参数,确保 V1 A/B 基线逐字节不变。
        """
        if system_prompt is None:
            user_content: list[dict[str, object]] = [
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image)},
                },
                {"type": "text", "text": prompt},
            ]
            messages: list[dict[str, object]] = [
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(image)},
                        },
                    ],
                },
            ]
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self._supports_thinking_options:
            if self._enable_thinking is not None:
                payload["enable_thinking"] = self._enable_thinking
            if self._thinking_budget is not None:
                payload["thinking_budget"] = self._thinking_budget
        return payload

    @staticmethod
    def _extract_text(payload: object) -> str:
        """从兼容响应中提取非空文本，不暴露原始响应。"""
        try:
            if not isinstance(payload, dict):
                raise TypeError("invalid payload")
            choices = payload["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError("invalid choices")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError("invalid choice")
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("invalid message")
            content = message["content"]
            if isinstance(content, str) and content.strip():
                return content
            if not isinstance(content, list):
                raise TypeError("invalid content")
            parts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].strip()
            ]
            if not parts:
                raise TypeError("missing text")
            return "".join(parts)
        except (KeyError, TypeError) as exception:
            raise DashScopeAPINonRetryableError(
                "API 响应格式无效。",
            ) from exception

    @staticmethod
    def _is_transport_retryable(exception: Exception) -> bool:
        """识别内置及 requests 的临时连接或超时错误。"""
        if isinstance(exception, (TimeoutError, ConnectionError)):
            return True
        try:
            requests_module = importlib.import_module("requests")
            exceptions = getattr(requests_module, "exceptions")
            retryable = (
                getattr(exceptions, "Timeout"),
                getattr(exceptions, "ConnectionError"),
            )
        except Exception:
            return False
        return isinstance(exception, retryable)

    def _post(self, payload: dict[str, object]) -> HTTPResponse:
        """通过注入 transport 或现有 requests 依赖发送一次请求。"""
        transport = self._transport
        if transport is None:
            try:
                requests_module = importlib.import_module("requests")
                session_factory = getattr(requests_module, "Session", None)
                if not callable(session_factory):
                    raise TypeError("requests Session unavailable")
                transport = session_factory()
                self._transport = transport
            except Exception as exception:
                raise DashScopeAPINonRetryableError(
                    "API transport 不可用。",
                ) from exception
        try:
            # OBSERVABILITY_ONLY:模型调用边界计时;挂起时表现为
            # diag_model_begin 出现而 diag_model_end 缺失(in-flight)。
            with diag_phase(
                "diag_model",
                model=self._model,
                timeout_seconds=self._timeout_seconds,
            ):
                response = transport.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout_seconds,
                    allow_redirects=False,
                )
            diag_log(
                "diag_model_response",
                status_category=(str(getattr(response, "status_code", "?"))[:1] + "xx"),
            )
        except Exception as exception:
            error_type = (
                DashScopeAPIRetryableError
                if self._is_transport_retryable(exception)
                else DashScopeAPINonRetryableError
            )
            raise error_type("API transport 调用失败。") from exception
        return response

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        usage_out: dict[str, object] | None = None,
    ) -> str:
        """执行一次通义千问开放平台请求并返回统一文本。

        Args:
            image: 待发送的 PIL 图像。
            prompt: 非空提示词文本。
            system_prompt: 分层协议的固定 system 文本;None 表示 V1 单消息。
            temperature: 分层协议显式采样温度;None 表示不发送该参数。
            usage_out: 可选字典;API 响应含 usage 时原样填入,不含时不写入。

        Returns:
            API 返回的非空文本。

        Raises:
            TypeError: image 或 prompt 类型不合法。
            ValueError: prompt 为空。
            DashScopeAPIConfigurationError: 配置缺失或不满足安全约束。
            DashScopeAPIRetryableError: 临时性 API 故障。
            DashScopeAPINonRetryableError: 不可重试的请求或响应错误。

        配置与图像编码通过后，本方法会向经验证的 HTTPS endpoint 发送
        至多一次同步请求；重定向被禁用。backend 自身不进行 retry，
        重试策略由上层 ModelClient 负责。
        """
        self._validate_generate_args(image, prompt)
        self._validate_configuration()
        response = self._post(
            self._payload(image, prompt, system_prompt, temperature),
        )
        status_code = getattr(response, "status_code", None)
        if status_code in _RETRYABLE_HTTP_STATUS:
            raise DashScopeAPIRetryableError("API 暂时不可用。")
        if not isinstance(status_code, int) or not 200 <= status_code < 300:
            raise DashScopeAPINonRetryableError("API 请求被拒绝。")
        try:
            payload = response.json()
        except Exception as exception:
            raise DashScopeAPINonRetryableError(
                "API 响应格式无效。",
            ) from exception
        if usage_out is not None and isinstance(payload, dict):
            usage = payload.get("usage")
            if isinstance(usage, dict):
                # 兼容 input_tokens 与 prompt_tokens 两种键名,统一为
                # input/output/cached 供 trace 使用,不估算缺失项。
                usage_out.update(usage)
                details = usage.get("prompt_tokens_details")
                usage_out["input_tokens"] = usage.get(
                    "input_tokens",
                    usage.get("prompt_tokens", "unknown"),
                )
                usage_out["output_tokens"] = usage.get(
                    "output_tokens",
                    usage.get("completion_tokens", "unknown"),
                )
                usage_out["cached_tokens"] = (
                    details.get("cached_tokens", "unknown")
                    if isinstance(details, dict)
                    else usage.get("cached_tokens", "unknown")
                )
        return self._extract_text(payload)
