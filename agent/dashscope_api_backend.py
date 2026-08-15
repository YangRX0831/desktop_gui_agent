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
        transport: HTTPTransport | None = None,
    ) -> None:
        """保存外部 API 配置并验证基础参数，不发送网络请求。

        Args:
            api_key: 通义千问 API 密钥，可省略。
            model: 模型名称，可省略。
            endpoint: 兼容接口端点地址。
            timeout_seconds: 单次请求超时秒数。
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
        if transport is not None and not callable(
            getattr(transport, "post", None),
        ):
            raise TypeError("transport 必须提供可调用的 post。")

        self._api_key = api_key.strip() if api_key is not None else None
        self._model = model.strip() if model is not None else None
        self._endpoint = endpoint.strip()
        self._timeout_seconds = float(timeout_seconds)
        self._transport = transport

    @classmethod
    def from_env(
        cls,
        *,
        transport: HTTPTransport | None = None,
    ) -> "DashScopeAPIBackend":
        """从环境变量创建后端，不读取配置文件或持久化秘密。

        Args:
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
                os.environ.get(_MODEL_ENV),
                endpoint=os.environ.get(_ENDPOINT_ENV, _DEFAULT_ENDPOINT),
                timeout_seconds=timeout_seconds,
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

    def _payload(self, image: Image.Image, prompt: str) -> dict[str, object]:
        """构造单轮图像与文本消息，保持现有完整 prompt 不变。"""
        return {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(image)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "stream": False,
        }

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
                transport = importlib.import_module("requests")
            except Exception as exception:
                raise DashScopeAPINonRetryableError(
                    "API transport 不可用。",
                ) from exception
        try:
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
        except Exception as exception:
            error_type = (
                DashScopeAPIRetryableError
                if self._is_transport_retryable(exception)
                else DashScopeAPINonRetryableError
            )
            raise error_type("API transport 调用失败。") from exception
        return response

    def generate(self, image: Image.Image, prompt: str) -> str:
        """执行一次通义千问开放平台请求并返回统一文本。

        Args:
            image: 待发送的 PIL 图像。
            prompt: 非空提示词文本。

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
        response = self._post(self._payload(image, prompt))
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
        return self._extract_text(payload)
