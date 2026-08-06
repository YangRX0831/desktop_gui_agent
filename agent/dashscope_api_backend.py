"""提供通义千问开放平台多模态 API 后端。"""

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
_CHAT_COMPLETIONS_PATH = "/compatible-mode/v1/chat/completions"
# 只允许官方域名与固定路径，避免 API Key 和图像被发送到任意 endpoint；
# 请求层同时禁用重定向作为第二层防护。
_ALLOWED_ENDPOINT_HOSTNAMES = frozenset(
    {
        "dashscope.aliyuncs.com",
        "token-plan.cn-beijing.maas.aliyuncs.com",
    },
)
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_TIMEOUT_SECONDS = 120.0
_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class DashScopeAPIError(RuntimeError):
    """表示通义千问开放平台调用失败。"""


class DashScopeAPIConfigurationError(DashScopeAPIError):
    """表示通义千问开放平台外部配置缺失或不安全。"""


class DashScopeAPIRetryableError(DashScopeAPIError):
    """表示可进行有限重试的临时 API 故障。"""


class DashScopeAPINonRetryableError(DashScopeAPIError):
    """表示不应重试的 API 请求、鉴权或响应故障。"""


class HTTPResponse(Protocol):
    """定义 API transport 返回值的最小合同。"""

    status_code: int

    def json(self) -> object:
        """返回已解析的 JSON 响应。"""


class HTTPTransport(Protocol):
    """定义可注入的同步 HTTP transport。"""

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
    """通过通义千问开放平台兼容接口执行多模态请求。"""

    def __init__(
        self,
        api_key: str | None,
        model: str | None,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: HTTPTransport | None = None,
    ) -> None:
        """保存外部 API 配置，不发送请求。"""
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
        """从环境变量创建后端，不读取配置文件或持久化秘密。"""
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
        """验证凭据、模型和仅限官方域名的 HTTPS endpoint。"""
        parsed = urlparse(self._endpoint)
        if (
            not self._api_key
            or not self._model
            or parsed.scheme != "https"
            or parsed.hostname not in _ALLOWED_ENDPOINT_HOSTNAMES
            or parsed.netloc != parsed.hostname
            or parsed.path != _CHAT_COMPLETIONS_PATH
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
            "enable_thinking": False,
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
        """执行一次通义千问开放平台请求并返回统一文本。"""
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
