"""测试模型客户端的路由、失败策略和日志隐私。"""

import io
import logging
from pathlib import Path

import pytest
from PIL import Image

from agent.action_parser import ACTION_SYSTEM_PROMPT
from agent.dashscope_api_backend import (
    DashScopeAPIBackend,
    DashScopeAPIConfigurationError,
    DashScopeAPIRetryableError,
)
from agent.model_client import ModelBackend, ModelClient, Qwen2VLLocalBackend


class FakeBackend:
    """按预设结果顺序响应的模型后端。"""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[Image.Image, str]] = []

    def generate(self, image: Image.Image, prompt: str) -> str:
        """记录调用并返回或抛出下一个预设结果。"""
        self.calls.append((image, prompt))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]  # 测试后端故意违反合同。


@pytest.fixture
def image() -> Image.Image:
    """创建不涉及文件或截图的内存图像。"""
    return Image.new("RGB", (2, 2), "white")


@pytest.fixture(autouse=True)
def clear_dashscope_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离用户环境，测试不得读取或使用任何真实 API 配置。"""
    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_API_MODEL",
        "DASHSCOPE_API_ENDPOINT",
        "DASHSCOPE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("retries", [0, 1, 2, 3])
def test_constructor_accepts_valid_retry_counts(retries: int) -> None:
    """允许 PRD 规定的零至三次失败后重试。"""
    ModelClient(FakeBackend(["ok"]), max_api_retries=retries)


@pytest.mark.parametrize("retries", [-1, 4])
def test_constructor_rejects_retry_count_range(retries: int) -> None:
    """拒绝范围外的 API 重试次数。"""
    with pytest.raises(ValueError):
        ModelClient(FakeBackend(["ok"]), max_api_retries=retries)


@pytest.mark.parametrize("retries", [True, False, 1.0, "3", None])
def test_constructor_rejects_non_integer_retry_count(
    retries: object,
) -> None:
    """bool 和其他非 int 类型不能作为重试次数。"""
    with pytest.raises(TypeError):
        ModelClient(
            FakeBackend(["ok"]),
            max_api_retries=retries,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_constructor_rejects_non_boolean_fallback(value: object) -> None:
    """fallback 开关必须严格为 bool。"""
    with pytest.raises(TypeError):
        ModelClient(
            FakeBackend(["ok"]),
            fallback_enabled=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("local", object()),
        ("local", type("BadBackend", (), {"generate": None})()),
        ("api", object()),
    ],
)
def test_constructor_rejects_invalid_backends(
    argument: str,
    value: object,
) -> None:
    """后端必须提供可调用的 generate。"""
    with pytest.raises(TypeError):
        if argument == "local":
            ModelClient(value)  # type: ignore[arg-type]
        else:
            ModelClient(FakeBackend(["ok"]), value)  # type: ignore[arg-type]


def test_protocol_allows_structural_backend() -> None:
    """协议不要求后端继承项目类。"""
    backend: ModelBackend = FakeBackend(["ok"])
    assert callable(backend.generate)


@pytest.mark.parametrize(
    ("image_value", "prompt", "mode", "error_type"),
    [
        (object(), "prompt", "local", TypeError),
        (None, "prompt", "local", TypeError),
        (Image.new("RGB", (1, 1)), 1, "local", TypeError),
        (Image.new("RGB", (1, 1)), "", "local", ValueError),
        (Image.new("RGB", (1, 1)), " \t", "local", ValueError),
        (Image.new("RGB", (1, 1)), "prompt", 1, TypeError),
        (Image.new("RGB", (1, 1)), "prompt", "LOCAL", ValueError),
        (Image.new("RGB", (1, 1)), "prompt", "unknown", ValueError),
    ],
)
def test_generate_validates_before_calling_backend(
    image_value: object,
    prompt: object,
    mode: object,
    error_type: type[Exception],
) -> None:
    """公共参数错误不得触发任何后端。"""
    local = FakeBackend(["local"])
    api = FakeBackend(["api"])
    client = ModelClient(local, api)
    with pytest.raises(error_type):
        client.generate(
            image_value,  # type: ignore[arg-type]
            prompt,  # type: ignore[arg-type]
            mode,  # type: ignore[arg-type]
        )
    assert local.calls == []
    assert api.calls == []


@pytest.mark.parametrize("response", ["result", ""])
def test_local_success_returns_exact_text(
    image: Image.Image,
    response: str,
) -> None:
    """本地后端文本包括空字符串均原样返回。"""
    local = FakeBackend([response])
    api = FakeBackend(["unused"])
    prompt = "  原始提示词  "
    assert ModelClient(local, api).generate(image, prompt) == response
    assert local.calls == [(image, prompt)]
    assert api.calls == []


def test_local_failure_falls_back_once_then_uses_api(
    image: Image.Image,
) -> None:
    """本地失败不计入 API 尝试次数。"""
    local = FakeBackend([RuntimeError("local secret")])
    api = FakeBackend(["api result"])
    result = ModelClient(local, api).generate(image, "prompt")
    assert result == "api result"
    assert len(local.calls) == 1
    assert len(api.calls) == 1


@pytest.mark.parametrize(
    ("fallback_enabled", "has_api"),
    [(False, False), (False, True), (True, False)],
)
def test_local_failure_without_available_fallback_returns_fixed_text(
    image: Image.Image,
    fallback_enabled: bool,
    has_api: bool,
) -> None:
    """无法回退时返回固定文本且不调用 API。"""
    local = FakeBackend([RuntimeError("local sensitive")])
    api = FakeBackend(["unused"]) if has_api else None
    client = ModelClient(local, api, fallback_enabled=fallback_enabled)
    result = client.generate(image, "prompt")
    expected = (
        "API 配置缺失或无效。"
        if fallback_enabled and not has_api
        else "本地模型调用失败。"
    )
    assert result == expected
    assert not result.startswith("Action:")
    assert len(local.calls) == 1
    assert api is None or api.calls == []


def test_non_text_local_response_is_a_failed_call(
    image: Image.Image,
) -> None:
    """本地非文本响应遵循本地失败和回退策略。"""
    local = FakeBackend([{"text": "secret"}])
    api = FakeBackend(["api"])
    assert ModelClient(local, api).generate(image, "prompt") == "api"


def test_api_mode_never_calls_local(image: Image.Image) -> None:
    """显式 API 模式绕过本地后端。"""
    local = FakeBackend([AssertionError("must not call")])
    api = FakeBackend(["api"])
    assert ModelClient(local, api).generate(image, "prompt", "api") == "api"
    assert local.calls == []


def test_missing_api_returns_fixed_error(
    image: Image.Image,
) -> None:
    """API 后端缺失时返回固定错误文本。"""
    client = ModelClient(FakeBackend(["unused"]))
    result = client.generate(image, "prompt", "api")
    assert result == "API 配置缺失或无效。"
    assert not result.startswith("Action:")


@pytest.mark.parametrize("success_attempt", [1, 2, 3, 4])
def test_api_returns_on_first_success(
    image: Image.Image,
    success_attempt: int,
) -> None:
    """API 在首次调用或三次重试内成功后立即停止。"""
    failures = [DashScopeAPIRetryableError("failure")] * (success_attempt - 1)
    api = FakeBackend([*failures, "success", AssertionError("fifth call")])
    client = ModelClient(
        FakeBackend(["unused"]),
        api,
        max_api_retries=3,
    )
    assert client.generate(image, " exact prompt ", "api") == "success"
    assert api.calls == [(image, " exact prompt ")] * success_attempt


@pytest.mark.parametrize("retries", [0, 1, 2, 3])
def test_api_exhaustion_returns_error_at_exact_limit(
    image: Image.Image,
    retries: int,
) -> None:
    """API 耗尽后返回固定文本，且绝不出现第五次调用。"""
    attempts = 1 + retries
    failures = [
        DashScopeAPIRetryableError(f"failure-{index}") for index in range(attempts)
    ]
    api = FakeBackend([*failures, AssertionError("extra call")])
    client = ModelClient(
        FakeBackend(["unused"]),
        api,
        max_api_retries=retries,
    )
    result = client.generate(image, "prompt", "api")
    assert result == "API 模型调用失败。"
    assert not result.startswith("Action:")
    assert len(api.calls) == attempts


def test_non_text_api_response_is_not_retried(image: Image.Image) -> None:
    """API 合同错误不可恢复，不得盲目重试。"""
    api = FakeBackend([None, 1, "success"])
    client = ModelClient(FakeBackend(["unused"]), api)
    assert client.generate(image, "prompt", "api") == "API 模型调用失败。"
    assert len(api.calls) == 1


class StopSignal(BaseException):
    """用于确认进程中止类异常保持传播。"""


@pytest.mark.parametrize("mode", ["local", "api"])
def test_base_exception_is_not_caught(
    image: Image.Image,
    mode: str,
) -> None:
    """模型客户端不捕获 BaseException 子类。"""
    local = FakeBackend([StopSignal()])
    api = FakeBackend([StopSignal()])
    with pytest.raises(StopSignal):
        ModelClient(local, api).generate(
            image,
            "prompt",
            mode,  # type: ignore[arg-type]
        )


def test_safe_log_does_not_expose_sensitive_values(
    image: Image.Image,
) -> None:
    """最终 Formatter 输出只包含安全事件和定位信息。"""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    target_logger = logging.getLogger("agent.model_client")
    old_level = target_logger.level
    target_logger.setLevel(logging.ERROR)
    target_logger.addHandler(handler)
    prompt_marker = "PROMPT_SECRET_MARKER"
    response_marker = "RESPONSE_SECRET_MARKER"
    exception_marker = "EXCEPTION_SECRET_MARKER"
    absolute_marker = "C:\\private\\secret.txt"
    local = FakeBackend(
        [RuntimeError(exception_marker + absolute_marker + prompt_marker)],
    )
    api = FakeBackend(
        [DashScopeAPIRetryableError(response_marker)] * 4,
    )
    try:
        result = ModelClient(local, api).generate(image, prompt_marker)
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(old_level)

    output = stream.getvalue()
    assert output.count("local_model_call_failed") == 1
    assert result == "API 模型调用失败。"
    assert output.count("api_model_call_failed") == 4
    for marker in (
        prompt_marker,
        response_marker,
        exception_marker,
        absolute_marker,
    ):
        assert marker not in output


def test_parameter_error_does_not_log(
    image: Image.Image,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """参数错误不记录模型运行失败。"""
    with caplog.at_level(logging.ERROR, logger="agent.model_client"):
        with pytest.raises(ValueError):
            ModelClient(FakeBackend(["unused"])).generate(image, " ")
    assert caplog.records == []


def test_qwen_backend_satisfies_model_backend_protocol(tmp_path: Path) -> None:
    """真实 Qwen 后端按结构提供统一 generate 接口。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    backend: ModelBackend = Qwen2VLLocalBackend(model_dir)
    assert callable(backend.generate)


def test_qwen_load_error_without_api_configuration_is_safe(
    tmp_path: Path,
) -> None:
    """真实后端加载错误由现有无 API fallback 合同安全处理。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    client = ModelClient(Qwen2VLLocalBackend(model_dir))
    assert client.generate(Image.new("RGB", (1, 1)), "prompt") == (
        "API 配置缺失或无效。"
    )


def test_qwen_load_error_can_fall_back_to_fake_api(tmp_path: Path) -> None:
    """真实后端加载错误可以进入既有 fake API 路由。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    api = FakeBackend(["api result"])
    client = ModelClient(Qwen2VLLocalBackend(model_dir), api)
    image = Image.new("RGB", (1, 1))
    assert client.generate(image, "prompt") == "api result"
    assert api.calls == [(image, "prompt")]


def test_qwen_client_parameter_error_does_not_trigger_api(
    tmp_path: Path,
) -> None:
    """公共参数错误仍在任何 Qwen 加载或 API fallback 之前抛出。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    api = FakeBackend(["unused"])
    client = ModelClient(Qwen2VLLocalBackend(model_dir), api)
    with pytest.raises(ValueError):
        client.generate(Image.new("RGB", (1, 1)), " ")
    assert api.calls == []


def test_qwen_load_error_log_hides_prompt_and_model_path(
    tmp_path: Path,
) -> None:
    """真实后端清单失败的最终日志不含 prompt 或绝对模型路径。"""
    model_dir = tmp_path / "MODEL_PATH_SECRET"
    model_dir.mkdir()
    prompt = "PROMPT_SECRET"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    target_logger = logging.getLogger("agent.model_client")
    target_logger.addHandler(handler)
    try:
        result = ModelClient(Qwen2VLLocalBackend(model_dir)).generate(
            Image.new("RGB", (1, 1)),
            prompt,
        )
    finally:
        target_logger.removeHandler(handler)
    output = stream.getvalue()
    assert result == "API 配置缺失或无效。"
    assert "local_model_call_failed" in output
    assert prompt not in output
    assert str(model_dir) not in output


class FakeHTTPResponse:
    """提供状态码和预设 JSON 的内存 HTTP 响应。"""

    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> object:
        """返回预设 JSON。"""
        return self.payload


class FakeHTTPTransport:
    """记录请求并按顺序返回或抛出预设结果。"""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
        allow_redirects: bool,
    ) -> FakeHTTPResponse:
        """记录完整调用形状，但测试断言绝不输出秘密。"""
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
                "allow_redirects": allow_redirects,
            },
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]  # 故意测试错误响应。


def _success_response(content: object) -> FakeHTTPResponse:
    """构造兼容接口的成功响应。"""
    return FakeHTTPResponse(
        200,
        {"choices": [{"message": {"content": content}}]},
    )


def test_dashscope_backend_builds_multimodal_request(
    image: Image.Image,
) -> None:
    """production 后端构造官方域名、鉴权、图像和完整 prompt。"""
    transport = FakeHTTPTransport([_success_response("Action: finish()")])
    backend = DashScopeAPIBackend(
        "TEST_API_KEY",
        "qwen3.6-flash",
        transport=transport,
    )
    prompt = ACTION_SYSTEM_PROMPT + "\n用户指令"

    assert backend.generate(image, prompt) == "Action: finish()"
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    assert call["headers"] == {
        "Authorization": "Bearer TEST_API_KEY",
        "Content-Type": "application/json",
    }
    assert call["timeout"] == 30.0
    assert call["allow_redirects"] is False
    payload = call["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen3.6-flash"
    assert payload["enable_thinking"] is False
    messages = payload["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert content[0]["image_url"]["url"].startswith(
        "data:image/png;base64,",
    )
    assert content[1] == {"type": "text", "text": prompt}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("text response", "text response"),
        (
            [
                {"type": "text", "text": "Action: "},
                {"type": "text", "text": "finish()"},
            ],
            "Action: finish()",
        ),
    ],
)
def test_dashscope_backend_parses_supported_response_content(
    image: Image.Image,
    content: object,
    expected: str,
) -> None:
    """兼容接口的文本和文本块响应均转换为统一字符串。"""
    transport = FakeHTTPTransport([_success_response(content)])
    backend = DashScopeAPIBackend(
        "TEST_KEY",
        "test-model",
        transport=transport,
    )
    assert backend.generate(image, "prompt") == expected


def test_dashscope_backend_ignores_reasoning_content(
    image: Image.Image,
) -> None:
    """只把最终 content 交给动作解析链，不读取思考过程。"""
    response = FakeHTTPResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "sensitive reasoning",
                        "content": "Action: finish()",
                    },
                },
            ],
        },
    )
    transport = FakeHTTPTransport([response])
    backend = DashScopeAPIBackend(
        "TEST_KEY",
        "qwen3.6-flash",
        transport=transport,
    )

    assert backend.generate(image, "prompt") == "Action: finish()"


def test_dashscope_environment_controls_model_endpoint_and_timeout(
    image: Image.Image,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用户只需设置环境变量，不必修改 Python 源码。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "TEST_KEY")
    monkeypatch.setenv("DASHSCOPE_API_MODEL", "qwen3.6-flash")
    monkeypatch.setenv(
        "DASHSCOPE_API_ENDPOINT",
        "https://token-plan.cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1/chat/completions",
    )
    monkeypatch.setenv("DASHSCOPE_TIMEOUT_SECONDS", "12.5")
    transport = FakeHTTPTransport([_success_response("ok")])
    backend = DashScopeAPIBackend.from_env(transport=transport)
    assert backend.generate(image, "prompt") == "ok"
    assert transport.calls[0]["timeout"] == 12.5
    assert transport.calls[0]["url"] == (
        "https://token-plan.cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1/chat/completions"
    )
    assert transport.calls[0]["json"]["model"] == "qwen3.6-flash"


@pytest.mark.parametrize(
    "hostname",
    [
        "dashscope.aliyuncs.com",
        "token-plan.cn-beijing.maas.aliyuncs.com",
    ],
)
def test_exact_endpoint_allowlist_accepts_approved_hosts(
    image: Image.Image,
    hostname: str,
) -> None:
    """只有两个精确批准的主机可以接收鉴权和多模态请求。"""
    transport = FakeHTTPTransport([_success_response("ok")])
    endpoint = f"https://{hostname}/compatible-mode/v1/chat/completions"
    backend = DashScopeAPIBackend(
        "TEST_KEY",
        "qwen3.6-flash",
        endpoint=endpoint,
        transport=transport,
    )

    assert backend.generate(image, "prompt") == "ok"
    assert transport.calls[0]["url"] == endpoint
    assert transport.calls[0]["allow_redirects"] is False


@pytest.mark.parametrize(
    "endpoint",
    [
        (
            "https://other.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1/chat/completions"
        ),
        (
            "http://token-plan.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1/chat/completions"
        ),
        "https://token-plan.cn-beijing.maas.aliyuncs.com/wrong/path",
        (
            "https://token-plan.cn-beijing.maas.aliyuncs.com:443/"
            "compatible-mode/v1/chat/completions"
        ),
    ],
)
def test_endpoint_allowlist_rejects_unapproved_variants_before_transport(
    image: Image.Image,
    endpoint: str,
) -> None:
    """模糊主机、HTTP、错误路径和替代 netloc 均在发送前被拒绝。"""
    transport = FakeHTTPTransport([AssertionError("must not call")])
    backend = DashScopeAPIBackend(
        "TEST_KEY",
        "qwen3.6-flash",
        endpoint=endpoint,
        transport=transport,
    )

    with pytest.raises(DashScopeAPIConfigurationError):
        backend.generate(image, "prompt")
    assert transport.calls == []


def test_missing_credentials_fail_safely_without_transport(
    image: Image.Image,
) -> None:
    """无凭据时明确失败，且在 transport 前停止。"""
    transport = FakeHTTPTransport([AssertionError("must not call")])
    backend = DashScopeAPIBackend(None, None, transport=transport)
    client = ModelClient(FakeBackend(["unused"]), backend)
    assert client.generate(image, "prompt", "api") == "API 配置缺失或无效。"
    assert transport.calls == []


def test_invalid_api_environment_does_not_block_local_success(
    image: Image.Image,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未使用 API 时，错误 API 配置不得破坏本地成功路径。"""
    monkeypatch.setenv("DASHSCOPE_TIMEOUT_SECONDS", "invalid")
    local = FakeBackend(["local result"])
    assert ModelClient(local).generate(image, "prompt") == "local result"
    assert len(local.calls) == 1


def test_invalid_api_environment_fails_safely_in_api_mode(
    image: Image.Image,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """选择 API 时，错误外部配置返回固定非敏感错误。"""
    monkeypatch.setenv("DASHSCOPE_TIMEOUT_SECONDS", "INVALID_SECRET")
    result = ModelClient(FakeBackend(["unused"])).generate(
        image,
        "prompt",
        "api",
    )
    assert result == "API 配置缺失或无效。"
    assert "INVALID_SECRET" not in result


def test_invalid_endpoint_fails_before_sending_secret(
    image: Image.Image,
) -> None:
    """非官方域名不得接收 Authorization 或图像。"""
    transport = FakeHTTPTransport([AssertionError("must not call")])
    backend = DashScopeAPIBackend(
        "TEST_KEY",
        "test-model",
        endpoint="https://example.invalid/chat/completions",
        transport=transport,
    )
    client = ModelClient(FakeBackend(["unused"]), backend)
    assert client.generate(image, "prompt", "api") == "API 配置缺失或无效。"
    assert transport.calls == []


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_retryable_http_status_uses_exact_retry_limit(
    image: Image.Image,
    status_code: int,
) -> None:
    """仅临时 HTTP 状态进入 ModelClient 有限重试。"""
    transport = FakeHTTPTransport(
        [
            FakeHTTPResponse(status_code, {"sensitive": "ignored"}),
            _success_response("ok"),
        ],
    )
    backend = DashScopeAPIBackend("TEST_KEY", "test-model", transport=transport)
    client = ModelClient(FakeBackend(["unused"]), backend, max_api_retries=1)
    assert client.generate(image, "prompt", "api") == "ok"
    assert len(transport.calls) == 2


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_non_retryable_http_status_stops_immediately(
    image: Image.Image,
    status_code: int,
) -> None:
    """鉴权和请求错误不得盲目重试。"""
    transport = FakeHTTPTransport(
        [
            FakeHTTPResponse(status_code, {"secret": "raw response"}),
            AssertionError("must not retry"),
        ],
    )
    backend = DashScopeAPIBackend("TEST_KEY", "test-model", transport=transport)
    client = ModelClient(FakeBackend(["unused"]), backend)
    assert client.generate(image, "prompt", "api") == "API 模型调用失败。"
    assert len(transport.calls) == 1


def test_transport_timeout_is_retryable(image: Image.Image) -> None:
    """连接超时可有限重试，成功后立即停止。"""
    transport = FakeHTTPTransport(
        [TimeoutError("sensitive timeout"), _success_response("ok")],
    )
    backend = DashScopeAPIBackend("TEST_KEY", "test-model", transport=transport)
    client = ModelClient(FakeBackend(["unused"]), backend, max_api_retries=1)
    assert client.generate(image, "prompt", "api") == "ok"
    assert len(transport.calls) == 2


def test_dashscope_backend_never_retries_by_itself(
    image: Image.Image,
) -> None:
    """provider 后端只执行一次请求，重试唯一归 ModelClient 所有。"""
    transport = FakeHTTPTransport(
        [FakeHTTPResponse(503, {}), AssertionError("backend retried")],
    )
    backend = DashScopeAPIBackend("TEST_KEY", "test-model", transport=transport)
    with pytest.raises(DashScopeAPIRetryableError):
        backend.generate(image, "prompt")
    assert len(transport.calls) == 1


def test_invalid_response_is_not_retried(image: Image.Image) -> None:
    """响应结构错误属于不可恢复错误。"""
    transport = FakeHTTPTransport(
        [FakeHTTPResponse(200, {"raw": "sensitive"}), _success_response("ok")],
    )
    backend = DashScopeAPIBackend("TEST_KEY", "test-model", transport=transport)
    client = ModelClient(FakeBackend(["unused"]), backend)
    assert client.generate(image, "prompt", "api") == "API 模型调用失败。"
    assert len(transport.calls) == 1


def test_api_secret_and_raw_response_never_enter_log_or_error(
    image: Image.Image,
) -> None:
    """最终 Formatter 输出和固定错误均不泄漏 API 敏感值。"""
    key_marker = "API_KEY_SECRET_MARKER"
    prompt_marker = "PROMPT_SECRET_MARKER"
    response_marker = "RAW_RESPONSE_SECRET_MARKER"
    transport = FakeHTTPTransport(
        [RuntimeError(key_marker + response_marker + prompt_marker)],
    )
    backend = DashScopeAPIBackend(key_marker, "test-model", transport=transport)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    target_logger = logging.getLogger("agent.model_client")
    old_level = target_logger.level
    target_logger.setLevel(logging.ERROR)
    target_logger.addHandler(handler)
    try:
        result = ModelClient(FakeBackend(["unused"]), backend).generate(
            image,
            prompt_marker,
            "api",
        )
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(old_level)
    output = stream.getvalue()
    assert result == "API 模型调用失败。"
    assert "api_model_call_failed" in output
    for marker in (key_marker, prompt_marker, response_marker, "Authorization"):
        assert marker not in output


def test_invalid_timeout_environment_is_a_safe_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法 timeout 不含原值地映射为配置异常。"""
    monkeypatch.setenv("DASHSCOPE_TIMEOUT_SECONDS", "SECRET_INVALID_TIMEOUT")
    with pytest.raises(DashScopeAPIConfigurationError) as error:
        DashScopeAPIBackend.from_env()
    assert str(error.value) == "API 配置缺失或无效。"
    assert "SECRET_INVALID_TIMEOUT" not in str(error.value)
