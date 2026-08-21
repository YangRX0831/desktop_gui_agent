"""production wiring、配置与 API transport 测试。"""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from agentscope.message import Msg
from PIL import Image

from agent.dashscope_api_backend import (
    DashScopeAPIBackend,
    DashScopeAPIConfigurationError,
    DashScopeAPIRetryableError,
)
from agent.task_manager import TaskManager
from config import AppConfig
from tests.agent_test_support import (
    MemoryControls,
    SequenceBackend,
    _FakeResponse,
    _FakeTransport,
    _make_api_backend,
    make_agent,
)


def test_config_failure_terminates_task_without_retry_budget_restart() -> None:
    """配置缺失属不可重试不变式:立即终态,不消耗 max_steps(C3)。"""
    from agent.task_manager import TaskManager, TaskStatus

    backend = SequenceBackend(["API 配置缺失或无效。", 'Action: finish(result="x")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(
            Msg("u", "任务", "user"),
        ),
    )

    assert result.content == "任务执行失败：模型调用失败。"
    assert manager.state.status is TaskStatus.FAILED
    assert backend.calls == 1


def test_api_retry_exhaustion_continues_next_step() -> None:
    """PRD 4.5.1:API 重试耗尽记录错误并继续下一步,不立即 fail(C1)。"""
    from agent.task_manager import TaskManager, TaskStatus

    backend = SequenceBackend(["API 模型调用失败。", 'Action: finish(result="x")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(
            Msg("u", "任务", "user"),
        ),
    )

    # step1 记录模型失败后前进;step2 重新感知并成功 finish。
    assert result.content == "x"
    assert manager.state.status is TaskStatus.SUCCESS
    assert backend.calls == 2
    step1 = manager.state.steps[0]
    assert step1.result is False
    assert step1.attempts[0].stage == "model"


def test_config_defaults_match_prd() -> None:
    """AppConfig 默认值与 PRD 4.4.1 一致。"""
    c = AppConfig()
    assert c.max_steps == 10
    assert c.retry_count == 3
    assert c.model_mode == "local"
    assert c.coordinate_mode == "normalized_1000"
    assert c.api_model is None
    assert c.api_enable_thinking is False
    assert c.api_thinking_budget is None
    assert c.api_thinking_options_supported is True


def test_config_rejects_invalid_max_steps() -> None:
    """非法 max_steps 在构造时失败。"""
    with pytest.raises(ValueError):
        AppConfig(max_steps=0)
    with pytest.raises(ValueError):
        AppConfig(max_steps=-1)
    with pytest.raises(TypeError):
        AppConfig(max_steps=3.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AppConfig(max_steps=True)


def test_config_rejects_invalid_retry_count() -> None:
    """非法 retry_count 在构造时失败。"""
    with pytest.raises(ValueError):
        AppConfig(retry_count=4)
    with pytest.raises(ValueError):
        AppConfig(retry_count=-1)
    with pytest.raises(TypeError):
        AppConfig(retry_count="3")  # type: ignore[arg-type]


def test_config_rejects_invalid_model_mode() -> None:
    """非法 model_mode 在构造时失败。"""
    with pytest.raises(ValueError):
        AppConfig(model_mode="invalid")  # type: ignore[arg-type]


def test_config_rejects_invalid_coordinate_mode() -> None:
    """坐标模式只接受显式白名单值。"""
    with pytest.raises(ValueError):
        AppConfig(coordinate_mode="auto")  # type: ignore[arg-type]


def test_config_accepts_valid_custom_values() -> None:
    """合法自定义值通过验证。"""
    from pathlib import Path

    c = AppConfig(
        model_mode="api",
        max_steps=5,
        retry_count=2,
        log_level="warning",
        log_dir=Path("/tmp/test"),
    )
    assert c.model_mode == "api"
    assert c.max_steps == 5
    assert c.retry_count == 2
    assert c.log_level == "WARNING"  # normalized


def test_config_invalid_log_level_rejected() -> None:
    """非法 log_level 在构造时失败。"""
    with pytest.raises(ValueError):
        AppConfig(log_level="VERBOSE")


def test_config_local_model_dir_from_env(monkeypatch) -> None:
    """local_model_dir_from_env 正确读取/忽略环境变量。"""
    from pathlib import Path

    from config import local_model_dir_from_env

    monkeypatch.delenv("GUI_AGENT_LOCAL_MODEL_DIR", raising=False)
    assert local_model_dir_from_env() is None
    monkeypatch.setenv("GUI_AGENT_LOCAL_MODEL_DIR", "  ")
    assert local_model_dir_from_env() is None
    monkeypatch.setenv("GUI_AGENT_LOCAL_MODEL_DIR", "/some/path")
    result = local_model_dir_from_env()
    assert result is not None
    assert isinstance(result, Path)


def test_config_coordinate_mode_from_env(monkeypatch) -> None:
    """坐标模式环境变量支持默认值、规范化和非法值拒绝。"""
    from config import coordinate_mode_from_env

    monkeypatch.delenv("GUI_AGENT_COORDINATE_MODE", raising=False)
    assert coordinate_mode_from_env() == "normalized_1000"
    monkeypatch.setenv("GUI_AGENT_COORDINATE_MODE", " IMAGE_PIXEL ")
    assert coordinate_mode_from_env() == "image_pixel"
    monkeypatch.setenv("GUI_AGENT_COORDINATE_MODE", "auto")
    with pytest.raises(ValueError):
        coordinate_mode_from_env()


def test_config_bool_rejected_for_int_fields() -> None:
    """bool 不能用于 int 字段(Python bool 是 int 子类但项目拒绝)。"""
    with pytest.raises(TypeError):
        AppConfig(max_steps=True)
    with pytest.raises(TypeError):
        AppConfig(retry_count=False)


def test_config_rejects_invalid_api_runtime_options() -> None:
    """API runtime options 在创建 backend 前完成严格类型和值域校验。"""
    with pytest.raises(ValueError):
        AppConfig(api_model=" ")
    with pytest.raises(TypeError):
        AppConfig(api_enable_thinking="false")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AppConfig(api_thinking_budget=0)
    with pytest.raises(TypeError):
        AppConfig(api_thinking_budget=True)
    with pytest.raises(TypeError):
        AppConfig(api_thinking_options_supported=1)  # type: ignore[arg-type]


def test_config_api_runtime_options_from_env(monkeypatch) -> None:
    """模型、thinking 三态、budget 与 capability 由统一配置层读取。"""
    from config import (
        api_enable_thinking_from_env,
        api_model_from_env,
        api_thinking_budget_from_env,
        api_thinking_options_supported_from_env,
    )

    monkeypatch.setenv("DASHSCOPE_API_MODEL", " model-a ")
    monkeypatch.delenv("GUI_AGENT_API_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("GUI_AGENT_API_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("GUI_AGENT_API_THINKING_OPTIONS_SUPPORTED", raising=False)
    assert api_model_from_env() == "model-a"
    assert api_enable_thinking_from_env() is False
    assert api_thinking_budget_from_env() is None
    assert api_thinking_options_supported_from_env() is True

    monkeypatch.setenv("GUI_AGENT_API_ENABLE_THINKING", "true")
    monkeypatch.setenv("GUI_AGENT_API_THINKING_BUDGET", "128")
    monkeypatch.setenv("GUI_AGENT_API_THINKING_OPTIONS_SUPPORTED", "false")
    assert api_enable_thinking_from_env() is True
    assert api_thinking_budget_from_env() == 128
    assert api_thinking_options_supported_from_env() is False

    monkeypatch.setenv("GUI_AGENT_API_ENABLE_THINKING", "none")
    assert api_enable_thinking_from_env() is None


def test_main_config_from_arguments(monkeypatch) -> None:
    """config_from_arguments 正确转换参数。"""
    from pathlib import Path

    from main import config_from_arguments, create_argument_parser

    monkeypatch.setenv("GUI_AGENT_LOCAL_MODEL_DIR", "/tmp/model")
    monkeypatch.setenv("GUI_AGENT_COORDINATE_MODE", "image_pixel")
    monkeypatch.setenv("DASHSCOPE_API_MODEL", "model-a")
    monkeypatch.delenv("GUI_AGENT_API_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("GUI_AGENT_API_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("GUI_AGENT_API_THINKING_OPTIONS_SUPPORTED", raising=False)
    parser = create_argument_parser()
    args = parser.parse_args(["--model-mode", "api", "--max-steps", "5"])
    config = config_from_arguments(args)
    assert config.model_mode == "api"
    assert config.api_model == "model-a"
    assert config.api_enable_thinking is False
    assert config.api_thinking_budget is None
    assert config.api_thinking_options_supported is True
    assert config.max_steps == 5
    assert config.local_model_dir == Path("/tmp/model")
    assert config.coordinate_mode == "image_pixel"


def test_main_build_production_agent_wiring(monkeypatch) -> None:
    """build_production_agent 正确连接依赖,不产生真实副作用。"""
    from pathlib import Path

    import main as main_module

    # monkeypatch 源模块(build_production_agent 内做延迟 import)。
    class FakeMouse:
        def click(self, *a, **kw):
            pass

        def right_click(self, *a, **kw):
            pass

        def double_click(self, *a, **kw):
            pass

        def drag_from_to(self, *a, **kw):
            pass

    class FakeKeyboard:
        def type(self, *a, **kw):
            pass

        def scroll(self, *a, **kw):
            pass

        def hotkey(self, *a, **kw):
            pass

    class FakeLocal:
        def __init__(self, model_dir, **kw):
            pass

        def generate(self, image, prompt):
            return "ok"

    api_runtime_options: dict[str, object] = {}

    class FakeAPI:
        @classmethod
        def from_env(cls, **kw):
            api_runtime_options.update(kw)
            return cls()

        def generate(self, image, prompt):
            return "ok"

    monkeypatch.setattr("control.mouse_controller.MouseController", FakeMouse)
    monkeypatch.setattr(
        "control.keyboard_controller.KeyboardController",
        FakeKeyboard,
    )
    monkeypatch.setattr("agent.model_client.Qwen2VLLocalBackend", FakeLocal)
    monkeypatch.setattr(
        "agent.dashscope_api_backend.DashScopeAPIBackend",
        FakeAPI,
    )

    class FakeOCR:
        def recognize(self, image):
            return []

    monkeypatch.setattr("perception.ocr_recognizer.OCRRecognizer", FakeOCR)

    config = AppConfig(
        model_mode="local",
        max_steps=7,
        retry_count=2,
        local_model_dir=Path("/tmp/model"),
        coordinate_mode="normalized_1000",
        api_model="configured-model",
        api_enable_thinking=False,
        api_thinking_budget=128,
        api_thinking_options_supported=True,
    )
    # 决策协议开关经环境变量进入 production settings;V3 之前漏接导致
    # --agent-protocol v3 实际静默运行 V1,此处固化两条 env→settings 通路。
    monkeypatch.setenv("GUI_AGENT_DECISION_PROTOCOL_V2", "0")
    monkeypatch.setenv("GUI_AGENT_DECISION_PROTOCOL_V3", "1")
    monkeypatch.setenv("GUI_AGENT_HIDE_OWN_WINDOW_DURING_RUN", "1")
    agent = main_module.build_production_agent(config, lambda *_: None)

    # 验证 wiring 传递的配置。
    assert agent._settings.max_steps == 7
    assert agent._settings.retry_count == 2
    assert agent._settings.model_mode == "local"
    assert agent._settings.coordinate_mode == "normalized_1000"
    assert agent._settings.decision_protocol_v2 is False
    assert agent._settings.decision_protocol_v3 is True
    assert agent._settings.hide_own_window_during_run is True
    assert agent._dependencies.action_dispatcher._coordinate_mode == "normalized_1000"
    assert api_runtime_options == {
        "model": "configured-model",
        "enable_thinking": False,
        "thinking_budget": 128,
        "supports_thinking_options": True,
    }

    # 2026-08-19 baseline 切换:环境变量全部未设置时研发默认协议为
    # CLEAN V3;显式 V1(=0)/V2(=1 且 V3=0)/V3(=1)仍完全可选。
    # 同日起 SEMANTIC EXECUTION 未设置时随 V3 默认启用;显式 0 关闭;
    # 显式 V1/V2 保持旧行为不启用。
    from config import (
        decision_protocol_v2_from_env,
        decision_protocol_v3_from_env,
        semantic_execution_from_env,
    )

    monkeypatch.delenv("GUI_AGENT_DECISION_PROTOCOL_V2", raising=False)
    monkeypatch.delenv("GUI_AGENT_DECISION_PROTOCOL_V3", raising=False)
    monkeypatch.delenv("GUI_AGENT_SEMANTIC_EXECUTION", raising=False)
    default_agent = main_module.build_production_agent(config, lambda *_: None)
    assert default_agent._settings.decision_protocol_v3 is True
    assert default_agent._settings.decision_protocol_v2 is False
    assert default_agent._settings.semantic_execution is True
    assert decision_protocol_v3_from_env() is True
    assert semantic_execution_from_env() is None
    monkeypatch.setenv("GUI_AGENT_SEMANTIC_EXECUTION", "0")
    assert default_agent is not None  # 显式关闭仍可构造
    semantic_off = main_module.build_production_agent(config, lambda *_: None)
    assert semantic_off._settings.semantic_execution is False
    monkeypatch.delenv("GUI_AGENT_SEMANTIC_EXECUTION", raising=False)
    monkeypatch.setenv("GUI_AGENT_DECISION_PROTOCOL_V3", "0")
    explicit_v1 = main_module.build_production_agent(config, lambda *_: None)
    assert explicit_v1._settings.decision_protocol_v3 is False
    assert explicit_v1._settings.semantic_execution is False
    assert decision_protocol_v3_from_env() is False
    monkeypatch.setenv("GUI_AGENT_DECISION_PROTOCOL_V2", "1")
    explicit_v2 = main_module.build_production_agent(config, lambda *_: None)
    assert explicit_v2._settings.decision_protocol_v2 is True
    assert explicit_v2._settings.semantic_execution is False
    assert decision_protocol_v2_from_env() is True
    monkeypatch.setenv("GUI_AGENT_DECISION_PROTOCOL_V3", "1")
    explicit_v3 = main_module.build_production_agent(config, lambda *_: None)
    assert explicit_v3._settings.decision_protocol_v3 is True
    assert explicit_v3._settings.semantic_execution is True
    assert agent._dependencies.action_dispatcher is not None
    assert agent._dependencies.model_client is not None
    assert agent._dependencies.protect_initial_foreground is True
    assert agent._dependencies.ocr_recognizer is not None


def test_api_backend_generate_success() -> None:
    """API 后端正常生成文本。"""
    resp = _FakeResponse(
        200,
        {
            "choices": [{"message": {"content": "hello"}}],
        },
    )
    transport = _FakeTransport([resp])
    backend = _make_api_backend(transport)
    result = backend.generate(Image.new("RGB", (2, 2)), "prompt")
    assert result == "hello"
    assert len(transport.calls) == 1
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_api_backend_retryable_status() -> None:
    """429/503 状态码抛出 retryable 异常。"""
    for status in (429, 503):
        resp = _FakeResponse(status, {})
        transport = _FakeTransport([resp])
        backend = _make_api_backend(transport)
        with pytest.raises(DashScopeAPIRetryableError):
            backend.generate(Image.new("RGB", (2, 2)), "p")


def test_api_backend_non_retryable_status() -> None:
    """400 状态码抛出 non-retryable 异常。"""
    resp = _FakeResponse(400, {})
    transport = _FakeTransport([resp])
    backend = _make_api_backend(transport)
    with pytest.raises(Exception) as exc_info:
        backend.generate(Image.new("RGB", (2, 2)), "p")
    assert not isinstance(exc_info.value, DashScopeAPIRetryableError)


def test_api_backend_config_error() -> None:
    """缺失 api_key 抛出 configuration error。"""
    backend = DashScopeAPIBackend(None, "model")
    with pytest.raises(DashScopeAPIConfigurationError):
        backend.generate(Image.new("RGB", (2, 2)), "p")


def test_api_backend_content_list_extraction() -> None:
    """content 为 list 时正确提取 text block。"""
    resp = _FakeResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "part1"},
                            {"type": "text", "text": "part2"},
                        ]
                    }
                }
            ],
        },
    )
    transport = _FakeTransport([resp])
    backend = _make_api_backend(transport)
    result = backend.generate(Image.new("RGB", (2, 2)), "p")
    assert result == "part1part2"


def test_api_backend_invalid_json_response() -> None:
    """json() 抛异常时返回 non-retryable。"""

    class _BadJsonResponse(_FakeResponse):
        def json(self):
            raise ValueError("bad json")

    resp = _BadJsonResponse(200, None)
    transport = _FakeTransport([resp])
    backend = _make_api_backend(transport)
    with pytest.raises(Exception) as exc_info:
        backend.generate(Image.new("RGB", (2, 2)), "p")
    assert not isinstance(exc_info.value, DashScopeAPIRetryableError)


def test_api_backend_from_env_missing_key(monkeypatch) -> None:
    """from_env 无 key 时接受但不验证(验证在 generate 时)。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_MODEL", raising=False)
    backend = DashScopeAPIBackend.from_env()
    # generate 时因 config 缺失抛出 configuration error
    with pytest.raises(DashScopeAPIConfigurationError):
        backend.generate(Image.new("RGB", (2, 2)), "p")


def test_api_backend_payload_structure() -> None:
    """payload 仅包含跨供应商兼容的标准字段。"""
    resp = _FakeResponse(
        200,
        {
            "choices": [{"message": {"content": "ok"}}],
        },
    )
    transport = _FakeTransport([resp])
    backend = _make_api_backend(transport)
    backend.generate(Image.new("RGB", (2, 2)), "test prompt")
    payload = transport.calls[0]["json"]
    assert payload["model"] == "test-model"
    assert "enable_thinking" not in payload
    assert payload["stream"] is False
    assert len(payload["messages"]) == 1
    message = payload["messages"][0]
    assert message["role"] == "user"
    content = message["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "test prompt"


def test_default_api_transport_reuses_one_session(monkeypatch) -> None:
    """默认 transport 在多次模型调用间复用同一个 HTTP Session。"""
    responses = [
        _FakeResponse(200, {"choices": [{"message": {"content": "one"}}]}),
        _FakeResponse(200, {"choices": [{"message": {"content": "two"}}]}),
    ]
    transport = _FakeTransport(responses)
    state = {"session_calls": 0}

    class _RequestsModule:
        @staticmethod
        def Session():
            state["session_calls"] += 1
            return transport

    from agent import dashscope_api_backend as backend_module

    monkeypatch.setattr(
        backend_module.importlib,
        "import_module",
        lambda name: _RequestsModule(),
    )
    backend = DashScopeAPIBackend("key", "model")
    assert backend.generate(Image.new("RGB", (2, 2)), "p1") == "one"
    assert backend.generate(Image.new("RGB", (2, 2)), "p2") == "two"
    assert state["session_calls"] == 1
    assert len(transport.calls) == 2


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_api_backend_sends_explicit_thinking_override(
    enable_thinking: bool,
) -> None:
    """false/true 均按配置发送，不依据模型名称猜测。"""
    transport = _FakeTransport(
        [_FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})],
    )
    backend = DashScopeAPIBackend(
        "key",
        "arbitrary-vision-model",
        enable_thinking=enable_thinking,
        supports_thinking_options=True,
        transport=transport,
    )

    backend.generate(Image.new("RGB", (2, 2)), "test prompt")

    assert transport.calls[0]["json"]["enable_thinking"] is enable_thinking


def test_api_backend_omits_none_thinking_override() -> None:
    """None 保留 provider/model 默认行为，不发送 override。"""
    transport = _FakeTransport(
        [_FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})],
    )
    backend = DashScopeAPIBackend(
        "key",
        "qwen3.6-flash",
        enable_thinking=None,
        supports_thinking_options=True,
        transport=transport,
    )

    backend.generate(Image.new("RGB", (2, 2)), "test prompt")

    assert "enable_thinking" not in transport.calls[0]["json"]


def test_api_backend_propagates_configured_thinking_budget() -> None:
    """thinking_budget 只在配置明确给出且 provider 支持时发送。"""
    transport = _FakeTransport(
        [_FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})],
    )
    backend = DashScopeAPIBackend(
        "key",
        "arbitrary-vision-model",
        enable_thinking=True,
        thinking_budget=128,
        supports_thinking_options=True,
        transport=transport,
    )

    backend.generate(Image.new("RGB", (2, 2)), "test prompt")

    payload = transport.calls[0]["json"]
    assert payload["enable_thinking"] is True
    assert payload["thinking_budget"] == 128


def test_api_backend_omits_unsupported_thinking_options() -> None:
    """第三方/provider capability 关闭时不接收 thinking 专用字段。"""
    transport = _FakeTransport(
        [_FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})],
    )
    backend = DashScopeAPIBackend(
        "key",
        "qwen3.6-flash",
        endpoint="https://example.com/v1/chat/completions",
        enable_thinking=False,
        thinking_budget=128,
        supports_thinking_options=False,
        transport=transport,
    )

    backend.generate(Image.new("RGB", (2, 2)), "test prompt")

    payload = transport.calls[0]["json"]
    assert "enable_thinking" not in payload
    assert "thinking_budget" not in payload


def test_api_backend_has_no_model_name_special_case() -> None:
    """同一显式配置对任意模型名生成相同 runtime options。"""
    payloads = []
    for model in ("qwen3.6-flash", "unrelated-vision-model"):
        transport = _FakeTransport(
            [_FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})],
        )
        backend = DashScopeAPIBackend(
            "key",
            model,
            enable_thinking=False,
            supports_thinking_options=True,
            transport=transport,
        )
        backend.generate(Image.new("RGB", (2, 2)), "test prompt")
        payloads.append(transport.calls[0]["json"])

    assert [payload["enable_thinking"] for payload in payloads] == [False, False]


def test_api_backend_validate_args() -> None:
    """generate 参数验证在发送前执行。"""
    backend = _make_api_backend(_FakeTransport([]))
    with pytest.raises(TypeError):
        backend.generate("not_image", "p")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        backend.generate(Image.new("RGB", (2, 2)), "")


def test_api_backend_init_validation() -> None:
    """构造器验证基础参数。"""
    with pytest.raises(TypeError):
        DashScopeAPIBackend("key", "model", endpoint=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DashScopeAPIBackend(
            "key",
            "model",
            timeout_seconds=0,
        )
    with pytest.raises(ValueError):
        DashScopeAPIBackend(
            "key",
            "model",
            timeout_seconds=200,
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://api.siliconflow.cn/v1/chat/completions",
        "https://example.com/custom/chat/completions",
        "https://example.com:443/v1/chat/completions",
    ],
)
def test_api_backend_accepts_https_chat_completion_endpoints(
    endpoint: str,
) -> None:
    """后端接受用户配置的 HTTPS OpenAI-compatible endpoint。"""
    transport = _FakeTransport(
        [_FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})],
    )
    backend = DashScopeAPIBackend(
        "key",
        "model",
        endpoint=endpoint,
        transport=transport,
    )

    assert backend.generate(Image.new("RGB", (2, 2)), "p") == "ok"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.example.com/v1/chat/completions",
        "https://user:password@example.com/v1/chat/completions",
        "https://example.com/v1/chat/completions?target=other",
        "https://example.com/v1/chat/completions#fragment",
        "https://example.com/v1/responses",
        "https://example.com:8443/v1/chat/completions",
        "https://example.com:invalid/v1/chat/completions",
    ],
)
def test_api_backend_rejects_unsafe_or_incompatible_endpoints(
    endpoint: str,
) -> None:
    """不安全或非 chat-completions 地址在请求前被拒绝。"""
    transport = _FakeTransport([])
    backend = DashScopeAPIBackend(
        "key",
        "model",
        endpoint=endpoint,
        transport=transport,
    )

    with pytest.raises(DashScopeAPIConfigurationError):
        backend.generate(Image.new("RGB", (2, 2)), "p")
    assert transport.calls == []


def test_api_backend_transport_exception_retryable() -> None:
    """transport TimeoutError 归类为 retryable。"""
    import requests as req_mod

    class _FailingTransport:
        def post(self, url, **kw):
            raise req_mod.exceptions.Timeout("timeout")

    backend = DashScopeAPIBackend(
        "test-key",
        "test-model",
        transport=_FailingTransport(),
    )
    with pytest.raises(DashScopeAPIRetryableError):
        backend.generate(Image.new("RGB", (2, 2)), "p")


def test_api_backend_transport_exception_non_retryable() -> None:
    """transport 普通异常归类为 non-retryable。"""

    class _FailingTransport:
        def post(self, url, **kw):
            raise RuntimeError("generic error")

    backend = DashScopeAPIBackend(
        "test-key",
        "test-model",
        transport=_FailingTransport(),
    )
    with pytest.raises(Exception) as exc_info:
        backend.generate(Image.new("RGB", (2, 2)), "p")
    assert not isinstance(exc_info.value, DashScopeAPIRetryableError)


def test_api_backend_missing_content_rejected() -> None:
    """content 缺失时抛出 non-retryable。"""
    resp = _FakeResponse(200, {"choices": [{"message": {}}]})
    transport = _FakeTransport([resp])
    backend = _make_api_backend(transport)
    with pytest.raises(Exception) as exc_info:
        backend.generate(Image.new("RGB", (2, 2)), "p")
    assert not isinstance(exc_info.value, DashScopeAPIRetryableError)


def test_api_backend_choices_empty_rejected() -> None:
    """choices 为空时抛出 non-retryable。"""
    resp = _FakeResponse(200, {"choices": []})
    transport = _FakeTransport([resp])
    backend = _make_api_backend(transport)
    with pytest.raises(Exception) as exc_info:
        backend.generate(Image.new("RGB", (2, 2)), "p")
    assert not isinstance(exc_info.value, DashScopeAPIRetryableError)


def test_result_message_contains_statistics() -> None:
    """成功消息携带步数/耗时/重试 metadata。"""
    from agent.gui_agent import GuiAgent

    manager = TaskManager("t")
    manager.start()
    manager.record_step({"action_type": "finish", "params": {}}, True, 1)
    manager.record_retry(1)
    manager.succeed()
    message = GuiAgent._result_message("done", manager)
    assert message.metadata["steps"] == 1
    assert message.metadata["retries"] == 1
    assert message.metadata["duration_seconds"] >= 0.0


def test_cli_formats_statistics_line() -> None:
    """CLI 把 metadata 统计格式化为单行输出。"""
    from main import _format_statistics

    line = _format_statistics(
        {"steps": 3, "duration_seconds": 12.5, "retries": 1},
    )
    assert line == "[统计] 步数=3 耗时(秒)=12.5 重试=1"
    assert _format_statistics(None) == ""
    assert _format_statistics({}) == ""


def test_openvino_backend_validates_model_dir(tmp_path) -> None:
    """OpenVINO 后端校验导出目录与参数,不在构造时加载模型。"""
    from agent.openvino_backend import Qwen2VLOpenVINOBackend

    with pytest.raises(TypeError, match="model_dir"):
        Qwen2VLOpenVINOBackend(123)
    with pytest.raises(ValueError, match="不存在"):
        Qwen2VLOpenVINOBackend(tmp_path / "missing")
    empty = tmp_path / "export"
    empty.mkdir()
    with pytest.raises(ValueError, match="openvino_language_model.xml"):
        Qwen2VLOpenVINOBackend(empty)
    with pytest.raises(ValueError, match="max_new_tokens"):
        Qwen2VLOpenVINOBackend(empty, max_new_tokens=0)

    marker = empty / "openvino_language_model.xml"
    marker.write_text("stub", encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        Qwen2VLOpenVINOBackend(empty, device=" ")


def test_openvino_backend_uses_single_image_vlm_contract(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenVINO VLM 调用包含视觉占位符并传入单张 image。"""
    from agent.openvino_backend import Qwen2VLOpenVINOBackend

    model_dir = tmp_path / "export"
    model_dir.mkdir()
    (model_dir / "openvino_language_model.xml").write_text(
        "stub",
        encoding="utf-8",
    )
    calls = []

    class FakePipeline:
        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return 'Action: finish(result="ok")'

    class FakeTensor:
        def __init__(self, array) -> None:
            self.array = array

    class FakeGenerationConfig:
        def __init__(self, **kwargs) -> None:
            self.options = kwargs

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Tensor=FakeTensor))
    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        SimpleNamespace(GenerationConfig=FakeGenerationConfig),
    )
    backend = Qwen2VLOpenVINOBackend(model_dir)
    backend._pipeline = FakePipeline()

    result = backend.generate(Image.new("RGB", (4, 3)), "用户任务")

    assert result == 'Action: finish(result="ok")'
    assert calls[0][0].startswith(
        "<|vision_start|><|image_pad|><|vision_end|>\n",
    )
    assert calls[0][0].endswith("用户任务")
    assert "image" in calls[0][1]
    assert "images" not in calls[0][1]


def test_local_runtime_config_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """本地运行时环境变量校验与默认 transformers 基线。"""
    from config import local_runtime_from_env, openvino_model_dir_from_env

    monkeypatch.delenv("GUI_AGENT_LOCAL_RUNTIME", raising=False)
    assert local_runtime_from_env() == "transformers"
    monkeypatch.setenv("GUI_AGENT_LOCAL_RUNTIME", "OpenVINO")
    assert local_runtime_from_env() == "openvino"
    monkeypatch.setenv("GUI_AGENT_LOCAL_RUNTIME", "tensorrt")
    with pytest.raises(ValueError, match="LOCAL_RUNTIME"):
        local_runtime_from_env()
    monkeypatch.delenv("GUI_AGENT_OPENVINO_MODEL_DIR", raising=False)
    assert openvino_model_dir_from_env() is None
    monkeypatch.setenv(
        "GUI_AGENT_OPENVINO_MODEL_DIR",
        r"C:\AI\OpenVINO\export",
    )
    assert openvino_model_dir_from_env() is not None


def test_repeated_api_exhaustion_fails_only_at_max_steps() -> None:
    """PRD 4.5.1:反复 API 耗尽不提前 fail,仅在 max_steps 处终态(C2)。"""
    from agent.task_manager import TaskManager, TaskStatus

    backend = SequenceBackend(["API 模型调用失败。"] * 3)
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, max_steps=3)(
            Msg("u", "任务", "user"),
        ),
    )
    assert manager.state.status is TaskStatus.FAILED
    assert "任务达到最大执行步数" in result.content
    assert backend.calls == 3
    assert all(step.result is False for step in manager.state.steps)


def test_local_image_max_dim_default_and_env(monkeypatch) -> None:
    """P5:local 模式图像长边上限默认 640,env 可覆盖并做下限校验。"""
    from config import local_model_image_max_dim_from_env

    monkeypatch.delenv("GUI_AGENT_LOCAL_IMAGE_MAX_DIM", raising=False)
    assert local_model_image_max_dim_from_env() == 640
    monkeypatch.setenv("GUI_AGENT_LOCAL_IMAGE_MAX_DIM", "512")
    assert local_model_image_max_dim_from_env() == 512
    monkeypatch.setenv("GUI_AGENT_LOCAL_IMAGE_MAX_DIM", "100")
    try:
        local_model_image_max_dim_from_env()
        raise AssertionError("应抛 ValueError")
    except ValueError:
        pass
