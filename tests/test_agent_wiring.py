"""production wiring、配置与 API transport 测试。"""

import asyncio

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


@pytest.mark.parametrize(
    "response",
    [
        "API 模型调用失败。",
        "API 配置缺失或无效。",
    ],
)
def test_terminal_api_failure_does_not_restart_retry_budget(
    response: str,
) -> None:
    """ModelClient 已终止的 API 失败不在 Agent 层重复重试。"""
    from agent.task_manager import TaskManager, TaskStatus

    backend = SequenceBackend([response, 'Action: finish(result="x")'])
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


def test_config_defaults_match_prd() -> None:
    """AppConfig 默认值与 PRD 4.4.1 一致。"""
    c = AppConfig()
    assert c.max_steps == 10
    assert c.retry_count == 3
    assert c.model_mode == "local"
    assert c.coordinate_mode == "normalized_1000"


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


def test_main_config_from_arguments(monkeypatch) -> None:
    """config_from_arguments 正确转换参数。"""
    from pathlib import Path

    from main import config_from_arguments, create_argument_parser

    monkeypatch.setenv("GUI_AGENT_LOCAL_MODEL_DIR", "/tmp/model")
    monkeypatch.setenv("GUI_AGENT_COORDINATE_MODE", "image_pixel")
    parser = create_argument_parser()
    args = parser.parse_args(["--model-mode", "api", "--max-steps", "5"])
    config = config_from_arguments(args)
    assert config.model_mode == "api"
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

    class FakeAPI:
        @classmethod
        def from_env(cls, **kw):
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
    )
    agent = main_module.build_production_agent(config, lambda *_: None)

    # 验证 wiring 传递的配置。
    assert agent._settings.max_steps == 7
    assert agent._settings.retry_count == 2
    assert agent._settings.model_mode == "local"
    assert agent._settings.coordinate_mode == "normalized_1000"
    assert agent._dependencies.action_dispatcher._coordinate_mode == "normalized_1000"
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
