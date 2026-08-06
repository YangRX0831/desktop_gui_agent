"""验证第二周四项模块之间的冻结公共合同。"""

import ast
from pathlib import Path

import pytest
from PIL import Image

from agent.action_parser import ACTION_SYSTEM_PROMPT, parse_action
from agent.dashscope_api_backend import DashScopeAPIBackend
from agent.model_client import ModelClient
from agent.task_manager import TaskManager, TaskStatus
from tests.keyboard_test_support import (
    FakeEnvironment,
    FakeKey,
    _controller,
    fake_environment,
)


class StaticBackend:
    """返回固定文本或固定异常的内存后端。"""

    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.call_count = 0

    def generate(self, image: Image.Image, prompt: str) -> str:
        """返回预设结果。"""
        self.call_count += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome  # type: ignore[return-value]  # fake 可测试合同错误。


@pytest.mark.parametrize(
    "response",
    [
        "Action: click(x=1, y=2)",
        'Action: type(text="文本")',
        'Action: scroll(direction="down", steps=1)',
        'Action: hotkey(key1="ctrl", key2="c")',
        'Action: finish(result="完成")',
    ],
)
def test_prompt_action_examples_match_parser_contract(response: str) -> None:
    """提示词声明的五种规范动作均有可解析实例。"""
    action_name = response.removeprefix("Action: ").split("(", 1)[0]
    assert action_name in ACTION_SYSTEM_PROMPT
    assert parse_action(response) is not None


def test_local_model_to_parser_to_task_manager() -> None:
    """本地文本可沿公共接口写入运行中任务。"""
    image = Image.new("RGB", (1, 1))
    client = ModelClient(StaticBackend("Action: click(x=3, y=4)"))
    action = parse_action(client.generate(image, "prompt"))
    assert action is not None

    manager = TaskManager("task")
    manager.start()
    manager.record_step(action, True)
    state = manager.state
    assert state.status is TaskStatus.RUNNING
    assert state.step_count == 1
    assert state.steps[0].action == action


def test_api_fallback_text_records_type_action() -> None:
    """本地失败后的 API 文本遵守同一动作和状态合同。"""
    local = StaticBackend(RuntimeError("local failure"))
    api = StaticBackend('Action: type(text="你好")')
    text = ModelClient(local, api).generate(
        Image.new("RGB", (1, 1)),
        "prompt",
    )
    action = parse_action(text)
    assert action == {"action_type": "type", "params": {"text": "你好"}}

    manager = TaskManager("task")
    manager.start()
    manager.record_step(action, True)
    assert manager.state.steps[0].action == action


def test_production_api_backend_preserves_model_protocol() -> None:
    """通义千问 production 后端遵守既有统一 generate 接口。"""
    backend = DashScopeAPIBackend(None, None)
    assert callable(backend.generate)


def test_hotkey_action_is_storable_mapping() -> None:
    """hotkey 的 TypedDict 运行时形状可直接交给状态管理器。"""
    action = parse_action(
        'Action: hotkey(key1="ctrl", key2="shift", key3="s")',
    )
    assert action is not None
    manager = TaskManager("task")
    manager.start()
    manager.record_step(action, True)
    assert manager.state.steps[0].action["params"] == {
        "keys": ("ctrl", "shift", "s"),
    }


@pytest.mark.parametrize(
    ("response", "expected_events"),
    [
        (
            'Action: hotkey(key1="enter")',
            [("press", FakeKey.enter), ("release", FakeKey.enter)],
        ),
        (
            'Action: hotkey(key1="ctrl", key2="c")',
            [
                ("press", FakeKey.ctrl),
                ("press", "c"),
                ("release", "c"),
                ("release", FakeKey.ctrl),
            ],
        ),
    ],
)
def test_parsed_hotkey_satisfies_keyboard_controller_contract(
    fake_environment: FakeEnvironment,
    response: str,
    expected_events: list[tuple[str, object]],
) -> None:
    """解析后的单键和组合键均可沿控制层公共接口执行。"""
    action = parse_action(response)
    assert action is not None
    assert action["action_type"] == "hotkey"
    keys = action["params"]["keys"]
    controller = _controller(fake_environment)

    result = controller.hotkey(*keys)

    assert result is None
    assert fake_environment.keyboard.events == expected_events
    assert fake_environment.sleeps == [0.1]


def test_finish_allows_caller_to_succeed_without_control_layer() -> None:
    """finish 由流程调用方解释为成功，不依赖控制模块。"""
    action = parse_action('Action: finish(result="done")')
    assert action is not None
    assert action["action_type"] == "finish"
    manager = TaskManager("task")
    manager.start()
    manager.succeed()
    assert manager.state.status is TaskStatus.SUCCESS
    assert manager.state.step_count == 0


def test_invalid_model_text_does_not_add_task_step() -> None:
    """解析失败时调用方能够保持步骤列表不变。"""
    client = ModelClient(StaticBackend("not an action"))
    manager = TaskManager("task")
    manager.start()
    action = parse_action(
        client.generate(Image.new("RGB", (1, 1)), "prompt"),
    )
    assert action is None
    assert manager.state.step_count == 0


def test_model_error_text_is_not_recorded_as_task_action() -> None:
    """固定运行错误文本不会被解析或写入任务步骤。"""
    client = ModelClient(
        StaticBackend(RuntimeError("failure")),
        fallback_enabled=False,
    )
    manager = TaskManager("task")
    manager.start()
    text = client.generate(Image.new("RGB", (1, 1)), "prompt")
    assert text == "本地模型调用失败。"
    assert not text.startswith("Action:")
    assert parse_action(text) is None
    assert manager.state.step_count == 0


def test_second_week_production_import_graph_has_expected_direction() -> None:
    """第二周模块只允许 ModelClient 依赖 DashScope 实现。"""
    root = Path(__file__).parents[1]
    files = {
        root
        / "agent"
        / "model_client.py": {
            "agent.dashscope_api_backend",
        },
        root / "agent" / "dashscope_api_backend.py": set(),
        root / "agent" / "action_parser.py": set(),
        root / "agent" / "task_manager.py": set(),
    }
    forbidden_roots = {
        "agentscope",
        "transformers",
        "requests",
        "httpx",
        "openai",
        "dashscope",
        "control",
    }
    second_week_modules = {
        "agent.model_client",
        "agent.dashscope_api_backend",
        "agent.action_parser",
        "agent.task_manager",
    }
    for path, allowed_internal in files.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not {name.split(".", 1)[0] for name in imported} & forbidden_roots
        assert imported & second_week_modules == allowed_internal


@pytest.mark.parametrize(
    "relative_path",
    ["agent/__init__.py", "config.py", "main.py"],
)
def test_protected_placeholders_remain_zero_bytes(relative_path: str) -> None:
    """第二周实现没有填充受保护的后续入口文件。"""
    path = Path(__file__).parents[1] / relative_path
    assert path.read_bytes() == b""
