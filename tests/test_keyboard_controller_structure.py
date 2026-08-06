"""测试键盘模块结构、动态导入、构造复用与依赖隔离。

结构检查不创建真实 pynput Controller 或 Listener。
"""

import ast
import inspect
import sys
from collections import defaultdict
from pathlib import Path

from control import keyboard_controller
from tests.keyboard_test_support import (
    FakeEnvironment,
    _controller,
    _source,
    fake_environment,
)


class FakeKey:
    """提供批准命名键的内存值。"""


class FakeKeyboardBackend:
    """记录键盘操作并支持按调用位置注入异常。"""

    def __init__(self, timeline: list[tuple[object, ...]] | None = None) -> None:
        self.events: list[tuple[str, object]] = []
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.failures: dict[tuple[str, int], Exception] = {}
        self.timeline = timeline

    def fail_on(self, operation: str, call_number: int, exc: Exception) -> None:
        self.failures[(operation, call_number)] = exc

    def press(self, key: object) -> None:
        self._record("press", key)

    def release(self, key: object) -> None:
        self._record("release", key)

    def _record(self, operation: str, key: object) -> None:
        self.counts[operation] += 1
        self.events.append((operation, key))
        if self.timeline is not None:
            self.timeline.append(("keyboard", operation, key))
        failure = self.failures.get((operation, self.counts[operation]))
        if failure is not None:
            raise failure


class FakeScrollBackend:
    """记录滚动操作并支持注入异常。"""

    def __init__(self) -> None:
        self.events: list[tuple[int, int]] = []
        self.failure: Exception | None = None

    def scroll(self, dx: int, dy: int) -> None:
        self.events.append((dx, dy))
        if self.failure is not None:
            raise self.failure


class FakeTextBackend:
    """记录 UTF-16 文本批次并支持按调用位置注入异常。"""

    def __init__(self, timeline: list[tuple[object, ...]] | None = None) -> None:
        self.batches: list[tuple[int, ...]] = []
        self.failures: dict[int, Exception] = {}
        self.timeline = timeline

    def fail_on(self, call_number: int, exc: Exception) -> None:
        self.failures[call_number] = exc

    def send(self, code_units: tuple[int, ...]) -> None:
        self.batches.append(code_units)
        if self.timeline is not None:
            self.timeline.append(("text", code_units))
        failure = self.failures.get(len(self.batches))
        if failure is not None:
            raise failure


def test_import_does_not_load_real_pynput_modules() -> None:
    assert "pynput.keyboard" not in sys.modules
    assert "pynput.mouse" not in sys.modules


def test_module_has_no_forbidden_top_level_behavior() -> None:
    tree = ast.parse(_source())
    imported_modules = {
        module_name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for module_name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
    }
    imported_names = {
        alias.asname or alias.name.rsplit(".", 1)[-1]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    top_level_forbidden_references = [
        node
        for statement in tree.body
        if not isinstance(statement, (ast.FunctionDef, ast.ClassDef))
        for node in ast.walk(statement)
        if (isinstance(node, ast.Name) and node.id in {"Controller", "Listener"})
        or (isinstance(node, ast.Attribute) and node.attr in {"Controller", "Listener"})
    ]

    assert "pynput" not in imported_modules
    assert "pynput.keyboard" not in imported_modules
    assert "pynput.mouse" not in imported_modules
    assert {"Controller", "Listener"}.isdisjoint(imported_names)
    assert top_level_forbidden_references == []


def test_module_has_no_forbidden_features() -> None:
    tree = ast.parse(_source())
    identifiers = {
        node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    class_names = {
        node.name.lower() for node in tree.body if isinstance(node, ast.ClassDef)
    }
    function_names = {
        node.name.lower()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported_modules = {
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden_fragments = {
        "listener",
        "clipboard",
        "pyperclip",
        "win32clipboard",
        "desktopcontroller",
        "operation_executor",
        "random",
    }
    code_names = (
        identifiers | attributes | class_names | function_names | imported_modules
    )
    bool_wrappers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bool"
    ]

    assert not any(
        fragment in name for name in code_names for fragment in forbidden_fragments
    )
    assert bool_wrappers == []


def test_dynamic_pynput_imports_are_confined_to_private_factories() -> None:
    tree = ast.parse(_source())
    imports_by_function: dict[str, list[str]] = {}
    all_dynamic_imports = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        imported = [
            child.args[0].value
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "importlib"
            and child.func.attr == "import_module"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and isinstance(child.args[0].value, str)
        ]
        if imported:
            imports_by_function[node.name] = imported

    assert imports_by_function == {
        "_create_keyboard_backend": ["pynput.keyboard"],
        "_create_scroll_backend": ["pynput.mouse"],
    }
    assert sorted(all_dynamic_imports) == ["pynput.keyboard", "pynput.mouse"]


def test_production_logging_never_formats_original_exceptions() -> None:
    tree = ast.parse(_source())
    exception_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "exception"
    ]
    exc_info_keywords = [
        keyword
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "exc_info"
    ]

    assert exception_calls == []
    assert exc_info_keywords == []


def test_public_interface_and_documentation() -> None:
    public_methods = {
        name
        for name, value in vars(keyboard_controller.KeyboardController).items()
        if callable(value) and not name.startswith("_")
    }
    expected = {"type", "press", "release", "hotkey", "scroll"}

    assert public_methods == expected
    assert "不保证线程安全" in keyboard_controller.KeyboardController.__doc__
    for method_name in expected:
        signature = inspect.signature(
            getattr(keyboard_controller.KeyboardController, method_name)
        )
        assert signature.return_annotation in {None, "None"}


def test_default_delays_and_factories(
    fake_environment: FakeEnvironment,
) -> None:
    controller = keyboard_controller.KeyboardController()

    assert controller._typing_interval == 0.05
    assert controller._action_delay == 0.1
    assert fake_environment.factory_calls == ["keyboard", "scroll"]


def test_backends_are_reused_for_multiple_operations(
    fake_environment: FakeEnvironment,
) -> None:
    controller = _controller(fake_environment)

    controller.press("a")
    controller.release("a")
    controller.scroll("up", 1)

    assert fake_environment.factory_calls == ["keyboard", "scroll"]
    assert fake_environment.keyboard.events == [
        ("press", "a"),
        ("release", "a"),
    ]
    assert fake_environment.scroll.events == [(0, 1)]


def test_source_has_no_sensitive_log_interpolation() -> None:
    source = _source()

    assert "完整文本" not in source
    assert "字符=%" not in source
    assert "key=%" not in source


def test_keyboard_module_does_not_depend_on_mouse_or_perception_modules() -> None:
    module_path = Path(__file__).parent.parent / "control" / "keyboard_controller.py"
    source = module_path.read_text(encoding="utf-8")

    # AST 检查避免运行时模块缓存或动态后端加载影响职责隔离结论。
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert "control.mouse_controller" not in imported_modules
    assert "mouse_controller" not in imported_modules
    assert all(
        module != "perception" and not module.startswith("perception.")
        for module in imported_modules
    )
