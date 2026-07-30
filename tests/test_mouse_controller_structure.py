"""测试鼠标模块结构、动态导入、构造复用与公共接口。

backend 和 mss 均由 fake 隔离，不创建真实鼠标控制器。
"""

import ast
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest

from control import mouse_controller
from utils.exceptions import MouseOperationError

from tests.mouse_test_support import FakeBackend
from tests.mouse_test_support import SafeEnvironment
from tests.mouse_test_support import _controller
from tests.mouse_test_support import _module_source
from tests.mouse_test_support import safe_environment


def test_module_has_no_top_level_backend_creation() -> None:
    tree = ast.parse(_module_source())

    top_level_calls = [
        node
        for statement in tree.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_create_backend"
        and not isinstance(statement, (ast.FunctionDef, ast.ClassDef))
    ]

    assert top_level_calls == []


def test_module_has_no_top_level_pynput_mouse_import() -> None:
    tree = ast.parse(_module_source())
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "pynput" not in imported_names
    assert "pynput.mouse" not in imported_names


def test_module_has_no_top_level_mss_instance() -> None:
    tree = ast.parse(_module_source())
    top_level_mss_calls = [
        node
        for statement in tree.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "MSS"
        and not isinstance(statement, (ast.FunctionDef, ast.ClassDef))
    ]

    assert top_level_mss_calls == []


def test_default_action_delay_is_point_one(
    safe_environment: SafeEnvironment,
) -> None:
    controller = mouse_controller.MouseController()

    controller.click()

    assert safe_environment.sleeps == [0.1]


def test_custom_action_delay_is_used(
    safe_environment: SafeEnvironment,
) -> None:
    controller = mouse_controller.MouseController(0.25)

    controller.click()

    assert safe_environment.sleeps == [0.25]


def test_integer_action_delay_is_stored_as_float(
    safe_environment: SafeEnvironment,
) -> None:
    controller = mouse_controller.MouseController(1)

    assert controller._action_delay == 1.0
    assert type(controller._action_delay) is float


def test_zero_action_delay_is_allowed(
    safe_environment: SafeEnvironment,
) -> None:
    controller = mouse_controller.MouseController(0)

    controller.click()

    assert safe_environment.sleeps == [0.0]


@pytest.mark.parametrize("value", ["0.1", None, object()])
def test_non_numeric_action_delay_raises_type_error(
    safe_environment: SafeEnvironment,
    value: object,
) -> None:
    with pytest.raises(TypeError):
        mouse_controller.MouseController(cast(Any, value))


def test_bool_action_delay_raises_type_error(
    safe_environment: SafeEnvironment,
) -> None:
    with pytest.raises(TypeError):
        mouse_controller.MouseController(True)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_action_delay_raises_value_error(
    safe_environment: SafeEnvironment,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        mouse_controller.MouseController(value)


def test_negative_action_delay_raises_value_error(
    safe_environment: SafeEnvironment,
) -> None:
    with pytest.raises(ValueError):
        mouse_controller.MouseController(-0.1)


@pytest.mark.parametrize("value", ["bad", True, math.nan, math.inf, -1])
def test_invalid_action_delay_does_not_create_backend(
    safe_environment: SafeEnvironment,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        mouse_controller.MouseController(cast(Any, value))

    assert safe_environment.factory_calls == []


def test_backend_creation_failure_is_converted_and_chained(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = OSError("backend unavailable")

    def fail() -> FakeBackend:
        raise failure

    monkeypatch.setattr(mouse_controller, "_create_backend", fail)

    with caplog.at_level(logging.ERROR), pytest.raises(
        MouseOperationError
    ) as exc_info:
        mouse_controller.MouseController()

    assert exc_info.value.__cause__ is failure
    assert "鼠标后端初始化失败" in caplog.text


def test_real_backend_factory_converts_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = OSError("controller failed")
    fake_module = SimpleNamespace(
        Controller=lambda: (_ for _ in ()).throw(failure),
        Button=SimpleNamespace(left=object(), right=object()),
    )
    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: fake_module,
    )

    with caplog.at_level(logging.ERROR), pytest.raises(
        MouseOperationError
    ) as exc_info:
        mouse_controller._create_backend()

    assert exc_info.value.__cause__ is failure
    assert "鼠标后端初始化失败" in caplog.text


def test_each_instance_creates_exactly_one_backend(
    safe_environment: SafeEnvironment,
) -> None:
    first = mouse_controller.MouseController()
    second = mouse_controller.MouseController()

    first.click()
    second.click()

    assert safe_environment.factory_calls == ["create", "create"]


def test_dynamic_backend_maps_only_left_and_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, int]] = []

    class FakeController:
        position = (0, 0)

        def click(self, button: object, count: int = 1) -> None:
            calls.append((button, count))

        def press(self, button: object) -> None:
            pass

        def release(self, button: object) -> None:
            pass

    left = object()
    right = object()
    fake_module = SimpleNamespace(
        Controller=FakeController,
        Button=SimpleNamespace(left=left, right=right, middle=object()),
    )
    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: fake_module,
    )

    backend = mouse_controller._create_backend()
    backend.click("left", 1)
    backend.click("right", 2)

    assert calls == [(left, 1), (right, 2)]


def test_virtual_bounds_reads_monitor_zero_without_grab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[str] = []
    context = SimpleNamespace(
        monitors=[{"left": -20, "top": -10, "width": 100, "height": 80}],
        entered=False,
        exited=False,
    )

    class FakeMSS:
        def __enter__(self) -> object:
            context.entered = True
            return context

        def __exit__(self, *args: object) -> None:
            context.exited = True

    def modern_factory() -> FakeMSS:
        constructor_calls.append("MSS")
        return FakeMSS()

    def legacy_factory() -> object:
        pytest.fail("MSS 存在时不得调用旧版 mss 入口")

    fake_module = SimpleNamespace(MSS=modern_factory, mss=legacy_factory)
    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: fake_module,
    )

    result = mouse_controller._get_virtual_screen_bounds()

    assert result == (-20, -10, 100, 80)
    assert constructor_calls == ["MSS"]
    assert context.entered is True
    assert context.exited is True
    assert not hasattr(context, "grab")


def test_virtual_bounds_falls_back_to_legacy_mss_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[str] = []
    context = SimpleNamespace(
        monitors=[{"left": -20, "top": -10, "width": 100, "height": 80}],
        entered=False,
        exited=False,
    )

    class FakeMSS:
        def __enter__(self) -> object:
            context.entered = True
            return context

        def __exit__(self, *args: object) -> None:
            context.exited = True

    def legacy_factory() -> FakeMSS:
        constructor_calls.append("mss")
        return FakeMSS()

    fake_module = SimpleNamespace(mss=legacy_factory)
    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: fake_module,
    )

    result = mouse_controller._get_virtual_screen_bounds()

    assert result == (-20, -10, 100, 80)
    assert constructor_calls == ["mss"]
    assert context.entered is True
    assert context.exited is True
    assert not hasattr(context, "grab")


def test_virtual_bounds_does_not_fallback_when_modern_constructor_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = OSError("modern constructor failed")

    def modern_factory() -> object:
        raise failure

    def legacy_factory() -> object:
        pytest.fail("现代入口构造失败时不得回退旧版入口")

    fake_module = SimpleNamespace(MSS=modern_factory, mss=legacy_factory)
    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: fake_module,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(MouseOperationError) as exc_info:
            mouse_controller._get_virtual_screen_bounds()

    assert exc_info.value.__cause__ is failure
    assert "获取虚拟桌面边界失败" in caplog.text


def test_virtual_bounds_converts_missing_mss_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mouse_controller.importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(MouseOperationError) as exc_info:
        mouse_controller._get_virtual_screen_bounds()

    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_mouse_operation_error_inherits_runtime_error() -> None:
    assert issubclass(MouseOperationError, RuntimeError)


def test_parameter_error_does_not_record_error_log(
    safe_environment: SafeEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = _controller(safe_environment)

    with caplog.at_level(logging.ERROR), pytest.raises(TypeError):
        controller.click(True, 1)

    assert caplog.records == []


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("move_to", (1, 2)),
        ("click", ()),
        ("right_click", ()),
        ("double_click", ()),
        ("drag_from_to", (0, 0, 1, 1, 0)),
    ],
)
def test_all_successful_public_methods_return_none(
    safe_environment: SafeEnvironment,
    method_name: str,
    args: tuple[object, ...],
) -> None:
    controller = _controller(safe_environment)

    result = getattr(controller, method_name)(*args)

    assert result is None


def test_public_methods_have_docstrings_and_return_annotations() -> None:
    method_names = (
        "__init__",
        "move_to",
        "click",
        "right_click",
        "double_click",
        "drag_from_to",
    )

    for method_name in method_names:
        method = getattr(mouse_controller.MouseController, method_name)
        assert method.__doc__
        assert method.__annotations__.get("return") in {None, "None"}


def test_module_does_not_reference_listener_or_boolean_wrapper() -> None:
    source = _module_source()

    assert "Listener" not in source
    assert "return True" not in source
    assert "return False" not in source


def test_test_module_does_not_import_real_mouse_or_mss() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert all(not name.startswith("pynput") for name in imported_modules)
    assert all(not name.startswith("mss") for name in imported_modules)
