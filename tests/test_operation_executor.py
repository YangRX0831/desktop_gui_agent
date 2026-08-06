"""测试 PRD 4.2.3 操作包装的成功语义、异常边界与依赖隔离。

操作、日志和失败均由内存替身承载，不创建真实控制 backend。
"""

import ast
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest

from control import operation_executor
from control.operation_executor import execute_operation
from utils.exceptions import KeyboardOperationError, MouseOperationError

FIXED_EVENT = "operation_execution_failed"


@contextmanager
def formatted_log_output() -> Iterator[StringIO]:
    """捕获包装层最终日志，并在退出时恢复其 logger 状态。"""
    target_logger = operation_executor.logger
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s|%(name)s|%(message)s"))
    previous_level = target_logger.level
    previous_propagate = target_logger.propagate
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.ERROR)
    target_logger.propagate = False
    try:
        yield stream
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)
        target_logger.propagate = previous_propagate


class SensitiveCallable:
    """抛出预设异常并提供不可写入日志的对象表示。"""

    def __init__(self, exception: Exception) -> None:
        self.exception = exception

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise self.exception

    def __repr__(self) -> str:
        return "SENSITIVE_OPERATION_REPR"


@pytest.mark.parametrize(
    "return_value",
    [None, False, True, 0, "", object()],
    ids=["none", "false", "true", "zero", "empty_string", "object"],
)
def test_normal_return_value_is_always_success(return_value: object) -> None:
    def operation() -> object:
        return return_value

    result = execute_operation(operation)

    assert result is True


def test_positional_arguments_are_forwarded_unchanged() -> None:
    calls: list[tuple[object, ...]] = []

    def operation(*args: object) -> None:
        calls.append(args)

    result = execute_operation(operation, "first", 2, None)

    assert result is True
    assert calls == [("first", 2, None)]


def test_keyword_arguments_are_forwarded_unchanged() -> None:
    calls: list[dict[str, object]] = []

    def operation(**kwargs: object) -> None:
        calls.append(kwargs)

    result = execute_operation(operation, text="value", count=2)

    assert result is True
    assert calls == [{"text": "value", "count": 2}]


def test_positional_and_keyword_arguments_are_forwarded_together() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def operation(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    result = execute_operation(operation, "value", enabled=True)

    assert result is True
    assert calls == [(("value",), {"enabled": True})]


def test_successful_operation_is_called_once() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    result = execute_operation(operation)

    assert result is True
    assert calls == 1


@pytest.mark.parametrize(
    "exception",
    [
        PermissionError("hidden"),
        TimeoutError("hidden"),
        RuntimeError("hidden"),
        TypeError("hidden"),
    ],
    ids=["permission", "timeout", "runtime", "type"],
)
def test_ordinary_exception_returns_false(exception: Exception) -> None:
    operation = SensitiveCallable(exception)

    result = execute_operation(operation)

    assert result is False


def test_non_callable_input_type_error_returns_false() -> None:
    operation = cast(Any, object())

    result = execute_operation(operation)

    assert result is False


def test_ordinary_exception_is_safely_logged_once() -> None:
    operation = SensitiveCallable(RuntimeError("hidden"))

    with formatted_log_output() as stream:
        result = execute_operation(operation)

    assert result is False
    assert stream.getvalue().count(FIXED_EVENT) == 1


def test_final_log_contains_fixed_event() -> None:
    operation = SensitiveCallable(RuntimeError("hidden"))

    with formatted_log_output() as stream:
        execute_operation(operation)

    assert FIXED_EVENT in stream.getvalue()


def test_final_log_contains_exception_type() -> None:
    operation = SensitiveCallable(PermissionError("hidden"))

    with formatted_log_output() as stream:
        execute_operation(operation)

    assert "PermissionError" in stream.getvalue()


def test_final_log_contains_safe_code_location() -> None:
    operation = SensitiveCallable(RuntimeError("hidden"))

    with formatted_log_output() as stream:
        execute_operation(operation)

    output = stream.getvalue()
    assert "文件=" in output
    assert "函数=" in output
    assert "行号=" in output
    assert "test_operation_executor.py" in output


def test_final_log_excludes_exception_message() -> None:
    operation = SensitiveCallable(RuntimeError("SENSITIVE_EXCEPTION_MESSAGE"))

    with formatted_log_output() as stream:
        execute_operation(operation)

    assert "SENSITIVE_EXCEPTION_MESSAGE" not in stream.getvalue()


def test_final_log_excludes_absolute_path() -> None:
    operation = SensitiveCallable(RuntimeError(r"C:\private\operation"))

    with formatted_log_output() as stream:
        execute_operation(operation)

    output = stream.getvalue()
    assert "C:\\" not in output
    assert "/desktop_gui_agent/" not in output.replace("\\", "/")


def test_final_log_excludes_operation_identity() -> None:
    operation = SensitiveCallable(RuntimeError("hidden"))

    with formatted_log_output() as stream:
        execute_operation(operation)

    output = stream.getvalue()
    assert "SensitiveCallable" not in output
    assert repr(operation) not in output


def test_final_log_excludes_positional_arguments() -> None:
    operation = SensitiveCallable(RuntimeError("hidden"))

    with formatted_log_output() as stream:
        execute_operation(operation, "SENSITIVE_POSITIONAL_ARGUMENT")

    assert "SENSITIVE_POSITIONAL_ARGUMENT" not in stream.getvalue()


def test_final_log_excludes_keyword_arguments() -> None:
    operation = SensitiveCallable(RuntimeError("hidden"))

    with formatted_log_output() as stream:
        execute_operation(operation, secret="SENSITIVE_KEYWORD_ARGUMENT")

    output = stream.getvalue()
    assert "secret" not in output
    assert "SENSITIVE_KEYWORD_ARGUMENT" not in output


def test_successful_return_value_is_not_logged() -> None:
    def operation() -> str:
        return "SENSITIVE_RETURN_VALUE"

    with formatted_log_output() as stream:
        result = execute_operation(operation)

    assert result is True
    assert stream.getvalue() == ""


def test_complex_sensitive_exception_message_is_not_logged() -> None:
    message = "密_TOKEN_password-123\n%s %(name)s " + "长" * 1000
    operation = SensitiveCallable(RuntimeError(message))

    with formatted_log_output() as stream:
        execute_operation(operation)

    output = stream.getvalue()
    assert all(part not in output for part in ("密", "TOKEN", "password", "%s", "长"))
    assert output.count("\n") == 1


@pytest.mark.parametrize(
    "exception",
    [
        MouseOperationError("hidden"),
        KeyboardOperationError("hidden"),
    ],
    ids=["mouse", "keyboard"],
)
def test_domain_exception_returns_false_without_wrapper_log(
    exception: Exception,
) -> None:
    operation = SensitiveCallable(exception)

    with formatted_log_output() as stream:
        result = execute_operation(operation)

    assert result is False
    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "exception",
    [
        KeyboardInterrupt(),
        SystemExit(),
        GeneratorExit(),
        type("CustomBaseException", (BaseException,), {})(),
    ],
    ids=["keyboard_interrupt", "system_exit", "generator_exit", "custom"],
)
def test_base_exception_subclass_propagates(exception: BaseException) -> None:
    def operation() -> None:
        raise exception

    with pytest.raises(type(exception)) as error_info:
        execute_operation(operation)

    assert error_info.value is exception


def test_failed_operation_is_not_retried() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("hidden")

    result = execute_operation(operation)

    assert result is False
    assert calls == 1


def test_module_has_only_approved_dependency_directions() -> None:
    module_path = Path(operation_executor.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    forbidden = {
        "control.mouse_controller",
        "control.keyboard_controller",
        "pynput",
        "mss",
        "paddleocr",
        "tkinter",
        "agentscope",
    }
    assert imported_modules.isdisjoint(forbidden)
    assert all(
        module != "perception" and not module.startswith("perception.")
        for module in imported_modules
    )


def test_module_source_has_no_real_backend_creation() -> None:
    source = Path(operation_executor.__file__).read_text(encoding="utf-8")

    assert "MouseController(" not in source
    assert "KeyboardController(" not in source
    assert "Listener(" not in source
    assert "Handler(" not in source


def test_execution_does_not_create_logger_handler() -> None:
    handlers_before = tuple(operation_executor.logger.handlers)

    result = execute_operation(lambda: None)

    assert result is True
    assert tuple(operation_executor.logger.handlers) == handlers_before


def test_module_public_api_only_contains_execute_operation() -> None:
    module_path = Path(operation_executor.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    public_definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and not node.name.startswith("_")
    }

    assert public_definitions == {"execute_operation"}


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda: None, True),
        (SensitiveCallable(RuntimeError("hidden")), False),
    ],
    ids=["success", "failure"],
)
def test_result_type_is_always_bool(
    operation: object,
    expected: bool,
) -> None:
    result = execute_operation(cast(Any, operation))

    assert result is expected
    assert type(result) is bool


def test_module_source_has_no_retry_sleep_or_unsafe_logging() -> None:
    source = Path(operation_executor.__file__).read_text(encoding="utf-8")

    assert "logger.exception" not in source
    assert "exc_info=True" not in source
    assert "sleep(" not in source
    assert "retry" not in source.lower()
    assert "while " not in source


def test_module_defines_no_classes() -> None:
    module_path = Path(operation_executor.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
