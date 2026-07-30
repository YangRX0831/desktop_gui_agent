"""测试 Windows Unicode SendInput 的 ABI、UTF-16 与事件构造。

user32 调用由内存 fake 代替；非 BMP 场景不代表真实输入验证。
"""

import ast
import io
import logging
from collections import defaultdict
from pathlib import Path

import pytest

from control import keyboard_controller
from utils.exceptions import KeyboardOperationError

from tests.keyboard_test_support import FakeEnvironment
from tests.keyboard_test_support import FakeSendInput
from tests.keyboard_test_support import _controller
from tests.keyboard_test_support import _keyboard_events
from tests.keyboard_test_support import _source
from tests.keyboard_test_support import fake_environment


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


def test_windows_type_preserves_exact_mixed_case(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    controller = _controller(fake_environment)

    result = controller.type("OpenAI Test")

    assert result is None
    assert fake_environment.text.batches == [
        (ord(character),) for character in "OpenAI Test"
    ]
    assert fake_environment.keyboard.events == []
    assert fake_environment.text_factory_calls == ["text"]
    assert fake_environment.factory_calls == ["keyboard", "scroll"]


@pytest.mark.parametrize("caps_lock_enabled", [False, True])
def test_windows_text_path_is_independent_of_caps_lock_state(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caps_lock_enabled: bool,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    observed_state = caps_lock_enabled
    controller = _controller(fake_environment)

    controller.type("Aa")

    assert observed_state is caps_lock_enabled
    assert fake_environment.text.batches == [(ord("A"),), (ord("a"),)]
    assert fake_environment.keyboard.events == []


def test_windows_text_path_has_no_lock_state_or_key_scan_calls() -> None:
    tree = ast.parse(_source())
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert {
        "GetKeyState",
        "GetKeyboardState",
        "SetKeyboardState",
        "VkKeyScan",
        "VkKeyScanW",
    }.isdisjoint(called_names | called_attributes)


@pytest.mark.parametrize(
    ("code_units", "expected_events"),
    [
        (
            (0x41,),
            [
                (0, 0x41, keyboard_controller._KEYEVENTF_UNICODE),
                (
                    0,
                    0x41,
                    keyboard_controller._KEYEVENTF_UNICODE
                    | keyboard_controller._KEYEVENTF_KEYUP,
                ),
            ],
        ),
        (
            (0xD83D, 0xDE00),
            [
                (0, 0xD83D, keyboard_controller._KEYEVENTF_UNICODE),
                (
                    0,
                    0xD83D,
                    keyboard_controller._KEYEVENTF_UNICODE
                    | keyboard_controller._KEYEVENTF_KEYUP,
                ),
                (0, 0xDE00, keyboard_controller._KEYEVENTF_UNICODE),
                (
                    0,
                    0xDE00,
                    keyboard_controller._KEYEVENTF_UNICODE
                    | keyboard_controller._KEYEVENTF_KEYUP,
                ),
            ],
        ),
    ],
)
def test_windows_backend_builds_unicode_key_down_up_batch(
    code_units: tuple[int, ...],
    expected_events: list[tuple[int, int, int]],
) -> None:
    send_input = FakeSendInput()
    backend = keyboard_controller._WindowsUnicodeTextBackend(send_input)

    backend.send(code_units)

    assert len(send_input.calls) == 1
    count, inputs, structure_size = send_input.calls[0]
    assert count == len(code_units) * 2
    assert structure_size == keyboard_controller.ctypes.sizeof(
        keyboard_controller._Input
    )
    assert _keyboard_events(inputs, count) == expected_events
    assert all(
        inputs[index].input_type == keyboard_controller._INPUT_KEYBOARD
        for index in range(count)
    )


@pytest.mark.parametrize("returned_count", [0, 1, 3])
def test_windows_backend_rejects_incomplete_send_input(
    returned_count: int,
) -> None:
    send_input = FakeSendInput()
    send_input.result = returned_count
    backend = keyboard_controller._WindowsUnicodeTextBackend(send_input)

    with pytest.raises(OSError, match="未完整插入"):
        backend.send((0x41,))

    assert len(send_input.calls) == 1


def test_windows_backend_structure_layout_matches_current_windows_abi() -> None:
    pointer_size = keyboard_controller.ctypes.sizeof(
        keyboard_controller.ctypes.c_void_p
    )
    expected_keyboard_size = 24 if pointer_size == 8 else 16
    expected_input_size = 40 if pointer_size == 8 else 28

    assert keyboard_controller.ctypes.sizeof(
        keyboard_controller._KeyboardInput
    ) == expected_keyboard_size
    assert (
        keyboard_controller.ctypes.sizeof(keyboard_controller._Input)
        == expected_input_size
    )
    assert keyboard_controller._KeyboardInput.virtual_key.offset == 0
    assert keyboard_controller._KeyboardInput.scan_code.offset == 2
    assert keyboard_controller._KeyboardInput.flags.offset == 4


def test_windows_backend_propagates_send_input_exception() -> None:
    original = RuntimeError("api failed")
    send_input = FakeSendInput()
    send_input.failure = original
    backend = keyboard_controller._WindowsUnicodeTextBackend(send_input)

    with pytest.raises(RuntimeError) as caught:
        backend.send((0x41,))

    assert caught.value is original
    assert len(send_input.calls) == 1


def test_public_type_wraps_partial_send_input_with_cause(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    send_input = FakeSendInput()
    send_input.result = 1
    backend = keyboard_controller._WindowsUnicodeTextBackend(send_input)
    monkeypatch.setattr(
        keyboard_controller,
        "_create_windows_text_backend",
        lambda: backend,
    )
    controller = _controller(fake_environment)

    with pytest.raises(KeyboardOperationError) as caught:
        controller.type("A")

    assert isinstance(caught.value.__cause__, OSError)
    assert len(send_input.calls) == 1
    assert fake_environment.sleeps == []


def test_windows_factory_declares_send_input_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send_input = FakeSendInput()

    class FakeUser32:
        SendInput = send_input

    calls: list[tuple[str, bool]] = []

    def fake_win_dll(name: str, use_last_error: bool) -> FakeUser32:
        calls.append((name, use_last_error))
        return FakeUser32()

    monkeypatch.setattr(keyboard_controller.ctypes, "WinDLL", fake_win_dll)

    backend = keyboard_controller._create_windows_text_backend()

    assert isinstance(
        backend,
        keyboard_controller._WindowsUnicodeTextBackend,
    )
    assert calls == [("user32", True)]
    assert send_input.argtypes == (
        keyboard_controller.ctypes.c_uint,
        keyboard_controller.ctypes.POINTER(keyboard_controller._Input),
        keyboard_controller.ctypes.c_int,
    )
    assert send_input.restype is keyboard_controller.ctypes.c_uint


def test_user32_loading_is_confined_to_private_factory() -> None:
    tree = ast.parse(_source())
    calls_by_function: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = [
            child.args[0].value
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "WinDLL"
            and child.args
            and isinstance(child.args[0], ast.Constant)
        ]
        if calls:
            calls_by_function[node.name] = calls

    assert calls_by_function == {
        "_create_windows_text_backend": ["user32"],
    }


def test_formatted_windows_text_log_redacts_sensitive_data(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive = "密_TOKEN_password-123"
    monkeypatch.setattr(keyboard_controller.sys, "platform", "win32")
    fake_environment.text.fail_on(1, RuntimeError(sensitive))
    controller = _controller(fake_environment)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    )
    monkeypatch.setattr(keyboard_controller.logger, "propagate", False)
    keyboard_controller.logger.addHandler(handler)

    try:
        with pytest.raises(KeyboardOperationError) as exc_info:
            controller.type("密")
    finally:
        keyboard_controller.logger.removeHandler(handler)

    output = stream.getvalue()
    cause = exc_info.value.__cause__
    assert isinstance(cause, RuntimeError)
    traceback = cause.__traceback__
    assert traceback is not None
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    expected_method = type(fake_environment.text).send
    expected_path = Path(expected_method.__code__.co_filename)
    assert traceback.tb_frame.f_code is expected_method.__code__
    assert sensitive not in output
    assert "密" not in output
    assert "TOKEN" not in output
    assert "password" not in output
    assert "123" not in output
    assert "Windows 文本输入失败" in output
    assert "RuntimeError" in output
    assert expected_path.name in output
    assert expected_method.__name__ in output
    assert f"行号={traceback.tb_lineno}" in output
    assert str(expected_path.resolve()) not in output
    assert "Traceback" not in output
    assert "raise failure" not in output


def test_formatted_log_redacts_original_exception_and_character(
    fake_environment: FakeEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive = "密_TOKEN_password-123"
    fake_environment.keyboard.fail_on("press", 1, RuntimeError(sensitive))
    controller = _controller(fake_environment)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    )
    monkeypatch.setattr(keyboard_controller.logger, "propagate", False)
    keyboard_controller.logger.addHandler(handler)

    try:
        with pytest.raises(KeyboardOperationError) as exc_info:
            controller.press("密")
    finally:
        keyboard_controller.logger.removeHandler(handler)

    output = stream.getvalue()
    cause = exc_info.value.__cause__
    assert isinstance(cause, RuntimeError)
    traceback = cause.__traceback__
    assert traceback is not None
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    expected_method = type(fake_environment.keyboard)._record
    expected_path = Path(expected_method.__code__.co_filename)
    assert traceback.tb_frame.f_code is expected_method.__code__
    assert sensitive not in output
    assert "密" not in output
    assert "TOKEN" not in output
    assert "password" not in output
    assert "123" not in output
    assert "按键失败" in output
    assert "RuntimeError" in output
    assert expected_path.name in output
    assert expected_method.__name__ in output
    assert f"行号={traceback.tb_lineno}" in output
    assert str(expected_path.resolve()) not in output
    assert "Traceback" not in output
    assert "raise failure" not in output
