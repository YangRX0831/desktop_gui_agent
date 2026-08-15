"""验证控制层公共接口、白名单分发和安全失败边界。"""

import logging
from io import StringIO
from types import SimpleNamespace

import pytest

import control.keyboard_controller as keyboard_module
import control.mouse_controller as mouse_module
from agent.action_dispatcher import ActionDispatcher, PermissionScope
from control.keyboard_controller import KeyboardController, is_supported_key
from control.mouse_controller import MouseController
from control.operation_executor import execute_operation
from utils.exceptions import KeyboardOperationError, MouseOperationError


class FakeKeyboardBackend:
    """记录键盘 press/release 调用。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def press(self, key: object) -> None:
        """记录按下。"""
        self.events.append(("press", key))

    def release(self, key: object) -> None:
        """记录释放。"""
        self.events.append(("release", key))


class FakeScrollBackend:
    """记录滚轮调用。"""

    def __init__(self) -> None:
        self.events: list[tuple[int, int]] = []

    def scroll(self, dx: int, dy: int) -> None:
        """记录二维滚动增量。"""
        self.events.append((dx, dy))


class FakeMouseBackend:
    """记录鼠标后端动作。"""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def move_to(self, x: int, y: int) -> None:
        """记录移动。"""
        self.events.append(("move", x, y))

    def click(self, button: str, count: int) -> None:
        """记录点击。"""
        self.events.append(("click", button, count))

    def press(self, button: str) -> None:
        """记录按下。"""
        self.events.append(("press", button))

    def release(self, button: str) -> None:
        """记录释放。"""
        self.events.append(("release", button))


def test_mouse_backend_enables_dpi_before_controller_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 pynput Controller 创建前必须先统一 Windows DPI 坐标空间。"""
    events: list[str] = []

    class FakeController:
        def __init__(self) -> None:
            events.append("controller")

    mouse_api = SimpleNamespace(
        Controller=FakeController,
        Button=SimpleNamespace(left="left", right="right"),
    )
    monkeypatch.setattr(
        mouse_module,
        "_enable_windows_dpi_awareness",
        lambda: events.append("dpi"),
    )
    monkeypatch.setattr(
        mouse_module.importlib,
        "import_module",
        lambda name: mouse_api,
    )

    mouse_module._create_backend()

    assert events == ["dpi", "controller"]


def test_keyboard_public_operations_use_injected_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """键盘输入、组合键和滚动沿公共接口调用隔离后端。"""
    keyboard = FakeKeyboardBackend()
    scroll = FakeScrollBackend()
    keys = SimpleNamespace(ctrl="CTRL", enter="ENTER")
    monkeypatch.setattr(
        keyboard_module,
        "_create_keyboard_backend",
        lambda: (keyboard, keys),
    )
    monkeypatch.setattr(keyboard_module, "_create_scroll_backend", lambda: scroll)
    monkeypatch.setattr(keyboard_module, "_sleep", lambda seconds: None)
    controller = KeyboardController(typing_interval=0, action_delay=0)

    controller.hotkey("ctrl", "a")
    controller.scroll("down", 2)

    assert keyboard.events == [
        ("press", "CTRL"),
        ("press", "a"),
        ("release", "a"),
        ("release", "CTRL"),
    ]
    assert scroll.events == [(0, -2)]


@pytest.mark.parametrize("key", ["ctrl", "f20", "a", "中"])
def test_supported_keys_are_accepted(key: str) -> None:
    """控制器公开的命名键和单字符合同保持可用。"""
    assert is_supported_key(key)


@pytest.mark.parametrize(
    "key",
    ["volume_up", "volume_mute", "media_play", "brightness_up"],
)
def test_media_keys_naming_convention(key: str) -> None:
    """媒体键必须使用 media_ 前缀;无前缀的简写不被接受。"""
    assert not is_supported_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "media_volume_up",
        "media_volume_down",
        "media_volume_mute",
        "media_play_pause",
        "media_next",
        "media_previous",
        "media_stop",
    ],
)
def test_media_system_keys_are_supported(key: str) -> None:
    """标准媒体/系统键在白名单中,可供通用 GUI 操作使用。"""
    assert is_supported_key(key)


@pytest.mark.parametrize("key", ["", "f21", 1, None])
def test_unsupported_keys_are_rejected(key: object) -> None:
    """未映射键名不能进入控制后端。"""
    assert not is_supported_key(key)


def test_mouse_public_click_validates_and_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """鼠标点击在边界校验后才调用后端。"""
    backend = FakeMouseBackend()
    monkeypatch.setattr(mouse_module, "_create_backend", lambda: backend)
    monkeypatch.setattr(
        mouse_module,
        "_get_virtual_screen_bounds",
        lambda: (0, 0, 100, 80),
    )
    monkeypatch.setattr(mouse_module, "_sleep", lambda seconds: None)
    controller = MouseController(action_delay=0)

    controller.click(10, 20)

    assert backend.events == [("move", 10, 20), ("click", "left", 1)]
    with pytest.raises(ValueError):
        controller.click(100, 20)


def test_action_dispatcher_maps_image_pixel_coordinates() -> None:
    """模型图像像素坐标直接作为截图内像素(全屏 offset=0 即桌面像素)。"""
    calls: list[tuple[object, ...]] = []

    class Controls:
        """实现分发器需要的最小控制接口。"""

        def click(self, x: int, y: int, button: str = "left") -> None:
            calls.append(("click", x, y, button))

        def right_click(self, x: int | None = None, y: int | None = None) -> None:
            calls.append(("right_click", x, y))

        def double_click(self, x: int | None = None, y: int | None = None) -> None:
            calls.append(("double_click", x, y))

        def drag_from_to(
            self,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            duration: float = 0.5,
        ) -> None:
            calls.append(("drag", x1, y1, x2, y2))

        def type(self, text: str) -> None:
            calls.append(("type", text))

        def scroll(self, direction: str, steps: int) -> None:
            calls.append(("scroll", direction, steps))

        def hotkey(self, *keys: str) -> None:
            calls.append(("hotkey", *keys))

    controls = Controls()
    dispatcher = ActionDispatcher(controls, controls)
    dispatcher.activate_run_scope(
        PermissionScope(
            allowed_actions=frozenset({"click"}),
            token=1,
        ),
    )

    # 图像像素坐标落在截图(200×100)内 -> 直接作为桌面像素(offset=0)。
    succeeded = dispatcher.dispatch(
        {"action_type": "click", "params": {"x": 100, "y": 99}},
        (200, 100),
    )
    assert succeeded is True
    assert calls == [("click", 100, 99, "left")]

    # 越界(>= 截图尺寸)的坐标被拒绝,不再被当作 0..999 归一化放行。
    calls.clear()
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": {"x": 200, "y": 50}},
            (200, 100),
        )
        is False
    )
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": {"x": 50, "y": 100}},
            (200, 100),
        )
        is False
    )
    assert calls == []


def test_dispatcher_denies_without_scope() -> None:
    """缺失当前 run 授权作用域时产生副作用的动作被拒绝,不产生控制调用。"""
    calls: list[tuple[object, ...]] = []

    class Controls:
        """实现分发器需要的最小控制接口。"""

        def click(self, x: int, y: int, button: str = "left") -> None:
            calls.append(("click", x, y, button))

        def right_click(self, x: int | None = None, y: int | None = None) -> None:
            calls.append(("right_click", x, y))

        def double_click(self, x: int | None = None, y: int | None = None) -> None:
            calls.append(("double_click", x, y))

        def drag_from_to(
            self,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            duration: float = 0.5,
        ) -> None:
            calls.append(("drag", x1, y1, x2, y2))

        def type(self, text: str) -> None:
            calls.append(("type", text))

        def scroll(self, direction: str, steps: int) -> None:
            calls.append(("scroll", direction, steps))

        def hotkey(self, *keys: str) -> None:
            calls.append(("hotkey", *keys))

    controls = Controls()
    dispatcher = ActionDispatcher(controls, controls)

    succeeded = dispatcher.dispatch(
        {"action_type": "click", "params": {"x": 500, "y": 999}},
        (200, 100),
    )

    assert succeeded is False
    assert calls == []


@pytest.mark.parametrize(
    "action",
    [
        {"action_type": "click", "params": {"x": 1000, "y": 0}},
        {"action_type": "scroll", "params": {"direction": "left", "steps": 1}},
        {"action_type": "hotkey", "params": {"keys": ("win",)}},
        {"action_type": "shell", "params": {}},
    ],
)
def test_dispatcher_rejects_invalid_actions_without_control(
    action: dict[str, object],
) -> None:
    """非法参数和非白名单动作在控制调用前返回失败。"""
    calls: list[object] = []

    class Controls:
        """任何调用都记录为安全边界失败。"""

        def click(self, *args: object) -> None:
            calls.append(args)

        def right_click(self, *args: object) -> None:
            calls.append(args)

        def double_click(self, *args: object) -> None:
            calls.append(args)

        def drag_from_to(self, *args: object) -> None:
            calls.append(args)

        def type(self, *args: object) -> None:
            calls.append(args)

        def scroll(self, *args: object) -> None:
            calls.append(args)

        def hotkey(self, *args: object) -> None:
            calls.append(args)

    controls = Controls()
    dispatcher = ActionDispatcher(controls, controls)

    assert (
        dispatcher.dispatch(  # type: ignore[arg-type]  # 不可信映射测试。
            action,
            (100, 100),
        )
        is False
    )
    assert calls == []


def test_operation_executor_logs_no_exception_message() -> None:
    """异常包装器返回 False，最终 Formatter 输出不包含异常正文。"""
    target = logging.getLogger("control.operation_executor")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous = (target.level, target.propagate)
    target.addHandler(handler)
    target.setLevel(logging.ERROR)
    target.propagate = False

    def failing_operation() -> None:
        raise RuntimeError("SENSITIVE_TEXT")

    try:
        assert execute_operation(failing_operation) is False
    finally:
        target.removeHandler(handler)
        target.setLevel(previous[0])
        target.propagate = previous[1]

    assert "SENSITIVE_TEXT" not in stream.getvalue()
    assert "RuntimeError" in stream.getvalue()


def test_controller_backend_failures_preserve_domain_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实后端构造失败由对应控制器异常承接并保留原因。"""
    mouse_error = RuntimeError("mouse")
    keyboard_error = RuntimeError("keyboard")
    monkeypatch.setattr(
        mouse_module,
        "_create_backend",
        lambda: (_ for _ in ()).throw(mouse_error),
    )
    monkeypatch.setattr(
        keyboard_module,
        "_create_keyboard_backend",
        lambda: (_ for _ in ()).throw(keyboard_error),
    )

    with pytest.raises(MouseOperationError) as mouse_info:
        MouseController()
    with pytest.raises(KeyboardOperationError) as keyboard_info:
        KeyboardController()

    assert mouse_info.value.__cause__ is mouse_error
    assert keyboard_info.value.__cause__ is keyboard_error


# ---------------------------------------------------------------------------
# PRD public API direct test coverage
# ---------------------------------------------------------------------------


def _make_mouse(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """创建注入 fake backend 的 MouseController 并返回 (controller, backend)。"""
    backend = FakeMouseBackend()
    monkeypatch.setattr(mouse_module, "_create_backend", lambda: backend)
    monkeypatch.setattr(
        mouse_module,
        "_get_virtual_screen_bounds",
        lambda: (0, 0, 100, 80),
    )
    monkeypatch.setattr(mouse_module, "_sleep", lambda seconds: None)
    return MouseController(action_delay=0), backend


def test_mouse_move_to_calls_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """move_to 调用后端 move 并验证坐标传递。"""
    controller, backend = _make_mouse(monkeypatch)
    controller.move_to(30, 40)
    assert backend.events == [("move", 30, 40)]


def test_mouse_right_click_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """right_click 使用 right 按钮调用后端。"""
    controller, backend = _make_mouse(monkeypatch)
    controller.right_click(10, 20)
    assert ("move", 10, 20) in backend.events
    assert ("click", "right", 1) in backend.events


def test_mouse_double_click_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """double_click 使用 left 按钮调用后端两次。"""
    controller, backend = _make_mouse(monkeypatch)
    controller.double_click(10, 20)
    assert ("move", 10, 20) in backend.events
    assert ("click", "left", 2) in backend.events


def test_mouse_drag_calls_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """drag_from_to 执行移动+按下+拖拽+释放序列。"""
    controller, backend = _make_mouse(monkeypatch)
    controller.drag_from_to(10, 10, 50, 50, duration=0)
    events = backend.events
    assert ("move", 10, 10) in events
    assert ("press", "left") in events
    assert ("release", "left") in events


def _make_keyboard(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple:
    """创建注入 fake backend 的 KeyboardController 并返回 (controller, backend)。"""
    keyboard = FakeKeyboardBackend()
    scroll = FakeScrollBackend()
    keys = SimpleNamespace(ctrl="CTRL", enter="ENTER", a="A")
    monkeypatch.setattr(
        keyboard_module,
        "_create_keyboard_backend",
        lambda: (keyboard, keys),
    )
    monkeypatch.setattr(keyboard_module, "_create_scroll_backend", lambda: scroll)
    monkeypatch.setattr(keyboard_module, "_sleep", lambda seconds: None)
    return KeyboardController(typing_interval=0, action_delay=0), keyboard


def test_keyboard_type_calls_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """type 通过公共接口处理输入文本并验证不抛异常。

    必须注入 fake text backend,否则 win32 下会调用真实 user32.SendInput
    向前台窗口发送真实键盘输入(deterministic test side-effect leak)。
    """
    controller, keyboard = _make_keyboard(monkeypatch)
    # 注入 fake Windows text backend,阻止真实 SendInput。
    monkeypatch.setattr(
        keyboard_module,
        "_create_windows_text_backend",
        lambda: type(
            "_StubBackend",
            (),
            {"send": lambda self, units: None},
        )(),
    )
    controller.type("Hi")


def test_keyboard_press_calls_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """press 调用后端 press。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    controller.press("a")
    assert ("press", "a") in keyboard.events


def test_keyboard_release_calls_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """release 调用后端 release。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    controller.release("a")
    assert ("release", "a") in keyboard.events


def test_keyboard_hotkey_duplicate_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """hotkey 重复键在首次后端调用前被拒绝。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    with pytest.raises(ValueError):
        controller.hotkey("ctrl", "ctrl")


def test_keyboard_hotkey_empty_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """hotkey 空参数被拒绝。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    with pytest.raises(ValueError):
        controller.hotkey()


def test_keyboard_invalid_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """press 无效键名被拒绝。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    with pytest.raises(ValueError):
        controller.press("not_a_key")


def test_keyboard_scroll_invalid_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scroll 无效方向被拒绝。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    with pytest.raises(ValueError):
        controller.scroll("left", 1)


def test_keyboard_scroll_invalid_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """scroll 非正步数被拒绝。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    with pytest.raises(ValueError):
        controller.scroll("down", 0)


def test_mouse_click_out_of_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """click 超出虚拟桌面范围抛出 ValueError。"""
    controller, backend = _make_mouse(monkeypatch)
    with pytest.raises(ValueError):
        controller.click(200, 20)


def test_mouse_move_to_out_of_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """move_to 超出虚拟桌面范围抛出 ValueError。"""
    controller, backend = _make_mouse(monkeypatch)
    with pytest.raises(ValueError):
        controller.move_to(200, 20)


def test_mouse_drag_out_of_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """drag_from_to 起点越界抛出 ValueError。"""
    controller, backend = _make_mouse(monkeypatch)
    with pytest.raises(ValueError):
        controller.drag_from_to(200, 10, 50, 50)


def test_mouse_invalid_button(monkeypatch: pytest.MonkeyPatch) -> None:
    """click 无效按钮名抛出 ValueError。"""
    controller, backend = _make_mouse(monkeypatch)
    with pytest.raises(ValueError):
        controller.click(10, 20, button="middle")  # type: ignore[arg-type]


def test_mouse_drag_with_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """drag_with duration>0 执行插值路径。"""
    controller, backend = _make_mouse(monkeypatch)
    controller.drag_from_to(10, 10, 50, 50, duration=0.05)
    assert ("press", "left") in backend.events
    assert ("release", "left") in backend.events
    assert len([e for e in backend.events if e[0] == "move"]) > 1


def test_keyboard_type_via_windows_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """type 在 win32 通过 Windows 文本后端发送非控制字符。"""
    controller, keyboard = _make_keyboard(monkeypatch)

    sent_units: list = []

    class _FakeTextBackend:
        def send(self, code_units: tuple[int, ...]) -> None:
            sent_units.append(code_units)

    # 通过 monkeypatch _create_windows_text_backend 注入 fake,
    # 覆盖 _get_text_backend 的 lazy creation 路径。
    monkeypatch.setattr(
        keyboard_module,
        "_create_windows_text_backend",
        lambda: _FakeTextBackend(),
    )
    controller.type("AB")
    assert len(sent_units) >= 1


def test_keyboard_hotkey_single_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """hotkey 单键执行 press + release。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    controller.hotkey("a")
    assert ("press", "a") in keyboard.events
    assert ("release", "a") in keyboard.events


def test_keyboard_type_tab_uses_named_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """type 中 tab 字符走命名键路径,不经过 Windows text backend。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    keys = SimpleNamespace(tab="TAB")
    monkeypatch.setattr(controller, "_keys", keys)
    controller.type("\t")
    assert ("press", "TAB") in keyboard.events
    assert ("release", "TAB") in keyboard.events


def test_keyboard_type_enter_uses_named_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """type 中换行字符走 enter 键路径。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    keys = SimpleNamespace(enter="ENTER")
    monkeypatch.setattr(controller, "_keys", keys)
    controller.type("\n")
    assert ("press", "ENTER") in keyboard.events


def test_keyboard_type_surrogate_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """type 中孤立 UTF-16 代理项被拒绝。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    with pytest.raises(ValueError):
        controller.type("\ud800")


def test_keyboard_type_empty_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """type 空字符串不产生任何后端调用。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    controller.type("")
    assert keyboard.events == []


def test_keyboard_type_non_string_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """type 非 string 输入被拒绝。"""
    controller, keyboard = _make_keyboard(monkeypatch)
    with pytest.raises(TypeError):
        controller.type(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dispatcher authorized dispatch for type/scroll/hotkey
# ---------------------------------------------------------------------------


def test_dispatcher_authorized_type_dispatches() -> None:
    """授权 scope 下 type 动作分发到控制器。"""
    calls: list[tuple[object, ...]] = []

    class Controls:
        def click(self, *a, **kw):
            calls.append(("click", *a))

        def right_click(self, *a, **kw):
            calls.append(("right_click", *a))

        def double_click(self, *a, **kw):
            calls.append(("double_click", *a))

        def drag_from_to(self, *a, **kw):
            calls.append(("drag_from_to", *a))

        def type(self, text: str) -> None:
            calls.append(("type", text))

        def scroll(self, d: str, s: int) -> None:
            calls.append(("scroll", d, s))

        def hotkey(self, *keys: str) -> None:
            calls.append(("hotkey", *keys))

    controls = Controls()
    dispatcher = ActionDispatcher(controls, controls)
    dispatcher.activate_run_scope(
        PermissionScope(
            allowed_actions=frozenset({"click", "type", "scroll", "hotkey"}),
            token=1,
        ),
    )
    assert dispatcher.dispatch(
        {"action_type": "type", "params": {"text": "hello"}},
        (100, 100),
    )
    assert ("type", "hello") in calls


def test_dispatcher_authorized_scroll_dispatches() -> None:
    """授权 scope 下 scroll 动作分发到控制器。"""
    calls: list[tuple[object, ...]] = []

    class Controls:
        def click(self, *a, **kw):
            pass

        def right_click(self, *a, **kw):
            pass

        def double_click(self, *a, **kw):
            pass

        def drag_from_to(self, *a, **kw):
            pass

        def type(self, *a, **kw):
            pass

        def scroll(self, d: str, s: int) -> None:
            calls.append(("scroll", d, s))

        def hotkey(self, *a, **kw):
            pass

    controls = Controls()
    dispatcher = ActionDispatcher(controls, controls)
    dispatcher.activate_run_scope(
        PermissionScope(
            allowed_actions=frozenset({"scroll"}),
            token=1,
        ),
    )
    assert dispatcher.dispatch(
        {"action_type": "scroll", "params": {"direction": "up", "steps": 3}},
        (100, 100),
    )
    assert ("scroll", "up", 3) in calls


def test_dispatcher_authorized_hotkey_dispatches() -> None:
    """授权 scope 下 hotkey 动作分发到控制器。"""
    calls: list[tuple[object, ...]] = []

    class Controls:
        def click(self, *a, **kw):
            pass

        def right_click(self, *a, **kw):
            pass

        def double_click(self, *a, **kw):
            pass

        def drag_from_to(self, *a, **kw):
            pass

        def type(self, *a, **kw):
            pass

        def scroll(self, *a, **kw):
            pass

        def hotkey(self, *keys: str) -> None:
            calls.append(("hotkey", *keys))

    controls = Controls()
    dispatcher = ActionDispatcher(controls, controls)
    dispatcher.activate_run_scope(
        PermissionScope(
            allowed_actions=frozenset({"hotkey"}),
            token=1,
        ),
    )
    assert dispatcher.dispatch(
        {"action_type": "hotkey", "params": {"keys": ("ctrl", "c")}},
        (100, 100),
    )
    assert ("hotkey", "ctrl", "c") in calls


def test_dispatcher_finish_valid_returns_true() -> None:
    """finish 动作在 dispatcher 中返回 True。"""
    controls = type(
        "C",
        (),
        {
            "click": lambda *a: None,
            "right_click": lambda *a: None,
            "double_click": lambda *a: None,
            "drag_from_to": lambda *a: None,
            "type": lambda *a: None,
            "scroll": lambda *a: None,
            "hotkey": lambda *a: None,
        },
    )()
    dispatcher = ActionDispatcher(controls, controls)
    assert dispatcher.dispatch(
        {"action_type": "finish", "params": {"result": "done"}},
        (100, 100),
    )


def test_dispatcher_maps_normalized_click_to_region_pixels() -> None:
    """0..1000 相对坐标映射到截图像素后再叠加 region offset。"""
    calls: list[tuple[object, ...]] = []

    class Controls:
        def click(self, *args: object, **kwargs: object) -> None:
            calls.append(("click", *args))

        def right_click(self, *args: object) -> None:
            calls.append(("right_click", *args))

        def double_click(self, *args: object) -> None:
            calls.append(("double_click", *args))

        def drag_from_to(self, *args: object) -> None:
            calls.append(("drag_from_to", *args))

        def type(self, *args: object) -> None:
            pass

        def scroll(self, *args: object) -> None:
            pass

        def hotkey(self, *args: object) -> None:
            pass

    controls = Controls()
    dispatcher = ActionDispatcher(
        controls,
        controls,
        coordinate_mode="normalized_1000",
    )
    dispatcher.activate_run_scope(
        PermissionScope(allowed_actions=frozenset({"click"}), token=1),
    )
    assert dispatcher.dispatch(
        {"action_type": "click", "params": {"x": 500, "y": 1000}},
        (201, 101),
        (40, 60),
    )
    assert calls == [("click", 140, 160)]


def test_dispatcher_rejects_invalid_coordinate_mode() -> None:
    """未知坐标模式在产生控制副作用前拒绝。"""
    controls = type(
        "C",
        (),
        {
            "click": lambda *args: None,
            "right_click": lambda *args: None,
            "double_click": lambda *args: None,
            "drag_from_to": lambda *args: None,
            "type": lambda *args: None,
            "scroll": lambda *args: None,
            "hotkey": lambda *args: None,
        },
    )()
    with pytest.raises(ValueError):
        ActionDispatcher(controls, controls, coordinate_mode="auto")


def test_dispatcher_invalid_action_dict() -> None:
    """非 dict action 被 dispatcher 拒绝。"""
    controls = type(
        "C",
        (),
        {
            "click": lambda *a: None,
            "right_click": lambda *a: None,
            "double_click": lambda *a: None,
            "drag_from_to": lambda *a: None,
            "type": lambda *a: None,
            "scroll": lambda *a: None,
            "hotkey": lambda *a: None,
        },
    )()
    dispatcher = ActionDispatcher(controls, controls)
    assert (
        dispatcher.dispatch(
            "not_dict",
            (100, 100),  # type: ignore[arg-type]
        )
        is False
    )


def test_dispatcher_params_not_dict() -> None:
    """params 非 dict 被 dispatcher 拒绝。"""
    controls = type(
        "C",
        (),
        {
            "click": lambda *a: None,
            "right_click": lambda *a: None,
            "double_click": lambda *a: None,
            "drag_from_to": lambda *a: None,
            "type": lambda *a: None,
            "scroll": lambda *a: None,
            "hotkey": lambda *a: None,
        },
    )()
    dispatcher = ActionDispatcher(controls, controls)
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": "bad"},  # type: ignore[dict-item]
            (100, 100),
        )
        is False
    )


def test_dispatcher_invalid_screenshot_size() -> None:
    """无效截图尺寸被 dispatcher 拒绝。"""
    controls = type(
        "C",
        (),
        {
            "click": lambda *a: None,
            "right_click": lambda *a: None,
            "double_click": lambda *a: None,
            "drag_from_to": lambda *a: None,
            "type": lambda *a: None,
            "scroll": lambda *a: None,
            "hotkey": lambda *a: None,
        },
    )()
    dispatcher = ActionDispatcher(controls, controls)
    dispatcher.activate_run_scope(
        PermissionScope(allowed_actions=frozenset({"click"}), token=1),
    )
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": {"x": 500, "y": 500}},
            (-1, 100),  # type: ignore[arg-type]
        )
        is False
    )


@pytest.mark.parametrize("key", ["win", "cmd"])
def test_system_key_aliases_are_supported(key: str) -> None:
    """win 与 cmd 都是合法系统键名,win 经别名映射到同一后端键。"""
    assert is_supported_key(key)
