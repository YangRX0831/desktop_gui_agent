"""Agent 安全层与任务目标绑定测试。"""

import asyncio
import threading

import pytest
from agentscope.message import Msg
from PIL import Image

from agent.task_manager import TaskManager, TaskStatus
from perception import prompt_context as pc
from tests.agent_test_support import (
    FrameSequenceCapture,
    MemoryControls,
    SequenceBackend,
    _make_timeline,
    make_agent,
)


def test_target_window_loss_retries_without_follow_up_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """起始目标关闭后不得把后续动作发送到其他窗口或桌面。"""
    from agent import gui_agent as ga

    backend = SequenceBackend(
        [
            'Action: hotkey(key1="alt", key2="f4")',
            'Action: hotkey(key1="alt", key2="f4")',
            'Action: finish(result="窗口已关闭")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("关闭当前窗口")
    window_states = iter((True, False))
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: next(window_states))

    result = asyncio.run(
        make_agent(backend, controls, manager)(
            Msg("u", "关闭当前窗口", "user"),
        ),
    )

    assert result.content == "窗口已关闭"
    assert controls.calls == [("hotkey", "alt", "f4")]
    assert backend.calls == 3
    assert "任务目标窗口已关闭" in backend.prompts[2]
    assert manager.state.retry_count == 1
    assert manager.state.status is TaskStatus.SUCCESS


def test_desktop_type_retries_without_text_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """桌面无应用焦点时拒绝 type，并把事实反馈给 fresh 决策。"""
    from agent import gui_agent as ga

    backend = SequenceBackend(
        [
            'Action: type(text="query")',
            'Action: hotkey(key1="cmd")',
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("搜索关键词")
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 0)

    result = asyncio.run(
        make_agent(backend, controls, manager)(
            Msg("u", "搜索关键词", "user"),
        ),
    )

    assert result.content == "done"
    assert controls.calls == [("hotkey", "cmd")]
    assert "不能直接输入文本" in backend.prompts[1]
    assert manager.state.retry_count == 1


def test_cli_foreground_rejects_type_until_focus_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI 任务输入窗口不能被模型当作目标文本框。"""
    from agent import gui_agent as ga

    backend = SequenceBackend(
        [
            'Action: type(text="打开记事本")',
            'Action: hotkey(key1="cmd", key2="s")',
            'Action: finish(result="目标界面已打开")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("打开目标应用")
    observed: list[str] = []
    app_foregrounds = iter((100, 100, 100, 200))
    monkeypatch.setattr(
        ga,
        "get_foreground_app_hwnd",
        lambda: next(app_foregrounds),
    )
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)

    result = asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            protect_initial_foreground=True,
            action_observer=lambda step, action: observed.append(
                action["action_type"],
            ),
        )(Msg("u", "打开目标应用", "user")),
    )

    assert result.content == "目标界面已打开"
    assert controls.calls == [("hotkey", "cmd", "s")]
    assert observed == ["hotkey", "finish"]
    assert "本智能体自身的命令行界面" in backend.prompts[1]
    assert manager.state.retry_count == 1


def test_cli_foreground_rejects_unverified_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未离开 CLI 时即使已有键盘动作也不能声明GUI任务完成。"""
    from agent import gui_agent as ga

    backend = SequenceBackend(
        [
            'Action: hotkey(key1="enter")',
            'Action: finish(result="已完成")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("打开目标应用")
    observed: list[str] = []
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)

    result = asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            max_steps=2,
            retry_count=0,
            protect_initial_foreground=True,
            action_observer=lambda step, action: observed.append(
                action["action_type"],
            ),
        )(Msg("u", "打开目标应用", "user")),
    )

    assert result.content == "任务执行失败：任务达到最大执行步数。"
    assert controls.calls == [("hotkey", "enter")]
    assert observed == ["hotkey"]
    assert manager.state.steps[1].attempts[0].stage == "finish"
    assert "没有可验证的目标界面进展" in (
        manager.state.steps[1].attempts[0].failure_reason or ""
    )


def test_initial_foreground_unlocks_after_major_content_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """起始前台自身发生大幅内容变化后解除 type 与 finish 只读保护。"""
    from agent import gui_agent as ga

    black = Image.new("RGB", (40, 40), "black")
    navigated = Image.new("RGB", (40, 40), "white")
    capture = FrameSequenceCapture(
        [black, navigated, navigated, navigated, navigated, navigated],
    )
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=10)",
            'Action: type(text="query")',
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("搜索关键词")
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: True)
    monkeypatch.setattr(ga, "get_window_process_name", lambda hwnd: "App.exe")
    monkeypatch.setattr(ga, "select_capture_region", lambda hwnd: (None, (0, 0)))

    result = asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            capture=capture,
            max_steps=3,
            retry_count=0,
            protect_initial_foreground=True,
            verify_action_effect=True,
        )(Msg("u", "搜索关键词", "user")),
    )

    assert result.content == "done"
    assert controls.calls == [
        ("click", 10, 10, "left"),
        ("type", "query"),
    ]


def test_initial_foreground_stays_locked_after_small_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """低于解锁阈值的局部变化不解除起始前台只读保护。"""
    from agent import gui_agent as ga

    black = Image.new("RGB", (40, 40), "black")
    slightly_changed = black.copy()
    for x in range(5):
        for y in range(5):
            slightly_changed.putpixel((x, y), (255, 255, 255))
    capture = FrameSequenceCapture(
        [black, slightly_changed, slightly_changed, slightly_changed],
    )
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=10)",
            'Action: type(text="query")',
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("搜索关键词")
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: True)
    monkeypatch.setattr(ga, "get_window_process_name", lambda hwnd: "App.exe")
    monkeypatch.setattr(ga, "select_capture_region", lambda hwnd: (None, (0, 0)))

    result = asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            capture=capture,
            max_steps=3,
            retry_count=0,
            protect_initial_foreground=True,
            verify_action_effect=True,
        )(Msg("u", "搜索关键词", "user")),
    )

    assert result.content == "任务执行失败：任务达到最大执行步数。"
    assert controls.calls == [("click", 10, 10, "left")]
    type_failure = manager.state.steps[1].attempts[0].failure_reason or ""
    assert "本智能体自身的命令行界面" in type_failure
    finish_failure = manager.state.steps[2].attempts[0].failure_reason or ""
    assert "本智能体自身的命令行界面" in finish_failure


def test_repeated_successful_type_is_not_dispatched_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """连续相同 type 在首次已分发后必须重新决策，避免文本叠加。"""
    from agent import gui_agent as ga

    backend = SequenceBackend(
        [
            'Action: type(text="query")',
            'Action: type(text="query")',
            'Action: hotkey(key1="enter")',
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("输入并提交")
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)

    result = asyncio.run(
        make_agent(backend, controls, manager)(
            Msg("u", "输入并提交", "user"),
        ),
    )

    assert result.content == "done"
    assert controls.calls == [
        ("type", "query"),
        ("hotkey", "enter"),
    ]
    assert "相同文本已经分发" in backend.prompts[2]
    assert manager.state.retry_count == 1


def test_finish_remains_allowed_after_close_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭动作后仍可接收无副作用 finish，正常结束任务。"""
    from agent import gui_agent as ga

    backend = SequenceBackend(
        [
            'Action: hotkey(key1="alt", key2="f4")',
            'Action: finish(result="窗口已关闭")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("关闭当前窗口")
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)

    result = asyncio.run(
        make_agent(backend, controls, manager)(
            Msg("u", "关闭当前窗口", "user"),
        ),
    )

    assert result.content == "窗口已关闭"
    assert controls.calls == [("hotkey", "alt", "f4")]
    assert manager.state.status is TaskStatus.SUCCESS


def test_desktop_alt_f4_is_blocked_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """桌面 Shell 上的 Alt+F4 不得打开 Windows 关机对话框。"""
    from agent import gui_agent as ga

    backend = SequenceBackend(['Action: hotkey(key1="alt", key2="f4")'])
    controls = MemoryControls()
    manager = TaskManager("关闭当前窗口")
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 0)

    result = asyncio.run(
        make_agent(backend, controls, manager)(
            Msg("u", "关闭当前窗口", "user"),
        ),
    )

    assert result.content == "任务执行失败：桌面环境禁止执行关闭窗口快捷键。"
    assert controls.calls == []
    assert manager.state.status is TaskStatus.FAILED


@pytest.mark.parametrize(
    ("focus_kind", "protect", "unlock", "expected"),
    [
        ("none", False, False, "false"),
        ("other", False, False, "unknown"),
        ("unknown", False, False, "unknown"),
        ("text_input", False, False, "true"),
        ("text_input", True, False, "false"),
        ("text_input", True, True, "true"),
    ],
)
def test_keyboard_input_ready_combines_focus_and_protection(
    monkeypatch: pytest.MonkeyPatch,
    focus_kind: str,
    protect: bool,
    unlock: bool,
    expected: str,
) -> None:
    """keyboard_input_ready 由焦点控件类别叠加起始前台保护计算。"""
    monkeypatch.setattr(
        pc,
        "get_foreground_app_hwnd",
        lambda: 100,
    )

    assert pc.keyboard_input_ready(focus_kind, protect, 100, unlock) == expected


def test_cli_foreground_rejects_alt_f4_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """受保护起始前台上的 alt+f4 被拒绝,避免关闭命令行自身。"""
    from agent import gui_agent as ga

    agent = make_agent(
        SequenceBackend(['Action: finish(result="unused")']),
        MemoryControls(),
        TaskManager("关闭当前窗口"),
        protect_initial_foreground=True,
    )
    agent._agent_ui_hwnd = 100
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    state = ga.ActionPromptState(step_number=1, max_steps=10)
    action = {"action_type": "hotkey", "params": {"keys": ("alt", "f4")}}

    failure = agent._get_dispatch_safety_failure(action, state, False)
    assert (
        failure
        == "当前聚焦的是本智能体自身的命令行界面，禁止对其使用关闭或系统菜单快捷键。"
    )

    unlocked_failure = agent._get_dispatch_safety_failure(action, state, True)
    assert unlocked_failure is None


def test_cli_foreground_rejects_clicks_inside_protected_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未解锁时落在起始前台窗口矩形内的鼠标动作被拒绝,矩形外放行。"""
    from agent import gui_agent as ga

    agent = make_agent(
        SequenceBackend(['Action: finish(result="unused")']),
        MemoryControls(),
        TaskManager("关闭当前窗口"),
        protect_initial_foreground=True,
    )
    agent._agent_ui_hwnd = 100
    # 受保护窗口物理矩形 (0,0)-(500,400);截图尺寸 1000x500,offset 0。
    monkeypatch.setattr(ga, "get_window_screen_rect", lambda hwnd: (0, 0, 500, 400))
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)

    inside = {"action_type": "click", "params": {"x": 250, "y": 200}}
    assert (
        ga.GuiAgent._protected_foreground_click_failure(
            agent,
            inside,
            (1000, 500),
            (0, 0),
            False,
        )
        == "坐标落在本智能体自身的命令行界面内，该界面只读，请勿再在此区域执行鼠标动作。"
    )

    outside = {"action_type": "click", "params": {"x": 900, "y": 200}}
    assert (
        ga.GuiAgent._protected_foreground_click_failure(
            agent,
            outside,
            (1000, 500),
            (0, 0),
            False,
        )
        is None
    )

    drag_inside = {
        "action_type": "drag",
        "params": {"x1": 900, "y1": 100, "x2": 250, "y2": 200},
    }
    assert (
        ga.GuiAgent._protected_foreground_click_failure(
            agent,
            drag_inside,
            (1000, 500),
            (0, 0),
            False,
        )
        is not None
    )

    unlocked = ga.GuiAgent._protected_foreground_click_failure(
        agent,
        inside,
        (1000, 500),
        (0, 0),
        True,
    )
    assert unlocked is None


@pytest.mark.parametrize(
    "keys",
    [("alt", "f4"), ("alt", "space"), ("ctrl", "w"), ("ctrl", "shift", "w")],
)
def test_close_hotkey_family_rejected_on_protected_foreground(
    monkeypatch: pytest.MonkeyPatch,
    keys: tuple[str, ...],
) -> None:
    """关闭类组合键族在受保护前台被拒;普通热键不受影响。"""
    from agent import gui_agent as ga

    agent = make_agent(
        SequenceBackend(['Action: finish(result="unused")']),
        MemoryControls(),
        TaskManager("关闭当前窗口"),
        protect_initial_foreground=True,
    )
    agent._agent_ui_hwnd = 100
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    state = ga.ActionPromptState(step_number=1, max_steps=10)

    action = {"action_type": "hotkey", "params": {"keys": keys}}
    assert agent._get_dispatch_safety_failure(action, state, False) is not None

    normal = {"action_type": "hotkey", "params": {"keys": ("win", "r")}}
    assert agent._get_dispatch_safety_failure(normal, state, False) is None


def test_agent_ui_window_state_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """agent_ui_window 序列化含 id/bbox/protected;几何不可得时bbox=unknown。"""
    agent = make_agent(
        SequenceBackend(['Action: finish(result="unused")']),
        MemoryControls(),
        TaskManager("t"),
        protect_initial_foreground=True,
    )
    agent._agent_ui_hwnd = 100
    monkeypatch.setattr(pc, "get_window_screen_rect", lambda h: (100, 100, 400, 200))
    line = pc.agent_ui_window_state(100, (1000, 500), (0, 0))
    assert line == "id:100, bbox:(100, 200, 500, 600), protected:true"

    monkeypatch.setattr(pc, "get_window_screen_rect", lambda h: None)
    line2 = pc.agent_ui_window_state(100, (1000, 500), (0, 0))
    assert line2 == "id:100, bbox:unknown, protected:true"


def test_task_target_window_state_and_close_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标窗口序列化含存在性;目标消失时效果优先判window_closed。"""
    from agent import gui_agent as ga

    agent = make_agent(
        SequenceBackend(['Action: finish(result="unused")']),
        MemoryControls(),
        TaskManager("关闭当前窗口"),
        protect_initial_foreground=True,
    )
    agent._task_target = {"hwnd": 555, "process": "notepad.exe"}
    agent._agent_ui_hwnd = 100
    monkeypatch.setattr(pc, "get_window_screen_rect", lambda h: (0, 0, 500, 400))
    monkeypatch.setattr(pc, "is_window_existing", lambda hwnd: hwnd != 555)
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)

    line = pc.task_target_window_state(agent._task_target, (1000, 500), (0, 0))
    assert "id:555, process:notepad.exe" in line
    assert "state:closed" in line

    observation = agent._verify_action_effect(
        "hotkey",
        __import__("PIL.Image", fromlist=["Image"]).new("RGB", (40, 40)),
        100,
        100,
    )
    assert observation.effect == "window_closed"


def test_case1_target_is_business_window_not_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case1: CLI抢焦点后提交任务,目标回溯为业务窗口而非终端。"""
    import main as main_module

    timeline = _make_timeline((111, "notepad.exe"), (222, "WindowsTerminal.exe"))
    monkeypatch.setattr(main_module, "get_foreground_app_hwnd", lambda: 222)
    monkeypatch.setattr(main_module, "is_window_existing", lambda hwnd: True)

    target = main_module.resolve_task_target_window(timeline)
    assert target == {"hwnd": 111, "process": "notepad.exe"}


def test_demonstrative_window_phrase_is_relative_reference() -> None:
    """“那个…窗口”与“当前窗口”一样绑定提交前业务窗口。"""
    import main as main_module

    assert main_module._has_relative_window_reference(
        "关闭桌面上那个测试专用的空白记事本窗口。",
    )
    assert not main_module._has_relative_window_reference("打开记事本")


def test_visible_end_state_intent_is_narrow_and_explicit() -> None:
    """仅明确要求保留在界面/窗口的任务抑制成功后的 CLI 抢焦点。"""
    from agent.gui_agent import _task_requests_visible_end_state

    assert _task_requests_visible_end_state("让最终结果保留在计算器界面")
    assert _task_requests_visible_end_state("保持在当前窗口")
    assert not _task_requests_visible_end_state("打开计算器并计算1+1")


def test_case3_same_process_windows_distinguished_by_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case3: 两个同进程窗口按hwnd区分,绑定最近操作的那个。"""
    import main as main_module

    timeline = _make_timeline(
        (111, "notepad.exe"),
        (333, "notepad.exe"),
        (222, "WindowsTerminal.exe"),
    )
    monkeypatch.setattr(main_module, "get_foreground_app_hwnd", lambda: 222)
    monkeypatch.setattr(main_module, "is_window_existing", lambda hwnd: True)

    target = main_module.resolve_task_target_window(timeline)
    assert target == {"hwnd": 333, "process": "notepad.exe"}


def test_reference_task_minimizes_cli_and_activates_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """含相对指代任务:CLI被最小化、目标被激活、metadata随消息传递。"""
    import main as main_module

    calls: list[tuple[str, int]] = []

    class FakeAgent:
        def __init__(self) -> None:
            self.messages: list[object] = []

        async def __call__(self, msg):
            self.messages.append(msg)
            from agentscope.message import Msg

            return Msg("agent", "任务执行成功：ok", "assistant")

    fake_agent = FakeAgent()
    timeline = _make_timeline((111, "notepad.exe"), (222, "WindowsTerminal.exe"))
    monkeypatch.setattr(main_module, "get_foreground_app_hwnd", lambda: 222)
    monkeypatch.setattr(main_module, "is_window_existing", lambda hwnd: True)
    monkeypatch.setattr(
        main_module,
        "minimize_window",
        lambda h: calls.append(("min", h)) or True,
    )
    monkeypatch.setattr(
        main_module,
        "activate_window",
        lambda h: calls.append(("act", h)) or True,
    )
    monkeypatch.setattr(
        main_module,
        "start_foreground_timeline",
        lambda: (timeline, threading.Event()),
    )

    inputs = iter(["关闭当前窗口", "exit"])
    result = asyncio.run(
        main_module.run_cli(
            main_module.AppConfig(),
            read_input=lambda *_: next(inputs),
            write_output=lambda *_: None,
            agent_factory=lambda *_: fake_agent,
        ),
    )
    assert result == 0
    assert ("min", 222) in calls
    assert ("act", 111) in calls
    assert ("act", 222) in calls
    sent = fake_agent.messages[0]
    assert sent.metadata["task_target_window"] == {
        "hwnd": 111,
        "process": "notepad.exe",
    }
    assert sent.metadata["agent_ui_window_hwnd"] == 222


@pytest.mark.parametrize("reference", ["当前浏览器", "当前网页"])
def test_browser_reference_resolves_target(
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
) -> None:
    """浏览器与网页相对指代沿用通用前台目标解析。"""
    import main as main_module

    class FakeAgent:
        def __init__(self) -> None:
            self.messages: list[object] = []

        async def __call__(self, msg):
            self.messages.append(msg)
            from agentscope.message import Msg

            return Msg("agent", "ok", "assistant")

    fake_agent = FakeAgent()
    timeline = _make_timeline((111, "chrome.exe"), (222, "WindowsTerminal.exe"))
    monkeypatch.setattr(main_module, "get_foreground_app_hwnd", lambda: 222)
    monkeypatch.setattr(main_module, "is_window_existing", lambda hwnd: True)
    monkeypatch.setattr(main_module, "minimize_window", lambda hwnd: True)
    monkeypatch.setattr(main_module, "activate_window", lambda hwnd: True)
    monkeypatch.setattr(
        main_module,
        "start_foreground_timeline",
        lambda: (timeline, threading.Event()),
    )

    inputs = iter([f"使用{reference}完成任务", "exit"])
    asyncio.run(
        main_module.run_cli(
            main_module.AppConfig(),
            read_input=lambda *_: next(inputs),
            write_output=lambda *_: None,
            agent_factory=lambda *_: fake_agent,
        ),
    )
    assert fake_agent.messages[0].metadata == {
        "task_target_window": {"hwnd": 111, "process": "chrome.exe"},
        "agent_ui_window_hwnd": 222,
    }


def test_browser_reference_skips_newer_non_browser(monkeypatch) -> None:
    """当前浏览器按能力类别解析，不把更近的普通业务窗口误绑定为浏览器。"""
    import main as main_module

    timeline = _make_timeline(
        (111, "chrome.exe"),
        (333, "ChatGPT.exe"),
        (222, "WindowsTerminal.exe"),
    )
    monkeypatch.setattr(main_module, "get_foreground_app_hwnd", lambda: 222)
    monkeypatch.setattr(main_module, "is_window_existing", lambda hwnd: True)
    target = main_module.resolve_task_target_window(
        timeline,
        main_module._BROWSER_PROCESS_NAMES,
    )
    assert target == {"hwnd": 111, "process": "chrome.exe"}


def test_browser_reference_falls_back_to_visible_zorder(monkeypatch) -> None:
    """时间线未观察到浏览器时，回退到最上层可见浏览器窗口。"""
    import main as main_module

    timeline = _make_timeline(
        (333, "ChatGPT.exe"),
        (222, "WindowsTerminal.exe"),
    )
    monkeypatch.setattr(main_module, "get_foreground_app_hwnd", lambda: 222)
    monkeypatch.setattr(main_module, "is_window_existing", lambda hwnd: True)
    monkeypatch.setattr(
        main_module,
        "list_visible_windows_zorder",
        lambda: [
            {"hwnd": 333, "process": "ChatGPT.exe"},
            {"hwnd": 444, "process": "chrome.exe"},
        ],
    )
    target = main_module.resolve_task_target_window(
        timeline,
        main_module._BROWSER_PROCESS_NAMES,
    )
    assert target == {"hwnd": 444, "process": "chrome.exe"}


def test_non_reference_task_sends_no_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无相对指代的任务不解析目标,不影响前台。"""
    import main as main_module

    class FakeAgent:
        def __init__(self) -> None:
            self.messages: list[object] = []

        async def __call__(self, msg):
            self.messages.append(msg)
            from agentscope.message import Msg

            return Msg("agent", "ok", "assistant")

    fake_agent = FakeAgent()
    timeline = _make_timeline((111, "notepad.exe"), (222, "WindowsTerminal.exe"))
    monkeypatch.setattr(
        main_module,
        "start_foreground_timeline",
        lambda: (timeline, threading.Event()),
    )

    inputs = iter(["打开记事本", "exit"])
    asyncio.run(
        main_module.run_cli(
            main_module.AppConfig(),
            read_input=lambda *_: next(inputs),
            write_output=lambda *_: None,
            agent_factory=lambda *_: fake_agent,
        ),
    )
    assert fake_agent.messages[0].metadata in (None, {}, None)


def test_case4_target_closed_effect_and_no_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case4: 目标关闭后效果为window_closed,绑定不随前台重绑。"""
    from agent import gui_agent as ga

    agent = make_agent(
        SequenceBackend(['Action: finish(result="unused")']),
        MemoryControls(),
        TaskManager("关闭当前窗口"),
    )
    agent._task_target = {"hwnd": 555, "process": "notepad.exe"}
    agent._agent_ui_hwnd = 100

    # 目标已消失:状态行 state=closed;新前台(400)不改变绑定。
    monkeypatch.setattr(pc, "is_window_existing", lambda hwnd: hwnd != 555)
    monkeypatch.setattr(pc, "get_window_screen_rect", lambda h: None)
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 400)
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 400)
    line = pc.task_target_window_state(agent._task_target, (1000, 500), (0, 0))
    assert "id:555, process:notepad.exe" in line
    assert "state:closed" in line
