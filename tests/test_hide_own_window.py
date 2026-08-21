"""run 期间最小化自身控制窗口(benchmark 去污染)的行为与安全语义测试。

覆盖:feature flag 默认值、启用路径、最小化发生在首次截图前、run 结束
恢复、最小化/可见两态下的受保护区域判定(不得残留旧矩形误拦任务窗口)。
"""

import asyncio

from agentscope.message import Msg
from PIL import Image

from agent import gui_agent as gui_agent_module
from agent.task_manager import TaskManager
from config import GuiAgentSettings, hide_own_window_during_run_from_env
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent

_FINISH = 'Action: finish(result="done")'
_OWN_HWND = 4242


def _click_action(x: int, y: int) -> dict[str, object]:
    return {"action_type": "click", "params": {"x": x, "y": y}}


def test_hide_flag_defaults_to_false(monkeypatch) -> None:
    """GuiAgentSettings 与 env 读取器默认关闭隐藏自身窗口。"""
    monkeypatch.delenv("GUI_AGENT_HIDE_OWN_WINDOW_DURING_RUN", raising=False)
    assert GuiAgentSettings().hide_own_window_during_run is False
    assert hide_own_window_during_run_from_env() is False


def test_hide_flag_enabled_explicitly(monkeypatch) -> None:
    """显式设置环境变量开启;非真值不开启。"""
    monkeypatch.setenv("GUI_AGENT_HIDE_OWN_WINDOW_DURING_RUN", "1")
    assert hide_own_window_during_run_from_env() is True
    monkeypatch.setenv("GUI_AGENT_HIDE_OWN_WINDOW_DURING_RUN", "0")
    assert hide_own_window_during_run_from_env() is False


def test_hide_minimizes_before_first_capture_and_restores(monkeypatch) -> None:
    """开启后:先最小化自身窗口再首次截图,run 结束恢复原窗口。"""
    events: list[str] = []

    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        lambda: _OWN_HWND,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "minimize_window",
        lambda hwnd: events.append(f"minimize:{hwnd}") or True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "activate_window",
        lambda hwnd: events.append(f"restore:{hwnd}") or True,
    )

    def capture(**kwargs):
        events.append("capture")
        return Image.new("RGB", (1000, 500))

    sleeps: list[float] = []
    backend = SequenceBackend([_FINISH])
    agent = make_agent(
        backend,
        MemoryControls(),
        TaskManager("打开记事本"),
        capture=capture,
        model_mode="api",
        reject_initial_finish=False,
        hide_own_window=True,
    )
    agent._dependencies = type(agent._dependencies)(
        model_client=agent._dependencies.model_client,
        action_dispatcher=agent._dependencies.action_dispatcher,
        capture=capture,
        task_manager_factory=agent._dependencies.task_manager_factory,
        action_observer=agent._dependencies.action_observer,
        sleep=sleeps.append,
        protect_initial_foreground=agent._dependencies.protect_initial_foreground,
        ocr_recognizer=agent._dependencies.ocr_recognizer,
        diagnostics_writer=agent._dependencies.diagnostics_writer,
        trace_writer=agent._dependencies.trace_writer,
    )
    result = asyncio.run(agent(Msg("u", "打开记事本", "user")))
    assert result.content == "done"
    assert events[0] == f"minimize:{_OWN_HWND}"
    assert events[1] == "capture"
    assert events[-1] == f"restore:{_OWN_HWND}"
    assert events.count(f"minimize:{_OWN_HWND}") == 1
    assert events.count(f"restore:{_OWN_HWND}") == 1
    # 最小化后执行了一次有界的桌面重绘等待。
    assert 0.5 in sleeps


def test_explicit_agent_ui_identity_is_not_replaced_by_target(monkeypatch) -> None:
    """CLI 已激活目标后，Agent 仍只隐藏 metadata 指定的 CLI 窗口。"""
    target_hwnd = 5151
    events: list[str] = []
    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        lambda: target_hwnd,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "minimize_window",
        lambda hwnd: events.append(f"minimize:{hwnd}") or True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "activate_window",
        lambda hwnd: events.append(f"restore:{hwnd}") or True,
    )
    agent = make_agent(
        SequenceBackend([_FINISH]),
        MemoryControls(),
        TaskManager("使用当前浏览器"),
        model_mode="api",
        reject_initial_finish=False,
        hide_own_window=True,
    )
    metadata = {
        "task_target_window": {"hwnd": target_hwnd, "process": "chrome.exe"},
        "agent_ui_window_hwnd": _OWN_HWND,
    }
    asyncio.run(agent(Msg("u", "使用当前浏览器", "user", metadata=metadata)))
    assert events == [f"minimize:{_OWN_HWND}", f"restore:{_OWN_HWND}"]


def test_hide_disabled_by_default_no_minimize(monkeypatch) -> None:
    """默认关闭:不最小化、不恢复。"""
    events: list[str] = []
    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        lambda: _OWN_HWND,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "minimize_window",
        lambda hwnd: events.append("minimize") or True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "activate_window",
        lambda hwnd: events.append("restore") or True,
    )
    agent = make_agent(
        SequenceBackend([_FINISH]),
        MemoryControls(),
        TaskManager("打开记事本"),
        model_mode="api",
        reject_initial_finish=False,
    )
    asyncio.run(agent(Msg("u", "打开记事本", "user")))
    assert events == []


def test_hide_restores_after_failed_run(monkeypatch) -> None:
    """任务失败路径同样恢复窗口(生命周期 try/finally 保护)。"""
    events: list[str] = []
    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        lambda: _OWN_HWND,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "minimize_window",
        lambda hwnd: events.append("minimize") or True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "activate_window",
        lambda hwnd: events.append("restore") or True,
    )
    agent = make_agent(
        SequenceBackend(["不是合法动作"]),
        MemoryControls(),
        TaskManager("打开记事本"),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        hide_own_window=True,
    )
    asyncio.run(agent(Msg("u", "打开记事本", "user")))
    assert events == ["minimize", "restore"]


def test_hide_skipped_when_no_own_hwnd(monkeypatch) -> None:
    """任务提交时前台不是应用窗口(hwnd=0)时不最小化。"""
    events: list[str] = []
    monkeypatch.setattr(gui_agent_module, "get_foreground_app_hwnd", lambda: 0)
    monkeypatch.setattr(
        gui_agent_module,
        "minimize_window",
        lambda hwnd: events.append("minimize") or True,
    )
    agent = make_agent(
        SequenceBackend([_FINISH]),
        MemoryControls(),
        TaskManager("打开记事本"),
        model_mode="api",
        reject_initial_finish=False,
        hide_own_window=True,
    )
    asyncio.run(agent(Msg("u", "打开记事本", "user")))
    assert events == []


def test_visible_own_window_rect_still_rejects_click(monkeypatch) -> None:
    """自身窗口可见时,命中其实时矩形的点击仍被安全拒绝。"""
    agent = make_agent(
        SequenceBackend([_FINISH]),
        MemoryControls(),
        TaskManager("打开记事本"),
        protect_initial_foreground=True,
    )
    agent._agent_ui_hwnd = _OWN_HWND
    monkeypatch.setattr(
        gui_agent_module,
        "get_window_screen_rect",
        lambda hwnd: (100, 600, 800, 400),
    )
    # click(x=500, y=700) 在 1000x1000 截图内映射到像素 (500, 699),
    # 位于受保护矩形 (100,600)-(900,1000) 内。
    failure = agent._protected_foreground_click_failure(
        _click_action(500, 700),
        (1000, 1000),
        (0, 0),
        False,
    )
    assert failure is not None


def test_minimized_own_window_rect_none_does_not_reject(monkeypatch) -> None:
    """自身窗口最小化时实时 rect 为 None,不得用旧矩形拦截任务窗口。"""
    agent = make_agent(
        SequenceBackend([_FINISH]),
        MemoryControls(),
        TaskManager("打开记事本"),
        protect_initial_foreground=True,
    )
    agent._agent_ui_hwnd = _OWN_HWND
    monkeypatch.setattr(
        gui_agent_module,
        "get_window_screen_rect",
        lambda hwnd: None,
    )
    failure = agent._protected_foreground_click_failure(
        _click_action(500, 699),
        (1000, 500),
        (0, 0),
        False,
    )
    assert failure is None


def test_minimized_then_visible_protection_resumes(monkeypatch) -> None:
    """最小化(放行)后窗口重新可见,同一坐标再次受保护。"""
    agent = make_agent(
        SequenceBackend([_FINISH]),
        MemoryControls(),
        TaskManager("打开记事本"),
        protect_initial_foreground=True,
    )
    agent._agent_ui_hwnd = _OWN_HWND
    rect_state = {"rect": None}

    def _fake_rect(hwnd):
        return rect_state["rect"]

    monkeypatch.setattr(gui_agent_module, "get_window_screen_rect", _fake_rect)
    action = _click_action(500, 700)
    assert (
        agent._protected_foreground_click_failure(
            action,
            (1000, 1000),
            (0, 0),
            False,
        )
        is None
    )
    rect_state["rect"] = (100, 600, 800, 400)
    assert (
        agent._protected_foreground_click_failure(
            action,
            (1000, 1000),
            (0, 0),
            False,
        )
        is not None
    )
