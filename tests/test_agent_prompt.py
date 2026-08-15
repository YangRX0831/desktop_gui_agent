"""动作 Prompt 构造与冻结测试。"""

import asyncio

import pytest
from agentscope.message import Msg
from PIL import Image

from agent.action_dispatcher import ActionDispatcher
from agent.action_parser import (
    ACTION_SYSTEM_PROMPT,
    PRD_ACTION_SYSTEM_PROMPT,
    ActionPromptState,
    compose_action_prompt,
)
from agent.task_manager import TaskManager
from perception import prompt_context as pc
from tests.agent_test_support import (
    FrameSequenceCapture,
    MemoryControls,
    SequenceBackend,
    make_agent,
)


def test_action_system_prompt_preserves_protocol_and_adds_guardrails() -> None:
    """分节 supplement 保留 PRD 协议、新动作语义且不含任务配方。"""
    assert ACTION_SYSTEM_PROMPT != PRD_ACTION_SYSTEM_PROMPT
    for marker in (
        "合法动作(唯一协议，每轮只输出一个)",
        "1. click(x=<整数>, y=<整数>) - 单击可见控件",
        "2. right_click(x=<整数>, y=<整数>) - 打开目标的上下文菜单",
        "3. double_click(x=<整数>, y=<整数>) - 双击打开或激活对象",
        "4. drag(x1=<整数>, y1=<整数>, x2=<整数>, y2=<整数>)",
        '8. finish(result="<结果描述>") - 任务完成',
        "必须以Action: 开头",
        "现在只输出Action",
        "不得猜测不可见目标的坐标",
        "不再用两个连续click模拟双击",
        "platform为windows时系统键使用win",
        "system_volume是当前系统主音量百分比",
        "keyboard_input_ready=true时可直接type",
        "OCR没有识别到某个控件不代表该控件不存在",
        "windows按层叠顺序自顶向下列出可见窗口",
        "任务只要求搜索关键词而未指明对象时，默认指网络搜索",
        "已经完成的子目标不得重复或回退",
        "能一次type完整输入时，不拆成多个click",
        "ui_change_signal只表示可观察UI变化程度",
        "agent_ui_window是智能体自身受保护的控制界面",
        "不得因前台变化重新绑定",
        "只有在程序可靠验证目标window id已关闭时才成立",
        "只有用户明确要求且当前目标对象与用户目标匹配时才执行",
        "click、right_click、double_click不得省略x=或y=",
    ):
        assert marker in ACTION_SYSTEM_PROMPT
    assert 'Action: type(text="Hello World")' not in ACTION_SYSTEM_PROMPT
    for benchmark_name in ("Chrome", "Python", "计算器", "记事本"):
        assert benchmark_name not in ACTION_SYSTEM_PROMPT
    for unsupported in ("press(", "release(", "move_to("):
        assert unsupported not in ACTION_SYSTEM_PROMPT


def test_production_prompt_keeps_prd_concat_contract() -> None:
    """动作 Prompt、运行状态和任务仍拼接为单个模型文本输入。"""
    state = ActionPromptState(
        step_number=2,
        max_steps=10,
        task_target_window=(
            "id:111, process:Notepad.exe, bbox:(0, 0, 100, 200), state:exists"
        ),
        agent_ui_window="id:222, bbox:(300, 0, 1000, 500), protected:true",
        last_action='type(text="query")',
        last_dispatch_status="success",
        last_error="none",
        foreground_after="SearchHost.exe",
        last_effect="foreground_window_changed",
        keyboard_input_ready="true",
        focused_control="text_input",
        ui_change_signal="strong",
        recent_actions=("hotkey:success:foreground_window_changed",),
        same_action_streak=1,
        platform="windows",
    )
    prompt = compose_action_prompt("打开浏览器", state, "normalized_1000")
    assert prompt.startswith(ACTION_SYSTEM_PROMPT)
    assert "step=2/10" in prompt
    assert "task_target_window=id:111, process:Notepad.exe" in prompt
    assert "agent_ui_window=id:222" in prompt
    assert 'last_action=type(text="query")' in prompt
    assert "last_dispatch_status=success" in prompt
    assert "last_error=none" in prompt
    assert "foreground_after=SearchHost.exe" in prompt
    assert "last_effect=foreground_window_changed" in prompt
    assert "ui_change_signal=strong" in prompt
    assert "recent_actions=hotkey:success:foreground_window_changed" in prompt
    assert "same_action_streak=1" in prompt
    assert "no_ui_change_streak=0" in prompt
    assert "platform=windows" in prompt
    assert "0到1000相对坐标" in prompt
    assert "用户指令：\n打开浏览器" in prompt
    assert prompt.endswith("现在只输出一行合法Action，不要输出其他内容。")

    unknown_state = ActionPromptState(step_number=1, max_steps=10)
    next_prompt = compose_action_prompt("打开浏览器", unknown_state, "image_pixel")
    assert "keyboard_input_ready=unknown" in next_prompt
    assert "图像像素坐标" in next_prompt


def test_compose_action_prompt_validates_public_inputs() -> None:
    """Prompt 公共拼装边界拒绝无效任务、状态和坐标模式。"""
    valid = ActionPromptState(step_number=1, max_steps=10)
    with pytest.raises(ValueError, match="task 不得为空"):
        compose_action_prompt(" ", valid, "normalized_1000")
    with pytest.raises(TypeError, match="ActionPromptState"):
        compose_action_prompt("任务", object(), "normalized_1000")
    invalid_step = ActionPromptState(step_number=0, max_steps=10)
    with pytest.raises(ValueError, match="step_number"):
        compose_action_prompt("任务", invalid_step, "normalized_1000")
    invalid_limit = ActionPromptState(step_number=2, max_steps=1)
    with pytest.raises(ValueError, match="不得超过"):
        compose_action_prompt("任务", invalid_limit, "normalized_1000")
    with pytest.raises(ValueError, match="coordinate_mode"):
        compose_action_prompt("任务", valid, "unknown")


def test_prd_baseline_remains_separate_from_production_optimization() -> None:
    """PRD 原文保留用于审计；production 顶部为统一 8 动作协议。"""
    assert "每次只输出一个动作" in PRD_ACTION_SYSTEM_PROMPT
    assert not ACTION_SYSTEM_PROMPT.startswith(PRD_ACTION_SYSTEM_PROMPT)
    assert "right_click(x=<整数>, y=<整数>)" in ACTION_SYSTEM_PROMPT
    assert "double_click(x=<整数>, y=<整数>)" in ACTION_SYSTEM_PROMPT
    assert "drag(x1=<整数>, y1=<整数>, x2=<整数>, y2=<整数>)" in ACTION_SYSTEM_PROMPT


def test_finish_works_without_scope() -> None:
    """finish 无桌面副作用,不需要 scope 即可返回 True。"""
    controls = MemoryControls()
    dispatcher = ActionDispatcher(controls, controls)
    assert (
        dispatcher.dispatch(
            {"action_type": "finish", "params": {"result": "done"}},
            (100, 100),
        )
        is True
    )
    assert controls.calls == []


def test_verified_window_close_reaches_next_prompt_as_observed_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """原前台 HWND 消失时只报告已验证关闭，并保持任务初始锚点。"""
    from agent import gui_agent as ga

    frame = Image.new("RGB", (40, 40), "black")
    capture = FrameSequenceCapture([frame.copy() for _ in range(4)])
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=10)",
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("关闭当前窗口")
    fg_calls = {"count": 0}

    def fake_foreground() -> int:
        fg_calls["count"] += 1
        return 100 if fg_calls["count"] == 1 else 200

    identities = {100: "Notepad.exe", 200: "ChatGPT.exe"}
    monkeypatch.setattr(ga, "get_foreground_hwnd", fake_foreground)
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: False)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    monkeypatch.setattr(
        ga,
        "get_window_process_name",
        lambda hwnd: identities.get(hwnd, ""),
    )
    monkeypatch.setattr(ga, "select_capture_region", lambda hwnd: (None, (0, 0)))

    result = asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            capture=capture,
            verify_action_effect=True,
        )(Msg("u", "关闭当前窗口", "user")),
    )

    assert result.content == "done"
    assert "last_action=click(x=10, y=10)" in backend.prompts[1]
    assert "foreground_after=ChatGPT.exe" in backend.prompts[1]
    assert "last_effect=window_closed" in backend.prompts[1]
    assert "ui_change_signal=strong" in backend.prompts[1]


def test_perception_to_agent_prompt_and_image_passed() -> None:
    """GuiAgent 把 screenshot + composed prompt 传给 ModelClient。"""
    from agent.action_parser import ACTION_SYSTEM_PROMPT

    captured: list[tuple] = []

    class PromptCaptureBackend:
        def __init__(self, response: str) -> None:
            self._response = response
            self.calls = 0

        def generate(self, image, prompt, mode="local"):
            self.calls += 1
            captured.append((image, prompt, mode))
            return self._response

    backend = PromptCaptureBackend('Action: finish(result="ok")')
    controls = MemoryControls()
    manager = TaskManager("任务")
    agent = make_agent(backend, controls, manager)

    asyncio.run(agent(Msg("u", "打开浏览器", "user")))

    assert backend.calls == 1
    img, prompt, mode = captured[0]
    # image 是 PIL.Image.Image 实例(fake capture 返回)。
    assert isinstance(img, Image.Image)
    # prompt 含批准的 PRD 协议优化 + 用户任务，仍是单个文本输入。
    assert prompt.startswith(ACTION_SYSTEM_PROMPT)
    assert "用户指令：\n打开浏览器" in prompt
    assert prompt.endswith("现在只输出一行合法Action，不要输出其他内容。")
    assert mode == "local"


def test_ocr_elements_flow_into_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """OCR 识别结果过滤、归一化后注入 Current perception 块。"""
    from agent import gui_agent as ga

    class FakeOCR:
        def recognize(self, image):
            return [
                {
                    "text": "计算器",
                    "bbox": (100, 200, 300, 250),
                    "confidence": 0.97,
                },
                {"text": "噪 声", "bbox": (0, 0, 10, 10), "confidence": 0.5},
            ]

    backend = SequenceBackend(
        [
            "Action: click(x=200, y=200)",
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("点击文字")
    agent = make_agent(
        backend,
        controls,
        manager,
        ocr_recognizer=FakeOCR(),
        max_steps=2,
        retry_count=0,
    )
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: True)
    monkeypatch.setattr(ga, "get_window_process_name", lambda hwnd: "App.exe")
    monkeypatch.setattr(ga, "select_capture_region", lambda hwnd: (None, (0, 0)))

    result = asyncio.run(agent(Msg("u", "点击文字", "user")))

    assert result.content == "done"
    prompt = backend.prompts[0]
    assert "Current perception:" in prompt
    assert 'text="计算器"' in prompt
    assert "bbox=(100, 400, 300, 500)" in prompt
    assert "噪声" not in prompt


def test_focus_control_state_flows_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每步焦点状态注入 Current execution state。"""
    from agent import gui_agent as ga

    monkeypatch.setattr(pc, "get_focus_control_kind", lambda: "text_input")
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=10)",
            'Action: finish(result="done")',
        ],
    )
    agent = make_agent(
        backend,
        MemoryControls(),
        TaskManager("点击后完成"),
        max_steps=2,
        retry_count=0,
    )
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 200)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: True)
    monkeypatch.setattr(ga, "get_window_process_name", lambda hwnd: "App.exe")
    monkeypatch.setattr(ga, "select_capture_region", lambda hwnd: (None, (0, 0)))

    result = asyncio.run(agent(Msg("u", "点击后完成", "user")))
    assert result.content == "done"
    assert "focused_control=text_input" in backend.prompts[0]
    assert "keyboard_input_ready=true" in backend.prompts[0]


def test_system_volume_flows_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每步读取系统主音量并注入动态状态。"""
    from agent import gui_agent as ga

    monkeypatch.setattr(pc, "system_volume_state", lambda: 78)
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=10)",
            'Action: finish(result="done")',
        ],
    )
    agent = make_agent(
        backend,
        MemoryControls(),
        TaskManager("调节音量"),
        max_steps=2,
        retry_count=0,
    )
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 200)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: True)
    monkeypatch.setattr(ga, "get_window_process_name", lambda hwnd: "App.exe")
    monkeypatch.setattr(ga, "select_capture_region", lambda hwnd: (None, (0, 0)))

    result = asyncio.run(agent(Msg("u", "调节音量", "user")))
    assert result.content == "done"
    assert "system_volume=78" in backend.prompts[0]


def test_windows_zorder_flows_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Z 序窗口列表换算为截图内相对矩形并注入 Current perception。"""
    monkeypatch.setattr(
        pc,
        "list_visible_windows_zorder",
        lambda: [
            {
                "hwnd": 1,
                "pid": 10,
                "process": "cmd.exe",
                "rect": (0, 0, 500, 400),
                "foreground": True,
            },
            {
                "hwnd": 2,
                "pid": 20,
                "process": "explorer.exe",
                "rect": (250, 200, 500, 300),
                "foreground": False,
            },
            {
                "hwnd": 3,
                "pid": 30,
                "process": "outside.exe",
                "rect": (2000, 2000, 100, 100),
                "foreground": False,
            },
        ],
    )
    entries = pc.perceive_windows((1000, 500), (0, 0))
    assert entries == (
        "id:1 fg=true process=cmd.exe bbox=(0, 0, 500, 800)",
        "id:2 fg=false process=explorer.exe bbox=(250, 400, 750, 1000)",
    )

    from agent.action_parser import ActionPromptState, compose_action_prompt

    state = ActionPromptState(step_number=1, max_steps=25, windows=entries)
    prompt = compose_action_prompt("t", state, "normalized_1000")
    assert "windows(按层叠顺序自顶向下, bbox为相对坐标):" in prompt
    assert "id:1 fg=true process=cmd.exe" in prompt
    assert "outside.exe" not in prompt
