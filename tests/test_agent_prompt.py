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
    """Production Prompt 明示 PRD 五动作协议且不含任务配方。"""
    assert ACTION_SYSTEM_PROMPT != PRD_ACTION_SYSTEM_PROMPT
    for marker in (
        "合法动作(唯一协议，只存在以下五种，每轮只输出一个)",
        "1. click(x=<整数>, y=<整数>) - 单击可见控件",
        '2. type(text="<文本>") - 输入文本',
        '3. scroll(direction="<up/down>", steps=<整数>) - 滚动屏幕',
        '4. hotkey(key1="<按键1>", key2="<按键2>", ...) - 按下组合键',
        '5. finish(result="<结果描述>") - 任务完成',
        "必须以Action: 开头",
        "参数名称不可省略",
        "禁止位置参数或命名参数与位置参数混用",
        "click必须同时包含",
        "x=<整数>和y=<整数>",
        "只能使用上述五种动作",
        "禁止发明drag",
        "正例：Action: click(x=123, y=456)",
        "反例：Action: click(x=123, 456)",
        "Action: click(123, 456)",
        "Action: drag(...)",
        "只有任务确实完成时才能使用finish",
        "现在只输出Action",
        "不得猜测不可见目标的坐标",
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
        "click不得省略x=或y=",
    ):
        assert marker in ACTION_SYSTEM_PROMPT


def test_repeated_strategy_feedback_is_structured_and_conditional() -> None:
    """被阻止的重复策略以通用结构化事实进入下一次动态上下文。"""
    default_prompt = compose_action_prompt(
        "执行桌面任务",
        ActionPromptState(step_number=2, max_steps=10),
        "normalized_1000",
    )
    assert "Recovery feedback:" not in default_prompt

    recovery_prompt = compose_action_prompt(
        "执行桌面任务",
        ActionPromptState(
            step_number=2,
            max_steps=10,
            last_action="click(x=480, y=883)",
            last_dispatch_status="failure",
            last_error="重复无效动作已被程序阻止，必须更换策略。",
            previous_strategy_failed=True,
            blocked_repeated_action="click(x=480, y=883)",
        ),
        "normalized_1000",
    )
    assert "Recovery feedback:" in recovery_prompt
    assert "- previous_action=click(x=480, y=883)" in recovery_prompt
    assert "- observed_result=动作未分发" in recovery_prompt
    assert "- required_change=不要再次返回同一动作" in recovery_prompt
    assert 'Action: type(text="Hello World")' not in ACTION_SYSTEM_PROMPT
    for benchmark_name in ("Chrome", "Python", "计算器", "记事本"):
        assert benchmark_name not in ACTION_SYSTEM_PROMPT
    for unsupported in ("press(", "release(", "move_to("):
        assert unsupported not in ACTION_SYSTEM_PROMPT

    grammar_section = ACTION_SYSTEM_PROMPT.split("输出协议：", 1)[0]
    for unsupported_action in ("right_click(", "double_click(", "drag("):
        assert unsupported_action not in grammar_section


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


def test_structured_entry_commit_feedback_is_generic_and_dynamic() -> None:
    """待提交恢复反馈只描述通用编辑状态，不泄漏任务或具体提交动作。"""
    prompt = compose_action_prompt(
        "录入多字段数据",
        ActionPromptState(
            step_number=2,
            max_steps=10,
            structured_entry_commit_feedback=True,
        ),
        "normalized_1000",
    )

    assert "Previous structured entry may still have its final field" in prompt
    assert "did not end with an explicit cell or row commit" in prompt
    assert "Verify or commit the final field before finishing" in prompt
    recovery = prompt.split("Recovery feedback:", 1)[1].split(
        "Current perception:",
        1,
    )[0]
    for forbidden in ("Enter", "Tab", "Excel", "C4", "M01"):
        assert forbidden not in recovery


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
    """PRD 原文保留用于审计；production 顶部仍限定同一五动作集合。"""
    assert "每次只输出一个动作" in PRD_ACTION_SYSTEM_PROMPT
    assert not ACTION_SYSTEM_PROMPT.startswith(PRD_ACTION_SYSTEM_PROMPT)
    grammar_section = ACTION_SYSTEM_PROMPT.split("输出协议：", 1)[0]
    for action in ("click(", "type(", "scroll(", "hotkey(", "finish("):
        assert action in grammar_section
    for unsupported in ("right_click(", "double_click(", "drag("):
        assert unsupported not in grammar_section


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

        def generate(self, image, prompt, mode="local", options=None):
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


class _ZoomAwareFakeRecognizer:
    """按图像尺寸区分全图/放大遍:返回不同识别结果。"""

    def __init__(self, full_items, zoom_items):
        self._full = full_items
        self._zoom = zoom_items
        self.calls: list[tuple[int, int]] = []

    def recognize(self, image):
        self.calls.append(image.size)
        if image.size == (200, 100):
            return self._full
        return self._zoom


def _item(text, bbox, confidence=0.9):
    return {"text": text, "bbox": bbox, "confidence": confidence}


def test_perceive_ocr_elements_focus_pass_takes_priority() -> None:
    """焦点放大遍的结果优先进入配额,并正确映射回全图坐标。"""
    from perception.prompt_context import perceive_ocr_elements

    full = [
        _item("功能区甲", (0, 0, 100, 20)),
        _item("功能区乙", (0, 25, 100, 45)),
        _item("功能区丙", (0, 50, 100, 70)),
    ]
    # 放大遍(以(100,50)为中心的裁剪,原点约(0,0),2x):识别出单元格
    zoom = [
        _item("姓名", (80, 60, 200, 100)),
        _item("部门", (280, 60, 400, 100)),
    ]
    recognizer = _ZoomAwareFakeRecognizer(full, zoom)
    elements = perceive_ocr_elements(
        recognizer,
        Image.new("RGB", (200, 100)),
        focus_point=(100, 50),
    )
    texts = [e.split('text="')[1].split('"')[0] for e in elements]
    # 焦点结果排最前,全图结果补足
    assert texts[:2] == ["姓名", "部门"]
    assert "功能区甲" in texts
    # 两次识别都被调用(全图 + 放大)
    assert len(recognizer.calls) == 2


def test_perceive_ocr_elements_focus_outside_image_skips_zoom() -> None:
    """焦点不在截图内时只做全图识别,行为与旧版一致。"""
    from perception.prompt_context import perceive_ocr_elements

    full = [_item("唯一文字", (10, 10, 60, 40))]
    recognizer = _ZoomAwareFakeRecognizer(full, [])
    elements = perceive_ocr_elements(
        recognizer,
        Image.new("RGB", (200, 100)),
        focus_point=(500, 500),
    )
    assert len(elements) == 1
    assert recognizer.calls == [(200, 100)]


def test_perceive_ocr_elements_without_focus_single_pass() -> None:
    """无焦点时只做一次全图识别,不触发放大裁剪遍。"""
    from perception.prompt_context import perceive_ocr_elements_detailed

    full = [_item("标题", (10, 10, 120, 40)), _item("按钮", (10, 60, 80, 90))]
    recognizer = _ZoomAwareFakeRecognizer(full, [])
    elements, detailed = perceive_ocr_elements_detailed(
        recognizer,
        Image.new("RGB", (200, 100)),
        focus_point=None,
    )
    assert recognizer.calls == [(200, 100)]
    assert len(elements) == 2
    assert [d["text"] for d in detailed] == ["标题", "按钮"]


def test_perceive_ocr_elements_detailed_matches_prompt_lines() -> None:
    """结构化条目与 Prompt 行的文本/归一化 bbox/置信度逐项一致。"""
    from perception.prompt_context import perceive_ocr_elements_detailed

    full = [_item("甲", (0, 0, 100, 50)), _item("乙", (100, 50, 200, 100), 0.85)]
    zoom = [_item("丙", (20, 20, 60, 40))]
    recognizer = _ZoomAwareFakeRecognizer(full, zoom)
    elements, detailed = perceive_ocr_elements_detailed(
        recognizer,
        Image.new("RGB", (200, 100)),
        focus_point=(50, 25),
    )
    assert len(elements) == len(detailed)
    for line, entry in zip(elements, detailed):
        assert f'text="{entry["text"]}"' in line
        assert f"bbox={entry['bbox']}" in line
        assert f"confidence={entry['confidence']:.2f}" in line


def test_perceive_ocr_elements_dense_focus_region_skips_zoom() -> None:
    """焦点区域已有足量高置信结果时跳过二遍识别,直接复用全图结果。"""
    from perception.prompt_context import FOCUS_REUSE_DENSE_MIN, perceive_ocr_elements

    full = [
        _item(f"条目{index}", (10, 10 + index * 10, 90, 20 + index * 10))
        for index in range(FOCUS_REUSE_DENSE_MIN)
    ]
    recognizer = _ZoomAwareFakeRecognizer(
        full, [_item("放大区文字", (80, 80, 120, 90))]
    )
    elements = perceive_ocr_elements(
        recognizer,
        Image.new("RGB", (200, 100)),
        focus_point=(100, 50),
    )
    assert recognizer.calls == [(200, 100)]
    assert all(f"条目{index}" in "\n".join(elements) for index in range(len(full)))


def test_perceive_ocr_elements_focus_density_boundary_keeps_zoom() -> None:
    """高置信条目数不足阈值或置信度不达标时,放大二遍识别仍然执行。"""
    from perception.prompt_context import FOCUS_REUSE_DENSE_MIN, perceive_ocr_elements

    sparse = [
        _item(f"条目{index}", (10, 10 + index * 10, 90, 20 + index * 10))
        for index in range(FOCUS_REUSE_DENSE_MIN - 1)
    ]
    recognizer = _ZoomAwareFakeRecognizer(
        sparse, [_item("放大区文字", (80, 80, 120, 90))]
    )
    elements = perceive_ocr_elements(
        recognizer,
        Image.new("RGB", (200, 100)),
        focus_point=(100, 50),
    )
    assert recognizer.calls == [(200, 100), (400, 200)]
    assert 'text="放大区文字"' in elements[0]

    low_confidence = [
        _item(f"条目{index}", (10, 10 + index * 10, 90, 20 + index * 10), 0.7)
        for index in range(FOCUS_REUSE_DENSE_MIN)
    ]
    recognizer_low = _ZoomAwareFakeRecognizer(
        low_confidence,
        [_item("放大区文字", (80, 80, 120, 90))],
    )
    perceive_ocr_elements(
        recognizer_low,
        Image.new("RGB", (200, 100)),
        focus_point=(100, 50),
    )
    assert recognizer_low.calls == [(200, 100), (400, 200)]


def test_perceive_ocr_elements_dedup_between_passes() -> None:
    """两遍都识别到的同位置同文本只保留一份。"""
    from perception.prompt_context import perceive_ocr_elements

    full = [_item("重复词", (95, 45, 105, 55))]
    zoom = [_item("重复词", (196, 96, 204, 104))]  # 映射回(98,48)-(102,52)
    recognizer = _ZoomAwareFakeRecognizer(full, zoom)
    elements = perceive_ocr_elements(
        recognizer,
        Image.new("RGB", (200, 100)),
        focus_point=(100, 50),
    )
    dup_count = sum('text="重复词"' in e for e in elements)
    assert dup_count == 1


def test_action_screen_point_mapping() -> None:
    """带坐标动作的屏幕锚点换算:相对坐标按截图尺寸缩放并叠加原点。"""
    from agent.gui_agent import _action_screen_point, _focus_local_point

    click = {"action_type": "click", "params": {"x": 500, "y": 500}}
    point = _action_screen_point(
        click,
        (1000, 500),
        (100, 200),
        "normalized_1000",
    )
    assert point == (100 + round(500 * 999 / 1000), 200 + round(500 * 499 / 1000))
    drag = {
        "action_type": "drag",
        "params": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
    }
    assert _action_screen_point(
        drag,
        (1000, 500),
        (0, 0),
        "normalized_1000",
    ) == (round(100 * 999 / 1000), round(100 * 499 / 1000))
    hotkey = {"action_type": "hotkey", "params": {"keys": ("enter",)}}
    assert _action_screen_point(hotkey, (1000, 500), (0, 0), "normalized_1000") is None
    assert _focus_local_point(None, (0, 0)) is None
    assert _focus_local_point((150, 260), (100, 200)) == (50, 60)
