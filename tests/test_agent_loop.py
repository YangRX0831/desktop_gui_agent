"""Agent 主循环、重试与效果验证测试。"""

import asyncio
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from agent.diagnostics import DiagnosticsRecord

from agentscope.message import Msg
from PIL import Image

from agent.action_dispatcher import ActionDispatcher, PermissionScope
from agent.dashscope_api_backend import DashScopeAPIRetryableError
from agent.gui_agent import GuiAgent, GuiAgentDependencies
from agent.model_client import (
    LocalModelInferenceError,
    LocalModelLoadError,
    LocalModelOutputError,
    ModelClient,
    Qwen2VLLocalBackend,
)
from agent.task_manager import TaskManager, TaskStatus
from config import AppConfig, GuiAgentSettings
from main import EXIT_INSTRUCTION, WELCOME_MESSAGE, run_cli
from tests.agent_test_support import (
    _REAL_MODEL_DIR,
    _RUN_ACTIONS,
    CountingCapture,
    FrameSequenceCapture,
    MemoryControls,
    RecordingBackend,
    SequenceBackend,
    _install_fake_transformers,
    _make_client,
    _RegionRecordingCapture,
    _stability_agent,
    make_agent,
)


def test_cli_constants_match_prd_4_4_2() -> None:
    """CLI welcome/exit 文案与 PRD 4.4.2 示例逐字一致。"""
    assert WELCOME_MESSAGE == "欢迎使用桌面GUI智能体！"
    assert EXIT_INSTRUCTION == "输入'exit'退出程序"


def test_cli_uses_prd_step_format_and_success_literal() -> None:
    """CLI 步骤格式与成功文案采用 PRD 4.4.2 canonical literal。"""
    outputs: list[str] = []
    inputs = iter(["任务", "exit"])

    class FakeAgent:
        def __init__(self, observer: object) -> None:
            self._observer = observer

        async def __call__(self, msg: Msg) -> Msg:
            observer = self._observer
            assert callable(observer)
            observer(1, {"action_type": "type", "params": {"text": "你好"}})
            return Msg("a", "完成", "assistant")

    def factory(config: AppConfig, observer: object) -> FakeAgent:
        return FakeAgent(observer)

    status = asyncio.run(
        run_cli(
            AppConfig(),
            agent_factory=factory,  # type: ignore[arg-type]
            read_input=lambda prompt: (outputs.append(prompt), next(inputs))[1],
            write_output=outputs.append,
        ),
    )

    assert status == 0
    assert WELCOME_MESSAGE in outputs
    assert EXIT_INSTRUCTION in outputs
    assert "正在初始化" in outputs[0]
    assert "请输入指令：" in outputs
    assert '[步骤1] 执行动作：type(text="你好")' in outputs
    # PRD canonical success literal:任务执行成功：
    assert "任务执行成功：完成" in outputs


def test_cli_displays_failure_without_double_prefix() -> None:
    """CLI 失败路径显示失败文案且不重复前缀(PRD 4.4.2 失败结果显示)。"""
    outputs: list[str] = []
    inputs = iter(["任务", "exit"])

    class FailingAgent:
        def __init__(self, observer: object) -> None:
            self._observer = observer

        async def __call__(self, msg: Msg) -> Msg:
            return Msg("a", "任务执行失败：模型调用失败。", "assistant")

    def factory(config: AppConfig, observer: object) -> FailingAgent:
        return FailingAgent(observer)

    status = asyncio.run(
        run_cli(
            AppConfig(),
            agent_factory=factory,  # type: ignore[arg-type]
            read_input=lambda prompt: next(inputs),
            write_output=outputs.append,
        ),
    )
    assert status == 0
    # 失败文案显示,且前缀只出现一次(不重复)
    failure_lines = [o for o in outputs if o.startswith("任务执行失败：")]
    assert len(failure_lines) == 1
    assert "任务执行失败：模型调用失败。" in failure_lines[0]
    assert failure_lines[0].count("任务执行失败：") == 1


def test_initial_success_no_retry() -> None:
    """initial attempt 成功时不产生 retry。"""
    backend = SequenceBackend(['Action: finish(result="done")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager)(Msg("u", "任务", "user"))
    )
    assert result.content == "done"
    assert backend.calls == 1
    assert manager.state.retry_count == 0


def test_dispatch_failure_triggers_fresh_screenshot_and_model() -> None:
    """dispatch 失败后做 fresh screenshot + fresh model decision,不重放同一动作。"""
    backend = SequenceBackend(
        ["Action: click(x=10, y=20)", 'Action: finish(result="ok")'],
    )
    controls = MemoryControls(fail_first=1)
    manager = TaskManager("任务")
    capture = CountingCapture()
    result = asyncio.run(
        make_agent(backend, controls, manager, capture, retry_count=3)(
            Msg("u", "任务", "user"),
        ),
    )
    assert result.content == "ok"
    # click 失败后进入 fresh perception:新截图 + 新模型决策。
    assert capture.calls == 2
    assert backend.calls == 2
    # click 只执行一次(失败的那次),finish 不触发控制。
    assert len(controls.calls) == 1
    assert manager.state.retry_count == 1


def test_success_after_retry_counts_one_retry() -> None:
    """initial 失败 + 1 次 fresh retry 成功 → retry_count=1。"""
    backend = SequenceBackend(
        ["Action: click(x=10, y=20)", 'Action: finish(result="ok")'],
    )
    controls = MemoryControls(fail_first=1)
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    assert manager.state.retry_count == 1
    assert manager.state.steps[0].retry_count == 1


def test_retry_count_boundary_is_three() -> None:
    """retry_count=3 表示 initial attempt 后最多 3 次 fresh retry。"""
    # 4 次尝试(initial+3 retry)全失败,第 5 次应是下一 logical step。
    outcomes = ["Action: click(x=1, y=1)"] * 4 + ['Action: finish(result="done")']
    backend = SequenceBackend(outcomes)
    controls = MemoryControls(fail_first=4)
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3, max_steps=2)(
            Msg("u", "任务", "user"),
        ),
    )
    assert result.content == "done"
    # 第一 logical step:initial + 3 retry = 4 次失败,第二 step finish。
    assert manager.state.steps[0].retry_count == 3
    assert manager.state.retry_count == 3


def test_retry_exhausted_continues_next_step() -> None:
    """retry 耗尽后记录该步失败并继续下一 logical step,不终止任务。"""
    outcomes = [
        "Action: click(x=1, y=1)",  # step1 initial
        "Action: click(x=2, y=2)",  # step1 retry1
        "Action: click(x=3, y=3)",  # step1 retry2
        "Action: click(x=4, y=4)",  # step1 retry3 → exhausted
        'Action: finish(result="ok")',  # step2 成功
    ]
    backend = SequenceBackend(outcomes)
    controls = MemoryControls(fail_first=4)
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3, max_steps=2)(
            Msg("u", "任务", "user"),
        ),
    )
    assert result.content == "ok"
    assert manager.state.status is TaskStatus.SUCCESS
    # step1 失败(retry_count=3),step2 成功。
    assert len(manager.state.steps) == 2
    assert manager.state.steps[0].result is False
    assert manager.state.steps[0].retry_count == 3


def test_max_steps_bounds_total_logical_steps() -> None:
    """未收到 finish 时,max_steps 限制 logical step 总数。"""
    backend = SequenceBackend(["Action: click(x=1, y=1)", "Action: click(x=2, y=2)"])
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, max_steps=2, retry_count=3)(
            Msg("u", "任务", "user"),
        ),
    )
    assert "达到最大执行步数" in result.content
    assert manager.state.status is TaskStatus.FAILED
    assert len(manager.state.steps) == 2


def test_task_manager_no_duplicate_retry_accounting() -> None:
    """record_step 的 per-step retry 与 record_retry 的 task 级 retry 不重复。"""
    manager = TaskManager("任务")
    manager.start()
    manager.record_step(
        {"action_type": "click", "params": {"x": 1, "y": 2}},
        False,
        retry_count=2,
    )
    manager.record_retry(2)
    assert manager.state.steps[0].retry_count == 2
    assert manager.state.retry_count == 2
    # 外部快照不能改写内部状态。
    snapshot = manager.state
    snapshot.steps[0].action["changed"] = True
    assert "changed" not in manager.state.steps[0].action


def test_load_failure_automatic_fallback_without_authorization() -> None:
    """PRD 4.3.1:load 失败无需任何授权即自动切换 API(F2)。"""
    local = RecordingBackend(raise_exc=LocalModelLoadError("load"))
    api = RecordingBackend(response="x")
    client = _make_client(local, api)
    assert client.generate(Image.new("RGB", (1, 1)), "p") == "x"
    assert local.calls == 1
    assert api.calls == 1


def test_load_failure_one_fallback_round() -> None:
    """load 失败进入一次 API fallback;backend 自身 1+3 retry(F4)。"""
    retryable = DashScopeAPIRetryableError("temp")
    api = SequenceBackend([retryable, retryable, retryable, retryable])
    local = RecordingBackend(raise_exc=LocalModelLoadError("load"))
    client = ModelClient(local, api, fallback_enabled=True, max_api_retries=3)
    assert client.generate(Image.new("RGB", (1, 1)), "p") == "API 模型调用失败。"
    assert local.calls == 1
    assert api.calls == 4


def test_inference_failure_no_fallback() -> None:
    """推理失败(非 load 失败)不触发 fallback(F6)。"""
    local = RecordingBackend(raise_exc=LocalModelInferenceError("infer"))
    api = RecordingBackend(response="x")
    client = _make_client(local, api)
    assert client.generate(Image.new("RGB", (1, 1)), "p") == "本地模型调用失败。"
    assert api.calls == 0


def test_output_failure_no_fallback() -> None:
    """输出失败不触发 fallback(F6b)。"""
    local = RecordingBackend(raise_exc=LocalModelOutputError("empty"))
    api = RecordingBackend(response="x")
    client = _make_client(local, api)
    assert client.generate(Image.new("RGB", (1, 1)), "p") == "本地模型调用失败。"
    assert api.calls == 0


def test_every_run_load_failure_falls_back() -> None:
    """每个 run 的 load 失败都自动 fallback;无跨 run 授权状态。"""
    local = RecordingBackend(raise_exc=LocalModelLoadError("load"))
    api = RecordingBackend(response="x")
    client = ModelClient(local, api, fallback_enabled=True)
    client.generate(Image.new("RGB", (1, 1)), "p")
    assert api.calls == 1
    local2 = RecordingBackend(raise_exc=LocalModelLoadError("load"))
    api2 = RecordingBackend(response="y")
    client._local_backend = local2
    client._api_backend = api2
    assert client.generate(Image.new("RGB", (1, 1)), "p") == "y"
    assert api2.calls == 1


def test_no_retry_multiplication_between_layers() -> None:
    """local 1 次 + API 至多 1+3=4 次,无乘法放大(C4)。"""
    retryable = DashScopeAPIRetryableError("temp")
    api = SequenceBackend([retryable, retryable, retryable, retryable])
    local = RecordingBackend(raise_exc=LocalModelLoadError("load"))
    client = ModelClient(local, api, fallback_enabled=True, max_api_retries=3)
    client.generate(Image.new("RGB", (1, 1)), "p")
    assert local.calls == 1
    assert api.calls == 4


def test_no_scope_denies_side_effect_action() -> None:
    """无授权作用域时 click 被拒绝,不产生控制调用。"""
    controls = MemoryControls()
    dispatcher = ActionDispatcher(controls, controls)
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": {"x": 500, "y": 500}},
            (100, 100),
        )
        is False
    )
    assert controls.calls == []


def test_cleared_scope_denies_side_effect_action() -> None:
    """scope 被 clear 后产生副作用的动作被拒绝。"""
    controls = MemoryControls()
    dispatcher = ActionDispatcher(controls, controls)
    scope = PermissionScope(allowed_actions=_RUN_ACTIONS, token=1)
    dispatcher.activate_run_scope(scope)
    dispatcher.clear_run_scope()
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": {"x": 500, "y": 500}},
            (100, 100),
        )
        is False
    )
    assert controls.calls == []


def test_authorized_scope_dispatches_once() -> None:
    """已激活 scope 授权有效动作时产生且仅产生一次控制调用。"""
    controls = MemoryControls()
    dispatcher = ActionDispatcher(controls, controls)
    scope = PermissionScope(allowed_actions=_RUN_ACTIONS, token=1)
    dispatcher.activate_run_scope(scope)
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": {"x": 50, "y": 50}},
            (100, 100),
        )
        is True
    )
    assert controls.calls == [("click", 50, 50, "left")]


def test_scope_token_differs_per_run() -> None:
    """每个 PermissionScope 有独立 token,确保 task1 scope != task2 scope。"""
    s1 = PermissionScope(allowed_actions=_RUN_ACTIONS, token=1)
    s2 = PermissionScope(allowed_actions=_RUN_ACTIONS, token=2)
    assert s1 != s2
    assert s1.token != s2.token


def test_click_region_offset_remaps_local_to_global() -> None:
    """region 截图:crop-local 像素 + region_offset = 全局桌面坐标。"""
    controls = MemoryControls()
    dispatcher = ActionDispatcher(controls, controls)
    dispatcher.activate_run_scope(
        PermissionScope(allowed_actions=_RUN_ACTIONS, token=71),
    )
    # region 200x100 at offset (50,30);image-pixel click(100,50)->global(150,80)
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": {"x": 100, "y": 50}},
            (200, 100),
            region_offset=(50, 30),
        )
        is True
    )
    assert controls.calls == [("click", 150, 80, "left")]


def test_click_full_screen_default_offset_zero_unchanged() -> None:
    """全屏截图 region_offset 默认 (0,0),映射与历史完全一致。"""
    controls = MemoryControls()
    dispatcher = ActionDispatcher(controls, controls)
    dispatcher.activate_run_scope(
        PermissionScope(allowed_actions=_RUN_ACTIONS, token=72),
    )
    assert (
        dispatcher.dispatch(
            {"action_type": "click", "params": {"x": 50, "y": 50}},
            (100, 100),
        )
        is True
    )
    assert controls.calls == [("click", 50, 50, "left")]


def test_dispatch_rejects_invalid_region_offset() -> None:
    """非法 region_offset 在控制调用前拒绝,不产生副作用。"""
    controls = MemoryControls()
    dispatcher = ActionDispatcher(controls, controls)
    dispatcher.activate_run_scope(
        PermissionScope(allowed_actions=_RUN_ACTIONS, token=73),
    )
    for bad in [(0,), (0, 0, 0), (0, "a"), (-1, 0), [0, 0], None]:
        assert (
            dispatcher.dispatch(
                {"action_type": "click", "params": {"x": 500, "y": 500}},
                (100, 100),
                region_offset=bad,
            )
            is False
        )
    assert controls.calls == []


def test_capture_uses_active_window_region_after_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """前台切换到新应用窗口后,_capture 截该窗口 region 且 click 映射全局坐标。"""
    from perception import screenshot as screen

    monkeypatch.setattr("agent.gui_agent.get_foreground_app_hwnd", lambda: 10)
    monkeypatch.setattr("agent.gui_agent.is_window_available", lambda hwnd: True)
    monkeypatch.setattr(screen, "get_foreground_app_hwnd", lambda: 20)
    monkeypatch.setattr(
        screen,
        "_window_region",
        lambda hwnd: (40, 60, 200, 100),
    )
    capture = _RegionRecordingCapture()
    backend = SequenceBackend(
        ["Action: click(x=100, y=50)", 'Action: finish(result="ok")'],
    )
    controls = MemoryControls()
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(backend, controls, manager, capture=capture)(
            Msg("u", "任务", "user"),
        )
    )
    assert any(c.get("region") == (40, 60, 200, 100) for c in capture.calls)
    # image-pixel click(100,50) on 200x100 region at offset(40,60) -> global(140,110)
    assert controls.calls[0] == ("click", 140, 110, "left")


def test_capture_falls_back_to_full_screen_when_foreground_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """前台未切换(与任务开始相同)或无前台应用时,回退全屏 region=None。"""
    from perception import screenshot as screen

    monkeypatch.setattr("agent.gui_agent.get_foreground_app_hwnd", lambda: 0)
    monkeypatch.setattr(screen, "get_foreground_app_hwnd", lambda: 0)
    capture = _RegionRecordingCapture()
    backend = SequenceBackend(['Action: finish(result="ok")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(backend, controls, manager, capture=capture)(
            Msg("u", "任务", "user"),
        )
    )
    assert capture.calls, "capture 应至少被调用一次"
    assert all(c.get("region") is None for c in capture.calls)


def test_capture_region_follows_dialog_and_cross_app_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """对话框置前或跨应用切换时,每步 region 跟随当前前台窗口。

    覆盖特性 C/D/E:弹出对话框变为前台后 ROI 即对话框区域,不隐藏
    新窗口;跨应用往返切换时 region 随每步前台更新。
    """
    from perception import screenshot as screen

    rects = {
        200: (40, 60, 300, 200),  # 应用窗口
        300: (120, 140, 200, 120),  # 对话框(更小矩形,叠在应用上)
    }
    foreground_sequence = iter([200, 300, 200])
    monkeypatch.setattr("agent.gui_agent.get_foreground_app_hwnd", lambda: 10)
    monkeypatch.setattr("agent.gui_agent.is_window_available", lambda hwnd: True)
    monkeypatch.setattr(
        screen,
        "get_foreground_app_hwnd",
        lambda: next(foreground_sequence),
    )
    monkeypatch.setattr(screen, "_window_region", lambda hwnd: rects[hwnd])
    capture = _RegionRecordingCapture()
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=10)",
            "Action: click(x=20, y=20)",
            'Action: finish(result="ok")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("跨窗口任务")
    asyncio.run(
        make_agent(backend, controls, manager, capture=capture, max_steps=3)(
            Msg("u", "跨窗口任务", "user"),
        )
    )
    region_calls = [c["region"] for c in capture.calls if c.get("region")]
    assert region_calls == [
        (40, 60, 300, 200),
        (120, 140, 200, 120),
        (40, 60, 300, 200),
    ]
    # 前两步 click 分别按各自 region offset 映射到全局坐标。
    assert controls.calls == [
        ("click", 40 + 10, 60 + 10, "left"),
        ("click", 120 + 20, 140 + 20, "left"),
    ]


class _SizeKeyedRecognizer:
    """按图像尺寸返回预置 OCR 结果并记录调用尺寸的 fake 识别器。"""

    def __init__(self, items_by_size: dict[tuple[int, int], list[dict]]) -> None:
        self._items = items_by_size
        self.calls: list[tuple[int, int]] = []

    def recognize(self, image: Image.Image) -> list[dict]:
        self.calls.append(image.size)
        return [dict(item) for item in self._items.get(image.size, [])]


def test_roi_empty_ocr_escalates_to_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """特性 F:region 感知无元素时升级全帧重识别,坐标基准一并切回全帧。"""
    from perception import screenshot as screen

    monkeypatch.setattr("agent.gui_agent.get_foreground_app_hwnd", lambda: 10)
    monkeypatch.setattr("agent.gui_agent.is_window_available", lambda hwnd: True)
    monkeypatch.setattr(screen, "get_foreground_app_hwnd", lambda: 20)
    monkeypatch.setattr(
        screen,
        "_window_region",
        lambda hwnd: (40, 60, 200, 100),
    )
    recognizer = _SizeKeyedRecognizer(
        {
            (1000, 500): [
                {"text": "背景目标", "bbox": (10, 10, 200, 60), "confidence": 0.95},
            ],
        },
    )
    capture = _RegionRecordingCapture()
    backend = SequenceBackend(
        ["Action: click(x=100, y=50)", 'Action: finish(result="ok")'],
    )
    controls = MemoryControls()
    manager = TaskManager("找背景目标")
    asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            capture=capture,
            ocr_recognizer=recognizer,
        )(
            Msg("u", "找背景目标", "user"),
        )
    )
    # region 空结果后对全帧重识别,模型看到全帧文字并按全帧坐标点击。
    assert (1000, 500) in recognizer.calls
    assert 'text="背景目标"' in backend.prompts[0]
    assert controls.calls[0] == ("click", 100, 50, "left")


def test_roi_nonempty_ocr_skips_full_frame_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """region 感知已有元素时不升级,不做第二次全帧识别。"""
    from perception import screenshot as screen

    monkeypatch.setattr("agent.gui_agent.get_foreground_app_hwnd", lambda: 10)
    monkeypatch.setattr("agent.gui_agent.is_window_available", lambda hwnd: True)
    monkeypatch.setattr(screen, "get_foreground_app_hwnd", lambda: 20)
    monkeypatch.setattr(
        screen,
        "_window_region",
        lambda hwnd: (40, 60, 200, 100),
    )
    recognizer = _SizeKeyedRecognizer(
        {
            (200, 100): [
                {"text": "窗口目标", "bbox": (10, 10, 120, 40), "confidence": 0.9},
            ],
        },
    )
    capture = _RegionRecordingCapture()
    backend = SequenceBackend(['Action: finish(result="ok")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            capture=capture,
            ocr_recognizer=recognizer,
        )(
            Msg("u", "任务", "user"),
        )
    )
    assert recognizer.calls == [(200, 100)]
    assert 'text="窗口目标"' in backend.prompts[0]


def test_full_frame_empty_ocr_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全屏路径本身无元素时不重复识别,每步只识别一次。"""
    from perception import screenshot as screen

    monkeypatch.setattr("agent.gui_agent.get_foreground_app_hwnd", lambda: 0)
    monkeypatch.setattr(screen, "get_foreground_app_hwnd", lambda: 0)
    recognizer = _SizeKeyedRecognizer({})
    capture = _RegionRecordingCapture()
    backend = SequenceBackend(['Action: finish(result="ok")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            capture=capture,
            ocr_recognizer=recognizer,
        )(
            Msg("u", "任务", "user"),
        )
    )
    assert recognizer.calls == [(1000, 500)]


def test_select_region_prefers_foreground_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切换到新前台应用且区域可靠时使用该窗口截图。"""
    from perception import screenshot as screen

    monkeypatch.setattr(screen, "get_foreground_app_hwnd", lambda: 200)
    monkeypatch.setattr(screen, "_window_region", lambda h: (100, 50, 800, 600))
    region, offset = screen.select_capture_region(100)
    assert region == (100, 50, 800, 600)
    assert offset == (100, 50)


def test_select_region_full_screen_when_nothing_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无新前台应用时始终保留完整当前屏幕。"""
    from perception import screenshot as screen

    monkeypatch.setattr(screen, "get_foreground_app_hwnd", lambda: 0)
    region, offset = screen.select_capture_region(0)
    assert region is None and offset == (0, 0)


def test_region_logic_tolerates_chinese_window_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中文标题窗口(如"计算器")不破坏 region 选择:基于 HWND/class/几何,非英文标题。"""
    from perception import screenshot as screen

    monkeypatch.setattr(screen, "get_foreground_app_hwnd", lambda: 555)
    monkeypatch.setattr(
        screen,
        "_window_region",
        lambda h: (210, 158, 666, 1078),
    )
    region, _ = screen.select_capture_region(1)
    assert region == (210, 158, 666, 1078)


def test_window_region_clamps_to_virtual_desktop() -> None:
    """窗口物理矩形超出虚拟桌面右/下边界时按可见范围裁剪。"""
    from perception.screenshot import _normalize_window_region

    # 真实案例:Chrome 窗口右缘超出 2880 物理屏宽 20 像素。
    region = _normalize_window_region(
        (752, 54, 2148, 1670),
        scale=1.0,
        bounds=(0, 0, 2880, 1800),
    )
    assert region == (752, 54, 2128, 1670)

    # DPI 缩放后的窗口矩形同样先缩放再裁剪。
    scaled = _normalize_window_region(
        (376, 27, 1074, 835),
        scale=2.0,
        bounds=(0, 0, 2880, 1800),
    )
    assert scaled == (752, 54, 2128, 1670)


def test_window_region_rejects_negative_and_tiny_after_clamp() -> None:
    """负原点窗口拒绝回退全屏;裁剪后低于最小边长的区域同样拒绝。"""
    from perception.screenshot import _normalize_window_region

    interior = _normalize_window_region(
        (100, 100, 800, 600),
        scale=1.0,
        bounds=(0, 0, 2880, 1800),
    )
    assert interior == (100, 100, 800, 600)

    negative = _normalize_window_region(
        (-16, 0, 1920, 1080),
        scale=1.0,
        bounds=(0, 0, 2880, 1800),
    )
    assert negative is None

    tiny_after_clamp = _normalize_window_region(
        (2860, 100, 800, 600),
        scale=1.0,
        bounds=(0, 0, 2880, 1800),
    )
    assert tiny_after_clamp is None

    without_bounds = _normalize_window_region(
        (752, 54, 2148, 1670),
        scale=1.0,
        bounds=None,
    )
    assert without_bounds == (752, 54, 2148, 1670)


def test_virtual_desktop_bounds_reads_monitor_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """虚拟桌面边界读取 mss monitor[0];异常时返回 None 跳过裁剪。"""
    from perception import screenshot as screen

    class _FakeMss:
        monitors = [
            {"left": 0, "top": 0, "width": 2880, "height": 1800},
        ]

    monkeypatch.setattr(
        screen,
        "_get_mss_instance_unlocked",
        lambda: _FakeMss(),
    )
    assert screen._virtual_desktop_bounds() == (0, 0, 2880, 1800)

    class _BrokenMss:
        monitors = [{"left": 0}]

    monkeypatch.setattr(
        screen,
        "_get_mss_instance_unlocked",
        lambda: _BrokenMss(),
    )
    assert screen._virtual_desktop_bounds() is None


def test_frames_stable_identical_and_different() -> None:
    """相同帧稳定;大面积变化帧不稳定。"""
    from PIL import ImageDraw

    from perception.ui_locator import frames_stable

    same = Image.new("RGB", (200, 200), (10, 10, 10))
    assert frames_stable(same, same) is True
    other = Image.new("RGB", (200, 200), (10, 10, 10))
    ImageDraw.Draw(other).rectangle([40, 40, 160, 160], fill=(255, 255, 255))
    assert frames_stable(same, other) is False


def test_frame_change_ratio_detects_visible_change() -> None:
    """动作结果帧差对相同画面为零，对显著变化返回正比例。"""
    from perception.ui_locator import frame_change_ratio

    before = Image.new("RGB", (100, 100), "black")
    after = Image.new("RGB", (100, 100), "white")
    assert frame_change_ratio(before, before) == 0.0
    assert frame_change_ratio(before, after) == 1.0


def test_action_without_ui_progress_retries_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """执行成功但画面未变化时重新观察、反馈并生成不同动作。"""
    from agent import gui_agent as ga

    before = Image.new("RGB", (100, 100), "black")
    after = Image.new("RGB", (100, 100), "white")
    capture = FrameSequenceCapture(
        [before, before, before, before, after, after, after],
    )
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=10)",
            "Action: click(x=20, y=20)",
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("点击后完成")
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 0)
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 0)
    monkeypatch.setattr(
        ga,
        "select_capture_region",
        lambda *args: (None, (0, 0)),
    )

    result = asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            capture,
            max_steps=2,
            retry_count=1,
            reject_initial_finish=True,
            verify_action_effect=True,
        )(Msg("u", "点击后完成", "user")),
    )

    assert result.content == "done"
    assert controls.calls == [
        ("click", 10, 10, "left"),
        ("click", 20, 20, "left"),
    ]
    assert manager.state.retry_count == 1
    assert manager.state.steps[0].attempts[0].stage == "verify"
    assert manager.state.steps[0].attempts[0].succeeded is False
    assert "last_dispatch_status=success" in backend.prompts[1]
    assert "last_error=操作后未检测到界面变化。" in backend.prompts[1]
    assert "ui_change_signal=none" in backend.prompts[1]
    assert "recent_actions=click:failure:none" in backend.prompts[1]
    assert "same_action_streak=1" in backend.prompts[1]
    assert "no_ui_change_streak=1" in backend.prompts[1]


def test_foreground_change_is_valid_action_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """前台窗口变化即使帧相同也可证明动作产生界面结果。"""
    from agent import gui_agent as ga

    frame = Image.new("RGB", (40, 40), "black")
    agent, _ = _stability_agent(lambda: frame.copy())
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 200)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: True)

    observation = agent._verify_action_effect(
        "click",
        frame,
        100,
        100,
    )

    assert observation.succeeded is True
    assert observation.failure_reason is None
    assert observation.after_foreground == 200
    assert observation.screen_changed is False
    assert observation.effect == "foreground_window_changed"


def test_small_visual_change_allows_action_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """高分辨率应用中的 25 像素变化足以证明动作产生效果。"""
    from agent import gui_agent as ga

    before = Image.new("RGB", (1000, 1000), "black")
    after = before.copy()
    for x in range(5):
        for y in range(5):
            after.putpixel((x, y), (255, 255, 255))
    capture = FrameSequenceCapture([after, after])
    backend = SequenceBackend(['Action: finish(result="unused")'])
    controls = MemoryControls()
    agent = make_agent(
        backend,
        controls,
        TaskManager("任务"),
        capture=capture,
    )
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: True)

    observation = agent._verify_action_effect(
        "click",
        before,
        100,
        100,
    )

    assert observation.succeeded is True
    assert observation.failure_reason is None
    assert observation.screen_changed is True
    assert observation.effect == "visible_content_changed"

    shell_observation = agent._verify_action_effect(
        "click",
        before,
        100,
        0,
    )
    assert shell_observation.succeeded is False
    assert shell_observation.failure_reason == "操作后未检测到界面变化。"
    assert shell_observation.screen_changed is False
    assert shell_observation.effect == "none"


def test_finish_after_verified_action_does_not_require_semantic_pixel_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """动作效果验证与任务 finish 保持独立，不用像素比例臆测语义完成。"""
    from agent import gui_agent as ga

    before = Image.new("RGB", (100, 100), "black")
    after = before.copy()
    for x in range(5):
        for y in range(5):
            after.putpixel((x, y), (255, 255, 255))
    capture = FrameSequenceCapture([before, after, after, after])
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=10)",
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("任务")
    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)
    monkeypatch.setattr(ga, "select_capture_region", lambda hwnd: (None, (0, 0)))

    result = asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            capture=capture,
            verify_action_effect=True,
        )(Msg("u", "任务", "user")),
    )

    assert result.content == "done"
    assert controls.calls == [("click", 10, 10, "left")]
    assert backend.calls == 2
    assert manager.state.status is TaskStatus.SUCCESS


def test_action_effect_capture_failure_is_retryable_failure() -> None:
    """动作后截图失败时不能伪造成功，返回固定验证失败原因。"""
    from utils.exceptions import ScreenCaptureError

    def fail_capture(**kwargs):
        raise ScreenCaptureError("simulated")

    backend = SequenceBackend(['Action: finish(result="unused")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    agent = make_agent(backend, controls, manager, capture=fail_capture)

    observation = agent._verify_action_effect(
        "click",
        Image.new("RGB", (10, 10)),
        0,
        0,
    )

    assert observation.succeeded is False
    assert observation.failure_reason == "操作结果验证失败。"
    assert observation.screen_changed is None
    assert observation.effect == "none"


def test_gui_agent_settings_validate_action_effect_flag() -> None:
    """production 默认启用动作结果验证并拒绝非 bool 配置。"""
    assert GuiAgentSettings().verify_action_effect is True
    with pytest.raises(TypeError):
        GuiAgentSettings(verify_action_effect=1)  # type: ignore[arg-type]


def test_wait_for_ui_stable_returns_when_stable() -> None:
    """帧已稳定:prev+curr 两次截图即返回。"""
    agent, state = _stability_agent(lambda: Image.new("RGB", (40, 40), (10, 10, 10)))
    agent._wait_for_ui_stable()
    assert state["captures"] == 2


def test_wait_for_ui_stable_recaptures_until_stable() -> None:
    """先变化后稳定:重采到收敛后返回(>2 次截图,但有限)。"""
    seq = iter(
        [
            Image.new("RGB", (40, 40), (10, 10, 10)),
            Image.new("RGB", (40, 40), (200, 200, 200)),
            Image.new("RGB", (40, 40), (200, 200, 200)),
        ],
    )

    def factory():
        try:
            return next(seq)
        except StopIteration:
            return Image.new("RGB", (40, 40), (200, 200, 200))

    agent, state = _stability_agent(factory)
    agent._wait_for_ui_stable()
    assert state["captures"] == 3


def test_wait_for_ui_stable_bounded_when_always_changing() -> None:
    """帧持续变化:有界重采,不无限等待。"""
    toggle = {"on": False}

    def factory():
        toggle["on"] = not toggle["on"]
        c = (200, 200, 200) if toggle["on"] else (10, 10, 10)
        return Image.new("RGB", (40, 40), c)

    agent, state = _stability_agent(factory)
    agent._wait_for_ui_stable()
    # 1 初始 + 至多 _STABILITY_MAX_ITERS 次,有限
    from agent.gui_agent import _STABILITY_MAX_ITERS

    assert 2 <= state["captures"] <= _STABILITY_MAX_ITERS + 1


def test_wait_for_ui_stable_fallback_on_capture_failure() -> None:
    """截图失败:回退固定等待,不抛异常。"""
    from utils.exceptions import ScreenCaptureError

    def factory():
        raise ScreenCaptureError("simulated")

    agent, _ = _stability_agent(factory)
    agent._wait_for_ui_stable()  # 不抛异常即可


def test_run_scoped_permission_in_agent_loop() -> None:
    """Agent 每 run 新建 scope;finish 正常完成。"""
    backend = SequenceBackend(['Action: finish(result="done")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager)(Msg("u", "任务", "user"))
    )
    assert result.content == "done"
    assert controls.calls == []


def test_backend_construction_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造后端不触发 from_pretrained。"""
    captured = _install_fake_transformers(monkeypatch)
    local = Qwen2VLLocalBackend(_REAL_MODEL_DIR)
    assert local._model is None
    assert captured["load"] == 0


def test_backend_generate_uses_4bit_and_local_only(monkeypatch: pytest.MonkeyPatch):
    """generate 通过 BitsAndBytesConfig 传 4-bit + local_files_only。"""
    captured = _install_fake_transformers(monkeypatch)
    local = Qwen2VLLocalBackend(_REAL_MODEL_DIR, max_new_tokens=16)
    assert local.generate(Image.new("RGB", (2, 2)), "p") == "ok"
    model_kwargs = captured["model_kwargs"]
    # quantization_config 传给 from_pretrained,不是直接 load_in_4bit kwarg。
    qconfig = model_kwargs.get("quantization_config")
    assert qconfig is not None
    assert qconfig.load_in_4bit is True
    assert "load_in_4bit" not in model_kwargs
    assert model_kwargs.get("local_files_only") is True
    assert "trust_remote_code" not in model_kwargs
    proc_kwargs = captured["processor_kwargs"]
    assert proc_kwargs.get("local_files_only") is True


def test_backend_load_failure_raises_load_error(monkeypatch: pytest.MonkeyPatch):
    """加载失败转换为 LocalModelLoadError。"""
    _install_fake_transformers(monkeypatch, load_exc=RuntimeError("fail"))
    local = Qwen2VLLocalBackend(_REAL_MODEL_DIR)
    with pytest.raises(LocalModelLoadError):
        local.generate(Image.new("RGB", (2, 2)), "p")


def test_backend_inference_failure_raises_inference_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """推理失败转换为 LocalModelInferenceError(不触发 fallback)。"""
    _install_fake_transformers(monkeypatch, generate_exc=RuntimeError("gpu"))
    local = Qwen2VLLocalBackend(_REAL_MODEL_DIR)
    with pytest.raises(LocalModelInferenceError):
        local.generate(Image.new("RGB", (2, 2)), "p")


def test_backend_output_empty_raises_output_error(monkeypatch: pytest.MonkeyPatch):
    """空输出转换为 LocalModelOutputError(不触发 fallback)。"""
    _install_fake_transformers(monkeypatch, decode_texts=["   "])
    local = Qwen2VLLocalBackend(_REAL_MODEL_DIR)
    with pytest.raises(LocalModelOutputError):
        local.generate(Image.new("RGB", (2, 2)), "p")


def test_backend_rejects_non_qwen2_vl(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 qwen2_vl 模型目录在加载边界被拒绝。"""
    _install_fake_transformers(
        monkeypatch,
        config_model_type="bert",
        stub_validate=False,
    )
    local = Qwen2VLLocalBackend(_REAL_MODEL_DIR)
    with pytest.raises(LocalModelLoadError):
        local.generate(Image.new("RGB", (2, 2)), "p")


def test_vertical_slice_executes_actions_until_finish() -> None:
    """截图、模型、Parser、控制、状态形成 PRD 纵向闭环。"""
    backend = SequenceBackend(
        ["Action: click(x=100, y=200)", 'Action: finish(result="完成")'],
    )
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager)(Msg("u", "任务", "user"))
    )
    assert result.content == "完成"
    assert controls.calls == [("click", 100, 200, "left")]
    assert manager.state.status is TaskStatus.SUCCESS


def test_explanatory_response_is_rejected_before_retry_action_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """带解释的 Action 在 API 模式不被抽取执行,fresh retry 后只执行严格响应。

    local 模式自 LOCAL 2B BASELINE 起允许确定性唯一 Action 行修复
    (见 tests/test_local_output_repair.py),本契约显式固定为 API 路径。
    """
    backend = SequenceBackend(
        [
            '好的，我来操作。\nAction: hotkey(key1="alt", key2="f4")',
            'Action: hotkey(key1="alt", key2="f4")',
            'Action: finish(result="窗口已关闭")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("关闭当前窗口")
    from agent import gui_agent as ga

    monkeypatch.setattr(ga, "get_foreground_app_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_available", lambda hwnd: True)

    result = asyncio.run(
        make_agent(
            backend,
            controls,
            manager,
            retry_count=3,
            model_mode="api",
        )(
            Msg("u", "关闭当前窗口", "user"),
        ),
    )

    assert result.content == "窗口已关闭"
    assert controls.calls == [("hotkey", "alt", "f4")]
    assert backend.calls == 3
    assert manager.state.retry_count == 1


def test_verify_action_effect_reports_change_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_verify_action_effect 的 change_ratio 反映帧差像素比例。"""
    from agent import gui_agent as ga

    black = Image.new("RGB", (40, 40), "black")
    half_white = Image.new("RGB", (40, 40), "white")
    capture = FrameSequenceCapture([half_white, half_white])
    backend = SequenceBackend(['Action: finish(result="unused")'])
    controls = MemoryControls()
    agent = make_agent(
        backend,
        controls,
        TaskManager("任务"),
        capture=capture,
        verify_action_effect=True,
    )
    monkeypatch.setattr(ga, "get_foreground_hwnd", lambda: 100)
    monkeypatch.setattr(ga, "is_window_existing", lambda hwnd: True)

    changed = agent._verify_action_effect("click", black, 100, 100)
    assert changed.change_ratio == pytest.approx(1.0)

    stable = agent._verify_action_effect("click", half_white, 100, 100)
    assert stable.change_ratio == pytest.approx(0.0)


def test_model_failure_does_not_immediately_terminate() -> None:
    """模型调用失败后进入 PRD retry,不立即终止整个任务。"""
    # 第 1 次模型失败 → retry → 第 2 次成功 finish。
    backend = SequenceBackend(
        ["本地模型调用失败。", 'Action: finish(result="ok")'],
    )
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    assert result.content == "ok"
    assert manager.state.status is TaskStatus.SUCCESS
    # 模型被调用 2 次(失败 + 成功),fresh observation。
    assert backend.calls == 2


def test_model_failure_retries_then_exhausts_continues_next_step() -> None:
    """模型连续失败至 retry 耗尽后继续下一 logical step。"""
    # step1: 4 次模型失败(initial+3 retry),step2: finish。
    failures = ["本地模型调用失败。"] * 4 + ['Action: finish(result="done")']
    backend = SequenceBackend(failures)
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3, max_steps=2)(
            Msg("u", "任务", "user"),
        ),
    )
    assert result.content == "done"
    assert manager.state.status is TaskStatus.SUCCESS


def test_attempt_history_preserves_all_attempts() -> None:
    """一个 logical step 内 3 次 attempt(2 失败 + 1 成功)全部保留。"""
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=20)",
            "Action: click(x=30, y=40)",
            "Action: click(x=50, y=60)",
        ],
    )
    controls = MemoryControls(fail_first=2)
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    step = manager.state.steps[0]
    # 3 次 attempt 全部保留,不只有最后一次。
    assert len(step.attempts) == 3
    # 每次 attempt 的具体坐标。
    assert step.attempts[0].action["params"]["x"] == 10
    assert step.attempts[1].action["params"]["x"] == 30
    assert step.attempts[2].action["params"]["x"] == 50
    # 每次 attempt 的结果。
    assert step.attempts[0].succeeded is False
    assert step.attempts[1].succeeded is False
    assert step.attempts[2].succeeded is True
    # retry_index。
    assert step.attempts[0].retry_index == 0
    assert step.attempts[1].retry_index == 1
    assert step.attempts[2].retry_index == 2
    # logical step 最终结果。
    assert step.result is True
    assert step.retry_count == 2


def test_attempt_history_single_attempt() -> None:
    """initial 成功时 attempt 历史只有 1 条。"""
    backend = SequenceBackend(['Action: finish(result="ok")'])
    controls = MemoryControls()
    manager = TaskManager("任务")
    asyncio.run(make_agent(backend, controls, manager)(Msg("u", "任务", "user")))
    step = manager.state.steps[0]
    assert len(step.attempts) == 1
    assert step.attempts[0].retry_index == 0
    assert step.attempts[0].succeeded is True


def test_task_step_action_property_backward_compat() -> None:
    """TaskStep.action 属性返回最后一个 attempt 的动作。"""
    backend = SequenceBackend(
        ["Action: click(x=10, y=20)", "Action: click(x=30, y=40)"],
    )
    controls = MemoryControls(fail_first=1)
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    step = manager.state.steps[0]
    # action 属性返回最后一个(成功的)attempt 的动作。
    assert step.action["params"]["x"] == 30


def test_local_load_failure_automatic_fallback_finishes() -> None:
    """PRD 4.3.1:local load 失败自动切 API,无需授权,任务经 API 完成(F2/F3)。"""
    api = SequenceBackend(['Action: finish(result="fallback_ok")'])
    local = RecordingBackend(raise_exc=LocalModelLoadError("load"))
    client = ModelClient(local, api, fallback_enabled=True)

    controls = MemoryControls()
    manager = TaskManager("任务")
    dependencies = GuiAgentDependencies(
        model_client=client,
        action_dispatcher=ActionDispatcher(controls, controls),
        capture=CountingCapture(),
        task_manager_factory=lambda task: manager,
        sleep=lambda seconds: None,
    )
    settings = GuiAgentSettings(
        model_mode="local",
        reject_initial_finish=False,
        verify_action_effect=False,
    )
    agent = GuiAgent(dependencies, settings)

    result = asyncio.run(agent(Msg("u", "任务", "user")))

    # load failure → 一次自动 API fallback,成功 finish,无需任何授权调用。
    assert result.content == "fallback_ok"
    assert local.calls == 1
    assert api.calls == 1


def test_next_local_run_without_reauthorization_zero_api() -> None:
    """同一 agent 下一 local run 不重新授权 → 零 API 调用。"""
    api1 = SequenceBackend(['Action: finish(result="run1")'])
    local1 = RecordingBackend(raise_exc=LocalModelLoadError("load"))
    client = ModelClient(local1, api1, fallback_enabled=True)

    controls = MemoryControls()
    dependencies = GuiAgentDependencies(
        model_client=client,
        action_dispatcher=ActionDispatcher(controls, controls),
        capture=CountingCapture(),
        task_manager_factory=lambda task: TaskManager(task),
        sleep=lambda seconds: None,
    )
    settings = GuiAgentSettings(
        model_mode="local",
        reject_initial_finish=False,
        verify_action_effect=False,
    )
    agent = GuiAgent(dependencies, settings)

    # run1:load 失败 → 自动 fallback 成功。
    r1 = asyncio.run(agent(Msg("u", "任务", "user")))
    assert r1.content == "run1"
    assert api1.calls == 1

    # run2:load 失败 → 再次自动 fallback(PRD 4.3.1 无跨 run 授权状态)。
    local2 = RecordingBackend(raise_exc=LocalModelLoadError("load"))
    api2 = SequenceBackend(['Action: finish(result="run2")'])
    client._local_backend = local2
    client._api_backend = api2
    r2 = asyncio.run(agent(Msg("u", "任务2", "user")))
    assert r2.content == "run2"
    assert api2.calls == 1


def test_manual_api_mode_unchanged_by_fallback_removal() -> None:
    """manual API 模式行为不变:直接走 API 后端,不触碰 local(F5)。"""
    api = SequenceBackend(['Action: finish(result="api_mode")'])
    local = RecordingBackend(response="never")
    client = ModelClient(local, api, fallback_enabled=True)

    controls = MemoryControls()
    manager = TaskManager("任务")
    dependencies = GuiAgentDependencies(
        model_client=client,
        action_dispatcher=ActionDispatcher(controls, controls),
        capture=CountingCapture(),
        task_manager_factory=lambda task: manager,
        sleep=lambda seconds: None,
    )
    settings = GuiAgentSettings(
        model_mode="api",
        reject_initial_finish=False,
        verify_action_effect=False,
    )
    agent = GuiAgent(dependencies, settings)
    result = asyncio.run(agent(Msg("u", "任务", "user")))
    assert result.content == "api_mode"
    assert local.calls == 0
    assert api.calls == 1


def test_model_fail_all_retries_step_record_exists() -> None:
    """model fail × initial+3 retries → step record 存在,retry_count=3。"""
    failures = ["本地模型调用失败。"] * 4 + ['Action: finish(result="done")']
    backend = SequenceBackend(failures)
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3, max_steps=2)(
            Msg("u", "任务", "user"),
        ),
    )
    assert result.content == "done"
    # step1:4 次 model failure(initial+3 retry),有完整 step record。
    step1 = manager.state.steps[0]
    assert step1.result is False
    assert step1.retry_count == 3
    assert len(step1.attempts) == 4
    for a in step1.attempts:
        assert a.action is None  # 不伪造 action
        assert a.succeeded is False
        assert a.stage == "model"
        assert a.failure_reason is not None  # 有 failure reason


def test_model_fail_then_success_preserves_both_attempts() -> None:
    """model fail → retry → valid action success:两次 attempt 都保留。"""
    backend = SequenceBackend(
        ["本地模型调用失败。", "Action: click(x=50, y=60)"],
    )
    controls = MemoryControls()
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    # click 成功后,任务在下一 step 需要 finish 才结束;但这里只验证 step1。
    step1 = manager.state.steps[0]
    assert len(step1.attempts) == 2
    # attempt 0: model failure,无 action。
    assert step1.attempts[0].action is None
    assert step1.attempts[0].stage == "model"
    assert step1.attempts[0].succeeded is False
    # attempt 1: click 成功。
    assert step1.attempts[1].action is not None
    assert step1.attempts[1].stage == "dispatch"
    assert step1.attempts[1].succeeded is True
    assert step1.result is True
    assert step1.retry_count == 1


def test_no_ghost_steps_in_history() -> None:
    """每个 logical step 都在 TaskManager 留下 record,无幽灵步骤。"""
    # step1: model fail 4 次 → fail record;step2: finish。
    outcomes = ["本地模型调用失败。"] * 4 + ['Action: finish(result="ok")']
    backend = SequenceBackend(outcomes)
    controls = MemoryControls()
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(backend, controls, manager, retry_count=3, max_steps=2)(
            Msg("u", "任务", "user"),
        ),
    )
    # 2 个 logical step 都存在,无幽灵。
    assert len(manager.state.steps) == 2
    assert manager.state.step_count == 2


def test_agent_to_control_permission_boundary() -> None:
    """Agent → ActionDispatcher: authorized action 执行,unauthorized 拒绝。"""
    # finish 不需要 scope → 成功;click 在无 scope 时被拒。
    backend = SequenceBackend(
        ["Action: click(x=50, y=50)", 'Action: finish(result="done")'],
    )
    controls = MemoryControls()
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    # click 被 PermissionScope 拒绝(已在 reply 中激活 scope,但测试验证
    # GuiAgent 的 vertical chain 真正经过 dispatcher authorization)。
    assert result.content == "done"


def test_deterministic_e2e_multi_step_with_retry() -> None:
    """E2E: step1 click(fail→retry→success),step2 finish。"""
    backend = SequenceBackend(
        [
            "Action: click(x=10, y=20)",
            "Action: click(x=30, y=40)",
            'Action: finish(result="complete")',
        ],
    )
    controls = MemoryControls(fail_first=1)
    manager = TaskManager("任务")
    result = asyncio.run(
        make_agent(backend, controls, manager, retry_count=3)(Msg("u", "任务", "user")),
    )
    assert result.content == "complete"
    assert manager.state.status is TaskStatus.SUCCESS
    assert len(manager.state.steps) == 2
    # step1 有 2 个 attempt(fail + success)。
    assert len(manager.state.steps[0].attempts) == 2


def test_dispatcher_maps_extended_mouse_actions() -> None:
    """right_click/double_click/drag 按归一化坐标映射到控制器。"""
    calls: list[tuple[object, ...]] = []

    class Controls:
        def click(self, x=None, y=None, button="left"):
            calls.append(("click", x, y))

        def right_click(self, x=None, y=None):
            calls.append(("right_click", x, y))

        def double_click(self, x=None, y=None):
            calls.append(("double_click", x, y))

        def drag_from_to(self, x1, y1, x2, y2, duration=0.5):
            calls.append(("drag", x1, y1, x2, y2))

        def type(self, text):
            calls.append(("type", text))

        def scroll(self, direction, steps):
            calls.append(("scroll", direction, steps))

        def hotkey(self, *keys):
            calls.append(("hotkey", *keys))

    controls = Controls()
    dispatcher = ActionDispatcher(
        controls,
        controls,
        coordinate_mode="normalized_1000",
    )
    dispatcher.activate_run_scope(
        PermissionScope(
            allowed_actions=frozenset(
                {"right_click", "double_click", "drag"},
            ),
            token=1,
        ),
    )
    assert dispatcher.dispatch(
        {"action_type": "right_click", "params": {"x": 0, "y": 1000}},
        (201, 101),
        (40, 60),
    )
    assert dispatcher.dispatch(
        {"action_type": "double_click", "params": {"x": 1000, "y": 0}},
        (201, 101),
        (40, 60),
    )
    assert dispatcher.dispatch(
        {
            "action_type": "drag",
            "params": {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000},
        },
        (201, 101),
        (40, 60),
    )
    assert calls == [
        ("right_click", 40, 160),
        ("double_click", 240, 60),
        ("drag", 40, 60, 240, 160),
    ]


def test_repeated_strategy_allows_quantified_progress() -> None:
    """同一动作正在推进量化事实时不得被无进展保护误拦。"""
    from dataclasses import replace

    from agent.action_parser import ActionPromptState

    agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager("任务"),
        decision_protocol_v3=True,
    )
    action = {"action_type": "hotkey", "params": {"keys": ("media_volume_up",)}}
    serialized = 'hotkey(key1="media_volume_up")'
    base = ActionPromptState(
        step_number=3,
        max_steps=10,
        last_action=serialized,
        same_action_streak=2,
        ui_change_signal="weak",
    )
    assert agent._repeated_strategy_blocked(action, base)
    progressed = replace(base, progress_status="INSUFFICIENT_RATE")
    assert not agent._repeated_strategy_blocked(action, progressed)


def test_hidden_raw_foreground_disappearance_is_not_window_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最小化终端的 raw HWND 消失不能冒充业务应用窗口关闭。"""
    from agent import gui_agent as gui_agent_module

    agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager("任务"),
    )
    frame = Image.new("RGB", (40, 40), "black")
    monkeypatch.setattr(agent, "_wait_for_ui_stable", lambda: frame)
    monkeypatch.setattr(gui_agent_module, "get_foreground_hwnd", lambda: 200)
    monkeypatch.setattr(
        gui_agent_module,
        "is_window_existing",
        lambda hwnd: hwnd != 100,
    )
    observation = agent._verify_action_effect(
        "click",
        frame,
        before_foreground=100,
        before_app_foreground=300,
    )
    assert observation.effect == "foreground_window_changed"


def test_make_agent_accepts_ocr_recognizer_none() -> None:
    """make_agent 支持省略 OCR(默认 None 不启用)。"""
    agent = make_agent(
        SequenceBackend(['Action: finish(result="ok")']),
        MemoryControls(),
        TaskManager("t"),
    )
    assert agent._dependencies.ocr_recognizer is None


def test_scale_image_for_model_downscales_and_preserves_small() -> None:
    """超过上限的截图等比缩放,小图保持原样。"""
    from PIL import Image

    from perception.prompt_context import scale_image_for_model

    large = Image.new("RGB", (2880, 1800))
    scaled = scale_image_for_model(large, 1280)
    assert scaled.size == (1280, 800)

    small = Image.new("RGB", (640, 480))
    assert scale_image_for_model(small, 1280).size == (640, 480)

    portrait = Image.new("RGB", (600, 2400))
    scaled_portrait = scale_image_for_model(portrait, 1280)
    assert scaled_portrait.size == (320, 1280)

    four_k = Image.new("RGB", (3840, 2160))
    scaled_4k = scale_image_for_model(four_k, 1280)
    assert scaled_4k.size == (1280, 720)

    low_res = Image.new("RGB", (1024, 768))
    assert scale_image_for_model(low_res, 1280).size == (1024, 768)


class _RecordingDiagnosticsWriter:
    """内存 fake 诊断写入器:记录事件,可选抛异常验证失败安全。"""

    def __init__(self, raise_exc: Exception | None = None) -> None:
        self.records: list[DiagnosticsRecord] = []
        self._exc = raise_exc

    def record(self, record: "DiagnosticsRecord") -> None:
        self.records.append(record)
        if self._exc is not None:
            raise self._exc


def test_parse_failure_records_diagnostics_with_response() -> None:
    """解析失败时记录 prompt、原始响应、步骤与尝试序号。"""
    writer = _RecordingDiagnosticsWriter()
    backend = SequenceBackend(["这不是动作"] * 4)
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(
            backend,
            MemoryControls(),
            manager,
            max_steps=1,
            retry_count=3,
            diagnostics_writer=writer,
        )(
            Msg("u", "任务", "user"),
        ),
    )
    # 单步内 initial + 3 次 fresh retry 均解析失败,每次 attempt 都有诊断。
    assert len(writer.records) == 4
    first = writer.records[0]
    assert first.response == "这不是动作"
    assert first.step_number == 1
    assert first.attempt == 0
    assert first.exception_type is None
    assert first.failure_reason.startswith("模型动作解析失败。")
    assert first.run_id.startswith("run_")
    # prompt 与发送给后端的是同一份完整文本。
    assert first.prompt == backend.prompts[0]
    assert "任务" in first.prompt


def test_model_exception_records_diagnostics_without_response() -> None:
    """模型调用抛异常时记录 response=None 与异常类型名。"""
    writer = _RecordingDiagnosticsWriter()
    backend = SequenceBackend([RuntimeError("boom")])
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(
            backend,
            MemoryControls(),
            manager,
            max_steps=1,
            retry_count=0,
            diagnostics_writer=writer,
        )(
            Msg("u", "任务", "user"),
        ),
    )
    assert len(writer.records) == 1
    record = writer.records[0]
    assert record.response is None
    assert record.exception_type == "RuntimeError"
    assert record.failure_reason == "模型调用失败。"


def test_api_exhausted_failure_records_diagnostics() -> None:
    """API 重试耗尽的固定响应进入诊断记录,reason 为 api_retry_exhausted。"""
    writer = _RecordingDiagnosticsWriter()
    backend = SequenceBackend(["API 模型调用失败。"] * 3)
    manager = TaskManager("任务")
    asyncio.run(
        make_agent(
            backend,
            MemoryControls(),
            manager,
            max_steps=2,
            diagnostics_writer=writer,
        )(
            Msg("u", "任务", "user"),
        ),
    )
    # max_steps=2 内每步一次 API 耗尽,均留诊断记录。
    assert len(writer.records) == 2
    record = writer.records[0]
    assert record.response == "API 模型调用失败。"
    assert record.failure_reason == "api_retry_exhausted"


def test_successful_run_records_no_diagnostics() -> None:
    """成功运行不写任何诊断事件。"""
    writer = _RecordingDiagnosticsWriter()
    backend = SequenceBackend(
        ["Action: click(x=1, y=1)", 'Action: finish(result="done")'],
    )
    asyncio.run(
        make_agent(
            backend,
            MemoryControls(),
            TaskManager("任务"),
            diagnostics_writer=writer,
        )(
            Msg("u", "任务", "user"),
        ),
    )
    assert writer.records == []


def test_diagnostics_writer_failure_does_not_break_task() -> None:
    """诊断写入器抛异常时任务流程不受影响。"""
    writer = _RecordingDiagnosticsWriter(raise_exc=OSError("disk full"))
    backend = SequenceBackend(["这不是动作", 'Action: finish(result="ok")'])
    result = asyncio.run(
        make_agent(
            backend,
            MemoryControls(),
            TaskManager("任务"),
            diagnostics_writer=writer,
        )(
            Msg("u", "任务", "user"),
        ),
    )
    assert result.content == "ok"


def test_diagnostics_run_id_scoped_to_single_run() -> None:
    """run 标识在任务结束后清空,两次 run 的诊断归属各自 JSONL。"""
    writer = _RecordingDiagnosticsWriter()
    backend = SequenceBackend(["这不是动作", 'Action: finish(result="ok")'])
    agent = make_agent(
        backend,
        MemoryControls(),
        TaskManager("任务"),
        diagnostics_writer=writer,
    )
    asyncio.run(agent(Msg("u", "任务", "user")))
    backend2 = SequenceBackend(["这不是动作", 'Action: finish(result="ok2")'])
    agent2 = make_agent(
        backend2,
        MemoryControls(),
        TaskManager("任务2"),
        diagnostics_writer=writer,
    )
    asyncio.run(agent2(Msg("u", "任务2", "user")))
    assert len(writer.records) == 2
    assert writer.records[0].run_id != writer.records[1].run_id


def test_diagnostics_writer_validation_rejects_non_callable() -> None:
    """diagnostics_writer 必须提供可调用 record,否则构造时拒绝。"""
    controls = MemoryControls()
    with pytest.raises(TypeError):
        GuiAgentDependencies(
            model_client=SequenceBackend([]),
            action_dispatcher=ActionDispatcher(controls, controls),
            capture=CountingCapture(),
            task_manager_factory=lambda task: TaskManager("任务"),
            sleep=lambda seconds: None,
            diagnostics_writer=123,  # type: ignore[arg-type]
        )


def test_diagnostics_writer_writes_jsonl_and_png(tmp_path) -> None:
    """production 写入器把 JSONL 行与 PNG 截图落盘到 diagnosis 子目录。"""
    import json

    from PIL import Image

    from agent.diagnostics import ActionDiagnosticsWriter, DiagnosticsRecord

    writer = ActionDiagnosticsWriter(tmp_path)
    image = Image.new("RGB", (4, 4))
    writer.record(
        DiagnosticsRecord(
            run_id="run_x",
            step_number=2,
            attempt=1,
            image=image,
            prompt="prompt文本",
            response="原始响应",
            failure_reason="模型动作解析失败。 (invalid_format)",
        )
    )
    diagnosis_dir = tmp_path / "diagnosis"
    png = diagnosis_dir / "run_x_step2_att1.png"
    assert png.exists()
    jsonl = diagnosis_dir / "run_x.jsonl"
    assert jsonl.exists()
    entry = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert entry["run_id"] == "run_x"
    assert entry["step_number"] == 2
    assert entry["attempt"] == 1
    assert entry["prompt"] == "prompt文本"
    assert entry["response"] == "原始响应"
    assert entry["screenshot"] == "run_x_step2_att1.png"
    assert entry["failure_reason"] == "模型动作解析失败。 (invalid_format)"
    assert entry["exception_type"] is None
    assert entry["timestamp"]


def test_diagnostics_writer_validates_fields(tmp_path) -> None:
    """字段类型或值域错误在写入副作用前抛出。"""
    from PIL import Image

    from agent.diagnostics import ActionDiagnosticsWriter, DiagnosticsRecord

    writer = ActionDiagnosticsWriter(tmp_path)
    image = Image.new("RGB", (2, 2))
    base = dict(
        run_id="run_x",
        step_number=1,
        attempt=0,
        image=image,
        prompt="p",
        response="r",
        failure_reason="失败",
    )

    def expect_invalid(exc: type[Exception], **override: object) -> None:
        with pytest.raises(exc):
            writer.record(
                DiagnosticsRecord(  # type: ignore[arg-type]  # 负向
                    **{**base, **override}
                )
            )

    expect_invalid(ValueError, run_id="")
    expect_invalid(ValueError, step_number=0)
    expect_invalid(ValueError, attempt=-1)
    expect_invalid(TypeError, image="not-image")
    expect_invalid(TypeError, prompt=1)
    expect_invalid(TypeError, response=1)
    with pytest.raises(ValueError):
        writer.record(DiagnosticsRecord(**{**base, "failure_reason": ""}))
    with pytest.raises(TypeError):
        writer.record(DiagnosticsRecord(**{**base, "exception_type": 5}))
    # 全部校验失败都不产生任何文件。
    assert not (tmp_path / "diagnosis").exists()


def test_diagnostics_writer_io_failure_is_swallowed(
    tmp_path, monkeypatch, caplog
) -> None:
    """落盘 I/O 失败只记 warning,不向调用方抛出。"""
    import logging

    from PIL import Image

    from agent.diagnostics import ActionDiagnosticsWriter, DiagnosticsRecord

    writer = ActionDiagnosticsWriter(tmp_path)

    def _fail_save(self, path, format=None):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", _fail_save)
    with caplog.at_level(logging.WARNING):
        writer.record(
            DiagnosticsRecord(
                run_id="run_x",
                step_number=1,
                attempt=0,
                image=Image.new("RGB", (2, 2)),
                prompt="p",
                response="r",
                failure_reason="失败",
            )
        )
    assert any(
        "diagnostics_write_failed" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_diagnostics_writer_rejects_non_path_log_dir() -> None:
    """log_dir 类型错误在构造时拒绝。"""
    from agent.diagnostics import ActionDiagnosticsWriter

    with pytest.raises(TypeError):
        ActionDiagnosticsWriter("logs")  # type: ignore[arg-type]


def test_parse_failure_retry_prompt_contains_format_feedback() -> None:
    """解析失败后的 fresh retry 携带格式纠正提示(经既有 last_error 通道)。"""
    backend = SequenceBackend(
        ["不是动作", 'Action: finish(result="ok")'],
    )
    result = asyncio.run(
        make_agent(
            backend,
            MemoryControls(),
            TaskManager("任务"),
            retry_count=3,
        )(
            Msg("u", "任务", "user"),
        ),
    )
    assert result.content == "ok"
    # 第二次调用的 Prompt 应包含针对上次解析失败的纠正提示
    assert len(backend.prompts) == 2
    assert "上次" in backend.prompts[1]
    assert "Action" in backend.prompts[1]


# ======================================================================
# E2-B characterization:_ocr_target_window_text 的显示器坐标帧
# ======================================================================


@pytest.mark.parametrize(
    (
        "virtual_origin",
        "window_rect",
        "expected_region",
        "case_label",
    ),
    [
        ((0, 0), (100, 50, 300, 200), (100, 50, 300, 200), "主屏原点"),
        ((0, 0), (2000, 60, 300, 200), (2000, 60, 300, 200), "正偏移副屏"),
        (
            (-1920, 0),
            (100, 50, 300, 200),
            (2020, 50, 300, 200),
            "负 X 虚拟原点",
        ),
    ],
)
def test_ocr_window_region_monitor_frames(
    monkeypatch,
    virtual_origin,
    window_rect,
    expected_region,
    case_label,
) -> None:
    """窗口矩形按虚拟桌面绝对坐标换算为 monitors[0] 相对 region。

    坐标空间:窗口矩形=VIRTUAL_DESKTOP_ABSOLUTE;capture_screen 的
    region=MONITOR_LOCAL(monitors[0]);换算=减虚拟桌面原点。
    """
    from PIL import Image

    import perception.screenshot as screenshot_module
    from agent import gui_agent as ga_module

    monkeypatch.setattr(
        screenshot_module,
        "get_window_screen_rect_clipped",
        lambda hwnd: window_rect,
        raising=True,
    )
    monkeypatch.setattr(
        screenshot_module,
        "virtual_desktop_origin",
        lambda: virtual_origin,
        raising=True,
    )
    captured: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> Image.Image:
        captured.update(kwargs)
        return Image.new("RGB", (8, 8))

    agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager("任务"),
        capture=fake_capture,
    )
    monkeypatch.setattr(
        ga_module.prompt_context,
        "perceive_ocr_elements_detailed",
        lambda recognizer, image, point: ((), [{"text": "ok"}]),
        raising=True,
    )
    agent._ocr_target_window_text(4242)
    assert captured["region"] == expected_region, case_label


def test_ocr_window_result_band_uses_relative_height(monkeypatch) -> None:
    """结果带按窗口相对高度裁剪，不依赖桌面固定坐标。"""
    import perception.screenshot as screenshot_module
    from agent import gui_agent as gui_agent_module

    monkeypatch.setattr(
        screenshot_module,
        "get_window_screen_rect_clipped",
        lambda hwnd: (100, 50, 301, 201),
    )
    monkeypatch.setattr(screenshot_module, "virtual_desktop_origin", lambda: (0, 0))
    captured: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> Image.Image:
        captured.update(kwargs)
        return Image.new("RGB", (8, 8))

    agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        TaskManager("任务"),
        capture=fake_capture,
    )
    monkeypatch.setattr(
        gui_agent_module.prompt_context,
        "perceive_ocr_elements_detailed",
        lambda recognizer, image, point: ((), ()),
    )
    agent._ocr_target_window_text(4242, 0.32)
    assert captured["region"] == (100, 50, 301, 64)


def test_calculator_operand_is_not_treated_as_evaluated_result() -> None:
    """结果带只有当前操作数时仍属未完成，必须出现等号完成态。"""
    from agent.gui_agent import _evaluated_calculator_result_text

    assert _evaluated_calculator_result_text("1 + 2") == ""
    assert _evaluated_calculator_result_text("1 + 1 = 2") == "1 + 1 = 2"


@pytest.mark.parametrize(
    ("response", "current", "dispatched", "expected"),
    [
        (r'Action: type(text="a\tb\nc\td")', False, True, True),
        (r'Action: type(text="a\tb\nc\td\n")', True, True, False),
        ('Action: type(text="hello")', False, True, False),
        (r'Action: type(text="a\tb")', False, False, False),
    ],
)
def test_structured_entry_pending_classification(
    response: str,
    current: bool,
    dispatched: bool,
    expected: bool,
) -> None:
    """只有成功分发且末项未提交的 structured type 才设置 pending。"""
    from agent.action_parser import parse_prd_action
    from agent.gui_agent import _updated_structured_entry_pending

    action = parse_prd_action(response)
    assert action is not None
    assert _updated_structured_entry_pending(current, action, dispatched) is expected


def test_structured_entry_pending_clears_only_on_explicit_commit_navigation() -> None:
    """pending 只由成功分发的单键提交/导航动作清除。"""
    from agent.action_parser import parse_prd_action
    from agent.gui_agent import _updated_structured_entry_pending

    enter = parse_prd_action('Action: hotkey(key1="enter")')
    tab = parse_prd_action('Action: hotkey(key1="tab")')
    unrelated = parse_prd_action('Action: hotkey(key1="ctrl", key2="s")')
    assert enter is not None
    assert tab is not None
    assert unrelated is not None

    assert not _updated_structured_entry_pending(True, enter, True)
    assert not _updated_structured_entry_pending(True, tab, True)
    assert _updated_structured_entry_pending(True, tab, False)
    assert _updated_structured_entry_pending(True, unrelated, True)


@pytest.mark.parametrize("model_mode", ["api", "local"])
def test_structured_entry_finish_guard_recovers_then_allows_finish(
    model_mode: str,
) -> None:
    """API/local 共用 guard：拒绝过早 finish，提交后允许正常完成。"""
    backend = SequenceBackend(
        [
            r'Action: type(text="a\tb\nc\td")',
            'Action: finish(result="premature")',
            'Action: hotkey(key1="tab")',
            'Action: finish(result="done")',
        ],
    )
    controls = MemoryControls()
    manager = TaskManager("录入多字段数据")
    agent = make_agent(
        backend,
        controls,
        manager,
        max_steps=4,
        retry_count=0,
        model_mode=model_mode,
        decision_protocol_v3=True,
    )

    result = asyncio.run(agent(Msg("u", "录入多字段数据", "user")))

    assert result.content == "done"
    assert controls.calls == [
        ("type", "a\tb\nc\td"),
        ("hotkey", "tab"),
    ]
    recovery_prompt = backend.prompts[2]
    assert (
        "Previous structured entry may still have its final field" in recovery_prompt
        or "上一结构化输入可能仍有最后字段处于编辑状态" in recovery_prompt
    )
    feedback_slice = (
        recovery_prompt.split("Recovery feedback:", 1)[1].split(
            "Current perception:",
            1,
        )[0]
        if "Recovery feedback:" in recovery_prompt
        else next(
            line
            for line in recovery_prompt.splitlines()
            if "上一结构化输入可能仍有最后字段处于编辑状态" in line
        )
    )
    for forbidden in ("Enter", "Tab", "Excel", "C4", "M01"):
        assert forbidden not in feedback_slice
    finish_attempt = manager.state.steps[1].attempts[0]
    assert finish_attempt.stage == "finish"
    assert finish_attempt.succeeded is False
    assert "最后字段处于编辑状态" in (finish_attempt.failure_reason or "")


def test_structured_entry_pending_resets_at_new_task_boundary() -> None:
    """同一 Agent 开始新任务时清空上一任务的 pending 状态。"""
    from dataclasses import replace

    backend = SequenceBackend(
        [
            r'Action: type(text="a\tb")',
            'Action: finish(result="new-task-done")',
        ],
    )
    agent = make_agent(
        backend,
        MemoryControls(),
        TaskManager("首次任务"),
        max_steps=1,
        retry_count=0,
        decision_protocol_v3=True,
    )
    agent._dependencies = replace(
        agent._dependencies,
        task_manager_factory=TaskManager,
    )

    first = asyncio.run(agent(Msg("u", "首次任务", "user")))
    assert "任务执行失败" in first.content
    agent._pending_structured_entry_commit = True

    second = asyncio.run(agent(Msg("u", "新任务", "user")))

    assert second.content == "new-task-done"
    assert agent._pending_structured_entry_commit is False
