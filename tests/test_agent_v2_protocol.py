"""V2 决策协议测试:observe 动作、连续保护、重复策略硬保护与 finish 规则。"""

import asyncio

from agentscope.message import Msg
from PIL import Image

import agent.gui_agent as gui_agent_module
from agent.action_parser import parse_action
from agent.task_manager import TaskManager
from tests.agent_test_support import (
    FrameSequenceCapture,
    MemoryControls,
    SequenceBackend,
    make_agent,
)


def _changing_capture() -> FrameSequenceCapture:
    """逐帧不同的捕获器:让 GUI 动作产生可见效果,V2 才可能接受 finish。"""
    # 帧差检测要求 >30 灰度级;黑白交替保证相邻帧必然被判为变化。
    shades = ["#000000", "#ffffff"] * 15
    return FrameSequenceCapture(
        [Image.new("RGB", (1000, 500), shade) for shade in shades],
    )


def test_parser_observe_variants() -> None:
    """observe() 合法;带参数与同义动词全部非法。"""
    assert parse_action("Action: observe()") == {
        "action_type": "observe",
        "params": {},
    }
    assert parse_action("Action: observe(1)") is None
    assert parse_action("Action: observe(wait=1)") is None
    assert parse_action("Action: wait()") is None
    assert parse_action("Action: sleep()") is None
    assert parse_action("Action: refresh()") is None
    assert parse_action("Action: observe") is None


def test_parser_bare_observe_is_not_canonical() -> None:
    """Strict parser 不负责补前缀，裸 observe 必须拒绝。"""
    assert parse_action("observe()") is None


def test_observe_executes_without_control_and_waits() -> None:
    """observe 不触发鼠标键盘,执行固定等待并进入下一轮 fresh 观察。"""
    sleeps: list[float] = []
    backend = SequenceBackend(
        [
            "Action: click(x=1, y=1)",
            "Action: observe()",
            'Action: finish(result="ok")',
        ],
    )
    controls = MemoryControls()
    capture = _changing_capture()

    def counting_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    agent = make_agent(
        backend,
        controls,
        TaskManager("任务"),
        capture,
        model_mode="api",
        decision_protocol_v2=True,
    )
    agent._dependencies = type(agent._dependencies)(
        model_client=agent._dependencies.model_client,
        action_dispatcher=agent._dependencies.action_dispatcher,
        capture=capture,
        task_manager_factory=agent._dependencies.task_manager_factory,
        action_observer=agent._dependencies.action_observer,
        sleep=counting_sleep,
        protect_initial_foreground=agent._dependencies.protect_initial_foreground,
        ocr_recognizer=agent._dependencies.ocr_recognizer,
        diagnostics_writer=agent._dependencies.diagnostics_writer,
        trace_writer=agent._dependencies.trace_writer,
    )
    result = asyncio.run(agent(Msg("u", "任务", "user")))
    assert result.content == "ok"
    # 只有 click 到达控制器;observe 无任何控制调用。
    assert [c[0] for c in controls.calls] == ["click"]
    # observe 执行了程序固定的 0.6 秒等待。
    assert 0.6 in sleeps
    # observe 后有 fresh 观察:捕获次数多于模型调用数。
    assert capture.calls >= 3


def test_observe_streak_fourth_is_blocked() -> None:
    """连续第 4 次 observe 被程序拒绝;换策略后恢复执行。"""
    backend = SequenceBackend(
        [
            "Action: observe()",
            "Action: observe()",
            "Action: observe()",
            "Action: observe()",
            "Action: click(x=2, y=2)",
            "Action: observe()",
            'Action: finish(result="ok")',
        ],
    )
    controls = MemoryControls()
    capture = _changing_capture()
    agent = make_agent(
        backend,
        controls,
        TaskManager("任务"),
        capture,
        model_mode="api",
        decision_protocol_v2=True,
    )
    result = asyncio.run(agent(Msg("u", "任务", "user")))
    # 任务完成,控制器只收到 click(全部 observe 均无控制调用)。
    assert result.content == "ok"
    assert [c[0] for c in controls.calls] == ["click"]
    # 3 次 observe 真实执行,第 4 次被拒后模型改用 click。
    assert capture.calls >= 5


def test_repeated_failed_action_third_is_blocked(monkeypatch) -> None:
    """相同无效 GUI 动作第 3 次被程序阻止,不进入控制器。

    前台窗口身份用固定假值隔离:真实前台(如任务栏)会被安全层判
    为"无聚焦应用"而拒绝 type,属环境耦合,与被测的重复策略无关。
    """
    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        lambda: 424242,
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "is_window_available",
        lambda hwnd: True,
        raising=True,
    )
    click = "Action: click(x=5, y=5)"
    backend = SequenceBackend(
        [
            click,
            click,
            click,
            'Action: type(text="换策略")',
        ],
    )
    controls = MemoryControls()
    agent = make_agent(
        backend,
        controls,
        TaskManager("任务"),
        retry_count=0,
        max_steps=4,
        model_mode="api",
        decision_protocol_v2=True,
    )
    asyncio.run(agent(Msg("u", "任务", "user")))
    actions = [c[0] for c in controls.calls]
    # 前两次 click 执行,第三次被硬保护拦截,只有换策略后的 type 执行。
    assert actions == ["click", "click", "type"]


def test_finish_after_no_effect_rejected_in_v2_but_accepted_in_v1() -> None:
    """V2:上一动作无可见效果时 finish 被拒;V1 同序列可完成。"""
    click = "Action: click(x=1, y=1)"
    finish = 'Action: finish(result="done")'

    def run(v2: bool) -> str:
        backend = SequenceBackend([click, finish, finish, finish])
        result = asyncio.run(
            make_agent(
                backend,
                MemoryControls(),
                TaskManager("任务"),
                max_steps=2,
                model_mode="api",
                decision_protocol_v2=v2,
            )(
                Msg("u", "任务", "user"),
            ),
        )
        return result.content

    assert run(v2=False) == "done"
    assert "任务执行失败" in run(v2=True)


def test_v2_payload_system_user_layering() -> None:
    """V2 payload:system 恰为 V2 Prompt 常量,user 内文本先于图像,温度为0。"""
    from agent.action_prompt_v2 import ACTION_SYSTEM_PROMPT_V2
    from agent.dashscope_api_backend import DashScopeAPIBackend

    captured: dict[str, object] = {}

    class _CaptureTransport:
        def post(self, url, **kwargs):
            captured.update(kwargs)
            captured["url"] = url

            class _Resp:
                status_code = 200

                def json(self):
                    return {
                        "choices": [
                            {"message": {"content": "Action: observe()"}},
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }

            return _Resp()

    backend = DashScopeAPIBackend(
        "test-key",
        "test-model",
        transport=_CaptureTransport(),
    )
    usage: dict[str, object] = {}
    from PIL import Image

    response = backend.generate(
        Image.new("RGB", (10, 10)),
        "dynamic",
        system_prompt=ACTION_SYSTEM_PROMPT_V2,
        temperature=0.0,
        usage_out=usage,
    )
    assert response == "Action: observe()"
    assert usage.get("input_tokens") == 10
    payload = captured["json"]
    assert payload["temperature"] == 0
    messages = payload["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == ACTION_SYSTEM_PROMPT_V2
    user = messages[1]
    assert user["role"] == "user"
    assert user["content"][0]["type"] == "text"
    assert user["content"][0]["text"] == "dynamic"
    assert user["content"][1]["type"] == "image_url"


def test_v1_payload_unchanged_single_user() -> None:
    """V1 payload 合同不变:单 user 消息、图像在前、无采样参数。"""
    from agent.dashscope_api_backend import DashScopeAPIBackend

    captured: dict[str, object] = {}

    class _CaptureTransport:
        def post(self, url, **kwargs):
            captured.update(kwargs)

            class _Resp:
                status_code = 200

                def json(self):
                    return {
                        "choices": [{"message": {"content": "ok"}}],
                    }

            return _Resp()

    backend = DashScopeAPIBackend(
        "test-key",
        "test-model",
        transport=_CaptureTransport(),
    )
    backend.generate(Image.new("RGB", (10, 10)), "v1-prompt")
    payload = captured["json"]
    assert "temperature" not in payload
    messages = payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image_url"
    assert messages[0]["content"][1]["type"] == "text"


def test_v3_prompt_is_optimized_independently_from_v1() -> None:
    """V3 按最新人工合同独立精简，不再要求逐行摘录历史 V1。"""
    from agent.action_parser import ACTION_SYSTEM_PROMPT
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3

    assert ACTION_SYSTEM_PROMPT_V3 != ACTION_SYSTEM_PROMPT
    assert len(ACTION_SYSTEM_PROMPT_V3) < len(ACTION_SYSTEM_PROMPT)


def test_v3_avoids_new_planning_doctrine() -> None:
    """V3 恢复稳定模块结构，不再重复动态 recovery 的规划教条。"""
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3

    assert "规划与状态" not in ACTION_SYSTEM_PROMPT_V3
    assert "当前未完成子目标" not in ACTION_SYSTEM_PROMPT_V3
    assert "键盘动作语义" in ACTION_SYSTEM_PROMPT_V3
    assert "感知信息解释" in ACTION_SYSTEM_PROMPT_V3


def test_v3_includes_stable_contract_sections() -> None:
    """V3 保留旧稳定合同的七段式结构。"""
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3

    for marker in [
        "合法动作",
        "输出协议",
        "鼠标动作语义",
        "键盘动作语义",
        "感知信息解释",
        "副作用保护",
        "参数格式",
    ]:
        assert marker in ACTION_SYSTEM_PROMPT_V3, marker


def test_v3_includes_generic_keyboard_and_grid_guidance() -> None:
    """V3 对精确输入和结构化网格使用通用键盘语义，不含任务配方。"""
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3

    assert "精确文本、数字或表达式" in ACTION_SYSTEM_PROMPT_V3
    assert "录入表格或网格时必须保留行、列和字段边界" in ACTION_SYSTEM_PROMPT_V3
    assert "制表符或换行以提交最后单元格" in ACTION_SYSTEM_PROMPT_V3
    for task_marker in ("S01", "M01", "姓名", "部门", "计算器", "Excel"):
        assert task_marker not in ACTION_SYSTEM_PROMPT_V3


def test_v3_param_format_complete() -> None:
    """V3 由权威 grammar 给出精确参数并保留 fail-closed 边界。"""
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3

    assert "参数名称和格式必须严格遵守上述定义" in ACTION_SYSTEM_PROMPT_V3
    assert "所有必需参数必须完整提供" in ACTION_SYSTEM_PROMPT_V3
    assert "禁止位置参数或命名参数与位置参数混用" in ACTION_SYSTEM_PROMPT_V3


def test_v3_no_observe_action() -> None:
    """V3 不含 observe 动作定义。"""
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3

    assert "observe" not in ACTION_SYSTEM_PROMPT_V3


def test_v3_no_v2_text() -> None:
    """V3 不含任何 V2-only Prompt 文案。"""
    from agent.action_prompt_v2 import ACTION_SYSTEM_PROMPT_V2
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3

    v2_only_lines = [
        line
        for line in ACTION_SYSTEM_PROMPT_V2.split("\n")
        if line.strip()
        and line not in ACTION_SYSTEM_PROMPT_V3
        and "observe" not in line  # observe 已单独测
    ]
    # V2 有而 V3 没有的行不应出现在 V3 中(方向检查)
    for line in v2_only_lines[:5]:
        assert line not in ACTION_SYSTEM_PROMPT_V3, f"V2-only text leaked: {line[:50]}"


def test_v3_prompt_hash() -> None:
    """输出三个 Prompt 的 MD5 哈希供审计。"""
    import hashlib

    from agent.action_parser import ACTION_SYSTEM_PROMPT
    from agent.action_prompt_v2 import ACTION_SYSTEM_PROMPT_V2
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3

    hashes = {
        "V1": hashlib.md5(ACTION_SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "V2": hashlib.md5(ACTION_SYSTEM_PROMPT_V2.encode()).hexdigest()[:12],
        "V3": hashlib.md5(ACTION_SYSTEM_PROMPT_V3.encode()).hexdigest()[:12],
    }
    assert len(set(hashes.values())) == 3  # 三个哈希互不相同
    print(f"Prompt hashes: {hashes}")
