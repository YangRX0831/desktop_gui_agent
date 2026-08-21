"""V3 八动作 payload、规划提示与消息分层回归测试。

背景:V3 曾把 ``compose_action_prompt`` 的完整输出(含 3081 字符 V1 静态
Prompt)作为 user message 发送,同时 system message 又携带 V3 Core,导致
静态规则重复且已删除的 G/H/I/J 模块经 user 侧回流。本文件固化修复后的
合同:V3 system 只含优化后的 ``ACTION_SYSTEM_PROMPT_V3``,user 只含既有
可信动态状态块;V1/V2 消息分层保持不变。
"""

import asyncio
import hashlib

from agentscope.message import Msg
from PIL import Image

from agent.action_parser import (
    ACTION_SYSTEM_PROMPT,
    CANONICAL_V3_ACTIONS,
    ActionPromptState,
    compose_action_prompt,
)
from agent.action_prompt_v2 import ACTION_SYSTEM_PROMPT_V2, compose_action_prompt_v2
from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3, compose_action_prompt_v3
from agent.gui_agent import _PARSE_FEEDBACK_HINTS
from agent.model_client import ModelCallOptions
from agent.task_manager import TaskManager
from tests.agent_test_support import MemoryControls, make_agent

# 2026-08-21 generic keyboard/grid guidance 后,按固定 canonical state
# 实测的输出 SHA-256;用于防止后续 prompt 或动态块静默漂移。
_V1_FULL_SHA256 = "5031acbde5df13de16b9f864b06410d73f6deeaea182753b0937aff51611d4e9"
_V1_DYNAMIC_SHA256 = "d1dee7133d63d10fa277f8a0e981f39dd90209684fff00b8dfa38b472779a1a3"
_V2_USER_SHA256 = "e52799c4b86ccd25076a2f4ef33fa12f66b3b5e38e7609823e4c3d95d72cf17b"
_CANONICAL_TASK = "在记事本中输入你好世界"


def _canonical_state() -> ActionPromptState:
    """构造覆盖全部动态字段的固定状态,供哈希级不变性断言使用。"""
    return ActionPromptState(
        step_number=3,
        max_steps=10,
        task_target_window="notepad.exe:id=101",
        agent_ui_window="windowsterminal.exe:id=7",
        last_action="click(x=500, y=300)",
        last_dispatch_status="success",
        last_error="none",
        foreground_after="notepad.exe",
        last_effect="visible_content_changed",
        ui_change_signal="weak",
        keyboard_input_ready="true",
        focused_control="text_input",
        system_volume_percent=42,
        recent_actions=(
            "click:success:none",
            "type:success:visible_content_changed",
            "click:success:visible_content_changed",
        ),
        same_action_streak=1,
        no_ui_change_streak=0,
        platform="windows",
        ocr_elements=(
            "文件(F) bbox=[0.01,0.02,0.05,0.03]",
            "编辑(E) bbox=[0.06,0.02,0.10,0.03]",
        ),
        windows=("记事本 id=101 bbox=[0.1,0.1,0.8,0.8] fg=true",),
        current_goal="打开记事本",
    )


def _sha256(text: str) -> str:
    """返回 UTF-8 SHA-256 十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _PayloadCaptureBackend:
    """记录 ModelClient.generate 全部入参的 fake 模型客户端。"""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        mode: str = "local",
        options: ModelCallOptions | None = None,
        **kwargs: object,
    ) -> str:
        fields: dict[str, object] = {"image": image, "prompt": prompt, "mode": mode}
        if options is not None:
            if options.system_prompt is not None:
                fields["system_prompt"] = options.system_prompt
            if options.temperature is not None:
                fields["temperature"] = options.temperature
            if options.usage_out is not None:
                fields["usage_out"] = options.usage_out
        self.calls.append({**fields, **kwargs})
        return self._response


def _capture_first_call(
    *,
    model_mode: str = "api",
    decision_protocol_v2: bool = False,
    decision_protocol_v3: bool = False,
) -> dict[str, object]:
    """跑一次立即 finish 的任务,返回首次模型调用的完整入参。

    max_steps=1、retry_count=0 保证无论协议是否接受首次 finish(V2 的
    no-effect finish guard 会拒绝),每次 run 恰好产生一次模型调用。
    """
    backend = _PayloadCaptureBackend('Action: finish(result="done")')
    agent = make_agent(
        backend,
        MemoryControls(),
        TaskManager(_CANONICAL_TASK),
        max_steps=1,
        retry_count=0,
        model_mode=model_mode,
        reject_initial_finish=False,
        decision_protocol_v2=decision_protocol_v2,
        decision_protocol_v3=decision_protocol_v3,
    )
    asyncio.run(
        agent(
            Msg(
                "u",
                _CANONICAL_TASK,
                "user",
                metadata={"trace_task_id": "S05"},
            ),
        ),
    )
    assert len(backend.calls) == 1
    return backend.calls[0]


def test_v3_system_equals_v3_constant() -> None:
    """V3 api 模式:system 恰为 V3 常量,temperature 为 0。"""
    call = _capture_first_call(decision_protocol_v3=True)
    assert call["system_prompt"] == ACTION_SYSTEM_PROMPT_V3
    assert call["temperature"] == 0.0
    assert call["mode"] == "api"


def test_v3_trace_protocol_version_is_v3() -> None:
    """CLEAN_V3 模型调用必须在 trace 中标为 v3。"""
    recorded: list[dict[str, object]] = []

    class _TraceWriter:
        def record_model_call(self, record: dict[str, object]) -> None:
            recorded.append(record)

    agent = make_agent(
        _PayloadCaptureBackend('Action: finish(result="done")'),
        MemoryControls(),
        TaskManager(_CANONICAL_TASK),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        trace_writer=_TraceWriter(),  # type: ignore[arg-type]
    )
    asyncio.run(
        agent(
            Msg(
                "u",
                _CANONICAL_TASK,
                "user",
                metadata={"trace_task_id": "S05"},
            ),
        ),
    )

    model_call = next(row for row in recorded if row.get("record_type") == "model_call")
    assert model_call["protocol_version"] == "v3"
    assert model_call["task_id"] == "S05"
    assert model_call["task_digest"]
    assert "screenshot_sha256" in model_call
    perception_summary = model_call["perception_summary"]
    assert isinstance(perception_summary, dict)
    assert perception_summary["ocr_element_count"] == 0
    assert isinstance(perception_summary["window_count"], int)
    assert perception_summary["window_count"] >= 0
    assert perception_summary["grounding_candidate_count"] == 0
    assert model_call["symbolic_grounding_resolution"] is None
    assert model_call["structured_entry_commit_pending"] is False


def test_v3_blocks_third_identical_click_without_strong_progress() -> None:
    """V3 中光标/hover 微变不能放行第三次完全相同点击。"""
    agent = make_agent(
        _PayloadCaptureBackend('Action: finish(result="done")'),
        MemoryControls(),
        TaskManager(_CANONICAL_TASK),
        decision_protocol_v3=True,
    )
    state = ActionPromptState(
        step_number=3,
        max_steps=10,
        same_action_streak=2,
        last_action="click(x=500, y=69)",
        last_effect="visible_content_changed",
        ui_change_signal="weak",
    )
    action = {
        "action_type": "click",
        "params": {"x": 500, "y": 69},
    }
    assert agent._repeated_strategy_blocked(action, state)  # type: ignore[arg-type]


def test_v3_user_does_not_contain_full_v1_prompt() -> None:
    """V3 user 文本不再包含完整 V1 ACTION_SYSTEM_PROMPT(污染修复核心)。"""
    call = _capture_first_call(decision_protocol_v3=True)
    assert ACTION_SYSTEM_PROMPT not in call["prompt"]
    assert ACTION_SYSTEM_PROMPT_V2 not in call["prompt"]


def test_v3_user_does_not_contain_v3_prompt_duplicate() -> None:
    """V3 user 文本不含 V3 Core 静态重复(分层后静态只出现一次)。"""
    call = _capture_first_call(decision_protocol_v3=True)
    assert ACTION_SYSTEM_PROMPT_V3 not in call["prompt"]
    for line in ACTION_SYSTEM_PROMPT_V3.split("\n"):
        if line.strip():
            assert line not in call["prompt"], line


def test_v3_system_prompt_contains_exact_eight_action_contract() -> None:
    """真实 API system message 使用确切八动作合同与精简格式边界。"""
    for marker in (
        "合法动作(唯一协议，每轮只输出一个)",
        "禁止位置参数或命名参数与位置参数混用",
        "整个回答只能有一行",
        "禁止解释、计划、Markdown、代码块、前后缀或第二行",
        "只有任务确实完成时才能finish",
    ):
        assert marker in ACTION_SYSTEM_PROMPT_V3

    grammar_section = ACTION_SYSTEM_PROMPT_V3.split("输出协议：", 1)[0]
    positions = [
        grammar_section.index(f". {action}(") for action in CANONICAL_V3_ACTIONS
    ]
    assert positions == sorted(positions)
    for unsupported in ("observe(", "move_to(", "press(", "release("):
        assert unsupported not in grammar_section


def test_v3_prompt_restores_stable_module_structure() -> None:
    """稳定 V3 恢复七段式结构，不再加入独立规划教条。"""
    assert len(ACTION_SYSTEM_PROMPT_V3) < len(ACTION_SYSTEM_PROMPT)
    for marker in (
        "输出协议：",
        "鼠标动作语义：",
        "键盘动作语义：",
        "感知信息解释：",
        "副作用保护：",
        "参数格式：",
        "录入表格或网格时必须保留行、列和字段边界",
        "不因前台变化重新绑定",
    ):
        assert marker in ACTION_SYSTEM_PROMPT_V3
    assert "规划与状态：" not in ACTION_SYSTEM_PROMPT_V3


def test_parse_failure_feedback_lists_exact_eight_actions() -> None:
    """unsupported-action 反馈与 V3 单一事实源一致且排除 observe。"""
    feedback = _PARSE_FEEDBACK_HINTS["unsupported_action"]
    assert feedback == (
        "上次动作不在支持列表;只允许 " + "、".join(CANONICAL_V3_ACTIONS) + "。"
    )
    assert "observe" not in feedback


def test_v3_user_contains_only_dynamic_state_not_planning_copy() -> None:
    """规划规则只在 system 出现，user 保持可信动态事实与任务。"""
    call = _capture_first_call(decision_protocol_v3=True)
    prompt = str(call["prompt"])
    assert prompt.startswith("Current execution state:\n")
    assert "动作选择：" not in prompt
    assert "规划与状态：" not in prompt


def test_v3_static_prompt_occurrence_count_is_one() -> None:
    """API 可见全文本中 V3 静态恰一次;local 模式使用 Compact Prompt。"""
    api_call = _capture_first_call(decision_protocol_v3=True)
    combined = f"{api_call['system_prompt']}\n\n{api_call['prompt']}"
    assert combined.count(ACTION_SYSTEM_PROMPT_V3) == 1
    assert combined.count(ACTION_SYSTEM_PROMPT) == 0
    assert combined.count(ACTION_SYSTEM_PROMPT_V2) == 0
    # local+V3 自 KEYBOARD-FIRST 阶段起使用 LOCAL Compact Prompt:
    # 不含 V3/V1 静态,不含 grounding 候选,单文本无采样参数。
    from agent.local_compact_prompt import LOCAL_ACTION_SYSTEM_PROMPT

    local_call = _capture_first_call(
        model_mode="local",
        decision_protocol_v3=True,
    )
    assert "system_prompt" not in local_call
    assert "temperature" not in local_call
    local_text = str(local_call["prompt"])
    assert local_text.count(LOCAL_ACTION_SYSTEM_PROMPT) == 1
    assert ACTION_SYSTEM_PROMPT_V3 not in local_text
    assert ACTION_SYSTEM_PROMPT not in local_text
    assert "Interactive elements" not in local_text


def test_v1_payload_contract_shape_and_hardened_hash() -> None:
    """V1 保持单 user 消息形态,并冻结本批 hardening 后的 prompt。"""
    call = _capture_first_call()
    assert "system_prompt" not in call
    assert "temperature" not in call
    prompt = str(call["prompt"])
    assert prompt.startswith(
        f"{ACTION_SYSTEM_PROMPT}\n\nCurrent execution state:\n",
    )
    assert prompt.endswith("现在只输出一行合法Action，不要输出其他内容。")
    # composer 级冻结:固定状态的完整输出与 hardening 后哈希一致。
    v1_full = compose_action_prompt(
        _CANONICAL_TASK,
        _canonical_state(),
        "normalized_1000",
    )
    assert _sha256(v1_full) == _V1_FULL_SHA256


def test_v2_payload_unaffected() -> None:
    """V2 payload 不受修复影响:system=V2 常量、temperature=0、用户块不变。"""
    call = _capture_first_call(decision_protocol_v2=True)
    assert call["system_prompt"] == ACTION_SYSTEM_PROMPT_V2
    assert call["temperature"] == 0.0
    assert str(call["prompt"]).startswith("OVERALL TASK:")
    assert ACTION_SYSTEM_PROMPT not in str(call["prompt"])
    v2_user = compose_action_prompt_v2(_CANONICAL_TASK, _canonical_state())
    assert _sha256(v2_user) == _V2_USER_SHA256


def test_v3_dynamic_block_equals_v1_dynamic_block() -> None:
    """V3 动态块与 V1 动态块逐字一致:仅去掉静态前缀,未重写任何文字。"""
    state = _canonical_state()
    v1_full = compose_action_prompt(_CANONICAL_TASK, state, "normalized_1000")
    v1_dynamic = v1_full[len(ACTION_SYSTEM_PROMPT) + 2 :]
    v3_user = compose_action_prompt_v3(_CANONICAL_TASK, state, "normalized_1000")
    assert v1_full == f"{ACTION_SYSTEM_PROMPT}\n\n{v1_dynamic}"
    assert v3_user == v1_dynamic
    assert v3_user.startswith("Current execution state:\n")
    assert _sha256(v1_dynamic) == _V1_DYNAMIC_SHA256
    assert _sha256(v3_user) == _V1_DYNAMIC_SHA256
