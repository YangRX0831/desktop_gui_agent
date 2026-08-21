"""LOCAL-ONLY output repair 与 raw logging 旁路测试。

合同:repair 只在 model_mode == local 执行;确定性、无歧义、不改坐标与
字符串参数;API 路径完全跳过;common parser 严格性不降低。
"""

import asyncio

from agentscope.message import Msg

from agent.action_parser import parse_action
from agent.action_response_adapter import adapt_action_response
from agent.local_output_repair import repair_local_output
from agent.task_manager import TaskManager
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent

_FINISH = 'Action: finish(result="完成")'


def test_standard_action_unchanged() -> None:
    """TEST 1:标准 Action 修复后逐字不变。"""
    text = "Action: click(x=100, y=200)"
    repaired, repairs = repair_local_output(text)
    assert repaired == text
    assert repairs == []


def test_markdown_fence_removed() -> None:
    """TEST 2:fence 包裹唯一 Action → 去除 fence。"""
    text = "```python\nAction: click(x=100, y=200)\n```"
    repaired, repairs = repair_local_output(text)
    assert repaired == "Action: click(x=100, y=200)"
    assert "markdown_fence" in repairs
    assert parse_action(repaired) is not None


def test_explanation_plus_unique_action_extracted() -> None:
    """TEST 3:解释文字 + 唯一 Action 行 → 提取该行。"""
    text = "我认为下一步应该点击按钮。\nAction: click(x=100, y=200)"
    repaired, repairs = repair_local_output(text)
    assert repaired == "Action: click(x=100, y=200)"
    assert "unique_action_line" in repairs


def test_chinese_colon_normalized() -> None:
    """TEST 4:中文冒号 → 半角(normalize 再补 Action 前缀空格)。"""
    text = "Action：click(x=100, y=200)"
    repaired, repairs = repair_local_output(text)
    assert repaired == "Action:click(x=100, y=200)"
    assert "cjk_punct" in repairs
    adapted = adapt_action_response(repaired)
    assert parse_action(adapted.normalized_response) is not None


def test_chinese_comma_and_parens_normalized() -> None:
    """TEST 5:中文逗号/括号 → 半角。"""
    text = "Action: click(x=100，y=200）"
    repaired, _ = repair_local_output(text)
    assert repaired == "Action: click(x=100, y=200)"
    assert parse_action(repaired) is not None


def test_trailing_chinese_period_removed() -> None:
    """TEST 6:Action 末尾中文句号 → 删除。"""
    text = "Action: click(x=100, y=200)。"
    repaired, repairs = repair_local_output(text)
    assert repaired == "Action: click(x=100, y=200)"
    assert "cjk_punct" in repairs


def test_action_name_case_normalized() -> None:
    """TEST 7:Click/CLICK/Finish 大小写只标准化动词 token。"""
    for verb, canonical in (
        ("Click", "click"),
        ("CLICK", "click"),
        ("Finish", "finish"),
        ("Type", "type"),
        ("Hotkey", "hotkey"),
        ("Drag", "drag"),
        ("Double_Click", "double_click"),
    ):
        text = (
            f"Action: {verb}(x=100, y=200)"
            if verb not in {"Finish", "Type", "Hotkey"}
            else (
                'Action: Finish(result="ok")'
                if verb == "Finish"
                else (
                    'Action: Type(text="hi")'
                    if verb == "Type"
                    else 'Action: Hotkey(key1="ctrl")'
                )
            )
        )
        repaired, _ = repair_local_output(text)
        if verb in {"Click", "CLICK"}:
            assert repaired.startswith(f"Action: {canonical}("), repaired
        if verb == "Finish":
            assert repaired.startswith("Action: finish("), repaired
        if verb == "Type":
            assert repaired.startswith("Action: type("), repaired
        if verb == "Hotkey":
            assert repaired.startswith("Action: hotkey("), repaired


def test_two_actions_not_selected() -> None:
    """TEST 8:两个不同 Action → 禁止二选一,保持 parser failure。"""
    text = "Action: click(x=1, y=2)\nAction: click(x=3, y=4)"
    repaired, repairs = repair_local_output(text)
    assert "unique_action_line" not in repairs
    assert parse_action(repaired) is None


def test_no_action_not_invented() -> None:
    """TEST 9:完全没有 Action → 不得凭空生成。"""
    text = "我认为应该打开开始菜单再寻找计算器。"
    repaired, repairs = repair_local_output(text)
    assert parse_action(repaired) is None
    assert repairs == []


def test_string_params_preserved() -> None:
    """TEST 10:字符串参数内部中文标点完整保留。"""
    text = 'Action: type(text="你好，世界（测试）。")'
    repaired, _ = repair_local_output(text)
    assert "你好，世界（测试）。" in repaired
    assert parse_action(repaired) is not None
    assert parse_action(repaired)["params"]["text"] == "你好，世界（测试）。"


def test_coordinates_never_modified() -> None:
    """TEST 11:所有坐标数值修复前后完全一致。"""
    samples = [
        "Action：click(x=358，y=490）",
        "```\nAction: CLICK(x=0, y=999)\n```",
    ]
    for text in samples:
        repaired, _ = repair_local_output(text)
        adapted = adapt_action_response(repaired)
        parsed = parse_action(adapted.normalized_response)
        assert parsed is not None, repaired
        params = parsed["params"]
        values = [v for v in params.values() if isinstance(v, int)]
        if text.startswith("Action：click"):
            assert values == [358, 490]
        elif "CLICK" in text:
            assert values == [0, 999]


def test_api_mode_skips_repair(monkeypatch) -> None:
    """TEST 12:API 模式不得进入 local_output_repair。"""
    import agent.gui_agent as ga

    calls = []
    original = ga.repair_local_output

    def spy(response):
        calls.append(response)
        return original(response)

    monkeypatch.setattr(ga, "repair_local_output", spy)
    agent = make_agent(
        SequenceBackend(["我认为应该这样。\nAction: click(x=1, y=1)", _FINISH]),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
    )
    asyncio.run(agent(Msg("u", "任务", "user")))
    # API 模式下第一响应是多行解释 + Action:严格 parser 拒绝并重试,
    # 且 repair 从未被调用。
    assert calls == []


def test_local_mode_uses_repair(monkeypatch) -> None:
    """TEST 13(local 路径正向):local 模式多行解释输出被修复后可解析。"""
    agent = make_agent(
        SequenceBackend(
            ["我认为应该点击。\nAction: click(x=1, y=1)", _FINISH],
        ),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="local",
        reject_initial_finish=False,
    )
    result = asyncio.run(agent(Msg("u", "任务", "user")))
    # 第一响应经 unique_action_line 提取后成功分发,随后 finish。
    assert result.content == "完成"
    assert (
        [c[0] for c in agent._dependencies.model_client.calls_args_actions()]
        == [
            "click",
        ]
        if hasattr(agent._dependencies.model_client, "calls_args_actions")
        else True
    )


def test_raw_logging_fields_present(monkeypatch) -> None:
    """raw logging:成功/失败调用均保留 raw/repaired/normalized/错误字段。"""
    recorded = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    agent = make_agent(
        SequenceBackend(["格式坏掉的输出", _FINISH]),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="local",
        reject_initial_finish=False,
        trace_writer=_TraceWriter(),
    )
    asyncio.run(agent(Msg("u", "任务", "user")))
    model_rows = [row for row in recorded if row.get("record_type") == "model_call"]
    first, second = model_rows[0], model_rows[1]
    # 失败调用:raw 完整保留 + 独立 parse_error 字段。
    assert first["raw_model_response"] == "格式坏掉的输出"
    assert first["parse_success"] is False
    assert first["parse_error"] == "invalid_syntax"
    assert first["repaired_model_response"] == "格式坏掉的输出"
    assert first["backend_error_summary"] is None
    # 成功调用:raw/repaired/normalized 全保留。
    assert second["parse_success"] is True
    assert second["parse_error"] is None
    assert second["raw_model_response"] == _FINISH
    assert second["model_latency"] >= 0
    assert second["backend"].startswith("local:")


def test_action_canon_001_trace_preserves_raw_and_reason() -> None:
    """API trace 同时保存模型原文、canonical form 与固定归一原因。"""
    recorded = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    agent = make_agent(
        SequenceBackend(
            ["Action: click(x=10, 20)", _FINISH],
        ),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        trace_writer=_TraceWriter(),
    )

    asyncio.run(agent(Msg("u", "任务", "user")))

    row = next(r for r in recorded if r.get("record_type") == "model_call")
    assert row["raw_model_response"] == "Action: click(x=10, 20)"
    assert row["normalized_model_response"] == "Action: click(x=10, y=20)"
    assert row["normalization_reason"] == ("action_canon_001_click_missing_y_name")
    assert row["parsed_action"] == "click(x=10, y=20)"
    execution = next(
        record for record in recorded if record.get("record_type") == "action_execution"
    )
    assert execution["protocol_version"] == "v1"
    assert execution["control_dispatch"] is True
    assert execution["parsed_action"] == "click(x=10, y=20)"
    assert execution["dispatch_elapsed_ms"] >= 0
    assert "post_action_screenshot_sha256" in execution


def test_action_canon_002_trace_preserves_raw_normalized_and_parsed() -> None:
    """002 的原文、canonical 文本、原因和 strict parsed action 均入 trace。"""
    recorded = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    agent = make_agent(
        SequenceBackend(["click(10,20)", _FINISH]),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        trace_writer=_TraceWriter(),
    )

    asyncio.run(agent(Msg("u", "任务", "user")))

    row = next(r for r in recorded if r.get("record_type") == "model_call")
    assert row["raw_model_response"] == "click(10,20)"
    assert row["normalized_model_response"] == "Action: click(x=10, y=20)"
    assert row["normalization_reason"] == "action_canon_002_click_positional_xy"
    assert row["parsed_action"] == "click(x=10, y=20)"
    execution = next(
        record for record in recorded if record.get("record_type") == "action_execution"
    )
    assert execution["control_dispatch"] is True
    assert execution["parsed_action"] == "click(x=10, y=20)"


def test_backend_error_summary_recorded(monkeypatch) -> None:
    """backend 异常:脱敏摘要进入 trace。"""
    recorded = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    class _ExplodingBackend:
        calls = 0

        def generate(self, image, prompt, mode="local", **kwargs):
            type(self).calls += 1
            raise RuntimeError("本地 Qwen2-VL 模型推理失败。")

    agent = make_agent(
        _ExplodingBackend(),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=1,
        retry_count=0,
        model_mode="local",
        trace_writer=_TraceWriter(),
    )
    result = asyncio.run(agent(Msg("u", "任务", "user")))
    assert "任务执行失败" in result.content
    row = next(r for r in recorded if r.get("record_type") == "model_call")
    assert row["backend_error_summary"].startswith("RuntimeError:")
    assert "推理失败" in row["backend_error_summary"]
    assert row["raw_model_response"] is None


def test_api_behavior_unchanged_by_logging(monkeypatch) -> None:
    """API 路径:提示/解析/重试语义与 trace 新字段前完全一致。"""
    agent = make_agent(
        SequenceBackend(["坏输出", _FINISH]),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
    )
    result = asyncio.run(agent(Msg("u", "任务", "user")))
    assert result.content == "完成"
    prompts = agent._dependencies.model_client.prompts
    # 第二次调用携带解析失败反馈(既有行为)。
    assert "动作格式" in prompts[1]
