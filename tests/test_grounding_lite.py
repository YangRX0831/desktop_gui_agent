"""PHASE 2B STRUCTURED GROUNDING LITE 测试:候选构建、渲染与集成。

坐标体系与模型动作一致(截图内 0..1000);Agent 自身窗口绝不成为候选;
feature 关闭时动态 Prompt 不出现 Interactive elements 块(字节不变性由
test_agent_v3_payload 的哈希钉死测试保证)。
"""

import asyncio

from agentscope.message import Msg

from agent.action_parser import compose_action_dynamic_prompt
from perception.prompt_context import (
    build_grounding_candidates,
    render_interactive_elements,
)
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent

_OCR_LINES = (
    'text="计算器" bbox=(100, 200, 300, 250) confidence=0.97',
    'text="很长的界面说明文字标题超过十二个字符" bbox=(10, 20, 30, 40) confidence=0.99',
    'text="取消" bbox=(500, 600, 560, 640) confidence=0.85',
    'text="确定" bbox=(700, 600, 760, 640) confidence=0.95',
)
_WINDOW_LINES = (
    "id:101 fg=true process=notepad.exe bbox=(80, 60, 700, 500)",
    "id:4242 fg=false process=windowsterminal.exe bbox=(100, 600, 900, 950)",
    "id:303 fg=false process=explorer.exe bbox=(0, 900, 1000, 1000)",
)


def test_grounding_candidates_exclude_agent_window_and_low_confidence() -> None:
    """Agent 自身窗口被排除;低置信与超长 OCR 文本被过滤。"""
    candidates = build_grounding_candidates(_OCR_LINES, _WINDOW_LINES, 4242)
    names = [c["name"] for c in candidates if c["source"] == "window"]
    assert "windowsterminal.exe" not in names
    assert names[0] == "notepad.exe"  # 前台窗口优先
    texts = [c["text"] for c in candidates if c["source"] == "ocr"]
    assert "计算器" in texts
    assert "确定" in texts
    assert "取消" not in texts  # confidence 0.85 < 0.9
    assert all(len(t) <= 12 for t in texts)


def test_grounding_candidate_caps() -> None:
    """窗口 ≤5、OCR ≤15,总量有界。"""
    ocr_many = tuple(
        f'text="t{i}" bbox=({i}, {i}, {i + 5}, {i + 5}) confidence=0.99'
        for i in range(40)
    )
    windows_many = tuple(
        f"id:{i} fg=false process=app{i}.exe bbox=(0, 0, 100, 100)" for i in range(10)
    )
    candidates = build_grounding_candidates(ocr_many, windows_many, 0)
    assert sum(1 for c in candidates if c["source"] == "window") == 5
    assert sum(1 for c in candidates if c["source"] == "ocr") == 15
    assert len(candidates) <= 20


def test_grounding_render_format_and_bbox_domain() -> None:
    """渲染行格式紧凑;bbox 值域 0..1000,与动作坐标体系一致。"""
    candidates = build_grounding_candidates(_OCR_LINES, _WINDOW_LINES, 4242)
    lines = render_interactive_elements(candidates)
    assert lines[0].startswith("E1 source=window role=window name=notepad.exe")
    assert any('text="计算器"' in line for line in lines)
    for candidate in candidates:
        assert all(0 <= v <= 1000 for v in candidate["bbox"])


def test_interactive_elements_block_rendering_rules() -> None:
    """非空候选渲染独立块;空候选输出无块(与此前逐字节一致)。"""
    from dataclasses import replace

    from agent.action_parser import ActionPromptState

    state = ActionPromptState(
        step_number=1,
        max_steps=10,
        interactive_elements=(
            'E1 source=ocr role=text text="确定" bbox=(700, 600, 760, 640)',
        ),
    )
    out = compose_action_dynamic_prompt("任务", state, "normalized_1000")
    assert "Interactive elements:\n  E1 source=ocr" in out
    plain = replace(state, interactive_elements=())
    out_plain = compose_action_dynamic_prompt("任务", plain, "normalized_1000")
    assert "Interactive elements" not in out_plain


def test_semantic_agent_prompt_contains_elements(monkeypatch) -> None:
    """semantic 开启时真实 Agent 首轮 Prompt 含 Interactive elements 块。"""
    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_ocr_elements",
        lambda recognizer, image, focus_point=None: _OCR_LINES,
        raising=True,
    )
    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_windows",
        lambda size, offset: _WINDOW_LINES,
        raising=True,
    )
    agent = make_agent(
        SequenceBackend(['Action: finish(result="done")']),
        MemoryControls(),
        make_task_manager(),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
    )
    # 使用不触发 app-launch route 的任务文本(含"新建"排除词)。
    asyncio.run(agent(Msg("u", "打开记事本并新建文档输入内容", "user")))
    prompt = agent._dependencies.model_client.prompts[0]
    assert "Interactive elements:" in prompt
    assert "E1 source=window role=window name=notepad.exe" in prompt


def test_local_symbolic_grounding_is_current_turn_scoped_and_traced(
    monkeypatch,
) -> None:
    """Local E1 经当轮候选解析、严格 parser、trace 后再分发。"""
    recorded = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_ocr_elements",
        lambda recognizer, image, focus_point=None: _OCR_LINES,
        raising=True,
    )
    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_windows",
        lambda size, offset: _WINDOW_LINES,
        raising=True,
    )
    controls = MemoryControls()
    backend = SequenceBackend(
        ["click(E1)", 'Action: finish(result="done")'],
    )
    agent = make_agent(
        backend,
        controls,
        make_task_manager(),
        max_steps=2,
        retry_count=0,
        model_mode="local",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
        trace_writer=_TraceWriter(),  # type: ignore[arg-type]
    )

    asyncio.run(agent(Msg("u", "完成当前界面任务", "user")))

    assert "当轮交互候选:" in backend.prompts[0]
    assert "E1 source=window role=window name=notepad.exe" in backend.prompts[0]
    assert controls.calls[0] == ("click", 390, 280, "left")
    row = next(record for record in recorded if record["record_type"] == "model_call")
    assert row["raw_model_response"] == "click(E1)"
    assert row["normalized_model_response"] == "Action: click(x=390, y=280)"
    assert row["normalization_reason"] == ("action_adapt_001_symbolic_grounding_click")
    assert row["parse_success"] is True


def make_task_manager():
    from agent.task_manager import TaskManager

    return TaskManager("打开记事本")
