"""SEMANTIC EXECUTION PHASE 2A 测试:三态完成验证、进度速率与 finish 分支。

finish 分支测试通过 monkeypatch 程序事实源(音量/OCR/前台)驱动真实
GuiAgent.finish 处理路径;静态 Prompt/动态 Prompt 的字节不变性由
test_agent_v3_payload 的哈希钉死测试保证(Phase 2A 字段全 None 时不渲染)。
"""

import asyncio

import pytest
from agentscope.message import Msg

from agent import gui_agent as gui_agent_module
from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3
from agent.task_expectation import (
    ProgressTracker,
    TaskExpectation,
    extract_task_expectation,
    verify_completion,
)
from agent.task_manager import TaskManager
from tests.agent_test_support import (
    MemoryControls,
    SequenceBackend,
    make_agent,
    make_semantic_agent,
)

_FINISH = 'Action: finish(result="完成")'


def _patch_facts(monkeypatch, volume=None, ocr="", foreground="explorer.exe"):
    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.system_volume_state",
        lambda: volume,
        raising=True,
    )
    monkeypatch.setattr(
        "agent.gui_agent.prompt_context.perceive_ocr_elements",
        lambda recognizer, image, focus_point=None: (
            tuple(ocr.split("|")) if ocr else ()
        ),
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "get_foreground_app_hwnd",
        lambda: 0 if foreground == "desktop" else 777,
        raising=True,
    )
    monkeypatch.setattr(
        gui_agent_module,
        "get_window_process_name",
        lambda hwnd: foreground,
        raising=True,
    )


def test_finish_verified_accepted(monkeypatch) -> None:
    """S03 类任务:音量已达容差 → VERIFIED → finish 接受。"""
    agent = make_semantic_agent([_FINISH])
    agent._run_expectation = TaskExpectation(expected_volume=20)
    _patch_facts(monkeypatch, volume=22, foreground="explorer.exe")
    result = asyncio.run(agent(Msg("u", "将音量调整到大约20%", "user")))
    assert result.content == "完成"


def test_finish_not_verified_rejected(monkeypatch) -> None:
    """S03 类任务:音量未达标 → NOT_VERIFIED → finish 拒绝并继续。"""
    agent = make_semantic_agent(
        [_FINISH, _FINISH],
        reject_initial_finish=False,
        max_steps=2,
    )
    agent._run_expectation = TaskExpectation(expected_volume=20)
    _patch_facts(monkeypatch, volume=50, foreground="explorer.exe")
    result = asyncio.run(agent(Msg("u", "将音量调整到大约20%", "user")))
    # 两次 finish 均被拒,步数耗尽 → 任务失败,而非宣称成功。
    assert "任务执行失败" in result.content
    # 拒绝原因进入下一轮动态状态(§19)。
    prompts = agent._dependencies.model_client.prompts
    assert "completion_verification=NOT_VERIFIED" in prompts[1]
    assert "current_volume=50" in prompts[1]


def test_finish_unknown_keeps_legacy_behavior(monkeypatch) -> None:
    """无结构化期望 → UNKNOWN → 保持既有 finish 行为(接受)。"""
    agent = make_semantic_agent([_FINISH])
    agent._run_expectation = TaskExpectation()
    _patch_facts(monkeypatch, volume=None, ocr="", foreground="explorer.exe")
    result = asyncio.run(agent(Msg("u", "随便一个任务", "user")))
    assert result.content == "完成"


def test_volume_verifier_three_states() -> None:
    """S03 结构化音量验证:容差内 VERIFIED,未达 NOT_VERIFIED,无读数 UNKNOWN。"""
    exp = TaskExpectation(expected_volume=60)
    assert verify_completion(exp, {"current_volume": 64}).status == "VERIFIED"
    assert verify_completion(exp, {"current_volume": 40}).status == "NOT_VERIFIED"
    unknown = verify_completion(exp, {"current_volume": None})
    assert unknown.status == "UNKNOWN"


def test_window_exists_verifier() -> None:
    """窗口关闭验证:仍存在 NOT_VERIFIED,已关闭 VERIFIED,无跟踪 UNKNOWN。"""
    exp = TaskExpectation(expected_window_closed=123)
    assert (
        verify_completion(exp, {"tracked_window_exists": True}).status == "NOT_VERIFIED"
    )
    assert verify_completion(exp, {"tracked_window_exists": False}).status == "VERIFIED"
    assert verify_completion(exp, {"tracked_window_exists": None}).status == "UNKNOWN"


def test_text_evidence_verifier() -> None:
    """文本证据验证:OCR 命中 VERIFIED;前台是搜索叠层 NOT_VERIFIED。"""
    exp = TaskExpectation(expected_text="晨星Agent952")
    facts_hit = {
        "ocr_text": 'text="晨星Agent952" bbox=(1,2,3,4)',
        "foreground_process": "Notepad.exe",
    }
    assert verify_completion(exp, facts_hit).status == "VERIFIED"
    facts_overlay = {
        "ocr_text": "",
        "foreground_process": "SearchHost.exe",
    }
    assert verify_completion(exp, facts_overlay).status == "NOT_VERIFIED"
    facts_miss = {"ocr_text": "别的文字", "foreground_process": "Notepad.exe"}
    assert verify_completion(exp, facts_miss).status == "UNKNOWN"


def test_numeric_result_verifier() -> None:
    """数值结果验证:OCR 含结果 VERIFIED;Start/Search 前台 NOT_VERIFIED。"""
    exp = TaskExpectation(expected_numeric_result=42)
    assert (
        verify_completion(
            exp,
            {"ocr_text": "6×7 = 42", "foreground_process": "CalculatorApp.exe"},
        ).status
        == "VERIFIED"
    )
    false_finish = {
        "ocr_text": "计算器 开始 搜索",
        "foreground_process": "SearchHost.exe",
    }
    assert verify_completion(exp, false_finish).status == "NOT_VERIFIED"
    miss = {"ocr_text": "计算器", "foreground_process": "CalculatorApp.exe"}
    assert verify_completion(exp, miss).status == "UNKNOWN"


def test_hosted_app_title_match_can_verify_expected_foreground() -> None:
    """UWP 宿主进程仅在标题别名命中时视为目标应用。"""
    exp = TaskExpectation(
        expected_app="Calculator",
        expected_app_processes=("calculatorapp.exe",),
        expected_app_title_keywords=("Calculator", "计算器"),
    )
    hosted = {
        "foreground_process": "ApplicationFrameHost.exe",
        "foreground_title_matches_expected": True,
    }
    assert verify_completion(exp, hosted).status == "VERIFIED"
    unrelated = {
        "foreground_process": "ApplicationFrameHost.exe",
        "foreground_title_matches_expected": False,
    }
    assert verify_completion(exp, unrelated).status == "NOT_VERIFIED"


def test_steps_remaining_rendered_and_correct(monkeypatch) -> None:
    """steps_remaining = max_steps - current_step,并进入动态状态。"""
    agent = make_semantic_agent([_FINISH])
    agent._run_expectation = TaskExpectation()
    _patch_facts(monkeypatch, volume=None, foreground="explorer.exe")
    asyncio.run(agent(Msg("u", "任务", "user")))
    prompt = agent._dependencies.model_client.prompts[0]
    assert "- steps_remaining=2" in prompt


def test_progress_tracker_detects_progressed() -> None:
    """速率充足时识别 PROGRESSED。"""
    tracker = ProgressTracker(80.0)
    tracker.record("hotkey", 50, 52)
    tracker.record("hotkey", 52, 54)
    status, _ = tracker.evaluate(54, 20)
    assert status == "PROGRESSED"


def test_progress_tracker_detects_insufficient_rate() -> None:
    """§16 示例:current=42 target=20 每 action≈-2 剩 5 步 → 不足。"""
    tracker = ProgressTracker(20.0)
    tracker.record("hotkey", 44, 42)
    status, reason = tracker.evaluate(42, 5)
    assert status == "INSUFFICIENT_RATE"
    assert "steps_remaining=5" in reason
    assert "required_delta=22" in reason


def test_progress_tracker_unknown_without_quantitative_fact() -> None:
    """无量化事实(无目标/无观察)时 UNKNOWN。"""
    tracker = ProgressTracker(None)
    assert tracker.evaluate(42, 5)[0] == "UNKNOWN"
    fresh = ProgressTracker(20.0)
    assert fresh.evaluate(42, 5)[0] == "UNKNOWN"


def test_progress_tracker_no_progress() -> None:
    """连续动作零推进 → NO_PROGRESS。"""
    tracker = ProgressTracker(20.0)
    tracker.record("click", 50, 50)
    tracker.record("click", 50, 50)
    status, reason = tracker.evaluate(50, 5)
    assert status == "NO_PROGRESS"
    assert "not moving" in reason


def test_progress_tracker_latest_improvement_survives_prior_oscillation() -> None:
    """最新距离缩短时不得被此前振荡错误覆盖为 NO_PROGRESS。"""
    tracker = ProgressTracker(60.0)
    tracker.record("click", 50, 70)
    tracker.record("click", 70, 34)
    tracker.record("click", 34, 47)
    status, reason = tracker.evaluate(47, 6)
    assert status == "PROGRESSED"
    assert "current=47" in reason


def test_progress_tracker_latest_stall_is_no_progress() -> None:
    """历史曾推进但最新动作停滞时仍须明确报告 NO_PROGRESS。"""
    tracker = ProgressTracker(60.0)
    tracker.record("click", 20, 47)
    tracker.record("click", 47, 47)
    status, reason = tracker.evaluate(47, 6)
    assert status == "NO_PROGRESS"
    assert "latest action" in reason


def test_narrow_expectation_extractor() -> None:
    """窄模式抽取:音量/算式保留;输入标识抽取已按 PRD-BC-001 移除。"""
    volume = extract_task_expectation("将系统输出音量调整到大约65%。")
    assert volume.expected_volume == 65
    calc = extract_task_expectation("打开系统计算器，计算23+48，并让结果保留。")
    assert calc.expected_numeric_result == 71
    assert calc.expected_app == "Calculator"
    assert "计算器" in calc.expected_app_title_keywords
    text = extract_task_expectation(
        "打开记事本并输入以下内容：桌面 GUI 智能体测试 晨星Agent952。",
    )
    assert text.expected_text is None
    assert text.expected_volume is None
    empty = extract_task_expectation("帮我整理桌面文件")
    assert empty.is_empty()


def test_removed_marker_extraction_never_verifies_text() -> None:
    """PRD-BC-001 移除后:输入标识不再生成期望,finish 文本判据不可达。

    即便 OCR 中出现该文本,verify_completion 也不得凭空 VERIFIED。
    """
    expectation = extract_task_expectation(
        "打开记事本并输入以下内容：桌面 GUI 智能体测试 晨星Agent952。",
    )
    facts = {
        "ocr_text": "桌面 GUI 智能体测试 晨星Agent952",
        "foreground_process": "notepad.exe",
    }
    verdict = verify_completion(expectation, facts)
    assert verdict.status == "UNKNOWN"


def test_general_expectation_families_independent_of_input_marker() -> None:
    """通用期望族(保存文件名/目录/投递)不依赖输入标识抽取。

    PRD-BC-001 特性锁:即使移除输入标识(marker)特殊抽取,基于独立
    产品语义的期望族必须继续工作。
    """
    save = extract_task_expectation("把页面里的图片另存为 photo.png。")
    assert save.save_image_intent is True
    assert save.expected_save_filename == "photo.png"
    folder = extract_task_expectation('把图片保存到桌面的"Reports"文件夹中')
    assert folder.expected_save_folder == "Desktop\\Reports"
    delivery = extract_task_expectation('给 user1 发送"任务完成"')
    assert delivery.delivery_intent is True
    assert delivery.expected_delivery_payload == "任务完成"


def test_benchmark_s02_validator_independent_of_marker_rule() -> None:
    """PRD-BC-001 特性锁:S02 验证器使用自带标识,不依赖生产 marker 抽取。"""
    import inspect

    from benchmark.tasks import S02TextInput

    source = inspect.getsource(S02TextInput)
    assert "task_expectation" not in source
    assert "_name_marker" in source
    assert "_number_marker" in source


def test_v3_static_prompt_matches_stable_contract() -> None:
    """V3 静态 Prompt 恢复稳定七段结构并保留八动作合同。"""
    from agent.action_parser import ACTION_SYSTEM_PROMPT
    from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3 as current

    assert current == ACTION_SYSTEM_PROMPT_V3
    assert 1000 < len(current) < len(ACTION_SYSTEM_PROMPT) * 0.75
    assert "键盘动作语义" in current
    assert "规划与状态" not in current
    assert "right_click" in current
    assert "observe" not in current


def test_semantic_trace_fields_present(monkeypatch) -> None:
    """trace 新字段进入记录;flag 关闭时不出现新动态行。"""
    recorded: list[dict] = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    backend = SequenceBackend([_FINISH])
    agent = make_agent(
        backend,
        MemoryControls(),
        TaskManager("任务"),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
        trace_writer=_TraceWriter(),
    )
    agent._run_expectation = TaskExpectation()
    _patch_facts(monkeypatch, volume=None, foreground="explorer.exe")
    asyncio.run(agent(Msg("u", "任务", "user")))
    assert recorded
    row = next(r for r in recorded if r.get("record_type") == "model_call")
    assert row["finish_proposed"] is True
    assert row["completion_verification"] is None
    assert row["steps_remaining"] == 0
    assert row["structured_facts"]["system_volume_percent"] is None
    # 接受路径的验证结论通过独立的 finish_decision 记录进入 trace。
    decision = next(r for r in recorded if r.get("record_type") == "finish_decision")
    assert decision["accepted"] is True
    assert decision["completion_verification"] == "UNKNOWN"
    # flag off 的对照:动态 Prompt 无任何 Phase 2A 行。
    plain = make_agent(
        SequenceBackend([_FINISH]),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=1,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
    )
    _patch_facts(monkeypatch, volume=None, foreground="explorer.exe")
    asyncio.run(plain(Msg("u", "任务", "user")))
    assert "steps_remaining" not in plain._dependencies.model_client.prompts[0]
    assert "completion_verification" not in (
        plain._dependencies.model_client.prompts[0]
    )


def test_finish_decision_trace_records_rejection(monkeypatch) -> None:
    """NOT_VERIFIED 拒绝路径同样写入 finish_decision 记录。"""
    recorded: list[dict] = []

    class _TraceWriter:
        def record_model_call(self, record):
            recorded.append(record)

    agent = make_agent(
        SequenceBackend([_FINISH, _FINISH]),
        MemoryControls(),
        TaskManager("任务"),
        max_steps=2,
        retry_count=0,
        model_mode="api",
        reject_initial_finish=False,
        decision_protocol_v3=True,
        semantic_execution=True,
        trace_writer=_TraceWriter(),
    )
    agent._run_expectation = TaskExpectation(expected_volume=20)
    _patch_facts(monkeypatch, volume=50, foreground="explorer.exe")
    asyncio.run(agent(Msg("u", "将音量调整到大约20%", "user")))
    decisions = [r for r in recorded if r.get("record_type") == "finish_decision"]
    assert len(decisions) == 2
    assert decisions[0]["accepted"] is False
    assert decisions[0]["completion_verification"] == "NOT_VERIFIED"
    assert "current_volume=50" in decisions[0]["completion_reason"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("3+4", 7),
        ("10-4", 6),
        ("6×7", 42),
        ("8÷2", 4),
        ("7÷2", None),
        ("5÷0", None),
        ("3++4", None),
        ("3\t+4", None),
    ],
)
def test_calc_expression_evaluation(expression: str, expected: int | None) -> None:
    """S01 算式求值:显式运算分支替代 eval 后的等价行为锁定。

    非整数商与除零按"无可验证期望"返回 None;不匹配/含制表符的
    输入不产生数值期望。
    """
    expectation = extract_task_expectation(f"打开计算器，计算{expression}")
    assert expectation.expected_numeric_result == expected
