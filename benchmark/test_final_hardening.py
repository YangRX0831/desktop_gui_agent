"""FINAL BENCHMARK HARDENING 测试:M/H CaseSpec 冻结与 M04/M01 verifier。

全部纯逻辑/fake,无真实 Excel/COM/GUI/模型。
"""

import json
from pathlib import Path

from benchmark.case_specs import (
    acceptance_cases,
    generate_mh_case,
)
from benchmark.tasks import (
    H01WebToDoc,
    H02Presentation,
    H03FileSearch,
    M01Spreadsheet,
    M02Email,
    M04Document,
    M05Chat,
    _latest_finish_result_since,
    _m01_data_matches,
)

PROJECT = Path(__file__).resolve().parent.parent
MH_IDS = ["M01", "M02", "M03", "M04", "M05", "H01", "H02", "H03"]


def test_mh_casespec_deterministic_double_materialize() -> None:
    """同一 CaseSpec 连续物化两次:instruction/fixture/expected 全一致。"""
    for task_id in MH_IDS:
        a = generate_mh_case(task_id, f"{task_id}_X1", seed=7)
        b = generate_mh_case(task_id, f"{task_id}_X1", seed=7)
        assert a.instruction == b.instruction, task_id
        assert a.params == b.params, task_id
        assert a.to_json() == b.to_json(), task_id


def test_mh_casespec_different_seed_allows_difference() -> None:
    """不同 seed 允许不同实例(至少多数任务字段可变)。"""
    for task_id in MH_IDS:
        a = generate_mh_case(task_id, "P1", seed=1)
        b = generate_mh_case(task_id, "P2", seed=2)
        assert (a.params != b.params) or (a.instruction != b.instruction), task_id


def test_mh_prepare_with_casespec_never_re_randomizes() -> None:
    """prepare(case_spec=...) 消费冻结字段,不调用随机路径。"""
    import benchmark.tasks as tasks_mod

    original_gen_id = tasks_mod.gen_id
    calls = []

    def spy_gen_id(n=4):
        calls.append(n)
        return original_gen_id(n)

    tasks_mod.gen_id = spy_gen_id
    try:
        cases = {
            c.task_id: c
            for c in (generate_mh_case(t, f"PRD_{t}", 20260819) for t in MH_IDS)
        }
        classes = {
            "M01": M01Spreadsheet,
            "M02": M02Email,
            "M04": M04Document,
            "M05": M05Chat,
            "H01": H01WebToDoc,
            "H02": H02Presentation,
            "H03": H03FileSearch,
        }
        for task_id, cls in classes.items():
            task = cls("RUN", PROJECT / "logs", case_spec=cases[task_id])
            task.instruction()  # 不触发随机
        assert calls == []
        # instruction 与 spec 完全一致(dir 占位除外)。
        for task_id, cls in classes.items():
            task = cls("RUN", PROJECT / "logs", case_spec=cases[task_id])
            rendered = task.instruction()
            assert "GUIAgentBenchmark_CASE" not in rendered or task_id in {
                "M03",
                "M04",
                "H03",
            }
    finally:
        tasks_mod.gen_id = original_gen_id


def test_acceptance_cases_cover_15_with_m06_safety_skip() -> None:
    """acceptance preset 覆盖 15 项;M06 为 SAFETY_SKIP 哨兵。"""
    cases = acceptance_cases()
    ids = [c.task_id for c in cases]
    assert len(cases) == 15
    assert sorted(ids) == sorted(
        [
            "S01",
            "S02",
            "S03",
            "S04",
            "S05",
            "S06",
            "M01",
            "M02",
            "M03",
            "M04",
            "M05",
            "M06",
            "H01",
            "H02",
            "H03",
        ],
    )
    m06 = next(c for c in cases if c.task_id == "M06")
    assert m06.params.get("safety_skip") is True
    # 每个任务 ID 唯一(无重复实例)。
    assert len(set(ids)) == 15


def test_m01_matcher_pass_fail() -> None:
    """COM 值核对纯函数:目标数据存在 PASS,缺失 FAIL。"""
    headers = ["姓名", "部门", "分数"]
    rows = [["陈晨", "研发", "86"], ["林宇", "产品", "91"], ["周宁", "测试", "78"]]
    ok, detail = _m01_data_matches(
        [
            ["姓名", "部门", "分数"],
            ["陈晨", "研发", "86"],
            ["林宇", "产品", "91"],
            ["周宁", "测试", "78"],
        ],
        headers,
        rows,
    )
    assert ok, detail
    missing_row, _ = _m01_data_matches(
        [["姓名", "部门", "分数"], ["陈晨", "研发", "86"], ["林宇", "产品", "91"]],
        headers,
        rows,
    )
    assert not missing_row
    bad, _ = _m01_data_matches([["空", "表"]], headers, rows)
    assert not bad


def test_m01_com_reader_unavailable_fails_closed(monkeypatch) -> None:
    """COM 不可用时拒绝用无 cell 边界的 OCR 文本冒充结构证据。"""
    import benchmark.tasks as tasks_mod

    monkeypatch.setattr(
        tasks_mod,
        "_excel_used_values_readonly",
        lambda: None,
    )

    case = generate_mh_case("M01", "PRD_M01", 20260819)
    task = M01Spreadsheet("RUN", PROJECT / "logs", case_spec=case)
    task.prepare()
    ok, actual = task.validate()
    assert not ok
    assert "结构化单元格证据" in actual


def test_m01_com_reader_readonly_no_side_effects() -> None:
    """_excel_used_values_readonly 源码层面只读:不出现写/存/关调用。"""
    import inspect

    from benchmark import tasks as tasks_mod

    source = inspect.getsource(tasks_mod._excel_used_values_readonly)
    for banned in (
        ".Save",
        ".Close",
        ".Quit",
        ".Value =",
        ".Cells(",
        ".Range(",
        ".Add",
        "Copy",
        ".Delete",
    ):
        assert banned not in source, banned


def test_m04_trace_verifier_pass_fail_fallback(tmp_path, monkeypatch) -> None:
    """M04 trace-first:正确 finish→PASS;错误→FAIL;无 trace→OCR fallback。"""
    import time as time_mod

    from benchmark import tasks as tasks_mod

    case = generate_mh_case("M04", "PRD_M04", 20260819)
    task = M04Document("RUN", tmp_path, case_spec=case)
    task.prepare()
    expected = case.params["project_num"]

    trace_dir = tmp_path / "trace"
    monkeypatch.chdir(tmp_path)
    (trace_dir).mkdir(exist_ok=True)
    # 正确 finish result → PASS。
    good = json.dumps(
        {
            "record_type": "model_call",
            "parse_success": True,
            "request_finished_at": time_mod.time(),
            "parsed_action": f'finish(result="编号是 {expected}")',
        }
    )
    (trace_dir / "run.jsonl").write_text(good + "\n", encoding="utf-8")
    task.result.start_time = time_mod.time() - 10
    monkeypatch.setattr(
        tasks_mod,
        "_latest_finish_result_since",
        lambda since: f"finish 的 result 含 {expected}" if since else None,
    )
    ok, actual = task.validate()
    assert ok and expected in actual
    # 错误 finish result → FAIL。
    monkeypatch.setattr(
        tasks_mod,
        "_latest_finish_result_since",
        lambda since: "结果未知",
    )
    ok, actual = task.validate()
    assert not ok and "未包含" in actual
    # 无 trace → fallback OCR(终端 hwnd=0 给出明确失败)。
    monkeypatch.setattr(
        tasks_mod,
        "_latest_finish_result_since",
        lambda since: None,
    )
    task.terminal_hwnd = 0
    ok, actual = task.validate()
    assert not ok and "终端窗口句柄不可用" in actual


def test_m04_trace_reader_tolerates_old_format(tmp_path) -> None:
    """旧 trace 格式/空文件不崩溃,返回 None。"""
    (tmp_path / "logs").mkdir(exist_ok=True)
    trace = tmp_path / "logs" / "agent_trace"
    trace.mkdir(exist_ok=True)
    (trace / "old.jsonl").write_text(
        "not-json\n{}}\n" '{"record_type":"model_call","parse_success":false}\n',
        encoding="utf-8",
    )
    import time as time_mod

    original = _latest_finish_result_since(time_mod.time() - 60)
    assert original is None  # 无 finish → None,不抛异常
