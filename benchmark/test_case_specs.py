"""CaseSpec 配对正确性测试(benchmark 层;不触发 GUI/COM/网络副作用)。

S03 的 prepare 含真实音量 reset、S04 含窗口枚举快照,因此这两类任务在
单元测试中只验证 spec 字段与 instruction,不调用 prepare。
"""

import json
from pathlib import Path

from benchmark.case_specs import CaseSpec, fixed_pairs, generate_case, generate_mh_case
from benchmark.tasks import (
    H03FileSearch,
    S01Calculator,
    S02TextInput,
    S04OpenApp,
    S05Search,
    S06CloseWindow,
)

PROJECT = Path(__file__).resolve().parent.parent


def _make(task_cls, spec: CaseSpec):
    return task_cls(
        run_id=spec.pair_id,
        desktop_dir=PROJECT / "logs",
        case_spec=spec,
    )


def test_same_pair_spec_identical_for_both_arms() -> None:
    """同一 pair_id 的两个 arm 读取同一 spec,字段逐字一致。"""
    spec = generate_case("S01", "S01_P01", seed=7)
    arm_a = _make(S01Calculator, spec)
    arm_b = _make(S01Calculator, spec)
    assert arm_a.prepare() and arm_b.prepare()
    assert arm_a.expression == arm_b.expression
    assert arm_a.expected == arm_b.expected
    assert arm_a.instruction() == arm_b.instruction() == spec.instruction


def test_h03_files_live_in_instruction_named_directory(tmp_path: Path) -> None:
    """H03 CaseSpec 不得把文件藏进指令未声明的额外子目录。"""
    spec = generate_mh_case("H03", "H03_LAYOUT", seed=20260821)
    task = H03FileSearch("H03_LAYOUT", tmp_path, case_spec=spec)
    try:
        assert task.prepare()
        assert task.search_dir == tmp_path
        assert all((tmp_path / name).is_file() for name in spec.params["files"])
        assert spec.params["directory"] not in task.instruction()
    finally:
        task.cleanup()


def test_different_pair_id_allows_different_instance() -> None:
    """不同 pair_id 的固定实例互不相同;同 pair_id 生成结果确定。"""
    s01_specs = fixed_pairs("S01")
    expressions = {s.params["expression"] for s in s01_specs}
    assert len(expressions) == 3
    s03_targets = {s.params["target_volume"] for s in fixed_pairs("S03")}
    assert len(s03_targets) == 3
    assert (
        generate_case("S02", "S02_P01", seed=3).to_json()
        == generate_case(
            "S02",
            "S02_P01",
            seed=3,
        ).to_json()
    )


def test_s01_pair_expression_and_expected_identical() -> None:
    """S01:pair 内表达式与期望结果一致(固定三对)。"""
    for spec in fixed_pairs("S01"):
        arms = [_make(S01Calculator, spec) for _ in range(2)]
        assert all(t.prepare() for t in arms)
        assert arms[0].expression == arms[1].expression
        assert arms[0].expected == arms[1].expected
        assert spec.params["expected"] == eval(
            spec.params["expression"].replace("×", "*").replace("÷", "/"),
        )


def test_s02_pair_text_and_markers_identical() -> None:
    """S02:pair 内输入文本与随机标识一致。"""
    spec = generate_case("S02", "S02_P05", seed=11)
    arms = [_make(S02TextInput, spec) for _ in range(2)]
    assert all(t.prepare() for t in arms)
    assert arms[0].text == arms[1].text
    assert arms[0]._name_marker == arms[1]._name_marker
    assert arms[0]._number_marker == arms[1]._number_marker
    assert spec.params["name_marker"] in spec.params["text"]
    assert spec.params["number_marker"] in spec.params["text"]


def test_s03_pair_initial_and_target_volume_identical() -> None:
    """S03:pair 内 initial_volume 与 target_volume 一致(固定三档难度)。"""
    specs = fixed_pairs("S03")
    for spec in specs:
        assert spec.initial_state["volume"] == 50
        assert 0 <= spec.params["target_volume"] <= 100
        assert str(spec.params["target_volume"]) in spec.instruction
    initials = {s.initial_state["volume"] for s in specs}
    targets = {s.params["target_volume"] for s in specs}
    assert initials == {50}
    assert targets == {60, 20, 80}


def test_s04_pair_target_app_identical() -> None:
    """S04:pair 内目标应用四元组一致。"""
    spec = generate_case("S04", "S04_P02", seed=5)
    arms = [_make(S04OpenApp, spec) for _ in range(2)]
    for task in arms:
        assert task.instruction() == spec.instruction
    keys = ("application", "process", "window_class", "title_keyword")
    assert all(key in spec.params for key in keys)
    assert spec.params["application"] in spec.instruction


def test_s05_pair_search_query_identical() -> None:
    """S05:pair 内搜索词一致。"""
    spec = generate_case("S05", "S05_P03", seed=9)
    arms = [_make(S05Search, spec) for _ in range(2)]
    for task in arms:
        task.query = str(spec.params["query"])
    assert arms[0].query == arms[1].query == spec.params["query"]
    assert spec.params["query"] in spec.instruction


def test_s06_pair_target_fixture_identical() -> None:
    """S06:pair 内指令一致(该任务无随机字段)。"""
    spec = generate_case("S06", "S06_P01", seed=2)
    arms = [_make(S06CloseWindow, spec) for _ in range(2)]
    assert arms[0].instruction() == arms[1].instruction() == spec.instruction
    assert "记事本" in spec.instruction


def test_case_spec_serializes_into_report() -> None:
    """CaseSpec 可稳定序列化并包含全部难度相关字段组。"""
    spec = generate_case("S03", "S03_P01", seed=1)
    data = json.loads(spec.to_json())
    assert data["task_id"] == "S03"
    assert data["pair_id"] == "S03_P01"
    assert data["case_seed"] == 1
    assert data["instruction"]
    assert "target_volume" in data["params"]
    assert "volume" in data["initial_state"]
    assert spec.to_json() == json.dumps(data, ensure_ascii=False, sort_keys=True)


def test_case_spec_does_not_touch_production_agent() -> None:
    """CaseSpec 只存在与 benchmark 层;production 源码零引用。"""
    for pattern in ("agent", "config.py", "main.py", "perception", "control"):
        base = PROJECT / pattern
        paths = (
            [base] if base.is_file() else base.rglob("*.py") if base.exists() else []
        )
        for path in paths:
            content = path.read_text(encoding="utf-8", errors="replace")
            assert "case_spec" not in content, f"{path} 引用了 case_spec"


def test_s05_cleanup_selects_only_benchmark_chrome_windows() -> None:
    """S05 cleanup 只关闭 benchmark 新建或含本任务搜索词的 Chrome 窗口。"""
    from benchmark.tasks import select_benchmark_chrome_windows

    preexisting = {1, 2}
    windows = [
        {"hwnd": 1},
        {"hwnd": 2},
        {"hwnd": 3},
    ]
    titles = {
        1: "Gmail - 收件箱 - Chrome",
        2: "桌面智能体 - Google 搜索",
        3: "新标签页 - Chrome",
    }
    selected = select_benchmark_chrome_windows(
        windows,
        preexisting,
        "桌面智能体",
        lambda hwnd: titles[hwnd],
    )
    assert [w["hwnd"] for w in selected] == [2, 3]


def test_s01_verifier_matches_real_calculator_window() -> None:
    """A:ApplicationFrameHost 宿主 + 标题含"计算器" → MATCH。"""
    from benchmark.tasks import select_calculator_windows

    windows = [{"hwnd": 10, "process": "ApplicationFrameHost.exe"}]
    selected = select_calculator_windows(windows, lambda _h: "计算器")
    assert [w["hwnd"] for w in selected] == [10]


def test_s01_verifier_rejects_searchhost_title_match() -> None:
    """B:SearchHost 标题恰为搜索词"计算器" → NO MATCH。"""
    from benchmark.tasks import select_calculator_windows

    windows = [{"hwnd": 11, "process": "SearchHost.exe"}]
    assert select_calculator_windows(windows, lambda _h: "计算器") == []


def test_s01_verifier_rejects_start_menu_query_window() -> None:
    """C:Start/Search 查询"计算器"的叠层窗口 → NO MATCH。"""
    from benchmark.tasks import select_calculator_windows

    windows = [
        {"hwnd": 12, "process": "StartMenuExperienceHost.exe"},
        {"hwnd": 13, "process": "explorer.exe"},
        {"hwnd": 14, "process": "WindowsTerminal.exe"},
    ]
    assert select_calculator_windows(windows, lambda _h: "计算器") == []


def test_s01_verifier_prefers_calculator_over_search() -> None:
    """D:Calculator 与 Search 同屏 → 只选 Calculator。"""
    from benchmark.tasks import select_calculator_windows

    windows = [
        {"hwnd": 20, "process": "SearchHost.exe"},
        {"hwnd": 21, "process": "ApplicationFrameHost.exe"},
    ]
    selected = select_calculator_windows(windows, lambda _h: "计算器")
    assert [w["hwnd"] for w in selected] == [21]


def test_s01_validate_flow_with_stub_monitor() -> None:
    """E:S01 validate 既有回归:真计算器窗口 + OCR 含结果 → PASS。"""
    from benchmark.tasks import S01Calculator

    task = S01Calculator(
        run_id="T",
        desktop_dir=PROJECT / "logs",
        case_spec=fixed_pairs("S01")[0],
    )
    task.prepare()

    class _StubMonitor:
        def visible_windows(self):
            return [
                {"hwnd": 30, "process": "SearchHost.exe"},
                {"hwnd": 31, "process": "ApplicationFrameHost.exe"},
            ]

        def get_window_title(self, hwnd):
            return "计算器" if hwnd == 31 else "计算器 - 搜索"

        def ocr_window_text(self, hwnd):
            return "标准  23+48  71" if hwnd == 31 else ""

        def ocr_window_band_text(self, hwnd, top_fraction, height_fraction, zoom=1):
            # HARNESS H3 后 S01 只读显示区带;stub 模拟显示区 OCR 结果。
            return "71" if hwnd == 31 else ""

    task.monitor = _StubMonitor()
    ok, actual = task.validate()
    assert ok, actual
    assert "71" in actual
