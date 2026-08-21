"""Canonical 非 GUI 质量入口的命令选择与报告测试。"""

import json
from pathlib import Path

import pytest

from tools import run_checks


def test_default_selection_is_all() -> None:
    """无显式选择时运行 unit 与 quality，但不隐式运行 GUI task。"""
    args = run_checks.parse_args([])
    checks = run_checks.build_checks(
        unit=args.unit or args.all,
        quality=args.quality or args.all,
    )
    names = {item.name for item in checks}
    assert "pytest-tests" in names
    assert "mypy-production" in names
    assert all("paired_runner" not in item.command for item in checks)


def test_help_describes_every_mode_and_excludes_gui(capsys) -> None:
    """正式测试入口的 help 明确选择范围与非 GUI 边界。"""
    with pytest.raises(SystemExit) as raised:
        run_checks.parse_args(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    for option in ("--unit", "--quality", "--all", "--report"):
        assert option in output
    assert "不运行 GUI task" in output


def test_unit_selection_excludes_quality() -> None:
    """unit 模式只含两组 pytest。"""
    checks = run_checks.build_checks(unit=True, quality=False)
    assert [item.name for item in checks] == [
        "pytest-tests",
        "pytest-benchmark-self-tests",
    ]


def test_quality_scopes_include_tooling_but_mypy_stays_production_only() -> None:
    """格式检查覆盖工具脚本，production mypy 口径保持既有范围。"""
    checks = run_checks.build_checks(unit=False, quality=True)
    by_name = {item.name: item for item in checks}
    assert "tools" in by_name["black"].command
    assert "tools" in by_name["flake8"].command
    assert "tools" not in by_name["mypy-production"].command


def test_report_payload_exposes_failure(monkeypatch) -> None:
    """任一非零退出码必须使汇总为 FAIL。"""
    monkeypatch.setattr(run_checks, "_git_value", lambda *args: "head")
    results = [
        run_checks.CheckResult("ok", "unit", ("python",), 0, "passed"),
        run_checks.CheckResult("bad", "quality", ("git",), 1, "failed"),
    ]
    payload = run_checks.report_payload(results, generated_at="now")
    assert payload["overall"] == "FAIL"
    assert payload["git_dirty"] is True
    assert len(payload["checks"]) == 2  # type: ignore[arg-type]


def test_write_report_creates_matching_json_and_markdown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """JSON 与 Markdown 报告必须基于同一汇总状态。"""
    monkeypatch.setattr(run_checks, "ROOT", tmp_path)
    payload = {
        "generated_at": "2026-08-21T20:00:00+08:00",
        "overall": "PASS",
        "git_head": "abc",
        "git_dirty": False,
        "scope_note": "Non-GUI only",
        "checks": [
            {
                "name": "pytest-tests",
                "category": "unit",
                "command": ["python"],
                "exit_code": 0,
                "output": "passed",
            }
        ],
    }
    json_path, markdown_path = run_checks.write_report(payload, "stamp")
    assert json.loads(json_path.read_text(encoding="utf-8"))["overall"] == "PASS"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| pytest-tests | unit | 0 |" in markdown
    assert "Official" not in markdown
