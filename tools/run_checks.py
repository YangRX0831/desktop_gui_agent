"""运行项目的非 GUI canonical 质量检查并可生成本地报告。"""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
QUALITY_SCOPES = (
    "agent",
    "perception",
    "control",
    "utils",
    "tools",
    "main.py",
    "config.py",
    "verify_env.py",
    "tests",
    "benchmark",
)
PRODUCTION_SCOPES = (
    "config.py",
    "main.py",
    "verify_env.py",
    "agent",
    "perception",
    "control",
    "utils",
)


@dataclass(frozen=True)
class CheckSpec:
    """一项质量检查的名称、类别与参数列表。"""

    name: str
    category: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class CheckResult:
    """一项检查的可序列化执行结果。"""

    name: str
    category: str
    command: tuple[str, ...]
    exit_code: int
    output: str


def build_checks(*, unit: bool, quality: bool) -> list[CheckSpec]:
    """按选择构造 canonical 命令；不包含任何 GUI 或 official task。"""
    checks: list[CheckSpec] = []
    if unit:
        checks.extend(
            [
                CheckSpec(
                    "pytest-tests",
                    "unit",
                    (
                        sys.executable,
                        "-B",
                        "-m",
                        "pytest",
                        "tests",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                    ),
                ),
                CheckSpec(
                    "pytest-benchmark-self-tests",
                    "unit",
                    (
                        sys.executable,
                        "-B",
                        "-m",
                        "pytest",
                        "benchmark",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                    ),
                ),
            ]
        )
    if quality:
        checks.extend(
            [
                CheckSpec(
                    "mypy-production",
                    "quality",
                    (
                        sys.executable,
                        "-m",
                        "mypy",
                        "--ignore-missing-imports",
                        *PRODUCTION_SCOPES,
                    ),
                ),
                CheckSpec(
                    "black",
                    "quality",
                    (sys.executable, "-m", "black", "--check", *QUALITY_SCOPES),
                ),
                CheckSpec(
                    "isort",
                    "quality",
                    (
                        sys.executable,
                        "-m",
                        "isort",
                        "--check-only",
                        "--profile",
                        "black",
                        *QUALITY_SCOPES,
                    ),
                ),
                CheckSpec(
                    "flake8",
                    "quality",
                    (
                        sys.executable,
                        "-m",
                        "flake8",
                        "--max-line-length=88",
                        "--extend-ignore=E203,W503",
                        *QUALITY_SCOPES,
                    ),
                ),
                CheckSpec(
                    "git-diff-check",
                    "quality",
                    ("git", "diff", "--check"),
                ),
            ]
        )
    return checks


def run_check(spec: CheckSpec) -> CheckResult:
    """在项目根目录运行单项检查，并保留合并后的文本输出。"""
    completed = subprocess.run(
        list(spec.command),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CheckResult(
        spec.name,
        spec.category,
        spec.command,
        completed.returncode,
        completed.stdout.rstrip(),
    )


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def report_payload(
    results: Sequence[CheckResult],
    *,
    generated_at: str,
) -> dict[str, object]:
    """构造包含环境、Git 状态及逐项结果的报告对象。"""
    return {
        "generated_at": generated_at,
        "python": sys.version,
        "platform": platform.platform(),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "overall": "PASS" if all(item.exit_code == 0 for item in results) else "FAIL",
        "checks": [asdict(item) for item in results],
        "scope_note": "Non-GUI checks only; no targeted or official task is executed.",
    }


def write_report(payload: dict[str, object], timestamp: str) -> tuple[Path, Path]:
    """把同一报告写为 JSON 与便于人工阅读的 Markdown。"""
    output_dir = ROOT / "reports" / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"quality_{timestamp}.json"
    markdown_path = output_dir / f"quality_{timestamp}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Quality Report",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Overall: {payload['overall']}",
        f"- Git HEAD: `{payload['git_head']}`",
        f"- Dirty worktree: `{payload['git_dirty']}`",
        f"- Scope: {payload['scope_note']}",
        "",
        "| Check | Category | Exit |",
        "|---|---|---:|",
    ]
    checks = payload["checks"]
    assert isinstance(checks, list)
    for item in checks:
        assert isinstance(item, dict)
        lines.append(f"| {item['name']} | {item['category']} | {item['exit_code']} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 unit/quality/all 与本地报告开关。"""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--unit",
        action="store_true",
        help="仅运行 tests/ 与 benchmark self-tests；不运行 GUI task。",
    )
    selection.add_argument(
        "--quality",
        action="store_true",
        help="仅运行 mypy、Black、isort、flake8 与 git diff --check。",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="运行全部 unit 与 quality 检查；不运行 GUI task。",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="在 reports/quality/ 写入 JSON 与 Markdown 结果。",
    )
    args = parser.parse_args(arguments)
    if not (args.unit or args.quality or args.all):
        args.all = True
    return args


def main(arguments: Sequence[str] | None = None) -> int:
    """运行所选检查，按需落盘报告，并以汇总状态作为退出码。"""
    args = parse_args(arguments)
    checks = build_checks(
        unit=args.unit or args.all,
        quality=args.quality or args.all,
    )
    results: list[CheckResult] = []
    for spec in checks:
        print(f"[{spec.category}] {spec.name}", flush=True)
        result = run_check(spec)
        results.append(result)
        if result.output:
            print(result.output, flush=True)
        print(f"exit={result.exit_code}", flush=True)
    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    payload = report_payload(results, generated_at=now.isoformat())
    if args.report:
        paths = write_report(payload, timestamp)
        print(f"report_json={paths[0]}")
        print(f"report_markdown={paths[1]}")
    return 0 if payload["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
