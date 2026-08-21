"""配对验证驱动:同一 CaseSpec 下 V1 vs CLEAN_V3 的逐 pair A/B 运行。

用法:
    python -B benchmark/paired_runner.py --tasks S01,S03

每个 pair 内两臂读取同一份 CaseSpec(同 instruction、同 initial state),
S03 在每臂 prepare 时显式 reset 初始音量并读回确认。结果增量写入
benchmark/reports/PAIRED_<ts>/paired_results.jsonl。
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.case_specs import CaseSpec, fixed_pairs, generate_case  # noqa: E402
from benchmark.core import Status  # noqa: E402
from benchmark.core import WebFixture  # noqa: E402
from benchmark.runner import (  # noqa: E402
    DESKTOP,
    REPORT_DIR,
    TASK_TIMEOUT,
    TASK_TIMEOUT_MEDIUM_HIGH,
    AgentSession,
    restore_benchmark_desktop_state,
)
from benchmark.tasks import ALL_TASKS  # noqa: E402
from benchmark.tasks import (  # noqa: E402
    FIXTURE_CHAT_CONTACTS,
    FIXTURE_EMAIL_RECIPIENTS,
    FIXTURE_GALLERY,
    FIXTURE_TASK_IDS,
    H01_SECTIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("paired")

TASK_CLASSES = {t.task_id: t for t in ALL_TASKS}


def gen_pair_id() -> str:
    return time.strftime("PAIRED_%Y%m%d_%H%M%S")


LOCAL_MODEL_DIR = (
    r"C:\AI\OpenVINO\Qwen2-VL-2B-Instruct-INT4"
    r"\895c3a49bc3fa70a340399125c650a463535e71c"
    r"-ov2026.2.1-oi1.26.0-nncf3.2.0-int4"
)


def run_arm(
    spec: CaseSpec,
    arm: str,
    report_dir: Path,
    desktop_dir: Path,
    semantic: bool = False,
    local: bool = False,
    timeout: float = TASK_TIMEOUT,
    fixture_ok: bool = True,
) -> dict:
    """按 CaseSpec 运行一个 arm;返回逐 run 结果记录。

    HARNESS H2:依赖 WebFixture 的任务在 fixture 未启动时直接按
    ENV_ERROR 分类,不创建会话、不注入指令、不计普通 Agent FAIL。
    """
    if spec.task_id in FIXTURE_TASK_IDS and not fixture_ok:
        log.error("%s %s: fixture 未启动,标记 ENV_ERROR", spec.pair_id, arm)
        return {
            "pair_id": spec.pair_id,
            "arm": "V1" if arm == "v1" else "CLEAN_V3",
            "status": "ENV_ERROR",
            "failure_reason": "fixture_start_failed",
        }
    env_overrides = {
        "GUI_AGENT_TRACE": "1",
        "GUI_AGENT_TRACE_TASK_ID": spec.task_id,
        "GUI_AGENT_DECISION_PROTOCOL_V2": "0",
        "GUI_AGENT_DECISION_PROTOCOL_V3": "1" if arm == "v3" else "0",
        "GUI_AGENT_HIDE_OWN_WINDOW_DURING_RUN": "1",
    }
    if semantic:
        env_overrides["GUI_AGENT_SEMANTIC_EXECUTION"] = "1"
    cli_args = None
    if local:
        env_overrides["GUI_AGENT_LOCAL_RUNTIME"] = "openvino"
        env_overrides["GUI_AGENT_OPENVINO_MODEL_DIR"] = LOCAL_MODEL_DIR
        # run_api.bat 已含 --model-mode api;argparse 后值覆盖 → local。
        cli_args = ["--model-mode", "local"]
    session = AgentSession(env_overrides=env_overrides)
    if not restore_benchmark_desktop_state(session.monitor):
        log.error("%s %s: 桌面状态恢复失败", spec.pair_id, arm)
        session.stop()
        return {"pair_id": spec.pair_id, "arm": arm, "status": "ENV_ERROR"}
    if not session.start(cli_args):
        session.stop()
        return {"pair_id": spec.pair_id, "arm": arm, "status": "ENV_ERROR"}

    entry = {
        "pair_id": spec.pair_id,
        "arm": "V1" if arm == "v1" else "CLEAN_V3",
        "semantic_execution": semantic,
        "case_spec": json.loads(spec.to_json()),
    }
    task_cls = TASK_CLASSES[spec.task_id]
    task = task_cls(
        run_id=spec.pair_id,
        desktop_dir=desktop_dir,
        case_spec=spec,
    )
    shot = report_dir / "screenshots"
    shot.mkdir(parents=True, exist_ok=True)
    try:
        session.monitor.screenshot(
            str(shot / f"{spec.pair_id}_{arm}_before.png"),
        )
        task.terminal_hwnd = session.terminal_hwnd
        # FIX A:统一经 task.instruction() 解析占位符(如
        # GUIAgentBenchmark_CASE → 实际 desktop_dir 名);Agent 与
        # TaskResult 都使用 resolved 版本,原始 CaseSpec 不变。
        task.result.instruction = task.instruction()
        prepared = False
        try:
            prepared = task.prepare()
        except Exception as exception:
            log.error("prepare 异常: %s", type(exception).__name__)
        if not prepared:
            if spec.task_id == "M06":
                # 破坏性系统操作按设计不执行:SAFETY_SKIP 单列,不算
                # 环境异常,不计入可执行任务分母,也不算 PASS。
                entry.update(
                    status="SAFETY_SKIP",
                    failure_reason="破坏性系统操作按安全设计跳过",
                )
            else:
                entry.update(
                    status="SKIP",
                    failure_reason=task.result.failure_reason or "环境准备失败",
                )
        else:
            entry["initial"] = dict(task.result.params)
            task.result.start_time = time.time()
            if session.send_task_with_retry(task.result.instruction):
                outcome, _line = session.wait_task(timeout)
                task.result.end_time = time.time()
                task.result.elapsed = task.result.end_time - task.result.start_time
                if outcome == "done":
                    ok, actual = task.validate()
                    task.result.actual = actual
                    task.result.status = Status.PASS if ok else Status.FAIL
                    if not ok:
                        task.result.failure_reason = actual
                elif outcome == "cli_died":
                    task.result.status = Status.ENV_ERROR
                    task.result.failure_reason = "Agent CLI 进程中途退出"
                else:
                    task.result.status = Status.TIMEOUT
                    task.result.failure_reason = f"outcome={outcome}"
                entry.update(
                    status=task.result.status.name,
                    actual=task.result.actual,
                    failure_reason=task.result.failure_reason,
                    elapsed_seconds=round(task.result.elapsed, 1),
                )
            else:
                entry.update(status="ENV_ERROR", failure_reason="键盘注入失败")
        session.monitor.screenshot(
            str(shot / f"{spec.pair_id}_{arm}_after.png"),
        )
        try:
            task.cleanup()
        except Exception as exception:
            log.error("cleanup 异常: %s", type(exception).__name__)
    finally:
        session.stop()
        restore_benchmark_desktop_state(session.monitor)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired benchmark runner")
    parser.add_argument("--tasks", type=str, default="S01,S03")
    parser.add_argument(
        "--pair-count",
        type=int,
        default=None,
        help="每个任务只运行前 N 个 pair(快速迭代预算控制;默认全部)",
    )
    parser.add_argument(
        "--pair-ids",
        type=str,
        default=None,
        help="只运行指定 pair(逗号分隔,如 S03_P02);覆盖 --pair-count",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="v1,v3",
        help="运行的 arm(逗号分隔,如 v3;默认 v1,v3)",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="启用 SEMANTIC EXECUTION PHASE 2A(GUI_AGENT_SEMANTIC_EXECUTION=1)",
    )
    parser.add_argument(
        "--preset",
        choices=["paired", "acceptance"],
        default="paired",
        help="acceptance=PRD SIMPLE 固定六实例,V3+semantic 各跑一次",
    )
    parser.add_argument("--timeout", type=float, default=TASK_TIMEOUT)
    parser.add_argument(
        "--local",
        action="store_true",
        help="OpenVINO CPU 本地模式(LOCAL 2B BASELINE)",
    )
    args = parser.parse_args()
    task_ids = [t.strip().upper() for t in args.tasks.split(",")]

    report_dir = REPORT_DIR / gen_pair_id()
    report_dir.mkdir(parents=True, exist_ok=True)
    desktop_dir = DESKTOP / "GUIAgentBenchmark_Paired"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    results_path = report_dir / "paired_results.jsonl"

    specs: list[CaseSpec] = []
    if args.preset == "acceptance":
        from benchmark.case_specs import acceptance_cases

        specs = [case for case in acceptance_cases() if case.task_id in task_ids]
    for task_id in task_ids:
        if specs:
            break
        if task_id in ("S01", "S03"):
            task_specs = list(fixed_pairs(task_id))
        else:
            task_specs = [
                generate_case(task_id, f"{task_id}_P{n:02d}", seed=n)
                for n in range(1, 11)
            ]
        if args.pair_count is not None:
            task_specs = task_specs[: args.pair_count]
        specs.extend(task_specs)
    if args.pair_ids:
        wanted = {p.strip().upper() for p in args.pair_ids.split(",")}
        specs = [s for s in specs if s.pair_id.upper() in wanted]
    # 按命令行 --tasks 顺序执行(acceptance 单臂按需排序)。
    specs.sort(key=lambda s: task_ids.index(s.task_id) if s.task_id in task_ids else 99)
    arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    if args.preset == "acceptance":
        arms = ["v3"]
        args.semantic = True

    # M02/M03/M05/H01 依赖本地 WebFixture(webmail/gallery/chat/article)。
    needs_fixture = any(s.task_id in FIXTURE_TASK_IDS for s in specs)
    fixture = WebFixture(port=18888)
    fixture_ok = True
    if needs_fixture:
        fixture.configure(
            gen_pair_id(),
            {
                "gallery_images": FIXTURE_GALLERY,
                "article_sections": H01_SECTIONS,
                "email_recipients": FIXTURE_EMAIL_RECIPIENTS,
                "chat_contacts": FIXTURE_CHAT_CONTACTS,
            },
        )
        try:
            fixture_ok = fixture.start()
        except Exception as exception:
            log.error("fixture 启动异常: %s", type(exception).__name__)
            fixture_ok = False
        if not fixture_ok:
            # HARNESS H2:环境失败按 ENV_ERROR 分类,依赖任务不再执行。
            log.error("WebFixture 启动失败,依赖任务将标记 fixture_start_failed")
    try:
        with open(results_path, "w", encoding="utf-8") as out:
            for spec in specs:
                for arm in arms:
                    log.info(
                        "=== %s %s: %s",
                        spec.pair_id,
                        arm.upper(),
                        spec.instruction[:50],
                    )
                    # HARNESS H5:未显式指定 --timeout 时按难度取统一
                    # 常量:S 类 TASK_TIMEOUT,M/H 类 TASK_TIMEOUT_MEDIUM_HIGH
                    # (与 runner.py 主线同源,不再对 M/H 写死 480s)。
                    arm_timeout = args.timeout
                    if arm_timeout is None:
                        task_cls = TASK_CLASSES.get(spec.task_id)
                        arm_timeout = (
                            TASK_TIMEOUT
                            if task_cls is not None and task_cls.difficulty == "简单"
                            else TASK_TIMEOUT_MEDIUM_HIGH
                        )
                    entry = run_arm(
                        spec,
                        arm,
                        report_dir,
                        desktop_dir,
                        semantic=args.semantic,
                        local=args.local,
                        timeout=arm_timeout,
                        fixture_ok=fixture_ok,
                    )
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out.flush()
                    log.info(
                        "=== %s %s -> %s %s",
                        spec.pair_id,
                        arm.upper(),
                        entry.get("status"),
                        str(entry.get("actual"))[:60],
                    )
                    time.sleep(3)
    finally:
        if needs_fixture:
            fixture.stop()
    shutil.rmtree(desktop_dir, ignore_errors=True)
    log.info("paired validation 完成: %s", results_path)
    # 正式口径:PRD_TOTAL 15,M06 单列 SAFETY_SKIP;成功率分别按
    # /15 与 /14 executable 输出,SKIP 不计作 PASS。
    entries = [json.loads(line) for line in open(results_path, encoding="utf-8")]
    passes = sum(1 for e in entries if e.get("status") == "PASS")
    skips = sum(1 for e in entries if e.get("status") == "SAFETY_SKIP")
    executable = len(entries) - skips
    log.info(
        "PRD_TOTAL=%d EXECUTABLE=%d SAFETY_SKIP=%d | "
        "PASS %d/%d | PASS %d/%d executable",
        len(entries),
        executable,
        skips,
        passes,
        len(entries),
        passes,
        executable,
    )


if __name__ == "__main__":
    main()
