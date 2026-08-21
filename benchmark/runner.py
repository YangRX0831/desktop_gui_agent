"""Benchmark 主入口:启动 Agent → 注入任务 → 验证 → 中文报告。

用法:
    python benchmark/runner.py                    # 全部 15 项
    python benchmark/runner.py --task S01         # 单项
    python benchmark/runner.py --runs 3           # 每项 3 次
    python benchmark/runner.py --seed 42          # 固定随机种子
    python benchmark/runner.py --clean            # 清除全部测试痕迹
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 脚本引导:支持 `python benchmark/runner.py` 直接运行,须先注入
# 项目根与 benchmark 包路径,再导入本地模块(因此豁免 E402)。
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.core import (  # noqa: E402
    Injector,
    Monitor,
    Status,
    TaskResult,
    WebFixture,
    find_marker_line_since,
    log_candidate_paths,
)
from benchmark.tasks import (  # noqa: E402
    ALL_TASKS,
    FIXTURE_CHAT_CONTACTS,
    FIXTURE_EMAIL_RECIPIENTS,
    FIXTURE_GALLERY,
    FIXTURE_TASK_IDS,
    H01_SECTIONS,
    gen_run_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("benchmark")

PROJECT_DIR = Path(__file__).resolve().parent.parent
BAT_PATH = PROJECT_DIR / "run_api.bat"
APP_LOG = PROJECT_DIR / "logs" / "desktop_gui_agent.log"
REPORT_DIR = PROJECT_DIR / "benchmark" / "reports"
DESKTOP = Path(os.environ.get("USERPROFILE", "")) / "Desktop"

# 分阶段任务超时(可用 --task-timeout 统一覆盖):S 类按 PRD 默认
# 10 步给 480s;M/H 类按 25 步 × 单步 20-30s 给 720s 余量。
TASK_TIMEOUT = 480
TASK_TIMEOUT_MEDIUM_HIGH = 1000
# M/H 类任务的 max_steps:实测 Excel 逐格录入自然需求约 27 步,取 30。
MAX_STEPS_MEDIUM_HIGH = 30
# Agent 启动等待秒数(等待 cli_ready 日志信号;实测冷启动含依赖导入约 87s)
STARTUP_TIMEOUT = 180
# 任务执行中日志无任何新增字节达到该秒数时判定卡死,立即强制中断;
# 须容纳两次连续慢 API 调用(单请求上限120s),过短会把合法重试误判为卡死
STALL_TIMEOUT = 240
# 监控轮询间隔秒数
POLL_INTERVAL = 2.0
# 终端宿主进程名(按进程名匹配新出现的终端窗口)
_TERMINAL_PROCESSES = ("windowsterminal.exe", "cmd.exe", "conhost.exe")
# 任务终态与就绪日志标记
_TERMINAL_MARKERS = ("gui_agent_task_succeeded", "gui_agent_task_failed")
# 当前会话终态行的时间戳余量:覆盖行时间戳秒级粒度与轮询相位差;
# 历史 run 的终态行至少早于本会话一个 CLI 启动周期(>10s),不会误入。
TERMINAL_SINCE_MARGIN_S = 5.0
_READY_MARKER = "cli_ready"


# =========================================================================
# 清理
# =========================================================================


def clean_all() -> None:
    """无痕清除所有测试痕迹。"""
    removed = []
    # 清除桌面测试目录
    for d in DESKTOP.iterdir():
        if d.is_dir() and d.name.startswith("GUIAgentBenchmark"):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
    # 清除报告目录
    if REPORT_DIR.exists():
        for d in REPORT_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))
    # 终止残留测试进程
    for proc in ("CalculatorApp.exe", "Notepad.exe", "EXCEL.EXE", "POWERPNT.EXE"):
        subprocess.run(
            ["taskkill", "/IM", proc, "/F"],
            capture_output=True,
            errors="replace",
        )
    if removed:
        print(f"已清除 {len(removed)} 项:")
        for r in removed:
            print(f"  {r}")
    else:
        print("无测试痕迹需要清除。")


# =========================================================================
# Agent 启动与交互
# =========================================================================

_SHELL_OVERLAY_CLASSES = frozenset({"Windows.UI.Core.CoreWindow"})


def restore_benchmark_desktop_state(monitor: Monitor) -> bool:
    """统一恢复桌面到可测试状态;失败返回 False。

    Agent run 可能遗留 Windows 搜索/开始菜单/运行对话框等临时 Shell
    叠层,这类窗口持有前台时,下一 run 的 CLI 无法获取前台(实测
    SetForegroundWindow 被拒)。用 Esc 安全关闭临时 Shell UI——不杀
    explorer、不重启 Shell、不动系统进程;对 V1/V2 完全一致。
    """
    import ctypes

    user32 = ctypes.windll.user32
    for attempt in range(3):
        hwnd = user32.GetForegroundWindow()
        window_class = monitor.get_window_class(hwnd)
        title = monitor.get_window_title(hwnd)
        is_overlay = window_class in _SHELL_OVERLAY_CLASSES or (
            window_class == "#32770" and title == "运行"
        )
        if not is_overlay:
            return True
        log.info("检测到 Shell 叠层(class=%s),发送 Esc 关闭", window_class)
        user32.keybd_event(0x1B, 0, 0, 0)
        user32.keybd_event(0x1B, 0, 2, 0)
        time.sleep(1.5)
    hwnd = user32.GetForegroundWindow()
    return monitor.get_window_class(hwnd) not in _SHELL_OVERLAY_CLASSES


class AgentSession:
    """管理 Agent CLI 进程的启动、指令注入、监控和强制中断。

    通过 Popen 持有 CLI 进程树 PID,只终止自己启动的进程,绝不按
    映像名全局 taskkill(避免误杀 benchmark 自身或用户其他终端)。
    """

    def __init__(self, env_overrides: dict[str, str] | None = None) -> None:
        self.injector = Injector()
        self.monitor = Monitor()
        self.terminal_hwnd = 0
        self._proc: subprocess.Popen | None = None
        self._user_terminal_hwnds: set[int] = set()
        self._session_seq = 0
        self._console_title = ""
        self._env_overrides = dict(env_overrides or {})

    def start(self, cli_args: list[str] | None = None) -> bool:
        """启动 run_api.bat,等待 cli_ready 日志并定位新终端窗口。

        cli_args 透传给 main.py(如 --max-steps 25),用于 M/H 阶段的
        深步数运行;None 表示使用 config 默认(PRD 默认 10 步)。
        """
        log.info("启动 %s ...", BAT_PATH.name)
        if not self._user_terminal_hwnds:
            # 首次启动前记录用户自己的终端窗口;兜底路径只考虑这之外
            # 的终端窗口,避免把指令注进用户终端。
            self._user_terminal_hwnds = {
                w["hwnd"]
                for w in self.monitor.visible_windows()
                if str(w["process"]).lower() in _TERMINAL_PROCESSES
            }
        # 控制台标题标记:每次会话唯一,cmd 的 title 命令写入控制台标题,
        # 终端宿主(Windows Terminal/conhost)会把它显示为窗口标题;
        # 按标题定位可根治"死会话残留窗口与存活 CLI 无法区分"的问题。
        self._session_seq += 1
        self._console_title = f"GUIAGENT_CLI_{self._session_seq}"
        bat_args = " ".join(str(a) for a in cli_args) if cli_args else ""
        command = f"title {self._console_title} && " f"{BAT_PATH} {bat_args}".rstrip()
        if cli_args:
            log.info("CLI 附加参数: %s", cli_args)
        child_env = dict(os.environ)
        child_env.update(self._env_overrides)
        self._proc = subprocess.Popen(
            ["cmd", "/c", command],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=child_env,
        )
        ready = self.monitor.wait_for_log(
            APP_LOG,
            [_READY_MARKER],
            STARTUP_TIMEOUT,
        )
        if ready is None:
            log.error("等待 cli_ready 超时(%.0fs)", STARTUP_TIMEOUT)
            self.stop()
            return False
        # 首选:按唯一标题定位本次会话的终端窗口
        deadline = time.time() + 15
        while time.time() < deadline and not self.terminal_hwnd:
            self.terminal_hwnd = self._find_titled_window()
            time.sleep(1)
        if not self.terminal_hwnd:
            # 兜底:启动前后差集 + 排除用户终端(旧行为)
            windows_before = {w["hwnd"] for w in self.monitor.visible_windows()}
            time.sleep(2)
            for w in self.monitor.visible_windows():
                if w["hwnd"] in windows_before:
                    continue
                if str(w["process"]).lower() in _TERMINAL_PROCESSES:
                    self.terminal_hwnd = w["hwnd"]
                    break
        if not self.terminal_hwnd:
            log.error("未找到 Agent 终端窗口(标题=%s)", self._console_title)
            self.stop()
            return False
        log.info(
            "Agent 就绪,终端 hwnd=%d 标题=%s",
            self.terminal_hwnd,
            self._console_title,
        )
        return True

    def _find_titled_window(self) -> int:
        """按本会话唯一控制台标题查找终端窗口;未找到返回 0。"""
        for w in self.monitor.visible_windows():
            if self._console_title in self.monitor.get_window_title(w["hwnd"]):
                return w["hwnd"]
        return 0

    def send_task(self, instruction: str) -> bool:
        """注入一条任务指令。"""
        return self.injector.type_and_enter(
            self.terminal_hwnd,
            instruction,
        )

    def reacquire_terminal(self) -> bool:
        """重新定位 Agent 自己的终端窗口,不动 CLI 进程。

        注入失败的常见原因是终端窗口句柄失效或焦点被占,而非 CLI
        死亡;杀掉 CLI 会损失 OCR/模型预热,仅在窗口无法恢复时才允许
        重启会话。重获范围排除用户自有终端窗口。
        """
        if self._proc is not None and self._proc.poll() is not None:
            return False
        titled = self._find_titled_window()
        if titled:
            self.terminal_hwnd = titled
            log.info("终端窗口重获(标题匹配) hwnd=%d", titled)
            return True
        candidates = [
            w["hwnd"]
            for w in self.monitor.visible_windows()
            if str(w["process"]).lower() in _TERMINAL_PROCESSES
            and w["hwnd"] not in self._user_terminal_hwnds
        ]
        if not candidates:
            return False
        self.terminal_hwnd = candidates[0]
        log.info("终端窗口重获(兜底) hwnd=%d", self.terminal_hwnd)
        return True

    def send_task_with_retry(
        self,
        instruction: str,
        max_retries: int = 2,
    ) -> bool:
        """注入指令;失败时优先重获窗口保持 CLI 存活,重启是最后手段。

        GUI Agent 进程承载 OCR 与模型预热,正常任务间的窗口清理不得
        连带关闭它;只有窗口重获失败(或 CLI 已死)才重启会话。
        """
        stale_hwnd_failures = 0
        for attempt in range(1 + max_retries):
            if self.send_task(instruction):
                return True
            log.warning(
                "指令注入失败(第%d/%d次),尝试重获终端窗口...",
                attempt + 1,
                1 + max_retries,
            )
            time.sleep(2)
            previous_hwnd = self.terminal_hwnd
            if self.reacquire_terminal():
                if self.terminal_hwnd == previous_hwnd:
                    # 窗口仍在但持续无法获得前台:重获无意义,计数并强制重启。
                    stale_hwnd_failures += 1
                else:
                    stale_hwnd_failures = 0
                if stale_hwnd_failures >= 2:
                    log.warning("同一终端窗口持续无法置前,重启 CLI 会话...")
                    self.stop()
                    if not self.start():
                        return False
                    stale_hwnd_failures = 0
                continue
            log.warning("终端窗口无法重获,重启 CLI 会话...")
            self.stop()
            if not self.start():
                return False
            time.sleep(2)
        return False

    def is_alive(self) -> bool:
        """检查 Agent CLI 进程是否仍在运行。"""
        return self._proc is not None and self._proc.poll() is None

    def ensure_alive(self) -> bool:
        """CLI 死亡时自动重启;存活时返回 True。"""
        if self.is_alive():
            return True
        log.warning("Agent CLI 已退出,正在重启...")
        self.stop()
        return self.start()

    def stop(self) -> None:
        """按 PID 进程树终止本次启动的 CLI;绝不按映像名全局杀进程。"""
        # 先向终端窗口发 WM_CLOSE,减少被杀会话留下的死壳窗口干扰
        # 后续的窗口差集定位;失败不影响强杀兜底。
        if self.terminal_hwnd:
            import ctypes

            user32 = ctypes.windll.user32
            if user32.IsWindow(self.terminal_hwnd):
                user32.PostMessageW(self.terminal_hwnd, 0x0010, 0, 0)
            time.sleep(1)
        if self._proc is not None and self._proc.poll() is None:
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(self._proc.pid),
                    "/T",
                    "/F",
                ],
                capture_output=True,
                errors="replace",
            )
            time.sleep(2)
        self._proc = None
        self.terminal_hwnd = 0

    def wait_task(self, timeout: float) -> tuple[str, str | None]:
        """持续监控的任务等待,返回 (outcome, 终态日志行)。

        outcome:
            done      — 日志出现本次会话的终态标记;
            timeout   — 达到硬超时上限;
            stalled   — 日志无任何新增字节超过 STALL_TIMEOUT,判定卡死;
            cli_died  — CLI 进程中途退出。

        HARNESS H1:终态判定不再依赖"当前文件计数超过 baseline"
        (午夜 TimedRotatingFileHandler 轮转后新文件计数归零,永不超
        基线,V2 M05 因此被误判 TIMEOUT)。改为按行首时间戳做当前
        会话关联:只接受时间戳不早于本次等待开始(留 5s 时钟余量)
        的终态行,且当前文件与最新轮转副本都扫描。历史 run 的终态
        行时间更早,天然隔离。
        """
        since = time.time() - TERMINAL_SINCE_MARGIN_S
        last_size = self._log_size()
        last_progress = time.time()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_alive():
                return "cli_died", None
            size = self._log_size()
            if size != last_size:
                last_size = size
                last_progress = time.time()
            line = find_marker_line_since(
                APP_LOG,
                list(_TERMINAL_MARKERS),
                since,
            )
            if line is not None:
                return "done", line
            if time.time() - last_progress >= STALL_TIMEOUT:
                return "stalled", None
            time.sleep(POLL_INTERVAL)
        return "timeout", None

    def _log_size(self) -> int:
        """返回当前日志与最新轮转副本的字节总量;不存在时为 0。

        求和使午夜轮转(当前文件变小)不再被误读为"无进展"。
        """
        total = 0
        for path in log_candidate_paths(APP_LOG):
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total


# =========================================================================
# 报告生成
# =========================================================================


def gen_report(
    results: list[TaskResult],
    run_id: str,
    report_dir: Path,
    trace_metrics: dict[str, int | float] | None = None,
) -> None:
    """生成中文 Markdown 报告 + JSON(含决策协议 trace 指标)。"""
    report_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = report_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    total = len(results)
    passed = sum(1 for r in results if r.status == Status.PASS)
    failed = sum(1 for r in results if r.status == Status.FAIL)
    skipped = sum(1 for r in results if r.status == Status.SKIP)
    env_err = sum(1 for r in results if r.status == Status.ENV_ERROR)
    timeouts = sum(1 for r in results if r.status == Status.TIMEOUT)
    safety_skips = sum(1 for r in results if r.status == Status.SAFETY_SKIP)
    executable = total - skipped - safety_skips
    success_rate = f"{passed / executable * 100:.1f}%" if executable else "N/A"
    overall_rate = f"{passed / total * 100:.1f}%" if total else "N/A"
    avg_time = f"{sum(r.elapsed for r in results if r.elapsed) / max(1, total):.1f} 秒"

    lines = [
        "# 桌面 GUI 智能体系统测试报告",
        "",
        "## 一、测试基本信息",
        "",
        f"- 测试编号：{run_id}",
        f"- 测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 操作系统：{sys.platform}",
        f"- 测试任务数量：{total}",
        "",
        "## 二、总体测试结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 测试任务总数 | {total} |",
        f"| 通过 | {passed} |",
        f"| 失败 | {failed} |",
        f"| 跳过 | {skipped} |",
        f"| 安全跳过 | {safety_skips} |",
        f"| 环境异常 | {env_err} |",
        f"| 超时 | {timeouts} |",
        f"| 可执行任务成功率 | {success_rate} |",
        f"| 总体任务成功率 | {overall_rate} |",
        f"| 平均执行时间 | {avg_time} |",
        "",
        "## 三、不同难度测试结果",
        "",
    ]

    for diff, label in [("简单", "简单"), ("中等", "中等"), ("复杂", "复杂")]:
        dr = [r for r in results if r.difficulty == diff]
        if dr:
            dp = sum(1 for r in dr if r.status == Status.PASS)
            rate = f"{dp / len(dr) * 100:.1f}%"
            lines.append(
                f"| {label} | {len(dr)} | {dp} | {len(dr) - dp} | {rate} |",
            )
    lines.insert(
        len(lines) - 3,
        "| 难度 | 任务数 | 通过 | 失败 | 成功率 |",
    )
    lines.insert(len(lines) - 3, "| --- | ---: | ---: | ---: | ---: |")

    lines += [
        "",
        "## 四、任务测试结果",
        "",
        "| 编号 | 测试任务 | 难度 | 结果 | 执行时间 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for r in results:
        elapsed = f"{r.elapsed:.1f} 秒" if r.elapsed else "-"
        lines.append(
            f"| {r.task_id} | {r.task_name} | {r.difficulty} "
            f"| {r.status.value} | {elapsed} |",
        )

    lines += ["", "## 五、失败任务汇总", ""]
    failures = [r for r in results if r.status in (Status.FAIL, Status.TIMEOUT)]
    if failures:
        lines += ["| 任务 | 失败现象 |", "| --- | --- |"]
        for r in failures:
            lines.append(f"| {r.task_id} | {r.failure_reason or r.actual} |")
    else:
        lines.append("本轮测试无失败任务。")

    lines += [
        "",
        "## 六、测试结论",
        "",
        f"本轮共执行 {total} 项桌面 GUI 智能体测试任务。",
        f"其中通过 {passed} 项，失败 {failed} 项，跳过 {skipped} 项。",
        f"可执行任务成功率为 {success_rate}。",
    ]

    md_path = report_dir / "测试报告.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("报告已保存: %s", md_path)

    # JSON
    json_data = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "safety_skip": safety_skips,
            "env_error": env_err,
            "timeout": timeouts,
        },
        "tasks": [
            {
                "task_id": r.task_id,
                "task_name": r.task_name,
                "difficulty": r.difficulty,
                "instruction": r.instruction,
                "params": r.params,
                "status": r.status.name,
                "status_cn": r.status.value,
                "elapsed_seconds": round(r.elapsed, 1),
                "actual": r.actual,
                "failure_reason": r.failure_reason,
            }
            for r in results
        ],
    }
    if trace_metrics:
        json_data["trace_metrics"] = trace_metrics
    json_path = report_dir / "测试结果.json"
    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("JSON已保存: %s", json_path)


def aggregate_trace_metrics(since: float) -> dict[str, int | float]:
    """聚合 since 之后写入的 agent trace,产出 V1/V2 通用的对比指标。"""
    import glob as glob_mod

    metrics: dict[str, int | float] = {
        "model_calls": 0,
        "gui_actions": 0,
        "observe_count": 0,
        "repeated_action_block_count": 0,
        "parse_failure_count": 0,
        "finish_count": 0,
    }
    latencies: list[float] = []
    for path in glob_mod.glob(
        str(PROJECT_DIR / "logs" / "agent_trace" / "*.jsonl"),
    ):
        if os.path.getmtime(path) < since:
            continue
        try:
            for line in open(path, encoding="utf-8"):
                record = json.loads(line)
                if record.get("record_type") != "model_call":
                    continue
                metrics["model_calls"] += 1
                parsed = str(record.get("parsed_action") or "")
                if not record.get("parse_success"):
                    metrics["parse_failure_count"] += 1
                    continue
                if "observe()" in parsed:
                    metrics["observe_count"] += 1
                elif "finish(" in parsed:
                    metrics["finish_count"] += 1
                else:
                    metrics["gui_actions"] += 1
                latency = record.get("api_latency_ms")
                if isinstance(latency, (int, float)):
                    latencies.append(float(latency))
        except (OSError, ValueError):
            continue
    if latencies:
        ordered = sorted(latencies)
        metrics["api_latency_mean_ms"] = round(
            sum(ordered) / len(ordered),
        )
        metrics["api_latency_p50_ms"] = round(
            ordered[len(ordered) // 2],
        )
        metrics["api_latency_p95_ms"] = round(
            ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        )
    return metrics


# =========================================================================
# 主流程
# =========================================================================


def run_benchmark(
    task_ids: list[str] | None = None,
    runs: int = 1,
    seed: int | None = None,
    task_timeout: float | None = None,
    agent_protocol: str = "v1",
) -> None:
    """执行测试;task_timeout 为用户显式统一覆盖,None 表示按阶段默认。

    agent_protocol 选择子 CLI 的决策协议版本(v1/v2),两者除该开关外
    配置完全一致,用于严格 A/B。
    """
    run_id = gen_run_id()
    desktop_dir = DESKTOP / f"GUIAgentBenchmark_{run_id}"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    report_dir = REPORT_DIR / run_id

    log.info("=" * 60)
    log.info("测试编号: %s", run_id)
    log.info("桌面测试目录: %s", desktop_dir)
    log.info("报告目录: %s", report_dir)
    log.info(
        "卡死阈值: %.0fs | 超时: S类%.0fs M/H类%.0fs",
        STALL_TIMEOUT,
        TASK_TIMEOUT,
        TASK_TIMEOUT_MEDIUM_HIGH,
    )
    log.info("=" * 60)

    # 启动 Web Fixture
    fixture = WebFixture(port=18888)
    # Fixture 数据全部取自 tasks.py 的单一数据源常量,保证页面内容
    # 与任务指令抽样范围一致(指令提到的对象页面上一定存在)。
    fixture.configure(
        run_id,
        {
            "gallery_images": FIXTURE_GALLERY,
            "article_sections": H01_SECTIONS,
            "email_recipients": FIXTURE_EMAIL_RECIPIENTS,
            "chat_contacts": FIXTURE_CHAT_CONTACTS,
        },
    )
    # HARNESS H2:fixture 启动失败/异常显式记录;依赖任务在循环内
    # 按 ENV_ERROR 分类,不再进入 Agent 执行后误记为 FAIL。
    try:
        fixture_started = fixture.start()
    except Exception as exception:
        log.error("Web Fixture 启动异常: %s", type(exception).__name__)
        fixture_started = False
    if fixture_started:
        log.info("Web Fixture 已启动: http://127.0.0.1:18888")
    else:
        log.error("Web Fixture 启动失败:依赖任务将标记 fixture_start_failed")

    # 启动 Agent(协议开关与 trace 经环境变量注入子 CLI)
    run_started_at = time.time()
    child_env = {
        "GUI_AGENT_TRACE": "1",
        "GUI_AGENT_DECISION_PROTOCOL_V2": "1" if agent_protocol == "v2" else "0",
        # 污染消除:benchmark 子 CLI 在 run 期间最小化自身控制窗口,
        # 使 Agent CLI 不进入模型视野(2026-08-19 PROTECTED_UI_DECONTAMINATION)。
        "GUI_AGENT_HIDE_OWN_WINDOW_DURING_RUN": "1",
    }
    session = AgentSession(env_overrides=child_env)
    if not session.start():
        log.error("Agent 启动失败,终止测试。")
        fixture.stop()
        if desktop_dir.exists():
            shutil.rmtree(desktop_dir, ignore_errors=True)
        return

    # 选择任务
    task_classes = ALL_TASKS
    if task_ids:
        task_classes = [t for t in ALL_TASKS if t.task_id in task_ids]

    all_results: list[TaskResult] = []

    # 阶段划分:S 类用 PRD 默认步数;M/H 类需要约 25 步,在阶段边界
    # 一次性带参数重启 CLI(仅此一次重启,阶段内保持 Agent 存活)。
    simple_tasks = [t for t in task_classes if t.difficulty == "简单"]
    medium_high_tasks = [t for t in task_classes if t.difficulty != "简单"]
    phase_all: list[tuple[str, list]] = []
    if simple_tasks:
        phase_all.append(("S", simple_tasks))
    if medium_high_tasks:
        phase_all.append(("MH", medium_high_tasks))

    environment_invalid = False
    try:
        for phase_name, phase_tasks in phase_all:
            if environment_invalid:
                break
            if task_timeout is None:
                phase_timeout: float = float(
                    TASK_TIMEOUT_MEDIUM_HIGH if phase_name == "MH" else TASK_TIMEOUT
                )
            else:
                phase_timeout = float(task_timeout)
            log.info("阶段 %s 开始,单任务超时 %.0fs", phase_name, phase_timeout)
            if phase_name == "MH":
                log.info(
                    "进入 M/H 阶段:重启 CLI(max_steps=%d)",
                    MAX_STEPS_MEDIUM_HIGH,
                )
                session.stop()
                if not session.start(
                    ["--max-steps", str(MAX_STEPS_MEDIUM_HIGH)],
                ):
                    log.error("M/H 阶段 CLI 启动失败")
                    break
            for task_cls in phase_tasks:
                for run_num in range(runs):
                    suffix = f"(第{run_num + 1}次)" if runs > 1 else ""
                    log.info(
                        "--- %s %s%s ---",
                        task_cls.task_id,
                        task_cls.task_name,
                        suffix,
                    )

                    task = task_cls(
                        run_id=run_id,
                        desktop_dir=desktop_dir,
                        seed=seed + run_num if seed else None,
                    )

                    # HARNESS H2:依赖 WebFixture 的任务在 fixture 未启动
                    # 时按环境异常分类:不恢复桌面、不注入指令、不计
                    # 普通 Agent FAIL。
                    if task_cls.task_id in FIXTURE_TASK_IDS and not fixture_started:
                        task.result.status = Status.ENV_ERROR
                        task.result.failure_reason = "fixture_start_failed"
                        all_results.append(task.result)
                        log.error("  → 环境异常(fixture_start_failed)")
                        continue

                    # 运行前统一恢复桌面(搜索/开始菜单等 Shell 叠层);
                    # 恢复失败按合同标记 environment_invalid 并停止后续 run。
                    if not restore_benchmark_desktop_state(session.monitor):
                        task.result.status = Status.ENV_ERROR
                        task.result.failure_reason = (
                            "environment_invalid:桌面状态恢复失败"
                        )
                        all_results.append(task.result)
                        log.error("  → environment_invalid,停止后续 benchmark")
                        environment_invalid = True
                        break

                    # 截图:任务开始前
                    shot_before = (
                        report_dir / "screenshots" / f"{task.task_id}_before.png"
                    )
                    shot_before.parent.mkdir(parents=True, exist_ok=True)
                    session.monitor.screenshot(str(shot_before))

                    # 确保 Agent CLI 存活(上项任务可能导致其退出)
                    if not session.ensure_alive():
                        task.result.status = Status.ENV_ERROR
                        task.result.failure_reason = "Agent CLI 重启失败"
                        all_results.append(task.result)
                        log.error("  → 环境异常(CLI重启失败)")
                        continue

                    # 准备(prepare 异常不得炸掉整个测试运行)
                    try:
                        prepared = task.prepare()
                    except Exception as exception:
                        prepared = False
                        log.error(
                            "  prepare 异常: %s",
                            type(exception).__name__,
                        )
                    if not prepared:
                        if task_cls.task_id == "M06":
                            # 破坏性系统操作按设计不执行:SAFETY_SKIP 单列,
                            # 不计入可执行分母,也不算 PASS(与 paired 口径一致)。
                            task.result.status = Status.SAFETY_SKIP
                            task.result.failure_reason = "破坏性系统操作按安全设计跳过"
                        else:
                            task.result.status = Status.SKIP
                            task.result.failure_reason = (
                                task.result.failure_reason or "环境准备失败"
                            )
                            task.result.actual = "跳过"
                        all_results.append(task.result)
                        log.warning("  → 跳过(%s)", task.result.failure_reason)
                        continue

                    task.terminal_hwnd = session.terminal_hwnd
                    task.result.instruction = task.instruction()
                    # 传输完整性守卫:指令经键盘注入到终端,任何换行都会
                    # 提前提交并使后续行变成碎片输入;Tab 在控制台输入中
                    # 同样不可靠。发现即判环境错误,绝不静默注入坏指令。
                    if (
                        "\n" in task.result.instruction
                        or "\r" in (task.result.instruction)
                        or "\t" in task.result.instruction
                    ):
                        task.result.status = Status.ENV_ERROR
                        task.result.failure_reason = "指令含换行或Tab,无法注入"
                        all_results.append(task.result)
                        log.error("  → 环境异常(指令含换行或Tab)")
                        continue
                    log.info("  指令: %s", task.result.instruction[:60])

                    # 注入指令(失败时内部重置会话并重试)
                    task.result.start_time = time.time()
                    if not session.send_task_with_retry(task.result.instruction):
                        task.result.status = Status.ENV_ERROR
                        task.result.failure_reason = "键盘注入失败"
                        all_results.append(task.result)
                        log.error("  → 环境异常(注入失败)")
                        continue

                    # 等待完成(持续监控进程存活、日志增量与终态标记)
                    outcome, _line = session.wait_task(phase_timeout)
                    task.result.end_time = time.time()
                    task.result.elapsed = task.result.end_time - task.result.start_time

                    # 截图:任务完成后
                    shot_after = (
                        report_dir / "screenshots" / f"{task.task_id}_after.png"
                    )
                    session.monitor.screenshot(str(shot_after))

                    if outcome == "done":
                        try:
                            ok, actual = task.validate()
                        except Exception as exception:
                            log.error(
                                "  validate 异常: %s",
                                type(exception).__name__,
                            )
                            ok, actual = False, f"验证器异常{type(exception).__name__}"
                        task.result.actual = actual
                        task.result.status = Status.PASS if ok else Status.FAIL
                        if not ok:
                            task.result.failure_reason = actual
                        log.info(
                            "  → %s (%.1fs): %s",
                            task.result.status.value,
                            task.result.elapsed,
                            actual,
                        )
                    else:
                        if outcome == "timeout":
                            task.result.status = Status.TIMEOUT
                            task.result.failure_reason = (
                                f"等待{phase_timeout:.0f}秒超时,强制中断"
                            )
                        elif outcome == "stalled":
                            task.result.status = Status.TIMEOUT
                            task.result.failure_reason = (
                                f"日志无进展达{STALL_TIMEOUT:.0f}秒,判定卡死并强制中断"
                            )
                        else:
                            task.result.status = Status.ENV_ERROR
                            task.result.failure_reason = "Agent CLI 进程中途退出"
                        log.warning(
                            "  → %s (%.1fs): %s",
                            task.result.status.value,
                            task.result.elapsed,
                            task.result.failure_reason,
                        )
                        # 果断中断:卡死/超时的 CLI 不再复用,按 PID 树终止并重启,
                        # 避免下一任务的指令注入到仍在执行旧任务的进程;
                        # M/H 阶段重启时保留 25 步配置。
                        restart_args = (
                            ["--max-steps", str(MAX_STEPS_MEDIUM_HIGH)]
                            if phase_name == "MH"
                            else None
                        )
                        session.stop()
                        if not session.start(restart_args):
                            log.error("  中断后会话重启失败,交由下轮 ensure_alive 重试")

                    # 清理本任务(清理异常不得中断后续任务)
                    try:
                        task.cleanup()
                    except Exception as exception:
                        log.error(
                            "  cleanup 异常: %s",
                            type(exception).__name__,
                        )
                    all_results.append(task.result)

                    # 任务间等待
                    time.sleep(3)

    except KeyboardInterrupt:
        log.warning("用户中断测试。")
    finally:
        session.stop()
        fixture.stop()
        restore_benchmark_desktop_state(session.monitor)

    # 生成报告(含 trace 指标聚合)
    trace_metrics = aggregate_trace_metrics(run_started_at)
    gen_report(all_results, run_id, report_dir, trace_metrics)

    # 清除桌面测试目录
    if desktop_dir.exists():
        shutil.rmtree(desktop_dir, ignore_errors=True)
        log.info("桌面测试目录已清除: %s", desktop_dir)

    log.info("测试完成。报告: %s/测试报告.md", report_dir)


# =========================================================================
# CLI
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="桌面 GUI 智能体系统测试平台",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="只运行指定任务(如S01,M03,H02),逗号分隔",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="每类任务运行次数(默认1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="固定随机种子以复现",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=None,
        help="统一覆盖单任务超时秒数(默认:S类480,M/H类720)",
    )
    parser.add_argument(
        "--agent-protocol",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="子 CLI 决策协议版本(A/B 对比用)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清除所有测试痕迹",
    )
    args = parser.parse_args()

    if args.clean:
        clean_all()
        return

    task_ids = None
    if args.task:
        task_ids = [t.strip().upper() for t in args.task.split(",")]

    run_benchmark(
        task_ids=task_ids,
        runs=args.runs,
        seed=args.seed,
        task_timeout=args.task_timeout,
        agent_protocol=args.agent_protocol,
    )


if __name__ == "__main__":
    main()
