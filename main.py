"""提供桌面 GUI 智能体的命令行入口。

本模块负责解析运行参数、配置日志、延迟组装运行依赖，并实现欢迎语、退出
说明、任务输入、步骤动作和成功或失败结果展示。

模型、截图和控制器只在创建运行实例时初始化。具体 Agent、模型和控制器采用
函数内延迟导入，以保持模块导入阶段无桌面和模型副作用。
"""

import argparse
import asyncio
import json
import logging
import os
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from agentscope.message import Msg

from agent.action_parser import ParsedAction
from config import (
    DEFAULT_MAX_STEPS,
    DEFAULT_RETRY_COUNT,
    AppConfig,
    GuiAgentSettings,
    api_enable_thinking_from_env,
    api_model_from_env,
    api_thinking_budget_from_env,
    api_thinking_options_supported_from_env,
    coordinate_mode_from_env,
    decision_protocol_v2_from_env,
    decision_protocol_v3_from_env,
    hide_own_window_during_run_from_env,
    local_model_dir_from_env,
    local_runtime_from_env,
    openvino_model_dir_from_env,
    semantic_execution_from_env,
    trace_enabled_from_env,
)
from perception.screenshot import (
    activate_window,
    get_foreground_app_hwnd,
    get_window_process_name,
    is_window_existing,
    list_visible_windows_zorder,
    minimize_window,
)
from utils.logger import setup_logging

# PRD 4.4.2 规定的命令行交互文案。
WELCOME_MESSAGE = "欢迎使用桌面GUI智能体！"
EXIT_INSTRUCTION = "输入'exit'退出程序"
EXIT_MESSAGE = "程序已退出。"
LOCAL_MODEL_CONFIGURATION_MESSAGE = (
    "local 模式需要通过 GUI_AGENT_LOCAL_MODEL_DIR 配置本地模型目录。"
)
OPENVINO_MODEL_CONFIGURATION_MESSAGE = (
    "openvino 运行时需要通过 GUI_AGENT_OPENVINO_MODEL_DIR 配置导出模型目录。"
)


class _Agent(Protocol):
    """描述命令行入口调用智能体所需的最小异步接口。

    运行时使用 ``GuiAgent``；测试可以注入不产生外部副作用的替代实现。
    """

    async def __call__(self, msg: Msg) -> Msg:
        """执行一项用户任务。"""


ActionObserver = Callable[[int, ParsedAction], None]

# 命中相对指代时，CLI 从前台窗口时间线中解析用户提交任务前的目标窗口。
_RELATIVE_REFERENCE_KEYWORDS = (
    "当前窗口",
    "当前应用",
    "当前浏览器",
    "当前网页",
)
_BROWSER_PROCESS_NAMES = frozenset(
    {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"}
)


def _has_relative_window_reference(task: str) -> bool:
    """判断任务是否以相对指代引用提交前的业务窗口。"""
    if any(keyword in task for keyword in _RELATIVE_REFERENCE_KEYWORDS):
        return True
    return "窗口" in task and any(marker in task for marker in ("这个", "那个", "该"))


def start_foreground_timeline() -> tuple[deque, threading.Event]:
    """启动轻量前台窗口时间线监听并记录最近八次窗口变化。

    用户提交任务时可据此回溯 CLI 聚焦前的业务窗口。监听线程为守护线程，
    每 0.5 秒读取一次前台窗口，不产生桌面控制副作用。
    """
    timeline: deque = deque(maxlen=8)
    stop = threading.Event()

    def _poll() -> None:
        """轮询前台窗口变化，直到收到停止信号。"""
        last_seen = 0
        while not stop.is_set():
            hwnd = get_foreground_app_hwnd()
            if hwnd and hwnd != last_seen:
                process = get_window_process_name(hwnd) or "unknown"
                timeline.append((hwnd, process))
                last_seen = hwnd
            stop.wait(0.5)

    threading.Thread(target=_poll, daemon=True).start()
    return timeline, stop


def resolve_task_target_window(
    timeline: deque,
    required_processes: frozenset[str] | None = None,
) -> dict[str, object] | None:
    """按可选进程类别回溯最近业务窗口；必要时检查当前可见窗口。"""
    current_fg = get_foreground_app_hwnd()
    for hwnd, process in reversed(list(timeline)):
        process_name = str(process).lower()
        if (
            hwnd != current_fg
            and is_window_existing(hwnd)
            and (required_processes is None or process_name in required_processes)
        ):
            return {"hwnd": hwnd, "process": process}
    if required_processes is not None:
        for window in list_visible_windows_zorder():
            hwnd = int(cast(int, window["hwnd"]))
            process = str(window.get("process") or "")
            if (
                hwnd != current_fg
                and process.lower() in required_processes
                and is_window_existing(hwnd)
            ):
                return {"hwnd": hwnd, "process": process}
    return None


class _AgentFactory(Protocol):
    """定义命令行入口创建智能体所需的最小工厂接口。"""

    def __call__(
        self,
        config: AppConfig,
        action_observer: ActionObserver,
    ) -> _Agent:
        """按配置和展示回调创建智能体。"""


class _UnavailableLocalBackend:
    """防止 API 模式的依赖组装意外进入本地模型路径。"""

    def generate(self, image: object, prompt: str) -> str:
        """任何实际调用都明确报告本地模型未配置。"""
        raise RuntimeError(LOCAL_MODEL_CONFIGURATION_MESSAGE)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器，不执行模型、截图或控制操作。"""
    parser = argparse.ArgumentParser(description="Desktop GUI Agent")
    parser.add_argument(
        "--model-mode",
        choices=("local", "api"),
        default="local",
        help="选择本地模型或现有 DashScope API。默认：local。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=(
            f"单任务基础逻辑步数上限；满足可测推进条件时最多扩展 3 步。"
            f"默认：{DEFAULT_MAX_STEPS}。"
        ),
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=DEFAULT_RETRY_COUNT,
        help=("单步动作执行失败后的最大重试次数。" f"默认：{DEFAULT_RETRY_COUNT}。"),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="标准库 logging 级别。默认：INFO。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="日志目录。默认：./logs。",
    )
    return parser


def config_from_arguments(arguments: argparse.Namespace) -> AppConfig:
    """把已解析命令行参数转换为经过验证的项目配置。

    Args:
        arguments: ``create_argument_parser`` 产生的参数命名空间。

    Returns:
        可直接用于运行组件组装的不可变配置。

    Raises:
        TypeError: 参数字段类型不符合 ``AppConfig`` 合同。
        ValueError: 参数字段值超出允许范围。
    """
    return AppConfig(
        model_mode=arguments.model_mode,
        max_steps=arguments.max_steps,
        retry_count=arguments.retry_count,
        log_level=arguments.log_level,
        log_dir=arguments.log_dir,
        local_model_dir=local_model_dir_from_env(),
        coordinate_mode=coordinate_mode_from_env(),
        local_runtime=local_runtime_from_env(),
        openvino_model_dir=openvino_model_dir_from_env(),
        api_model=api_model_from_env(),
        api_enable_thinking=api_enable_thinking_from_env(),
        api_thinking_budget=api_thinking_budget_from_env(),
        api_thinking_options_supported=api_thinking_options_supported_from_env(),
    )


def build_production_agent(
    config: AppConfig,
    action_observer: ActionObserver,
) -> _Agent:
    """按显式配置延迟构造生产运行依赖链。

    Args:
        config: 已验证的运行配置。
        action_observer: 接收已解析动作的只读 CLI 展示回调。

    Returns:
        已连接截图、模型、解析器、分发器和任务管理器的智能体。

    Raises:
        ValueError: 本地模式缺少模型目录等必要配置。
    """
    from agent.action_dispatcher import ActionDispatcher
    from agent.dashscope_api_backend import DashScopeAPIBackend
    from agent.diagnostics import ActionDiagnosticsWriter
    from agent.gui_agent import GuiAgent, GuiAgentDependencies
    from agent.model_client import ModelBackend, ModelClient, Qwen2VLLocalBackend
    from control.keyboard_controller import KeyboardController
    from control.mouse_controller import MouseController

    # 未显式配置时，程序化完成验证跟随 V3 协议启用；显式配置始终优先。
    protocol_v3 = decision_protocol_v3_from_env()
    semantic_execution = semantic_execution_from_env()
    if semantic_execution is None:
        semantic_execution = protocol_v3

    local_backend: ModelBackend
    if config.model_mode == "local" and config.local_runtime == "openvino":
        from agent.openvino_backend import Qwen2VLOpenVINOBackend

        if config.openvino_model_dir is None:
            raise ValueError(OPENVINO_MODEL_CONFIGURATION_MESSAGE)
        local_backend = Qwen2VLOpenVINOBackend(config.openvino_model_dir)
    elif config.model_mode == "local":
        if config.local_model_dir is None:
            raise ValueError(LOCAL_MODEL_CONFIGURATION_MESSAGE)
        local_backend = Qwen2VLLocalBackend(config.local_model_dir)
    else:
        local_backend = _UnavailableLocalBackend()

    api_backend = DashScopeAPIBackend.from_env(
        model=config.api_model,
        enable_thinking=config.api_enable_thinking,
        thinking_budget=config.api_thinking_budget,
        supports_thinking_options=config.api_thinking_options_supported,
    )
    # 本地模型加载失败时保留 PRD 规定的 API 回退能力。
    model_client = ModelClient(local_backend, api_backend, fallback_enabled=True)
    # 动作分发权限由 GuiAgent 为每个任务创建独立 PermissionScope。
    dispatcher = ActionDispatcher(
        MouseController(),
        KeyboardController(),
        coordinate_mode=config.coordinate_mode,
    )
    from perception.ocr_recognizer import OCRRecognizer

    trace_writer = None
    if trace_enabled_from_env():
        from agent.agent_trace import AgentTraceWriter

        trace_writer = AgentTraceWriter(config.log_dir)
    dependencies = GuiAgentDependencies(
        model_client=model_client,
        action_dispatcher=dispatcher,
        action_observer=action_observer,
        protect_initial_foreground=True,
        ocr_recognizer=OCRRecognizer(),
        diagnostics_writer=ActionDiagnosticsWriter(config.log_dir),
        trace_writer=trace_writer,
    )
    settings = GuiAgentSettings(
        max_steps=config.max_steps,
        retry_count=config.retry_count,
        model_mode=config.model_mode,
        coordinate_mode=config.coordinate_mode,
        decision_protocol_v2=decision_protocol_v2_from_env(),
        decision_protocol_v3=protocol_v3,
        hide_own_window_during_run=hide_own_window_during_run_from_env(),
        semantic_execution=semantic_execution,
    )
    return GuiAgent(dependencies, settings)


async def run_cli(
    config: AppConfig,
    *,
    agent_factory: _AgentFactory = build_production_agent,
    read_input: Callable[[str], str] = input,
    write_output: Callable[[str], None] = print,
) -> int:
    """运行命令行交互循环并实时展示已验证动作。

    Args:
        config: 运行配置。
        agent_factory: 延迟构造智能体的工厂。
        read_input: 用户输入函数。
        write_output: CLI 文本输出函数。

    Returns:
        用户正常退出时返回 0。

    交互终端可以显示当前动作；持久化业务日志仍只记录固定诊断事件，不复用
    用户输入、动作参数或模型原文。
    """
    write_output("正在初始化感知模块与模型后端...")
    try:
        agent = agent_factory(config, _action_observer(write_output))
    except Exception as exception:
        write_output(f"初始化失败：{type(exception).__name__}")
        write_output("请检查模型目录与依赖配置后重试。")
        return 1

    _clear_screen()
    write_output(WELCOME_MESSAGE)
    write_output(EXIT_INSTRUCTION)
    # 初始化完成后写入就绪标记，供外部测试框架判断 CLI 可以接收任务。
    logging.getLogger(__name__).info("cli_ready")
    timeline, stop_watcher = start_foreground_timeline()
    try:
        return await _cli_loop(
            config,
            agent_factory,
            _CliIO(read_input, write_output),
            timeline,
            agent,
        )
    finally:
        stop_watcher.set()


@dataclass(frozen=True)
class _CliIO:
    """保存可注入的 CLI 输入与输出函数。"""

    read_input: Callable[[str], str]
    write_output: Callable[[str], None]


async def _cli_loop(
    config: AppConfig,
    agent_factory: _AgentFactory,
    console: _CliIO,
    timeline: deque,
    agent: _Agent | None = None,
) -> int:
    """运行指令输入循环，并在需要时解析相对窗口指代。"""
    while True:
        try:
            task = console.read_input("请输入指令：")
        except EOFError:
            task = "exit"
        if task.strip().lower() == "exit":
            console.write_output(EXIT_MESSAGE)
            return 0
        if not task.strip():
            console.write_output("任务不能为空。")
            continue
        if agent is None:
            agent = agent_factory(
                config,
                _action_observer(console.write_output),
            )
        logging.getLogger(__name__).info("cli_task_received")
        trace_task_id = os.environ.get("GUI_AGENT_TRACE_TASK_ID", "").strip()
        metadata: dict[str, object] | None = (
            {"trace_task_id": trace_task_id} if trace_task_id else None
        )
        agent_ui_hwnd = get_foreground_app_hwnd()
        if _has_relative_window_reference(task):
            required_processes = (
                _BROWSER_PROCESS_NAMES
                if "当前浏览器" in task or "当前网页" in task
                else None
            )
            target = resolve_task_target_window(timeline, required_processes)
            if target is not None:
                if metadata is None:
                    metadata = {}
                metadata.update(
                    {
                        "task_target_window": target,
                        "agent_ui_window_hwnd": agent_ui_hwnd,
                    },
                )
                # 相对指代任务提交后最小化 CLI 并恢复目标窗口，避免遮挡截图。
                minimize_window(agent_ui_hwnd)
                activate_window(int(cast(int, target["hwnd"])))
        try:
            result = await agent(
                Msg(
                    "user",
                    task,
                    "user",
                    metadata=metadata,  # type: ignore[arg-type]  # AgentScope JSON 类型
                ),
            )
        finally:
            if metadata is not None:
                activate_window(agent_ui_hwnd)
        if isinstance(result.content, str) and result.content.startswith(
            "任务执行失败："
        ):
            # GuiAgent 已包含失败前缀，直接输出避免重复包装。
            console.write_output(result.content)
        else:
            console.write_output(f"任务执行成功：{result.content}")
        statistics = _format_statistics(getattr(result, "metadata", None))
        if statistics:
            console.write_output(statistics)


def _clear_screen() -> None:
    """清空终端屏幕；系统命令失败时由调用环境自行容忍。"""
    import sys

    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")


def _format_statistics(metadata: object) -> str:
    """把最终消息 metadata 中的任务统计格式化为 CLI 单行输出。"""
    if not isinstance(metadata, dict) or not metadata:
        return ""
    parts = []
    for key, label in (
        ("steps", "步数"),
        ("duration_seconds", "耗时(秒)"),
        ("retries", "重试"),
    ):
        if key in metadata:
            parts.append(f"{label}={metadata[key]}")
    return f"[统计] {' '.join(parts)}" if parts else ""


def _action_observer(
    write_output: Callable[[str], None],
) -> ActionObserver:
    """创建实时步骤展示回调。"""

    def _display(step_number: int, action: ParsedAction) -> None:
        """按命令行展示格式输出一个已验证动作。"""
        write_output(
            f"[步骤{step_number}] 执行动作：{_format_action(action)}",
        )

    return _display


def _format_action(action: ParsedAction) -> str:
    """把已验证动作还原为 CLI 可读格式，不用于持久化业务日志。

    必须显式覆盖全部动作类型。新增动作时需要同步补齐分支，避免未知动作
    静默落入错误的参数访问路径。
    """
    if action["action_type"] == "click":
        return f"click(x={action['params']['x']}, y={action['params']['y']})"
    if action["action_type"] == "right_click":
        return f"right_click(x={action['params']['x']}, y={action['params']['y']})"
    if action["action_type"] == "double_click":
        return f"double_click(x={action['params']['x']}, y={action['params']['y']})"
    if action["action_type"] == "drag":
        return (
            f"drag(x1={action['params']['x1']}, y1={action['params']['y1']}, "
            f"x2={action['params']['x2']}, y2={action['params']['y2']})"
        )
    if action["action_type"] == "type":
        text = json.dumps(action["params"]["text"], ensure_ascii=False)
        return f"type(text={text})"
    if action["action_type"] == "scroll":
        direction = json.dumps(action["params"]["direction"], ensure_ascii=False)
        steps = action["params"]["steps"]
        return f"scroll(direction={direction}, steps={steps})"
    if action["action_type"] == "hotkey":
        items = ", ".join(
            f"key{index}={json.dumps(key, ensure_ascii=False)}"
            for index, key in enumerate(action["params"]["keys"], start=1)
        )
        return f"hotkey({items})"
    if action["action_type"] == "finish":
        result = json.dumps(action["params"]["result"], ensure_ascii=False)
        return f"finish(result={result})"
    raise ValueError(f"不支持的动作类型：{action['action_type']}")


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数、配置日志并启动 CLI。

    Args:
        argv: 可选命令行参数；None 时读取当前进程参数。

    Returns:
        CLI 正常退出返回 0，参数或配置错误返回 argparse 的非零状态。
    """
    parser = create_argument_parser()
    arguments = parser.parse_args(argv)
    config = config_from_arguments(arguments)
    setup_logging(config.log_dir, config.log_level)
    try:
        return asyncio.run(run_cli(config))
    except (TypeError, ValueError) as exception:
        logging.getLogger(__name__).error(
            "cli_configuration_failed：异常类型=%s",
            type(exception).__name__,
        )
        parser.error(str(exception))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
