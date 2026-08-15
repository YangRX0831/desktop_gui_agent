"""提供桌面 GUI 智能体的最小命令行 vertical-slice 入口。

职责：
    解析运行参数、配置日志、延迟组装 production 依赖，并实现 PRD 4.4.2
    的欢迎语、退出说明、任务输入、步骤动作和成功/失败结果展示。

构造约束：
    production 后端只在收到第一项非空任务后创建；请求 exit、空输入或仅
    解析参数不会加载模型、截图、创建控制器或读取桌面。

展示与日志：
    CLI 是用户主动查看的交互界面，按 PRD 显示已验证动作的完整参数。
    logging 是持久化诊断通道，只记录固定脱敏事件，不能复用 CLI 正文。

依赖方向：
    具体 Agent、模型和控制器采用函数内延迟导入，目的仅是保持无副作用
    CLI 导入边界，不作为掩盖循环依赖的手段。
"""

import argparse
import asyncio
import json
import logging
import os
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from agentscope.message import Msg

from agent.action_parser import ParsedAction
from config import (
    DEFAULT_MAX_STEPS,
    DEFAULT_RETRY_COUNT,
    AppConfig,
    GuiAgentSettings,
    coordinate_mode_from_env,
    local_model_dir_from_env,
    local_runtime_from_env,
    openvino_model_dir_from_env,
)
from perception.screenshot import (
    activate_window,
    get_foreground_app_hwnd,
    get_window_process_name,
    is_window_existing,
    minimize_window,
)
from utils.logger import setup_logging

# PRD 4.4.2 交互示例逐字文案。欢迎使用桌面GUI智能体！输入'exit'退出程序。
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
    """描述 CLI 调用 AgentScope 智能体的最小异步接口。

    Attributes:
        具体任务状态由 Agent 实现私有保存，CLI 不读取内部属性。

    production 使用 ``GuiAgent``，测试可注入只产生内存消息的 fake。
    """

    async def __call__(self, msg: Msg) -> Msg:
        """执行一项用户任务。"""


ActionObserver = Callable[[int, ParsedAction], None]

# 相对指代词表:命中时 CLI 从前台时间线解析任务目标窗口。这是通用
# 语言指代解析,不是应用名或任务答案硬编码。
_RELATIVE_REFERENCE_KEYWORDS = ("当前窗口", "当前应用")


def start_foreground_timeline() -> tuple[deque, threading.Event]:
    """启动轻量前台时间线监听,记录窗口变化(去重,最近8个)。

    用户提交任务时据此回溯CLI聚焦前的业务窗口;线程为daemon,轮询
    间隔0.5秒,不产生桌面副作用。
    """
    timeline: deque = deque(maxlen=8)
    stop = threading.Event()

    def _poll() -> None:
        """轮询前台变化并追加时间线,直到收到停止信号。"""
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


def resolve_task_target_window(timeline: deque) -> dict[str, object] | None:
    """回溯时间线中最近一个非当前前台的仍存在窗口作为任务目标。"""
    current_fg = get_foreground_app_hwnd()
    for hwnd, process in reversed(list(timeline)):
        if hwnd != current_fg and is_window_existing(hwnd):
            return {"hwnd": hwnd, "process": process}
    return None


class _AgentFactory(Protocol):
    """定义 CLI 创建 production Agent 的最小合同。"""

    def __call__(
        self,
        config: AppConfig,
        action_observer: ActionObserver,
    ) -> _Agent:
        """按配置和展示回调创建 Agent。"""


class _UnavailableLocalBackend:
    """阻止 api-only production wiring 意外进入本地模型路径。

    Attributes:
        本类无可变状态，只实现 ModelBackend 的 generate 形状。

    当 CLI 选择 API 模式时作为占位依赖；任何实际调用都明确失败。
    """

    def generate(self, image: object, prompt: str) -> str:
        """防止 api-only wiring 意外进入本地模型路径。"""
        raise RuntimeError(LOCAL_MODEL_CONFIGURATION_MESSAGE)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建不执行模型、截图或控制操作的 CLI 参数解析器。

    Returns:
        包含模型模式、最大步数、格式重试和日志选项的参数解析器。
    """
    parser = argparse.ArgumentParser(description="Desktop GUI Agent W3 demo")
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
        help=f"单任务最大模型轮次。默认：{DEFAULT_MAX_STEPS}。",
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
    """把已解析 CLI 参数转换为经过验证的项目配置。

    Args:
        arguments: ``create_argument_parser`` 产生的参数命名空间。

    Returns:
        可直接用于 production wiring 的不可变配置。

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
    )


def build_production_agent(
    config: AppConfig,
    action_observer: ActionObserver,
) -> _Agent:
    """按显式配置延迟构造唯一 production 依赖链。

    Args:
        config: 已验证的 production 配置。
        action_observer: 接收已解析动作的只读 CLI 展示回调。

    Returns:
        已连接截图、模型、Parser、Dispatcher 和 TaskManager 的 Agent。

    Raises:
        ValueError: 本地模式缺少模型目录等必要配置。
    """
    from agent.action_dispatcher import ActionDispatcher
    from agent.dashscope_api_backend import DashScopeAPIBackend
    from agent.gui_agent import GuiAgent, GuiAgentDependencies
    from agent.model_client import ModelBackend, ModelClient, Qwen2VLLocalBackend
    from control.keyboard_controller import KeyboardController
    from control.mouse_controller import MouseController

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

    api_backend = DashScopeAPIBackend.from_env()
    # fallback_enabled 表示保留 PRD model-load-failure -> API capability;
    # 真正的远程发送授权由每个 run 显式设置(run-scoped, non-persistent)。
    model_client = ModelClient(local_backend, api_backend, fallback_enabled=True)
    # ActionDispatcher 的动作授权由 GuiAgent 每个 run 新建 PermissionScope 注入,
    # production wiring 不再使用静态 allow-all authorizer。
    dispatcher = ActionDispatcher(
        MouseController(),
        KeyboardController(),
        coordinate_mode=config.coordinate_mode,
    )
    from perception.ocr_recognizer import OCRRecognizer

    dependencies = GuiAgentDependencies(
        model_client=model_client,
        action_dispatcher=dispatcher,
        action_observer=action_observer,
        protect_initial_foreground=True,
        ocr_recognizer=OCRRecognizer(),
    )
    settings = GuiAgentSettings(
        max_steps=config.max_steps,
        retry_count=config.retry_count,
        model_mode=config.model_mode,
        coordinate_mode=config.coordinate_mode,
    )
    return GuiAgent(dependencies, settings)


async def run_cli(
    config: AppConfig,
    *,
    agent_factory: _AgentFactory = build_production_agent,
    read_input: Callable[[str], str] = input,
    write_output: Callable[[str], None] = print,
) -> int:
    """运行 PRD 4.4.2 命令行交互循环并实时展示动作。

    Args:
        config: production 运行配置。
        agent_factory: 延迟构造 Agent 的工厂。
        read_input: 用户输入函数。
        write_output: CLI 文本输出函数。

    Returns:
        用户正常退出时返回 0。

    交互终端是用户主动查看的任务界面，可以显示当前动作；业务日志仍只
    记录脱敏事件，不记录动作参数、用户输入或模型原文。
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
    timeline, stop_watcher = start_foreground_timeline()
    try:
        return await _cli_loop(
            config,
            agent_factory,
            read_input,
            write_output,
            timeline,
            agent,
        )
    finally:
        stop_watcher.set()


async def _cli_loop(
    config: AppConfig,
    agent_factory: _AgentFactory,
    read_input: Callable[[str], str],
    write_output: Callable[[str], None],
    timeline: deque,
    agent: _Agent | None = None,
) -> int:
    """运行指令输入循环;含相对指代时解析目标窗口并管理CLI前后台。"""
    while True:
        try:
            task = read_input("请输入指令：")
        except EOFError:
            task = "exit"
        if task.strip().lower() == "exit":
            write_output(EXIT_MESSAGE)
            return 0
        if not task.strip():
            write_output("任务不能为空。")
            continue
        if agent is None:
            agent = agent_factory(config, _action_observer(write_output))
        logging.getLogger(__name__).info("cli_task_received")
        metadata: dict[str, object] | None = None
        agent_ui_hwnd = get_foreground_app_hwnd()
        if any(keyword in task for keyword in _RELATIVE_REFERENCE_KEYWORDS):
            target = resolve_task_target_window(timeline)
            if target is not None:
                metadata = {"task_target_window": target}
                # 提交含相对指代任务时,最小化CLI并恢复目标窗口到前台,
                # 避免CLI遮挡截图与目标歧义;失败均容忍。
                minimize_window(agent_ui_hwnd)
                activate_window(int(target["hwnd"]))
        try:
            result = await agent(Msg("user", task, "user", metadata=metadata))
        finally:
            if metadata is not None:
                activate_window(agent_ui_hwnd)
        if isinstance(result.content, str) and result.content.startswith(
            "任务执行失败："
        ):
            # result.content 已由 GuiAgent 格式化为 "任务执行失败：<reason>",
            # 直接输出避免重复前缀;PRD 4.4.2 只给出 success literal。
            write_output(result.content)
        else:
            write_output(f"任务执行成功：{result.content}")
        statistics = _format_statistics(getattr(result, "metadata", None))
        if statistics:
            write_output(statistics)


def _clear_screen() -> None:
    """清空终端屏幕;平台差异由系统命令处理,失败时容忍。"""
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
    """创建符合 PRD 示例的实时步骤展示回调。"""

    def _display(step_number: int, action: ParsedAction) -> None:
        """按 PRD 4.4.2 示例格式显示一个已验证动作。"""
        write_output(
            f"[步骤{step_number}] 执行动作：{_format_action(action)}",
        )

    return _display


def _format_action(action: ParsedAction) -> str:
    """把已验证动作还原为 CLI 可读格式，不用于业务日志。

    必须显式覆盖全部八种动作，未知动作类型显式抛 ValueError；新增动作
    类型时同步补齐分支，防止静默落到错误的参数访问。
    """
    action_type = action["action_type"]
    params = action["params"]
    if action_type == "click":
        return f"click(x={params['x']}, y={params['y']})"
    if action_type == "right_click":
        return f"right_click(x={params['x']}, y={params['y']})"
    if action_type == "double_click":
        return f"double_click(x={params['x']}, y={params['y']})"
    if action_type == "drag":
        return (
            f"drag(x1={params['x1']}, y1={params['y1']}, "
            f"x2={params['x2']}, y2={params['y2']})"
        )
    if action_type == "type":
        text = json.dumps(params["text"], ensure_ascii=False)
        return f"type(text={text})"
    if action_type == "scroll":
        direction = json.dumps(params["direction"], ensure_ascii=False)
        return f"scroll(direction={direction}, steps={params['steps']})"
    if action_type == "hotkey":
        items = ", ".join(
            f"key{index}={json.dumps(key, ensure_ascii=False)}"
            for index, key in enumerate(params["keys"], start=1)
        )
        return f"hotkey({items})"
    if action_type == "finish":
        result = json.dumps(params["result"], ensure_ascii=False)
        return f"finish(result={result})"
    raise ValueError(f"不支持的动作类型：{action_type}")


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
