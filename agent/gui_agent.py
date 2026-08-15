"""协调桌面 GUI 智能体的单一感知、规划与执行循环。

职责：
    按 PRD 4.4.1 串行完成截图、模型生成、严格解析、白名单分发、结果记录
    和下一轮感知。每个任务有明确的最大步骤数，不创建后台循环。

重试约束：
    一个 logical step 内的 retry 按 PRD 4.5.1 + AGENTS 副作用安全组合实现:
    initial attempt 失败 → 等待 → fresh screenshot → fresh model decision →
    fresh parse → fresh dispatch。这才算一次 retry;严禁在同一截图与同一
    ParsedAction 上机械 replay。``retry_count`` 是 PRD 单步最大重试次数
    (默认 3),即 initial attempt 后最多 ``retry_count`` 次 fresh-observation
    retry;该步 retry 耗尽后按 PRD 记录错误并继续下一 logical step,不直接
    终止任务。API transport retry 完全由 ``ModelClient`` 管理,本编排层不在
    fresh retry 中乘法放大。

失败语义：
    截图失败和已由 ModelClient 耗尽的模型失败无法产生可靠新状态,因此结束
    当前任务。Parser 失败结束当前 attempt,在 retry 预算内继续 fresh
    observation;步 retry 耗尽记录该步失败并进入下一 logical step。
    ``max_steps`` 限制 logical step 总数。

安全边界：
    本模块不直接操作鼠标键盘，不读取模型私有状态，也不记录任务、prompt
    或模型原文。CLI observer 只接收已解析动作，观察失败不能改变任务结果。
"""

import itertools
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol, cast

from agentscope.agent import AgentBase
from agentscope.message import Msg
from PIL import Image

from agent.action_dispatcher import ActionDispatcher, PermissionScope
from agent.action_parser import (
    ActionPromptState,
    FinishParams,
    ParsedAction,
    PromptEffect,
    PromptProgress,
    classify_action_parse_error,
    compose_action_prompt,
    parse_action,
)
from agent.dashscope_api_backend import DASHSCOPE_CONFIGURATION_MESSAGE
from agent.task_manager import TaskManager
from config import GuiAgentSettings, model_image_max_dim_from_env
from perception import prompt_context
from perception.prompt_context import OCRRecognizerProtocol
from perception.screenshot import (
    capture_screen,
    get_foreground_app_hwnd,
    get_foreground_hwnd,
    get_window_process_name,
    get_window_screen_rect,
    is_window_available,
    is_window_existing,
    select_capture_region,
)
from perception.ui_locator import frame_change_ratio, frames_stable
from utils.exceptions import ScreenCaptureError

logger = logging.getLogger(__name__)

_AGENT_NAME = "gui_agent"
_TERMINAL_MODEL_FAILURE_RESPONSES = {
    "API 模型调用失败。",
    DASHSCOPE_CONFIGURATION_MESSAGE,
}
_PREMATURE_FINISH_REASON = "尚未成功执行GUI动作。"
_UNRESOLVED_FAILURE_FINISH_REASON = "上一动作失败且没有强可验证进展。"
_NO_UI_PROGRESS_REASON = "操作后未检测到界面变化。"
_ACTION_VERIFICATION_FAILURE_REASON = "操作结果验证失败。"
_MODEL_FAILURE_RESPONSES = _TERMINAL_MODEL_FAILURE_RESPONSES | {
    "本地模型调用失败。",
}
_MODEL_FAILURE_REASON = "模型调用失败。"
_TERMINAL_MODEL_FAILURE = "terminal_model_failure"
_PARSE_FAILURE_REASON = "模型动作解析失败。"
_CAPTURE_FAILURE_REASON = "屏幕截图失败。"
_TARGET_WINDOW_LOST_REASON = "任务目标窗口已关闭，已停止后续桌面操作。"
_DESKTOP_TYPE_REASON = "当前没有已聚焦的应用输入目标，不能直接输入文本。"
_PROTECTED_FOREGROUND_TYPE_REASON = (
    "当前聚焦的是本智能体自身的命令行界面，不能向它输入文本。"
)
_PROTECTED_FOREGROUND_FINISH_REASON = (
    "当前仍停留在本智能体自身的命令行界面，没有可验证的目标界面进展。"
)
_PROTECTED_FOREGROUND_CLOSE_REASON = (
    "当前聚焦的是本智能体自身的命令行界面，禁止对其使用关闭或系统菜单快捷键。"
)
_PROTECTED_FOREGROUND_CLICK_REASON = (
    "坐标落在本智能体自身的命令行界面内，该界面只读，请勿再在此区域执行鼠标动作。"
)
# 可在当前窗口触发关闭或系统菜单的常见组合键族;仅在受保护未解锁前台
# 拒绝,作用于其他窗口时不受限制。
_CLOSE_HOTKEY_SETS = (
    frozenset({"alt", "f4"}),
    frozenset({"alt", "space"}),
    frozenset({"ctrl", "w"}),
    frozenset({"ctrl", "shift", "w"}),
)
_REPEATED_TYPE_REASON = "相同文本已经分发，不能在没有新焦点的情况下重复输入。"
_SHELL_CLOSE_HOTKEY_REASON = "桌面环境禁止执行关闭窗口快捷键。"
_MAX_STEPS_REASON = "任务达到最大执行步数。"
_RETRY_DELAY_SECONDS = 1.0
# PRD 4.2.1 "操作之间添加适当延迟":每次桌面动作成功后,等待 UI 稳定再
# 截取下一帧,避免 capture 到 transition/loading/旧窗口状态。
_ACTION_STABILIZATION_SECONDS = 2.0
# 通用 UI 稳定检测(§7):动作后用帧差判断动画是否收敛,有界等待,避免在弹窗/菜单
# 动画过程中截下一帧。无 OpenCV 或截图失败时回退固定等待。不增加 logical step。
_STABILITY_INTERVAL = 0.3
_STABILITY_MAX_ITERS = 5
_ACTION_PROGRESS_DIFF_RATIO = 0.0001
_ACTION_PROGRESS_MIN_PIXELS = 25
_SHELL_CLICK_PROGRESS_DIFF_RATIO = 0.005
# 起始前台窗口内容因已分发动作发生该比例以上的帧差(页面导航、面板展开
# 量级)后，不再视为未触碰的只读任务输入界面；终端内点击产生的局部
# 高亮远低于该值，CLI 保护语义不受影响。
_INITIAL_WINDOW_UNLOCK_RATIO = 0.05
# 每个用户 task/run 提交后,视为对该 run 授予 PRD 基础桌面控制动作的执行权限。
# 这是一次性 run grant,每次 reply 新建独立 PermissionScope,不跨 run 继承。
_RUN_PERMISSION_ACTIONS = frozenset(
    {
        "click",
        "right_click",
        "double_click",
        "drag",
        "type",
        "scroll",
        "hotkey",
        "finish",
    },
)
_permission_token_counter = itertools.count(1)


class ModelClientProtocol(Protocol):
    """定义 GuiAgent 需要的最小模型调用合同。

    Attributes:
        实现自行保存后端和 transport retry 状态，GuiAgent 不读取这些状态。

    ``ModelClient`` 是 production 实现；测试可注入不产生网络调用的 fake。
    """

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        mode: Literal["local", "api"] = "local",
    ) -> str:
        """根据截图和提示词生成模型文本。"""


@dataclass(frozen=True)
class _ActionObservation:
    """保存动作后能够可靠观察到的桌面事实。"""

    succeeded: bool
    failure_reason: str | None
    after_foreground: int
    screen_changed: bool | None
    effect: PromptEffect
    change_ratio: float


@dataclass(frozen=True)
class _AttemptState:
    """内层循环某次 attempt 的只读上下文,供分支 handler 共享。

    封装 attempt 级可变循环状态,使 ``_handle_*`` 方法参数不超过 5 个;
    handler 通过 ``_AttemptOutcome`` 返回更新后的 ``prompt_state`` 与
    ``step_retries``,其余字段在分支内只读。
    """

    attempt: int
    step_retries: int
    prompt_state: ActionPromptState
    successful_action_count: int
    initial_foreground_unlocked: bool


@dataclass(frozen=True)
class _AttemptOutcome:
    """分支处理后的控制流与状态更新。

    flow 为 ``terminal`` 时 ``final_msg`` 非空,调用方应立即返回该消息;
    flow 为 ``retry`` 时调用方继续内层循环;flow 为 ``exhausted`` 时调用方
    跳出内层循环。``prompt_state`` 与 ``step_retries`` 始终为分支处理后的
    最新值,供调用方写回循环状态。
    """

    flow: Literal["terminal", "retry", "exhausted"]
    final_msg: Msg | None
    prompt_state: ActionPromptState
    step_retries: int


@dataclass(frozen=True)
class GuiAgentDependencies:
    """集中保存编排器依赖，保持构造接口简洁且便于隔离测试。

    Attributes:
        model_client: 负责模型调用与 transport retry 的唯一客户端。
        action_dispatcher: 负责白名单校验与控制调用的分发器。
        capture: 当前屏幕截图函数。
        task_manager_factory: 为每项用户任务创建状态管理器的工厂。
        action_observer: 可选的只读步骤展示回调。
        sleep: 注入的等待函数。
        protect_initial_foreground: 是否把任务开始时的前台窗口视为只读任务
            输入界面，阻止模型向其输入或在未离开时声明完成。
        ocr_recognizer: 可选 OCR 感知器；每步截图后识别文字并注入 Prompt
            的 Current perception 块。None 表示不启用 OCR 辅助感知。

    production wiring 由 ``main.build_production_agent`` 一次性组装这些依赖。
    """

    model_client: ModelClientProtocol
    action_dispatcher: ActionDispatcher
    capture: Callable[..., Image.Image] = capture_screen
    task_manager_factory: Callable[[str], TaskManager] = TaskManager
    action_observer: Callable[[int, ParsedAction], None] | None = None
    sleep: Callable[[float], None] = time.sleep
    protect_initial_foreground: bool = False
    ocr_recognizer: OCRRecognizerProtocol | None = None

    def __post_init__(self) -> None:
        """在构造 Agent 前验证所有注入边界。"""
        if not callable(getattr(self.model_client, "generate", None)):
            raise TypeError("model_client 必须提供可调用的 generate。")
        if not callable(getattr(self.action_dispatcher, "dispatch", None)):
            raise TypeError("action_dispatcher 必须提供可调用的 dispatch。")
        if not callable(self.capture):
            raise TypeError("capture 必须可调用。")
        if not callable(self.task_manager_factory):
            raise TypeError("task_manager_factory 必须可调用。")
        if self.action_observer is not None and not callable(self.action_observer):
            raise TypeError("action_observer 必须可调用或为 None。")
        if not callable(self.sleep):
            raise TypeError("sleep 必须可调用。")
        if type(self.protect_initial_foreground) is not bool:
            raise TypeError("protect_initial_foreground 必须是 bool。")
        if self.ocr_recognizer is not None and not callable(
            getattr(self.ocr_recognizer, "recognize", None),
        ):
            raise TypeError("ocr_recognizer 必须提供可调用的 recognize。")


class GuiAgent(AgentBase):
    """串行执行 PRD 定义的感知、模型、解析、控制和反馈闭环。

    Attributes:
        依赖和配置保持为私有状态，外部只通过 AgentScope 调用接口执行任务。

    典型用法是注入 ``GuiAgentDependencies`` 和 ``GuiAgentSettings``，然后
    向实例发送包含自然语言任务的 ``Msg``。模型 transport retry 完全归
    ``ModelClient`` 所有；本类的步内重试始终基于全新截图与模型决策，
    不重放同一 ParsedAction(重试约束见模块 docstring)，避免把两层上限
    相乘成多次 API 请求。
    """

    def __init__(
        self,
        dependencies: GuiAgentDependencies,
        settings: GuiAgentSettings | None = None,
    ) -> None:
        """初始化无副作用的桌面任务编排器。"""
        super().__init__()
        if not isinstance(dependencies, GuiAgentDependencies):
            raise TypeError("dependencies 必须是 GuiAgentDependencies。")
        if settings is not None and not isinstance(settings, GuiAgentSettings):
            raise TypeError("settings 必须是 GuiAgentSettings 或 None。")
        self._dependencies = dependencies
        self._settings = settings or GuiAgentSettings()
        # 一次性 local fallback 授权;调用方通过 authorize_next_run_fallback
        # 设置后,仅对下一个 run 生效,使用后自动清零,不跨 run 继承。
        self._next_run_fallback_authorized = False
        # Agent 自身控制界面窗口 HWND(任务提交时的前台);保护与截图基准用。
        self._agent_ui_hwnd = 0
        # 任务目标窗口绑定(id/process),由 CLI 从前台时间线解析后经 metadata 传入。
        self._task_target: dict[str, object] | None = None
        # 最近一次全屏截图,用于动作前后效果比较。
        self._latest_full: Image.Image | None = None

    def authorize_next_run_fallback(self) -> None:
        """为下一个 local run 授权一次 PRD model-load-failure -> API fallback。

        该授权是一次性的:仅对紧接的下一次 ``reply`` 生效,使用后或 run 结束
        后自动清零。默认 local run 不授权(deny-by-default);调用方需显式调用
        本方法才允许当前 local run 的 load-failure fallback。
        """
        self._next_run_fallback_authorized = True

    async def reply(self, msg: Msg) -> Msg:
        """执行一个桌面任务并返回最终结果消息。

        Args:
            msg: 包含非空用户任务文本的 AgentScope 用户消息。

        Returns:
            finish 结果或固定的安全失败消息。

        Raises:
            TypeError: 消息或消息内容类型错误。
            ValueError: 消息角色或任务文本不合法。
        """
        task = self._validate_task_message(msg)
        self._task_target = self._extract_task_target(msg)
        manager = self._dependencies.task_manager_factory(task)
        manager.start()
        logger.info("gui_agent_task_started")

        # 每个 run 新建独立的临时授权作用域;run 结束立即清空,不跨 run 继承。
        run_scope = PermissionScope(
            allowed_actions=_RUN_PERMISSION_ACTIONS,
            token=next(_permission_token_counter),
        )
        dispatcher = self._dependencies.action_dispatcher
        dispatcher.activate_run_scope(run_scope)
        # API 模式:当前 run 明确选择远程调用,授权 fallback。
        # local 模式:默认不授权远程 fallback(AGENTS §11.2 deny-by-default)。
        model_client = self._dependencies.model_client
        fallback_setter = getattr(
            model_client,
            "set_run_fallback_authorization",
            None,
        )
        fallback_clearer = getattr(
            model_client,
            "clear_run_fallback_authorization",
            None,
        )
        if callable(fallback_setter):
            # API 模式:当前 run 明确选择远程调用。
            # local 模式:若调用方通过 authorize_next_run_fallback 设置了
            # 一次性授权,则本次 run 允许 load-failure fallback;否则 deny。
            run_authorized = (
                self._settings.model_mode == "api" or self._next_run_fallback_authorized
            )
            fallback_setter(run_authorized)
            self._next_run_fallback_authorized = False
        try:
            return await self._run_task(task, manager)
        finally:
            dispatcher.clear_run_scope()
            if callable(fallback_clearer):
                fallback_clearer()

    async def _run_task(
        self,
        task: str,
        manager: TaskManager,
    ) -> Msg:
        """执行任务的主循环;调用前由 reply 完成 scope 激活,本方法不处理 scope。"""
        # agent_ui_hwnd 在整个 run 内固定;任务目标由 metadata 绑定,不随界面漂移。
        prompt_state = ActionPromptState(
            step_number=1,
            max_steps=self._settings.max_steps,
            task_target_window="none",
            agent_ui_window="none",
            platform=self._platform_name(),
        )
        # 应用 HWND 仅作为"发现阶段全屏、切换后 region"的截图基准。
        self._agent_ui_hwnd = get_foreground_app_hwnd()
        self._latest_full = None
        successful_action_count = 0
        # 起始前台窗口内容发生已分发动作引起的大幅变化后解除只读保护;
        # run 级状态,不跨任务继承。
        initial_foreground_unlocked = False
        for step_number in range(1, self._settings.max_steps + 1):
            step_succeeded = False
            step_retries = 0
            for attempt in range(self._settings.retry_count + 1):
                capture = self._capture(manager, step_number)
                if capture is None:
                    return self._failure_message(_CAPTURE_FAILURE_REASON, manager)
                image, region_offset = capture

                focus_kind = prompt_context.focus_control_state()
                target_window_state = prompt_context.task_target_window_state(
                    self._task_target,
                    image.size,
                    region_offset,
                )
                agent_window_state = prompt_context.agent_ui_window_state(
                    self._agent_ui_hwnd,
                    image.size,
                    region_offset,
                )
                windows_state = prompt_context.perceive_windows(
                    image.size,
                    region_offset,
                )
                action, failure = self._generate_action(
                    image,
                    task,
                    replace(
                        prompt_state,
                        step_number=step_number,
                        ocr_elements=prompt_context.perceive_ocr_elements(
                            self._dependencies.ocr_recognizer,
                            image,
                        ),
                        focused_control=focus_kind,
                        keyboard_input_ready=prompt_context.keyboard_input_ready(
                            focus_kind,
                            self._dependencies.protect_initial_foreground,
                            self._agent_ui_hwnd,
                            initial_foreground_unlocked,
                        ),
                        system_volume_percent=(prompt_context.system_volume_state()),
                        task_target_window=target_window_state,
                        agent_ui_window=agent_window_state,
                        windows=windows_state,
                    ),
                )
                if action is None:
                    # model/parse failure 属 PRD 4.5.1 retryable;记录每次
                    # attempt(stage + reason),不伪造 ParsedAction。
                    outcome = self._handle_model_failure(
                        failure,
                        _AttemptState(
                            attempt=attempt,
                            step_retries=step_retries,
                            prompt_state=prompt_state,
                            successful_action_count=successful_action_count,
                            initial_foreground_unlocked=initial_foreground_unlocked,
                        ),
                        manager,
                    )
                    if outcome.flow == "terminal":
                        return outcome.final_msg
                    prompt_state = outcome.prompt_state
                    step_retries = outcome.step_retries
                    if outcome.flow == "retry":
                        continue
                    break

                if action["action_type"] == "finish":
                    outcome = self._handle_finish_action(
                        action,
                        step_number,
                        _AttemptState(
                            attempt=attempt,
                            step_retries=step_retries,
                            prompt_state=prompt_state,
                            successful_action_count=successful_action_count,
                            initial_foreground_unlocked=initial_foreground_unlocked,
                        ),
                        manager,
                    )
                    if outcome.flow == "terminal":
                        return outcome.final_msg
                    prompt_state = outcome.prompt_state
                    step_retries = outcome.step_retries
                    if outcome.flow == "retry":
                        continue
                    break

                safety_failure = self._get_dispatch_safety_failure(
                    action,
                    prompt_state,
                    initial_foreground_unlocked,
                ) or self._protected_foreground_click_failure(
                    action,
                    image.size,
                    region_offset,
                    initial_foreground_unlocked,
                )
                if safety_failure is not None:
                    outcome = self._handle_safety_rejection(
                        action,
                        safety_failure,
                        _AttemptState(
                            attempt=attempt,
                            step_retries=step_retries,
                            prompt_state=prompt_state,
                            successful_action_count=successful_action_count,
                            initial_foreground_unlocked=initial_foreground_unlocked,
                        ),
                        manager,
                    )
                    if outcome.flow == "terminal":
                        return outcome.final_msg
                    prompt_state = outcome.prompt_state
                    step_retries = outcome.step_retries
                    if outcome.flow == "retry":
                        continue
                    break

                self._notify_action(step_number, action)
                before_foreground = get_foreground_hwnd()
                before_app_foreground = get_foreground_app_hwnd()
                before_full = None
                if self._settings.verify_action_effect:
                    before_full = (
                        self._latest_full.copy()
                        if self._latest_full is not None
                        else image.copy()
                    )
                dispatched = self._dependencies.action_dispatcher.dispatch(
                    action,
                    image.size,
                    region_offset,
                )
                if dispatched and self._settings.verify_action_effect:
                    assert before_full is not None
                    observation = self._verify_action_effect(
                        action["action_type"],
                        before_full,
                        before_foreground,
                        before_app_foreground,
                    )
                    stage = "verify"
                else:
                    observation = _ActionObservation(
                        succeeded=dispatched,
                        failure_reason=None,
                        after_foreground=get_foreground_hwnd(),
                        screen_changed=None,
                        effect="none",
                        change_ratio=0.0,
                    )
                    stage = "dispatch"
                manager.record_attempt(
                    action,
                    observation.succeeded,
                    attempt,
                    stage,
                    observation.failure_reason,
                )
                if (
                    not initial_foreground_unlocked
                    and observation.screen_changed
                    and observation.change_ratio >= _INITIAL_WINDOW_UNLOCK_RATIO
                    and observation.after_foreground == self._agent_ui_hwnd
                ):
                    initial_foreground_unlocked = True
                prompt_state = self._advance_prompt_state(
                    prompt_state,
                    action,
                    dispatched,
                    observation,
                )
                if observation.succeeded:
                    successful_action_count += 1
                    manager.finalize_step(True, retry_count=step_retries)
                    if step_retries:
                        manager.record_retry(step_retries)
                    step_succeeded = True
                    break

                if not dispatched:
                    prompt_state = replace(
                        prompt_state,
                        last_error="动作执行失败。",
                    )
                # 失败后重新观察并重新决策,绝不机械重放同一 ParsedAction。
                outcome = self._retry_or_exhausted(
                    attempt,
                    step_retries,
                    manager,
                    prompt_state,
                )
                step_retries = outcome.step_retries
                if outcome.flow == "retry":
                    continue
                break

            if step_succeeded:
                continue
            logger.warning(
                "gui_agent_step_retry_exhausted：step=%d，retries=%d",
                step_number,
                step_retries,
            )

        manager.fail(_MAX_STEPS_REASON)
        return self._failure_message(_MAX_STEPS_REASON, manager)

    def _handle_model_failure(
        self,
        failure: str | None,
        state: _AttemptState,
        manager: TaskManager,
    ) -> _AttemptOutcome:
        """处理模型调用或动作解析失败。

        terminal_model_failure(ModelClient 已耗尽 transport retry 或配置错误)
        结束当前任务,不在 Agent 层重开第二套重试预算;其余 model/parse
        failure 属 PRD 4.5.1 retryable,在 retry 预算内继续 fresh observation。
        """
        # failure 形如 _MODEL_FAILURE_REASON / _TERMINAL_MODEL_FAILURE /
        # f"{_PARSE_FAILURE_REASON} (category)";stage 与 recorded_failure 按
        # PRD 4.5.1 分类,不伪造 ParsedAction。
        stage = (
            "model"
            if failure in {_MODEL_FAILURE_REASON, _TERMINAL_MODEL_FAILURE}
            else "parse"
        )
        recorded_failure = (
            _MODEL_FAILURE_REASON if failure == _TERMINAL_MODEL_FAILURE else failure
        )
        manager.record_attempt(
            None,
            False,
            state.attempt,
            stage,
            recorded_failure,
        )
        prompt_state = replace(
            state.prompt_state,
            last_action="none",
            last_dispatch_status="none",
            last_error=recorded_failure or _MODEL_FAILURE_REASON,
        )
        if failure == _TERMINAL_MODEL_FAILURE:
            # ModelClient 已按 PRD 4.3.1 用尽 API 的 3 次重试，或确认
            # 配置错误；不得在 Agent 层重新开启第二套重试预算。
            manager.finalize_step(False, retry_count=state.step_retries)
            manager.fail(_MODEL_FAILURE_REASON)
            return _AttemptOutcome(
                "terminal",
                self._failure_message(_MODEL_FAILURE_REASON, manager),
                prompt_state,
                state.step_retries,
            )
        return self._retry_or_exhausted(
            state.attempt,
            state.step_retries,
            manager,
            prompt_state,
        )

    def _handle_finish_action(
        self,
        action: ParsedAction,
        step_number: int,
        state: _AttemptState,
        manager: TaskManager,
    ) -> _AttemptOutcome:
        """处理 finish 动作:拒绝过早完成或记录任务成功。

        未满足保护/过早/未解决失败条件时记录成功并返回任务结果消息;
        否则按 retryable 处理,在 retry 预算内继续 fresh observation。
        """
        prompt_state = state.prompt_state
        protected_finish = (
            self._dependencies.protect_initial_foreground
            and self._agent_ui_hwnd != 0
            and get_foreground_app_hwnd() == self._agent_ui_hwnd
            and not state.initial_foreground_unlocked
        )
        unresolved_failure = (
            self._settings.reject_initial_finish
            and prompt_state.last_dispatch_status == "failure"
            and prompt_state.ui_change_signal != "strong"
        )
        if not (
            protected_finish
            or (
                self._settings.reject_initial_finish
                and state.successful_action_count == 0
            )
            or unresolved_failure
        ):
            self._notify_action(step_number, action)
            result = cast(FinishParams, action["params"])["result"]
            manager.record_attempt(
                action,
                True,
                state.attempt,
                "finish",
            )
            manager.finalize_step(True, retry_count=state.step_retries)
            if state.step_retries:
                manager.record_retry(state.step_retries)
            manager.succeed()
            logger.info(
                "gui_agent_task_succeeded：steps=%d",
                manager.state.step_count,
            )
            return _AttemptOutcome(
                "terminal",
                self._result_message(result, manager),
                prompt_state,
                state.step_retries,
            )
        finish_reason = (
            _PROTECTED_FOREGROUND_FINISH_REASON
            if protected_finish
            else (
                _PREMATURE_FINISH_REASON
                if state.successful_action_count == 0
                else _UNRESOLVED_FAILURE_FINISH_REASON
            )
        )
        manager.record_attempt(
            action,
            False,
            state.attempt,
            "finish",
            finish_reason,
        )
        current_foreground = get_foreground_hwnd()
        current_identity = self._window_identity(current_foreground)
        prompt_state = replace(
            prompt_state,
            last_action=self._serialize_action(action),
            last_dispatch_status="none",
            last_error=finish_reason,
            foreground_after=current_identity,
            last_effect="none",
        )
        return self._retry_or_exhausted(
            state.attempt,
            state.step_retries,
            manager,
            prompt_state,
        )

    def _handle_safety_rejection(
        self,
        action: ParsedAction,
        safety_failure: str,
        state: _AttemptState,
        manager: TaskManager,
    ) -> _AttemptOutcome:
        """处理控制分发前的安全层拒绝。

        shell close hotkey(桌面关闭快捷键)为不可重试失败,结束当前任务;
        其余安全拒绝按 retryable 处理,在 retry 预算内继续 fresh observation。
        """
        manager.record_attempt(
            action,
            False,
            state.attempt,
            "dispatch",
            safety_failure,
        )
        current_foreground = get_foreground_hwnd()
        prompt_state = replace(
            state.prompt_state,
            last_action=self._serialize_action(action),
            last_dispatch_status="failure",
            last_error=safety_failure,
            foreground_after=self._window_identity(current_foreground),
        )
        if safety_failure == _SHELL_CLOSE_HOTKEY_REASON:
            manager.finalize_step(False, retry_count=state.step_retries)
            if state.step_retries:
                manager.record_retry(state.step_retries)
            manager.fail(safety_failure)
            return _AttemptOutcome(
                "terminal",
                self._failure_message(safety_failure, manager),
                prompt_state,
                state.step_retries,
            )
        return self._retry_or_exhausted(
            state.attempt,
            state.step_retries,
            manager,
            prompt_state,
        )

    def _retry_or_exhausted(
        self,
        attempt: int,
        step_retries: int,
        manager: TaskManager,
        prompt_state: ActionPromptState,
    ) -> _AttemptOutcome:
        """统一处理 retryable 失败的重试或步终止决策。

        attempt 未耗尽 retry 预算时,本方法先等待重试间隔再返回 retry,
        调用方继续 fresh observation;否则 finalize_step 并返回 exhausted
        (跳出内层循环,进入下一 logical step)。``prompt_state`` 用于写回
        caller 的循环状态;各分支在调用前已将其更新为本次 attempt 产出
        的最新值。
        """
        if attempt < self._settings.retry_count:
            self._dependencies.sleep(_RETRY_DELAY_SECONDS)
            return _AttemptOutcome(
                "retry",
                None,
                prompt_state,
                step_retries + 1,
            )
        manager.finalize_step(False, retry_count=step_retries)
        if step_retries:
            manager.record_retry(step_retries)
        return _AttemptOutcome(
            "exhausted",
            None,
            prompt_state,
            step_retries,
        )

    def _protected_foreground_click_failure(
        self,
        action: ParsedAction,
        screenshot_size: tuple[int, int],
        region_offset: tuple[int, int],
        initial_foreground_unlocked: bool,
    ) -> str | None:
        """拒绝落在未解锁起始前台窗口矩形内的鼠标动作。

        命令行等任务输入界面是只读的:模型点击其内部(包括关闭按钮)没有
        合法用途,坐标命中即拒绝;窗口几何不可得时不猜测,放行交由后续
        控制层校验。坐标映射与 ActionDispatcher 保持一致。
        """
        action_type = action["action_type"]
        if action_type not in {"click", "right_click", "double_click", "drag"}:
            return None
        if (
            not self._dependencies.protect_initial_foreground
            or not self._agent_ui_hwnd
            or initial_foreground_unlocked
        ):
            return None
        rect = get_window_screen_rect(self._agent_ui_hwnd)
        if rect is None:
            return None
        left, top, width, height = rect
        params = action["params"]
        if action_type == "drag":
            raw_points = (
                (params["x1"], params["y1"]),
                (params["x2"], params["y2"]),
            )
        else:
            raw_points = ((params["x"], params["y"]),)
        screen_w, screen_h = screenshot_size
        for raw_x, raw_y in raw_points:
            if self._settings.coordinate_mode == "normalized_1000":
                if not 0 <= raw_x <= 1000 or not 0 <= raw_y <= 1000:
                    continue
                pixel_x = round(raw_x * (screen_w - 1) / 1000) + region_offset[0]
                pixel_y = round(raw_y * (screen_h - 1) / 1000) + region_offset[1]
            else:
                pixel_x = raw_x + region_offset[0]
                pixel_y = raw_y + region_offset[1]
            if left <= pixel_x < left + width and top <= pixel_y < top + height:
                return _PROTECTED_FOREGROUND_CLICK_REASON
        return None

    def _get_dispatch_safety_failure(
        self,
        action: ParsedAction,
        state: ActionPromptState,
        initial_foreground_unlocked: bool,
    ) -> str | None:
        """在控制分发前阻止目标丢失和桌面关闭快捷键。"""
        if self._agent_ui_hwnd and not is_window_available(
            self._agent_ui_hwnd,
        ):
            return _TARGET_WINDOW_LOST_REASON
        if action["action_type"] == "type":
            foreground = get_foreground_app_hwnd()
            if foreground == 0:
                return _DESKTOP_TYPE_REASON
            if (
                self._dependencies.protect_initial_foreground
                and foreground == self._agent_ui_hwnd
                and not initial_foreground_unlocked
            ):
                return _PROTECTED_FOREGROUND_TYPE_REASON
            if (
                state.last_dispatch_status == "success"
                and state.last_action == self._serialize_action(action)
            ):
                return _REPEATED_TYPE_REASON
        if action["action_type"] != "hotkey":
            return None
        keys = cast(tuple[str, ...], action["params"]["keys"])
        key_set = frozenset(keys)
        if key_set == {"alt", "f4"} and get_foreground_app_hwnd() == 0:
            return _SHELL_CLOSE_HOTKEY_REASON
        if (
            key_set in _CLOSE_HOTKEY_SETS
            and self._dependencies.protect_initial_foreground
            and self._agent_ui_hwnd
            and get_foreground_app_hwnd() == self._agent_ui_hwnd
            and not initial_foreground_unlocked
        ):
            return _PROTECTED_FOREGROUND_CLOSE_REASON
        return None

    def _capture(
        self,
        manager: TaskManager,
        step_number: int,
    ) -> tuple[Image.Image, tuple[int, int]] | None:
        """截取当前画面并返回 ``(图像, region_offset)``。

        先截取完整当前屏幕；新前台应用有可靠窗口区域时再截该区域，否则把
        完整截图发送给模型。region 越界或失败时回退全屏。完整帧保留给动作
        效果验证；截图失败返回 None。
        """
        try:
            full = self._dependencies.capture(screen_id=0, region=None)
        except ScreenCaptureError:
            manager.fail(_CAPTURE_FAILURE_REASON)
            return None
        region, offset = select_capture_region(self._agent_ui_hwnd)
        self._latest_full = full
        logger.debug(
            "visual_focus_step：step=%d，region=%s，offset=%s",
            step_number,
            region,
            offset,
        )
        if region is not None:
            try:
                image = self._dependencies.capture(screen_id=0, region=region)
                return image, offset
            except (ScreenCaptureError, ValueError):
                pass  # region 越界/负坐标/失败 → 回退全屏
        return full, (0, 0)

    def _wait_for_ui_stable(self) -> Image.Image | None:
        """动作后等待 UI 稳定:帧差收敛则提前返回,否则有界重采,失败回退固定等待。

        通用(非 task-specific):适用于开始菜单、搜索、快速设置、对话框、页面切换等
        瞬态 UI 动画。不增加 logical step,有界。
        """
        try:
            previous = self._dependencies.capture(screen_id=0, region=None)
        except ScreenCaptureError:
            self._dependencies.sleep(_ACTION_STABILIZATION_SECONDS)
            return None
        for _ in range(_STABILITY_MAX_ITERS):
            self._dependencies.sleep(_STABILITY_INTERVAL)
            try:
                current = self._dependencies.capture(screen_id=0, region=None)
            except ScreenCaptureError:
                return None
            if frames_stable(previous, current):
                return current
            previous = current
        return previous

    def _verify_action_effect(
        self,
        action_type: str,
        before_full: Image.Image,
        before_foreground: int,
        before_app_foreground: int,
    ) -> _ActionObservation:
        """返回前台、窗口存在性和稳定帧差能够证明的动作效果。"""
        after_full = self._wait_for_ui_stable()
        after_foreground = get_foreground_hwnd()
        if after_full is None:
            return _ActionObservation(
                False,
                _ACTION_VERIFICATION_FAILURE_REASON,
                after_foreground,
                None,
                "none",
                0.0,
            )
        self._latest_full = after_full
        change_ratio = frame_change_ratio(before_full, after_full)
        progress_threshold = (
            _SHELL_CLICK_PROGRESS_DIFF_RATIO
            if action_type == "click" and before_app_foreground == 0
            else _ACTION_PROGRESS_DIFF_RATIO
        )
        changed_pixels = round(
            change_ratio * before_full.width * before_full.height,
        )
        app_has_small_progress = (
            before_app_foreground != 0 and changed_pixels >= _ACTION_PROGRESS_MIN_PIXELS
        )
        screen_changed = change_ratio >= progress_threshold or app_has_small_progress
        foreground_changed = after_foreground != before_foreground
        if self._task_target is not None and not is_window_existing(
            int(self._task_target["hwnd"]),
        ):
            effect: PromptEffect = "window_closed"
        elif before_foreground and not is_window_existing(before_foreground):
            effect: PromptEffect = "window_closed"
        elif foreground_changed:
            effect = "foreground_window_changed"
        elif screen_changed:
            effect = "visible_content_changed"
        else:
            effect = "none"
        succeeded = foreground_changed or screen_changed
        return _ActionObservation(
            succeeded,
            None if succeeded else _NO_UI_PROGRESS_REASON,
            after_foreground,
            screen_changed,
            effect,
            change_ratio,
        )

    @staticmethod
    def _window_identity(hwnd: int) -> str:
        """把 HWND 转换为不含窗口标题的安全进程标识。"""
        if not hwnd:
            return "desktop"
        return get_window_process_name(hwnd) or "unknown"

    @staticmethod
    def _platform_name() -> Literal["windows", "macos", "linux", "unknown"]:
        """把 Python 平台标识收敛为 Prompt 允许的四种值。"""
        if sys.platform == "win32":
            return "windows"
        if sys.platform == "darwin":
            return "macos"
        if sys.platform.startswith("linux"):
            return "linux"
        return "unknown"

    @staticmethod
    def _serialize_action(action: ParsedAction) -> str:
        """把已解析动作重建为单行、JSON 转义的 Prompt 状态。"""
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
            direction = json.dumps(params["direction"])
            return f"scroll(direction={direction}, steps={params['steps']})"
        if action_type == "hotkey":
            keys = cast(tuple[str, ...], params["keys"])
            arguments = ", ".join(
                f"key{index}={json.dumps(key)}" for index, key in enumerate(keys, 1)
            )
            return f"hotkey({arguments})"
        result = json.dumps(params["result"], ensure_ascii=False)
        return f"finish(result={result})"

    @staticmethod
    def _ui_change_signal(effect: PromptEffect) -> PromptProgress:
        """把已验证效果映射为不含任务语义的 UI 变化强度。"""
        if effect in {"window_closed", "foreground_window_changed"}:
            return "strong"
        if effect == "visible_content_changed":
            return "weak"
        return "none"

    def _advance_prompt_state(
        self,
        state: ActionPromptState,
        action: ParsedAction,
        dispatched: bool,
        observation: _ActionObservation,
    ) -> ActionPromptState:
        """记录已解析动作的分发结果、视觉效果和有界历史。"""
        serialized = self._serialize_action(action)
        status = "success" if observation.succeeded else "failure"
        dispatch_status = "success" if dispatched else "failure"
        progress = self._ui_change_signal(observation.effect)
        summary = f"{action['action_type']}:{status}:{observation.effect}"
        recent_actions = (*state.recent_actions, summary)[-3:]
        same_action_streak = (
            state.same_action_streak + 1 if state.last_action == serialized else 1
        )
        no_ui_change_streak = state.no_ui_change_streak + 1 if progress == "none" else 0
        return replace(
            state,
            last_action=serialized,
            last_dispatch_status=dispatch_status,
            last_error=observation.failure_reason or "none",
            foreground_after=self._window_identity(
                observation.after_foreground,
            ),
            last_effect=observation.effect,
            ui_change_signal=progress,
            recent_actions=recent_actions,
            same_action_streak=same_action_streak,
            no_ui_change_streak=no_ui_change_streak,
        )

    def _generate_action(
        self,
        image: Image.Image,
        task: str,
        state: ActionPromptState,
    ) -> tuple[ParsedAction | None, str | None]:
        """生成并严格解析一个动作，且只进入一次 ModelClient 调用边界。"""
        try:
            scaled = prompt_context.scale_image_for_model(
                image,
                model_image_max_dim_from_env(),
            )
            response = self._dependencies.model_client.generate(
                scaled,
                compose_action_prompt(
                    task,
                    state,
                    self._settings.coordinate_mode,
                ),
                self._settings.model_mode,
            )
        except Exception as exception:
            logger.warning(
                "gui_agent_model_call_failed：exception_type=%s",
                type(exception).__name__,
            )
            return None, _MODEL_FAILURE_REASON
        if response in _TERMINAL_MODEL_FAILURE_RESPONSES:
            return None, _TERMINAL_MODEL_FAILURE
        if response in _MODEL_FAILURE_RESPONSES:
            return None, _MODEL_FAILURE_REASON
        action = parse_action(response)
        if action is None:
            category = classify_action_parse_error(response) or "unknown"
            return None, f"{_PARSE_FAILURE_REASON} ({category})"
        return action, None

    def _notify_action(self, step_number: int, action: ParsedAction) -> None:
        """把已解析动作发送给可选的只读展示回调。"""
        observer = self._dependencies.action_observer
        if observer is None:
            return
        try:
            observer(step_number, action)
        except Exception as exception:
            logger.warning(
                "gui_agent_action_observer_failed：exception_type=%s",
                type(exception).__name__,
            )

    @staticmethod
    def _validate_task_message(msg: object) -> str:
        """验证用户任务消息并返回原始任务文本。"""
        if not isinstance(msg, Msg):
            raise TypeError("msg 必须是 AgentScope Msg。")
        if msg.role != "user":
            raise ValueError("msg.role 必须是 user。")
        if not isinstance(msg.content, str):
            raise TypeError("msg.content 必须是 str。")
        if not msg.content.strip():
            raise ValueError("用户任务不得为空。")
        return msg.content

    def _extract_task_target(self, msg: Msg) -> dict[str, object] | None:
        """从消息 metadata 提取 CLI 解析的任务目标窗口绑定。

        仅接受含 hwnd(int>0) 与 process(str) 的字典;非法数据一律视为
        未绑定,不由编排层猜测。
        """
        metadata = getattr(msg, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        target = metadata.get("task_target_window")
        if not isinstance(target, dict):
            return None
        hwnd = target.get("hwnd")
        process = target.get("process")
        if type(hwnd) is not int or hwnd <= 0:
            return None
        if not isinstance(process, str) or not process:
            return None
        return {"hwnd": hwnd, "process": process}

    @staticmethod
    def _result_message(content: str, manager: TaskManager | None) -> Msg:
        """创建携带任务统计 metadata 的最终消息。"""
        metadata: dict[str, object] = {}
        if manager is not None:
            state = manager.state
            duration = 0.0
            if state.ended_at is not None and state.started_at is not None:
                duration = (state.ended_at - state.started_at).total_seconds()
            metadata = {
                "steps": state.step_count,
                "duration_seconds": round(duration, 1),
                "retries": state.retry_count,
            }
        return Msg(_AGENT_NAME, content, "assistant", metadata=metadata)

    def _failure_message(
        self,
        reason: str,
        manager: TaskManager | None = None,
    ) -> Msg:
        """创建固定失败消息，不包含模型原文或用户数据。"""
        logger.warning("gui_agent_task_failed：reason=%s", reason)
        return self._result_message(f"任务执行失败：{reason}", manager)
