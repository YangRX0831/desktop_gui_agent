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
    ``max_steps`` 是基础 logical step 上限；仅在最近步骤有可测推进且没有
    安全、重复策略或不可恢复阻断时，通用预算策略可一次性扩展最多 3 步。

安全边界：
    本模块不直接操作鼠标键盘，也不读取模型私有状态。业务日志不记录任务、
    prompt 或模型原文；经 AGENTS.md §9.2 诊断例外批准，失败事件可把
    prompt、响应原文与截图交给可选诊断写入器落盘到本地 logs/diagnosis/。
    CLI observer 只接收已解析动作，观察失败不能改变任务结果。
"""

import hashlib
import itertools
import json
import logging
import os
import re
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, cast

from agentscope.agent import AgentBase
from agentscope.message import Msg
from PIL import Image

from agent.action_dispatcher import ActionDispatcher, PermissionScope
from agent.action_parser import (
    CANONICAL_V3_ACTIONS,
    ActionPromptState,
    FinishParams,
    ParsedAction,
    PromptEffect,
    PromptProgress,
    PromptStatus,
    classify_action_parse_error,
    classify_prd_action_parse_error,
    compose_action_prompt,
    is_click_like_action,
    parse_action,
    parse_prd_action,
)
from agent.action_prompt_v2 import ACTION_SYSTEM_PROMPT_V2, compose_action_prompt_v2
from agent.action_prompt_v3 import ACTION_SYSTEM_PROMPT_V3, compose_action_prompt_v3
from agent.action_response_adapter import NormalizationReason, adapt_action_response
from agent.agent_trace import AgentTraceWriterProtocol
from agent.cross_app_transfer import (
    FOCUS_SETTLE_SECONDS,
    PendingCrossAppTransfer,
    evaluate_transfer_transition,
    extract_selection_text,
    get_clipboard_sequence,
)
from agent.dashscope_api_backend import DASHSCOPE_CONFIGURATION_MESSAGE
from agent.delivery_confirmation import (
    PendingSubmission,
    classify_submit_candidate,
    evaluate_delivery_transition,
    find_payload_occurrences,
)
from agent.diagnostics import DiagnosticsRecord, DiagnosticsWriterProtocol
from agent.dialog_transition import (
    CLOSE_POLL_INTERVAL_S,
    CLOSE_POLL_TOTAL_S,
    TRIGGER_DIALOG_CONFIRMED,
    TRIGGER_NONE,
    DialogPollTiming,
    DialogWindowHistory,
    PendingDialogTransition,
    classify_save_menu_candidate,
    dialog_title_supports_save,
    evaluate_post_save,
    poll_for_dialog,
    retroactive_save_dialog_arm,
)
from agent.local_compact_prompt import compose_local_compact_prompt
from agent.local_output_repair import repair_local_output
from agent.model_client import ModelCallOptions
from agent.semantic_routes import (
    AppStateSnapshot,
    SemanticRoute,
    build_app_launch_route,
    build_browser_search_route,
    build_calculator_expression_route,
    build_file_open_route,
    build_file_search_route,
    build_save_dialog_route,
    build_save_dialog_route_v2,
    build_transfer_paste_route,
    calculator_foreground_is_reliable,
    calculator_title_keywords,
    enumerate_save_dialog_windows,
    extract_app_launch_info,
    extract_browser_search_info,
    extract_calculator_expression,
    extract_file_route_info,
    is_save_download_task,
    process_name_of_hwnd,
    resolve_save_folder_location,
    snapshot_app_state,
)
from agent.task_expectation import (
    ProgressTracker,
    TaskExpectation,
    VerificationResult,
    extract_task_expectation,
    verify_completion,
)
from agent.task_manager import (
    ProgressDependentStepBudget,
    StepExtensionEvidence,
    TaskManager,
)
from config import (
    OBSERVE_WAIT_SECONDS,
    GuiAgentSettings,
    local_model_image_max_dim_from_env,
    model_image_max_dim_from_env,
)
from perception import prompt_context
from perception.prompt_context import OCRRecognizerProtocol
from perception.screenshot import (
    activate_window,
    capture_screen,
    get_foreground_app_hwnd,
    get_foreground_hwnd,
    get_window_process_name,
    get_window_screen_rect,
    is_window_available,
    is_window_existing,
    minimize_window,
    select_capture_region,
)
from perception.ui_locator import frame_change_ratio, frames_stable
from utils.exceptions import ScreenCaptureError
from utils.run_diagnostics import diag_log, diag_phase, text_digest, watchdog_clear

logger = logging.getLogger(__name__)

_AGENT_NAME = "gui_agent"
_TERMINAL_MODEL_FAILURE_RESPONSES = {
    DASHSCOPE_CONFIGURATION_MESSAGE,
}
# API 重试耗尽的固定响应:属 PRD 4.5.1 "重试失败后记录错误并继续下一步",
# 不再按终态短路任务。
_API_EXHAUSTED_MODEL_RESPONSE = "API 模型调用失败。"
_API_EXHAUSTED_MODEL_FAILURE = "api_retry_exhausted"
_PREMATURE_FINISH_REASON = "尚未成功执行GUI动作。"
_UNRESOLVED_FAILURE_FINISH_REASON = "上一动作失败且没有强可验证进展。"
_NO_UI_PROGRESS_REASON = "操作后未检测到界面变化。"
# SEMANTIC EXECUTION PHASE 2A:finish 是提案,程序化验证 NOT_VERIFIED 时拒绝。
_FINISH_NOT_VERIFIED_REASON = "任务完成未通过程序化验证："
_ACTION_VERIFICATION_FAILURE_REASON = "操作结果验证失败。"
_MODEL_FAILURE_RESPONSES = _TERMINAL_MODEL_FAILURE_RESPONSES | {
    _API_EXHAUSTED_MODEL_RESPONSE,
    "本地模型调用失败。",
}
_MODEL_FAILURE_REASON = "模型调用失败。"
# V2 决策协议新增的拒绝原因
_OBSERVE_LIMIT_REASON = "连续observe已达上限，必须更换策略。"
_REPEATED_STRATEGY_REASON = "重复无效动作已被程序阻止，必须更换策略。"
_NUMERIC_RESULT_BAND_HEIGHT_FRACTION = 0.32
_NO_EFFECT_FINISH_REASON = "上一动作无可见效果，不能据此宣称任务完成。"
_STRUCTURED_ENTRY_FINISH_REASON = (
    "上一结构化输入可能仍有最后字段处于编辑状态，因为输入未以显式单元格或行提交"
    "结束。完成前应先验证或提交最后字段。"
)
_STRUCTURED_ENTRY_SEPARATORS = ("\t", "\n", "\r")
_STRUCTURED_ENTRY_COMMIT_KEYS = frozenset({"enter", "tab"})
_OBSERVE_ACTION_TEXT = "Action: observe()"
_TERMINAL_MODEL_FAILURE = "terminal_model_failure"
_PARSE_FAILURE_REASON = "模型动作解析失败。"
# 解析失败的格式纠正提示:模型无关,指向动作语法合同本身;经既有
# last_error 状态块进入 fresh retry 的 Prompt,不改动 Prompt 模板。
_PARSE_FEEDBACK_HINTS = {
    "invalid_syntax": (
        "上次输出不符合动作格式;只输出一行,形如 Action: click(x=100, y=200),"
        "不要解释、Markdown 或多个动作。"
    ),
    "invalid_parameters": (
        "上次动作参数不合法;坐标必须是0到1000的整数,hotkey 必须逐个编号"
        '写成 key1="ctrl", key2="c" 的形式,文本用双引号括起。'
    ),
    "unsupported_action": "上次动作不在支持列表;只允许 "
    + "、".join(CANONICAL_V3_ACTIONS)
    + "。",
    "multiline_response": "上次输出包含换行;只输出一行动作,不要换行。",
    "empty_response": "上次没有输出动作;必须输出一个动作。",
}


def _parse_feedback_hint(category: str) -> str:
    """按解析失败类别返回通用格式纠正提示;未知类别给兜底提示。"""
    return _PARSE_FEEDBACK_HINTS.get(
        category,
        "上次输出不符合动作格式;严格按动作定义输出一行 Action。",
    )


def _task_requests_visible_end_state(task: str) -> bool:
    """任务明确要求结果留在目标界面时，不在成功边界抢回自身窗口。"""
    return bool(
        re.search(
            r"(?:保留|保持|留在).{0,16}(?:界面|窗口)",
            task,
        )
    )


def _evaluated_calculator_result_text(text: str) -> str:
    """只接受带等号的计算完成态，拒绝把当前操作数误当最终结果。"""
    return text if "=" in text or "＝" in text else ""


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
# 自身控制窗口最小化后的桌面重绘等待:一次性、有界,确保第一张发给
# 模型的截图不包含最小化动画残留。
_OWN_WINDOW_HIDE_SETTLE_SECONDS = 0.5
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
        options: ModelCallOptions | None = None,
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
class _TraceExtras:
    """模型调用 trace 的修复、归一化与解析附加字段。"""

    repaired_response: str | None = None
    repair_types: list[str] | None = None
    parse_error: str | None = None
    raw_parse_success: bool | None = None
    normalization_reason: NormalizationReason | None = None


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
        diagnostics_writer: 可选诊断写入器；仅在模型调用或解析失败时落盘
            prompt、响应原文与截图。None 表示不启用诊断。

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
    diagnostics_writer: DiagnosticsWriterProtocol | None = None
    trace_writer: AgentTraceWriterProtocol | None = None

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
        if self.diagnostics_writer is not None and not callable(
            getattr(self.diagnostics_writer, "record", None),
        ):
            raise TypeError("diagnostics_writer 必须提供可调用的 record。")
        if self.trace_writer is not None and not callable(
            getattr(self.trace_writer, "record_model_call", None),
        ):
            raise TypeError("trace_writer 必须提供可调用的 record_model_call。")


def _resolve_backend_identity(model_mode: str) -> str:
    """返回旁路日志用的后端标识;任何环境读取失败都退化为固定占位。"""
    if model_mode == "api":
        return f"api:{os.environ.get('DASHSCOPE_API_MODEL', 'unknown')}"
    runtime = os.environ.get("GUI_AGENT_LOCAL_RUNTIME", "transformers")
    if runtime.strip().lower() == "openvino":
        path = os.environ.get("GUI_AGENT_OPENVINO_MODEL_DIR", "")
        return f"local:openvino:{path or 'unconfigured'}"
    path = os.environ.get("GUI_AGENT_LOCAL_MODEL_DIR", "")
    return f"local:transformers:{path or 'unconfigured'}"


def _action_screen_point(
    action: ParsedAction,
    image_size: tuple[int, int],
    region_offset: tuple[int, int],
    coordinate_mode: str,
) -> tuple[int, int] | None:
    """返回带坐标动作的锚点屏幕位置;无坐标动作返回 None。

    click 族取其坐标,drag 取终点;相对坐标按当前截图尺寸换算为
    像素后叠加截图区域原点。只用作感知放大中心,不参与分发。"""
    width, height = image_size
    if is_click_like_action(action):
        point_params = action["params"]
        x, y = point_params["x"], point_params["y"]
    elif action["action_type"] == "drag":
        drag_params = action["params"]
        x, y = drag_params["x2"], drag_params["y2"]
    else:
        return None
    if coordinate_mode == "normalized_1000":
        x = round(x * (width - 1) / 1000)
        y = round(y * (height - 1) / 1000)
    return (x + region_offset[0], y + region_offset[1])


def _focus_local_point(
    focus_screen: tuple[int, int] | None,
    region_offset: tuple[int, int],
) -> tuple[int, int] | None:
    """把屏幕锚点换算到当前截图本地坐标;不在截图内时返回 None。"""
    if focus_screen is None:
        return None
    return (
        focus_screen[0] - region_offset[0],
        focus_screen[1] - region_offset[1],
    )


def _updated_structured_entry_pending(
    current: bool,
    action: ParsedAction,
    dispatched: bool,
) -> bool:
    """按成功分发的结构化输入或明确提交键更新待提交状态。"""
    if not dispatched:
        return current
    if action["action_type"] == "type":
        text = action["params"]["text"]
        if any(separator in text for separator in _STRUCTURED_ENTRY_SEPARATORS):
            return not text.endswith(_STRUCTURED_ENTRY_SEPARATORS)
        return current
    if current and action["action_type"] == "hotkey":
        keys = action["params"]["keys"]
        if len(keys) == 1 and keys[0].strip().lower() in _STRUCTURED_ENTRY_COMMIT_KEYS:
            return False
    return current


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
        # Agent 自身控制界面窗口 HWND(任务提交时的前台);保护与截图基准用。
        self._agent_ui_hwnd = 0
        # 本次 run 是否由本 Agent 主动最小化了自身窗口;reply 结束时恢复。
        self._own_window_minimized = False
        # 任务目标窗口绑定(id/process),由 CLI 从前台时间线解析后经 metadata 传入。
        self._task_target: dict[str, object] | None = None
        # 最近一次全屏截图,用于动作前后效果比较。
        self._latest_full: Image.Image | None = None
        # 当前 run 的诊断标识;由 reply 生成、run 结束清空,失败分支用它
        # 把诊断事件归入同一 JSONL。
        self._current_run_id = ""
        self._current_task_id = "unknown"
        # 当前 run 的后端标识(api:模型名 / 本地模型目录);reply 内覆写。
        self._run_backend_identity: str = ""
        # SEMANTIC EXECUTION PHASE 2A 的 run 级上下文;reply 内初始化,
        # finally 清空,不跨 run 继承。
        self._run_expectation: TaskExpectation | None = None
        self._run_progress: ProgressTracker | None = None
        self._last_completion_evidence: dict[str, object] | None = None
        # P1:最近一次投递状态转移判定的证据(trace 用)。
        self._last_delivery_evidence: dict[str, object] | None = None
        # PHASE 2B 最近一次模型调用的 grounding 候选(结构化,供 trace)。
        self._last_grounding_candidates: list[dict[str, object]] = []
        # PHASE 2C proactive 检查产出的待复用感知(image, offset, OCR 行,
        # 结构化 OCR 条目):未提前结束时直接供下一次模型调用使用,避免
        # 同屏重复 capture/OCR;第四元素供 P1 投递确认复用同一份 box 证据。
        self._pending_observation: (
            tuple[
                Image.Image,
                tuple[int, int],
                tuple[str, ...],
                tuple[dict[str, object], ...],
            ]
            | None
        ) = None
        # P1 POST_SUBMIT_DELIVERY_CONFIRMATION:task 局部待验证投递状态;
        # 每个任务开始时清空,绝不跨 task/run 泄漏。
        self._pending_submission: object | None = None
        # 任务级安全网上下文:当前逻辑步号(意外异常终态化时记录)。
        self._current_step: int = 0
        # P2 MULTI_STEP_DOWNLOAD_SAVE_COMPLETION:task 局部保存对话框
        # 转移状态;任务开始时清空,不跨 task/run 泄漏。
        self._pending_dialog: PendingDialogTransition | None = None
        # P2 recovery:task 局部对话框窗口历史(每步真实快照)与
        # 近期动作类型(retroactive arm 的触发回看)。
        self._dialog_history = DialogWindowHistory()
        self._recent_action_types: list[str] = []
        # P3 CROSS_APP_CONTENT_TRANSFER:task 局部搬运状态;任务开始与
        # reply 结束时整体清空,不跨 task/run 泄漏。
        self._pending_transfer: PendingCrossAppTransfer | None = None
        self._last_transfer_evidence: dict[str, object] | None = None
        # FINAL FIX 2+3:确定性 keyboard-native 语义路线;步骤耗尽或
        # run 结束后清空,归还模型决策权。
        self._active_route: SemanticRoute | None = None
        # Calculator keyboard capability 每个 task 最多提交一次表达式。
        self._calculator_expression_submitted = False
        # S04 SEMANTIC FIX:目标应用 task-start 快照(before/after 判定)。
        self._app_snapshot: AppStateSnapshot | None = None
        # 结构化多字段输入的末项可能仍在编辑态；task/run 边界严格清空。
        self._pending_structured_entry_commit = False

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
        self._agent_ui_hwnd = self._extract_agent_ui_hwnd(msg)
        self._current_task_id = self._extract_trace_task_id(msg)
        manager = self._dependencies.task_manager_factory(task)
        manager.start()
        logger.info("gui_agent_task_started")

        # 每个 run 新建独立的临时授权作用域;run 结束立即清空,不跨 run 继承。
        run_scope = PermissionScope(
            allowed_actions=_RUN_PERMISSION_ACTIONS,
            token=next(_permission_token_counter),
        )
        # 诊断 run 标识:秒级时间戳加 scope token,同秒并发 run 也不混写。
        self._current_run_id = (
            f"run_{time.strftime('%Y%m%d_%H%M%S')}_t{run_scope.token}"
        )
        # 旁路日志用模型标识(api:模型名 / 本地模型目录);不含凭据。
        self._run_backend_identity = _resolve_backend_identity(
            self._settings.model_mode,
        )
        dispatcher = self._dependencies.action_dispatcher
        dispatcher.activate_run_scope(run_scope)
        try:
            return await self._run_task(task, manager)
        finally:
            # 生命周期保护:由本 Agent 最小化的自身窗口在 run 边界恢复,
            # 正常模式继续可用。若成功任务明确要求结果保留在目标界面，
            # 则遵守该可见终态语义，不用 CLI 遮挡最终结果。
            if self._own_window_minimized:
                keep_result_visible = (
                    manager.state.status.value == "success"
                    and _task_requests_visible_end_state(task)
                )
                if not keep_result_visible and not activate_window(self._agent_ui_hwnd):
                    # run 结束恢复失败只影响下次交互体验;debug 留痕一次。
                    logger.debug(
                        "own_window_restore_failed：hwnd=%d",
                        self._agent_ui_hwnd,
                    )
                self._own_window_minimized = False
            self._current_run_id = ""
            self._current_task_id = "unknown"
            self._current_step = 0
            self._run_expectation = None
            self._run_progress = None
            self._last_completion_evidence = None
            self._last_delivery_evidence = None
            self._last_grounding_candidates = []
            self._pending_observation = None
            self._active_route = None
            self._calculator_expression_submitted = False
            self._app_snapshot = None
            self._pending_structured_entry_commit = False
            self._pending_dialog = None
            self._pending_transfer = None
            self._dialog_history = DialogWindowHistory()
            self._recent_action_types = []
            watchdog_clear()
            dispatcher.clear_run_scope()

    async def _run_task(
        self,
        task: str,
        manager: TaskManager,
    ) -> Msg:
        """任务级安全网:任何意外 Exception 终态化为 FAIL,不静默卡死。

        只捕获 ``Exception``;KeyboardInterrupt/SystemExit/GeneratorExit
        保持系统语义继续向外传播。异常信息按安全白名单记录(类型名 +
        帧摘要 文件:行号:函数),不落敏感值;终态事件经
        ``_failure_message`` 的 gui_agent_task_failed 正常发射一次。
        """
        try:
            return await self._run_task_inner(task, manager)
        except Exception as exception:
            return self._terminalize_unexpected_exception(exception, manager)

    def _terminalize_unexpected_exception(
        self,
        exception: Exception,
        manager: TaskManager,
    ) -> Msg:
        """把逃逸到任务边界的意外异常转为可观测终态 FAIL。"""
        frames = traceback.extract_tb(exception.__traceback__)[-5:]
        stack_summary = " <- ".join(
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
            for frame in frames
        )
        logger.error(
            "gui_agent_task_unexpected_exception：exception_type=%s，"
            "step=%s，run=%s，stack=%s",
            type(exception).__name__,
            self._current_step,
            self._current_run_id,
            stack_summary,
        )
        reason = f"任务执行内部异常（{type(exception).__name__}）"
        manager.fail(reason)
        return self._failure_message(reason, manager)

    async def _run_task_inner(
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
            current_goal=task,
        )
        # 应用 HWND 仅作为"发现阶段全屏、切换后 region"的截图基准。
        if not self._agent_ui_hwnd:
            self._agent_ui_hwnd = get_foreground_app_hwnd()
        # benchmark/实验模式:在首次模型截图前最小化自身控制窗口,使
        # CLI 不再出现在模型视野;默认关闭。
        self._minimize_own_window_for_run()
        # SEMANTIC EXECUTION PHASE 2A:窄模式任务期望 + 可量化事实进度
        # 跟踪(默认关闭;关闭时动态 Prompt 与 trace 均不出现新字段)。
        if self._settings.semantic_execution:
            self._run_expectation = extract_task_expectation(task)
            self._run_progress = ProgressTracker(
                (
                    float(self._run_expectation.expected_volume)
                    if self._run_expectation.expected_volume is not None
                    else None
                ),
            )
            self._initialize_semantic_route(task)
        self._latest_full = None
        self._calculator_expression_submitted = False
        self._pending_structured_entry_commit = False
        step_budget = ProgressDependentStepBudget(self._settings.max_steps)
        self._record_step_budget_trace(step_budget, 0)
        # P1:投递确认状态为 task 局部,任务开始时整体清空。
        self._pending_submission = None
        # P2:保存对话框转移状态为 task 局部,任务开始时整体清空。
        self._pending_dialog = None
        self._pending_transfer = None
        self._dialog_history = DialogWindowHistory()
        self._recent_action_types = []
        successful_action_count = 0
        # 起始前台窗口内容发生已分发动作引起的大幅变化后解除只读保护;
        # run 级状态,不跨任务继承。
        initial_foreground_unlocked = False
        # 焦点放大 OCR 的锚点:最近一次带坐标动作的全局屏幕位置;
        # type/hotkey 等无坐标动作作用于当前焦点,沿用旧锚点。
        focus_screen: tuple[int, int] | None = None
        step_number = 1
        latest_focus_transition = False
        latest_change_ratio = 0.0
        while step_number <= step_budget.effective_hard_limit:
            step_budget.begin_step(step_number)
            if step_number > step_budget.configured_max_steps:
                self._record_step_budget_trace(step_budget, step_number)
            diag_log("diag_step_begin", step=step_number)
            self._current_step = step_number
            # P2 recovery:每步观察点做一次真实对话框窗口快照
            # (轻量 Win32 枚举,毫秒级),构成真实 pre/post 历史。
            if self._settings.semantic_execution and (
                self._run_expectation is not None
                and self._run_expectation.save_image_intent
            ):
                self._dialog_history.snapshot(
                    step_number,
                    enumerate_save_dialog_windows(),
                    process_name_of_hwnd(get_foreground_app_hwnd()),
                )
            step_succeeded = False
            step_retries = 0
            latest_focus_transition = False
            latest_change_ratio = 0.0
            for attempt in range(self._settings.retry_count + 1):
                pending = self._pending_observation if attempt == 0 else None
                if pending is not None:
                    # PHASE 2C:复用 proactive 检查已产出的同一份 post-action
                    # 感知,不重复 capture/OCR。
                    self._pending_observation = None
                    image, region_offset, attempt_ocr, attempt_boxes = pending
                else:
                    # OBSERVABILITY_ONLY:粗粒度感知阶段(含截图/OCR/
                    # 窗口状态),用于隔离 screenshot/ocr 之外的挂起。
                    diag_log("diag_perception_begin", step=step_number)
                    capture = self._capture(manager, step_number)
                    if capture is None:
                        diag_log("diag_perception_end", step=step_number)
                        return self._failure_message(
                            _CAPTURE_FAILURE_REASON,
                            manager,
                        )
                    image, region_offset = capture
                    (
                        image,
                        region_offset,
                        attempt_ocr,
                        attempt_boxes,
                    ) = self._perceive_with_escalation(
                        image,
                        region_offset,
                        focus_screen,
                    )
                    diag_log("diag_perception_end", step=step_number)

                with diag_phase("diag_window_state", step=step_number):
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
                # PHASE 2B:把既有 OCR/窗口感知转成 grounding 候选;坐标
                # 与动作同为截图 0..1000 体系,Agent 自身窗口绝不入列。
                if self._settings.semantic_execution:
                    grounding = prompt_context.build_grounding_candidates(
                        attempt_ocr,
                        windows_state,
                        self._agent_ui_hwnd,
                    )
                    turn_token = f"{self._current_run_id}:{step_number}:{attempt}"
                    grounding = [
                        {
                            **candidate,
                            "symbol": f"E{index}",
                            "turn_token": turn_token,
                        }
                        for index, candidate in enumerate(grounding, 1)
                    ]
                    self._last_grounding_candidates = grounding
                    interactive_lines = prompt_context.render_interactive_elements(
                        grounding
                    )
                else:
                    interactive_lines = ()
                action, failure = self._generate_action(
                    image,
                    task,
                    replace(
                        prompt_state,
                        step_number=step_number,
                        steps_remaining=(
                            step_budget.effective_hard_limit - step_number
                            if self._settings.semantic_execution
                            else None
                        ),
                        ocr_elements=attempt_ocr,
                        interactive_elements=interactive_lines,
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
                        successful_action_count=successful_action_count,
                    ),
                    attempt,
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
                        assert outcome.final_msg is not None
                        return outcome.final_msg
                    prompt_state = outcome.prompt_state
                    step_retries = outcome.step_retries
                    if outcome.flow == "retry":
                        continue
                    break

                if (
                    self._settings.decision_protocol_v2
                    and action["action_type"] == "observe"
                ):
                    outcome = self._handle_observe_action(
                        action,
                        image,
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
                        assert outcome.final_msg is not None
                        return outcome.final_msg
                    prompt_state = outcome.prompt_state
                    step_retries = outcome.step_retries
                    if outcome.flow == "retry":
                        continue
                    step_succeeded = True
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
                        assert outcome.final_msg is not None
                        return outcome.final_msg
                    prompt_state = outcome.prompt_state
                    step_retries = outcome.step_retries
                    if outcome.flow == "retry":
                        continue
                    break

                if self._repeated_strategy_blocked(action, prompt_state):
                    outcome = self._handle_safety_rejection(
                        action,
                        _REPEATED_STRATEGY_REASON,
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
                        assert outcome.final_msg is not None
                        return outcome.final_msg
                    prompt_state = replace(
                        outcome.prompt_state,
                        previous_strategy_failed=True,
                        blocked_repeated_action=self._serialize_action(action),
                    )
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
                        assert outcome.final_msg is not None
                        return outcome.final_msg
                    prompt_state = outcome.prompt_state
                    step_retries = outcome.step_retries
                    if outcome.flow == "retry":
                        continue
                    break

                self._notify_action(step_number, action)
                before_foreground = get_foreground_hwnd()
                before_app_foreground = get_foreground_app_hwnd()
                # P1:dispatch 前记录 type payload / submit 候选与提交前
                # 感知快照(纯本地证据,不改动作本身)。
                self._track_delivery_action(
                    action,
                    step_number,
                    attempt_boxes,
                    before_app_foreground,
                )
                # P2:dispatch 前识别 save-like 上下文菜单候选并武装
                # 对话框转移跟踪(纯本地证据,不改动作本身)。
                self._track_save_dialog_action(
                    action,
                    step_number,
                    attempt_boxes,
                )
                # P3:dispatch 前记录 drag 选择源签名 / Ctrl+C 序号前值 /
                # Ctrl+V 粘贴标记(纯本地证据,不改动作本身)。
                self._track_transfer_action(action, step_number, attempt_boxes)
                before_full = None
                if self._settings.verify_action_effect:
                    latest = self._latest_full
                    if latest is not None:
                        before_full = latest.copy()
                    else:
                        before_full = image.copy()
                new_focus = _action_screen_point(
                    action,
                    image.size,
                    region_offset,
                    self._settings.coordinate_mode,
                )
                if new_focus is not None:
                    focus_screen = new_focus
                volume_before_dispatch = (
                    prompt_context.system_volume_state()
                    if self._run_progress is not None
                    and self._run_progress.target is not None
                    else None
                )
                # OBSERVABILITY_ONLY:动作分发计时;type 动作只记长度+摘要。
                _diag_dispatch_fields: dict[str, object] = {
                    "action_type": action["action_type"],
                }
                if action["action_type"] == "type":
                    _typed = action["params"].get("text")
                    if isinstance(_typed, str):
                        _diag_dispatch_fields["text_len"] = len(_typed)
                        _diag_dispatch_fields["text_digest"] = text_digest(_typed)
                dispatch_started_at = time.perf_counter()
                with diag_phase(
                    "diag_dispatch", step=step_number, **_diag_dispatch_fields
                ):
                    dispatched = self._dependencies.action_dispatcher.dispatch(
                        action,
                        image.size,
                        region_offset,
                    )
                dispatch_elapsed_ms = round(
                    (time.perf_counter() - dispatch_started_at) * 1000,
                    3,
                )
                if dispatched:
                    self._recent_action_types.append(action["action_type"])
                # 步骤级进度信号:仅记录步骤号、动作类型名与分发结果,
                # 供外部监控判断任务是否仍在推进,不含坐标、文本等参数。
                logger.info(
                    "gui_agent_step_dispatched：step=%d，action=%s，dispatched=%s",
                    step_number,
                    action["action_type"],
                    dispatched,
                )
                verification_started_at = time.perf_counter()
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
                verification_elapsed_ms = round(
                    (time.perf_counter() - verification_started_at) * 1000,
                    3,
                )
                focus_after_dispatch = prompt_context.focus_control_state()
                latest_focus_transition = bool(
                    dispatched and focus_after_dispatch not in {"unknown", focus_kind}
                )
                latest_change_ratio = observation.change_ratio
                self._pending_structured_entry_commit = (
                    _updated_structured_entry_pending(
                        self._pending_structured_entry_commit,
                        action,
                        dispatched,
                    )
                )
                post_image = self._latest_full
                post_screenshot_sha256: str | None = None
                post_screenshot_path: str | None = None
                if post_image is not None:
                    (
                        post_screenshot_sha256,
                        post_screenshot_path,
                    ) = self._trace_image_evidence(
                        post_image,
                        step_number,
                        attempt,
                        "after",
                    )
                trace_writer = self._dependencies.trace_writer
                if trace_writer is not None:
                    trace_writer.record_model_call(
                        {
                            "record_type": "action_execution",
                            "run_id": self._current_run_id,
                            "task_id": self._current_task_id,
                            "task_digest": text_digest(task),
                            "step": step_number,
                            "attempt": attempt,
                            "protocol_version": self._decision_protocol_version(),
                            "parsed_action": self._serialize_action(action),
                            "control_dispatch": dispatched,
                            "dispatch_elapsed_ms": dispatch_elapsed_ms,
                            "verification_elapsed_ms": verification_elapsed_ms,
                            "foreground_before_dispatch": self._window_identity(
                                before_foreground,
                            ),
                            "post_action_foreground": self._window_identity(
                                observation.after_foreground,
                            ),
                            "post_action_effect": observation.effect,
                            "post_action_succeeded": observation.succeeded,
                            "post_action_failure_reason": (observation.failure_reason),
                            "screen_changed": observation.screen_changed,
                            "screen_change_ratio": observation.change_ratio,
                            "focus_transition": latest_focus_transition,
                            "structured_entry_commit_pending": (
                                self._pending_structured_entry_commit
                            ),
                            "post_action_screenshot_sha256": (post_screenshot_sha256),
                            "post_action_screenshot_path": post_screenshot_path,
                        },
                    )
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
                    # Phase 2A:记录可量化事实的推进速率并评估预算充足性;
                    # 只注入事实与预算约束,不指定具体 GUI 策略。
                    if self._run_progress is not None:
                        volume_after = prompt_context.system_volume_state()
                        self._run_progress.record(
                            action["action_type"],
                            volume_before_dispatch,
                            volume_after,
                        )
                        progress_status, progress_reason = self._run_progress.evaluate(
                            volume_after,
                            step_budget.effective_hard_limit - step_number,
                        )
                        prompt_state = replace(
                            prompt_state,
                            progress_status=progress_status,
                            progress_reason=progress_reason,
                        )
                    manager.finalize_step(True, retry_count=step_retries)
                    if step_retries:
                        manager.record_retry(step_retries)
                    # P2:保存触发后的有界到达轮询 / 路线注入 /
                    # post-save 关闭验证(全部本地,零模型调用)。
                    if self._settings.semantic_execution:
                        prompt_state, save_done = self._handle_save_dialog_transition(
                            prompt_state,
                            manager,
                            step_number,
                            step_retries,
                        )
                        if save_done is not None:
                            return save_done
                    # P3:复制确认 / 目标焦点判定 / 粘贴路线注入 /
                    # post-paste 内容转移验证(全部本地,零模型调用)。
                    if self._settings.semantic_execution:
                        prompt_state, transfer_done = self._handle_cross_app_transfer(
                            prompt_state,
                            manager,
                            step_number,
                            step_retries,
                        )
                        if transfer_done is not None:
                            return transfer_done
                    # PHASE 2C:动作后(UI 已稳定)主动完成检测;VERIFIED
                    # 立即成功,不再等待模型自行 finish。
                    if self._settings.semantic_execution:
                        early_result = self._proactive_completion_check(
                            prompt_state,
                            manager,
                            step_number,
                            step_retries,
                            focus_screen,
                        )
                        if early_result is not None:
                            return early_result
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
                diag_log("diag_step_end", step=step_number, outcome="success")
            else:
                diag_log(
                    "diag_step_end",
                    step=step_number,
                    outcome="retry_exhausted",
                    retries=step_retries,
                )
                logger.warning(
                    "gui_agent_step_retry_exhausted：step=%d，retries=%d",
                    step_number,
                    step_retries,
                )

            if step_number == step_budget.configured_max_steps:
                evidence = self._step_extension_evidence(
                    prompt_state,
                    manager,
                    step_succeeded,
                    latest_focus_transition,
                    latest_change_ratio,
                    step_retries,
                )
                step_budget.evaluate(evidence)
                self._record_step_budget_trace(step_budget, step_number)
                if step_budget.extension_granted:
                    prompt_state = replace(
                        prompt_state,
                        max_steps=step_budget.effective_hard_limit,
                    )
                else:
                    break
            elif step_number > step_budget.configured_max_steps and (
                not step_succeeded
                or prompt_state.previous_strategy_failed
                or prompt_state.progress_status == "NO_PROGRESS"
                or prompt_state.no_ui_change_streak > 0
            ):
                break
            step_number += 1

        # PHASE 2C:max_steps 兜底终检:任务客观已完成时不因模型未
        # finish 而误判失败(正常情况已在更早 step 提前结束)。
        if self._settings.semantic_execution:
            final_result = self._final_max_steps_check(prompt_state, manager)
            if final_result is not None:
                return final_result
        manager.fail(_MAX_STEPS_REASON)
        return self._failure_message(_MAX_STEPS_REASON, manager)

    def _initialize_semantic_route(self, task: str) -> None:
        """FINAL FIX 3+PART A:窄模式识别文件/应用启动路线,预构建步骤。

        优先级:file route > browser search > app launch route > 无路线。
        仅对明确的文件打开/搜索、当前浏览器搜索或纯应用启动任务触发。
        """
        route_info = extract_file_route_info(task)
        if route_info is not None:
            if route_info["route_type"] == "exact_path":
                self._active_route = build_file_open_route(
                    route_info["folder"],
                    route_info["filename"],
                )
                logger.info(
                    "semantic_route_initialized：route=file_open，folder=%s，file=%s",
                    route_info["folder"],
                    route_info["filename"],
                )
            elif route_info["route_type"] == "search_substring":
                self._active_route = build_file_search_route(
                    route_info["folder"],
                    route_info["substring"],
                    route_info["extension"],
                )
                logger.info(
                    "semantic_route_initialized：route=file_search，folder=%s，"
                    "substring=%s",
                    route_info["folder"],
                    route_info["substring"],
                )
            return
        browser_search = extract_browser_search_info(task)
        if browser_search is not None:
            self._active_route = build_browser_search_route(
                browser_search["query"],
            )
            logger.info("semantic_route_initialized：route=browser_search")
            return
        # PART A:纯应用启动 intent(如"打开Chrome浏览器")。
        app_info = extract_app_launch_info(task)
        if app_info is not None:
            # PART 1:task-start 应用状态快照(before/after 判定基础)。
            self._app_snapshot = snapshot_app_state(
                tuple(app_info["process_names"]),
            )
            pre_existing = bool(self._app_snapshot.target_hwnds)
            self._active_route = build_app_launch_route(
                app_info["search_text"],
                app_id=app_info["app_id"],
                pre_existing=pre_existing,
            )
            logger.info(
                "semantic_route_initialized：route=app_launch，app=%s，"
                "pre_existing=%s，pre_hwnds=%d，pre_fg=%s",
                app_info["canonical_name"],
                pre_existing,
                len(self._app_snapshot.target_hwnds),
                self._app_snapshot.was_foreground,
            )

    def _check_save_dialog_route(self, task: str) -> SemanticRoute | None:
        """FIX B + P2 recovery:检测前台 Save-As 对话框并构建完成路线。

        触发条件:①semantic on;②前台为 #32770 Save/另存为 对话框;
        ③任务语义属于保存/下载;④当前无其他活跃路线。
        P2 recovery:检出同时向转移状态上报 SAVE_DIALOG_OBSERVED
        (legacy 检测成为 P2 的感知源,而非旁路);路线按四象限构建:
        目录(解析成功时导航)+ 文件名(有则设置,无则保留默认名)。
        """
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)
        if class_buf.value != "#32770":
            return None
        title_buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title_buf, 256)
        title = title_buf.value
        if not dialog_title_supports_save(title):
            return None
        if not is_save_download_task(task):
            return None
        expectation = self._run_expectation
        # 期望文件名只来自通用期望抽取(explicit "另存为 X.png" 措辞);
        # IMG 形状的 fixture 推断已按 PRD-BC-001 裁决删除,无替代规则。
        expected_filename = (
            expectation.expected_save_filename if expectation is not None else None
        )
        folder_location = None
        if expectation is not None and expectation.expected_save_folder:
            folder_location = resolve_save_folder_location(
                expectation.expected_save_folder,
            )
        # P2:legacy 检出上报转移状态(真实当前前台对话框)。
        self._report_save_dialog_observed(hwnd, title)
        logger.info(
            "semantic_route_save_dialog：filename=%s，folder=%s",
            expected_filename or "(default)",
            folder_location or "(default)",
        )
        if folder_location is not None:
            return build_save_dialog_route_v2(folder_location, expected_filename)
        return build_save_dialog_route(expected_filename)

    def _report_save_dialog_observed(self, hwnd: int, title: str) -> None:
        """P2:legacy 前台检出上报;按真实历史差集判定证据等级。

        legacy 检出后紧跟路线注入,因此同时置 save_route_dispatched,
        使 post-save 关闭验证在路线耗尽后运行(修复:M03 首次 recovery
        运行中路线经 legacy 注入但未置该标志,post-save 验证未触发)。
        """
        expectation = self._run_expectation
        if expectation is None or not expectation.save_image_intent:
            return
        pending = self._pending_dialog
        if isinstance(pending, PendingDialogTransition):
            if pending.dialog_hwnd is None:
                pending.dialog_hwnd = hwnd
                pending.dialog_title = title
                pending.dialog_process = process_name_of_hwnd(hwnd)
                if pending.last_trigger_result in ("", "PENDING", "NO_DIALOG"):
                    pending.last_trigger_result = "DIALOG_ARRIVED"
            pending.save_route_dispatched = True
            return
        # 无 pending:按窗口历史做 retroactive arm(真实 pre 差集)。
        current = enumerate_save_dialog_windows()
        hit = retroactive_save_dialog_arm(
            self._dialog_history,
            current,
            self._recent_action_types,
            save_intent=True,
        )
        evidence = TRIGGER_DIALOG_CONFIRMED if hit is not None else TRIGGER_NONE
        new_pending = PendingDialogTransition(
            expected_filename=expectation.expected_save_filename,
        )
        new_pending.dialog_hwnd = hwnd
        new_pending.dialog_title = title
        new_pending.dialog_process = process_name_of_hwnd(hwnd)
        new_pending.last_trigger_result = "DIALOG_ARRIVED"
        new_pending.trigger_process = self._dialog_history.last_foreground_process
        new_pending.save_route_dispatched = True
        self._pending_dialog = new_pending
        new_pending.pre_dialog_hwnds = frozenset(
            w["hwnd"] for w in self._dialog_history.previous_dialogs
        )
        new_pending.other_dialog_hwnds_at_save = frozenset(
            w["hwnd"] for w in current
        ) - {hwnd}
        diag_log(
            "diag_save_dialog_observed",
            hwnd=hwnd,
            title_supports=dialog_title_supports_save(title),
            evidence=evidence,
        )

    def _minimize_own_window_for_run(self) -> None:
        """按设置在首次模型截图前最小化自身控制窗口。

        只用 MINIMIZE,不用 HIDE:最小化窗口仍通过 IsWindowVisible
        存在性检查;实时 rect 查询对最小化窗口返回 None,受保护区域
        判定随之放行,不会用旧矩形拦截已暴露的任务区域;窗口身份
        (HWND)保持不变,重新可见后保护自动恢复。最小化失败仅记录
        安全事件并保持原可见行为,不中断任务。
        """
        if not self._settings.hide_own_window_during_run or not self._agent_ui_hwnd:
            return
        if minimize_window(self._agent_ui_hwnd):
            self._own_window_minimized = True
            self._dependencies.sleep(_OWN_WINDOW_HIDE_SETTLE_SECONDS)
        else:
            logger.warning("gui_agent_hide_own_window_failed")

    def _handle_model_failure(
        self,
        failure: str | None,
        state: _AttemptState,
        manager: TaskManager,
    ) -> _AttemptOutcome:
        """处理模型调用或动作解析失败。

        terminal_model_failure(配置错误等不可重试不变式)结束当前任务;
        api_retry_exhausted(ModelClient 已耗尽 API 的 1+3 重试)按 PRD
        4.5.1 记录错误并前进下一步,不重开第二套重试预算;其余 model/parse
        failure 属 PRD 4.5.1 retryable,在 retry 预算内继续 fresh observation。
        解析失败额外注入模型无关的格式纠正提示,让 fresh retry 带着明确
        的违约原因,而不是盲重试同一错误。
        """
        # failure 形如 _MODEL_FAILURE_REASON / _TERMINAL_MODEL_FAILURE /
        # _API_EXHAUSTED_MODEL_FAILURE / f"{_PARSE_FAILURE_REASON} (category)";
        # stage 与 recorded_failure 按 PRD 4.5.1 分类,不伪造 ParsedAction。
        stage = (
            "model"
            if failure
            in {
                _MODEL_FAILURE_REASON,
                _TERMINAL_MODEL_FAILURE,
                _API_EXHAUSTED_MODEL_FAILURE,
            }
            else "parse"
        )
        recorded_failure = (
            _MODEL_FAILURE_REASON
            if failure in {_TERMINAL_MODEL_FAILURE, _API_EXHAUSTED_MODEL_FAILURE}
            else failure
        )
        manager.record_attempt(
            None,
            False,
            state.attempt,
            stage,
            recorded_failure,
        )
        last_error_value = recorded_failure or _MODEL_FAILURE_REASON
        if stage == "parse":
            category = str(failure).rsplit("(", 1)[-1].removesuffix(")")
            last_error_value = f"{last_error_value}。{_parse_feedback_hint(category)}"
        prompt_state = replace(
            state.prompt_state,
            last_action="none",
            last_dispatch_status="none",
            last_error=last_error_value,
        )
        if failure == _API_EXHAUSTED_MODEL_FAILURE:
            # PRD 4.5.1:重试失败后记录错误并继续下一步。本步以失败终结
            # 但不 fail 任务,也不步内重试(避免 API 层与 Agent 层重试相乘)。
            manager.finalize_step(False, retry_count=state.step_retries)
            return _AttemptOutcome(
                "exhausted",
                None,
                prompt_state,
                state.step_retries,
            )
        if failure == _TERMINAL_MODEL_FAILURE:
            # 配置错误等不可重试不变式:立即结束任务,不浪费 max_steps。
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

    def _handle_observe_action(
        self,
        action: ParsedAction,
        image: Image.Image,
        step_number: int,
        state: _AttemptState,
        manager: TaskManager,
    ) -> _AttemptOutcome:
        """执行 observe:不操作任何控件,有界等待后交给下一轮 fresh 观察。

        连续第 4 次 observe 被程序拒绝(走安全拒绝重试路径);前 3 次
        允许,次数注入状态供 System 规则约束。等待期间用帧差检测界面
        是否发生独立更新,结果写入 last_effect 支撑 finish 新规则。
        observe 不计入 successful_action_count。
        """
        prompt_state = state.prompt_state
        consecutive = prompt_state.consecutive_observe_count + 1
        if consecutive > 3:
            manager.record_attempt(
                action,
                False,
                state.attempt,
                "observe",
                _OBSERVE_LIMIT_REASON,
            )
            return self._handle_safety_rejection(
                action,
                _OBSERVE_LIMIT_REASON,
                state,
                manager,
            )
        self._notify_action(step_number, action)
        self._dependencies.sleep(OBSERVE_WAIT_SECONDS)
        try:
            after = self._dependencies.capture(screen_id=0, region=None)
        except ScreenCaptureError:
            after = None
        effect: PromptEffect = "none"
        if after is not None:
            ratio = frame_change_ratio(image, after)
            if ratio >= _ACTION_PROGRESS_DIFF_RATIO or (
                round(ratio * image.width * image.height) >= _ACTION_PROGRESS_MIN_PIXELS
            ):
                effect = "visible_content_changed"
        manager.record_attempt(action, True, state.attempt, "observe")
        manager.finalize_step(True, retry_count=state.step_retries)
        summary = f"observe:success:{effect}"
        return _AttemptOutcome(
            "exhausted",
            None,
            replace(
                prompt_state,
                last_action=_OBSERVE_ACTION_TEXT,
                last_dispatch_status="success",
                last_error="none",
                last_effect=effect,
                ui_change_signal="strong" if effect != "none" else "none",
                recent_actions=(*prompt_state.recent_actions, summary)[-3:],
                consecutive_observe_count=consecutive,
                same_action_streak=0,
            ),
            state.step_retries,
        )

    def _repeated_strategy_blocked(
        self,
        action: ParsedAction,
        prompt_state: ActionPromptState,
    ) -> bool:
        """判定第三次完全相同的无效 GUI 动作是否需要程序阻止。

        条件(V2/V3 协议):相同动作已连续两次、没有强状态转移，且当前
        动作与其完全一致。微小视觉变化可能只是光标/hover，不足以证明
        重复点击产生了任务进展。observe 不走本规则(有自己的计数)。
        """
        if not (
            self._settings.decision_protocol_v2 or self._settings.decision_protocol_v3
        ):
            return False
        if action["action_type"] == "observe":
            return False
        return (
            prompt_state.same_action_streak >= 2
            and prompt_state.ui_change_signal != "strong"
            and prompt_state.progress_status not in {"PROGRESSED", "INSUFFICIENT_RATE"}
            and prompt_state.last_action == self._serialize_action(action)
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
        # V2:上一动作无可见效果时不得宣称完成;observe 后确有独立 UI
        # 更新的情形在 observe 处理中已把 last_effect 置为非 none。
        no_effect_finish = (
            self._settings.decision_protocol_v2
            and not self._settings.decision_protocol_v3
            and prompt_state.last_effect == "none"
        )
        # Phase 2A:finish 是提案,程序化验证 NOT_VERIFIED 时拒绝;
        # VERIFIED 接受;UNKNOWN 保持既有 finish 安全行为。
        completion = (
            self._verify_completion_proposal(prompt_state)
            if self._settings.semantic_execution
            else None
        )
        finish_not_verified = (
            completion is not None and completion.status == "NOT_VERIFIED"
        )
        numeric_finish_without_evidence = (
            completion is not None
            and completion.status == "UNKNOWN"
            and self._run_expectation is not None
            and self._run_expectation.expected_numeric_result is not None
        )
        structured_entry_finish = self._pending_structured_entry_commit
        accepted = not (
            protected_finish
            or finish_not_verified
            or numeric_finish_without_evidence
            or structured_entry_finish
            or (
                self._settings.reject_initial_finish
                and state.successful_action_count == 0
            )
            or unresolved_failure
            or no_effect_finish
        )
        # 验证结论发生在模型调用之后,单独追加 finish_decision trace 记录,
        # 保证接受路径的 completion 状态同样进入 trace(§Phase2A-20)。
        trace_writer = self._dependencies.trace_writer
        if trace_writer is not None and (
            completion is not None or structured_entry_finish
        ):
            trace_writer.record_model_call(
                {
                    "record_type": "finish_decision",
                    "run_id": self._current_run_id,
                    "task_id": self._current_task_id,
                    "step": step_number,
                    "attempt": state.attempt,
                    "finish_proposed": True,
                    "completion_trigger": "model_finish",
                    "accepted": accepted,
                    "completion_verification": (
                        completion.status if completion is not None else None
                    ),
                    "completion_reason": (
                        completion.reason if completion is not None else None
                    ),
                    "completion_evidence": self._last_completion_evidence,
                    "structured_entry_commit_pending": structured_entry_finish,
                },
            )
        if accepted:
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
        # finish_not_verified 定义蕴含 completion 非 None;在分支内先行
        # 断言并取出 reason,使 mypy 收窄对后续三元表达式可见。
        if finish_not_verified:
            assert completion is not None
            not_verified_reason = f"{_FINISH_NOT_VERIFIED_REASON}{completion.reason}"
        else:
            not_verified_reason = ""
        finish_reason = (
            _PROTECTED_FOREGROUND_FINISH_REASON
            if protected_finish
            else (
                _STRUCTURED_ENTRY_FINISH_REASON
                if structured_entry_finish
                else (
                    not_verified_reason
                    if finish_not_verified or numeric_finish_without_evidence
                    else (
                        _PREMATURE_FINISH_REASON
                        if state.successful_action_count == 0
                        else _UNRESOLVED_FAILURE_FINISH_REASON
                    )
                )
            )
        )
        manager.record_attempt(
            action,
            False,
            state.attempt,
            "finish",
            finish_reason,
        )
        logger.info(
            "gui_agent_finish_rejected：reason_class=%s，completion=%s",
            "protected" if protected_finish else "policy",
            completion.status if completion is not None else "off",
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
            completion_verification=(
                completion.status if completion is not None else None
            ),
            completion_reason=(completion.reason if completion is not None else None),
            structured_entry_commit_feedback=structured_entry_finish,
        )
        return self._retry_or_exhausted(
            state.attempt,
            state.step_retries,
            manager,
            prompt_state,
        )

    def _verify_completion_proposal(
        self,
        prompt_state: ActionPromptState,
    ) -> VerificationResult:
        """用程序化事实裁决 finish 提案;OCR 取当前完整截图的新识别。

        事实来源全部为程序可靠读取:系统音量、前台进程标识与当前截图
        OCR;窗口关闭类证据来自已验证的 last_effect(见 _completion_facts)。
        """
        expectation = self._run_expectation or TaskExpectation()
        ocr_elements: tuple[str, ...] = ()
        recognizer = self._dependencies.ocr_recognizer
        if (
            expectation.expected_numeric_result is not None
            and expectation.expected_app is not None
        ):
            result_text = self._ocr_target_window_text(
                get_foreground_app_hwnd(),
                _NUMERIC_RESULT_BAND_HEIGHT_FRACTION,
            )
            result_text = _evaluated_calculator_result_text(result_text)
            ocr_elements = (result_text,) if result_text else ()
        elif recognizer is not None and self._latest_full is not None:
            ocr_elements = prompt_context.perceive_ocr_elements(
                recognizer,
                self._latest_full,
            )
        result = verify_completion(
            expectation,
            self._completion_facts(prompt_state, "\n".join(ocr_elements)),
        )
        self._last_completion_evidence = dict(result.evidence) or {
            "foreground_process": "unknown",
        }
        return result

    def _completion_facts(
        self,
        prompt_state: ActionPromptState,
        ocr_text: str,
    ) -> dict[str, object]:
        """组装三态验证的程序事实;窗口关闭类由已验证效果提供证据。"""
        facts: dict[str, object] = {
            "current_volume": prompt_context.system_volume_state(),
            "ocr_text": ocr_text,
            "foreground_process": self._window_identity(
                get_foreground_app_hwnd(),
            ),
            "tracked_window_exists": None,
        }
        expectation = self._run_expectation
        if expectation is not None and expectation.expected_app_title_keywords:
            facts["foreground_title_matches_expected"] = self._window_title_matches(
                get_foreground_app_hwnd(),
                expectation.expected_app_title_keywords,
            )
        # S04 SEMANTIC FIX:注入 app 状态快照(pre/post HWND 集合)。
        if self._app_snapshot is not None and self._run_expectation:
            expectation = self._run_expectation
            if expectation.expected_app_processes:
                post = snapshot_app_state(
                    expectation.expected_app_processes,
                )
                facts["pre_target_hwnds"] = set(
                    self._app_snapshot.target_hwnds,
                )
                facts["post_target_hwnds"] = set(post.target_hwnds)
        closed_expectation = self._run_expectation
        if (
            closed_expectation is not None
            and closed_expectation.expected_window_closed is not None
        ):
            if self._task_target is not None:
                facts["tracked_window_exists"] = is_window_existing(
                    int(cast(int, self._task_target["hwnd"])),
                )
            else:
                # 未绑定具体目标时只接受 app-level 窗口关闭效果；不能把
                # 被最小化的 Agent 终端子窗口消失当作业务窗口完成证据。
                facts["tracked_window_exists"] = (
                    prompt_state.last_effect != "window_closed"
                )
        return facts

    def _emit_completion_decision(
        self,
        trigger: str,
        step_number: int,
        status: str,
        reason: str,
    ) -> None:
        """追加 completion 决策 trace 记录(PHASE 2C)。"""
        writer = self._dependencies.trace_writer
        if writer is None:
            return
        writer.record_model_call(
            {
                "record_type": "completion_decision",
                "run_id": self._current_run_id,
                "task_id": self._current_task_id,
                "step": step_number,
                "completion_trigger": trigger,
                "completion_verification": status,
                "completion_reason": reason,
                "completion_evidence": self._last_completion_evidence,
            },
        )

    def _complete_early(
        self,
        manager: TaskManager,
        step_retries: int,
        reason: str,
    ) -> Msg:
        """以程序化完成验证结束任务;不伪造模型 finish。"""
        manager.succeed()
        logger.info("gui_agent_task_succeeded：steps=%d", manager.state.step_count)
        return self._result_message(f"任务完成(程序化验证)：{reason}", manager)

    def _track_delivery_action(
        self,
        action: ParsedAction,
        step_number: int,
        boxes: tuple[dict, ...],
        foreground_app: int | None,
    ) -> None:
        """P1:在 dispatch 前维护待验证投递状态(纯记录,不改动作)。

        type 动作建立/刷新 pending payload(以实际输入文本为准);
        随后的 submit-like 动作(click 命中发送类控件词,或投递上下文
        中的 Enter)武装提交记录,并对当前感知快照 payload occurrence
        作为 composer 前置证据。非投递任务零开销。
        """
        expectation = self._run_expectation
        if expectation is None or not expectation.delivery_intent:
            return
        action_type = action["action_type"]
        if action_type == "type":
            text = action["params"].get("text")
            if isinstance(text, str) and text.strip():
                self._pending_submission = PendingSubmission(
                    payload=text.strip(),
                    typed_step=step_number,
                    foreground_app=foreground_app,
                    typed_timestamp=time.time(),
                )
            return
        pending = self._pending_submission
        if not isinstance(pending, PendingSubmission):
            return
        candidate = classify_submit_candidate(
            action_type,
            action["params"],
            list(boxes),
            has_pending_payload=True,
            delivery_intent=True,
        )
        if candidate is None:
            return
        pending.arm_submit(
            step_number,
            candidate.action_type,
            candidate.control_word,
            candidate.control_bbox,
        )
        occurrences = find_payload_occurrences(pending.payload, list(boxes))
        if occurrences:
            pending.snapshot_pre(occurrences)

    def _track_save_dialog_action(
        self,
        action: ParsedAction,
        step_number: int,
        boxes: tuple[dict, ...],
    ) -> None:
        """P2:dispatch 前识别 save-like 菜单点击并武装对话框转移跟踪。

        仅当任务为图片保存意图且 click 命中 STRONG 标签(完整"图片
        另存为"类)时武装(证据层 OCR_STRONG_TRIGGER);同签名短窗内
        重复且上次无对话框时不重复武装。对话框已到达
        (SUCCEEDED_TO_DIALOG)后不再武装——消灭"到达→Enter→又右键"
        循环。目录要求仅在可解析时放行(解析失败保持保守不武装)。
        """
        expectation = self._run_expectation
        if expectation is None or not expectation.save_image_intent:
            return
        if expectation.save_folder_required and (
            resolve_save_folder_location(expectation.expected_save_folder) is None
        ):
            return
        if action["action_type"] != "click":
            return
        x = action["params"].get("x")
        y = action["params"].get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return
        candidate = classify_save_menu_candidate(list(boxes), (x, y))
        if candidate is None or candidate.level != "STRONG":
            return
        pending = self._pending_dialog
        if isinstance(pending, PendingDialogTransition) and (
            pending.dialog_hwnd is not None
            or pending.last_trigger_result == "DIALOG_ARRIVED"
        ):
            # 对话框已到达过:本 task 该触发已 SUCCEEDED_TO_DIALOG。
            return
        if pending is None:
            pending = PendingDialogTransition(
                expected_filename=expectation.expected_save_filename,
            )
            self._pending_dialog = pending
        if pending.is_duplicate_trigger(candidate, step_number):
            diag_log("diag_save_duplicate_trigger", step=step_number)
            return
        pre_dialogs = frozenset(
            window["hwnd"] for window in enumerate_save_dialog_windows()
        )
        trigger_process = process_name_of_hwnd(get_foreground_app_hwnd())
        pending.arm_trigger(
            step_number, "click", candidate, trigger_process, pre_dialogs
        )
        diag_log(
            "diag_save_trigger_armed",
            step=step_number,
            label=candidate.label,
            process=trigger_process or "(unknown)",
            pre_dialogs=len(pre_dialogs),
        )

    def _handle_save_dialog_transition(
        self,
        prompt_state: ActionPromptState,
        manager: TaskManager,
        step_number: int,
        step_retries: int,
    ) -> tuple[ActionPromptState, Msg | None]:
        """P2 主处理:retroactive arm → 到达轮询 → 路线注入 → post-save。

        全部为本地证据计算(有界轮询,零模型调用);弱证据只形成
        policy 事实,绝不提前 finish。OCR 触发未武装时,以真实窗口
        历史差集 retroactively 确认保存对话框(DIALOG_CONFIRMED_TRIGGER)
        并注入同一完成路线。
        """
        expectation = self._run_expectation
        pending = self._pending_dialog
        if expectation is None or not expectation.save_image_intent:
            return prompt_state, None
        if not isinstance(pending, PendingDialogTransition):
            # P2 recovery:dialog-first retroactive arm(真实 pre 快照差集)。
            current = enumerate_save_dialog_windows()
            hit = retroactive_save_dialog_arm(
                self._dialog_history,
                current,
                self._recent_action_types,
                save_intent=True,
            )
            if hit is None:
                return prompt_state, None
            _evidence, window = hit
            new_pending = PendingDialogTransition(
                expected_filename=expectation.expected_save_filename,
            )
            new_pending.dialog_hwnd = window.get("hwnd")
            new_pending.dialog_process = str(window.get("process", ""))
            new_pending.dialog_title = str(window.get("title", ""))
            new_pending.dialog_arrived_step = step_number
            new_pending.last_trigger_result = "DIALOG_ARRIVED"
            new_pending.trigger_process = self._dialog_history.last_foreground_process
            new_pending.pre_dialog_hwnds = frozenset(
                w["hwnd"] for w in self._dialog_history.previous_dialogs
            )
            new_pending.other_dialog_hwnds_at_save = frozenset(
                w["hwnd"] for w in current
            ) - {window.get("hwnd")}
            self._pending_dialog = new_pending
            diag_log(
                "diag_save_retro_arm",
                step=step_number,
                hwnd=window.get("hwnd"),
                process=new_pending.dialog_process or "(unknown)",
            )
            prompt_state = replace(
                prompt_state,
                last_error=(
                    "SAVE_TRIGGER_CONFIRMED_BY_DIALOG：保存对话框已出现并"
                    "进入完成路线,无需再次打开菜单。"
                ),
            )
            pending = new_pending
            return prompt_state, None
        if (
            pending.trigger_step == step_number
            and pending.dialog_hwnd is None
            and pending.last_trigger_result != "NO_DIALOG"
        ):
            diag_log("diag_dialog_wait_begin", step=step_number)
            arrival = poll_for_dialog(
                enumerate_save_dialog_windows,
                pending.pre_dialog_hwnds,
                pending.trigger_process,
                timing=DialogPollTiming(
                    sleep_fn=self._dependencies.sleep,
                ),
            )
            if arrival.status == "ARRIVED":
                pending.dialog_hwnd = arrival.hwnd
                pending.dialog_process = arrival.process
                pending.dialog_title = arrival.title
                pending.dialog_arrived_step = step_number
                pending.last_trigger_result = "DIALOG_ARRIVED"
                arrival_windows = enumerate_save_dialog_windows()
                pending.other_dialog_hwnds_at_save = frozenset(
                    w["hwnd"] for w in arrival_windows
                ) - {arrival.hwnd}
                diag_log(
                    "diag_dialog_arrived",
                    step=step_number,
                    hwnd=arrival.hwnd,
                    process=arrival.process or "(unknown)",
                    title_supports=dialog_title_supports_save(arrival.title),
                )
                # 复用既有 FIX 4 保存路线(经 ActionDispatcher 真实键盘);
                # P2 recovery:目录期望可解析时按四象限 v2 路线导航。
                if self._active_route is None or self._active_route.is_exhausted:
                    folder_location = resolve_save_folder_location(
                        expectation.expected_save_folder
                    )
                    if folder_location is not None:
                        self._active_route = build_save_dialog_route_v2(
                            folder_location,
                            pending.expected_filename,
                        )
                    else:
                        self._active_route = build_save_dialog_route(
                            pending.expected_filename,
                        )
                    pending.save_route_dispatched = True
                    diag_log(
                        "diag_save_route_dispatched",
                        step=step_number,
                        filename=pending.expected_filename or "(default)",
                        folder=folder_location or "(default)",
                    )
            else:
                diag_log("diag_dialog_wait_timeout", step=step_number)
                pending.mark_trigger_failed(step_number)
                prompt_state = replace(
                    prompt_state,
                    last_error=(
                        "SAVE_TRIGGER_ATTEMPT_NO_DIALOG：保存触发后未出现"
                        "保存对话框,请重新感知当前界面后再决定下一步。"
                    ),
                )
            return prompt_state, None
        if (
            pending.save_route_dispatched
            and pending.dialog_hwnd is not None
            and (self._active_route is None or self._active_route.is_exhausted)
        ):
            windows = self._poll_dialog_closed(pending.dialog_hwnd)
            foreground_process = process_name_of_hwnd(get_foreground_app_hwnd())
            result = evaluate_post_save(
                pending.dialog_hwnd,
                windows,
                foreground_process,
                pending.trigger_process,
                pending.other_dialog_hwnds_at_save,
            )
            if result.status == "VERIFIED":
                diag_log(
                    "diag_dialog_closed",
                    step=step_number,
                    hwnd=pending.dialog_hwnd,
                    foreground=foreground_process or "(unknown)",
                )
                self._pending_dialog = None
                self._emit_completion_decision(
                    "post_dialog_save_completion",
                    step_number,
                    "VERIFIED",
                    result.reason,
                )
                return prompt_state, self._complete_early(
                    manager,
                    step_retries,
                    "保存对话框已完成并关闭(post_dialog_save_completion)",
                )
            diag_log(
                (
                    "diag_secondary_dialog"
                    if result.status == "SECONDARY_DIALOG_PRESENT"
                    else "diag_save_post_incomplete"
                ),
                step=step_number,
                reason=result.reason,
            )
            prompt_state = replace(
                prompt_state,
                last_error=f"SAVE_{result.status}：{result.reason}",
            )
        return prompt_state, None

    def _poll_dialog_closed(self, dialog_hwnd: int) -> list[dict]:
        """有界轮询对话框关闭;返回最终窗口列表(供 post-save 判定)。"""
        windows = enumerate_save_dialog_windows()
        deadline = time.monotonic() + CLOSE_POLL_TOTAL_S
        while any(w["hwnd"] == dialog_hwnd for w in windows):
            if time.monotonic() >= deadline:
                break
            self._dependencies.sleep(CLOSE_POLL_INTERVAL_S)
            windows = enumerate_save_dialog_windows()
        return windows

    def _track_transfer_action(
        self,
        action: ParsedAction,
        step_number: int,
        boxes: tuple[dict, ...],
    ) -> None:
        """P3:dispatch 前维护跨应用搬运状态(纯记录,不改动作)。

        drag → 从结构化 OCR box 提取选择区文本形成源签名;
        Ctrl+C → 记录 clipboard 序号前值(只观察,不读内容);
        Ctrl+V → 记录粘贴步;目标窗口内 type 前置标题 → prefix 完成。
        """
        expectation = self._run_expectation
        if expectation is None or not expectation.cross_app_transfer_intent:
            return
        pending = self._pending_transfer
        if pending is None:
            pending = PendingCrossAppTransfer(
                target_app=expectation.transfer_target_app or "",
            )
            pending.target_prefix_text = expectation.transfer_target_prefix_text
            self._pending_transfer = pending
        action_type = action["action_type"]
        # §14:copy 有效且未粘贴时抑制重复源捕获(drag/Ctrl+C),
        # 只记录事实让 policy 走目标获取,不无谓执行重复动作。
        if action_type in ("drag",) and pending.copy_state in (
            "CONFIRMED",
            "PROVISIONAL",
        ):
            diag_log("diag_transfer_duplicate_source_suppressed", step=step_number)
            return
        if action_type == "drag":
            x1 = action["params"].get("x1")
            y1 = action["params"].get("y1")
            x2 = action["params"].get("x2")
            y2 = action["params"].get("y2")
            if all(isinstance(v, int) for v in (x1, y1, x2, y2)):
                drag_bbox = (
                    cast(int, x1),
                    cast(int, y1),
                    cast(int, x2),
                    cast(int, y2),
                )
                selected, _bboxes = extract_selection_text(drag_bbox, list(boxes))
                pending.record_selection(
                    step_number,
                    drag_bbox,
                    selected,
                    get_foreground_app_hwnd(),
                    process_name_of_hwnd(get_foreground_app_hwnd()),
                )
                diag_log(
                    "diag_transfer_selection",
                    step=step_number,
                    selected_len=len(selected),
                    source_process=pending.source_process or "(unknown)",
                )
            return
        if action["action_type"] == "hotkey":
            raw_keys = action["params"]["keys"]
            keys = {str(key) for key in raw_keys}
            if keys == {"ctrl", "c"} and pending.copy_state in ("NONE", "DISPATCHED"):
                pending.copy_step = step_number
                pending.copy_state = "DISPATCHED"
                pending.clipboard_sequence_before = get_clipboard_sequence()
                # P3 stability:pre-dispatch 快照复制时的前台身份——
                # 复制是否在正确 source 执行由此刻事实证明;之后的
                # 前台漂移不能撤销已成立的 copy(§2/§13)。
                pre_hwnd = get_foreground_app_hwnd()
                pending.pre_copy_foreground_hwnd = pre_hwnd
                pending.pre_copy_foreground_process = process_name_of_hwnd(pre_hwnd)
                return
            if keys == {"ctrl", "v"} and not pending.paste_dispatched:
                pending.paste_step = step_number
                pending.paste_dispatched = True
                return
        if (
            action_type == "type"
            and pending.target_window_hwnd is not None
            and pending.target_prefix_text is not None
            and not pending.prefix_completed
        ):
            typed = action["params"].get("text")
            if isinstance(typed, str) and (
                typed.strip() == pending.target_prefix_text.strip()
            ):
                pending.prefix_completed = True
                diag_log("diag_transfer_prefix_typed", step=step_number)

    def _ocr_target_window_text(
        self,
        hwnd: int | None,
        height_fraction: float = 1.0,
    ) -> str:
        """OCR 目标窗口区域文本(P3 pre/post-paste 专用)。

        只识别目标窗口矩形(可见部分截断),避免源浏览器窗口仍在
        桌面上时其文本污染 post-paste 判定(§20 假阳性防线)。
        ``height_fraction`` 允许完成验证只读取应用内稳定的结果带；
        矩形不可用或比例非法时返回空串,由调用方按证据不足处理。
        """
        if not hwnd or not 0 < height_fraction <= 1:
            return ""
        from perception.screenshot import (
            get_window_screen_rect_clipped,
            virtual_desktop_origin,
        )

        rect = get_window_screen_rect_clipped(hwnd)
        if rect is None:
            return ""
        # 窗口矩形是虚拟桌面绝对坐标;capture_screen 的 region 相对
        # monitors[0],必须减去虚拟桌面原点,负原点布局下才不会双重偏移。
        origin = virtual_desktop_origin()
        if origin is None:
            return ""
        capture_height = max(1, round(rect[3] * height_fraction))
        try:
            image = self._dependencies.capture(
                screen_id=0,
                region=(
                    rect[0] - origin[0],
                    rect[1] - origin[1],
                    rect[2],
                    capture_height,
                ),
            )
        except Exception as exception:
            # 证据采集失败静默降级会掩盖系统性截图问题;debug 级留痕,
            # 只记异常类型,不刷 warning。
            logger.debug(
                "ocr_target_window_capture_failed：exception_type=%s",
                type(exception).__name__,
            )
            return ""
        _lines, boxes = prompt_context.perceive_ocr_elements_detailed(
            self._dependencies.ocr_recognizer,
            image,
            None,
        )
        return "".join(str(item.get("text", "")) for item in boxes)

    def _transfer_target_acquired(
        self,
        pending,
        hwnd: int,
        step_number: int,
    ) -> None:
        """标记目标已获取:记录身份 + pre-paste 基线 + 注入粘贴路线。"""
        pending.target_window_hwnd = hwnd
        pending.target_process = process_name_of_hwnd(hwnd)
        pending.target_pre_paste_text = self._ocr_target_window_text(hwnd)
        diag_log(
            "diag_transfer_target_acquired",
            step=step_number,
            hwnd=hwnd,
            process=pending.target_process,
            pre_text_len=len(pending.target_pre_paste_text),
        )
        if pending.target_prefix_text is not None and not pending.prefix_completed:
            if self._active_route is None or self._active_route.is_exhausted:
                self._active_route = build_transfer_paste_route(
                    pending.target_prefix_text,
                )
        else:
            if self._active_route is None or self._active_route.is_exhausted:
                self._active_route = build_transfer_paste_route(None)

    def _transfer_bounded_focus_scan(
        self,
        pending,
        prompt_state,
        step_number,
        target_processes,
        expectation,
    ):
        """有界 GUI-native Alt+Tab 目标搜索(每次一个真实 Alt+Tab)。

        上界 = min(可见可切换窗口数 + 1, 全局上限);Agent UI/Shell/
        来源循环都只继续搜索,不粘贴、不回源重复制;目标进程不存在
        时输出 TRANSFER_TARGET_LAUNCH_NEEDED(app-launch 路线负责)。
        """
        from agent.cross_app_transfer import TARGET_ACQUISITION_GLOBAL_CAP

        visible = prompt_context.perceive_windows((1000, 1000), (0, 0))
        switchable = len(visible) if visible else 0
        max_attempts = min(switchable + 1, TARGET_ACQUISITION_GLOBAL_CAP)
        target_exists = any(
            str(w.get("process", "")).lower() in target_processes for w in visible
        )
        if not target_exists:
            prompt_state = replace(
                prompt_state,
                last_error=(
                    "TRANSFER_TARGET_LAUNCH_NEEDED：目标应用当前未运行,"
                    f"请先打开{expectation.transfer_target_app};"
                    "已复制的内容无需重新获取。"
                ),
            )
            return prompt_state, False
        from agent.action_parser import parse_action

        for attempt in range(1, max_attempts + 1):
            before_hwnd = get_foreground_app_hwnd()
            action, _failure = parse_action('Action: hotkey(key1="alt", key2="tab")')
            self._dependencies.action_dispatcher.dispatch(
                action,
                (1000, 1000),
                (0, 0),
            )
            pending.focus_scan_dispatched += 1
            self._recent_action_types.append("hotkey")
            self._dependencies.sleep(FOCUS_SETTLE_SECONDS)
            after_hwnd = get_foreground_app_hwnd()
            after_process = process_name_of_hwnd(after_hwnd)
            diag_log(
                "diag_transfer_focus_scan",
                step=step_number,
                attempt=attempt,
                from_process=process_name_of_hwnd(before_hwnd) or "(unknown)",
                to_process=after_process or "(unknown)",
            )
            if after_process.lower() in target_processes:
                self._transfer_target_acquired(pending, after_hwnd, step_number)
                return prompt_state, True
        prompt_state = replace(
            prompt_state,
            last_error=(
                "TRANSFER_TARGET_FOCUS_NOT_ACQUIRED：有界搜索已用尽"
                f"(attempts={max_attempts});请重新感知界面后再决定。"
            ),
        )
        return prompt_state, False

    def _handle_cross_app_transfer(
        self,
        prompt_state: ActionPromptState,
        manager: TaskManager,
        step_number: int,
        step_retries: int,
    ) -> tuple[ActionPromptState, Msg | None]:
        """P3 主处理:复制确认 → 目标焦点 → 粘贴路线 → 转移验证。

        全部本地证据(clipboard 序号只观察变化;焦点按通用进程身份
        匹配;post-paste 用目标窗口 OCR 做内容转移验证)。弱证据只
        形成 policy 事实,绝不提前 finish;已确认复制后不回退重复制。
        """
        expectation = self._run_expectation
        pending = self._pending_transfer
        if (
            expectation is None
            or not expectation.cross_app_transfer_intent
            or not isinstance(pending, PendingCrossAppTransfer)
        ):
            return prompt_state, None
        target_processes = {
            name.lower() for name in expectation.transfer_target_app_processes
        }
        # 1) 复制状态确认:pre-dispatch source 身份 + 序号变化(§2-4)。
        #    pre-dispatch 时刻前台==记录的 source 才可能 CONFIRMED;
        #    Ctrl+C 后的前台漂移只分类记录,不撤销 copy(§13)。
        if pending.copy_state == "DISPATCHED" and pending.copy_step == step_number:
            pending.clipboard_sequence_after = get_clipboard_sequence()
            sequence_changed = (
                pending.clipboard_sequence_after is not None
                and pending.clipboard_sequence_before is not None
                and pending.clipboard_sequence_after
                != pending.clipboard_sequence_before
            )
            pre_source_ok = (
                pending.selection_evidence_sufficient
                and bool(pending.source_process)
                and pending.pre_copy_foreground_process
                and pending.pre_copy_foreground_process.lower()
                == pending.source_process.lower()
            )
            post_hwnd = get_foreground_app_hwnd()
            post_process = process_name_of_hwnd(post_hwnd)
            if pre_source_ok and sequence_changed:
                pending.copy_confirmed = True
                pending.copy_state = "CONFIRMED"
                # 前台漂移三分类(§13):target/source/unrelated。
                if post_process.lower() in target_processes:
                    drift = "POST_COPY_FOREGROUND_IS_TARGET"
                elif (
                    pending.source_process
                    and post_process.lower() == pending.source_process.lower()
                ):
                    drift = "POST_COPY_FOREGROUND_STILL_SOURCE"
                else:
                    drift = "POST_COPY_FOREGROUND_DRIFT"
                diag_log(
                    "diag_transfer_copy_confirmed",
                    step=step_number,
                    drift=drift,
                    post_process=post_process or "(unknown)",
                )
            elif pre_source_ok and not sequence_changed:
                pending.copy_state = "PROVISIONAL"
                diag_log(
                    "diag_transfer_copy_provisional",
                    step=step_number,
                    reason="sequence_unchanged_or_unavailable",
                )
            else:
                pending.copy_state = "NONE"
                diag_log(
                    "diag_transfer_copy_rejected",
                    step=step_number,
                    reason="pre_dispatch_identity_or_selection_insufficient",
                )
            return prompt_state, None
        # 2) 目标窗口获取(§6-13):copy 有效(PROVISIONAL 起)即进入
        #    bounded GUI-native Alt+Tab 搜索;目标已前台零切换直接
        #    ACQUIRED;搜索有界(可见窗口数+1 与全局上限取小)。
        copy_state_valid = pending.copy_state in ("CONFIRMED", "PROVISIONAL")
        if (
            copy_state_valid
            and pending.target_window_hwnd is None
            and not pending.paste_dispatched
        ):
            foreground_hwnd = get_foreground_app_hwnd()
            foreground_process = process_name_of_hwnd(foreground_hwnd)
            if foreground_process.lower() in target_processes:
                self._transfer_target_acquired(pending, foreground_hwnd, step_number)
            else:
                prompt_state, _acquired = self._transfer_bounded_focus_scan(
                    pending,
                    prompt_state,
                    step_number,
                    target_processes,
                    expectation,
                )
            if pending.target_window_hwnd is not None:
                prompt_state = replace(
                    prompt_state,
                    last_error=(
                        "TRANSFER_TARGET_ACQUIRED：目标窗口已就绪,"
                        "粘贴路线已准备,无需再次切换窗口或重新复制。"
                    ),
                )
            return prompt_state, None
        # 3) post-paste 验证:目标窗 OCR 的 source 新增转移。
        if pending.paste_dispatched and not pending.transfer_verified:
            post_text = self._ocr_target_window_text(pending.target_window_hwnd)
            if not post_text:
                prompt_state = replace(
                    prompt_state,
                    last_error="PASTE_VERIFICATION_INSUFFICIENT：目标窗口OCR无证据",
                )
                return prompt_state, None
            foreground_process = process_name_of_hwnd(get_foreground_app_hwnd())
            verdict = evaluate_transfer_transition(
                pending,
                post_text,
                get_foreground_app_hwnd(),
                foreground_process,
            )
            if verdict.status == "VERIFIED":
                pending.transfer_verified = True
                self._last_transfer_evidence = dict(verdict.evidence)
                diag_log(
                    "diag_transfer_verified",
                    step=step_number,
                    post_overlap=verdict.evidence.get("post_overlap"),
                    source_digest=verdict.evidence.get("source_digest"),
                )
                self._pending_transfer = None
                self._emit_completion_decision(
                    "cross_app_content_transfer",
                    step_number,
                    "VERIFIED",
                    verdict.reason,
                )
                return prompt_state, self._complete_early(
                    manager,
                    step_retries,
                    "跨应用内容搬运完成(cross_app_content_transfer)",
                )
            diag_log(
                "diag_transfer_insufficient",
                step=step_number,
                reason=verdict.reason,
                post_overlap=verdict.evidence.get("post_overlap"),
            )
            prompt_state = replace(
                prompt_state,
                last_error=f"PASTE_VERIFICATION_INSUFFICIENT：{verdict.reason}",
            )
        return prompt_state, None

    def _proactive_completion_check(
        self,
        prompt_state: ActionPromptState,
        manager: TaskManager,
        step_number: int,
        step_retries: int,
        focus_screen: tuple[int, int] | None,
    ) -> Msg | None:
        """OBSERVABILITY_ONLY 薄包装:为完成检测记录 diag 阶段计时。"""
        with diag_phase("diag_completion", step=step_number):
            return self._proactive_completion_check_inner(
                prompt_state,
                manager,
                step_number,
                step_retries,
                focus_screen,
            )

    def _proactive_completion_check_inner(
        self,
        prompt_state: ActionPromptState,
        manager: TaskManager,
        step_number: int,
        step_retries: int,
        focus_screen: tuple[int, int] | None,
    ) -> Msg | None:
        """动作后主动完成检测;VERIFIED 提前成功,否则缓存感知继续。

        音量/窗口关闭类只读廉价结构化事实;数值/文本类对 post-action
        稳定帧做一次感知,未完成时同一份感知缓存给下一次模型调用,
        不产生重复 capture/OCR,也不增加任何模型调用。
        """
        expectation = self._run_expectation
        if expectation is None or expectation.is_empty():
            return None
        pending = self._pending_submission
        delivery_armed = (
            isinstance(pending, PendingSubmission) and pending.submit_step is not None
        )
        needs_ocr = (
            expectation.expected_numeric_result is not None
            or expectation.expected_text is not None
            or delivery_armed
        )
        ocr_text = ""
        ocr_boxes: tuple[dict, ...] = ()
        if needs_ocr:
            capture = self._capture(manager, step_number)
            if capture is None:
                return None
            image, region_offset = capture
            image, region_offset, ocr_lines, ocr_boxes = self._perceive_with_escalation(
                image, region_offset, focus_screen
            )
            self._pending_observation = (image, region_offset, ocr_lines, ocr_boxes)
            ocr_text = "\n".join(ocr_lines)
            if (
                expectation.expected_numeric_result is not None
                and expectation.expected_app is not None
            ):
                ocr_text = self._ocr_target_window_text(
                    get_foreground_app_hwnd(),
                    _NUMERIC_RESULT_BAND_HEIGHT_FRACTION,
                )
                ocr_text = _evaluated_calculator_result_text(ocr_text)
        # P1:submit 后对 post-action 感知做投递状态转移判定;强证据
        # 成立才提前成功,弱证据一律 INSUFFICIENT 并解除本次提交记录。
        if delivery_armed and isinstance(pending, PendingSubmission):
            verdict = evaluate_delivery_transition(
                pending,
                list(ocr_boxes),
                get_foreground_app_hwnd(),
            )
            self._last_delivery_evidence = dict(verdict.evidence)
            # OBSERVABILITY_ONLY:投递判定事实计数,不改变 verdict。
            diag_log(
                "diag_delivery_facts",
                step=step_number,
                verdict=verdict.status,
                pre_occurrences=verdict.evidence.get("pre_occurrence_count"),
                post_occurrences=len(
                    cast(
                        list[object],
                        verdict.evidence.get("post_occurrence_bboxes") or [],
                    )
                ),
                composer_cleared=bool(verdict.evidence.get("composer_cleared")),
            )
            if verdict.status == "VERIFIED":
                self._pending_submission = None
                self._emit_completion_decision(
                    "post_submit_delivery",
                    step_number,
                    "VERIFIED",
                    verdict.reason,
                )
                return self._complete_early(manager, step_retries, verdict.reason)
            pending.disarm_submit()
        result = verify_completion(
            expectation,
            self._completion_facts(prompt_state, ocr_text),
        )
        self._last_completion_evidence = dict(result.evidence) or {
            "foreground_process": "unknown",
        }
        if result.status == "VERIFIED":
            self._emit_completion_decision(
                "proactive_post_action",
                step_number,
                result.status,
                result.reason,
            )
            return self._complete_early(manager, step_retries, result.reason)
        return None

    def _final_max_steps_check(
        self,
        prompt_state: ActionPromptState,
        manager: TaskManager,
    ) -> Msg | None:
        """max_steps 兜底终检:程序证据确凿时按成功结束。"""
        expectation = self._run_expectation
        if expectation is None or expectation.is_empty():
            return None
        result = self._verify_completion_proposal(prompt_state)
        if result.status != "VERIFIED":
            return None
        self._emit_completion_decision(
            "max_steps_final_check",
            self._settings.max_steps,
            result.status,
            result.reason,
        )
        return self._complete_early(manager, 0, result.reason)

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
        raw_points: tuple[tuple[int, int], ...]
        if action["action_type"] == "drag":
            drag_params = action["params"]
            raw_points = (
                (drag_params["x1"], drag_params["y1"]),
                (drag_params["x2"], drag_params["y2"]),
            )
        elif is_click_like_action(action):
            point_params = action["params"]
            raw_points = ((point_params["x"], point_params["y"]),)
        else:
            return None
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

    def _step_extension_evidence(
        self,
        prompt_state: ActionPromptState,
        manager: TaskManager,
        step_succeeded: bool,
        focus_transition: bool,
        change_ratio: float,
        step_retries: int,
    ) -> StepExtensionEvidence:
        """把 base 边界的通用运行态收敛为扩展资格证据。"""
        strong_effect = prompt_state.last_effect in {
            "foreground_window_changed",
            "window_closed",
        }
        measurable_visual_progress = change_ratio >= 0.005
        quantitative_progress = prompt_state.progress_status in {
            "PROGRESSED",
            "INSUFFICIENT_RATE",
        }
        route_result_unverified = bool(
            isinstance(self._pending_dialog, PendingDialogTransition)
            and self._pending_dialog.save_route_dispatched
            and self._pending_dialog.dialog_hwnd is not None
            and (self._active_route is None or self._active_route.is_exhausted)
        )
        same_strategy_retry_exhausted = route_result_unverified
        state = manager.state
        if not step_succeeded and step_retries >= self._settings.retry_count:
            latest = state.steps[-1] if state.steps else None
            serialized = {
                self._serialize_action(cast(ParsedAction, attempt.action))
                for attempt in (latest.attempts if latest is not None else [])
                if attempt.action is not None
            }
            same_strategy_retry_exhausted = len(serialized) <= 1
        meaningful_progress = bool(
            step_succeeded
            and (
                strong_effect
                or focus_transition
                or measurable_visual_progress
                or quantitative_progress
            )
        )
        return StepExtensionEvidence(
            task_unfinished=True,
            recent_step_succeeded=step_succeeded,
            meaningful_progress=meaningful_progress,
            latest_dispatch_transition=meaningful_progress,
            repeated_action_blocked=(
                prompt_state.previous_strategy_failed
                or prompt_state.blocked_repeated_action != "none"
            ),
            no_progress_active=(
                prompt_state.progress_status == "NO_PROGRESS"
                or prompt_state.no_ui_change_streak > 0
            ),
            same_strategy_retry_exhausted=same_strategy_retry_exhausted,
            safety_blocked=(
                not step_succeeded and prompt_state.last_dispatch_status == "failure"
            ),
            unrecoverable_error=False,
        )

    def _record_step_budget_trace(
        self,
        budget: ProgressDependentStepBudget,
        step_number: int,
    ) -> None:
        """写入不含任务文本的统一 step-budget trace。"""
        writer = self._dependencies.trace_writer
        if writer is None:
            return
        writer.record_model_call(
            {
                "record_type": "step_budget",
                "run_id": self._current_run_id,
                "task_id": self._current_task_id,
                "step": step_number,
                **budget.trace_fields(),
            },
        )

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

    def _perceive_with_escalation(
        self,
        image: Image.Image,
        region_offset: tuple[int, int],
        focus_screen: tuple[int, int] | None,
    ) -> tuple[Image.Image, tuple[int, int], tuple[str, ...], tuple[dict, ...]]:
        """ROI 优先感知:region 感知无任何元素时升级为全帧重识别。

        前台窗口 region 的 OCR 结果为空(如窗口矩形漂移到无文字区域、
        或目标内容不在当前前台窗口内)且当前图不是全帧时,对同一步的
        完整帧重跑一次感知,并把图像与 region_offset 一并切回全帧,
        保证 OCR 文本与 click 映射处于同一坐标系;焦点锚点按全帧坐标
        直接使用,与全屏步骤的既有行为一致。无识别器、已是全帧或
        全帧缺失时保持原结果,不产生第二次识别。
        """
        recognizer = self._dependencies.ocr_recognizer
        focus_local = _focus_local_point(focus_screen, region_offset)
        ocr_lines, boxes = prompt_context.perceive_ocr_elements_detailed(
            recognizer,
            image,
            focus_local,
        )
        full = self._latest_full
        if (
            recognizer is not None
            and not ocr_lines
            and full is not None
            and full is not image
            and full.size != image.size
        ):
            diag_log("diag_roi_escalation")
            ocr_lines, boxes = prompt_context.perceive_ocr_elements_detailed(
                recognizer,
                full,
                focus_screen,
            )
            return full, (0, 0), ocr_lines, boxes
        return image, region_offset, ocr_lines, boxes

    def _capture(
        self,
        manager: TaskManager,
        step_number: int,
    ) -> tuple[Image.Image, tuple[int, int]] | None:
        """OBSERVABILITY_ONLY 薄包装:为截图记录 diag 阶段计时与尺寸。"""
        with diag_phase("diag_screenshot", step=step_number):
            result = self._capture_inner(manager, step_number)
        if result is not None:
            image, _offset = result
            diag_log(
                "diag_screenshot_stats",
                step=step_number,
                image_w=image.size[0],
                image_h=image.size[1],
            )
        return result

    def _capture_inner(
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
        """OBSERVABILITY_ONLY 薄包装:动作后稳定等待阶段计时。"""
        with diag_phase("diag_ui_stable"):
            return self._wait_for_ui_stable_inner()

    def _wait_for_ui_stable_inner(self) -> Image.Image | None:
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
            int(cast(int, self._task_target["hwnd"])),
        ):
            effect: PromptEffect = "window_closed"
        elif before_app_foreground and not is_window_existing(before_app_foreground):
            effect = "window_closed"
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
    def _window_title_matches(hwnd: int, keywords: tuple[str, ...]) -> bool:
        """只在内存中判断窗口标题是否命中应用别名，不返回或记录标题。"""
        if not hwnd or not keywords:
            return False
        try:
            import ctypes

            user32 = ctypes.windll.user32
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return False
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.casefold()
            return any(keyword.casefold() in title for keyword in keywords)
        except (AttributeError, OSError):
            return False

    def _decision_protocol_version(self) -> Literal["v1", "v2", "v3"]:
        """返回本次模型决策实际使用的协议版本。"""
        if self._settings.decision_protocol_v3:
            return "v3"
        if self._settings.decision_protocol_v2:
            return "v2"
        return "v1"

    def _trace_image_evidence(
        self,
        image: Image.Image,
        step_number: int,
        attempt: int,
        phase: Literal["before", "after"],
    ) -> tuple[str | None, str | None]:
        """保存 trace 截图并返回像素摘要和相对路径。

        仅在显式配置 trace writer 时工作；任何摘要或写入异常都退化为
        ``(None, None)``，不得影响任务动作与控制流。
        """
        writer = self._dependencies.trace_writer
        if writer is None:
            return None, None
        trace_phase = f"attempt_{attempt:02d}_{phase}"
        path = f"{self._current_run_id}/" f"step_{step_number:02d}_{trace_phase}.png"
        try:
            digest = hashlib.sha256()
            digest.update(image.mode.encode("ascii", errors="replace"))
            digest.update(str(image.size).encode("ascii"))
            digest.update(image.tobytes())
            save = getattr(writer, "save_step_screenshot", None)
            if callable(save):
                save(
                    self._current_run_id,
                    step_number,
                    trace_phase,
                    image,
                )
                return digest.hexdigest(), path
            return digest.hexdigest(), None
        except Exception as exception:
            logger.warning(
                "agent_trace_image_evidence_failed：exception_type=%s",
                type(exception).__name__,
            )
            return None, None

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
        if action["action_type"] == "click":
            return f"click(x={action['params']['x']}, y={action['params']['y']})"
        if action["action_type"] == "right_click":
            return (
                f"right_click(x={action['params']['x']}, " f"y={action['params']['y']})"
            )
        if action["action_type"] == "double_click":
            return (
                f"double_click(x={action['params']['x']}, "
                f"y={action['params']['y']})"
            )
        if action["action_type"] == "drag":
            return (
                f"drag(x1={action['params']['x1']}, y1={action['params']['y1']}, "
                f"x2={action['params']['x2']}, y2={action['params']['y2']})"
            )
        if action["action_type"] == "type":
            text = json.dumps(action["params"]["text"], ensure_ascii=False)
            return f"type(text={text})"
        if action["action_type"] == "scroll":
            direction = json.dumps(action["params"]["direction"])
            steps = action["params"]["steps"]
            return f"scroll(direction={direction}, steps={steps})"
        if action["action_type"] == "hotkey":
            keys = action["params"]["keys"]
            arguments = ", ".join(
                f"key{index}={json.dumps(key)}" for index, key in enumerate(keys, 1)
            )
            return f"hotkey({arguments})"
        if action["action_type"] == "observe":
            return _OBSERVE_ACTION_TEXT
        result = json.dumps(action["params"]["result"], ensure_ascii=False)
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
        dispatch_status: PromptStatus = "success" if dispatched else "failure"
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
            consecutive_observe_count=0,
            previous_strategy_failed=False,
            blocked_repeated_action="none",
            structured_entry_commit_feedback=(
                state.structured_entry_commit_feedback
                and self._pending_structured_entry_commit
            ),
        )

    def _generate_action(
        self,
        image: Image.Image,
        task: str,
        state: ActionPromptState,
        attempt: int,
    ) -> tuple[ParsedAction | None, str | None]:
        """生成并严格解析一个动作，且只进入一次 ModelClient 调用边界。"""
        # FINAL FIX 2+3:确定性语义路线优先于模型调用。
        route_action = self._consume_semantic_route_step(task)
        if route_action is not None:
            return route_action, None
        return self._generate_model_action(image, task, state, attempt)

    def _consume_semantic_route_step(
        self,
        task: str,
    ) -> ParsedAction | None:
        """消费当前路线的下一步;无路线或耗尽返回 None。

        wait-only 步(目录导航后的 settle 等待)就地睡眠并继续消费
        下一步,不落回模型调用。
        """
        if not self._settings.semantic_execution:
            return None
        # FIX 2:Save-As 对话框检测(动态触发,不在 initialize 时);
        # P2 recovery:legacy 检出同时向转移状态上报(P2 感知源)。
        if self._active_route is None or self._active_route.is_exhausted:
            save_route = self._check_save_dialog_route(task)
            if save_route is not None:
                self._active_route = save_route
        if self._active_route is None or self._active_route.is_exhausted:
            calculator_route = self._check_calculator_expression_route(task)
            if calculator_route is not None:
                self._active_route = calculator_route
        while self._active_route is not None and not self._active_route.is_exhausted:
            step = self._active_route.next_step()
            if step is None:
                return None
            if step.wait_seconds > 0:
                self._dependencies.sleep(step.wait_seconds)
            if step.action is None:
                continue
            logger.info(
                "semantic_route_step：route=%s，desc=%s",
                self._active_route.name,
                step.description,
            )
            return step.action
        return None

    def _check_calculator_expression_route(
        self,
        task: str,
    ) -> SemanticRoute | None:
        """当前台可靠属于 Calculator 时构造一次安全键盘算式路线。"""
        if self._calculator_expression_submitted:
            return None
        expression = extract_calculator_expression(task)
        if expression is None:
            return None
        foreground = get_foreground_app_hwnd()
        process = process_name_of_hwnd(foreground)
        title_matches = self._window_title_matches(
            foreground,
            calculator_title_keywords(),
        )
        if not calculator_foreground_is_reliable(process, title_matches):
            return None
        self._calculator_expression_submitted = True
        logger.info("application_capability_activated：calculator_keyboard_expression")
        return build_calculator_expression_route(expression)

    def _generate_model_action(
        self,
        image: Image.Image,
        task: str,
        state: ActionPromptState,
        attempt: int,
    ) -> tuple[ParsedAction | None, str | None]:
        """生成并严格解析一个动作，且只进入一次 ModelClient 调用边界。"""
        started_at = time.time()
        usage: dict[str, object] = {}

        def emit_trace(
            raw_response: str | None,
            normalized_response: str | None,
            parsed: ParsedAction | None,
            extras: _TraceExtras | None = None,
        ) -> None:
            repaired_response = extras.repaired_response if extras else None
            repair_types = extras.repair_types if extras else None
            parse_error = extras.parse_error if extras else None
            raw_parse_success = extras.raw_parse_success if extras else None
            normalization_reason = extras.normalization_reason if extras else None
            writer = self._dependencies.trace_writer
            if writer is None:
                return
            writer.record_model_call(
                {
                    "record_type": "model_call",
                    "run_id": self._current_run_id,
                    "task_id": self._current_task_id,
                    "task_digest": text_digest(task),
                    "step": state.step_number,
                    "attempt": attempt,
                    "protocol_version": self._decision_protocol_version(),
                    "model": self._settings.model_mode,
                    "backend": self._run_backend_identity,
                    "model_identifier_or_path": self._run_backend_identity,
                    "logical_step": state.step_number,
                    "fresh_attempt": attempt,
                    "image_width": scaled.width,
                    "image_height": scaled.height,
                    "prompt_character_count": len(prompt_text),
                    "request_started_at": started_at,
                    "request_finished_at": time.time(),
                    "api_latency_ms": round((time.time() - started_at) * 1000),
                    "model_latency": round((time.time() - started_at) * 1000),
                    "raw_model_response": raw_response,
                    "repaired_model_response": repaired_response,
                    "repair_types": repair_types,
                    "normalized_model_response": normalized_response,
                    "normalization_reason": normalization_reason,
                    "parse_error": parse_error,
                    "backend_error_summary": backend_error_summary,
                    "raw_parse_success": raw_parse_success,
                    "parsed_action": (
                        self._serialize_action(parsed) if parsed is not None else None
                    ),
                    "parse_success": parsed is not None,
                    "finish_proposed": (
                        parsed is not None and parsed["action_type"] == "finish"
                    ),
                    "completion_verification": state.completion_verification,
                    "completion_reason": state.completion_reason,
                    "completion_evidence": self._last_completion_evidence,
                    "progress_status": state.progress_status,
                    "progress_reason": state.progress_reason,
                    "steps_remaining": state.steps_remaining,
                    "structured_facts": {
                        "system_volume_percent": state.system_volume_percent,
                    },
                    "grounding_candidates": self._last_grounding_candidates,
                    "candidate_count": len(self._last_grounding_candidates),
                    "perception_summary": {
                        "ocr_element_count": len(state.ocr_elements),
                        "window_count": len(state.windows),
                        "grounding_candidate_count": len(
                            self._last_grounding_candidates
                        ),
                    },
                    "symbolic_grounding_resolution": (
                        {
                            "status": "resolved",
                            "canonical_action": self._serialize_action(parsed),
                            "normalization_reason": normalization_reason,
                        }
                        if parsed is not None
                        and normalization_reason
                        == "action_adapt_001_symbolic_grounding_click"
                        else None
                    ),
                    "structured_entry_commit_pending": (
                        self._pending_structured_entry_commit
                    ),
                    "input_tokens": usage.get("input_tokens", "unknown"),
                    "output_tokens": usage.get("output_tokens", "unknown"),
                    "cached_tokens": usage.get("cached_tokens", "unknown"),
                    "same_action_streak": state.same_action_streak,
                    "consecutive_observe_count": state.consecutive_observe_count,
                    "previous_strategy_failed": state.previous_strategy_failed,
                    "foreground_before_model": self._window_identity(
                        get_foreground_hwnd(),
                    ),
                    "visible_windows": list(state.windows),
                    "retry_reason": state.last_error if attempt > 0 else None,
                    "screenshot_sha256": trace_screenshot_sha256,
                    "screenshot_path": trace_screenshot_path,
                }
            )

        scaled = image
        (
            trace_screenshot_sha256,
            trace_screenshot_path,
        ) = self._trace_image_evidence(
            image,
            state.step_number,
            attempt,
            "before",
        )
        prompt_text = ""
        if self._settings.decision_protocol_v3:
            if self._settings.model_mode == "local":
                # LOCAL-ONLY Compact Prompt(键盘优先):2B 已被诊断连续
                # 坐标 grounding 不可靠,精简状态+通用键盘策略;API 的
                # Clean V3 路径完全不受影响。
                prompt_text = compose_local_compact_prompt(
                    task,
                    state,
                    self._settings.coordinate_mode,
                )
                system_prompt: str | None = None
                temperature: float | None = None
            else:
                prompt_text = compose_action_prompt_v3(
                    task,
                    state,
                    self._settings.coordinate_mode,
                )
                system_prompt = ACTION_SYSTEM_PROMPT_V3
                temperature = 0.0
        elif self._settings.decision_protocol_v2:
            prompt_text = compose_action_prompt_v2(task, state)
            system_prompt = ACTION_SYSTEM_PROMPT_V2
            temperature = 0.0
            if self._settings.model_mode != "api":
                # 本地后端不支持 role 分层,合并为单文本(文档已记录)。
                prompt_text = f"{system_prompt}\n\n{prompt_text}"
                temperature = None
        else:
            prompt_text = compose_action_prompt(
                task,
                state,
                self._settings.coordinate_mode,
            )
            system_prompt = None
            temperature = None
        # 旁路日志:backend 异常的脱敏摘要(项目固定异常文本,无凭据)。
        backend_error_summary: str | None = None
        try:
            max_dim = (
                local_model_image_max_dim_from_env()
                if self._settings.model_mode == "local"
                else model_image_max_dim_from_env()
            )
            scaled = prompt_context.scale_image_for_model(
                image,
                max_dim,
            )
            call_options = ModelCallOptions(
                system_prompt=system_prompt,
                temperature=temperature,
                usage_out=(
                    usage if self._dependencies.trace_writer is not None else None
                ),
            )
            response = self._dependencies.model_client.generate(
                scaled,
                prompt_text,
                self._settings.model_mode,
                options=call_options,
            )
        except Exception as exception:
            backend_error_summary = f"{type(exception).__name__}:{str(exception)[:120]}"
            logger.warning(
                "gui_agent_model_call_failed：exception_type=%s",
                type(exception).__name__,
            )
            self._record_diagnostics(
                DiagnosticsRecord(
                    run_id=self._current_run_id,
                    step_number=state.step_number,
                    attempt=attempt,
                    image=image,
                    prompt=prompt_text,
                    response=None,
                    failure_reason=_MODEL_FAILURE_REASON,
                    exception_type=type(exception).__name__,
                )
            )
            emit_trace(None, None, None)
            return None, _MODEL_FAILURE_REASON
        if response in _TERMINAL_MODEL_FAILURE_RESPONSES:
            self._record_diagnostics(
                DiagnosticsRecord(
                    run_id=self._current_run_id,
                    step_number=state.step_number,
                    attempt=attempt,
                    image=image,
                    prompt=prompt_text,
                    response=response,
                    failure_reason=_TERMINAL_MODEL_FAILURE,
                )
            )
            emit_trace(response, response, None)
            return None, _TERMINAL_MODEL_FAILURE
        if response == _API_EXHAUSTED_MODEL_RESPONSE:
            # API 重试已在 ModelClient 内耗尽(1+3);按 PRD 4.5.1 记录错误
            # 并结束本步,由外层步循环继续下一步,不在 Agent 层重开重试预算。
            self._record_diagnostics(
                DiagnosticsRecord(
                    run_id=self._current_run_id,
                    step_number=state.step_number,
                    attempt=attempt,
                    image=image,
                    prompt=prompt_text,
                    response=response,
                    failure_reason=_API_EXHAUSTED_MODEL_FAILURE,
                )
            )
            emit_trace(response, response, None)
            return None, _API_EXHAUSTED_MODEL_FAILURE
        if response in _MODEL_FAILURE_RESPONSES:
            self._record_diagnostics(
                DiagnosticsRecord(
                    run_id=self._current_run_id,
                    step_number=state.step_number,
                    attempt=attempt,
                    image=image,
                    prompt=prompt_text,
                    response=response,
                    failure_reason=_MODEL_FAILURE_REASON,
                )
            )
            emit_trace(response, response, None)
            return None, _MODEL_FAILURE_REASON
        # LOCAL-ONLY 表面修复:API 模式完全跳过(repaired == raw),
        # 决策链 repair → normalize → strict parser 与 API 完全一致。
        repaired_response = response
        repair_types: list[str] = []
        raw_parse_success: bool | None = None
        parse_model_action = (
            parse_prd_action if self._settings.decision_protocol_v3 else parse_action
        )
        classify_model_error = (
            classify_prd_action_parse_error
            if self._settings.decision_protocol_v3
            else classify_action_parse_error
        )
        if self._settings.model_mode == "local":
            repaired_response, repair_types = repair_local_output(response)
            raw_parse_success = parse_model_action(response) is not None
        adapted = adapt_action_response(
            repaired_response,
            self._last_grounding_candidates,
            grounding_turn_token=(
                f"{self._current_run_id}:{state.step_number}:{attempt}"
            ),
        )
        normalized_final = adapted.normalized_response
        normalization_reason = adapted.normalization_reason
        action = parse_model_action(normalized_final)
        if action is None:
            category = classify_model_error(normalized_final) or "unknown"
            self._record_diagnostics(
                DiagnosticsRecord(
                    run_id=self._current_run_id,
                    step_number=state.step_number,
                    attempt=attempt,
                    image=image,
                    prompt=prompt_text,
                    response=response,
                    failure_reason=f"{_PARSE_FAILURE_REASON} ({category})",
                )
            )
            emit_trace(
                response,
                normalized_final,
                None,
                _TraceExtras(
                    repaired_response,
                    repair_types,
                    category,
                    raw_parse_success,
                    normalization_reason,
                ),
            )
            return None, f"{_PARSE_FAILURE_REASON} ({category})"
        emit_trace(
            response,
            normalized_final,
            action,
            _TraceExtras(
                repaired_response,
                repair_types,
                None,
                raw_parse_success,
                normalization_reason,
            ),
        )
        return action, None

    def _record_diagnostics(self, record: DiagnosticsRecord) -> None:
        """把失败事件交给可选诊断写入器;诊断失败不影响任务流程。"""
        writer = self._dependencies.diagnostics_writer
        if writer is None:
            return
        try:
            writer.record(record)
        except Exception as exception:
            logger.warning(
                "gui_agent_diagnostics_failed：exception_type=%s",
                type(exception).__name__,
            )

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
    def _extract_agent_ui_hwnd(msg: Msg) -> int:
        """从消息 metadata 提取提交任务时的 Agent UI 窗口句柄。"""
        metadata = getattr(msg, "metadata", None)
        if not isinstance(metadata, dict):
            return 0
        hwnd = metadata.get("agent_ui_window_hwnd")
        if type(hwnd) is not int or hwnd <= 0:
            return 0
        return hwnd

    @staticmethod
    def _extract_trace_task_id(msg: Msg) -> str:
        """提取仅用于 trace 关联的短任务标识，非法或缺失时返回 unknown。"""
        metadata = getattr(msg, "metadata", None)
        if not isinstance(metadata, dict):
            return "unknown"
        task_id = metadata.get("trace_task_id")
        if not isinstance(task_id, str):
            return "unknown"
        normalized = task_id.strip()
        if not normalized or len(normalized) > 64:
            return "unknown"
        if not all(char.isalnum() or char in "_-" for char in normalized):
            return "unknown"
        return normalized

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
        return Msg(
            _AGENT_NAME,
            content,
            "assistant",
            metadata=metadata,  # type: ignore[arg-type]  # AgentScope 递归 JSON 类型
        )

    def _failure_message(
        self,
        reason: str,
        manager: TaskManager | None = None,
    ) -> Msg:
        """创建固定失败消息，不包含模型原文或用户数据。"""
        logger.warning("gui_agent_task_failed：reason=%s", reason)
        return self._result_message(f"任务执行失败：{reason}", manager)
