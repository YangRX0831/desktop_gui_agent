"""提供桌面 GUI 智能体的运行配置与环境变量解析。

``AppConfig`` 保存命令行入口与 ``GuiAgent`` 之间的运行配置快照。默认
``max_steps`` 和 ``retry_count`` 统一在本模块定义，避免入口与编排层重复
维护数值。

所有类型和值域都在配置对象构造时验证。环境变量读取函数只负责解析配置，
不会创建目录、加载模型或修改系统环境。本模块不保存 API 密钥；DashScope
凭据由专用后端在调用边界读取。
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

ModelMode = Literal["local", "api"]
CoordinateMode = Literal["image_pixel", "normalized_1000"]
LocalRuntime = Literal["transformers", "openvino"]
# PRD 4.4.1 默认基础步数为 10，可通过 CLI --max-steps 调整。
DEFAULT_MAX_STEPS = 10
DEFAULT_RETRY_COUNT = 3
# 决策协议版本开关。V2 保留 observe 等兼容行为，V3 为当前默认协议。
DECISION_PROTOCOL_V2_ENV = "GUI_AGENT_DECISION_PROTOCOL_V2"
DECISION_PROTOCOL_V3_ENV = "GUI_AGENT_DECISION_PROTOCOL_V3"
# 测试运行期间可最小化 Agent 自身控制窗口，避免其进入模型截图。
HIDE_OWN_WINDOW_ENV = "GUI_AGENT_HIDE_OWN_WINDOW_DURING_RUN"
# 程序化完成验证与进度状态开关。未显式设置时由调用方结合协议版本决定。
SEMANTIC_EXECUTION_ENV = "GUI_AGENT_SEMANTIC_EXECUTION"
# observe() 的固定等待秒数由程序控制，模型不能指定。
OBSERVE_WAIT_SECONDS = 0.6
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = Path("logs")
LOCAL_MODEL_DIR_ENV = "GUI_AGENT_LOCAL_MODEL_DIR"
# 本地推理支持 Transformers 与 OpenVINO 两种运行时，由环境变量显式选择。
LOCAL_RUNTIME_ENV = "GUI_AGENT_LOCAL_RUNTIME"
OPENVINO_MODEL_DIR_ENV = "GUI_AGENT_OPENVINO_MODEL_DIR"
DEFAULT_LOCAL_RUNTIME: LocalRuntime = "transformers"
COORDINATE_MODE_ENV = "GUI_AGENT_COORDINATE_MODE"
TRACE_ENV = "GUI_AGENT_TRACE"
DEFAULT_COORDINATE_MODE: CoordinateMode = "normalized_1000"
MODEL_IMAGE_MAX_DIM_ENV = "GUI_AGENT_MODEL_IMAGE_MAX_DIM"
# API 模型输入图像长边上限；超过时等比缩放以控制视觉 token 数量。
DEFAULT_MODEL_IMAGE_MAX_DIM = 1280
# 本地模型使用较小图像上限以降低视觉 token 数量和 CPU 推理开销。
LOCAL_MODEL_IMAGE_MAX_DIM_ENV = "GUI_AGENT_LOCAL_IMAGE_MAX_DIM"
DEFAULT_LOCAL_MODEL_IMAGE_MAX_DIM = 640
API_MODEL_ENV = "DASHSCOPE_API_MODEL"
API_ENABLE_THINKING_ENV = "GUI_AGENT_API_ENABLE_THINKING"
API_THINKING_BUDGET_ENV = "GUI_AGENT_API_THINKING_BUDGET"
API_THINKING_OPTIONS_SUPPORTED_ENV = "GUI_AGENT_API_THINKING_OPTIONS_SUPPORTED"
DEFAULT_API_ENABLE_THINKING = False
DEFAULT_API_THINKING_OPTIONS_SUPPORTED = True


@dataclass(frozen=True)
class AppConfig:
    """保存命令行入口与生产运行组件所需的配置。

    Attributes:
        model_mode: 模型调用模式。
        max_steps: 单任务基础逻辑步数上限；符合推进条件时可有界扩展。
        log_level: 标准库日志级别。
        log_dir: 运行日志输出目录。
        retry_count: 已解析动作执行失败后的单步重试上限。
        local_model_dir: 可选的本地模型目录。
        coordinate_mode: 模型点击坐标使用截图像素或 0..1000 相对坐标。
        local_runtime: 本地模型运行时。
        openvino_model_dir: 可选的 OpenVINO 模型目录。
        api_model: API 模型名称；None 表示未配置。
        api_enable_thinking: thinking 三态覆盖；None 表示不发送该字段。
        api_thinking_budget: 可选 thinking token budget。
        api_thinking_options_supported: 服务端是否支持 thinking 字段。

    CLI 使用 ``config_from_arguments`` 创建该对象，再交给生产运行组件组装。
    """

    model_mode: ModelMode = "local"
    max_steps: int = DEFAULT_MAX_STEPS
    log_level: str = DEFAULT_LOG_LEVEL
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    retry_count: int = DEFAULT_RETRY_COUNT
    local_model_dir: Path | None = None
    coordinate_mode: CoordinateMode = DEFAULT_COORDINATE_MODE
    local_runtime: LocalRuntime = DEFAULT_LOCAL_RUNTIME
    openvino_model_dir: Path | None = None
    api_model: str | None = None
    api_enable_thinking: bool | None = DEFAULT_API_ENABLE_THINKING
    api_thinking_budget: int | None = None
    api_thinking_options_supported: bool = DEFAULT_API_THINKING_OPTIONS_SUPPORTED

    def __post_init__(self) -> None:
        """验证配置，但不访问模型、网络或桌面。"""
        if self.model_mode not in {"local", "api"}:
            raise ValueError("model_mode 只能是 local 或 api。")
        self._validate_positive_integer(self.max_steps, "max_steps")
        if not isinstance(self.log_level, str):
            raise TypeError("log_level 必须是 str。")
        normalized_level = self.log_level.upper()
        if not isinstance(logging.getLevelName(normalized_level), int):
            raise ValueError("log_level 不是有效日志级别。")
        object.__setattr__(self, "log_level", normalized_level)
        if not isinstance(self.log_dir, Path):
            raise TypeError("log_dir 必须是 pathlib.Path。")
        self._validate_retry_count(self.retry_count)
        if self.local_model_dir is not None and not isinstance(
            self.local_model_dir,
            Path,
        ):
            raise TypeError("local_model_dir 必须是 pathlib.Path 或 None。")
        if self.coordinate_mode not in {"image_pixel", "normalized_1000"}:
            raise ValueError(
                "coordinate_mode 只能是 image_pixel 或 normalized_1000。",
            )
        if self.local_runtime not in {"transformers", "openvino"}:
            raise ValueError("local_runtime 只能是 transformers 或 openvino。")
        if self.openvino_model_dir is not None and not isinstance(
            self.openvino_model_dir,
            Path,
        ):
            raise TypeError("openvino_model_dir 必须是 pathlib.Path 或 None。")
        if self.api_model is not None:
            if not isinstance(self.api_model, str):
                raise TypeError("api_model 必须是 str 或 None。")
            if not self.api_model.strip():
                raise ValueError("api_model 不得为空。")
            object.__setattr__(self, "api_model", self.api_model.strip())
        if (
            self.api_enable_thinking is not None
            and type(
                self.api_enable_thinking,
            )
            is not bool
        ):
            raise TypeError("api_enable_thinking 必须是 bool 或 None。")
        if self.api_thinking_budget is not None:
            self._validate_positive_integer(
                self.api_thinking_budget,
                "api_thinking_budget",
            )
        if type(self.api_thinking_options_supported) is not bool:
            raise TypeError("api_thinking_options_supported 必须是 bool。")

    @staticmethod
    def _validate_positive_integer(value: object, name: str) -> None:
        """验证严格正整数配置，拒绝 bool 的隐式整数行为。"""
        if type(value) is not int:
            raise TypeError(f"{name} 必须是 int。")
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0。")

    @staticmethod
    def _validate_retry_count(value: object) -> None:
        """把单步重试限制在默认上限三次以内。"""
        if type(value) is not int:
            raise TypeError("retry_count 必须是 int。")
        if not 0 <= value <= DEFAULT_RETRY_COUNT:
            raise ValueError("retry_count 必须在 0 到 3 之间。")


@dataclass(frozen=True)
class GuiAgentSettings:
    """保存任务循环使用的运行设置。"""

    max_steps: int = DEFAULT_MAX_STEPS
    retry_count: int = DEFAULT_RETRY_COUNT
    model_mode: ModelMode = "local"
    coordinate_mode: CoordinateMode = DEFAULT_COORDINATE_MODE
    reject_initial_finish: bool = True
    verify_action_effect: bool = True
    decision_protocol_v2: bool = False
    decision_protocol_v3: bool = False
    hide_own_window_during_run: bool = False
    semantic_execution: bool = False

    def __post_init__(self) -> None:
        """在 Agent 创建带副作用的依赖前验证全部设置。"""
        AppConfig._validate_positive_integer(self.max_steps, "max_steps")
        AppConfig._validate_retry_count(self.retry_count)
        if self.model_mode not in {"local", "api"}:
            raise ValueError("model_mode 只能是 local 或 api。")
        if self.coordinate_mode not in {"image_pixel", "normalized_1000"}:
            raise ValueError("coordinate_mode 不是受支持的坐标模式。")
        if type(self.reject_initial_finish) is not bool:
            raise TypeError("reject_initial_finish 必须是 bool。")
        if type(self.decision_protocol_v2) is not bool:
            raise TypeError("decision_protocol_v2 必须是 bool。")
        if type(self.decision_protocol_v3) is not bool:
            raise TypeError("decision_protocol_v3 必须是 bool。")
        if type(self.verify_action_effect) is not bool:
            raise TypeError("verify_action_effect 必须是 bool。")
        if type(self.hide_own_window_during_run) is not bool:
            raise TypeError("hide_own_window_during_run 必须是 bool。")
        if type(self.semantic_execution) is not bool:
            raise TypeError("semantic_execution 必须是 bool。")


def local_model_dir_from_env() -> Path | None:
    """读取可选的本地模型目录，不记录或修改环境变量。"""
    value = os.environ.get(LOCAL_MODEL_DIR_ENV)
    if value is None or not value.strip():
        return None
    return Path(value)


def local_model_image_max_dim_from_env() -> int:
    """读取本地模型图像长边上限，未配置时使用默认值 640。"""
    value = os.environ.get(LOCAL_MODEL_IMAGE_MAX_DIM_ENV)
    if value is None or not value.strip():
        return DEFAULT_LOCAL_MODEL_IMAGE_MAX_DIM
    try:
        dim = int(value.strip())
    except ValueError:
        raise ValueError(
            "GUI_AGENT_LOCAL_MODEL_IMAGE_MAX_DIM 必须是正整数。",
        )
    if dim < 256:
        raise ValueError(
            "GUI_AGENT_LOCAL_MODEL_IMAGE_MAX_DIM 不得低于 256。",
        )
    return dim


def model_image_max_dim_from_env() -> int:
    """读取 API 模型输入图像长边上限，未配置时使用默认值。"""
    value = os.environ.get(MODEL_IMAGE_MAX_DIM_ENV)
    if value is None or not value.strip():
        return DEFAULT_MODEL_IMAGE_MAX_DIM
    try:
        dim = int(value.strip())
    except ValueError:
        raise ValueError(
            "GUI_AGENT_MODEL_IMAGE_MAX_DIM 必须是正整数。",
        )
    if dim < 256:
        raise ValueError(
            "GUI_AGENT_MODEL_IMAGE_MAX_DIM 不得低于 256。",
        )
    return dim


def api_model_from_env() -> str | None:
    """读取 API 模型名称，空值视为未配置。"""
    value = os.environ.get(API_MODEL_ENV)
    if value is None or not value.strip():
        return None
    return value.strip()


def api_enable_thinking_from_env() -> bool | None:
    """读取 thinking 三态覆盖，未设置时保持默认关闭。"""
    value = os.environ.get(API_ENABLE_THINKING_ENV)
    if value is None:
        return DEFAULT_API_ENABLE_THINKING
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"none", "default"}:
        return None
    raise ValueError(
        "GUI_AGENT_API_ENABLE_THINKING 必须是 true、false 或 none。",
    )


def api_thinking_budget_from_env() -> int | None:
    """读取可选 thinking budget，未设置或空值时不发送覆盖值。"""
    value = os.environ.get(API_THINKING_BUDGET_ENV)
    if value is None or not value.strip():
        return None
    try:
        budget = int(value.strip())
    except ValueError:
        raise ValueError("GUI_AGENT_API_THINKING_BUDGET 必须是正整数。")
    AppConfig._validate_positive_integer(budget, "api_thinking_budget")
    return budget


def api_thinking_options_supported_from_env() -> bool:
    """读取服务端 thinking 能力开关，默认使用当前 DashScope 能力。"""
    value = os.environ.get(API_THINKING_OPTIONS_SUPPORTED_ENV)
    if value is None:
        return DEFAULT_API_THINKING_OPTIONS_SUPPORTED
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "GUI_AGENT_API_THINKING_OPTIONS_SUPPORTED 必须是 bool。",
    )


def local_runtime_from_env() -> LocalRuntime:
    """读取本地推理运行时，未配置时使用 Transformers。"""
    value = os.environ.get(LOCAL_RUNTIME_ENV, DEFAULT_LOCAL_RUNTIME)
    normalized = value.strip().lower()
    if normalized not in {"transformers", "openvino"}:
        raise ValueError(
            "GUI_AGENT_LOCAL_RUNTIME 只能是 transformers 或 openvino。",
        )
    return cast(LocalRuntime, normalized)


def openvino_model_dir_from_env() -> Path | None:
    """读取可选的 OpenVINO 导出模型目录，空值视为未配置。"""
    value = os.environ.get(OPENVINO_MODEL_DIR_ENV)
    if value is None or not value.strip():
        return None
    return Path(value)


def decision_protocol_v2_from_env() -> bool:
    """读取决策协议 V2 开关，默认关闭。"""
    value = os.environ.get(DECISION_PROTOCOL_V2_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def decision_protocol_v3_from_env() -> bool:
    """读取决策协议 V3 开关；未设置时默认启用。

    显式设置为 0、false、no 或 off 时关闭。V3 与 V2 同时开启时由调用方
    按当前协议优先级处理。
    """
    value = os.environ.get(DECISION_PROTOCOL_V3_ENV)
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def semantic_execution_from_env() -> bool | None:
    """读取程序化完成验证开关，未设置时返回 None 交由调用方决定。"""
    value = os.environ.get(SEMANTIC_EXECUTION_ENV)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def hide_own_window_during_run_from_env() -> bool:
    """读取任务运行期间最小化自身控制窗口的开关，默认关闭。"""
    value = os.environ.get(HIDE_OWN_WINDOW_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def trace_enabled_from_env() -> bool:
    """读取 Agent 决策追踪开关，默认关闭。"""
    value = os.environ.get(TRACE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def coordinate_mode_from_env() -> CoordinateMode:
    """读取模型坐标模式，未配置时使用 0..1000 相对坐标。"""
    value = os.environ.get(COORDINATE_MODE_ENV, DEFAULT_COORDINATE_MODE)
    normalized = value.strip().lower()
    if normalized not in {"image_pixel", "normalized_1000"}:
        raise ValueError(
            "GUI_AGENT_COORDINATE_MODE 只能是 image_pixel 或 normalized_1000。",
        )
    return cast(CoordinateMode, normalized)
