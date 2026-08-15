"""提供桌面 GUI 智能体的最小运行配置。

职责：
    ``AppConfig`` 是 CLI 与 ``GuiAgent`` 之间唯一的运行配置快照；默认
    ``max_steps`` 和 ``retry_count`` 也只在本模块定义，避免入口和编排器
    分别复制数值后发生漂移。

校验约束：
    所有类型和值域在构造时验证，后端、截图和控制器尚未创建，因此非法
    配置不会产生外部副作用。bool 不作为 int 接受。

环境边界：
    本地模型路径只能由专用环境变量读取，空值视为未配置。读取函数不创建
    目录、不加载模型，也不修改进程或系统环境。

该模块不保存 API key。DashScope 凭据由专用后端在其环境边界读取，避免
把秘密信息混入可打印的通用配置对象。
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

ModelMode = Literal["local", "api"]
CoordinateMode = Literal["image_pixel", "normalized_1000"]
LocalRuntime = Literal["transformers", "openvino"]
# PRD 4.4.1 默认步数上限 10;可通过 CLI --max-steps 调整。
DEFAULT_MAX_STEPS = 10
DEFAULT_RETRY_COUNT = 3
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = Path("logs")
LOCAL_MODEL_DIR_ENV = "GUI_AGENT_LOCAL_MODEL_DIR"
# 本地推理双路线:transformers 为 canonical 默认,openvino 为纯 CPU
# 环境的加速路线,经环境变量显式选择。
LOCAL_RUNTIME_ENV = "GUI_AGENT_LOCAL_RUNTIME"
OPENVINO_MODEL_DIR_ENV = "GUI_AGENT_OPENVINO_MODEL_DIR"
DEFAULT_LOCAL_RUNTIME: LocalRuntime = "transformers"
COORDINATE_MODE_ENV = "GUI_AGENT_COORDINATE_MODE"
DEFAULT_COORDINATE_MODE: CoordinateMode = "normalized_1000"
MODEL_IMAGE_MAX_DIM_ENV = "GUI_AGENT_MODEL_IMAGE_MAX_DIM"
# 模型输入图像长边上限(像素):超过时等比缩放。1280 兼顾 UI 细节与
# visual token 数量;可通过环境变量调整,适配不同模型能力。
DEFAULT_MODEL_IMAGE_MAX_DIM = 1280


@dataclass(frozen=True)
class AppConfig:
    """保存 CLI 与 production dependency wiring 所需的最小配置。

    Attributes:
        model_mode: 模型调用模式。
        max_steps: 单任务最大执行轮次。
        log_level: 标准库日志级别。
        log_dir: 周期日志输出目录。
        retry_count: 已解析动作执行失败后的单步重试上限。
        local_model_dir: 可选的本地模型目录。
        coordinate_mode: 模型 click 坐标使用截图像素或 0..1000 相对坐标。

    CLI 使用 ``config_from_arguments`` 创建该对象，再交给 production wiring。
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

    @staticmethod
    def _validate_positive_integer(value: object, name: str) -> None:
        """验证严格正整数配置，拒绝 bool 的隐式整数行为。"""
        if type(value) is not int:
            raise TypeError(f"{name} 必须是 int。")
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0。")

    @staticmethod
    def _validate_retry_count(value: object) -> None:
        """把单步重试限制在 PRD 默认上限三次以内。"""
        if type(value) is not int:
            raise TypeError("retry_count 必须是 int。")
        if not 0 <= value <= DEFAULT_RETRY_COUNT:
            raise ValueError("retry_count 必须在 0 到 3 之间。")


@dataclass(frozen=True)
class GuiAgentSettings:
    """保存 PRD 4.4.1 任务循环的运行设置。"""

    max_steps: int = DEFAULT_MAX_STEPS
    retry_count: int = DEFAULT_RETRY_COUNT
    model_mode: ModelMode = "local"
    coordinate_mode: CoordinateMode = DEFAULT_COORDINATE_MODE
    reject_initial_finish: bool = True
    verify_action_effect: bool = True

    def __post_init__(self) -> None:
        """在 Agent 创建副作用依赖前验证全部设置。"""
        AppConfig._validate_positive_integer(self.max_steps, "max_steps")
        AppConfig._validate_retry_count(self.retry_count)
        if self.model_mode not in {"local", "api"}:
            raise ValueError("model_mode 只能是 local 或 api。")
        if self.coordinate_mode not in {"image_pixel", "normalized_1000"}:
            raise ValueError("coordinate_mode 不是受支持的坐标模式。")
        if type(self.reject_initial_finish) is not bool:
            raise TypeError("reject_initial_finish 必须是 bool。")
        if type(self.verify_action_effect) is not bool:
            raise TypeError("verify_action_effect 必须是 bool。")


def local_model_dir_from_env() -> Path | None:
    """读取可选的本地模型目录，不记录或修改环境变量。"""
    value = os.environ.get(LOCAL_MODEL_DIR_ENV)
    if value is None or not value.strip():
        return None
    return Path(value)


def model_image_max_dim_from_env() -> int:
    """读取模型输入图像长边上限;未配置时使用默认值。"""
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


def local_runtime_from_env() -> LocalRuntime:
    """读取本地推理运行时;未配置时保持 transformers canonical 默认。"""
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


def coordinate_mode_from_env() -> CoordinateMode:
    """读取模型坐标模式；未配置时使用 Qwen-VL 相对坐标基线。"""
    value = os.environ.get(COORDINATE_MODE_ENV, DEFAULT_COORDINATE_MODE)
    normalized = value.strip().lower()
    if normalized not in {"image_pixel", "normalized_1000"}:
        raise ValueError(
            "GUI_AGENT_COORDINATE_MODE 只能是 image_pixel 或 normalized_1000。",
        )
    return cast(CoordinateMode, normalized)
