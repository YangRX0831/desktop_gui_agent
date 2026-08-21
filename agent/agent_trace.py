"""记录模型决策与步骤截图，供本地调试和 Benchmark 分析使用。

该功能默认关闭，可通过 ``GUI_AGENT_TRACE`` 启用。追踪文件可能包含
模型调用和界面状态信息，因此只应保存在本地受控目录中。写入器不会对
调用方传入的任意字段执行内容脱敏，调用方不得传入 API Key、密码、Token
等凭据或其他不应落盘的敏感信息。
"""

import json
import logging
import time
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

_UNKNOWN = "unknown"


class AgentTraceWriterProtocol(Protocol):
    """定义编排器使用的最小追踪写入接口。"""

    def record_model_call(self, fields: dict[str, object]) -> None:
        """追加一条模型调用或决策记录。"""

    def save_step_screenshot(
        self,
        run_id: str,
        step_number: int,
        phase: str,
        image: object,
    ) -> None:
        """保存某一步骤的 before、after 或 observe 截图。"""


def default_trace_dir(log_dir: Path) -> Path:
    """返回追踪文件根目录 ``logs/agent_trace/``。"""
    return log_dir / "agent_trace"


class AgentTraceWriter:
    """把决策追踪写入 JSONL 文件及对应的步骤截图目录。"""

    def __init__(self, log_dir: Path) -> None:
        if not isinstance(log_dir, Path):
            raise TypeError("log_dir 必须是 pathlib.Path。")
        self._root = default_trace_dir(log_dir)

    def record_model_call(self, fields: dict[str, object]) -> None:
        """追加一条 JSONL 记录；写入失败只记录异常类型，不影响主流程。"""
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            run_id = str(fields.get("run_id", "unknown"))
            record = {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **fields,
            }
            with open(
                self._root / f"{run_id}.jsonl",
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exception:
            logger.warning(
                "agent_trace_write_failed：exception_type=%s",
                type(exception).__name__,
            )

    def save_step_screenshot(
        self,
        run_id: str,
        step_number: int,
        phase: str,
        image: object,
    ) -> None:
        """按步骤编号和阶段保存截图；写入失败不影响主流程。"""
        try:
            directory = self._root / run_id
            directory.mkdir(parents=True, exist_ok=True)
            name = f"step_{step_number:02d}_{phase}.png"
            save = getattr(image, "save", None)
            if callable(save):
                save(directory / name, "PNG")
        except Exception as exception:
            logger.warning(
                "agent_trace_screenshot_failed：exception_type=%s",
                type(exception).__name__,
            )
