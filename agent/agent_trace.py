"""决策协议共用的 debug/benchmark 追踪写入器。

默认关闭(GUI_AGENT_TRACE 开启);记录每次模型调用的完整决策证据与
每步 before/after 截图,供失败分类(perception/grounding/policy/
parsing/execution/completion)与 benchmark 指标聚合使用。凭据类信息
硬禁;不估算 token(usage 缺失时记 unknown)。
"""

import json
import logging
import time
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

_UNKNOWN = "unknown"


class AgentTraceWriterProtocol(Protocol):
    """编排器需要的最小 trace 合同;测试可注入内存 fake。"""

    def record_model_call(self, fields: dict[str, object]) -> None:
        """追加一条模型调用/决策记录。"""

    def save_step_screenshot(
        self,
        run_id: str,
        step_number: int,
        phase: str,
        image: object,
    ) -> None:
        """保存某步的 before/after/observe 截图。"""


def default_trace_dir(log_dir: Path) -> Path:
    """返回 trace 根目录 logs/agent_trace/。"""
    return log_dir / "agent_trace"


class AgentTraceWriter:
    """把决策追踪写入 logs/agent_trace/<run_id>.jsonl 与截图目录。"""

    def __init__(self, log_dir: Path) -> None:
        if not isinstance(log_dir, Path):
            raise TypeError("log_dir 必须是 pathlib.Path。")
        self._root = default_trace_dir(log_dir)

    def record_model_call(self, fields: dict[str, object]) -> None:
        """追加一条 JSONL 记录;写入失败只记安全日志不抛出。"""
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
        """按 step_XX_before/after/observe 命名保存截图;失败不抛出。"""
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
