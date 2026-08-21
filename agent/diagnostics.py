"""模型失败事件的本地诊断文件写入器。

职责：
    仅在模型调用失败或动作解析失败时，把 prompt、模型响应原文和当步截图
    写入 ``logs/diagnosis/`` 下的 JSONL 与 PNG 文件，供问题定位使用。成功
    响应不记录，常规业务日志仍保持脱敏输出。

隐私边界：
    诊断文件可能包含 prompt、模型响应和截图，因此仅应保存在本地受控目录。
    本写入器不对这些内容自动执行凭据过滤；调用方不得传入密码、Token、
    API Key 等敏感信息，手动分享诊断文件前也应再次检查并脱敏。

失败安全：
    写入失败只记录异常类型，不向调用方抛出，也不影响主任务流程。
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

logger = logging.getLogger(__name__)

_DIAGNOSIS_SUBDIR = "diagnosis"


class DiagnosticsWriterProtocol(Protocol):
    """定义编排器需要的最小诊断写入接口。

    ``ActionDiagnosticsWriter`` 是默认实现；测试可注入内存实现，None 表示
    不启用诊断。
    """

    def record(self, record: "DiagnosticsRecord") -> None:
        """写入单个失败事件。"""


@dataclass(frozen=True)
class DiagnosticsRecord:
    """保存单个失败事件的诊断数据。

    Attributes:
        run_id: 任务运行标识，用于把同一运行的事件归入同一 JSONL。
        step_number: 失败发生的逻辑步骤号（从 1 开始）。
        attempt: 步内尝试序号（从 0 开始，0 为首次尝试）。
        image: 失败时使用的模型输入截图。
        prompt: 失败时发送的完整 prompt 文本。
        response: 模型响应原文；模型调用抛异常时为 None。
        failure_reason: 失败原因摘要。
        exception_type: 模型调用抛出的异常类型名；无异常时为 None。
    """

    run_id: str
    step_number: int
    attempt: int
    image: Image.Image
    prompt: str
    response: str | None
    failure_reason: str
    exception_type: str | None = None


class ActionDiagnosticsWriter:
    """把失败事件追加写入本地诊断目录。"""

    def __init__(self, log_dir: Path) -> None:
        """初始化写入器。

        Args:
            log_dir: 业务日志目录；诊断文件位于其 ``diagnosis`` 子目录。

        Raises:
            TypeError: log_dir 不是 pathlib.Path。
        """
        if not isinstance(log_dir, Path):
            raise TypeError("log_dir 必须是 pathlib.Path。")
        self._dir = log_dir / _DIAGNOSIS_SUBDIR

    def record(self, record: DiagnosticsRecord) -> None:
        """把单个失败事件写为 JSONL 记录和 PNG 截图。

        目录按需创建，写入失败只记录异常类型；参数错误在写入前抛出。

        Args:
            record: 已由调用方组装的失败事件数据。

        Raises:
            TypeError: 字段类型不符合约定。
            ValueError: run_id、failure_reason 为空或步骤号无效。
        """
        self._validate_record(record)
        screenshot_name = (
            f"{record.run_id}_step{record.step_number}_att{record.attempt}.png"
        )
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": record.run_id,
            "step_number": record.step_number,
            "attempt": record.attempt,
            "failure_reason": record.failure_reason,
            "exception_type": record.exception_type,
            "prompt": record.prompt,
            "response": record.response,
            "screenshot": screenshot_name,
        }
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            record.image.save(self._dir / screenshot_name, "PNG")
            with open(
                self._dir / f"{record.run_id}.jsonl",
                "a",
                encoding="utf-8",
            ) as jsonl:
                jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exception:
            logger.warning(
                "diagnostics_write_failed：exception_type=%s",
                type(exception).__name__,
            )

    @staticmethod
    def _validate_record(record: DiagnosticsRecord) -> None:
        """在产生任何写入副作用前校验诊断字段。"""
        if not isinstance(record.run_id, str) or not record.run_id:
            raise ValueError("run_id 必须是非空字符串。")
        if type(record.step_number) is not int or record.step_number < 1:
            raise ValueError("step_number 必须是正整数。")
        if type(record.attempt) is not int or record.attempt < 0:
            raise ValueError("attempt 必须是非负整数。")
        if not isinstance(record.image, Image.Image):
            raise TypeError("image 必须是 PIL.Image.Image。")
        if not isinstance(record.prompt, str):
            raise TypeError("prompt 必须是 str。")
        if record.response is not None and not isinstance(record.response, str):
            raise TypeError("response 必须是 str 或 None。")
        if not isinstance(record.failure_reason, str) or not record.failure_reason:
            raise ValueError("failure_reason 必须是非空字符串。")
        if record.exception_type is not None and not isinstance(
            record.exception_type,
            str,
        ):
            raise TypeError("exception_type 必须是 str 或 None。")
