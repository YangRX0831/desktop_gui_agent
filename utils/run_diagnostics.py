"""被动运行诊断(OBSERVABILITY_ONLY):阶段计时 + 纯观察 watchdog。

仅当环境变量 ``GUI_AGENT_RUN_DIAGNOSTICS=1`` 时启用;关闭时全部函数
为零开销直通,不改变任何 Agent 决策与流程。事件写入 gui_agent 业务
日志(``diag_<phase>_begin/end``),字段仅允许非敏感事实(步号、时长、
模型名、动作类型、文本长度与摘要、异常类型名);密钥/全文/头信息一律
不出现在输出。watchdog 只观察并记录(30s 警告、60s/120s 各一次线程
栈快照),绝不终止进程、不重试、不干预执行。
"""

import hashlib
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger("gui_agent")

_DIAG_ENV = "GUI_AGENT_RUN_DIAGNOSTICS"
# 敏感字段名(小写):按块名单直接丢弃,值绝不进入日志。
_SENSITIVE_FIELDS = frozenset(
    {
        "api_key",
        "authorization",
        "token",
        "secret",
        "password",
        "headers",
        "prompt",
        "text",
        "image",
    },
)
_WARNING_SECONDS = 30.0
_STACK_SECONDS = 60.0
_STACK_AGAIN_SECONDS = 120.0
_POLL_SECONDS = 5.0


def diagnostics_enabled() -> bool:
    """被动诊断是否启用(默认关闭)。"""
    return os.environ.get(_DIAG_ENV, "") == "1"


def text_digest(text: str) -> str:
    """文本短摘要(sha1 前 8 位);诊断日志中替代全文。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _safe_fields(fields: dict) -> dict:
    """过滤敏感键并截断超长值;敏感键直接丢弃而非脱敏保留。"""
    safe: dict[str, object] = {}
    for key, value in fields.items():
        if str(key).lower() in _SENSITIVE_FIELDS:
            continue
        text_value = str(value)
        safe[key] = text_value[:80]
    return safe


def diag_log(event: str, **fields: object) -> None:
    """记录一条诊断事件;未启用时为空操作。"""
    if not diagnostics_enabled():
        return
    rendered = "".join(
        f"，{key}={value}" for key, value in _safe_fields(fields).items()
    )
    logger.info("%s%s", event, rendered)


@contextmanager
def diag_phase(
    name: str,
    step: int | None = None,
    **fields: object,
) -> Generator[None, None, None]:
    """阶段计时上下文:begin/end + duration_ms;异常路径同样记 end。

    同时向 watchdog 注册/注销当前阶段,使长阶段可被观察定位。
    """
    if not diagnostics_enabled():
        yield
        return
    begin = time.monotonic()
    diag_log(f"{name}_begin", step=step, **fields)
    watchdog_register(name, begin, step)
    try:
        yield
    except BaseException as exception:
        diag_log(
            f"{name}_end",
            step=step,
            duration_ms=round((time.monotonic() - begin) * 1000, 1),
            exception_type=type(exception).__name__,
        )
        watchdog_clear()
        raise
    diag_log(
        f"{name}_end",
        step=step,
        duration_ms=round((time.monotonic() - begin) * 1000, 1),
    )
    watchdog_clear()


def thread_stack_snapshot() -> str:
    """当前进程全部线程的调用栈快照(每线程最多保留顶部 3 帧)。"""
    names = {t.ident: t.name for t in threading.enumerate()}
    chunks: list[str] = []
    for tid, frame in sys._current_frames().items():
        entries = traceback.extract_stack(frame)
        top = " ".join(
            f"{e.filename.rsplit('/', 1)[-1]}:{e.lineno}:{e.name}" for e in entries[-3:]
        )
        chunks.append(f"thread={names.get(tid, tid)} frames_top={top}")
    return " || ".join(chunks) if chunks else "no_frames"


class _StallWatchdog:
    """纯观察 watchdog:跟踪当前阶段,超阈值仅写日志。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase: str | None = None
        self._step: int | None = None
        self._begin: float | None = None
        self._warned = False
        self._stacked_at: list[float] = []
        self._thread: threading.Thread | None = None

    def register(self, name: str, begin: float, step: int | None) -> None:
        """登记当前阶段并确保看门狗线程在运行。"""
        with self._lock:
            self._phase = name
            self._step = step
            self._begin = begin
            self._warned = False
            self._stacked_at = []
            self._ensure_thread_locked()

    def clear(self) -> None:
        """清空当前阶段登记;不影响看门狗线程。"""
        with self._lock:
            self._phase = None
            self._begin = None
            self._step = None

    def _ensure_thread_locked(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._loop,
                name="diag_stall_watchdog",
                daemon=True,
            )
            self._thread.start()

    def _loop(self) -> None:
        while True:
            time.sleep(_POLL_SECONDS)
            self.check()

    def check(self, now: float | None = None) -> None:
        """检查当前阶段耗时并按阈值记录;now 可注入便于测试。"""
        with self._lock:
            if self._phase is None or self._begin is None:
                return
            elapsed = (now if now is not None else time.monotonic()) - self._begin
            if elapsed >= _WARNING_SECONDS and not self._warned:
                self._warned = True
                diag_log(
                    "DIAG_STALL_WARNING",
                    phase=self._phase,
                    step=self._step,
                    elapsed_s=round(elapsed, 1),
                    thread=threading.current_thread().name,
                )
            for threshold in (_STACK_SECONDS, _STACK_AGAIN_SECONDS):
                if elapsed >= threshold and threshold not in self._stacked_at:
                    self._stacked_at.append(threshold)
                    diag_log(
                        "DIAG_STALL_STACK",
                        phase=self._phase,
                        step=self._step,
                        elapsed_s=round(elapsed, 1),
                        snapshot=thread_stack_snapshot(),
                    )


_WATCHDOG = _StallWatchdog()


def watchdog_register(name: str, begin: float, step: int | None) -> None:
    """登记当前阶段起点(仅诊断启用时有意义)。"""
    if diagnostics_enabled():
        _WATCHDOG.register(name, begin, step)


def watchdog_clear() -> None:
    """清除当前阶段登记(任务结束/切换时调用,防止误报)。"""
    _WATCHDOG.clear()
