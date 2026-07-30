"""提供鼠标测试共享 fake、时钟和依赖隔离辅助。

本模块只服务测试，不创建真实 Controller 或访问桌面。
"""

import logging
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from control import mouse_controller

SENSITIVE_LOG_PARTS = (
    "SENSITIVE_EXCEPTION_MESSAGE",
    "private",
    "model",
    "region=(10,20,30,40)",
    "user_text_marker",
)


@contextmanager
def formatted_log_output(logger_name: str) -> Iterator[StringIO]:
    """捕获最终 Formatter 输出，并在退出时恢复 logger 状态。"""
    target_logger = logging.getLogger(logger_name)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s|%(name)s|%(message)s"))
    previous_level = target_logger.level
    previous_propagate = target_logger.propagate
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.ERROR)
    target_logger.propagate = False
    try:
        yield stream
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)
        target_logger.propagate = previous_propagate


class FakeBackend:
    """记录鼠标调用并支持在指定调用处模拟失败。"""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.failures: dict[tuple[str, int], Exception] = {}

    def fail_on(self, operation: str, call_number: int, exc: Exception) -> None:
        self.failures[(operation, call_number)] = exc

    def move_to(self, x: int, y: int) -> None:
        self._record("move_to", x, y)

    def click(self, button: str, count: int) -> None:
        self._record("click", button, count)

    def press(self, button: str) -> None:
        self._record("press", button)

    def release(self, button: str) -> None:
        self._record("release", button)

    def _record(self, operation: str, *values: object) -> None:
        self.counts[operation] += 1
        self.events.append((operation, *values))
        failure = self.failures.get((operation, self.counts[operation]))
        if failure is not None:
            raise failure


@dataclass
class SafeEnvironment:
    """保存完全模拟的鼠标测试依赖及调用记录。"""

    backend: FakeBackend
    factory_calls: list[str]
    bounds_calls: list[str]
    sleeps: list[float]
    bounds: tuple[int, int, int, int]


@pytest.fixture
def safe_environment(monkeypatch: pytest.MonkeyPatch) -> SafeEnvironment:
    """为每个测试提供全新的后端、边界和等待记录。"""
    backend = FakeBackend()
    factory_calls: list[str] = []
    bounds_calls: list[str] = []
    sleeps: list[float] = []
    bounds = (-100, -50, 800, 600)

    def create_backend() -> FakeBackend:
        factory_calls.append("create")
        return backend

    def get_bounds() -> tuple[int, int, int, int]:
        bounds_calls.append("bounds")
        return bounds

    monkeypatch.setattr(mouse_controller, "_create_backend", create_backend)
    monkeypatch.setattr(
        mouse_controller,
        "_get_virtual_screen_bounds",
        get_bounds,
    )
    monkeypatch.setattr(mouse_controller, "_sleep", sleeps.append)
    return SafeEnvironment(
        backend=backend,
        factory_calls=factory_calls,
        bounds_calls=bounds_calls,
        sleeps=sleeps,
        bounds=bounds,
    )


def _module_source() -> str:
    return Path(mouse_controller.__file__).read_text(encoding="utf-8")


def _controller(safe_environment: SafeEnvironment) -> mouse_controller.MouseController:
    return mouse_controller.MouseController()
