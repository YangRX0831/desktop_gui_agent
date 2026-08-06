"""提供单任务生命周期、步骤和重试统计管理。"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """任务生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class TaskStep:
    """单个动作的执行记录。"""

    action: dict[str, object]
    result: bool
    recorded_at: datetime
    retry_count: int


@dataclass
class TaskState:
    """任务当前状态的独立数据快照。"""

    task_description: str
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    ended_at: datetime | None = None
    steps: list[TaskStep] = field(default_factory=list)
    step_count: int = 0
    retry_count: int = 0
    failure_reason: str | None = None


class TaskManager:
    """按固定状态机管理一个任务，不保证线程安全。"""

    def __init__(
        self,
        task_description: str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """初始化待执行任务。

        Args:
            task_description: 原样保留的非空任务描述。
            clock: 可注入的当前时间提供者。

        Raises:
            TypeError: 参数类型不符合合同。
            ValueError: 任务描述只包含空白。
        """
        if not isinstance(task_description, str):
            raise TypeError("task_description 必须是 str。")
        if not task_description.strip():
            raise ValueError("task_description 不得为空。")
        if clock is not None and not callable(clock):
            raise TypeError("clock 必须可调用。")

        self._clock = clock if clock is not None else datetime.now
        self._state = TaskState(task_description=task_description)

    def _now(self) -> datetime:
        """读取一次时钟并验证返回类型。"""
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock 必须返回 datetime。")
        return value

    @property
    def state(self) -> TaskState:
        """返回与内部状态完全隔离的深复制快照。"""
        return deepcopy(self._state)

    def start(self) -> None:
        """把待执行任务转换为运行中状态。

        Raises:
            ValueError: 当前状态不是 pending。
            TypeError: 注入时钟返回类型错误。
        """
        if self._state.status is not TaskStatus.PENDING:
            raise ValueError("只有 pending 任务可以开始。")
        started_at = self._now()
        self._state.started_at = started_at
        self._state.status = TaskStatus.RUNNING

    def record_step(
        self,
        action: Mapping[str, object],
        result: bool,
        retry_count: int = 0,
    ) -> None:
        """记录运行中任务的一次动作执行。

        Args:
            action: 非空动作映射。
            result: 动作是否成功。
            retry_count: 本步骤已经发生的重试次数。

        Raises:
            TypeError: 参数类型或时钟返回类型错误。
            ValueError: 状态、动作内容或重试次数不合法。
        """
        if self._state.status is not TaskStatus.RUNNING:
            raise ValueError("只有 running 任务可以记录步骤。")
        if not isinstance(action, Mapping):
            raise TypeError("action 必须是 Mapping。")
        if not action:
            raise ValueError("action 不得为空。")
        if type(result) is not bool:
            raise TypeError("result 必须是 bool。")
        if type(retry_count) is not int:
            raise TypeError("retry_count 必须是 int。")
        if retry_count < 0:
            raise ValueError("retry_count 不得小于 0。")

        action_copy = deepcopy(dict(action))
        recorded_at = self._now()
        step = TaskStep(
            action=action_copy,
            result=result,
            recorded_at=recorded_at,
            retry_count=retry_count,
        )
        self._state.steps.append(step)
        self._state.step_count += 1
        self._state.retry_count += retry_count

    def succeed(self) -> None:
        """把运行中任务转换为成功终态。

        Raises:
            ValueError: 当前状态不是 running。
            TypeError: 注入时钟返回类型错误。
        """
        if self._state.status is not TaskStatus.RUNNING:
            raise ValueError("只有 running 任务可以成功结束。")
        ended_at = self._now()
        self._state.ended_at = ended_at
        self._state.status = TaskStatus.SUCCESS

    def fail(self, reason: str) -> None:
        """把运行中任务转换为失败终态。

        Args:
            reason: 原样保留的非空失败原因。

        Raises:
            TypeError: 参数类型或时钟返回类型错误。
            ValueError: 当前状态或失败原因不合法。
        """
        if self._state.status is not TaskStatus.RUNNING:
            raise ValueError("只有 running 任务可以失败结束。")
        if not isinstance(reason, str):
            raise TypeError("reason 必须是 str。")
        if not reason.strip():
            raise ValueError("reason 不得为空。")
        ended_at = self._now()
        self._state.ended_at = ended_at
        self._state.failure_reason = reason
        self._state.status = TaskStatus.FAILED
