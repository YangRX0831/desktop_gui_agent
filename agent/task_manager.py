"""提供单任务生命周期、步骤和重试统计管理。

职责：
    为一个用户任务保存 pending、running、success、failed 四态生命周期，
    同时记录已执行动作、布尔结果、单步重试次数和任务累计重试次数。

状态约束：
    只有 running 任务可以写入步骤、重试或终态；终态不可再次转换。时间由
    可注入时钟提供，便于确定性测试，但每次读取仍验证返回类型。

隔离约束：
    动作和公开 ``state`` 都经过深复制。调用方无法通过修改原始动作或状态
    快照反向改写任务历史，避免审计证据随外部可变对象发生漂移。

本模块只管理内存状态，不执行控制、不写文件、不调用模型，也不决定任务
是否语义完成；finish 的解释属于 ``GuiAgent`` 编排职责。
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """枚举单任务允许的四种生命周期状态。

    Attributes:
        PENDING: 已创建但尚未开始。
        RUNNING: 可以记录步骤、重试或终态。
        SUCCESS: 收到 finish 后的成功终态。
        FAILED: 无法继续或步骤耗尽后的失败终态。

    调用方通过 ``TaskManager.state`` 的深复制快照读取状态。
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class TaskAttempt:
    """保存一个 logical step 内单次 attempt 的动作与结果。

    Attributes:
        action: 与外部可变对象隔离的动作深复制;model/parse 失败无动作时为 None。
        succeeded: 本次 attempt 是否成功。
        retry_index: 0 = initial attempt, >=1 = fresh retry。
        stage: attempt 发生的阶段:"model"、"parse"、"dispatch" 或 "finish"。
        failure_reason: 失败原因摘要;succeeded 时为 None。
        recorded_at: 记录时的时钟值。
    """

    action: dict[str, object] | None
    succeeded: bool
    retry_index: int
    stage: str
    failure_reason: str | None
    recorded_at: datetime


@dataclass
class TaskStep:
    """保存一个 logical step 的全部 attempt 历史和最终结果。

    Attributes:
        attempts: 该 step 内所有 attempt 的有序记录(initial + retries)。
        result: 该 logical step 的最终布尔结果。
        retry_count: 该 step 实际发生的 fresh retry 次数。
        recorded_at: 完成本步骤统计时的时钟值。

    PRD 4.3.4 要求"记录每一步执行的动作与结果";由于 fresh retry 产生新
    动作,每个 attempt 都需要保存。``attempts`` 保留完整证据,``action``
    属性提供对最后一个 attempt 动作的便捷只读访问。
    """

    attempts: list[TaskAttempt]
    result: bool
    retry_count: int
    recorded_at: datetime

    @property
    def action(self) -> dict[str, object]:
        """返回最后一个含动作的 attempt 的动作(向后兼容只读访问)。"""
        for attempt in reversed(self.attempts):
            if attempt.action is not None:
                return attempt.action
        return {}


@dataclass
class TaskState:
    """保存任务生命周期、步骤和失败信息。

    Attributes:
        task_description: 原始任务文本，仅保存在内存状态。
        status: 当前 ``TaskStatus``。
        started_at: 开始时间，pending 时为 None。
        ended_at: 成功或失败结束时间。
        steps: 已执行动作记录。
        step_count: ``steps`` 的显式统计值。
        retry_count: 所有动作的累计重试次数。
        failure_reason: 固定安全失败原因。

    该对象只通过深复制公开，适合只读展示和测试断言。
    """

    task_description: str
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    ended_at: datetime | None = None
    steps: list[TaskStep] = field(default_factory=list)
    step_count: int = 0
    retry_count: int = 0
    failure_reason: str | None = None


class TaskManager:
    """按固定状态机管理一个任务，不保证线程安全。

    Attributes:
        clock: 返回 datetime 的可注入时钟。
        state: 私有可变状态，只通过深复制属性公开。

    典型用法是 start、逐动作 record_step，最后 succeed 或 fail。非法状态
    迁移立即抛出 ValueError，不自动修正历史。
    """

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
        # 当前 logical step 的 attempt 缓冲区;finalize_step 时转为 TaskStep。
        self._current_step_attempts: list[TaskAttempt] = []

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

    def record_attempt(
        self,
        action: Mapping[str, object] | None,
        succeeded: bool,
        retry_index: int = 0,
        stage: str = "dispatch",
        failure_reason: str | None = None,
    ) -> None:
        """记录当前 logical step 的一次 attempt 的动作与结果。

        Args:
            action: 动作映射;model/parse 失败无动作时为 None(不伪造动作)。
            succeeded: 本次 attempt 是否成功。
            retry_index: 0 = initial attempt, >=1 = fresh retry 序号。
            stage: attempt 阶段:"model"、"parse"、"dispatch" 或 "finish"。
            failure_reason: 失败原因摘要;succeeded 时应为 None。

        Raises:
            TypeError: 参数类型错误。
            ValueError: 状态非 running 或 action 为空映射(非 None)。
        """
        if self._state.status is not TaskStatus.RUNNING:
            raise ValueError("只有 running 任务可以记录 attempt。")
        if action is not None:
            if not isinstance(action, Mapping):
                raise TypeError("action 必须是 Mapping 或 None。")
            if not action:
                raise ValueError("action 不得为空映射(使用 None 表示无动作)。")
        if type(succeeded) is not bool:
            raise TypeError("succeeded 必须是 bool。")
        if type(retry_index) is not int:
            raise TypeError("retry_index 必须是 int。")
        if retry_index < 0:
            raise ValueError("retry_index 不得小于 0。")
        if not isinstance(stage, str):
            raise TypeError("stage 必须是 str。")
        if failure_reason is not None and not isinstance(failure_reason, str):
            raise TypeError("failure_reason 必须是 str 或 None。")
        action_copy = deepcopy(dict(action)) if action is not None else None
        attempt = TaskAttempt(
            action=action_copy,
            succeeded=succeeded,
            retry_index=retry_index,
            stage=stage,
            failure_reason=failure_reason,
            recorded_at=self._now(),
        )
        self._current_step_attempts.append(attempt)

    def finalize_step(self, result: bool, retry_count: int = 0) -> None:
        """结束当前 logical step,保存全部 attempt 历史和最终结果。

        Args:
            result: 该 logical step 的最终布尔结果。
            retry_count: 该 step 实际发生的 fresh retry 次数。

        Raises:
            ValueError: 状态非 running 或当前无 attempt 缓冲。
        """
        if self._state.status is not TaskStatus.RUNNING:
            raise ValueError("只有 running 任务可以结束步骤。")
        if type(result) is not bool:
            raise TypeError("result 必须是 bool。")
        if type(retry_count) is not int:
            raise TypeError("retry_count 必须是 int。")
        if retry_count < 0:
            raise ValueError("retry_count 不得小于 0。")
        step = TaskStep(
            attempts=self._current_step_attempts,
            result=result,
            retry_count=retry_count,
            recorded_at=self._now(),
        )
        self._state.steps.append(step)
        self._state.step_count += 1
        self._current_step_attempts = []

    def record_step(
        self,
        action: Mapping[str, object],
        result: bool,
        retry_count: int = 0,
    ) -> None:
        """单次 attempt 的便捷记录(等价于 record_attempt + finalize_step)。

        Args:
            action: 非空动作映射。
            result: 动作是否成功。
            retry_count: 此动作执行前发生的非负重试次数。
        Raises:
            TypeError: 参数类型或时钟返回类型错误。
            ValueError: 状态或动作内容不合法。
        """
        self.record_attempt(action, result, 0)
        self.finalize_step(result, retry_count)

    def record_retry(self, count: int = 1) -> None:
        """记录未产生控制步骤的有限重试次数。

        Args:
            count: 本次累计的正整数重试次数。

        Raises:
            TypeError: count 不是严格 int。
            ValueError: 当前状态不是 running，或 count 不是正整数。
        """
        if self._state.status is not TaskStatus.RUNNING:
            raise ValueError("只有 running 任务可以记录重试。")
        if type(count) is not int:
            raise TypeError("count 必须是 int。")
        if count <= 0:
            raise ValueError("count 必须大于 0。")
        self._state.retry_count += count

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
