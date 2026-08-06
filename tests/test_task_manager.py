"""测试任务状态转换、时钟原子性和快照隔离。"""

from collections import UserDict
from datetime import datetime, timezone

import pytest

from agent.task_manager import TaskManager, TaskState, TaskStatus, TaskStep

STARTED_AT = datetime(2026, 7, 31, 9, 0, tzinfo=timezone.utc)
RECORDED_AT = datetime(2026, 7, 31, 9, 1, tzinfo=timezone.utc)
ENDED_AT = datetime(2026, 7, 31, 9, 2, tzinfo=timezone.utc)


class SequenceClock:
    """按顺序返回预设时间或异常的可调用时钟。"""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.call_count = 0

    def __call__(self) -> datetime:
        """返回下一个预设结果。"""
        self.call_count += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]  # 测试非法时钟返回。


def test_initial_state_preserves_description() -> None:
    """初始快照完整且任务描述不被 strip。"""
    manager = TaskManager("  保留空格  ")
    state = manager.state
    assert state == TaskState(task_description="  保留空格  ")
    assert state.status is TaskStatus.PENDING
    assert state.started_at is None
    assert state.ended_at is None
    assert state.steps == []
    assert state.step_count == 0
    assert state.retry_count == 0
    assert state.failure_reason is None


@pytest.mark.parametrize("description", ["", " ", "\t\r\n"])
def test_constructor_rejects_blank_description(description: str) -> None:
    """拒绝空或纯空白任务描述。"""
    with pytest.raises(ValueError):
        TaskManager(description)


@pytest.mark.parametrize("description", [None, 1, [], object()])
def test_constructor_rejects_non_string_description(
    description: object,
) -> None:
    """任务描述必须为字符串。"""
    with pytest.raises(TypeError):
        TaskManager(description)  # type: ignore[arg-type]


@pytest.mark.parametrize("clock", [1, "clock", object()])
def test_constructor_rejects_non_callable_clock(clock: object) -> None:
    """注入时钟必须可调用。"""
    with pytest.raises(TypeError):
        TaskManager("task", clock=clock)  # type: ignore[arg-type]


def test_default_clock_produces_datetime() -> None:
    """默认时钟能够建立 datetime 开始时间。"""
    manager = TaskManager("task")
    manager.start()
    assert isinstance(manager.state.started_at, datetime)


def test_start_calls_clock_once_and_sets_running() -> None:
    """pending 只通过一次时钟读取进入 running。"""
    clock = SequenceClock([STARTED_AT])
    manager = TaskManager("task", clock=clock)
    assert manager.start() is None
    state = manager.state
    assert clock.call_count == 1
    assert state.status is TaskStatus.RUNNING
    assert state.started_at == STARTED_AT
    assert state.ended_at is None


@pytest.mark.parametrize("clock_outcome", [None, "time", 1])
def test_start_rejects_non_datetime_without_partial_change(
    clock_outcome: object,
) -> None:
    """非法时钟返回不改变 pending 状态。"""
    manager = TaskManager("task", clock=SequenceClock([clock_outcome]))
    before = manager.state
    with pytest.raises(TypeError):
        manager.start()
    assert manager.state == before


def test_start_clock_exception_does_not_change_state() -> None:
    """时钟异常发生时不产生部分状态更新。"""
    failure = RuntimeError("clock failed")
    manager = TaskManager("task", clock=SequenceClock([failure]))
    before = manager.state
    with pytest.raises(RuntimeError) as caught:
        manager.start()
    assert caught.value is failure
    assert manager.state == before


@pytest.mark.parametrize(
    "operation",
    [
        lambda manager: manager.succeed(),
        lambda manager: manager.fail("reason"),
        lambda manager: manager.record_step({"action_type": "click"}, True),
    ],
)
def test_pending_rejects_non_start_operations(operation: object) -> None:
    """pending 不允许记录或直接进入终态。"""
    manager = TaskManager("task")
    with pytest.raises(ValueError):
        operation(manager)  # type: ignore[operator]
    assert manager.state.status is TaskStatus.PENDING


def _running_manager(
    clock_outcomes: list[object] | None = None,
) -> tuple[TaskManager, SequenceClock]:
    """创建使用可控时钟的运行中任务。"""
    clock = SequenceClock(clock_outcomes or [STARTED_AT])
    manager = TaskManager("task", clock=clock)
    manager.start()
    return manager, clock


@pytest.mark.parametrize(
    ("action", "error_type"),
    [
        (None, TypeError),
        ([], TypeError),
        ("action", TypeError),
        ({}, ValueError),
        (UserDict(), ValueError),
    ],
)
def test_record_step_rejects_invalid_action(
    action: object,
    error_type: type[Exception],
) -> None:
    """步骤动作必须为非空 Mapping。"""
    manager, _ = _running_manager()
    before = manager.state
    with pytest.raises(error_type):
        manager.record_step(action, True)  # type: ignore[arg-type]
    assert manager.state == before


@pytest.mark.parametrize("result", [1, 0, None, "true"])
def test_record_step_rejects_non_boolean_result(result: object) -> None:
    """步骤结果必须严格为 bool。"""
    manager, _ = _running_manager()
    before = manager.state
    with pytest.raises(TypeError):
        manager.record_step(
            {"action_type": "click"},
            result,  # type: ignore[arg-type]
        )
    assert manager.state == before


@pytest.mark.parametrize("retry_count", [True, False, 1.0, "1", None])
def test_record_step_rejects_non_integer_retry_count(
    retry_count: object,
) -> None:
    """重试次数拒绝 bool 和其他非 int 类型。"""
    manager, _ = _running_manager()
    before = manager.state
    with pytest.raises(TypeError):
        manager.record_step(
            {"action_type": "click"},
            True,
            retry_count,  # type: ignore[arg-type]
        )
    assert manager.state == before


def test_record_step_rejects_negative_retry_count() -> None:
    """重试次数不得为负数。"""
    manager, _ = _running_manager()
    before = manager.state
    with pytest.raises(ValueError):
        manager.record_step({"action_type": "click"}, True, -1)
    assert manager.state == before


@pytest.mark.parametrize("retry_count", [0, 1, 3])
def test_record_step_saves_mapping_and_updates_totals(
    retry_count: int,
) -> None:
    """合法 Mapping 被保存并更新步骤及重试计数。"""
    manager, clock = _running_manager([STARTED_AT, RECORDED_AT])
    action = UserDict(
        {"action_type": "click", "params": {"x": 1, "y": 2}},
    )
    assert manager.record_step(action, False, retry_count) is None
    state = manager.state
    assert clock.call_count == 2
    assert state.step_count == 1
    assert state.retry_count == retry_count
    assert state.steps == [
        TaskStep(
            action=dict(action),
            result=False,
            recorded_at=RECORDED_AT,
            retry_count=retry_count,
        ),
    ]


def test_record_step_deep_copies_nested_action() -> None:
    """调用方后续修改动作及嵌套参数不影响内部状态。"""
    manager, _ = _running_manager([STARTED_AT, RECORDED_AT])
    action = {"action_type": "click", "params": {"x": 1, "y": 2}}
    manager.record_step(action, True)
    action["action_type"] = "changed"
    action["params"]["x"] = 99  # type: ignore[index]
    saved = manager.state.steps[0].action
    assert saved == {
        "action_type": "click",
        "params": {"x": 1, "y": 2},
    }


def test_multiple_steps_preserve_invariants() -> None:
    """多步骤后计数与步骤重试总和保持一致。"""
    manager, _ = _running_manager(
        [STARTED_AT, RECORDED_AT, ENDED_AT],
    )
    manager.record_step({"action_type": "click"}, True, 0)
    manager.record_step({"action_type": "type"}, False, 2)
    state = manager.state
    assert state.step_count == len(state.steps) == 2
    assert state.retry_count == 2
    assert state.retry_count == sum(step.retry_count for step in state.steps)


@pytest.mark.parametrize("clock_outcome", [None, "time", RuntimeError("bad")])
def test_record_step_clock_failure_has_no_partial_write(
    clock_outcome: object,
) -> None:
    """复制和参数校验完成后的时钟失败仍不写入步骤。"""
    manager, _ = _running_manager([STARTED_AT, clock_outcome])
    before = manager.state
    with pytest.raises((TypeError, RuntimeError)):
        manager.record_step({"action_type": "click"}, True, 2)
    assert manager.state == before


def test_succeed_sets_terminal_invariants() -> None:
    """running 可通过一次时钟读取进入 success。"""
    manager, clock = _running_manager([STARTED_AT, ENDED_AT])
    assert manager.succeed() is None
    state = manager.state
    assert clock.call_count == 2
    assert state.status is TaskStatus.SUCCESS
    assert state.started_at == STARTED_AT
    assert state.ended_at == ENDED_AT
    assert state.failure_reason is None


def test_fail_preserves_original_reason() -> None:
    """running 失败时原样保存非空原因。"""
    manager, clock = _running_manager([STARTED_AT, ENDED_AT])
    assert manager.fail("  原始原因  ") is None
    state = manager.state
    assert clock.call_count == 2
    assert state.status is TaskStatus.FAILED
    assert state.ended_at == ENDED_AT
    assert state.failure_reason == "  原始原因  "


@pytest.mark.parametrize("reason", ["", " ", "\t\r\n"])
def test_fail_rejects_blank_reason(reason: str) -> None:
    """失败原因不得为空或纯空白。"""
    manager, _ = _running_manager()
    before = manager.state
    with pytest.raises(ValueError):
        manager.fail(reason)
    assert manager.state == before


@pytest.mark.parametrize("reason", [None, 1, [], object()])
def test_fail_rejects_non_string_reason(reason: object) -> None:
    """失败原因必须为字符串。"""
    manager, _ = _running_manager()
    before = manager.state
    with pytest.raises(TypeError):
        manager.fail(reason)  # type: ignore[arg-type]
    assert manager.state == before


@pytest.mark.parametrize("terminal_method", ["succeed", "fail"])
def test_terminal_clock_failure_has_no_partial_change(
    terminal_method: str,
) -> None:
    """结束时钟失败不会部分写入终态字段。"""
    manager, _ = _running_manager([STARTED_AT, None])
    before = manager.state
    with pytest.raises(TypeError):
        if terminal_method == "succeed":
            manager.succeed()
        else:
            manager.fail("reason")
    assert manager.state == before


@pytest.mark.parametrize("terminal_status", ["success", "failed"])
@pytest.mark.parametrize("operation", ["start", "succeed", "fail", "record"])
def test_terminal_state_rejects_all_mutations(
    terminal_status: str,
    operation: str,
) -> None:
    """success 和 failed 终态拒绝全部后续修改。"""
    manager, _ = _running_manager([STARTED_AT, ENDED_AT])
    if terminal_status == "success":
        manager.succeed()
    else:
        manager.fail("reason")
    before = manager.state
    with pytest.raises(ValueError):
        if operation == "start":
            manager.start()
        elif operation == "succeed":
            manager.succeed()
        elif operation == "fail":
            manager.fail("new reason")
        else:
            manager.record_step({"action_type": "click"}, True)
    assert manager.state == before


def test_running_rejects_repeated_start() -> None:
    """running 状态不能再次 start。"""
    manager, _ = _running_manager()
    before = manager.state
    with pytest.raises(ValueError):
        manager.start()
    assert manager.state == before


def test_state_returns_fully_isolated_deep_copies() -> None:
    """快照对象、列表、步骤和嵌套动作均与内部状态隔离。"""
    manager, _ = _running_manager([STARTED_AT, RECORDED_AT])
    manager.record_step(
        {"action_type": "click", "params": {"x": 1, "y": 2}},
        True,
    )
    first = manager.state
    second = manager.state
    assert first is not second
    assert first.steps is not second.steps
    assert first.steps[0] is not second.steps[0]
    assert first.steps[0].action is not second.steps[0].action
    first.status = TaskStatus.FAILED
    first.steps.clear()
    second.steps[0].action["params"]["x"] = 99  # type: ignore[index]
    current = manager.state
    assert current.status is TaskStatus.RUNNING
    assert current.step_count == 1
    assert current.steps[0].action["params"] == {"x": 1, "y": 2}


def test_manager_instances_do_not_share_mutable_state() -> None:
    """不同任务实例之间没有共享步骤列表。"""
    first, _ = _running_manager([STARTED_AT, RECORDED_AT])
    second, _ = _running_manager([STARTED_AT])
    first.record_step({"action_type": "click"}, True)
    assert first.state.step_count == 1
    assert second.state.step_count == 0
