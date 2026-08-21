"""通用 progress-dependent logical-step extension 测试。"""

import asyncio

import pytest
from agentscope.message import Msg

from agent.task_manager import (
    ProgressDependentStepBudget,
    StepExtensionEvidence,
    TaskManager,
)
from config import DEFAULT_MAX_STEPS, GuiAgentSettings
from tests.agent_test_support import MemoryControls, SequenceBackend, make_agent


def _progress(**overrides: bool) -> StepExtensionEvidence:
    fields = {
        "task_unfinished": True,
        "recent_step_succeeded": True,
        "meaningful_progress": True,
        "latest_dispatch_transition": True,
    }
    fields.update(overrides)
    return StepExtensionEvidence(**fields)


def test_default_max_steps_remains_ten() -> None:
    assert DEFAULT_MAX_STEPS == 10
    assert GuiAgentSettings().max_steps == 10


def test_recent_real_progress_grants_extension() -> None:
    budget = ProgressDependentStepBudget(10)
    assert budget.evaluate(_progress())
    assert budget.extension_granted


def test_no_progress_streak_rejects_extension() -> None:
    budget = ProgressDependentStepBudget(10)
    assert not budget.evaluate(_progress(no_progress_active=True))
    assert budget.extension_reason == "no_progress_active"


def test_repeated_action_block_rejects_extension() -> None:
    budget = ProgressDependentStepBudget(10)
    assert not budget.evaluate(_progress(repeated_action_blocked=True))
    assert budget.extension_reason == "repeated_action_block_active"


def test_same_strategy_retry_exhaustion_rejects_extension() -> None:
    budget = ProgressDependentStepBudget(10)
    assert not budget.evaluate(_progress(same_strategy_retry_exhausted=True))
    assert budget.extension_reason == "same_strategy_retry_exhausted"


def test_earlier_progress_but_latest_no_progress_is_rejected() -> None:
    budget = ProgressDependentStepBudget(10)
    latest = _progress(meaningful_progress=False, latest_dispatch_transition=False)
    assert not budget.evaluate(latest)
    assert budget.extension_reason == "no_recent_meaningful_progress"


def test_extension_is_limited_to_three_steps() -> None:
    budget = ProgressDependentStepBudget(10)
    assert budget.evaluate(_progress())
    for step in (11, 12, 13):
        budget.begin_step(step)
    assert budget.extension_steps_used == 3
    assert budget.extension_steps_available == 0
    with pytest.raises(ValueError, match="hard limit"):
        budget.begin_step(14)


def test_extension_cannot_be_granted_recursively() -> None:
    budget = ProgressDependentStepBudget(10)
    assert budget.evaluate(_progress())
    assert not budget.evaluate(_progress())
    assert budget.effective_hard_limit == 13
    assert budget.extension_reason == "extension_already_evaluated"


def test_completion_during_extension_stops_immediately() -> None:
    backend = SequenceBackend(
        ["Action: click(x=10, y=20)", 'Action: finish(result="done")'],
    )
    controls = MemoryControls()
    manager = TaskManager("任务")
    agent = make_agent(
        backend,
        controls,
        manager,
        max_steps=1,
        retry_count=0,
        decision_protocol_v3=True,
    )
    agent._step_extension_evidence = (  # type: ignore[method-assign]
        lambda *args: _progress()
    )
    result = asyncio.run(agent(Msg("u", "任务", "user")))
    assert result.content == "done"
    assert backend.calls == 2
    assert len(controls.calls) == 1


def test_new_task_budget_starts_with_clean_state() -> None:
    first = ProgressDependentStepBudget(10)
    first.evaluate(_progress())
    first.begin_step(11)
    second = ProgressDependentStepBudget(10)
    assert second.trace_fields() == {
        "configured_max_steps": 10,
        "base_budget_exhausted": False,
        "extension_eligible": False,
        "extension_granted": False,
        "extension_steps_available": 0,
        "extension_steps_used": 0,
        "effective_hard_limit": 10,
        "extension_reason": "base_budget_not_exhausted",
    }


def test_custom_max_steps_is_preserved_and_optionally_extended() -> None:
    budget = ProgressDependentStepBudget(7)
    budget.evaluate(_progress())
    assert budget.configured_max_steps == 7
    assert budget.effective_hard_limit == 10


def test_trace_fields_are_accurate_after_one_extension_step() -> None:
    budget = ProgressDependentStepBudget(4)
    budget.evaluate(_progress())
    budget.begin_step(5)
    fields = budget.trace_fields()
    assert fields["configured_max_steps"] == 4
    assert fields["base_budget_exhausted"] is True
    assert fields["extension_eligible"] is True
    assert fields["extension_granted"] is True
    assert fields["extension_steps_available"] == 2
    assert fields["extension_steps_used"] == 1
    assert fields["effective_hard_limit"] == 7
    assert fields["extension_reason"] == "recent_measurable_progress"
