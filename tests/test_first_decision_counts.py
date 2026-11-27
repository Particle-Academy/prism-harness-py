"""The first decision for an approval stands (G-401).

Once an approval has a decision, a second decision for it records nothing,
and the resume acts on the first, whether the second came in the same batch,
after a turn that failed before acting, or after the step was resolved.
Entered at the public seam (``send``, ``record_approval``); the model is
scripted.
"""

from __future__ import annotations

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import HarnessError, LlmToolCall, Session, record_approval


def decisions_for(session: Session, approval: str) -> list[bool]:
    """Every decision stored for the approval, a resume row's restatement included."""
    return [
        decision["approved"]
        for entry in session.thread().messages()
        if entry.message["type"] == "tool_result"
        for decision in entry.message.get("tool_approval_responses", [])
        if decision["approval_id"] == approval
    ]


def results_for(session: Session, value: str) -> set[str]:
    return {
        result["result"]
        for entry in session.thread().messages()
        if entry.message["type"] == "tool_result"
        for result in entry.message["tool_results"]
        if result["args"].get("value") == value
    }


@pytest.mark.parametrize(
    ("first", "second", "runs"),
    [(False, True, False), (True, False, True)],
    ids=["deny-then-approve", "approve-then-deny"],
)
def test_a_second_decision_in_the_same_batch_records_nothing(
    first: bool, second: bool, runs: bool
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, first, None if first else "no")
    record_approval(session, approval, second, None if second else "no")
    runtime.send(session, "")
    runtime.send(session, "")

    assert set(decisions_for(session, approval)) == {first}
    assert counts.get("echo", 0) == (1 if runs else 0)
    assert results_for(session, "P") == ({"echo:P"} if runs else {"no"})


def test_a_second_decision_after_a_turn_that_failed_before_acting_records_nothing() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, False, "no")
    with pytest.raises(HarnessError):
        # Refused before any decision is acted on.
        runtime.send(session, "", ["not-a-tool"])
    record_approval(session, approval, True)
    runtime.send(session, "")
    runtime.send(session, "")

    assert set(decisions_for(session, approval)) == {False}
    assert counts == {}
    assert results_for(session, "P") == {"no"}


def test_a_decision_after_the_step_was_resolved_records_nothing() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, False, "no")
    runtime.send(session, "")
    record_approval(session, approval, True)
    runtime.send(session, "")

    assert set(decisions_for(session, approval)) == {False}
    assert counts == {}
    assert results_for(session, "P") == {"no"}


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "deny"])
def test_a_resent_identical_decision_writes_no_row_and_runs_nothing_more(approved: bool) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, approved, None if approved else "no")
    rows = len(session.thread().messages())
    record_approval(session, approval, approved, None if approved else "no")

    assert len(session.thread().messages()) == rows
    runtime.send(session, "")
    runtime.send(session, "")
    assert counts.get("echo", 0) == (1 if approved else 0)
