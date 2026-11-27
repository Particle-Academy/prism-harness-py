"""Any deny recorded for an approval is final (G-401, option denial-wins).

On write and on read: a deny is final, so an approve after a deny records
nothing, and a deny after an approve revokes it; a re-sent identical decision
records nothing. The resolver asks whether any deny exists for the approval,
so a decision pair already in the thread, or two writes that race, read the
same way. Entered at the public seam (``send``, ``record_approval``,
``Thread.record``); the model is scripted.
"""

from __future__ import annotations

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import HarnessError, LlmToolCall, Session, record_approval


def decision_rows(session: Session, approval: str) -> int:
    return sum(
        1
        for entry in session.thread().messages()
        if entry.message["type"] == "tool_result" and not entry.message["tool_results"]
        for decision in entry.message.get("tool_approval_responses", [])
        if decision["approval_id"] == approval
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [(False, True), (True, False)],
    ids=["deny-then-approve", "approve-then-deny"],
)
def test_in_one_batch_a_deny_is_final_whichever_came_first(first: bool, second: bool) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, first, None if first else "no")
    record_approval(session, approval, second, None if second else "no")
    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {}


def test_an_approve_after_a_failed_turn_does_not_lift_a_deny() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, False, "no")
    with pytest.raises(HarnessError):
        runtime.send(session, "", ["not-a-tool"])
    record_approval(session, approval, True)
    runtime.send(session, "")

    assert counts == {}
    assert decision_rows(session, approval) == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [(False, True), (True, False)],
    ids=["deny-then-approve", "approve-then-deny"],
)
def test_a_decision_pair_already_in_the_thread_is_read_deny_wins(first: bool, second: bool) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    session.thread().record(
        [
            {
                "type": "tool_result",
                "tool_results": [],
                "tool_approval_responses": [
                    {"approval_id": approval, "approved": verdict, "reason": None}
                ],
            }
            for verdict in (first, second)
        ]
    )
    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {}


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "deny"])
def test_a_resent_identical_decision_records_nothing(approved: bool) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, approved, None if approved else "no")
    record_approval(session, approval, approved, None if approved else "no")

    assert decision_rows(session, approval) == 1
    runtime.send(session, "")
    assert counts.get("echo", 0) == (1 if approved else 0)


def test_a_deny_after_the_step_was_resolved_runs_nothing_more() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    runtime.send(session, "")
    record_approval(session, approval, False, "late")
    runtime.send(session, "")

    assert counts == {"echo": 1}


def test_an_approve_never_acted_on_then_a_deny_runs_nothing() -> None:
    # (d): the approve is stored, a turn fails before acting on it, then a
    # deny arrives. A call a person denied never runs.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    with pytest.raises(HarnessError):
        # Refused before any decision is acted on.
        runtime.send(session, "", ["not-a-tool"])
    record_approval(session, approval, False, "no")
    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {}
