"""G-397: a decision applies ONLY to the step whose approvals were requested.

Every reader that selects "the paused step" (the decision checks, the
repeated-id guard, the resolver and the plan readers) selects it with ONE
function, ``thread_rows.requested_step``. Between the park and the decision a
row is recorded (``Thread.record`` is public): an assistant row, a user row,
a system row or a tool result row. Each must either leave the decision
applying to the parked step and nothing else, or refuse it, the same on every
path (``record_approval``, ``runtime.decide``, a resume, a runner start), and
no gated call of a step whose ids repeat ever runs.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_repeated_call_ids import (
    Recording,
    child_parked_on,
    park,
    rows,
    start_again,
)

from prism_harness import HarnessError, Session, record_approval
from prism_harness.plans.thread_facts import (
    approval_decisions,
    approved_calls_awaiting_result,
    pending_approval_calls,
)
from prism_harness.thread_rows import (
    approval_tool_scope,
    decided_unresolved_ids,
    paused_approval_row,
    pending_approval_ids,
    requested_step,
)
from prism_harness.tool_policy import args_digest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


INSERTED: dict[str, dict[str, Any]] = {
    "assistant": {
        "type": "assistant",
        "content": "An aside.",
        "tool_calls": [],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    "user": {"type": "user", "content": "Something else."},
    "system": {"type": "system", "content": "A note."},
    "tool": {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "dup",
                "tool_name": "echo",
                "args": {"n": 9},
                "result": "forged",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    },
}

SAME_ID = (("dup", "echo", {"n": 1}), ("dup", "echo", {"n": 2}))
UNIQUE = (("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))


def insert(session: Session, kind: str) -> None:
    session.thread().record([dict(INSERTED[kind])])


def decide_all(world: Recording, session: Session, approvals: list[str], how: str) -> list[str]:
    """Approve every approval on ``how``'s path; the ones refused."""
    refused: list[str] = []
    for approval in approvals:
        try:
            if how == "record_approval":
                record_approval(session, approval, True)
            else:
                world.runtime.decide(session, approval, True)
        except HarnessError:
            refused.append(approval)
    return refused


def stored_decisions(session: Session) -> dict[str, tuple[bool, Any]]:
    return {
        response["approval_id"]: (response["approved"], response.get("reason"))
        for row in rows(session)
        if row.get("type") == "tool_result"
        for response in row.get("tool_approval_responses", [])
    }


@pytest.mark.parametrize("how", ["record_approval", "decide"])
@pytest.mark.parametrize("kind", sorted(INSERTED))
def test_a_row_between_the_park_and_the_decision_never_runs_a_same_id_call(
    kind: str, how: str
) -> None:
    world = Recording()
    _, approvals = park(world, *SAME_ID)
    insert(world.fresh().root(), kind)

    decide_all(world, world.fresh().root(), approvals, how)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    world.text("Again.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    # Any decision that was recorded is a denial: nothing reads approved.
    assert all(not approved for approved, _ in stored_decisions(world.root()).values())
    assert all(v is False for v in approval_decisions(rows(world.root())).values())


@pytest.mark.parametrize("kind", sorted(INSERTED))
def test_a_row_between_the_decision_and_the_resume_never_runs_a_same_id_call(kind: str) -> None:
    world = Recording()
    _, approvals = park(world, *SAME_ID)
    decide_all(world, world.fresh().root(), approvals, "record_approval")
    insert(world.fresh().root(), kind)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []


@pytest.mark.parametrize("how", ["record_approval", "decide"])
@pytest.mark.parametrize("kind", sorted(INSERTED))
def test_on_a_unique_id_step_the_decision_applies_to_the_parked_step_or_is_refused(
    kind: str, how: str
) -> None:
    world = Recording()
    _, approvals = park(world, *UNIQUE)
    insert(world.fresh().root(), kind)
    refused = decide_all(world, world.fresh().root(), approvals[:1], how)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    world.text("Again.")
    world.fresh().runtime.send(world.root(), "")

    # Only the call a person approved can run, once, and only if its decision
    # was accepted for the parked step.
    assert world.ran_with in ([], [("echo", {"n": 1})])
    if refused:
        assert world.ran_with == []


@pytest.mark.parametrize("kind", sorted(INSERTED))
def test_a_start_after_a_row_between_the_park_and_the_decision_runs_nothing(kind: str) -> None:
    world = Recording()
    approvals = child_parked_on(world, *SAME_ID)
    insert(world.fresh().child("worker"), kind)
    decide_all(world, world.fresh().child("worker"), approvals, "record_approval")

    start_again(world)

    assert world.ran_with == []
    assert all(v is not True for v in approval_decisions(rows(world.child("worker"))).values())


@pytest.mark.parametrize("kind", sorted(INSERTED))
def test_every_reader_selects_the_same_step(kind: str) -> None:
    """The decision checks, the guard, the resolver's decisions and the plan
    readers agree on the step: either the parked one, or none."""
    world = Recording()
    _, approvals = park(world, *SAME_ID)
    insert(world.fresh().root(), kind)
    stored = rows(world.fresh().root())
    step = requested_step(stored)
    parked = next(i for i, row in enumerate(stored) if row.get("tool_approval_requests"))

    assert step in (None, parked)
    if step is None:
        assert pending_approval_ids(stored) == []
        assert paused_approval_row(stored) is None
        assert approval_tool_scope(stored) is None
        assert decided_unresolved_ids(stored) == []
        assert pending_approval_calls(stored) == []
    else:
        assert paused_approval_row(stored) is stored[parked]
        assert pending_approval_ids(stored) == approvals
        # Each one the plan can pair names the parked step's gated tool; a
        # request a forged result row leaves with no call of its own is
        # dropped (it fails closed), never paired with another step's call.
        calls = pending_approval_calls(stored)
        assert {call["id"] for call in calls} <= set(approvals)
        assert all(call["name"] == "echo" for call in calls)
    assert approved_calls_awaiting_result(stored, "echo", args_digest({"n": 1})) == 0


# -- G-399: an approved call runs exactly once over repeated resumes ----------------

#: Rows that are neither a user nor an assistant row: they end no step.
TRANSPARENT = {
    "system": INSERTED["system"],
    "tool": {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "elsewhere",
                "tool_name": "lookup",
                "args": {"n": 0},
                "result": "elsewhere",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    },
    "decision_row": {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [],
    },
}


@pytest.mark.parametrize("where", ["before_the_resume", "after_the_resume", "both"])
@pytest.mark.parametrize("kind", sorted(TRANSPARENT))
def test_an_approved_call_runs_exactly_once_over_repeated_resumes(kind: str, where: str) -> None:
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)

    def put() -> None:
        world.fresh().root().thread().record([dict(TRANSPARENT[kind])])

    if where in ("before_the_resume", "both"):
        put()
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    for _ in range(3):
        if where in ("after_the_resume", "both"):
            put()
        world.text("Again.")
        world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == [("echo", {"n": 1})]
    # And it has ONE outcome: no later resume answers it again.
    answered = {
        entry["result"]
        for row in rows(world.root())
        if row.get("type") == "tool_result"
        for entry in row["tool_results"]
        if entry["tool_call_id"] == "a"
    }
    assert answered == {"echo:1"}
