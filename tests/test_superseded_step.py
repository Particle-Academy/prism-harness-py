"""DECISION B (G-387 round 13): a superseded parked step denies its open
gated calls, once.

A parked step is SUPERSEDED by any later assistant row with tool calls, or by
an assistant text row or a user row recorded OUTSIDE the harness run in
progress (no run id, another run id, or a run id handed to the public
``Thread.record`` outside that run). Each of its gated calls with no result
then gets ONE denial, "Not run: a later step replaced it.", on a tool result
row marked ``harness_superseded``; nothing of it runs, and a later decision
is refused. The harness's own rows inside its run never supersede.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_repeated_call_ids import Recording, park, rows

from prism_harness import HarnessError, Session, record_approval
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.thread import enter_run
from prism_harness.thread_rows import (
    SUPERSEDED,
    SUPERSEDED_RESULT,
    pending_approval_ids,
    superseded_approval_ids,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


SUPERSEDING: dict[str, dict[str, Any]] = {
    "assistant_text": {
        "type": "assistant",
        "content": "An aside.",
        "tool_calls": [],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    "assistant_step": {
        "type": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "later",
                "name": "echo",
                "arguments": {"n": 7},
                "result_id": None,
                "reasoning_id": None,
                "reasoning_summary": None,
            }
        ],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    "user": {"type": "user", "content": "Something else."},
}

STEPS = {
    "unique": (("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}), ("c", "lookup", {"n": 3})),
    "repeated": (("dup", "echo", {"n": 1}), ("dup", "echo", {"n": 2})),
}


def parked_run_id(session: Session) -> str:
    for entry in reversed(session.thread().messages()):
        if entry.message.get("tool_approval_requests"):
            assert entry.run_id is not None
            return entry.run_id
    raise AssertionError("nothing parked")


def outcomes_of(session: Session, n: int) -> list[str]:
    return [
        entry["result"]
        for row in rows(session)
        if row.get("type") == "tool_result"
        for entry in row["tool_results"]
        if entry["args"].get("n") == n
    ]


def marks(session: Session) -> list[Any]:
    return [row[SUPERSEDED] for row in rows(session) if SUPERSEDED in row]


@pytest.mark.parametrize("step", sorted(STEPS))
@pytest.mark.parametrize("run", ["none", "another", "forged"])
@pytest.mark.parametrize("kind", sorted(SUPERSEDING))
def test_a_host_recorded_row_supersedes_the_parked_step(kind: str, run: str, step: str) -> None:
    world = Recording()
    _, approvals = park(world, *STEPS[step])
    # A person approves the first call while the step is answerable.
    record_approval(world.fresh().root(), approvals[0], True)
    host = world.fresh().root()
    run_id = {"none": None, "another": "host-run", "forged": parked_run_id(host)}[run]

    host.thread().record([dict(SUPERSEDING[kind])], run_id)

    stored = rows(world.fresh().root())
    marked = [row for row in stored if SUPERSEDED in row]
    assert [row[SUPERSEDED] for row in marked] == [{"approval_ids": approvals}]
    assert list(marked[0]) == ["type", "tool_results", "tool_approval_responses", SUPERSEDED]
    # Placed right before the row that superseded the step.
    at = stored.index(marked[0])
    assert stored[at + 1]["type"] == SUPERSEDING[kind]["type"]
    assert superseded_approval_ids(stored) == set(approvals)
    assert approval_decisions(stored) == {}
    assert pending_approval_ids(stored) == []
    # A later decision is refused, and no resume runs anything of the step.
    for approval in approvals:
        with pytest.raises(HarnessError) as caught:
            record_approval(world.fresh().root(), approval, True)
        assert caught.value.code == "approval_not_pending"
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    world.text("Again.")
    world.fresh().runtime.send(world.root(), "")
    assert not any(name == "echo" for name, _ in world.ran_with)
    # Exactly ONE outcome per gated call of the parked step.
    assert outcomes_of(world.root(), 1) == [SUPERSEDED_RESULT]
    assert outcomes_of(world.root(), 2) == [SUPERSEDED_RESULT]


def test_the_harness_own_rows_in_its_run_do_not_supersede() -> None:
    """A resume answers the step and then writes its own text answer in its
    run: no denial, and the approved call ran once."""
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == [("echo", {"n": 1})]
    assert marks(world.root()) == []


@pytest.mark.parametrize("kind", ["assistant_text", "user"])
def test_a_text_or_user_row_inside_the_harness_run_does_not_supersede(kind: str) -> None:
    """The rule at the seam every row goes through: a text or user row the
    harness records under the run in progress on that thread is in the run."""
    world = Recording()
    park(world, ("a", "echo", {"n": 1}))
    root = world.fresh().root()
    leave = enter_run(f"{root.key()}:thread", "run-in-progress")
    try:
        root.thread().record([dict(SUPERSEDING[kind])], "run-in-progress")
    finally:
        leave()

    assert marks(world.root()) == []


def test_a_run_id_named_outside_its_run_is_outside() -> None:
    """A run id handed to the public API from inside ANOTHER thread's run is
    outside: it supersedes."""
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}))
    root = world.fresh().root()
    leave = enter_run("someone-else:thread", "run-x")
    try:
        root.thread().record([dict(SUPERSEDING["user"])], "run-x")
    finally:
        leave()

    assert marks(world.root()) == [{"approval_ids": approvals}]


def test_a_later_step_with_tool_calls_supersedes_even_inside_the_run() -> None:
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}))
    root = world.fresh().root()
    leave = enter_run(f"{root.key()}:thread", "run-in-progress")
    try:
        root.thread().record([dict(SUPERSEDING["assistant_step"])], "run-in-progress")
    finally:
        leave()

    assert marks(world.root()) == [{"approval_ids": approvals}]


def test_a_call_with_a_result_is_not_denied_again() -> None:
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))
    record_approval(world.fresh().root(), approvals[0], True)
    record_approval(world.fresh().root(), approvals[1], False, "no")
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    world.fresh().root().thread().record([dict(SUPERSEDING["user"])])

    assert marks(world.root()) == []
    assert outcomes_of(world.root(), 1) == ["echo:1"]


@pytest.mark.parametrize("decided", [False, True], ids=["undecided", "approved"])
def test_a_new_message_supersedes_the_parked_step(decided: bool) -> None:
    """A new message through the harness is a new run: the parked step's open
    gated calls are denied once before its turn, even one a person approved,
    and a later decision is refused."""
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))
    if decided:
        record_approval(world.fresh().root(), approvals[0], True)
    world.text("Answered.")

    world.fresh().runtime.send(world.root(), "Something else.")

    assert not any(name == "echo" for name, _ in world.ran_with)
    assert marks(world.root()) == [{"approval_ids": approvals}]
    assert outcomes_of(world.root(), 1) == [SUPERSEDED_RESULT]
    assert outcomes_of(world.root(), 2) == [SUPERSEDED_RESULT]
    for approval in approvals:
        with pytest.raises(HarnessError):
            record_approval(world.fresh().root(), approval, True)
