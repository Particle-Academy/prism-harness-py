"""ABSOLUTE SEGMENT OWNERSHIP (G-387 round 13b).

A step's SEGMENT is the rows after its assistant row and before the next
assistant row with tool calls. A stored result counts ONLY for calls of the
step whose segment it sits in; within that segment it is matched to a call by
exact arguments, then by id and tool name, then by id. An earlier step's gated
calls with no result of their own get the replaced denial. Every gated call
has exactly one outcome, and nothing runs twice.

The case this pins: a result recorded in a LATER step's segment, under an id
that an EARLIER step's call also carries, whose stored arguments are LOSSY:
they differ from the later call's own and match the earlier call's exactly.
Another writer on a shared store can leave exactly that. A rule that let the
exact arguments carry the result back to the earlier call would leave the
approved later call with no result, so it would run again on every resume.
Here the result answers the later call (it fails closed: the call is done),
the earlier call keeps its one replaced denial, and the approved later call
runs at most once over three resumes.

Every test enters at the public seam (``send``, ``record_approval``,
``Thread.record``); the model is scripted and the network is refused.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_repeated_call_ids import Recording, park, rows, say

from prism_harness import record_approval
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.thread_rows import (
    REPLACED_STEP,
    SUPERSEDED,
    SUPERSEDED_RESULT,
    decided_unresolved_ids,
    pending_approval_ids,
    superseded_approval_ids,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


EARLIER = {"n": 1}
NEWEST = {"n": 2}


def lossy_result(text: str) -> dict[str, Any]:
    """A result for the newest call as another writer stored it: the newest
    call's id and tool, with the EARLIER call's arguments exactly."""
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "x",
                "tool_name": "echo",
                "args": dict(EARLIER),
                "result": text,
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    }


def segments(stored: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The results of each step with tool calls, read from ITS segment only
    (an independent reading, never the code under test)."""
    found: list[list[dict[str, Any]]] = []
    for row in stored:
        if row.get("type") == "assistant" and row.get("tool_calls"):
            found.append([])
        elif row.get("type") == "tool_result" and found:
            found[-1].extend(row["tool_results"])
    return found


def two_steps(world: Recording, approve_earlier: bool) -> tuple[str, str]:
    """An earlier step parks on call ``x``; a new message replaces it; the
    newest step parks on a call under the SAME id ``x``; a person approves it."""
    _, [old] = park(world, ("x", "echo", dict(EARLIER)))
    if approve_earlier:
        record_approval(world.fresh().root(), old, True)
    say(world, ("x", "echo", dict(NEWEST)))
    parked = world.fresh().runtime.send(world.root(), "Something else.")
    assert parked.finish_reason == "awaiting_approval"
    [new] = [pending.id for pending in parked.pending_approvals]
    record_approval(world.fresh().root(), new, True)
    return old, new


def resume(world: Recording) -> None:
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")


def assert_earlier_has_its_one_denial(stored: list[dict[str, Any]], old: str) -> None:
    earlier, _newest = segments(stored)
    assert [(e["args"], e["result"]) for e in earlier] == [(EARLIER, SUPERSEDED_RESULT)]
    # Answered once, in its own segment: never again, by a mark, after the newest step.
    assert not any(REPLACED_STEP in row for row in stored)
    assert [row[SUPERSEDED] for row in stored if SUPERSEDED in row] == [{"approval_ids": [old]}]
    assert old in superseded_approval_ids(stored)
    assert old not in approval_decisions(stored)


@pytest.mark.parametrize("approve_earlier", [False, True], ids=["earlier-undecided", "approved"])
def test_a_lossy_result_in_the_newest_segment_answers_the_newest_call(
    approve_earlier: bool,
) -> None:
    """The lossy result is the newest call's only result: it answers that
    call (the call is done and never runs), not the earlier one."""
    world = Recording()
    old, new = two_steps(world, approve_earlier)

    world.fresh().root().thread().record([lossy_result("echo:2 (another writer)")])
    for _ in range(3):
        resume(world)

    stored = rows(world.fresh().root())
    assert [name for name, _ in world.ran_with if name == "echo"] == []
    assert_earlier_has_its_one_denial(stored, old)
    _earlier, newest = segments(stored)
    assert {e["result"] for e in newest} == {"echo:2 (another writer)"}
    assert approval_decisions(stored) == {new: True}
    assert pending_approval_ids(stored) == []
    assert decided_unresolved_ids(stored) == []


@pytest.mark.parametrize("approve_earlier", [False, True], ids=["earlier-undecided", "approved"])
def test_the_approved_newest_call_runs_exactly_once_over_three_resumes(
    approve_earlier: bool,
) -> None:
    """The approved newest call runs on the first resume; a lossy result
    recorded after it changes nothing: two more resumes run nothing, and the
    earlier call still has exactly its one denial."""
    world = Recording()
    old, new = two_steps(world, approve_earlier)

    resume(world)
    assert world.ran_with == [("echo", NEWEST)]
    world.fresh().root().thread().record([lossy_result("echo:2 (restated)")])
    resume(world)
    resume(world)

    stored = rows(world.fresh().root())
    assert world.ran_with == [("echo", NEWEST)]
    assert_earlier_has_its_one_denial(stored, old)
    _earlier, newest = segments(stored)
    # The newest call's first result is its own run; nothing of it is pending.
    assert newest[0]["args"] == NEWEST and newest[0]["result"] == "echo:2"
    assert approval_decisions(stored) == {new: True}
    assert pending_approval_ids(stored) == []
    assert decided_unresolved_ids(stored) == []


def test_a_lossy_result_before_the_decision_still_answers_the_newest_call() -> None:
    """Recorded while the newest call is still pending: the call is done, a
    decision after it runs nothing, and the earlier call keeps its denial."""
    world = Recording()
    _, [old] = park(world, ("x", "echo", dict(EARLIER)))
    say(world, ("x", "echo", dict(NEWEST)))
    parked = world.fresh().runtime.send(world.root(), "Something else.")
    [new] = [pending.id for pending in parked.pending_approvals]

    world.fresh().root().thread().record([lossy_result("echo:2 (another writer)")])
    # The call has a result, so the decision is recorded and acts on nothing.
    record_approval(world.fresh().root(), new, True)
    for _ in range(3):
        resume(world)

    stored = rows(world.fresh().root())
    assert world.ran_with == []
    assert_earlier_has_its_one_denial(stored, old)
    _earlier, newest = segments(stored)
    assert {e["result"] for e in newest} == {"echo:2 (another writer)"}
