"""Each tool call is decided by its own approval (G-392).

A step is one assistant message with tool calls and the approval requests
issued for it. When a tool call id repeats among the step's calls (an empty id
counts), or a ``tool_call_id`` repeats among its approval requests, the step is
ambiguous: every gated call in it that has no result yet is denied and never
runs, whatever was decided, each with its own result. A step whose ids are all
unique is decided call by call, as before.

Some providers reuse an id within one response (one uses the function name),
so two calls can arrive under one id. Every test enters at the public seam
(``send``, ``record_approval``, ``SubagentRunner.run``); the model is scripted
and the network is refused.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_step_allowance import World, rewrite_paused_row
from test_step_allowance_r2 import drop_child_field

from prism_harness import (
    AgentRuntime,
    HarnessError,
    LlmResponse,
    LlmToolCall,
    RunBudget,
    RunContext,
    Session,
    SubagentRunner,
    record_approval,
)
from prism_harness.plans.thread_facts import pending_approval_calls
from prism_harness.thread_rows import ENDED_BY_START, ended_approval_ids, pending_approval_ids
from prism_harness.tool_policy import args_digest

REPEATED = "Tool call was not run because another call in this step has the same id"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


class Recording(World):
    """The World whose tools record the arguments they ran with."""

    def __init__(self) -> None:
        self.ran_with: list[tuple[str, dict[str, Any]]] = []
        super().__init__()

    def _runtime(self) -> AgentRuntime:
        runtime = super()._runtime()
        world = self

        class Tool:
            def __init__(self, name: str) -> None:
                self.name = name

            def handle(self, args: dict[str, Any]) -> Any:
                world.ran_with.append((self.name, dict(args)))
                return f"{self.name}:{args.get('n')}"

        runtime._tools.register(Tool("echo")).register(Tool("lookup"))
        return runtime


def say(world: World, *calls: tuple[str, str, dict[str, Any]]) -> None:
    world.queue.append(
        LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(call_id, name, args) for call_id, name, args in calls],
        )
    )


def rows(session: Session) -> list[dict[str, Any]]:
    """The rows as every reader takes them (:meth:`Thread.rows`): a harness
    mark counts only on a row read with the harness's own flag (G-387 round
    16), which the stored messages alone do not carry."""
    return session.thread().rows()


def stored_messages(session: Session) -> list[dict[str, Any]]:
    """The stored messages, byte for byte as written."""
    return [entry.message for entry in session.thread().messages()]


def results_after_park(session: Session) -> list[tuple[str, str, Any]]:
    """Every stored result written after the parked step, as stored (never the
    provider view, which folds results by call id)."""
    stored = rows(session)
    index = max(
        i for i, row in enumerate(stored) if row["type"] == "assistant" and row["tool_calls"]
    )
    return [
        (entry["tool_call_id"], entry["result"], entry["args"].get("n"))
        for row in stored[index + 1 :]
        if row["type"] == "tool_result"
        for entry in row["tool_results"]
    ]


def park(world: World, *calls: tuple[str, str, dict[str, Any]]) -> tuple[Session, list[str]]:
    """A root turn that parks on the step given."""
    session = world.root()
    say(world, *calls)
    parked = world.runtime.send(session, "Go")
    assert parked.finish_reason == "awaiting_approval"
    return session, [pending.id for pending in parked.pending_approvals]


SAME_ID = (("dup", "echo", {"n": 1}), ("dup", "echo", {"n": 2}))
EMPTY_ID = (("", "echo", {"n": 1}), ("", "echo", {"n": 2}))


@pytest.mark.parametrize("calls", [SAME_ID, EMPTY_ID], ids=["same-id", "empty-id"])
@pytest.mark.parametrize(
    "decisions",
    [(False, True), (True, False), (True, True), (False, False)],
    ids=["deny1-approve2", "approve1-deny2", "approve-both", "deny-both"],
)
def test_two_gated_calls_under_one_id_never_run(
    calls: tuple[tuple[str, str, dict[str, Any]], ...], decisions: tuple[bool, bool]
) -> None:
    world = Recording()
    _, approvals = park(world, *calls)
    assert len(approvals) == 2
    assert len(set(approvals)) == 2

    for approval, approved in zip(approvals, decisions, strict=True):
        record_approval(world.fresh().root(), approval, approved, None if approved else "no")
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    call_id = calls[0][0]
    assert results_after_park(world.root()) == [
        (call_id, REPEATED, 1),
        (call_id, REPEATED, 2),
    ]
    # No later decision or resume can make either run.
    for approval in approvals:
        with pytest.raises(HarnessError):
            record_approval(world.fresh().root(), approval, True)
    world.text("Again.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == []


def test_an_undecided_ambiguous_step_is_denied_with_the_same_reason() -> None:
    world = Recording()
    park(world, *SAME_ID)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    assert [result for _, result, _ in results_after_park(world.root())] == [REPEATED, REPEATED]


def test_an_ungated_call_in_an_ambiguous_step_keeps_running() -> None:
    world = Recording()
    _, approvals = park(
        world, ("dup", "echo", {"n": 1}), ("dup", "echo", {"n": 2}), ("solo", "lookup", {"n": 3})
    )
    assert world.ran_with == [("lookup", {"n": 3})]
    for approval in approvals:
        record_approval(world.fresh().root(), approval, True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == [("lookup", {"n": 3})]
    assert [r for r in results_after_park(world.root()) if r[1] == REPEATED] == [
        ("dup", REPEATED, 1),
        ("dup", REPEATED, 2),
    ]


def test_an_ungated_call_sharing_the_gated_calls_id_never_lets_it_run() -> None:
    world = Recording()
    _, approvals = park(world, ("", "lookup", {"n": 1}), ("", "echo", {"n": 2}))
    assert world.ran_with == [("lookup", {"n": 1})]
    for approval in approvals:
        try:
            record_approval(world.fresh().root(), approval, True)
        except HarnessError:
            pass
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert ("echo", {"n": 2}) not in world.ran_with


def test_unique_ids_are_still_decided_call_by_call() -> None:
    world = Recording()
    _, approvals = park(
        world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}), ("c", "echo", {"n": 3})
    )
    record_approval(world.fresh().root(), approvals[0], False, "not this one")
    record_approval(world.fresh().root(), approvals[1], True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == [("echo", {"n": 2})]
    assert results_after_park(world.root()) == [
        ("a", "not this one", 1),
        ("b", "echo:2", 2),
        ("c", "No approval response provided", 3),
    ]


def test_a_repeated_id_among_the_requests_alone_makes_the_step_ambiguous() -> None:
    """A stored step whose calls are unique but whose requests both name one
    call (written by another writer): neither call runs."""
    world = Recording()
    session, approvals = park(world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))

    def same_call(row: dict[str, Any]) -> None:
        for request in row["tool_approval_requests"]:
            request["tool_call_id"] = "a"

    rewrite_paused_row(session, same_call)
    for approval in approvals:
        record_approval(world.fresh().root(), approval, True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []


def test_each_pending_approval_names_its_own_call_for_a_plan() -> None:
    world = Recording()
    session, approvals = park(world, *SAME_ID)
    calls = pending_approval_calls(rows(session))
    assert [(c["id"], c["digest"]) for c in calls] == [
        (approvals[0], args_digest({"n": 1})),
        (approvals[1], args_digest({"n": 2})),
    ]


# -- the start path ---------------------------------------------------------------


def child_parked_on(world: World, *calls: tuple[str, str, dict[str, Any]]) -> list[str]:
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    say(world, *calls)
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", "go"
    )
    approvals = pending_approval_ids(rows(world.child("worker")))
    assert len(approvals) == len(calls)
    return approvals


def unanswerable(world: World) -> None:
    """Leave the child's parked approvals with nobody to answer them, so a
    runner start may go ahead and end them (as test_step_allowance_r6)."""
    child = world.child("worker")
    drop_child_field(child, "child_binding")
    rewrite_paused_row(child, lambda row: row.pop("harness_binding", None))


def start_again(world: World) -> None:
    unanswerable(world)
    world.fresh()
    world.queue.clear()
    world.text("Child answer.")
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("host", RunBudget(8)), "host", "New task."
    )


@pytest.mark.parametrize("calls", [SAME_ID, EMPTY_ID], ids=["same-id", "empty-id"])
@pytest.mark.parametrize("decided", [False, True], ids=["undecided", "decided"])
def test_a_start_ends_each_call_with_its_own_approval_id(
    calls: tuple[tuple[str, str, dict[str, Any]], ...], decided: bool
) -> None:
    world = Recording()
    approvals = child_parked_on(world, *calls)
    if decided:
        record_approval(world.fresh().child("worker"), approvals[0], False, "no")
        record_approval(world.fresh().child("worker"), approvals[1], True)

    start_again(world)

    stored = rows(world.fresh().child("worker"))
    marked = [row[ENDED_BY_START] for row in stored if ENDED_BY_START in row]
    assert marked == [{"approval_ids": approvals}]
    assert ended_approval_ids(stored) == set(approvals)
    assert world.ran_with == []
    for approval in approvals:
        with pytest.raises(HarnessError):
            record_approval(world.fresh().child("worker"), approval, True)


def test_a_start_ends_unique_calls_with_their_own_approval_ids() -> None:
    world = Recording()
    approvals = child_parked_on(world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))
    start_again(world)

    stored = rows(world.fresh().child("worker"))
    assert [row[ENDED_BY_START] for row in stored if ENDED_BY_START in row] == [
        {"approval_ids": approvals}
    ]
    assert world.ran_with == []
