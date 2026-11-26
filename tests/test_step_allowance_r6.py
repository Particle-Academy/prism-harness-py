"""G-387 round 6, at the public seam.

- Every ledger-only stop (steps, cost, time, cancel) a runner start reaches
  after the runner's pre-check refuses it before the child's records are
  written.
- The refusal sits immediately before the runner start's first durable write
  on the child. Once past it, the start is committed: a later cancel or
  time-out only stops model steps, and the task is recorded. Never "refused but
  the records rewritten".
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from test_step_allowance import World, rewrite_paused_row
from test_step_allowance_r2 import child_requests, drop_child_field

from prism_harness import (
    AgentRuntime,
    HarnessError,
    LlmResponse,
    LlmToolCall,
    RunBudget,
    RunContext,
    RunLedger,
    Session,
    SubagentRunner,
    record_approval,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def authority(session: Session) -> dict[str, Any]:
    """The child's saved authority and its thread, as stored."""
    store = session._durable  # a test standing in for a store reader
    return {suffix: store.get(f"{session.key()}{suffix}") for suffix in (":child", ":thread")}


def dead_approval_child(world: World) -> str:
    """A child parked on an approval nobody can answer any more."""
    world.run_child_in_runner("worker")
    child = world.child("worker")
    drop_child_field(child, "child_binding")
    rewrite_paused_row(child, lambda row: row.pop("harness_binding", None))
    approval = world.pending(child)[0]
    with pytest.raises(HarnessError):
        record_approval(world.fresh().child("worker"), approval, True)
    return approval


def start_again(world: World) -> Any:
    parent = world.fresh().root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    return SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root2", RunBudget(8)), "root2", "Again."
    )


def stop(context: RunContext, how: str) -> None:
    if how == "cancelled":
        context.ledger.cancel("stopped by host")
    elif how == "steps":
        context.ledger.record_steps(100)
    elif how == "time":
        object.__setattr__(context, "budget", RunBudget(8, max_seconds=1))
        context.ledger._started_at -= 100
    else:
        object.__setattr__(context, "budget", RunBudget(8, max_cost_usd=1.0))
        context.ledger.record_cost(None)


HOWS = ["cancelled", "steps", "time", "unpriced"]


# -- 1. every ledger-only stop before send refuses with nothing written -----


@pytest.mark.parametrize("how", HOWS)
def test_any_ledger_stop_before_send_refuses_the_start_writing_nothing(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    approval = dead_approval_child(world)
    before = authority(world.child("worker"))
    original = AgentRuntime.send

    def stopped_between(self: AgentRuntime, session: Session, prompt: str, **kwargs: Any) -> Any:
        if kwargs.get("context") is not None:
            stop(kwargs["context"], how)
        return original(self, session, prompt, **kwargs)

    monkeypatch.setattr(AgentRuntime, "send", stopped_between)
    result = start_again(world)

    assert result.outcome == ("cancelled" if how == "cancelled" else "exhausted")
    assert authority(world.child("worker")) == before
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().child("worker"), approval, True)
    assert caught.value.code == "child_binding_missing"
    assert child_requests(world) == 1


# -- 2. after the gate the start is committed; before it, nothing written ---


def hook_after(
    monkeypatch: pytest.MonkeyPatch, method: str, how: str, contexts: list[RunContext]
) -> None:
    original: Callable[..., Any] = getattr(Session, method)

    def hooked(self: Session, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if contexts and "::sub::" in self.scope:
            stop(contexts.pop(), how)
        return result

    monkeypatch.setattr(Session, method, hooked)


def capture_context(monkeypatch: pytest.MonkeyPatch) -> list[RunContext]:
    contexts: list[RunContext] = []
    original = AgentRuntime.send

    def capturing(self: AgentRuntime, session: Session, prompt: str, **kwargs: Any) -> Any:
        if kwargs.get("context") is not None:
            contexts.append(kwargs["context"])
        return original(self, session, prompt, **kwargs)

    monkeypatch.setattr(AgentRuntime, "send", capturing)
    return contexts


@pytest.mark.parametrize("how", ["cancelled", "time"])
def test_a_stop_during_begin_run_refuses_at_the_gate_with_records_unchanged(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    approval = dead_approval_child(world)
    before = authority(world.child("worker"))
    hook_after(monkeypatch, "begin_run", how, capture_context(monkeypatch))

    result = start_again(world)

    assert result.outcome == ("cancelled" if how == "cancelled" else "exhausted")
    assert authority(world.child("worker")) == before
    with pytest.raises(HarnessError):
        record_approval(world.fresh().child("worker"), approval, True)
    assert child_requests(world) == 1


@pytest.mark.parametrize("how", ["cancelled", "time"])
def test_a_stop_after_the_gate_commits_the_start_and_takes_no_model_step(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    dead_approval_child(world)
    hook_after(monkeypatch, "remember_child_binding", how, capture_context(monkeypatch))

    result = start_again(world)

    # Committed: the task is recorded as a runner start records it, the dead
    # approval is ended, and the loop's gate takes no model step.
    child = world.child("worker")
    rows = [entry.message for entry in child.thread().messages()]
    assert {"type": "user", "content": "Again."} in rows
    assert world.pending(child) == []
    assert child_requests(world) == 1
    assert result.outcome == "exhausted"


# -- 3. the unpriced host control, on a metered turn ------------------------


def test_a_host_context_with_an_unenforceable_cost_cap_stops_the_tool_in_a_metered_turn() -> None:
    world = World()

    def unpriced() -> RunContext:
        context = RunContext(ledger=RunLedger("host"), budget=RunBudget(100, max_cost_usd=1.0))
        context.ledger.record_cost(None)  # a call on the host's context reported no cost
        return context

    world.tool_context = unpriced
    world.fresh()
    # The turn's own calls report their cost, so only the host's state can stop it.
    world.queue.append(
        LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall("w1", "worker", {"task": "Look."})],
            cost_usd=0.01,
        )
    )
    world.queue.append(LlmResponse(text="Child answer.", finish_reason="stop", cost_usd=0.01))
    world.queue.append(LlmResponse(text="Root done.", finish_reason="stop", cost_usd=0.01))

    assert world.runtime.send(world.root(), "Go").finish_reason == "stop"

    assert child_requests(world) == 0
    assert "cost budget cannot be enforced" in str(world.requests[-1].messages)
