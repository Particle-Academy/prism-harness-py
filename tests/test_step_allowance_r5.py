"""G-387 round 5, at the public seam.

- A runner start refused because its run was cancelled, or ran out of time,
  between the runner's pre-check and the child's ``send`` writes nothing
  durable on the child, so a dead approval on it stays refused.
- A host's cancel (or exhaustion) of the context it built a subagent tool with
  reaches the tool inside a turn: framed, and no child model step.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_step_allowance import World, rewrite_paused_row
from test_step_allowance_r2 import child_requests, drop_child_field

from prism_harness import (
    AgentRuntime,
    HarnessError,
    RunBudget,
    RunContext,
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


def durable_records(session: Session) -> dict[str, Any]:
    """The child's durable records, as stored."""
    store = session._durable  # a test standing in for a store reader
    return {
        suffix: store.get(f"{session.key()}{suffix}")
        for suffix in (":child", ":thread", ":run-trees", ":durable")
    }


# -- 1. a refused runner start writes nothing durable -----------------------


@pytest.mark.parametrize("how", ["cancelled", "out of time"])
def test_a_runner_start_stopped_before_send_writes_nothing_durable(
    how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    world.run_child_in_runner("worker")
    child = world.child("worker")
    # Its approval is dead: no binding is saved for it any more.
    drop_child_field(child, "child_binding")
    rewrite_paused_row(child, lambda row: row.pop("harness_binding", None))
    approval = world.pending(child)[0]
    with pytest.raises(HarnessError):
        record_approval(world.fresh().child("worker"), approval, True)
    before = durable_records(world.child("worker"))

    original = AgentRuntime.send

    def stopped_between(
        self: AgentRuntime, session: Session, prompt: str, *args: Any, **kwargs: Any
    ) -> Any:
        context = kwargs.get("context")
        if context is not None:
            # The run is stopped after the runner's pre-check, before send.
            if how == "cancelled":
                context.ledger.cancel("stopped by host")
            else:
                object.__setattr__(context, "budget", RunBudget(8, max_seconds=1))
                context.ledger._started_at -= 100
        return original(self, session, prompt, *args, **kwargs)

    monkeypatch.setattr(AgentRuntime, "send", stopped_between)
    parent = world.fresh().root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    result = SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root2", RunBudget(8)), "root2", "Again."
    )

    assert result.outcome in ("cancelled", "exhausted")
    assert durable_records(world.child("worker")) == before
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().child("worker"), approval, True)
    assert caught.value.code == "child_binding_missing"
    assert child_requests(world) == 1


# -- 2. a host cancel reaches the subagent tool inside a turn ---------------


def delegate_in_a_turn(world: World) -> str:
    world.calls(("worker", {"task": "Look."}))
    world.text("Root done.")
    assert world.runtime.send(world.root(), "Go").finish_reason == "stop"
    return str(world.requests[-1].messages)


def test_a_host_cancel_reaches_the_subagent_tool_inside_a_turn() -> None:
    world = World()

    def cancelled() -> RunContext:
        context = RunContext.root("host", RunBudget(100))
        context.ledger.cancel("host stopped delegations")
        return context

    world.tool_context = cancelled
    world.fresh()

    following = delegate_in_a_turn(world)

    assert child_requests(world) == 0
    assert "did not run: host stopped delegations" in following


# The unenforceable-cost-cap control is in test_step_allowance_r6.py, on a
# metered turn, where the turn's own ledger cannot trip the cap by itself.


def test_a_live_host_context_still_delegates_inside_a_turn() -> None:
    world = World()
    world.calls(("worker", {"task": "Look."}))  # the root delegates
    world.text("Child answer.")  # the child
    world.text("Root done.")  # the root
    assert world.runtime.send(world.root(), "Go").finish_reason == "stop"
    assert child_requests(world) == 1
