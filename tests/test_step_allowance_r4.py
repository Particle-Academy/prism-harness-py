"""G-387 round 4: the third final review's findings, at the public seam.

- No exception text reaches the parent's model, by any route around a
  delegation: the subagent tool's own narrowing, the runner's checks, and
  everything inside the child's ``send`` (store writes, the child's client).
- The refusal of a re-delegation while a turn on the child holds steps is
  part of the reset's own locked write, and a refused start saves nothing.
- A resume under a context whose time is spent acts on no recorded approval.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator
from typing import Any

import pytest
from test_step_allowance import World, approve_and_resume
from test_step_allowance_r2 import child_key, child_requests, set_child_field

from prism_harness import (
    FileSessionStore,
    LlmRequest,
    LlmResponse,
    PrismHarness,
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


SECRET = "SECRET-TEXT postgres://user:pw@db/prism"


class PutFailingStore(FileSessionStore):
    """A file store whose write of one key fails, with text the model must
    never see, once armed."""

    armed: str | None = None

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        if PutFailingStore.armed is not None and key.endswith(PutFailingStore.armed):
            raise OSError(SECRET)
        super().put(key, payload, ttl_seconds)


class PutFailingWorld(World):
    def _harness(self) -> PrismHarness:
        return PrismHarness(
            drivers={"files": lambda: PutFailingStore(self.directory)},
            stores={"ephemeral": "files", "durable": "files"},
        )


class ChildClientFailingWorld(World):
    def _client(self, request: LlmRequest) -> LlmResponse:
        if request.system_prompt == "Work.":
            self.requests.append(request)
            raise RuntimeError(f"PROVIDER-BODY api_key=sk-live-XYZ {SECRET}")
        return super()._client(request)


def delegate_in_a_turn(world: World) -> str:
    """The root delegates to worker, then answers; the text of the next
    provider request after the delegation."""
    world.calls(("worker", {"task": "Look."}))
    world.text("Root done.")
    response = world.runtime.send(world.root(), "Go")
    assert response.finish_reason == "stop"
    return str(world.requests[-1].messages)


def assert_framed(following: str, name: str) -> None:
    assert "SECRET" not in following
    assert "sk-live" not in following
    assert "pw@db" not in following
    assert f"did not run: {name}" in following


# -- 1. no exception text by any route --------------------------------------


@pytest.mark.parametrize("failing", ["::sub::worker:child", "::sub::worker:thread"])
def test_a_failing_store_write_on_a_child_key_is_framed_without_its_text(failing: str) -> None:
    world = PutFailingWorld()
    PutFailingStore.armed = failing
    try:
        following = delegate_in_a_turn(world)
    finally:
        PutFailingStore.armed = None
    assert_framed(following, "OSError")


def test_a_child_client_raising_is_framed_without_its_text() -> None:
    world = ChildClientFailingWorld()
    following = delegate_in_a_turn(world)
    assert_framed(following, "RuntimeError")


class SecretLedger(RunLedger):
    @property
    def steps(self) -> int:
        raise OSError(SECRET)


def test_a_host_context_whose_ledger_throws_is_framed_without_its_text() -> None:
    world = World()
    world.tool_context = lambda: RunContext(ledger=SecretLedger("host"), budget=RunBudget(100))
    world.fresh()
    following = delegate_in_a_turn(world)
    assert_framed(following, "OSError")
    assert child_requests(world) == 0


def test_a_host_context_with_a_forged_fractional_max_steps_is_framed() -> None:
    world = World()

    def forged() -> RunContext:
        budget = RunBudget(100)
        object.__setattr__(budget, "max_steps", 1.5)  # past the constructor's check
        return RunContext.root("host", budget)

    world.tool_context = forged
    world.fresh()
    following = delegate_in_a_turn(world)
    assert "did not run: TypeError" in following
    assert "The tool failed" not in following
    assert child_requests(world) == 0


# -- 2. the live-hold refusal is part of the reset's locked write ----------


def test_a_resume_claiming_between_the_check_and_the_reset_refuses_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    world.run_child_in_runner("big")  # declared 4, one used
    record_approval(world.fresh().child("big"), world.pending(world.child("big"))[0], True)
    child = world.child("big")
    store, key = child_key(child)
    before = store.get(key)
    assert isinstance(before, dict)
    # A host resume claimed three steps just after the runner's own early
    # check read the count (forced: that check sees no live hold).
    record = dict(before["child_steps"])
    record["holds"] = {"resume_run": {"steps": 3, "at": int(time.time())}}
    set_child_field(child, "child_steps", record)
    claimed = store.get(key)
    monkeypatch.setattr(Session, "child_live_hold", lambda *_args, **_kwargs: False)

    parent = world.fresh().root()
    subagent = world.runtime.mode_for(parent).subagents["big"]
    world.text("New delegation.")
    result = SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root2", RunBudget(8)), "root2", "Again."
    )

    # Refused, and the refused start wrote nothing on the child: the running
    # turn's hold, the ceiling and the binding are as they were.
    assert result.outcome == "denied"
    assert store.get(key) == claimed
    assert child_requests(world) == 1


def test_a_ceiling_with_no_count_record_fails_closed() -> None:
    world = World()
    world.run_child_in_runner("big")
    child = world.child("big")
    store, key = child_key(child)
    stored = store.get(key)
    assert isinstance(stored, dict)
    stored.pop("child_steps")
    store.put(key, stored)

    approve_and_resume(world, lambda: world.child("big"), 6)

    assert child_requests(world) == 1
    assert world.ran == ["echo"]


# -- 4. spent time means no acting -----------------------------------------


def spent_time_context() -> RunContext:
    return RunContext(
        ledger=RunLedger("host", started_at=time.monotonic() - 100),
        budget=RunBudget(10, max_seconds=5),
    )


@pytest.mark.parametrize("approved", [True, False])
def test_a_resume_whose_time_is_spent_acts_on_no_recorded_decision(approved: bool) -> None:
    world = World()
    parked = world.runtime.send(world.root(), "Go")
    record_approval(world.fresh().root(), parked.pending_approvals[0].id, approved)
    rows = len(world.root().thread().messages())

    response = world.fresh().runtime.send(world.root(), "", context=spent_time_context())

    assert response.finish_reason == "budget_exhausted"
    assert world.ran == []
    assert len(world.root().thread().messages()) == rows  # nothing acted on, nothing written
    assert len(world.requests) == 1

    # The decision stays recorded: a later resume that can act on it does, once.
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran == (["echo"] if approved else [])


def test_a_new_message_under_a_spent_time_context_records_nothing() -> None:
    world = World()
    parked = world.runtime.send(world.root(), "Go")
    record_approval(world.fresh().root(), parked.pending_approvals[0].id, True)
    rows = len(world.root().thread().messages())

    response = world.fresh().runtime.send(world.root(), "Another", context=spent_time_context())

    # Ended on the spent time like a cancelled context: no decision acted on,
    # the new message not recorded, no model step.
    assert response.finish_reason == "budget_exhausted"
    assert world.ran == []
    assert len(world.requests) == 1
    assert len(world.root().thread().messages()) == rows
