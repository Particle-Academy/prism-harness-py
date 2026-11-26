"""G-387 round 3: the second final review's findings, at the public seam.

- Everything the subagent runner checks before or around a child runs inside
  its framing, and a framed failure names the exception's class, never its
  text.
- A child's durable step count is read strictly: missing or malformed fails
  closed, and only a runner start writes a fresh one.
- A legacy subagent child (its scope is a child's, nothing saved by the runner)
  fails closed: proved on a store RELEASED v0.5.3 wrote.
- Settling replaces the run's own charge, live or already expired.
- The runner refuses a re-delegation while a turn on the child holds steps,
  and blocks on a pending approval only when a person could still answer it.
- A blank task is defined by one explicit codepoint set.
"""

from __future__ import annotations

import math
import shutil
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_step_allowance import World, approve_and_resume, rewrite_paused_row
from test_step_allowance_r2 import child_key, child_requests, drop_child_field, set_child_field

from prism_harness import (
    FileSessionStore,
    HarnessError,
    PrismHarness,
    RunBudget,
    RunContext,
    Session,
    SubagentRunner,
    record_approval,
)

FIXTURE = Path(__file__).parent / "fixtures" / "legacy-v0.5.3-child"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


SECRET = "SECRET-STORE-TEXT /var/lib/store/credentials"


class ArmedStore(FileSessionStore):
    """A file store that fails reading one key, with text the model must
    never see, once armed."""

    armed: str | None = None

    def get(self, key: str) -> dict[str, Any] | None:
        if ArmedStore.armed is not None and key.endswith(ArmedStore.armed):
            raise OSError(SECRET)
        return super().get(key)


class ArmedWorld(World):
    def _harness(self) -> PrismHarness:
        return PrismHarness(
            drivers={"files": lambda: ArmedStore(self.directory)},
            stores={"ephemeral": "files", "durable": "files"},
        )


def runner(world: World) -> tuple[SubagentRunner, Session]:
    world.fresh()
    return SubagentRunner(world.harness, world.runtime), world.root()


def delegate(world: World, name: str, task: object, budget: int = 8) -> Any:
    run, parent = runner(world)
    subagent = world.runtime.mode_for(parent).subagents[name]
    return run.run(subagent, parent, RunContext.root("rootX", RunBudget(budget)), "rootX", task)  # type: ignore[arg-type]


def steps_record(session: Session) -> Any:
    store, key = child_key(session)
    stored = store.get(key)
    return None if not isinstance(stored, dict) else stored.get("child_steps")


# -- 1. the runner's framing -----------------------------------------------


@pytest.mark.parametrize("failing", ["::sub::worker:thread", "::sub::worker:child"])
def test_a_store_failure_around_a_child_is_framed_without_its_text(failing: str) -> None:
    world = ArmedWorld()
    world.calls(("worker", {"task": "Look."}))  # root step 1
    world.text("Root done.")  # root step 2
    ArmedStore.armed = failing
    try:
        response = world.runtime.send(world.root(), "Go")
    finally:
        ArmedStore.armed = None

    assert response.finish_reason == "stop"
    following = str(world.requests[-1].messages)
    assert "SECRET" not in following
    assert "credentials" not in following
    assert "OSError" in following


@pytest.mark.parametrize("task", [None, 3, 1.5, ["go"]])
def test_a_task_that_is_not_a_string_is_denied(task: object) -> None:
    world = World()
    result = delegate(world, "worker", task)
    assert result.outcome == "denied"
    assert world.child("worker").find_thread() is None


@pytest.mark.parametrize("steps", [math.nan, math.inf])
def test_a_forged_host_budget_is_framed_not_raised(steps: float) -> None:
    world = World()
    run, parent = runner(world)
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    budget = RunBudget(8)
    object.__setattr__(budget, "max_steps", steps)  # past the constructor's check

    result = run.run(subagent, parent, RunContext.root("host", budget), "host", "go")

    # Framed whatever it is: never raised, never past the declaration.
    assert result.outcome in ("failed", "denied", "exhausted", "completed")
    assert child_requests(world) <= 2


# -- 2. the durable count is read strictly ---------------------------------


def test_a_missing_count_fails_closed_on_resume_and_on_a_new_message() -> None:
    world = World()
    world.run_child_in_runner("big")  # declared 4, one used
    drop_child_field(world.child("big"), "child_steps")

    approve_and_resume(world, lambda: world.child("big"), 6)
    assert child_requests(world) == 1
    assert world.ran == ["echo"]

    with pytest.raises(HarnessError) as caught:
        world.fresh().runtime.send(world.child("big"), "A new question")
    assert caught.value.code == "run_not_permitted"


@pytest.mark.parametrize(
    "record",
    [
        {"steps": 1, "holds": {"r": {"steps": 0, "at": -1}}, "charged": {}},
        {"steps": 1, "holds": {"r": {"steps": 0, "at": 1.5}}, "charged": {}},
        {"steps": 1, "holds": {"r": {"steps": 0, "at": "0"}}, "charged": {}},
        {"steps": 1, "holds": {"r": {"steps": 0}}, "charged": {}},
        {"steps": 1, "charged": {}},
        {"steps": 1, "holds": [], "charged": {}},
        {"steps": 1, "holds": {}},
        {"steps": 1, "holds": {}, "charged": {"r": -1}},
        {"holds": {}, "charged": {}},
    ],
)
def test_a_malformed_count_fails_closed(record: dict[str, Any]) -> None:
    world = World()
    world.run_child_in_runner("big")
    set_child_field(world.child("big"), "child_steps", record)

    approve_and_resume(world, lambda: world.child("big"), 6)

    assert child_requests(world) == 1
    assert world.ran == ["echo"]


def test_a_garbage_count_is_never_repaired_when_a_run_settles() -> None:
    world = World()
    world.run_child_in_runner("big")
    record_approval(world.fresh().child("big"), world.pending(world.child("big"))[0], True)
    garbage = {"steps": "x", "holds": {}, "charged": {}}

    def corrupt() -> None:
        set_child_field(world.child("big"), "child_steps", garbage)

    world.calls(("lookup", {"q": 1}))
    world.on_request = corrupt
    world.fresh().runtime.send(world.child("big"), "")

    assert steps_record(world.child("big")) == garbage


# -- 3. a legacy child fails closed, on the released shape -----------------


def released_world() -> World:
    world = World()
    for file in FIXTURE.glob("*.json"):
        shutil.copy(file, world.directory)
    return world.fresh()


def test_a_child_parked_by_released_v0_5_3_takes_no_model_step_on_main() -> None:
    world = released_world()
    child = world.child("worker")
    waiting = world.pending(child)
    assert len(waiting) == 1
    assert child.child_ceiling_state() == ("absent", None)

    approve_and_resume(world, lambda: world.child("worker"), 6)

    # Declared 2 and one used under v0.5.3: main fails closed, acts on the
    # approval and takes no model step.
    assert world.requests == []
    assert world.ran == ["echo"]


def test_a_new_message_to_a_released_v0_5_3_child_is_refused_before_recording() -> None:
    world = released_world()
    rows = len(world.child("worker").thread().messages())

    with pytest.raises(HarnessError) as caught:
        world.runtime.send(world.child("worker"), "Another question")

    assert caught.value.code == "run_not_permitted"
    assert len(world.child("worker").thread().messages()) == rows


# -- 4. settling replaces the run's own charge, live or expired ------------


def test_a_turn_past_the_ttl_that_took_one_of_four_leaves_three_for_its_resume() -> None:
    world = World()
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["big"]  # declared 4
    refused: list[str] = []

    def slow_call() -> None:
        # The runner's first model call takes 400 s (the lock TTL is 300): its
        # hold is dated 400 s back, and meanwhile a turn on the child charges
        # the expired hold in full.
        record = steps_record(world.child("big"))
        for hold in record["holds"].values():
            hold["at"] -= 400
        set_child_field(world.child("big"), "child_steps", record)
        other = World.__new__(World)
        other.__dict__.update(world.__dict__)
        other.fresh()
        try:
            other.runtime.send(other.child("big"), "Meanwhile")
        except HarnessError as error:
            refused.append(error.code)

    world.on_request = slow_call
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", "go"
    )
    assert refused == ["run_not_permitted"]
    record = steps_record(world.child("big"))
    assert record["steps"] == 1 and record["holds"] == {}  # counted once

    for q in range(6):
        world.calls(("lookup", {"q": q}))
    approve_and_resume(world, lambda: world.child("big"), 1)

    assert child_requests(world) == 4


# -- 5. no re-delegation while a turn on the child holds steps -------------


def test_the_runner_refuses_to_delegate_into_a_child_a_live_turn_holds() -> None:
    world = World()
    world.run_child_in_runner("big")
    record_approval(world.fresh().child("big"), world.pending(world.child("big"))[0], True)
    seen: dict[str, Any] = {}

    def world_clone() -> World:
        other = World.__new__(World)
        other.__dict__.update(world.__dict__)
        return other

    def redelegate() -> None:
        before = steps_record(world.child("big"))
        result = delegate(world_clone(), "big", "Again.")
        seen["outcome"] = result.outcome
        seen["unchanged"] = steps_record(world.child("big")) == before

    for q in range(3):
        world.calls(("lookup", {"q": q}))
    world.on_request = redelegate
    world.fresh().runtime.send(world.child("big"), "")

    assert seen == {"outcome": "denied", "unchanged": True}
    assert child_requests(world) <= 4


def test_a_claim_for_a_run_that_already_holds_fails_closed() -> None:
    world = World()
    world.run_child_in_runner("big")
    child = world.child("big")

    first = child.claim_child_steps("run_same", 1, 300, False, lambda counted: 1)
    seen: list[int | None] = []

    def decide(counted: int | None) -> int:
        seen.append(counted)
        return 1

    second = child.claim_child_steps("run_same", 1, 300, False, decide)

    assert first is not None
    assert second is None
    assert seen == [None]


# -- 6. one codepoint set for a blank task --------------------------------

BLANK = (
    [chr(c) for c in range(0x09, 0x0E)]
    + [" ", "\u0085", " ", " "]
    + [chr(c) for c in range(0x2000, 0x200E)]
    + [" ", " ", " ", " ", "⁠", "　", "﻿"]
)


@pytest.mark.parametrize("codepoint", BLANK, ids=[f"U+{ord(c):04X}" for c in BLANK])
def test_a_task_of_one_blank_codepoint_is_denied(codepoint: str) -> None:
    world = World()
    assert delegate(world, "worker", codepoint).outcome == "denied"
    assert world.child("worker").find_thread() is None


def test_a_mixed_run_of_blank_codepoints_is_denied() -> None:
    world = World()
    assert delegate(world, "worker", "".join(BLANK) * 2).outcome == "denied"


@pytest.mark.parametrize("task", ["‎", "\u001c", "᠎", "a", " a "])
def test_a_task_with_any_other_character_delegates(task: str) -> None:
    world = World()
    world.text("Done.")
    assert delegate(world, "worker", task).outcome == "completed"


# -- 7. a pending approval blocks only when a person could answer it -------


def test_a_child_stuck_on_an_unanswerable_approval_can_be_started_again() -> None:
    world = World()
    world.run_child_in_runner("worker")
    child = world.child("worker")
    drop_child_field(child, "child_binding")
    rewrite_paused_row(child, lambda row: row.pop("harness_binding", None))
    with pytest.raises(HarnessError):
        record_approval(world.fresh().child("worker"), world.pending(child)[0], True)

    world.text("Started again.")
    result = delegate(world, "worker", "Again.")

    assert result.outcome == "completed"
    assert world.pending(world.child("worker")) == []
