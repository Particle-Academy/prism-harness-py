"""G-387 round 2: what the final review found, at the public seam.

- A child's saved ceiling is the DELEGATION's lifetime budget: a durable count
  of the steps the child used since the runner last started it, claimed in one
  locked write at run begin, so concurrent and repeated turns on the child
  never total more than its ceiling.
- A linked child with no saved ceiling, or one that is not a whole number,
  fails closed.
- A budget that is not a finite whole number never removes the cap.
- A decision the resuming turn could not act on is refused before it is
  recorded, and a cancelled tree runs no approved call.
- The runner refuses an empty task, and a delegation into a child a person's
  approval is pending on.

Driven through ``send``, ``record_approval``, ``decide``,
``SubagentRunner.run`` and host ``RunContext``s, over the ``World`` of
test_step_allowance (fresh harness and runtime objects over one file store).
"""

from __future__ import annotations

import math
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_step_allowance import World, approve_and_resume

from prism_harness import (
    MAX_DEPTH,
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


def child_requests(world: World) -> int:
    return sum(1 for request in world.requests if request.system_prompt == "Work.")


def child_key(session: Session) -> tuple[Any, str]:
    return session._durable, f"{session.key()}:child"  # a test standing in for a store writer


def set_child_field(session: Session, name: str, value: object) -> None:
    store, key = child_key(session)
    stored = store.get(key)
    assert isinstance(stored, dict)
    store.put(key, {**stored, name: value})


def drop_child_field(session: Session, name: str) -> None:
    store, key = child_key(session)
    stored = store.get(key)
    assert isinstance(stored, dict)
    stored.pop(name, None)
    store.put(key, stored)


def thread_length(session: Session) -> int:
    return len(session.thread().messages())


# -- 1. the saved ceiling is the delegation's lifetime budget --------------


def test_two_concurrent_resumes_of_a_runner_started_child_never_pass_its_ceiling() -> None:
    world = World()
    world.run_child_in_runner("big")  # declared 4, one step taken
    record_approval(world.fresh().child("big"), world.pending(world.child("big"))[0], True)
    seen: dict[str, Any] = {}

    def second_worker() -> None:
        other = World.__new__(World)
        other.__dict__.update(world.__dict__)
        other.fresh()
        try:
            seen["second"] = other.runtime.send(other.child("big"), "").finish_reason
        except HarnessError as error:
            seen["second"] = error.code

    for q in range(10):
        world.calls(("lookup", {"q": q}))  # the child would keep going
    world.on_request = second_worker
    world.fresh().runtime.send(world.child("big"), "")

    assert "second" in seen
    assert child_requests(world) <= 4


def test_two_concurrent_resumes_of_a_child_in_a_root_turn_never_pass_its_ceiling() -> None:
    world = World(lead_steps=20)
    world.calls(("big", {"task": "B"}))  # root step 1
    world.calls(("echo", {"who": "big"}))  # big parks after 1
    world.text("Root done.")
    world.runtime.send(world.root(), "Go")
    record_approval(world.fresh().child("big"), world.pending(world.child("big"))[0], True)

    def second_worker() -> None:
        other = World.__new__(World)
        other.__dict__.update(world.__dict__)
        other.fresh()
        try:
            other.runtime.send(other.child("big"), "")
        except HarnessError:
            pass

    for q in range(10):
        world.calls(("lookup", {"q": q}))
    world.on_request = second_worker
    world.fresh().runtime.send(world.child("big"), "")

    assert child_requests(world) <= 4


def test_three_new_messages_to_a_child_declared_two_total_two_model_requests() -> None:
    world = World()
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]  # declared 2
    world.text("First answer.")
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", "go"
    )
    for question in ("Q1", "Q2", "Q3"):
        world.text(f"Answer to {question}.")
        try:
            world.fresh().runtime.send(world.child("worker"), question)
        except HarnessError as error:
            assert error.code == "run_not_permitted"

    assert child_requests(world) == 2


def test_a_hold_whose_run_never_settled_is_charged_in_full_after_the_ttl() -> None:
    world = World()
    world.run_child_in_runner("big")  # declared 4, one used
    child = world.child("big")
    store, key = child_key(child)
    stored = store.get(key)
    assert isinstance(stored, dict) and isinstance(stored.get("child_steps"), dict)
    # A run that crashed long ago, holding three steps, never settled.
    stored["child_steps"]["holds"]["crashed_run"] = {"steps": 3, "at": 0}
    store.put(key, stored)

    approve_and_resume(world, lambda: world.child("big"), 6)

    # 1 used + 3 charged: the ceiling of 4 is spent, so no model step.
    assert child_requests(world) == 1
    assert world.ran == ["echo"]


@pytest.mark.parametrize("garbage", ["1", -1, True, None, 1.5, {"steps": 0}])
def test_a_garbage_durable_count_fails_closed(garbage: object) -> None:
    world = World()
    world.run_child_in_runner("big")
    set_child_field(world.child("big"), "child_steps", {"steps": garbage, "holds": {}})

    approve_and_resume(world, lambda: world.child("big"), 6)

    assert child_requests(world) == 1
    assert world.ran == ["echo"]


def test_a_runner_start_resets_the_count_for_the_new_delegation() -> None:
    world = World()
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]  # declared 2
    for _ in range(2):
        world.text("Answer.")
        world.text("Answer.")
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", "one"
    )
    world.fresh().runtime.send(world.child("worker"), "Q1")  # spends the delegation
    assert child_requests(world) == 2

    SubagentRunner(world.fresh().harness, world.runtime).run(
        subagent, world.root(), RunContext.root("root2", RunBudget(8)), "root2", "two"
    )

    # A new delegation: its own ceiling again.
    assert child_requests(world) == 3


# -- 2. a linked child with no usable saved ceiling fails closed -----------


@pytest.mark.parametrize("garbage", ["2", -1, True, 2.0, None, math.nan])
def test_a_garbage_saved_ceiling_fails_closed(garbage: object) -> None:
    world = World()
    world.run_child_in_runner("worker")  # declared 2, one used
    set_child_field(world.child("worker"), "child_ceiling", garbage)

    approve_and_resume(world, world.child, 6)

    assert child_requests(world) == 1
    assert world.ran == ["echo"]  # the approved call still ran


def test_a_linked_child_with_no_saved_ceiling_fails_closed() -> None:
    world = World()
    world.run_child_in_runner("worker")
    drop_child_field(world.child("worker"), "child_ceiling")

    approve_and_resume(world, world.child, 6)

    assert child_requests(world) == 1
    assert world.ran == ["echo"]


def test_a_new_message_to_a_linked_child_with_a_garbage_ceiling_is_refused() -> None:
    world = World()
    world.text("Answer.")
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", "go"
    )
    set_child_field(world.child("worker"), "child_ceiling", "2")
    rows = thread_length(world.child("worker"))

    with pytest.raises(HarnessError) as caught:
        world.fresh().runtime.send(world.child("worker"), "Another question")

    assert caught.value.code == "run_not_permitted"
    assert thread_length(world.child("worker")) == rows


def test_a_session_that_was_never_a_child_uses_its_mode() -> None:
    world = World(lead_steps=3)
    world.runtime.send(world.root(), "Go")
    approve_and_resume(world, world.root, 6)
    assert len(world.requests) == 3


# -- 3. a budget that is not a finite whole number never removes the cap ---


@pytest.mark.parametrize("steps", [math.nan, math.inf, -math.inf, -1, 2.0, 2.5, True, None, "8"])
def test_a_run_budget_refuses_a_max_steps_that_is_not_a_whole_number(steps: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RunBudget(steps)  # type: ignore[arg-type]


@pytest.mark.parametrize("steps", [math.nan, math.inf, "8", 2.0])
def test_a_subagent_declaration_that_is_not_a_whole_number_is_refused(steps: object) -> None:
    from prism_harness.subagents import subagent_from_config

    with pytest.raises((TypeError, ValueError)):
        subagent_from_config("worker", {"mode": "work", "max_steps": steps})


@pytest.mark.parametrize("steps", [math.nan, math.inf])
def test_a_host_context_whose_budget_was_forced_non_finite_takes_no_model_step(
    steps: float,
) -> None:
    world = World()
    parked = world.runtime.send(world.root(), "Go")
    record_approval(world.fresh().root(), parked.pending_approvals[0].id, True)
    budget = RunBudget(10)
    object.__setattr__(budget, "max_steps", steps)  # past the constructor's check
    context = RunContext.root("host", budget)

    response = world.fresh().runtime.send(world.root(), "", context=context)

    assert response.finish_reason == "budget_exhausted"
    assert world.ran == ["echo"]
    assert len(world.requests) == 1


@pytest.mark.parametrize("ceiling", [math.nan, math.inf, 2.0])
def test_a_max_steps_ceiling_that_is_not_a_whole_number_is_refused(ceiling: float) -> None:
    with pytest.raises(ValueError):
        World(config={"max_steps_ceiling": ceiling})  # type: ignore[arg-type]


# -- 4. a decision the resume could not act on is refused before recording -


@pytest.mark.parametrize("path", ["record_approval", "decide"])
def test_a_decision_on_a_linked_child_with_no_binding_is_refused_recording_nothing(
    path: str,
) -> None:
    world = World()
    world.run_child_in_runner("worker")
    child = world.child("worker")
    drop_child_field(child, "child_binding")

    def strip(row: dict[str, Any]) -> None:
        row.pop("harness_binding", None)

    from test_step_allowance import rewrite_paused_row

    rewrite_paused_row(child, strip)
    approval = world.pending(child)[0]
    rows = thread_length(child)

    with pytest.raises(HarnessError) as caught:
        if path == "decide":
            world.fresh().runtime.decide(world.child("worker"), approval, True)
        else:
            record_approval(world.fresh().child("worker"), approval, True)

    assert caught.value.code == "child_binding_missing"
    assert thread_length(world.child("worker")) == rows
    assert world.pending(world.child("worker")) == [approval]


def test_a_decision_on_a_child_too_deep_to_run_is_refused_recording_nothing() -> None:
    world = World()
    world.run_child_in_runner("worker")
    child = world.child("worker")
    set_child_field(child, "child_depth", MAX_DEPTH)
    approval = world.pending(child)[0]
    rows = thread_length(child)

    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().child("worker"), approval, True)

    assert caught.value.code == "run_not_permitted"
    assert thread_length(world.child("worker")) == rows


def test_a_cancelled_tree_runs_no_approved_call_on_resume() -> None:
    world = World()
    parked = world.runtime.send(world.root(), "Go")
    record_approval(world.fresh().root(), parked.pending_approvals[0].id, True)
    context = RunContext.root("host", RunBudget(10))
    context.ledger.cancel("stopped by host")

    response = world.fresh().runtime.send(world.root(), "", context=context)

    assert response.stopped_because == "stopped by host"
    assert world.ran == []
    assert len(world.requests) == 1


# -- 5. the runner refuses an empty task and a child a person waits on -----


@pytest.mark.parametrize("task", ["", "   ", "\n\t"])
def test_the_runner_refuses_an_empty_task_writing_nothing(task: str) -> None:
    world = World()
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]

    result = SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", task
    )

    assert result.outcome == "denied"
    assert world.child("worker").find_thread() is None
    assert world.child("worker").child_binding() is None
    assert world.requests == []


def test_the_runner_refuses_to_delegate_into_a_child_a_person_waits_on() -> None:
    world = World()
    world.run_child_in_runner("worker")
    child = world.child("worker")
    waiting = world.pending(child)
    rows = thread_length(child)
    parent = world.fresh().root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]

    result = SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root2", RunBudget(8)), "root2", "Again."
    )

    assert result.outcome == "denied"
    assert world.pending(world.child("worker")) == waiting
    assert thread_length(world.child("worker")) == rows
    assert len(world.requests) == 1


def test_a_delegation_through_the_tool_into_a_waiting_child_is_refused() -> None:
    world = World()
    world.calls(("worker", {"task": "One."}))  # root step 1
    world.calls(("echo", {"who": "worker"}))  # the child parks
    world.calls(("worker", {"task": "Again, and dismiss it."}))  # root step 2
    world.text("Root done.")
    world.runtime.send(world.root(), "Go")
    child = world.child("worker")

    assert len(world.pending(child)) == 1
    assert child_requests(world) == 1
