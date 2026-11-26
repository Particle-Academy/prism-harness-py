"""G-387 round 7: ONE start gate for a runner start, at the public seam.

A runner start (``SubagentRunner.run``, and the subagent tool that calls it)
reads every fact a start depends on first, asks ``RunStarts.gate`` once, and
only then commits, in a fixed order: the tree hold, then the child's records in
one locked write, then the run itself. A refused start writes nothing: not the
child's records, not a run state or run history entry, not the parent's tree
record. Once committed, a later stop only stops model steps.

Every gate code is driven through the subagent TOOL inside a parent turn and
through the runner directly, asserting the framed outcome, no child model
request, the child's stored records byte-identical, no run state or history
entry for the refused start, and the parent unchanged.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from test_step_allowance import World
from test_step_allowance_r2 import child_requests, set_child_field

from prism_harness import (
    AgentRuntime,
    HarnessError,
    RunBudget,
    RunContext,
    RunLedger,
    Session,
    SubagentRunner,
    record_approval,
)
from prism_harness.plans import budget
from prism_harness.plans.service import Plans
from prism_harness.run_trees import RunTrees
from prism_harness.runtime import current_turn_run, current_turn_run_id
from prism_harness.subagents import MAX_DEPTH


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


# -- what is stored ---------------------------------------------------------


def stored(world: World) -> dict[str, str]:
    """Every stored record, by its key, as the exact text on disk."""
    found: dict[str, str] = {}
    for path in sorted(Path(world.directory).glob("*.json")):
        text = path.read_text(encoding="utf-8")
        found[str(json.loads(text)["key"])] = text
    return found


def child_records(world: World) -> dict[str, str]:
    """The children's records (binding, count, refs, thread, run state, run
    history), byte for byte."""
    return {key: text for key, text in stored(world).items() if "::sub::" in key}


def tree_records(world: World) -> dict[str, str]:
    """The durable run-tree records of the parent's runs."""
    return {key: text for key, text in stored(world).items() if ":run-tree:" in key}


def direct(world: World, parent: Session, context: RunContext, task: object) -> Any:
    """The runner driven directly, outside a turn."""
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    return SubagentRunner(world.harness, world.runtime).run(
        subagent,
        parent,
        context,
        "root2",
        task,  # type: ignore[arg-type]
    )


def delegate_in_a_turn(world: World, args: dict[str, Any], name: str = "worker") -> str:
    """The root delegates through the subagent TOOL, then answers; the text of
    the provider request after the delegation."""
    world.calls((name, args))
    world.text("Root done.")
    response = world.runtime.send(world.root(), "Go")
    assert response.finish_reason == "stop"
    return str(world.requests[-1].messages)


def plant_live_hold(world: World) -> None:
    """A turn on the child holds steps from its delegation right now."""
    child = world.child("worker")
    set_child_field(
        child,
        "child_steps",
        {
            "steps": 0,
            "holds": {"run_live": {"steps": 1, "at": world.runtime.plans.now()}},
            "charged": {},
        },
    )


def aged(context: RunContext) -> RunContext:
    context.ledger._started_at -= 100  # its 60 s ran out
    return context


def cancelled(context: RunContext) -> RunContext:
    context.ledger.cancel("SECRET-OPERATOR-REASON")
    return context


def spent_steps(context: RunContext) -> RunContext:
    context.ledger.record_steps(context.budget.max_steps)
    return context


def spent_cost(context: RunContext) -> RunContext:
    context.ledger.record_cost(2.0)
    return context


def unpriced(context: RunContext) -> RunContext:
    context.ledger.record_cost(None)
    return context


#: Each stop of the gate a context can carry: how to build that context, the
#: outcome the runner frames, and the harness-written reason.
CONTEXT_STOPS: dict[str, tuple[Callable[[], RunContext], str, str]] = {
    "too_deep": (
        lambda: RunContext(ledger=RunLedger("host"), budget=RunBudget(8), depth=MAX_DEPTH - 1),
        "denied",
        f"subagent nesting reached the maximum depth of {MAX_DEPTH}",
    ),
    "cancelled": (
        lambda: cancelled(RunContext.root("host", RunBudget(8))),
        "cancelled",
        "its run was cancelled",
    ),
    "time": (
        lambda: aged(RunContext.root("host", RunBudget(8, max_seconds=60))),
        "exhausted",
        "time budget exhausted",
    ),
    "steps": (
        lambda: spent_steps(RunContext.root("host", RunBudget(3))),
        "exhausted",
        "step budget exhausted",
    ),
    "cost": (
        lambda: spent_cost(RunContext.root("host", RunBudget(8, max_cost_usd=1.0))),
        "exhausted",
        "cost budget exhausted",
    ),
    "cost_unenforceable": (
        lambda: unpriced(RunContext.root("host", RunBudget(8, max_cost_usd=1.0))),
        "exhausted",
        "cost budget cannot be enforced",
    ),
}


def parked_child(world: World) -> None:
    """A child the runner started, parked on an approval a person can answer."""
    world.run_child_in_runner("worker")


# -- 1. every gate code, through the runner directly -------------------------


@pytest.mark.parametrize("code", list(CONTEXT_STOPS))
def test_a_context_stop_refuses_a_direct_runner_start_writing_nothing(code: str) -> None:
    world = World()
    parent = world.root()
    build, outcome, reason = CONTEXT_STOPS[code]
    context = build()
    before = stored(world)

    result = direct(world, parent, context, "Look.")

    assert result.outcome == outcome
    assert result.text == f"The [worker] subagent did not run: {reason}."
    assert "SECRET" not in result.text
    assert child_requests(world) == 0
    assert stored(world) == before  # nothing at all: child, run state, history, parent


@pytest.mark.parametrize(
    ("task", "reason"),
    [
        (7, "the task is not text"),
        (None, "the task is not text"),
        ("\u200b\u3000 \t", "the task is empty"),
    ],
)
def test_a_task_that_is_not_text_or_blank_refuses_a_direct_start_writing_nothing(
    task: object, reason: str
) -> None:
    world = World()
    parent = world.root()
    before = stored(world)

    result = direct(world, parent, RunContext.root("host", RunBudget(8)), task)

    assert result.outcome == "denied"
    assert result.text == f"The [worker] subagent did not run: {reason}."
    assert child_requests(world) == 0
    assert stored(world) == before


def test_a_busy_child_refuses_a_direct_start_writing_nothing() -> None:
    world = World()
    parked_child(world)
    approval = world.pending(world.child("worker"))[0]
    record_approval(world.child("worker"), approval, False)
    plant_live_hold(world)
    parent = world.fresh().root()
    before = stored(world)

    result = direct(world, parent, RunContext.root("host", RunBudget(8)), "Again.")

    assert result.outcome == "denied"
    assert result.text == "The [worker] subagent did not run: a turn on it is still running."
    assert child_requests(world) == 1
    assert stored(world) == before


def test_a_child_waiting_on_an_answerable_approval_refuses_a_direct_start() -> None:
    world = World()
    parked_child(world)
    parent = world.fresh().root()
    before = stored(world)

    result = direct(world, parent, RunContext.root("host", RunBudget(8)), "Again.")

    assert result.outcome == "denied"
    assert result.text == "The [worker] subagent did not run: it is waiting on a person's approval."
    assert child_requests(world) == 1
    assert stored(world) == before


def test_a_spent_child_agent_ceiling_refuses_the_start_as_cost() -> None:
    """The child agent's own spend ceiling is a cost stop of the gate."""
    world = World()
    parent = world.root()
    context = RunContext.root("host", RunBudget(8))
    before = stored(world)

    original = AgentRuntime._labelled

    def ceilinged(self: AgentRuntime, session: Session, run: RunContext) -> RunContext:
        labelled = original(self, session, run)
        if "::sub::" not in session.scope:
            return labelled
        from dataclasses import replace

        labelled.ledger.carry_agent_spend("researcher", 0.05, 0)
        return replace(labelled, agent="researcher", spend_ceiling=0.04)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AgentRuntime, "_labelled", ceilinged)
        result = direct(world, parent, context, "Look.")

    assert result.outcome == "exhausted"
    assert result.text == "The [worker] subagent did not run: cost budget exhausted."
    assert child_requests(world) == 0
    assert stored(world) == before


# -- 2. every gate code, through the subagent TOOL inside a parent turn -------


@pytest.mark.parametrize("code", list(CONTEXT_STOPS))
def test_a_host_context_stop_refuses_the_subagent_tool_inside_a_turn(code: str) -> None:
    world = World()
    build, _outcome, reason = CONTEXT_STOPS[code]
    world.tool_context = build
    world.fresh()

    following = delegate_in_a_turn(world, {"task": "Look."})

    assert f"The [worker] subagent did not run: {reason}." in following
    assert "SECRET" not in following
    assert child_requests(world) == 0
    assert child_records(world) == {}
    assert tree_records(world) == {}


@pytest.mark.parametrize(
    ("task", "reason"),
    [(7, "the task is not text"), ("\u200b\u3000 \t", "the task is empty")],
)
def test_a_task_that_is_not_text_or_blank_refuses_the_tool_inside_a_turn(
    task: object, reason: str
) -> None:
    world = World()

    following = delegate_in_a_turn(world, {"task": task})

    assert f"The [worker] subagent did not run: {reason}." in following
    assert child_requests(world) == 0
    assert child_records(world) == {}
    assert tree_records(world) == {}


def test_a_busy_child_refuses_the_tool_inside_a_turn() -> None:
    world = World()
    parked_child(world)
    approval = world.pending(world.child("worker"))[0]
    record_approval(world.child("worker"), approval, False)
    plant_live_hold(world)
    world.fresh()
    before = child_records(world)

    following = delegate_in_a_turn(world, {"task": "Again."})

    assert "The [worker] subagent did not run: a turn on it is still running." in following
    assert child_requests(world) == 1
    assert child_records(world) == before
    assert tree_records(world) == {}


def test_a_waiting_child_refuses_the_tool_inside_a_turn() -> None:
    world = World()
    parked_child(world)
    world.fresh()
    before = child_records(world)

    following = delegate_in_a_turn(world, {"task": "Again."})

    assert "it is waiting on a person's approval" in following
    assert child_requests(world) == 1
    assert child_records(world) == before
    assert tree_records(world) == {}


def plant_spent_tree(world: World) -> None:
    """The parent turn's tree record, already spent (a sibling settled, or
    another worker wrote it)."""
    root = world.root()
    run = root.run()
    assert run is not None
    root.plan_store().put(
        f"{root.key()}:run-tree:{run['id']}",
        {
            **budget.open_ledger(
                run["id"], {"mode": "shared", "max_steps": 1, "max_cost_micros": None}
            ),
            "steps": 1,
        },
    )


def test_a_spent_tree_refuses_the_tool_inside_a_turn_before_anything_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    original = AgentRuntime.send

    def spent_then_send(self: AgentRuntime, session: Session, prompt: str, **kwargs: Any) -> Any:
        if kwargs.get("context") is not None:
            plant_spent_tree(world)
        return original(self, session, prompt, **kwargs)

    monkeypatch.setattr(AgentRuntime, "send", spent_then_send)

    following = delegate_in_a_turn(world, {"task": "Look."})

    assert "The [worker] subagent did not run: its run tree has no steps left." in following
    assert child_requests(world) == 0
    assert child_records(world) == {}
    (record,) = tree_records(world).values()
    assert json.loads(record)["payload"]["holds"] == {}


class DirectWorld(World):
    """The lead's ``alpha`` tool calls the runner DIRECTLY with the turn's own
    context, as a host's own tool might."""

    def _runtime(self) -> AgentRuntime:
        runtime = super()._runtime()
        world = self

        class Direct:
            name = "alpha"

            def __init__(self, session: Session) -> None:
                self._session = session

            def handle(self, args: dict[str, Any]) -> Any:
                key = self._session.key()
                turn = current_turn_run(key)
                assert turn is not None
                subagent = world.runtime.mode_for(self._session).subagents["alpha"]
                result = SubagentRunner(world.harness, world.runtime).run(
                    subagent,
                    self._session,
                    turn,
                    current_turn_run_id(key) or "",
                    str(args.get("task")),
                )
                world.results.append(result)
                return result.to_tool_result()

        runtime._tools.register_factory("alpha", Direct)
        return runtime

    def __init__(self) -> None:
        self.results: list[Any] = []
        super().__init__()


def test_a_spent_tree_refuses_a_direct_runner_start_inside_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = DirectWorld()
    original = AgentRuntime.send

    def spent_then_send(self: AgentRuntime, session: Session, prompt: str, **kwargs: Any) -> Any:
        if kwargs.get("context") is not None:
            plant_spent_tree(world)
        return original(self, session, prompt, **kwargs)

    monkeypatch.setattr(AgentRuntime, "send", spent_then_send)

    delegate_in_a_turn(world, {"task": "Look."}, "alpha")

    (result,) = world.results
    assert result.outcome == "exhausted"
    assert result.text == "The [alpha] subagent did not run: its run tree has no steps left."
    assert child_requests(world) == 0
    assert child_records(world) == {}
    (record,) = tree_records(world).values()
    assert json.loads(record)["payload"]["holds"] == {}


# -- 3. the commit: a race since the read refuses it, releasing the hold -------


def test_a_tree_spent_between_the_read_and_the_hold_refuses_with_nothing_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    original = RunTrees.child_started
    planted: list[dict[str, Any]] = []

    def spent_first(self: RunTrees, session: Session, member: Any, *args: Any) -> Any:
        if not planted:
            plant_spent_tree(world)
            planted.append({})
        return original(self, session, member, *args)

    monkeypatch.setattr(RunTrees, "child_started", spent_first)

    following = delegate_in_a_turn(world, {"task": "Look."})

    assert "The [worker] subagent did not run: its run tree has no steps left." in following
    assert planted
    assert child_requests(world) == 0
    assert child_records(world) == {}  # no run state, no run history, no records
    (record,) = tree_records(world).values()
    assert json.loads(record)["payload"]["holds"] == {}


def test_a_resume_claiming_between_the_read_and_the_child_write_refuses_and_releases_the_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    parked_child(world)
    approval = world.pending(world.child("worker"))[0]
    record_approval(world.child("worker"), approval, False)
    world.fresh()
    original = Session.child_live_hold
    claimed: list[str] = []

    def then_claimed(self: Session, *args: Any) -> bool:
        found = original(self, *args)
        if not claimed and "::sub::" in self.scope:
            claimed.append(self.key())
            plant_live_hold(world)  # a resume on the child claimed right after the read
        return found

    monkeypatch.setattr(Session, "child_live_hold", then_claimed)
    before = child_records(world)

    following = delegate_in_a_turn(world, {"task": "Again."})

    assert "The [worker] subagent did not run: a turn on it is still running." in following
    assert claimed
    assert child_requests(world) == 1
    after = child_records(world)
    child_key = f"{world.child('worker').key()}:child"
    # Only the resume's own claim changed the child; the start wrote nothing.
    assert {key: text for key, text in after.items() if key != child_key} == {
        key: text for key, text in before.items() if key != child_key
    }
    saved = json.loads(after[child_key])["payload"]
    assert list(saved["child_steps"]["holds"]) == ["run_live"]
    assert saved["child_binding"] == json.loads(before[child_key])["payload"]["child_binding"]
    # The tree hold the start took is released: no record is left holding it.
    assert tree_records(world) == {}


def test_a_failed_child_write_releases_the_tree_hold_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child's records fail to save: the tree hold is released, and a
    second delegation in the same turn takes it again, exactly once."""
    world = World()
    original_with_lock = type(world.root()._durable).with_lock
    failed: list[str] = []

    def failing(self: Any, key: str, callback: Callable[[], Any], *args: Any) -> Any:
        if key.endswith("::sub::worker:child") and not failed:
            failed.append(key)
            raise OSError("store down")
        return original_with_lock(self, key, callback, *args)

    monkeypatch.setattr(type(world.root()._durable), "with_lock", failing)
    world.calls(("worker", {"task": "Look."}))  # fails at the child write
    world.calls(("alpha", {"task": "Look."}))  # starts
    world.text("Alpha answer.")
    world.text("Root done.")

    response = world.runtime.send(world.root(), "Go")

    assert response.finish_reason == "stop"
    assert failed
    assert any(
        "The [worker] subagent did not run: OSError." in str(request.messages)
        for request in world.requests
        if request.system_prompt == "Lead."
    )
    worker = {key: text for key, text in child_records(world).items() if "::sub::worker" in key}
    assert worker == {}
    (record,) = tree_records(world).values()
    payload = json.loads(record)["payload"]
    assert payload["holds"] == {}
    assert payload["steps"] == 4  # three root steps and alpha's one, counted once


# -- 4. once committed, a later stop only stops model steps ------------------


def test_a_committed_start_followed_by_a_cancel_records_its_task_and_takes_no_model_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    parent = world.root()
    context = RunContext.root("host", RunBudget(8))
    original = Session.begin_run

    def then_cancelled(self: Session, *args: Any, **kwargs: Any) -> Any:
        found = original(self, *args, **kwargs)
        if "::sub::" in self.scope:
            context.ledger.cancel("late cancel")
        return found

    monkeypatch.setattr(Session, "begin_run", then_cancelled)

    result = direct(world, parent, context, "Look.")

    assert result.outcome == "exhausted"
    assert child_requests(world) == 0
    rows = [entry.message for entry in world.child("worker").thread().messages()]
    assert {"type": "user", "content": "Look."} in rows


def test_a_cancel_after_the_read_never_commits_over_an_answerable_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child a person can still answer is refused at the gate, so no commit
    (and no task row) can ever end that approval, even when a cancel lands
    after the read."""
    world = World()
    parked_child(world)
    approval = world.pending(world.child("worker"))[0]
    parent = world.fresh().root()
    context = RunContext.root("host", RunBudget(8))
    original = Session.begin_run

    def then_cancelled(self: Session, *args: Any, **kwargs: Any) -> Any:
        context.ledger.cancel("late cancel")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Session, "begin_run", then_cancelled)
    before = stored(world)

    result = direct(world, parent, context, "Again.")

    assert result.outcome == "denied"
    assert stored(world) == before
    assert world.pending(world.child("worker")) == [approval]
    record_approval(world.fresh().child("worker"), approval, True)  # still answerable


# -- 5. new channels: every read fails closed --------------------------------


class RaisingCancelLedger(RunLedger):
    @property
    def cancelled(self) -> bool:
        raise OSError("SECRET host ledger")


def test_a_host_ledger_read_that_raises_fails_closed_writing_nothing() -> None:
    world = World()
    world.tool_context = lambda: RunContext(
        ledger=RaisingCancelLedger("host"), budget=RunBudget(100)
    )
    world.fresh()

    following = delegate_in_a_turn(world, {"task": "Look."})

    assert "The [worker] subagent did not run: OSError." in following
    assert "SECRET" not in following
    assert child_requests(world) == 0
    assert child_records(world) == {}
    assert tree_records(world) == {}


def test_a_child_read_that_raises_fails_closed_writing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    parent = world.root()
    before = stored(world)

    def raising(self: Session, *args: Any) -> bool:
        raise OSError("SECRET store text")

    monkeypatch.setattr(Session, "child_live_hold", raising)

    result = direct(world, parent, RunContext.root("host", RunBudget(8)), "Look.")

    assert result.outcome == "failed"
    assert result.text == "The [worker] subagent did not run: OSError."
    assert child_requests(world) == 0
    assert stored(world) == before


@pytest.mark.parametrize("field", ["max_cost_usd", "max_seconds"])
def test_a_forged_non_finite_limit_on_the_host_fails_closed(field: str) -> None:
    """A NaN cost or time cap compares false with everything: read as a cap
    that cannot be met, never as no cap."""
    world = World()

    def forged() -> RunContext:
        limits = RunBudget(100)
        object.__setattr__(limits, field, float("nan"))
        return RunContext.root("host", limits)

    world.tool_context = forged
    world.fresh()

    following = delegate_in_a_turn(world, {"task": "Look."})

    assert "The [worker] subagent did not run:" in following
    assert child_requests(world) == 0
    assert child_records(world) == {}


def test_a_plan_ledger_on_a_runner_start_fails_closed_writing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner start spends from its parent's run, never from a plan ledger
    of its own: a plan route that names one for it (the plan's own step run
    under a runner context) is refused before anything is written."""
    world = World()
    parent = world.root()
    original = Plans._ledger_for

    def routed(self: Plans, session: Session, *args: Any, **kwargs: Any) -> Any:
        if "::sub::" in session.scope:
            return {"session": session.key(), "plan": "pl_1", "refused": True}
        return original(self, session, *args, **kwargs)

    monkeypatch.setattr(Plans, "_ledger_for", routed)
    before = stored(world)

    result = direct(world, parent, RunContext.root("host", RunBudget(8)), "Look.")

    assert result.outcome == "failed"
    assert child_requests(world) == 0
    assert stored(world) == before


def test_a_live_start_still_delegates_through_both_seams() -> None:
    world = World()
    world.text("Child answer.")
    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Look.")
    assert result.outcome == "completed"

    world = World()
    world.calls(("worker", {"task": "Look."}))
    world.text("Child answer.")
    world.text("Root done.")
    assert world.runtime.send(world.root(), "Go").finish_reason == "stop"
    assert child_requests(world) == 1


def test_a_refused_start_is_no_harness_error_to_the_host() -> None:
    """The runner frames every refusal; a host calling it never sees a raise."""
    world = World()
    result = direct(world, world.root(), cancelled(RunContext.root("h", RunBudget(8))), "Look.")
    assert result.outcome == "cancelled"
    assert not isinstance(result, HarnessError)
