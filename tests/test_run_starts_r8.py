"""G-387 round 8: the round-7 review findings, at the public seam.

1. A runner start is narrowed from what its gate READ: steps spent on the
   context after the read count against the child.
2. The child's records written by commit (b) (binding, depth, ceiling, refs,
   count, session mode) are reversible until the task row is recorded: a
   failure in between restores them, so a dead approval stays dead and no
   live hold is left behind.
3. The child's pending approvals are re-read under (b)'s lock, with the
   gate's answerability rule.
4. The runner hands the parent model only fixed, harness-written reasons:
   never a host's cancel reason, an exception's message or a forged code.
5. A start releases only the tree hold it took itself.
6. The answerability read fails closed on any error it does not mean.
7. A non-finite or negative time or cost limit is refused, or read as spent.
8. A run's failure never records an exception's text: class name, a fixed
   description and the harness code.
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from test_run_starts import child_records, direct, parked_child, stored, tree_records
from test_step_allowance import World
from test_step_allowance_r2 import child_requests
from test_step_allowance_r6 import dead_approval_child

from prism_harness import (
    AgentRuntime,
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    RunBudget,
    RunContext,
    RunFailed,
    RunStarted,
    Session,
    SubagentRunner,
    ToolRegistry,
    record_approval,
)
from prism_harness.plans.service import Plans
from prism_harness.run_starts import StartRefused
from prism_harness.runtime import current_turn_run, current_turn_run_id
from prism_harness.thread_rows import pending_approval_ids


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def is_child(session: Session, name: str = "worker") -> bool:
    return session.scope.endswith(f"::sub::{name}")


def child_state(world: World, name: str = "worker") -> dict[str, Any]:
    """The child's records a runner start writes: the durable ``:child``
    record (binding, depth, ceiling, refs, count with its holds) and the
    session mode, as stored."""
    child = world.fresh().child(name)
    store = child._durable  # a test standing in for a store reader
    ephemeral = child._ephemeral
    state = ephemeral.get(child._ephemeral_key()) or {}
    return {"child": store.get(f"{child.key()}:child"), "mode": state.get("mode")}


def live_holds(world: World, name: str = "worker") -> dict[str, Any]:
    record = child_state(world, name)["child"]
    if not isinstance(record, dict) or not isinstance(record.get("child_steps"), dict):
        return {}
    holds = record["child_steps"].get("holds")
    return dict(holds) if isinstance(holds, dict) else {}


# -- 1. narrowed from what the gate read --------------------------------------


@pytest.mark.parametrize("hook", ["child_live_hold", "start_child", "begin_run"])
def test_steps_spent_after_the_read_count_against_the_child(
    hook: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner context has 2 of 8 steps left when the gate reads it. A
    concurrent spender takes those 2 after the read (at the last fact read,
    inside the child write, or inside the run's begin): the child takes no
    model step, and the context never goes past its 8."""
    world = World()
    world.text("Child answer.")
    world.text("Child answer.")
    context = RunContext.root("host", RunBudget(8))
    context.ledger.record_steps(6)
    original: Callable[..., Any] = getattr(Session, hook)
    spent: list[int] = []

    def spend(self: Session, *args: Any, **kwargs: Any) -> Any:
        if is_child(self) and not spent:
            spent.append(1)
            if hook == "child_live_hold":
                found = original(self, *args, **kwargs)
                context.ledger.record_steps(2)
                return found
            context.ledger.record_steps(2)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Session, hook, spend)

    result = direct(world, world.root(), context, "Look.")

    assert spent
    assert child_requests(world) == 0
    assert context.ledger.steps == 8
    assert result.outcome == "exhausted"


# -- 2. commit (b) is reversible until the task row ---------------------------


def fail_tools_for_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ToolRegistry.resolve_detailed

    def failing(self: ToolRegistry, names: Any, session: Session | None = None) -> Any:
        if session is not None and is_child(session):
            raise OSError("SECRET tool provider down")
        return original(self, names, session)

    monkeypatch.setattr(ToolRegistry, "resolve_detailed", failing)


def fail_plan_context_for_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Plans.context

    def failing(self: Plans, session: Session, prompt: str, *args: Any) -> str:
        if is_child(session):
            raise OSError("SECRET plan store down")
        return original(self, session, prompt, *args)

    monkeypatch.setattr(Plans, "context", failing)


def fail_run_begin_for_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Session.begin_run

    def failing(self: Session, *args: Any, **kwargs: Any) -> Any:
        if is_child(self):
            raise OSError("SECRET store down")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Session, "begin_run", failing)


FAILURES: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "tool_registry": fail_tools_for_the_child,
    "plan_context": fail_plan_context_for_the_child,
    "run_begin": fail_run_begin_for_the_child,
}


def start_again(world: World, task: str = "Again.") -> Any:
    parent = world.fresh().root()
    return direct(world, parent, RunContext.root("host", RunBudget(8)), task)


@pytest.mark.parametrize("failure", list(FAILURES))
def test_a_failure_after_the_child_write_keeps_a_dead_approval_dead(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    approval = dead_approval_child(world)
    before = child_state(world)

    with pytest.MonkeyPatch.context() as patch:
        FAILURES[failure](patch)
        result = start_again(world)

    assert result.outcome == "failed"
    assert result.text == "The [worker] subagent did not run: OSError."
    assert child_state(world) == before  # byte-identical: binding, count, refs, mode
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().child("worker"), approval, True)
    assert caught.value.code == "child_binding_missing"
    assert live_holds(world) == {}
    # No live hold remains: the next delegation is not refused as busy.
    world.text("Child answer.")
    again = start_again(world)
    assert again.outcome == "completed"


@pytest.mark.parametrize("failure", list(FAILURES))
def test_a_failure_after_the_child_write_leaves_a_fresh_child_unbound(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh child whose start fails after (b): no binding, no count, no
    mode left behind, so a host turn on it never runs in the lead's mode."""
    world = World()
    before = child_state(world)
    assert before == {"child": None, "mode": None}

    with pytest.MonkeyPatch.context() as patch:
        FAILURES[failure](patch)
        result = start_again(world, "Look.")

    assert result.outcome == "failed"
    assert child_state(world) == before
    assert world.fresh().child("worker").child_binding() is None
    assert child_requests(world) == 0


def test_a_restore_never_clobbers_a_later_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare-and-restore: when another writer changed the child's record
    after (b), the restore leaves that record alone and only removes the
    failed start's own hold."""
    world = World()
    original = Session.begin_run

    def later_writer_then_fail(self: Session, *args: Any, **kwargs: Any) -> Any:
        if is_child(self):
            store = self._durable
            key = f"{self.key()}:child"
            current = store.get(key)
            assert isinstance(current, dict)
            store.put(key, {**current, "child_depth": 2})  # someone else wrote since
            self.using_mode("free")  # and the mode too
            raise OSError("store down")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Session, "begin_run", later_writer_then_fail)

    result = start_again(world, "Look.")

    assert result.outcome == "failed"
    record = child_state(world)["child"]
    assert isinstance(record, dict)
    assert record["child_depth"] == 2  # the later write stands
    assert child_state(world)["mode"] == "free"  # so does the later mode
    assert live_holds(world) == {}  # the failed start's hold is gone


# -- 3. pending approvals re-read under (b)'s lock -----------------------------


def test_an_answerable_approval_parked_after_the_read_refuses_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    world.text("Child answer.")
    assert start_again(world, "First.").outcome == "completed"
    original = Session.child_live_hold
    parked: list[dict[str, Any]] = []

    def host_turn_parks(self: Session, *args: Any) -> bool:
        found = original(self, *args)
        if is_child(self) and not parked:
            # A host turn on the child, right after the gate read it, parks an
            # approval a person can answer.
            response = world.runtime.send(world.child("worker"), "Host asks.")
            assert response.finish_reason == "awaiting_approval"
            parked.append(child_state(world))
        return found

    monkeypatch.setattr(Session, "child_live_hold", host_turn_parks)
    requests = child_requests(world)

    result = start_again(world)

    assert parked
    assert result.outcome == "denied"
    assert result.text == "The [worker] subagent did not run: it is waiting on a person's approval."
    assert child_requests(world) == requests + 1  # only the host turn's
    assert child_state(world) == parked[0]
    (approval,) = world.pending(world.child("worker"))
    record_approval(world.fresh().child("worker"), approval, True)  # still answerable
    assert tree_records(world) == {}


def test_a_pending_re_read_that_raises_refuses_the_start_writing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-read under (b)'s lock fails closed: framed failed, nothing
    written on the child, and the dead approval stays dead."""
    world = World()
    approval = dead_approval_child(world)
    world.fresh()
    calls: list[int] = []
    original_pending = pending_approval_ids
    original_refs = Session.child_refs

    def counting(rows: Any) -> Any:
        calls.append(1)
        return original_pending(rows)

    def refs(self: Session) -> Any:
        if is_child(self) and len(calls) >= 2:
            raise HarnessError("session_locked", "SECRET store text")
        return original_refs(self)

    monkeypatch.setattr("prism_harness.runtime.pending_approval_ids", counting)
    monkeypatch.setattr(Session, "child_refs", refs)
    before = stored(world)

    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Again.")

    assert len(calls) >= 2  # the re-read under the lock ran, and raised
    assert result.outcome == "failed"
    assert "SECRET" not in result.text
    assert stored(world) == before
    monkeypatch.undo()
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().child("worker"), approval, True)
    assert caught.value.code == "child_binding_missing"


# -- 4. no foreign text reaches the parent model ------------------------------


def run_root_with_a_cancel(
    world: World, monkeypatch: pytest.MonkeyPatch, hook: str
) -> tuple[RunContext, list[Any]]:
    host = RunContext.root("host_run", RunBudget(8))
    original: Callable[..., Any] = getattr(Session, hook)
    landed: list[int] = []

    def cancel_there(self: Session, *args: Any, **kwargs: Any) -> Any:
        if is_child(self) and not landed:
            landed.append(1)
            if hook == "begin_run":
                found = original(self, *args, **kwargs)
                host.ledger.cancel("SECRET operator note")
                return found
            host.ledger.cancel("SECRET operator note")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Session, hook, cancel_there)
    world.calls(("worker", {"task": "Look."}))
    world.text("Child answer.")
    world.text("Root done.")
    world.runtime.send(world.root(), "Go", context=host)
    return host, landed


@pytest.mark.parametrize("hook", ["begin_run", "start_child"])
def test_a_host_cancel_reason_never_reaches_the_parent_model(
    hook: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(i) the cancel lands after the child's run began, (ii) between the gate
    and the commit. Neither the parent's tool_result row nor the parent's next
    provider request carries the host's reason."""
    world = World()
    _host, landed = run_root_with_a_cancel(world, monkeypatch, hook)
    assert landed
    monkeypatch.undo()

    root = world.fresh().root()
    rows = [entry.message for entry in root.thread().messages()]
    results = [row for row in rows if row["type"] == "tool_result"]
    assert results
    assert "SECRET" not in json.dumps(results)
    assert "its run was cancelled" in json.dumps(results)

    world.text("Next answer.")
    world.runtime.send(root, "Next question")
    assert "SECRET" not in str(world.requests[-1].messages)
    assert "SECRET" not in json.dumps(rows)


class CodeThatRaises(HarnessError):
    @property
    def code(self) -> str:
        raise OSError("SECRET getter")

    @code.setter
    def code(self, _value: str) -> None:
        pass


class CodeThatChanges(HarnessError):
    reads = 0

    @property
    def code(self) -> str:
        CodeThatChanges.reads += 1
        return "run_not_permitted" if CodeThatChanges.reads == 1 else "SECRET-GETTER-TEXT"

    @code.setter
    def code(self, _value: str) -> None:
        pass


def forged_refusal(outcome: str) -> HarnessError:
    return StartRefused("cancelled", outcome)


RAISED: dict[str, tuple[Callable[[], BaseException], str, str]] = {
    "forged_outcome": (lambda: forged_refusal("SECRET outcome"), "failed", "its run was cancelled"),
    "forged_refusal": (
        lambda: StartRefused("SECRET refusal", "denied"),
        "denied",
        "it was refused",
    ),
    "spend_ceiling": (
        lambda: HarnessError("agent_spend_ceiling_exceeded", "SECRET ceiling text"),
        "exhausted",
        "its agent's spend ceiling is spent",
    ),
    "unpriced": (
        lambda: HarnessError("budget_unpriced", "SECRET unpriced text"),
        "exhausted",
        "its agent's spend cannot be metered",
    ),
    "throwing_code": (lambda: CodeThatRaises("x", "SECRET message"), "failed", "CodeThatRaises"),
    "foreign_code": (
        lambda: HarnessError("SECRET-CODE", "SECRET message"),
        "failed",
        "HarnessError",
    ),
}


@pytest.mark.parametrize("case", list(RAISED))
def test_an_exception_is_framed_with_fixed_text_only(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    build, outcome, reason = RAISED[case]
    original = AgentRuntime.send

    def raising(self: AgentRuntime, session: Session, prompt: str, **kwargs: Any) -> Any:
        if is_child(session):
            raise build()
        return original(self, session, prompt, **kwargs)

    monkeypatch.setattr(AgentRuntime, "send", raising)

    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Look.")

    assert result.outcome == outcome
    assert result.text == f"The [worker] subagent did not run: {reason}."
    assert "SECRET" not in result.text


def test_a_code_getter_that_changes_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World()
    CodeThatChanges.reads = 0
    original = AgentRuntime.send

    def raising(self: AgentRuntime, session: Session, prompt: str, **kwargs: Any) -> Any:
        if is_child(session):
            raise CodeThatChanges("x", "SECRET message")
        return original(self, session, prompt, **kwargs)

    monkeypatch.setattr(AgentRuntime, "send", raising)

    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Look.")

    assert result.outcome == "failed"
    assert "SECRET" not in result.text
    assert CodeThatChanges.reads == 1


# -- 5. a start releases only its own hold ------------------------------------


def test_a_failed_reentrant_start_keeps_the_first_starts_tree_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    events = HarnessEvents()
    world.runtime._events = events  # a host listening for run starts
    worker_key = world.child("worker").key()
    reentered: list[Any] = []

    def start_alpha(event: HarnessEvent) -> None:
        if not isinstance(event, RunStarted) or event.session_key != worker_key or reentered:
            return
        root = world.root()
        turn = current_turn_run(root.key())
        assert turn is not None
        subagent = world.runtime.mode_for(root).subagents["alpha"]
        reentered.append(
            SubagentRunner(world.harness, world.runtime).run(
                subagent, root, turn, current_turn_run_id(root.key()) or "", "Also."
            )
        )

    events.listen(start_alpha)
    original = Session.start_child

    def alpha_fails(self: Session, *args: Any, **kwargs: Any) -> Any:
        if is_child(self, "alpha"):
            raise OSError("store down")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Session, "start_child", alpha_fails)
    seen: list[dict[str, str]] = []

    def snapshot() -> None:
        seen.append(tree_records(world))

    world.calls(("worker", {"task": "Look."}))
    world.text("Child answer.")
    world.text("Root done.")
    queue_hook = world._client

    def client(request: Any) -> Any:
        if request.system_prompt == "Work." and not seen:
            snapshot()
        return queue_hook(request)

    world.runtime._client = client

    world.runtime.send(world.root(), "Go")

    assert reentered and reentered[0].outcome == "failed"
    assert seen
    (record,) = seen[0].values()
    holds = json.loads(record)["payload"]["holds"]
    assert holds  # the first start's hold is still there while its child runs


# -- 6. the answerability read fails closed ------------------------------------


def test_a_foreign_error_in_the_answerability_read_refuses_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    parked_child(world)
    (approval,) = world.pending(world.child("worker"))
    world.fresh()
    armed: list[bool] = []
    original_pending = pending_approval_ids
    original_depth = Session.child_depth

    def arming(rows: Any) -> Any:
        found = original_pending(rows)
        if found and not armed:
            armed.append(True)
        return found

    def depth(self: Session) -> int:
        if is_child(self) and armed and armed[-1]:
            armed.append(False)
            raise HarnessError("session_locked", "SECRET store text")
        return original_depth(self)

    monkeypatch.setattr("prism_harness.runtime.pending_approval_ids", arming)
    monkeypatch.setattr(Session, "child_depth", depth)
    before = stored(world)

    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Again.")

    assert False in armed  # the read did raise
    assert result.outcome == "failed"
    assert "SECRET" not in result.text
    assert stored(world) == before
    monkeypatch.undo()
    assert world.pending(world.fresh().child("worker")) == [approval]


# -- 7. non-finite limits --------------------------------------------------------


@pytest.mark.parametrize("field", ["max_seconds", "max_cost_usd"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1])
def test_a_budget_refuses_a_non_finite_or_negative_limit(field: str, value: float) -> None:
    limits: dict[str, Any] = {field: value}
    with pytest.raises(ValueError):
        RunBudget(8, **limits)
    with pytest.raises(ValueError):
        RunBudget.from_config({"max_steps": 8, field: value})


@pytest.mark.parametrize("field", ["max_seconds", "max_cost_usd"])
def test_a_forged_non_finite_limit_is_spent_for_a_host_turn(field: str) -> None:
    """A limit forged past the constructor (NaN compares false with
    everything) is read as spent by the loop's gate, never as no limit."""
    world = World()
    world.text("Answer.")
    limits = RunBudget(8)
    object.__setattr__(limits, field, math.nan)

    try:
        response = world.runtime.send(world.root(), "Go", context=RunContext.root("host", limits))
    except ValueError:
        pass  # refused when the run's budget is narrowed: failed closed
    else:
        assert response.finish_reason == "budget_exhausted"
    assert world.requests == []


@pytest.mark.parametrize("field", ["max_seconds", "max_cost_usd"])
def test_a_forged_non_finite_limit_refuses_a_runner_start(field: str) -> None:
    world = World()
    limits = RunBudget(8)
    object.__setattr__(limits, field, math.nan)

    result = direct(world, world.root(), RunContext.root("host", limits), "Look.")

    # Refused (spent) or failed closed (unreadable): never run.
    assert result.outcome in ("exhausted", "failed")
    assert "SECRET" not in result.text
    assert child_requests(world) == 0
    assert child_records(world) == {}


# -- 8. no exception text in a run row or RunFailed -----------------------------


def stored_texts(world: World) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path(world.directory).glob("*.json"))
    )


@pytest.mark.parametrize(
    "error",
    [
        lambda: RuntimeError("SECRET exception text"),
        lambda: HarnessError("run_not_permitted", "SECRET exception text"),
    ],
)
def test_a_failed_run_records_no_exception_text(error: Callable[[], BaseException]) -> None:
    world = World()
    events = HarnessEvents()
    failed: list[RunFailed] = []
    events.listen(lambda event: failed.append(event) if isinstance(event, RunFailed) else None)
    world.runtime._events = events

    def provider_fails() -> None:
        raise error()

    world.on_request = provider_fails

    with pytest.raises(Exception, match="SECRET"):
        world.runtime.send(world.root(), "Go")

    (event,) = failed
    assert "SECRET" not in event.failure
    assert event.failure.startswith(type(error()).__name__)
    run = world.fresh().root().run()
    assert run is not None and run["status"] == "failed"
    assert "SECRET" not in json.dumps(run)
    assert "SECRET" not in stored_texts(world)


def test_a_failed_child_run_records_no_exception_text_either() -> None:
    world = World()

    def provider_fails() -> None:
        raise RuntimeError("SECRET exception text")

    world.on_request = provider_fails

    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Look.")

    assert result.outcome == "failed"
    assert "SECRET" not in result.text
    assert "SECRET" not in stored_texts(world)
