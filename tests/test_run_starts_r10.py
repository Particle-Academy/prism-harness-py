"""G-387 round 10: the round-9 review findings, one rule each.

1. An unconfirmed (provisional) child record reads as the last CONFIRMED run
   tree and plan ledger, never as none: a failed start whose undo also failed
   never detaches the child from the tree and plan budget it was saved with.
2. A decision a person recorded that a resume could still act on blocks a
   runner start exactly like an answerable pending approval: a model never
   voids a person's decision by delegating.
3. The session mode a runner start writes is provisional like its other
   records: absent (the previous confirmed mode) until its task row exists.
4. A ``BaseException`` from host code still releases the start's holds, and
   still propagates.
5. Confirmation is a READ, allowed in the read-only tool-resolution view.
6. The decision events for calls a runner start ends are emitted only after
   the task row recording them is durable.
7. A framed failure, ``RunFailed`` and the run row name an identifier-shaped
   class only: never a forged name carrying a path.
8. A call a runner start ended reads as ended, never as approved.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_plan_budget import Rig
from test_run_starts import child_records, direct, stored
from test_run_starts_r8 import fail_tools_for_the_child, is_child, live_holds
from test_run_starts_r9 import (
    fail_the_undo,
    in_the_window,
    later,
    listening,
    raise_from_the_child_tools,
    refused_as_unbound,
    start,
)
from test_step_allowance import World, rewrite_paused_row
from test_step_allowance_r2 import child_requests, drop_child_field
from test_step_allowance_r6 import dead_approval_child

from prism_harness import (
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    Participant,
    RunBudget,
    RunContext,
    RunFailed,
    Session,
    SubagentRunner,
    ToolDenied,
    ToolRegistry,
    record_approval,
)
from prism_harness.plans import budget
from prism_harness.plans.service import Plans
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.thread import Thread


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def tool_delegated_child(world: World) -> dict[str, str]:
    """A child the root delegated to through the subagent tool: saved in the
    root's run tree. Returns that tree."""
    world.calls(("worker", {"task": "First."}))
    world.text("Child answer.")
    world.text("Root done.")
    world.runtime.send(world.root(), "Go")
    world.queue.clear()
    refs = world.fresh().child("worker").child_refs()
    assert refs is not None and refs["tree"] is not None
    return refs["tree"]


def failed_start_then_recovered(world: World) -> None:
    """A direct start fails after (b) and its undo fails too; once its hold
    has expired, a direct start commits."""
    with pytest.MonkeyPatch.context() as patch:
        fail_tools_for_the_child(patch)
        fail_the_undo(patch)
        assert "did not run: OSError" in start(world, "direct", answer=False)
    with pytest.MonkeyPatch.context() as patch:
        later(patch)
        assert "Child answer." in start(world, "direct", "New task.")


# -- 1. an unconfirmed record keeps the confirmed tree and plan ledger ----------


def test_a_failed_start_whose_undo_fails_keeps_the_childs_run_tree() -> None:
    """PA2 (tree): after a failed start whose undo also failed and a later
    host-driven start, a host turn on the child is still in the root's tree."""
    world = World()
    tree = tool_delegated_child(world)

    with pytest.MonkeyPatch.context() as patch:
        fail_tools_for_the_child(patch)
        fail_the_undo(patch)
        start(world, "direct", answer=False)
    child = world.fresh().child("worker")
    assert child.child_provisional()
    refs = child.child_refs()
    assert refs is not None and refs["tree"] == tree  # the confirmed tree, never none

    with pytest.MonkeyPatch.context() as patch:
        later(patch)
        assert "Child answer." in start(world, "direct", "New task.")
    refs = world.fresh().child("worker").child_refs()
    assert refs is not None and refs["tree"] == tree

    world.text("Hello back.")
    response = world.fresh().runtime.send(world.child("worker"), "Hello.")
    assert response.finish_reason == "stop"
    # A store reader: the run continued the root's tree, not one of its own.
    assert world.child("worker")._run_tree(response.run_id) == tree


def test_a_failed_start_whose_undo_fails_keeps_the_child_bounded_by_its_tree() -> None:
    """PA2 (bounded): the root's tree is spent; the child, re-delegated by
    the host after a failed start, is still bounded by it."""
    world = World()
    tree = tool_delegated_child(world)
    failed_start_then_recovered(world)
    root = world.root()
    root.plan_store().put(
        f"{tree['session']}:run-tree:{tree['run']}",
        {
            **budget.open_ledger(
                tree["run"], {"mode": "shared", "max_steps": 1, "max_cost_micros": None}
            ),
            "steps": 1,
        },
    )
    requests = child_requests(world)

    with pytest.raises(HarnessError) as caught:
        world.fresh().runtime.send(world.child("worker"), "Hello.")

    assert caught.value.code == "run_not_permitted"
    assert child_requests(world) == requests


def research_child(rig: Rig) -> Session:
    return rig.harness.for_(Participant("User", "ada")).session("chat::sub::research")


def research_start(rig: Rig, task: str) -> Any:
    subagent = rig.runtime.mode_for(rig.session).subagents["research"]
    return SubagentRunner(rig.harness, rig.runtime).run(
        subagent, rig.session, RunContext.root("host", RunBudget(8)), "host", task
    )


def test_a_failed_start_whose_undo_fails_keeps_the_childs_plan_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PA2 (plan ledger): a child started in a shared plan step, after a
    failed start whose undo failed and a later host-driven start, still
    spends from the same plan ledger."""
    rig = Rig({"budget": {"mode": "shared", "max_steps": 10}}, subagents=True)
    rig.active(1)
    rig.calls(("research", {"task": "Look."}))
    rig.text("Child answer.")
    rig.calls(("plan_complete_step", {"summary": "Asked."}))
    rig.text("Step done.")
    rig.advance()
    refs = research_child(rig).child_refs()
    assert refs is not None and refs["plan_ledger"] is not None
    ledger = refs["plan_ledger"]

    with pytest.MonkeyPatch.context() as patch:
        original = ToolRegistry.resolve_detailed

        def failing(self: ToolRegistry, names: Any, session: Session | None = None) -> Any:
            if session is not None and is_child(session, "research"):
                raise OSError("store down")
            return original(self, names, session)

        patch.setattr(ToolRegistry, "resolve_detailed", failing)
        fail_the_undo(patch)
        assert research_start(rig, "Again.").outcome == "failed"
    found = research_child(rig).child_refs()
    assert found is not None and found["plan_ledger"] == ledger

    original_now = Plans.now
    monkeypatch.setattr(Plans, "now", lambda self: original_now(self) + 301)
    rig.text("Child answer.")
    assert research_start(rig, "New task.").outcome == "completed"
    found = research_child(rig).child_refs()
    assert found is not None and found["plan_ledger"] == ledger

    spent = rig.budget()["steps"]
    rig.text("Answer.")
    rig.runtime.send(research_child(rig), "New question")
    assert rig.budget()["steps"] == spent + 1


def test_an_unconfirmed_record_in_its_window_reads_the_confirmed_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ever narrowing: in a start's window the child still reads as
    bound to nothing (refused), but never as detached from its tree."""
    world = World()
    tree = tool_delegated_child(world)
    found: dict[str, Any] = {}

    def read() -> None:
        child = world.child("worker")
        found["refs"] = child.child_refs()
        found["binding"] = child.child_binding()
        try:
            world.runtime.send(world.child("worker"), "Hello.")
            found["message"] = "ran"
        except HarnessError as error:
            found["message"] = error.code

    with pytest.MonkeyPatch.context() as patch:
        in_the_window(world, patch, read)
        start(world, "direct", "New task.")

    assert found["refs"] == {"tree": tree, "plan_ledger": None}
    assert found["binding"] is None
    assert found["message"] == "child_binding_missing"


# -- 2. a decided approval a resume could act on blocks a runner start ----------


@pytest.mark.parametrize("path", ["direct", "tool"])
def test_a_decided_approval_refuses_a_re_delegation_until_it_is_resumed(path: str) -> None:
    world = World()
    world.run_child_in_runner("worker")
    (approval,) = world.pending(world.child("worker"))
    record_approval(world.fresh().child("worker"), approval, True)
    before = child_records(world)
    requests = child_requests(world)

    text = start(world, path, "New task.", answer=False)

    assert "it is waiting on a person's approval" in text
    assert child_requests(world) == requests
    assert child_records(world) == before
    assert "echo" not in world.ran

    # The host resumes the child: the approved call runs, once.
    world.text("Child done.")
    world.fresh().runtime.send(world.child("worker"), "")
    assert world.ran.count("echo") == 1


@pytest.mark.parametrize("approved", [True, False])
def test_a_decision_recorded_after_the_gate_read_refuses_the_start_under_the_lock(
    approved: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b)'s locked re-check: a host turn parks an approval and a person
    decides it between the gate read and (b): refused, nothing written."""
    world = World()
    world.text("Child answer.")
    assert start(world, "direct", "First.").count("Child answer.") >= 1
    original = Session.child_live_hold
    parked: list[dict[str, str]] = []

    def host_turn_and_decision(self: Session, *args: Any) -> bool:
        found = original(self, *args)
        if is_child(self) and not parked:
            response = world.runtime.send(world.child("worker"), "Host asks.")
            assert response.finish_reason == "awaiting_approval"
            record_approval(world.child("worker"), response.pending_approvals[0].id, approved)
            parked.append(child_records(world))
        return found

    monkeypatch.setattr(Session, "child_live_hold", host_turn_and_decision)

    text = start(world, "direct", "Again.", answer=False)

    assert parked
    assert "it is waiting on a person's approval" in text
    assert child_records(world) == parked[0]
    assert "echo" not in world.ran


# -- 3. the mode a runner start writes is provisional --------------------------


def test_a_failed_start_whose_undo_fails_leaves_the_confirmed_mode() -> None:
    world = World()
    assert "Child answer." in start(world, "direct", "First.")
    world.child("worker").using_mode("free")  # the host's own, confirmed

    with pytest.MonkeyPatch.context() as patch:
        fail_tools_for_the_child(patch)
        fail_the_undo(patch)
        start(world, "direct", answer=False)

    assert world.fresh().child("worker").mode() == "free"

    with pytest.MonkeyPatch.context() as patch:
        later(patch)
        assert "Child answer." in start(world, "direct", "New task.")
    assert world.fresh().child("worker").mode() == "work"


def test_a_starts_mode_is_absent_in_its_window(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World()
    found: dict[str, Any] = {}

    def read() -> None:
        found["mode"] = world.child("worker").mode()

    with pytest.MonkeyPatch.context() as patch:
        in_the_window(world, patch, read)
        start(world, "direct", "New task.")

    assert found == {"mode": None}  # never started: no confirmed mode
    assert world.fresh().child("worker").mode() == "work"


def test_a_host_mode_written_after_a_failed_start_reads_as_written() -> None:
    world = World()
    with pytest.MonkeyPatch.context() as patch:
        fail_tools_for_the_child(patch)
        fail_the_undo(patch)
        start(world, "direct", answer=False)
    assert world.fresh().child("worker").mode() is None

    world.child("worker").using_mode("free")

    assert world.fresh().child("worker").mode() == "free"


# -- 4. a BaseException from host code still releases the start's holds --------


class Abort(BaseException):
    """Host code stopping the process: not an Exception."""


def test_a_base_exception_in_a_start_releases_its_holds_and_propagates() -> None:
    world = World()
    approval = dead_approval_child(world)
    world.fresh()

    with pytest.MonkeyPatch.context() as patch:
        raise_from_the_child_tools(patch, Abort)
        with pytest.raises(Abort):
            direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Again.")

    assert live_holds(world) == {}
    refused_as_unbound(lambda: record_approval(world.fresh().child("worker"), approval, True))
    # Not blocked as busy: the re-delegation runs at once.
    world.text("Child answer.")
    again = direct(world.fresh(), world.root(), RunContext.root("host", RunBudget(8)), "New task.")
    assert again.outcome == "completed"
    assert "echo" not in world.ran


# -- 5. confirmation is a read, allowed in the read-only resolution view --------


@pytest.mark.parametrize("path", ["direct", "tool"])
def test_a_tool_provider_reading_the_child_during_a_runner_start(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    world.run_child_in_runner("worker")
    seen: list[dict[str, Any]] = []
    writes: list[str] = []
    original = ToolRegistry.resolve_detailed

    def providing(self: ToolRegistry, names: Any, session: Session | None = None) -> Any:
        if session is not None and is_child(session):
            seen.append(
                {
                    "run": session.run(),
                    "binding": session.child_binding(),
                    "refs": session.child_refs(),
                    "depth": session.child_depth(),
                    "ceiling": session.child_ceiling(),
                    "mode": session.mode(),
                    "provisional": session.child_provisional(),
                }
            )
            try:
                session.using_mode("free")
                writes.append("wrote")
            except ValueError:
                writes.append("refused")
        return original(self, names, session)

    monkeypatch.setattr(ToolRegistry, "resolve_detailed", providing)
    record_approval(world.fresh().child("worker"), world.pending(world.child("worker"))[0], False)
    world.text("Denied noted.")
    world.fresh().runtime.send(world.child("worker"), "")  # resolved; nothing waits

    text = start(world, path, "Look.")

    assert "Child answer." in text
    assert seen and writes == ["refused"] * len(seen)
    # Inside the start's own run (its own Session's view), the start's own
    # records read as committed (G-387 round 11); outside readers keep the
    # provisional rule (the invariant matrix).
    assert seen[-1]["provisional"] is False
    assert seen[-1]["binding"] is not None
    assert seen[-1]["mode"] == "work"


# -- 6. end events only after the task row is durable ---------------------------


def test_a_failed_task_row_write_emits_no_tool_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World()
    approval = dead_approval_child(world)
    world.fresh()
    events = HarnessEvents()
    seen: list[HarnessEvent] = []
    events.listen(seen.append)
    world.runtime._events = events
    original = Thread.record

    def failing(self: Thread, messages: Any, run_id: str | None = None) -> Any:
        if "::sub::worker" in self._key and any(row.get("type") == "user" for row in messages):
            raise OSError("store down")
        return original(self, messages, run_id)

    monkeypatch.setattr(Thread, "record", failing)

    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "New task.")

    assert result.outcome == "failed"
    assert not any(isinstance(event, ToolDenied) for event in seen)
    assert world.pending(world.fresh().child("worker")) == [approval]


# -- 7. a framed failure names an identifier-shaped class only ------------------


def forged_names() -> dict[str, type[BaseException]]:
    pathy = type("Leaky", (OSError,), {})
    pathy.__name__ = "C:\\srv\\app\\hooks.py:12"
    spaced = type("Boom", (RuntimeError,), {})
    spaced.__name__ = "Boom\n/srv/app/hooks.py:3"
    dotted = type("Dotted", (ValueError,), {})
    dotted.__name__ = "app.hooks.Failure"
    return {"path": pathy, "spaced": spaced, "dotted": dotted}


LEAKS = ("@anonymous", ".py", ".ts", ".js", ".php", "/srv", "\\srv", "hooks")


def no_leak(text: str) -> None:
    for leak in LEAKS:
        assert leak not in text, leak


@pytest.mark.parametrize("kind", list(forged_names()))
@pytest.mark.parametrize("path", ["direct", "tool"])
def test_a_forged_class_name_never_reaches_the_model_or_a_run_row(
    kind: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    error = forged_names()[kind]
    events = HarnessEvents()
    failed: list[RunFailed] = []
    events.listen(lambda event: failed.append(event) if isinstance(event, RunFailed) else None)
    original_fresh = World.fresh

    def keep_listening(self: World) -> World:
        found = original_fresh(self)
        self.runtime._events = events
        return found

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(World, "fresh", keep_listening)
        raise_from_the_child_tools(patch, lambda: error("SECRET"))
        text = start(world, path, "Look.", answer=False)

    assert "did not run" in text
    no_leak(text)
    no_leak(str(list(stored(world).values())))
    assert failed
    for event in failed:
        no_leak(event.failure)


@pytest.mark.parametrize("kind", list(forged_names()))
def test_a_forged_class_name_never_reaches_a_root_run_row(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    failed = listening(world)
    error = forged_names()[kind]

    def failing(self: ToolRegistry, *_args: Any, **_kwargs: Any) -> Any:
        raise error("SECRET")

    monkeypatch.setattr(ToolRegistry, "resolve", failing)
    monkeypatch.setattr(ToolRegistry, "resolve_detailed", failing)

    with pytest.raises(BaseException):  # noqa: B017 - the host's own class propagates
        world.runtime.send(world.root(), "Go")

    assert len(failed) == 1
    no_leak(failed[0].failure)
    run = world.fresh().root().run()
    assert run is not None and run["status"] == "failed"
    no_leak(str(run))


# -- 8. a call a runner start ended reads as ended, never approved --------------


def test_a_call_a_start_ended_never_reads_as_approved() -> None:
    world = World()
    world.run_child_in_runner("worker")
    (approval,) = world.pending(world.child("worker"))
    record_approval(world.fresh().child("worker"), approval, True)
    # The binding is then lost: the decision can no longer be acted on, so a
    # re-delegation is let through and its task row ends the call.
    drop_child_field(world.child("worker"), "child_binding")
    rewrite_paused_row(world.child("worker"), lambda row: row.pop("harness_binding", None))

    assert "Child answer." in start(world, "direct", "New task.")

    rows = [entry.message for entry in world.fresh().child("worker").thread().messages()]
    assert "Not run" in str(rows)
    assert approval not in approval_decisions(rows)
    assert world.pending(world.child("worker")) == []
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().child("worker"), approval, True)
    assert caught.value.code == "approval_not_pending"
    assert "echo" not in world.ran


# -- run-14 I1: an expired parked run never becomes the attempted one ----------


def test_a_failed_start_with_no_prior_run_state_names_no_run() -> None:
    world = World()
    world.text("Child answer.")
    start(world, "direct", "First.")
    world.child("worker").forget()  # the ephemeral half expired
    assert world.fresh().child("worker").run() is None

    with pytest.MonkeyPatch.context() as patch:
        fail_tools_for_the_child(patch)
        fail_the_undo(patch)
        start(world, "direct", answer=False)

    assert world.fresh().child("worker").run() is None
