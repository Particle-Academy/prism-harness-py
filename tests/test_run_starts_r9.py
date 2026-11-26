"""G-387 round 9: a runner start's writes are PROVISIONAL until its task row.

1. Everything commit (b) writes on the child (binding, depth, ceiling, count,
   refs, mode) carries the start's run id as ``provisional``, and every reader
   treats it as absent until that run's task row (a user row recorded under its
   id) is in the child's thread. Confirmation is read, never written, so a
   start that fails after (b) is harmless even when its undo cannot be written.
2. A runner start acts on NO recorded decision before its task row: the task
   row ends every pending call on the child, recorded as not run.
4. A failure handler cleans up first and reads an exception's attributes only
   guarded.
7. The attempted run is absent from the child's run state and run history
   until its task row exists: the run the child is parked on stays current.
8. A time or cost limit that is present but not a finite number of zero or
   more is a configuration error, never "no limit".
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from test_run_starts import delegate_in_a_turn, direct
from test_run_starts_r8 import child_state, fail_tools_for_the_child, is_child, live_holds
from test_step_allowance import World, approve_and_resume, rewrite_paused_row
from test_step_allowance_r2 import child_requests, drop_child_field, set_child_field
from test_step_allowance_r6 import dead_approval_child

from prism_harness import (
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    ModeRegistry,
    RunBudget,
    RunContext,
    RunFailed,
    Session,
    SubagentRunner,
    ToolRegistry,
    record_approval,
)
from prism_harness.plans.service import Plans
from prism_harness.run_trees import RunTrees
from prism_harness.thread import Thread
from prism_harness.thread_rows import tool_result_row


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


TTL = 300


def start(world: World, path: str, task: str = "Again.", answer: bool = True) -> str:
    """A runner start onto the worker child, through fresh objects: through
    the runner directly, or through the subagent tool inside a parent turn.
    The child answers once if it takes a model step. The framed text (or, in
    a turn, the parent's next request)."""
    world.fresh()
    world.queue.clear()
    if path == "direct":
        if answer:
            world.text("Child answer.")
        parent = world.root()
        text = str(direct(world, parent, RunContext.root("host", RunBudget(8)), task).text)
    else:
        world.calls(("worker", {"task": task}))
        if answer:
            world.text("Child answer.")
        world.text("Root done.")
        world.runtime.send(world.root(), "Go")
        text = str(world.requests[-1].messages)
    world.queue.clear()
    return text


def refused_as_unbound(action: Callable[[], Any]) -> None:
    with pytest.raises(HarnessError) as caught:
        action()
    assert caught.value.code == "child_binding_missing"


def fail_the_undo(patch: pytest.MonkeyPatch) -> None:
    def failing(self: Session, *_args: Any, **_kwargs: Any) -> None:
        raise OSError("store down")

    patch.setattr(Session, "undo_child_start", failing)


def outage(patch: pytest.MonkeyPatch) -> None:
    """One store outage: the start fails after (b), and every write that could
    clean up after it fails too."""
    fail_tools_for_the_child(patch)
    fail_the_undo(patch)
    for name in ("settle_child_steps", "fail_run", "confirm_child_start"):
        original = getattr(Session, name, None)
        if original is None:
            continue

        def failing(
            self: Session, *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            if is_child(self):
                raise OSError("store down")
            return _original(self, *args, **kwargs)

        patch.setattr(Session, name, failing)


def later(patch: pytest.MonkeyPatch, seconds: int = TTL + 1) -> None:
    original = Plans.now

    def now(self: Plans) -> int:
        return original(self) + seconds

    patch.setattr(Plans, "now", now)


def decision_row(approval: str) -> dict[str, Any]:
    return tool_result_row([], [{"approval_id": approval, "approved": True, "reason": None}])


def thread_text(world: World) -> str:
    return str([entry.message for entry in world.fresh().child("worker").thread().messages()])


# -- R12-1: the start fails after (b) and the undo fails too --------------------


@pytest.mark.parametrize("path", ["direct", "tool"])
@pytest.mark.parametrize("how", ["undo_fails", "outage"])
def test_a_failed_start_whose_undo_fails_keeps_a_dead_approval_dead(
    path: str, how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    approval = dead_approval_child(world)
    requests = child_requests(world)

    with pytest.MonkeyPatch.context() as patch:
        if how == "undo_fails":
            fail_tools_for_the_child(patch)
            fail_the_undo(patch)
        else:
            outage(patch)
        text = start(world, path)

    assert "did not run: OSError" in text
    refused_as_unbound(lambda: record_approval(world.fresh().child("worker"), approval, True))
    refused_as_unbound(lambda: world.fresh().runtime.send(world.child("worker"), ""))
    refused_as_unbound(lambda: world.fresh().runtime.send(world.child("worker"), "Hello."))
    refused_as_unbound(
        lambda: world.fresh().runtime.send(
            world.child("worker"), "", context=RunContext.root("host", RunBudget(10))
        )
    )
    assert child_requests(world) == requests
    assert "echo" not in world.ran

    # Once the start's hold has expired, a re-delegation is not blocked.
    with pytest.MonkeyPatch.context() as patch:
        later(patch)
        again = start(world, "direct", "New task.")
    assert "Child answer." in again
    assert "echo" not in world.ran


# -- R12-2: a decision recorded between (b) and the task row --------------------


def in_the_window(world: World, patch: pytest.MonkeyPatch, act: Callable[[], None]) -> list[int]:
    """Run ``act`` once, inside a runner start on the child, after (b) and
    before its task row (where the plan context is read)."""
    original = Plans.context
    ran: list[int] = []

    def hooked(self: Plans, session: Session, prompt: str) -> str:
        if is_child(session) and not ran:
            ran.append(1)
            act()
        return original(self, session, prompt)

    patch.setattr(Plans, "context", hooked)
    return ran


@pytest.mark.parametrize("path", ["direct", "tool"])
@pytest.mark.parametrize("ends", ["completes", "fails"])
def test_a_decision_recorded_in_the_window_is_never_acted_on(
    path: str, ends: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    approval = dead_approval_child(world)
    answers: list[str] = []

    def decide() -> None:
        try:
            record_approval(world.child("worker"), approval, True)
            answers.append("accepted")
        except HarnessError as error:
            answers.append(error.code)
        # A decision that landed anyway (a host writing the row itself).
        world.child("worker").thread().record([decision_row(approval)], None)

    with pytest.MonkeyPatch.context() as patch:
        ran = in_the_window(world, patch, decide)
        if ends == "fails":
            fail_tools_for_the_child(patch)
        start(world, path)

    assert ran
    assert answers == ["child_binding_missing"]
    assert "echo" not in world.ran  # not by this start

    start(world, "direct", "Next task.")
    assert "echo" not in world.ran  # nor by the next delegation

    try:
        world.fresh().runtime.send(world.child("worker"), "")
    except HarnessError:
        pass
    assert "echo" not in world.ran  # nor by a host resume


@pytest.mark.parametrize("path", ["direct", "tool"])
def test_a_decision_recorded_while_bound_is_not_run_by_a_re_delegation(path: str) -> None:
    """P3: a person approves while the binding is alive; the binding is then
    lost. A bare resume is refused, and a re-delegation runs nothing of it."""
    world = World()
    world.run_child_in_runner("worker")
    child = world.child("worker")
    (approval,) = world.pending(child)
    record_approval(world.fresh().child("worker"), approval, True)
    drop_child_field(world.child("worker"), "child_binding")
    rewrite_paused_row(world.child("worker"), lambda row: row.pop("harness_binding", None))

    refused_as_unbound(lambda: world.fresh().runtime.send(world.child("worker"), ""))
    assert "echo" not in world.ran

    text = start(world, path, "New task.")

    assert "echo" not in world.ran
    assert "Child answer." in text
    # The task row ended the old call: recorded as not run, never pending.
    assert world.pending(world.fresh().child("worker")) == []
    try:
        world.fresh().runtime.send(world.child("worker"), "")
    except HarnessError:
        pass
    assert "echo" not in world.ran


def test_a_runner_start_onto_an_unanswered_dead_approval_acts_on_nothing() -> None:
    world = World()
    dead_approval_child(world)
    events = HarnessEvents()
    seen: list[HarnessEvent] = []
    events.listen(seen.append)
    world.runtime._events = events
    world.text("Child answer.")

    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "New task.")

    assert result.outcome == "completed"
    assert "echo" not in world.ran
    assert not any(type(event).__name__ == "ToolApproved" for event in seen)
    assert world.pending(world.fresh().child("worker")) == []


# -- no regression on the normal path -------------------------------------------


@pytest.mark.parametrize("housekeeping", ["runs", "crashes"])
def test_a_confirmed_start_behaves_as_before(
    housekeeping: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provisional record whose task row exists is a confirmed one: the
    child parks, its approval is answered, it resumes, the delegation count
    holds (D1: declared 2, so 2 model requests in all). A crash right after
    the task row (the housekeeping clear never runs) changes nothing."""
    if housekeeping == "crashes":

        def crash(self: Session, *_args: Any, **_kwargs: Any) -> None:
            raise OSError("crashed after the task row")

        monkeypatch.setattr(Session, "confirm_child_start", crash, raising=False)
    world = World()
    world.run_child_in_runner("worker")
    child = world.fresh().child("worker")
    assert child.child_binding() is not None
    assert child.child_ceiling() == 2
    record = child_state(world)["child"]
    if housekeeping == "runs":
        assert "provisional" not in record

    response = approve_and_resume(world, lambda: world.child("worker"), 6)

    assert response is not None
    assert child_requests(world) == 2
    assert "echo" in world.ran


def test_a_legacy_record_without_the_mark_reads_as_before() -> None:
    world = World()
    world.run_child_in_runner("worker")
    drop_child_field(world.child("worker"), "provisional")
    (approval,) = world.pending(world.child("worker"))

    record_approval(world.fresh().child("worker"), approval, True)
    world.fresh().runtime.send(world.child("worker"), "")

    assert "echo" in world.ran


@pytest.mark.parametrize("garbage", [123, "", None, True, ["run"], "no-such-run"])
def test_a_garbage_provisional_mark_fails_closed(garbage: object) -> None:
    world = World()
    world.run_child_in_runner("worker")
    set_child_field(world.child("worker"), "provisional", garbage)
    (approval,) = world.pending(world.child("worker"))

    refused_as_unbound(lambda: record_approval(world.fresh().child("worker"), approval, True))
    refused_as_unbound(lambda: world.fresh().runtime.send(world.child("worker"), "Hello."))
    assert world.fresh().child("worker").child_ceiling_state() == ("absent", None)
    assert "echo" not in world.ran


# -- new channels ---------------------------------------------------------------


def failed_start_with_undo_failing(world: World) -> str:
    approval = dead_approval_child(world)
    with pytest.MonkeyPatch.context() as patch:
        fail_tools_for_the_child(patch)
        fail_the_undo(patch)
        start(world, "direct")
    return approval


def provisional_of(world: World) -> object:
    record = child_state(world)["child"]
    assert isinstance(record, dict)
    return record.get("provisional")


def test_a_row_under_another_run_or_of_another_kind_never_confirms() -> None:
    world = World()
    approval = failed_start_with_undo_failing(world)
    mark = provisional_of(world)
    assert isinstance(mark, str) and mark

    thread = world.child("worker").thread()
    # A user row recorded under ANOTHER run id, and a non-user row recorded
    # under the start's own run id: neither is that start's task row.
    thread.record([{"type": "user", "content": "Hi."}], "another-run")
    thread.record([{"type": "assistant", "content": "x", "tool_calls": []}], mark)

    refused_as_unbound(lambda: record_approval(world.fresh().child("worker"), approval, True))
    assert world.fresh().child("worker").child_binding() is None


def test_a_starts_mark_is_its_own_fresh_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host forging a mark that names an old task row gains nothing a host
    writing the record directly does not already have; the harness itself
    only ever marks with the run id of the start writing it, which no row
    carries yet."""
    world = World()
    world.run_child_in_runner("worker")
    first_runs = {
        entry.run_id for entry in world.child("worker").thread().messages() if entry.run_id
    }
    drop_child_field(world.child("worker"), "child_binding")
    rewrite_paused_row(world.child("worker"), lambda row: row.pop("harness_binding", None))
    seen: list[object] = []

    def look() -> None:
        mark = provisional_of(world)
        seen.append(mark)
        rows = world.child("worker").thread().messages()
        seen.append(any(entry.run_id == mark for entry in rows))

    with pytest.MonkeyPatch.context() as patch:
        in_the_window(world, patch, look)
        start(world, "direct", "New task.")

    mark, already = seen
    assert isinstance(mark, str) and mark not in first_runs
    assert already is False


def test_every_reader_sees_a_start_in_its_window_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume, a new message, a host turn and a decision racing the start
    (after (b), before its task row) are all refused, with no model step."""
    world = World()
    approval = dead_approval_child(world)
    parked = world.child("worker").run()
    requests = child_requests(world)
    found: dict[str, Any] = {}

    def race() -> None:
        child = world.child("worker")
        found["binding"] = child.child_binding()
        found["ceiling"] = child.child_ceiling_state()
        found["refs"] = child.child_refs()
        found["run"] = child.run()
        for name, act in {
            "decision": lambda: record_approval(world.child("worker"), approval, True),
            "resume": lambda: world.runtime.send(world.child("worker"), ""),
            "message": lambda: world.runtime.send(world.child("worker"), "Hello."),
            "host": lambda: world.runtime.send(
                world.child("worker"), "", context=RunContext.root("h", RunBudget(10))
            ),
        }.items():
            try:
                act()
                found[name] = "ran"
            except HarnessError as error:
                found[name] = error.code

    with pytest.MonkeyPatch.context() as patch:
        in_the_window(world, patch, race)
        start(world, "direct", "New task.")

    assert found["binding"] is None
    assert found["ceiling"] == ("absent", None)
    assert found["refs"] == {"tree": None, "plan_ledger": None}
    assert found["run"] == parked
    for name in ("decision", "resume", "message", "host"):
        assert found[name] == "child_binding_missing", name
    assert child_requests(world) == requests + 1  # only the start's own step
    assert "echo" not in world.ran


def test_a_resume_racing_the_task_row_write_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World()
    dead_approval_child(world)
    original = Thread.record
    found: list[str] = []

    def hooked(self: Thread, messages: Any, run_id: str | None = None) -> Any:
        if (
            not found
            and "::sub::worker" in self._key
            and any(row.get("type") == "user" for row in messages)
        ):
            try:
                world.runtime.send(world.child("worker"), "")
                found.append("ran")
            except HarnessError as error:
                found.append(error.code)
        return original(self, messages, run_id)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Thread, "record", hooked)
        start(world, "direct", "New task.")

    assert found == ["child_binding_missing"]
    assert "echo" not in world.ran


# -- 4. a failure handler cleans up first, reading attributes only guarded -----


class NoInit(HarnessError):
    """A host subclass that never ran HarnessError's constructor."""

    def __init__(self) -> None:
        Exception.__init__(self, "SECRET no init")


class BadDocument(HarnessError):
    def __init__(self) -> None:
        Exception.__init__(self, "SECRET bad document")

    @property
    def document(self) -> str | None:  # type: ignore[override]
        raise RuntimeError("SECRET document getter")


class BadCode(HarnessError):
    def __init__(self) -> None:
        Exception.__init__(self, "SECRET bad code")

    @property
    def code(self) -> str:  # type: ignore[override]
        raise RuntimeError("SECRET code getter")


ERRORS: dict[str, Callable[[], BaseException]] = {
    "no_init": NoInit,
    "document_raises": BadDocument,
    "code_raises": BadCode,
}


def raise_from_the_child_tools(patch: pytest.MonkeyPatch, error: Callable[[], BaseException]) -> None:
    original = ToolRegistry.resolve_detailed

    def failing(self: ToolRegistry, names: Any, session: Session | None = None) -> Any:
        if session is not None and is_child(session):
            raise error()
        return original(self, names, session)

    patch.setattr(ToolRegistry, "resolve_detailed", failing)


def listening(world: World) -> list[RunFailed]:
    events = HarnessEvents()
    failed: list[RunFailed] = []
    events.listen(lambda event: failed.append(event) if isinstance(event, RunFailed) else None)
    world.runtime._events = events
    return failed


@pytest.mark.parametrize("kind", list(ERRORS))
def test_a_host_error_without_its_attributes_still_cleans_up_a_start(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    approval = dead_approval_child(world)
    before = child_state(world)
    world.fresh()
    failed = listening(world)

    with pytest.MonkeyPatch.context() as patch:
        raise_from_the_child_tools(patch, ERRORS[kind])
        result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Again.")

    assert result.outcome == "failed"
    assert "SECRET" not in result.text
    assert len(failed) == 1 and "SECRET" not in failed[0].failure
    assert child_state(world) == before
    assert live_holds(world) == {}
    refused_as_unbound(lambda: record_approval(world.fresh().child("worker"), approval, True))


@pytest.mark.parametrize("kind", list(ERRORS))
def test_a_host_error_without_its_attributes_still_fails_a_root_run(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    failed = listening(world)

    def failing(self: ToolRegistry, *_args: Any, **_kwargs: Any) -> Any:
        raise ERRORS[kind]()

    monkeypatch.setattr(ToolRegistry, "resolve", failing)
    monkeypatch.setattr(ToolRegistry, "resolve_detailed", failing)

    with pytest.raises(HarnessError):
        world.runtime.send(world.root(), "Go")

    assert len(failed) == 1 and "SECRET" not in failed[0].failure
    run = world.fresh().root().run()
    assert run is not None and run["status"] == "failed"
    assert "SECRET" not in str(run)


# -- 7. the attempted run is absent until its task row --------------------------


def fail_run_trees_record_for_the_child(patch: pytest.MonkeyPatch) -> None:
    original = RunTrees.record

    def failing(self: RunTrees, session: Session, *args: Any, **kwargs: Any) -> Any:
        if is_child(session):
            raise OSError("store down")
        return original(self, session, *args, **kwargs)

    patch.setattr(RunTrees, "record", failing)


@pytest.mark.parametrize("where", ["run_trees_record", "tool_registry"])
@pytest.mark.parametrize("undo", ["lands", "fails"])
def test_a_failed_start_leaves_the_parked_run_current(
    where: str, undo: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()
    dead_approval_child(world)
    parked = world.fresh().child("worker").run()
    assert parked is not None
    attempted: list[str] = []
    original = Session.begin_run

    def begin(self: Session, run_id: str, *args: Any, **kwargs: Any) -> Any:
        if is_child(self):
            attempted.append(run_id)
        return original(self, run_id, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Session, "begin_run", begin)
        if where == "run_trees_record":
            fail_run_trees_record_for_the_child(patch)
        else:
            fail_tools_for_the_child(patch)
        if undo == "fails":
            fail_the_undo(patch)
        start(world, "direct")

    (attempt,) = attempted
    child = world.fresh().child("worker")
    assert child.run() == parked  # the parked run, never the attempted one
    assert child._run_tree(attempt) is None  # a store reader: absent from history


# -- 8. a limit present but unreadable is a configuration error -----------------


@pytest.mark.parametrize("field", ["max_seconds", "max_cost_usd"])
@pytest.mark.parametrize("value", ["NAN", "INF", "-INF", "0,50", "60", [], {}, "", -1])
def test_an_unreadable_limit_is_a_configuration_error(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        RunBudget.from_config({field: value})
    with pytest.raises((ValueError, TypeError)):
        RunBudget(8, **{field: value})  # type: ignore[arg-type]


def test_a_mode_declaring_a_quoted_limit_is_malformed() -> None:
    registry = ModeRegistry(
        {
            "default": "lead",
            "modes": {
                "lead": {
                    "system_prompt": "Lead.",
                    "subagents": {"worker": {"mode": "work", "max_seconds": "60"}},
                },
                "work": {"system_prompt": "Work."},
            },
        }
    )
    with pytest.raises(HarnessError) as caught:
        registry.resolve("lead")
    assert caught.value.code == "mode_malformed"


def test_the_framed_start_text_never_names_foreign_codes() -> None:
    """Item 3: only the harness's own codes are repeated (already true here)."""
    world = World()

    class Foreign(HarnessError):
        pass

    with pytest.MonkeyPatch.context() as patch:
        raise_from_the_child_tools(
            patch, lambda: Foreign("operator.note.db.password.hunter2", "x")
        )
        result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Look.")

    assert "hunter2" not in result.text
    assert result.text == "The [worker] subagent did not run: Foreign."
    assert SubagentRunner is not None
