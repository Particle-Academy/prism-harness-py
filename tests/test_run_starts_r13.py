"""G-387 round 13: the round-12 review findings, one rule each.

1. An approval reads ENDED when ANY result row after its own request row and
   before the next assistant row carries ``harness_ended_by_start`` naming
   it, not only the first row that carries its call id (P1, P1b).
3. On a step whose call ids repeat, every recorded decision is a DENIAL with
   the G-392 text, and results are matched to calls BY POSITION, never through
   a map keyed by call id: a gated call that shares its id with an UNGATED call
   is still pending, so a runner start ends it with its own approval, and a
   resume denies it (P3, four variants; P4).
4. The Session object that committed a start cannot re-enter that start's
   own-run view once the run has ended (P5); the BaseException path runs the
   plan's after-turn hook too.
5. The success path never undoes a start: the task row is recorded as
   written in memory at the write itself, so a store failure after that write
   (a lock release, a read in cleanup) keeps the start's records (ZB).

Every test enters at the public seam (``SubagentRunner.run``,
``runtime.send``, ``record_approval``); the model is scripted and the network
is refused.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from test_repeated_call_ids import (
    REPEATED,
    Recording,
    child_parked_on,
    rows,
    say,
    start_again,
    unanswerable,
)
from test_run_starts import direct
from test_run_starts_r8 import is_child
from test_run_starts_r9 import outage, refused_as_unbound, start
from test_run_starts_r10 import Abort
from test_step_allowance import World, rewrite_paused_row
from test_step_allowance_r6 import dead_approval_child

from prism_harness import (
    HarnessError,
    RunBudget,
    RunContext,
    Session,
    StructuredAgentResponse,
    SubagentRunner,
    record_approval,
)
from prism_harness.plans.service import Plans
from prism_harness.plans.thread_facts import approval_decisions, pending_approval_calls
from prism_harness.thread_rows import (
    ENDED_BY_START,
    ended_approval_ids,
    pending_approval_ids,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def child_rows(world: World) -> list[dict[str, Any]]:
    return rows(world.fresh().child("worker"))


def marks(world: World) -> list[Any]:
    return [row[ENDED_BY_START] for row in child_rows(world) if ENDED_BY_START in row]


def stored_decisions(world: World) -> list[dict[str, Any]]:
    return [
        dict(response)
        for row in child_rows(world)
        if row.get("type") == "tool_result"
        for response in row.get("tool_approval_responses", [])
    ]


GATED_FIRST = "gated-first"
UNGATED_FIRST = "ungated-first"


def mixed(call_id: str, order: str) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """A step whose gated ``echo`` shares its id with the ungated ``lookup``."""
    gated = (call_id, "echo", {"n": 1})
    ungated = (call_id, "lookup", {"n": 9})
    return (gated, ungated) if order == GATED_FIRST else (ungated, gated)


def park_mixed(world: Recording, call_id: str, order: str) -> str:
    """The worker child, started by the runner, parks on the mixed step: the
    ungated call runs at once (its result row carries the shared id) and the
    gated one asks for one approval."""
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    say(world, *mixed(call_id, order))
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", "go"
    )
    approvals = pending_approval_ids(rows(world.child("worker")))
    assert len(approvals) == 1
    assert world.ran_with == [("lookup", {"n": 9})]
    return approvals[0]


VARIANTS = [
    pytest.param("dup", GATED_FIRST, id="dup-gated-first"),
    pytest.param("dup", UNGATED_FIRST, id="dup-ungated-first"),
    pytest.param("", GATED_FIRST, id="empty-gated-first"),
    pytest.param("", UNGATED_FIRST, id="empty-ungated-first"),
]


# -- 1 and 3. a start ends the gated call whatever shares its id (P1, P3) --------


@pytest.mark.parametrize(("call_id", "order"), VARIANTS)
def test_a_start_ends_a_gated_call_that_shares_its_id_with_an_ungated_one(
    call_id: str, order: str
) -> None:
    world = Recording()
    approval = park_mixed(world, call_id, order)
    # A person approves while the approval is answerable.
    record_approval(world.fresh().child("worker"), approval, True)

    start_again(world)

    stored = child_rows(world)
    assert marks(world) == [{"approval_ids": [approval]}]
    assert approval in ended_approval_ids(stored)
    assert approval_decisions(stored).get(approval) is not True
    assert ("echo", {"n": 1}) not in world.ran_with
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().child("worker"), approval, True)
    assert caught.value.code == "approval_not_pending"


@pytest.mark.parametrize(("call_id", "order"), VARIANTS)
def test_the_pending_approval_names_the_gated_call_not_its_ungated_twin(
    call_id: str, order: str
) -> None:
    """The plan's view of what a person is asked about (TS I-5)."""
    world = Recording()
    approval = park_mixed(world, call_id, order)

    (pending,) = pending_approval_calls(child_rows(world))
    assert pending["id"] == approval
    assert pending["name"] == "echo"


@pytest.mark.parametrize(("call_id", "order"), VARIANTS)
def test_a_resume_denies_the_gated_call_with_the_repeated_id_text(call_id: str, order: str) -> None:
    world = Recording()
    approval = park_mixed(world, call_id, order)
    record_approval(world.fresh().child("worker"), approval, True)
    world.text("Done.")

    world.fresh().runtime.send(world.child("worker"), "")

    assert ("echo", {"n": 1}) not in world.ran_with
    stored = child_rows(world)
    echoes = [
        entry["result"]
        for row in stored
        if row.get("type") == "tool_result"
        for entry in row["tool_results"]
        if entry["tool_name"] == "echo"
    ]
    assert REPEATED in echoes
    assert approval_decisions(stored).get(approval) is False


@pytest.mark.parametrize(("call_id", "order"), VARIANTS)
def test_a_decision_on_a_mixed_repeated_step_is_recorded_as_a_denial(
    call_id: str, order: str
) -> None:
    world = Recording()
    approval = park_mixed(world, call_id, order)
    record_approval(world.fresh().child("worker"), approval, True)

    assert stored_decisions(world) == [
        {"approval_id": approval, "approved": False, "reason": REPEATED}
    ]


# -- 3 (P4 / I1). every decision on a repeated-id step is a denial ---------------


@pytest.mark.parametrize("call_id", ["dup", ""], ids=["same-id", "empty-id"])
def test_every_decision_on_a_repeated_id_step_is_recorded_as_a_denial(call_id: str) -> None:
    world = Recording()
    approvals = child_parked_on(world, (call_id, "echo", {"n": 1}), (call_id, "echo", {"n": 2}))
    record_approval(world.fresh().child("worker"), approvals[0], True)
    record_approval(world.fresh().child("worker"), approvals[1], False, "no")

    assert stored_decisions(world) == [
        {"approval_id": approvals[0], "approved": False, "reason": REPEATED},
        {"approval_id": approvals[1], "approved": False, "reason": REPEATED},
    ]
    assert approval_decisions(child_rows(world)) == dict.fromkeys(approvals, False)


def test_a_unique_id_step_still_records_each_decision_as_given() -> None:
    world = Recording()
    approvals = child_parked_on(world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))
    record_approval(world.fresh().child("worker"), approvals[0], True)
    record_approval(world.fresh().child("worker"), approvals[1], False, "no")

    assert stored_decisions(world) == [
        {"approval_id": approvals[0], "approved": True, "reason": None},
        {"approval_id": approvals[1], "approved": False, "reason": "no"},
    ]


# -- 1 (P1b). one function name, one call with arguments that are not an object ---


def test_one_name_under_one_id_with_unusable_arguments_never_runs_or_reads_approved() -> None:
    """Gemini-style ids: two ``echo`` calls share the id ``echo``, one with
    arguments that are not a JSON object. Neither runs, and no approval on
    the step reads approved after a start."""
    world = Recording()
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    say(world, ("echo", "echo", {"n": 1}), ("echo", "echo", "not an object"))  # type: ignore[arg-type]
    world.text("Child answer.")
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", "go"
    )
    approvals = pending_approval_ids(child_rows(world))
    for approval in approvals:
        record_approval(world.fresh().child("worker"), approval, True)
    unanswerable(world)
    start_again(world)

    stored = child_rows(world)
    assert world.ran_with == []
    assert all(approval_decisions(stored).get(approval) is not True for approval in approvals)


# -- 1. the read rule: any marked row in the step, not the first with the id -----


def test_an_unmarked_earlier_result_row_does_not_hide_the_starts_mark() -> None:
    """The rule as read from stored rows (the parity row pins it too): an
    approval reads ended when ANY result row after its request and before
    the next assistant row names it in the mark."""
    assistant: dict[str, Any] = {
        "type": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "w", "name": "echo", "arguments": {"n": 1}},
            {"id": "w", "name": "lookup", "arguments": {"n": 9}},
        ],
        "additional_content": {},
        "tool_approval_requests": [{"approval_id": "apr_x", "tool_call_id": "w"}],
    }
    unmarked: dict[str, Any] = {
        "type": "tool_result",
        "tool_results": [{"tool_call_id": "w", "tool_name": "lookup", "result": "ok"}],
        "tool_approval_responses": [],
    }
    decided: dict[str, Any] = {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [{"approval_id": "apr_x", "approved": True, "reason": None}],
    }
    marked: dict[str, Any] = {
        "type": "tool_result",
        "tool_results": [{"tool_call_id": "w", "tool_name": "echo", "result": "Not run"}],
        "tool_approval_responses": [],
        ENDED_BY_START: {"approval_ids": ["apr_x"]},
        # As read from a row the harness wrote (G-387 round 16).
        "harness": True,
    }
    task: dict[str, Any] = {"type": "user", "content": "New task."}
    found: list[dict[str, Any]] = [assistant, unmarked, decided, marked, task]

    assert ended_approval_ids(found) == {"apr_x"}
    assert approval_decisions(found) == {}
    # A mark after the NEXT assistant step names nothing of this one.
    later: dict[str, Any] = {**assistant, "tool_approval_requests": []}
    assert ended_approval_ids([assistant, unmarked, decided, later, marked]) == set()
    # The same mark on a row the harness did not write names nothing.
    forged = {key: value for key, value in marked.items() if key != "harness"}
    assert ended_approval_ids([assistant, unmarked, decided, forged, task]) == set()


# -- 4. the committing object cannot re-enter its view once its run ended (P5) ---


def test_the_committing_object_cannot_re_enter_its_view_on_its_own_crash_residue() -> None:
    world = World()
    approval = dead_approval_child(world)
    committed: list[Session] = []
    original = Session.start_child

    def commit(self: Session, *args: Any, **kwargs: Any) -> Any:
        handle = original(self, *args, **kwargs)
        if is_child(self):
            committed.append(self)
        return handle

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Session, "start_child", commit)
        outage(patch)
        start(world, "direct", answer=False)
    (own,) = committed
    residue = own._durable.get(f"{own.key()}:child")  # a store reader
    assert isinstance(residue, dict)
    run_id = residue["provisional"]

    leave = own.enter_own_start(run_id)
    try:
        own._cached_state = None
        assert own.child_binding() is None
        assert own.child_ceiling() is None
        assert own.child_provisional()
    finally:
        leave()
    refused_as_unbound(lambda: record_approval(world.fresh().child("worker"), approval, True))


def test_a_completed_starts_object_cannot_re_enter_its_view_either() -> None:
    world = World()
    committed: list[tuple[Session, str]] = []
    original = Session.start_child

    def commit(self: Session, run_id: str, *args: Any, **kwargs: Any) -> Any:
        handle = original(self, run_id, *args, **kwargs)
        if is_child(self):
            committed.append((self, run_id))
        return handle

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Session, "start_child", commit)
        assert "Child answer." in start(world, "direct", "First.")
    ((own, run_id),) = committed
    assert own._committed_start is None
    leave = own.enter_own_start(run_id)
    leave()


# -- 4 (I3). the BaseException path runs the plan's after-turn hook ---------------


def test_a_base_exception_still_runs_the_plans_after_turn_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    seen: list[str] = []
    original = Plans.after_turn

    def after_turn(self: Plans, session: Session) -> Any:
        seen.append(session.key())
        return original(self, session)

    monkeypatch.setattr(Plans, "after_turn", after_turn)

    def stop() -> None:
        raise Abort()

    world.on_request = stop
    with pytest.raises(Abort):
        world.runtime.send(world.root(), "Go")

    assert seen == [world.root().key()]


# -- 5. the success path never undoes a start (ZB) --------------------------------


def _kept(world: World) -> None:
    """The start stands: confirmed, bound, in its mode, and the next host turn
    on the child runs a model step."""
    child = world.fresh().child("worker")
    assert not child.child_provisional()
    assert child.child_binding() is not None
    assert child.mode() == "work"
    world.queue.clear()
    world.text("Host turn.")
    before = len(world.requests)
    response = world.runtime.send(world.fresh().child("worker"), "More.")
    assert response.text == "Host turn."
    assert len(world.requests) == before + 1


def _failing_after_the_task_row(patch: pytest.MonkeyPatch, world: World) -> None:
    """The child's lock returns from the write that recorded the start's task
    row, then fails (a store outage on release)."""
    original = Session.lock
    armed = [True]

    def lock(self: Session, callback: Callable[[Session], Any], *args: Any) -> Any:
        result = original(self, callback, *args)
        if armed[0] and is_child(self):
            tasks = [
                entry
                for entry in self.thread().messages()
                if entry.message.get("type") == "user" and entry.message.get("content") == "Task."
            ]
            if tasks:
                armed[0] = False
                raise OSError("store down")
        return result

    patch.setattr(Session, "lock", lock)


@pytest.mark.parametrize("prior", ["fresh", "parked"])
def test_a_store_failure_right_after_the_task_row_never_undoes_the_start(prior: str) -> None:
    world = World()
    if prior == "parked":
        dead_approval_child(world)
    world.fresh()
    world.queue.clear()
    world.text("Child answer.")
    with pytest.MonkeyPatch.context() as patch:
        _failing_after_the_task_row(patch, world)
        result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Task.")
    assert result.outcome == "failed"

    _kept(world)


def test_a_read_failure_at_the_end_of_a_completed_start_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    original = Plans.after_turn
    armed = [True]

    def after_turn(self: Plans, session: Session) -> Any:
        if armed[0] and is_child(session):
            armed[0] = False
            raise OSError("store read down")
        return original(self, session)

    monkeypatch.setattr(Plans, "after_turn", after_turn)
    world.text("Child answer.")
    direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Task.")
    monkeypatch.setattr(Plans, "after_turn", original)

    _kept(world)


def test_a_start_that_fails_after_its_task_row_keeps_its_records() -> None:
    world = World()

    def down() -> None:
        raise OSError("provider down")

    world.on_request = down
    result = direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Task.")
    assert result.outcome == "failed"

    _kept(world)


# -- new channels ------------------------------------------------------------------


def test_a_repeated_id_among_the_requests_alone_records_denials() -> None:
    """A stored step whose calls are unique but whose requests both name one
    call (written by another writer): every decision on it is a denial."""
    world = Recording()
    approvals = child_parked_on(world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))

    def same_call(row: dict[str, Any]) -> None:
        for request in row["tool_approval_requests"]:
            request["tool_call_id"] = "a"

    rewrite_paused_row(world.child("worker"), same_call)
    record_approval(world.fresh().child("worker"), approvals[0], True)

    assert stored_decisions(world) == [
        {"approval_id": approvals[0], "approved": False, "reason": REPEATED}
    ]


def test_a_start_whose_task_row_is_written_hands_its_tree_hold_over() -> None:
    """The kept start after a store failure leaves no live hold behind: the
    next delegation is not refused as busy, and the root's tree holds
    nothing."""
    world = World()
    world.text("Child answer.")
    with pytest.MonkeyPatch.context() as patch:
        _failing_after_the_task_row(patch, world)
        assert (
            direct(world, world.root(), RunContext.root("host", RunBudget(8)), "Task.").outcome
            == "failed"
        )
    assert world.runtime._trees.members == {}
    assert "Child answer." in start(world, "direct", "Again.")


# -- a structured send whose first step parks ---------------------------------------


def test_a_structured_send_whose_first_step_parks_returns_the_structured_shape() -> None:
    world = World()
    say(world, ("a", "echo", {"n": 1}))
    response = world.runtime.send_structured(
        world.root(),
        "Go",
        {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )

    assert isinstance(response, StructuredAgentResponse)
    assert response.finish_reason == "awaiting_approval"
    assert [pending.tool for pending in response.pending_approvals] == ["echo"]
    assert response.structured is None
