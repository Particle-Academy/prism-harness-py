"""G-387 round 11, item 7 (extended in round 12): THE INVARIANT MATRIX.

Every child-state reader, through every phase of a runner start, read from
INSIDE the start's own run (the Session object that run uses), from OUTSIDE it
(a fresh Session over the same stores, a decision through ``record_approval``)
and from the start's OWN Session object used outside its run. The expected
values are written first, as symbols, from the rules of BRIEF-R7..R12; each
symbol is then resolved per scenario:

- ``P`` prior: what the reader returned before the start began.
- ``C`` committed: what the start's commit (b) wrote (its mode ``work``, its
  binding, refs, depth and ceiling, its run, its delegation count and hold).
- ``A`` absent: no binding, depth 0, no ceiling, no delegation count, while
  the session still reads as a child the runner started (provisional).
- ``K`` kept: the last CONFIRMED run tree and plan ledger (round 10), which
  for a child never started is a pair of nones.
- ``E`` ended: the call the start's task row ended reads ended, nothing is
  pending, no paused row, and a decision on it is refused
  ``approval_not_pending``.
- ``Y`` / ``N`` / ``0``: true, false, none.
- ``R`` refused: a turn on the child is refused ``child_binding_missing``.
- ``D`` denied: a second runner start onto the child is ``denied``, writes
  nothing at all (every stored record byte-identical) and makes no model
  request.

Phases: before the gate (read while the gate reads its facts); in a
``RunStarted`` listener for the start's run; after (b) (the start's window,
before its task row); during the start's own run (its first model call); in a
``ToolDenied`` listener for the call its task row ended (parked only); after the
task row (the run finished); in a ``RunFailed`` listener and after a failed
start whose undo landed; the same for a failed start whose undo failed too. For
the ``failed_*`` phases "inside" reads the Session object the run used, after
its run ended: its own view is cleared with the run. The ``failing_*`` phases
read DURING the failure, inside the failing run's ``RunFailed`` listener.

Round 13 adds the scenarios ``repeated`` (a child parked on two gated calls
sharing one id) and ``mixed`` (a gated call sharing its id with an UNGATED call
that already ran, approved by a person while answerable: "mixed repeated");
the readers ``mark`` (the RAW stored bytes of every ended mark on the child's
thread), ``decided`` (the decision standing for the approval) and ``ran``
(every tool the child ran); the action ``host_turn`` (a new message to the
child from a host); and the start outcomes that end AFTER the task row: a
read inside a ``RunFinished`` listener, a start that fails after its task row
(``failing_kept`` / ``failed_kept``), a store failure right after the task-row
write (``failing_write`` / ``failed_write``), a start whose context is
cancelled inside its run before its first model step (``cancelled``), and a
completed start followed by a host turn (``completes_plus``). ``H`` is a host
turn that is never refused as unbound (it runs, or is refused for the
delegation's spent ceiling); ``L`` the later run a host turn began.

PHP's ``cancel S`` column has no Python counterpart: there is no cancel by run
id (no durable tree cancel, G-285); a host cancels a start through the context
it handed the runner, which the ``cancelled`` outcome covers.

Every future fix must keep this green.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from functools import cache
from typing import Any

import pytest
from test_run_starts import direct, stored
from test_run_starts_r8 import is_child
from test_run_starts_r9 import fail_the_undo, start
from test_step_allowance import World, rewrite_paused_row
from test_step_allowance_r2 import drop_child_field
from test_step_allowance_r6 import dead_approval_child

from prism_harness import (
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    LlmResponse,
    LlmToolCall,
    ModeRegistry,
    RunBudget,
    RunContext,
    RunFailed,
    RunFinished,
    RunStarted,
    Session,
    SubagentRunner,
    ToolDenied,
    ToolRegistry,
    record_approval,
)
from prism_harness.plans.service import Plans
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.session import outside_any_start
from prism_harness.thread_rows import ended_approval_ids, paused_approval_row, pending_approval_ids


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


READERS = (
    "mode",
    "binding",
    "refs",
    "depth",
    "ceiling",
    "run",
    "approval",
    "offer",
    "provisional",
    "plan_run",
    "paused",
    "plans",
    "delegation",
    "busy",
    "linked",
    "policy",
    "mark",
    "decided",
    "ran",
    "decision",
    "decision_same",
    "resume",
    "second_start",
    "host_turn",
)
#: The columns that ACT (an entry point), not read; each is exercised only in
#: the phases its rule covers, ``-`` elsewhere.
ACTIONS = ("decision", "decision_same", "resume", "second_start", "host_turn")
PHASES = (
    "before_gate",
    "run_started",
    "after_b",
    "own_run",
    "tool_denied",
    "at_run_finished",
    "after_task_row",
    "failing_undone",
    "failed_undone",
    "failing_undo_failed",
    "failed_undo_failed",
    "failing_kept",
    "failed_kept",
    "failing_write",
    "failed_write",
    "cancelled",
    "completes_plus",
)
SIDES = ("inside", "outside", "own_outside")
SCENARIOS = ("fresh", "confirmed", "parked", "repeated", "mixed")
#: A phase only some scenarios reach: a ToolDenied only for a call the task
#: row ended, so only for a child parked on one.
ONLY = {"tool_denied": ("parked", "repeated", "mixed")}


def _row(**cells: str) -> dict[str, str]:
    rest = cells.pop("rest", "?")
    return {reader: cells.get(reader, "-" if reader in ACTIONS else rest) for reader in READERS}


#: Outside, in the start's window: provisional, so absent, with the last
#: confirmed tree and plan ledger kept, and the delegation hold live.
_WINDOW_OUT = {
    "mode": "P",
    "binding": "A",
    "refs": "K",
    "depth": "A",
    "ceiling": "A",
    "run": "P",
    "approval": "P",
    "offer": "P",
    "provisional": "A",
    "plan_run": "P",
    "paused": "P",
    "plans": "P",
    "delegation": "A",
    "busy": "Y",
    "linked": "A",
    "policy": "P",
    "mark": "P",
    "decided": "P",
    "ran": "P",
}
#: Inside, in the window: what (b) committed; the approval and the paused row
#: are still the prior ones (the task row ends them).
_WINDOW_IN = {
    **dict.fromkeys(_WINDOW_OUT, "C"),
    "approval": "P",
    "paused": "P",
    "mark": "P",
    "decided": "P",
    "ran": "P",
}
#: Once the task row is in the thread, from any side: every call it ended
#: reads ended, its raw mark is stored, no decision on them stands, and
#: nothing ran.
_RECORDED = {
    **dict.fromkeys(_WINDOW_OUT, "C"),
    "approval": "E",
    "paused": "E",
    "mark": "E",
    "decided": "E",
    "ran": "P",
}
#: A failed start whose undo failed too: provisional for every reader, its
#: hold left to expire (charged in full then).
_UNDO_FAILED = dict(_WINDOW_OUT)
#: The start's run finished: no run in progress, its hold settled.
_FINISHED = {**_RECORDED, "plan_run": "0", "busy": "N"}
_PRIOR = {**dict.fromkeys(_WINDOW_OUT, "P"), "busy": "N"}
#: A completed start, then a host turn on the child: the start stands, the
#: child's run is the host turn's.
_LATER = {**_FINISHED, "run": "L"}

#: THE EXPECTED TABLE, written before the fix.
EXPECTED: dict[str, dict[str, dict[str, str]]] = {
    "before_gate": {
        "inside": _row(**_PRIOR, decision_same="P"),
        "outside": _row(**_PRIOR, decision="P"),
        "own_outside": _row(**_PRIOR, decision_same="P"),
    },
    "run_started": {
        "inside": _row(**_WINDOW_IN, decision_same="P", resume="R"),
        "outside": _row(**_WINDOW_OUT, decision="P", resume="R", second_start="D"),
        "own_outside": _row(**_WINDOW_OUT, decision_same="P", resume="R"),
    },
    "after_b": {
        "inside": _row(**_WINDOW_IN, decision_same="P", resume="R"),
        "outside": _row(**_WINDOW_OUT, decision="P", resume="R", second_start="D"),
        "own_outside": _row(**_WINDOW_OUT, decision_same="P", resume="R"),
    },
    "own_run": {
        "inside": _row(**_RECORDED, decision_same="E"),
        "outside": _row(**_RECORDED, decision="E", second_start="D"),
        "own_outside": _row(**_RECORDED, decision_same="E"),
    },
    "tool_denied": {
        "inside": _row(**_RECORDED, decision_same="E"),
        "outside": _row(**_RECORDED, decision="E", second_start="D"),
        "own_outside": _row(**_RECORDED, decision_same="E"),
    },
    # Inside the start's RunFinished listener: its run is over and settled,
    # and nothing it committed is undone.
    "at_run_finished": {
        "inside": _row(**_FINISHED, decision_same="E"),
        "outside": _row(**_FINISHED, decision="E"),
        "own_outside": _row(**_FINISHED, decision_same="E"),
    },
    "after_task_row": {
        "inside": _row(**_FINISHED, decision_same="E"),
        "outside": _row(**_FINISHED, decision="E"),
        "own_outside": _row(**_FINISHED, decision_same="E", host_turn="H"),
    },
    "failing_undone": {
        "inside": _row(**_PRIOR, decision_same="P"),
        "outside": _row(**_PRIOR, decision="P"),
        "own_outside": _row(**_PRIOR, decision_same="P"),
    },
    "failed_undone": {
        "inside": _row(**_PRIOR, decision_same="P"),
        "outside": _row(**_PRIOR, decision="P"),
        "own_outside": _row(**_PRIOR, decision_same="P"),
    },
    "failing_undo_failed": {
        "inside": _row(**_UNDO_FAILED, decision_same="P"),
        "outside": _row(**_UNDO_FAILED, decision="P"),
        "own_outside": _row(**_UNDO_FAILED, decision_same="P"),
    },
    "failed_undo_failed": {
        "inside": _row(**_UNDO_FAILED, decision_same="P"),
        "outside": _row(**_UNDO_FAILED, decision="P"),
        "own_outside": _row(**_UNDO_FAILED, decision_same="P"),
    },
    # Failed AFTER its task row (its model request raised): the start stands.
    "failing_kept": {
        "inside": _row(**_FINISHED, decision_same="E"),
        "outside": _row(**_FINISHED, decision="E"),
        "own_outside": _row(**_FINISHED, decision_same="E"),
    },
    "failed_kept": {
        "inside": _row(**_FINISHED, decision_same="E"),
        "outside": _row(**_FINISHED, decision="E"),
        "own_outside": _row(**_FINISHED, decision_same="E", host_turn="H"),
    },
    # A store failure right after the task-row write (the lock's release):
    # the write landed, so nothing is undone (round 13, ZB).
    "failing_write": {
        "inside": _row(**_FINISHED, decision_same="E"),
        "outside": _row(**_FINISHED, decision="E"),
        "own_outside": _row(**_FINISHED, decision_same="E"),
    },
    "failed_write": {
        "inside": _row(**_FINISHED, decision_same="E"),
        "outside": _row(**_FINISHED, decision="E"),
        "own_outside": _row(**_FINISHED, decision_same="E", host_turn="H"),
    },
    # Cancelled inside its run before its first model step: a committed
    # start still records its task (a stop never refuses it) and takes no
    # model step.
    "cancelled": {
        "inside": _row(**_FINISHED, decision_same="E"),
        "outside": _row(**_FINISHED, decision="E"),
        "own_outside": _row(**_FINISHED, decision_same="E"),
    },
    "completes_plus": {
        # The start's own Session object keeps its cached view of the run it
        # ran; the host turn ran on another object.
        "inside": _row(**_FINISHED, decision_same="E"),
        "outside": _row(**_LATER, decision="E"),
        "own_outside": _row(**_LATER, decision_same="E", host_turn="H"),
    },
}


def table_markdown() -> str:
    """The expected table as markdown, for the PR body."""
    lines = [
        "| phase | side | " + " | ".join(READERS) + " |",
        "|---|---|" + "---|" * len(READERS),
    ]
    for phase in PHASES:
        for side in SIDES:
            cells = EXPECTED[phase][side]
            lines.append(f"| {phase} | {side} | " + " | ".join(cells[r] for r in READERS) + " |")
    return "\n".join(lines)


# -- the readers -------------------------------------------------------------------

#: Plans on for ``work``, off for ``free`` and for no mode: the plan reader
#: follows the mode each side reads.
_PLAN_MODES = ModeRegistry(
    {
        "default": "free",
        "modes": {
            "work": {"system_prompt": "Work.", "tools": ["echo"], "plans": True},
            "free": {"system_prompt": "Free.", "tools": ["lookup"]},
        },
    }
)


def _plans() -> Plans:
    plans = Plans({"enabled": True})
    plans._modes = _PLAN_MODES  # the registry the runtime would hand it
    return plans


def _refs(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    def pair(found: object, first: str, second: str) -> dict[str, str] | None:
        if not isinstance(found, dict):
            return None
        a, b = found.get(first), found.get(second)
        ok = isinstance(a, str) and a and isinstance(b, str) and b
        return {first: a, second: b} if ok else None  # type: ignore[dict-item]

    return {
        "tree": pair(value.get("tree"), "session", "run"),
        "plan_ledger": pair(value.get("plan_ledger"), "session", "plan"),
    }


def _decision(session: Session, approval: str | None) -> str:
    try:
        record_approval(session, approval or "apr_none", True)
    except HarnessError as error:
        return str(error.code)
    return "recorded"


def _resume(world: World, session: Session) -> str:
    before = len(world.requests)
    try:
        world.runtime.send(session, "")
    except HarnessError as error:
        return f"{error.code}:{len(world.requests) - before}"
    return f"ran:{len(world.requests) - before}"


def _host_turn(world: World, session: Session) -> str:
    world.queue.clear()
    world.text("Host turn.")
    before = len(world.requests)
    try:
        world.runtime.send(session, "More.")
    except HarnessError as error:
        return f"{error.code}:{len(world.requests) - before}"
    finally:
        world.queue.clear()
    return f"ran:{len(world.requests) - before}"


def raw_marks(world: World) -> list[str]:
    """Every ended mark on the child's thread as the EXACT text stored on
    disk, never re-serialised."""
    found: list[str] = []
    decoder = json.JSONDecoder()
    key = '"harness_ended_by_start": '
    for name, text in stored(world).items():
        if not name.endswith("::sub::worker:thread"):
            continue
        at = text.find(key)
        while at != -1:
            begin = at + len(key)
            _, end = decoder.raw_decode(text, begin)
            found.append(text[begin:end])
            at = text.find(key, end)
    return found


def _second_start(world: World) -> tuple[str, bool, int]:
    before, requests = stored(world), len(world.requests)
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    result = SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("second", RunBudget(8)), "root3", "Second."
    )
    return result.outcome, stored(world) == before, len(world.requests) - requests


def _policy(world: World, session: Session) -> list[str]:
    """The offered tools a call to would ask a person first: the mode's
    approval list under the host's tool policy, in the mode this side reads."""
    try:
        mode = world.runtime.mode_for(session)
    except Exception:  # noqa: BLE001 - a mode that does not resolve gates nothing
        return []
    return sorted(
        tool["name"]
        for tool in world.runtime.effective_tools(session)
        if mode.needs_approval(tool["name"])
    )


def read(
    world: World,
    session: Session,
    side: str,
    approval: str | None,
    acts: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Every reader on ``session``, and the actions ``acts`` names."""
    run = session.run()
    rows = [entry.message for entry in world.child("worker").thread().messages()]
    found: dict[str, Any] = {
        "mode": session.mode(),
        "binding": session.child_binding(),
        "refs": session.child_refs(),
        "depth": session.child_depth(),
        "ceiling": session.child_ceiling(),
        "run": run.get("id") if isinstance(run, dict) else None,
        "approval": (
            tuple(pending_approval_ids(rows)),
            approval is not None and approval in ended_approval_ids(rows),
        ),
        "offer": sorted(tool["name"] for tool in world.runtime.effective_tools(session)),
        "provisional": session.child_provisional(),
        "plan_run": _plans().run_id(session),
        "paused": paused_approval_row(rows) is not None,
        "plans": _plans().enabled_for(session),
        "delegation": isinstance(session._child_field("child_steps"), dict),
        "busy": session.child_live_hold(world.runtime.plans.now(), world.runtime.lock_ttl(session)),
        "linked": session.child_refs() is not None,
        "policy": _policy(world, session),
        "mark": raw_marks(world),
        "decided": approval_decisions(rows).get(approval) if approval is not None else None,
        "ran": tuple(world.ran),
    }
    if "decision" in acts:
        found["decision"] = _decision(world.child("worker"), approval)
    if "decision_same" in acts:
        found["decision_same"] = _decision(session, approval)
    if "resume" in acts:
        found["resume"] = _resume(world, session)
    if "second_start" in acts:
        found["second_start"] = _second_start(world)
    if "host_turn" in acts:
        found["host_turn"] = _host_turn(world, session)
    return found


# -- the scenarios -------------------------------------------------------------------


def _parked_on(world: World, *calls: tuple[str, str, dict[str, Any]]) -> list[str]:
    """The worker child, started by the runner, parks on the step given."""
    world.queue.append(
        LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(call_id, name, args) for call_id, name, args in calls],
        )
    )
    parent = world.root()
    subagent = world.runtime.mode_for(parent).subagents["worker"]
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root", RunBudget(8)), "root", "go"
    )
    return pending_approval_ids(
        [entry.message for entry in world.child("worker").thread().messages()]
    )


def _unanswerable(world: World) -> None:
    child = world.child("worker")
    drop_child_field(child, "child_binding")
    rewrite_paused_row(child, lambda row: row.pop("harness_binding", None))


def _scenario(name: str) -> tuple[World, str | None, list[str]]:
    """A world whose worker child is in the scenario's state, the approval
    it is parked on (or None), and every approval the start will end."""
    world = World()
    approval: str | None = None
    ending: list[str] = []
    if name == "confirmed":
        assert "Child answer." in start(world, "direct", "First.")
        world.child("worker").using_mode("free")
    elif name == "parked":
        approval = dead_approval_child(world)
        ending = [approval]
        world.child("worker").using_mode("free")
    elif name == "repeated":
        ending = _parked_on(world, ("dup", "echo", {"n": 1}), ("dup", "echo", {"n": 2}))
        assert len(ending) == 2
        approval = ending[0]
        _unanswerable(world)
        world.child("worker").using_mode("free")
    elif name == "mixed":
        ending = _parked_on(world, ("dup", "echo", {"n": 1}), ("dup", "lookup", {"n": 9}))
        assert len(ending) == 1 and world.ran == ["lookup"]
        approval = ending[0]
        # A person answers while the approval is answerable.
        record_approval(world.fresh().child("worker"), approval, True)
        _unanswerable(world)
        world.child("worker").using_mode("free")
    world.fresh()
    world.queue.clear()
    return world, approval, ending


def _acts(phase: str, side: str) -> tuple[str, ...]:
    return tuple(name for name in ACTIONS if EXPECTED[phase][side][name] != "-")


@cache
def observed(scenario: str, outcome: str) -> dict[str, Any]:
    """One runner start onto the scenario's child: ``success``, ``undone``
    (fails after (b), its undo lands) or ``undo_failed``. What every reader
    returned, by phase and side, and the values each symbol stands for."""
    world, approval, ending = _scenario(scenario)
    outside = world.child("worker")
    prior = read(world, outside, "outside", approval, ("decision",))
    prior["decision_same"] = prior["decision"]
    reads: dict[str, dict[str, dict[str, Any]]] = {}
    own: list[Session] = []
    committed: dict[str, Any] = {}
    child_run: list[str] = []

    def every(phase: str, session: Session) -> None:
        def side(name: str) -> dict[str, Any]:
            return read(
                world,
                session if name != "outside" else world.child("worker"),
                name,
                approval,
                _acts(phase, name),
            )

        found = {"inside": side("inside"), "outside": side("outside")}
        with outside_any_start():
            found["own_outside"] = side("own_outside")
        reads[phase] = found

    original_hold = Session.child_live_hold
    original_start = Session.start_child
    original_context = Plans.context

    reading = outcome == "success"
    # A completed start followed by a host turn needs a step left for it.
    context = RunContext.root("host", RunBudget(2 if outcome == "completes_plus" else 1))
    gate: list[Session] = []

    def at_the_gate(self: Session, *args: Any) -> bool:
        if reading and is_child(self) and not own and not gate:
            gate.append(self)
            every("before_gate", self)
        return original_hold(self, *args)

    def commit_b(self: Session, run_id: str, *args: Any, **kwargs: Any) -> Any:
        handle = original_start(self, run_id, *args, **kwargs)
        if is_child(self) and not own:
            own.append(self)
            child_run.append(run_id)
            raw = self._durable.get(f"{self.key()}:child")  # a store reader
            assert isinstance(raw, dict)
            committed.update(
                {
                    "binding": raw.get("child_binding"),
                    "refs": _refs(raw.get("child_refs")),
                    "depth": raw.get("child_depth"),
                    "ceiling": raw.get("child_ceiling"),
                    "run": run_id,
                    "plan_run": run_id,
                }
            )
        return handle

    def in_the_window(self: Plans, session: Session, prompt: str, *args: Any) -> str:
        if reading and own and session is own[0] and "after_b" not in reads:
            every("after_b", session)
            world.on_request = lambda: every("own_run", own[0])
        return original_context(self, session, prompt, *args)

    def listener(event: HarnessEvent) -> None:
        if not own or getattr(event, "run_id", None) != child_run[0]:
            return
        if isinstance(event, RunStarted) and reading:
            every("run_started", own[0])
        elif isinstance(event, ToolDenied) and reading and "tool_denied" not in reads:
            every("tool_denied", own[0])
        elif isinstance(event, RunFinished) and reading:
            every("at_run_finished", own[0])
        elif isinstance(event, RunStarted) and outcome == "cancelled":
            context.ledger.cancel("Stopped by the host.")
        elif isinstance(event, RunFailed):
            every(
                {
                    "undone": "failing_undone",
                    "fails_after": "failing_kept",
                    "write_failed": "failing_write",
                }.get(outcome, "failing_undo_failed"),
                own[0],
            )

    events = HarnessEvents()
    events.listen(listener)
    world.runtime._events = events

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Session, "child_live_hold", at_the_gate)
        patch.setattr(Session, "start_child", commit_b)
        patch.setattr(Plans, "context", in_the_window)
        if outcome == "fails_after":

            def down() -> None:
                raise OSError("provider down")

            world.on_request = down
        if outcome == "write_failed":
            _fail_after_the_task_row(patch)
        if outcome in ("undone", "undo_failed"):
            # Once: the start's own tool resolution fails, the readers after it do not.
            failures = [OSError("down")]
            resolve = ToolRegistry.resolve_detailed

            def failing_once(self: ToolRegistry, names: Any, session: Session | None = None) -> Any:
                if session is not None and is_child(session) and failures:
                    raise failures.pop()
                return resolve(self, names, session)

            patch.setattr(ToolRegistry, "resolve_detailed", failing_once)
        if outcome == "undo_failed":
            fail_the_undo(patch)
        world.text("Child answer.")
        result = direct(world, world.root(), context, "Matrix.")
    assert result.outcome == _OUTCOME.get(outcome, "failed"), result.text
    later: list[str] = []
    if outcome == "completes_plus":
        world.queue.clear()
        world.text("Host turn.")
        later.append(world.fresh().runtime.send(world.child("worker"), "More.").run_id)
        world.queue.clear()
    finished = {
        "success": "after_task_row",
        "undone": "failed_undone",
        "fails_after": "failed_kept",
        "write_failed": "failed_write",
        "cancelled": "cancelled",
        "completes_plus": "completes_plus",
    }.get(outcome, "failed_undo_failed")
    every(finished, own[0])

    kept = prior["refs"] if prior["refs"] is not None else {"tree": None, "plan_ledger": None}
    work = world.child("worker")
    values = {
        "P": prior,
        "A": {
            "binding": None,
            "depth": 0,
            "ceiling": None,
            "provisional": True,
            "delegation": False,
            "linked": True,
        },
        "K": {"refs": kept},
        "E": {
            "approval": ((), approval is not None),
            "paused": False,
            "decision": "approval_not_pending",
            "decision_same": "approval_not_pending",
            # The exact bytes the store holds: the reference shape, the
            # ended approvals in call order.
            "mark": ['{"approval_ids": [' + ", ".join(f'"{each}"' for each in ending) + "]}"]
            if ending
            else [],
            "decided": None,
        },
        "C": {
            **committed,
            "mode": "work",
            "offer": sorted(tool["name"] for tool in world.runtime.effective_tools(work, "work")),
            "provisional": False,
            "plans": True,
            "delegation": True,
            "busy": True,
            "linked": True,
            "policy": ["echo"],
        },
        "Y": dict.fromkeys(READERS, True),
        "N": dict.fromkeys(READERS, False),
        "0": dict.fromkeys(READERS, None),
        "R": {"resume": "child_binding_missing:0"},
        "D": {"second_start": ("denied", True, 0)},
        "H": {"host_turn": _HOST_TURN.get(outcome)},
        "L": {"run": later[0] if later else None},
    }
    return {"reads": reads, "values": values}


#: A host turn on the child once the start stands: it is never refused as
#: unbound. It runs one model step when the delegation's ceiling has a step
#: left (the start took none), and is refused for the spent ceiling when the
#: start (and, for ``completes_plus``, the host turn after it) took them all.
_HOST_TURN = {
    "success": "run_not_permitted:0",
    "fails_after": "ran:1",
    "write_failed": "ran:1",
    "completes_plus": "run_not_permitted:0",
}

#: The framed outcome each start outcome reports to its parent.
#: A cancelled context stops the child before its first model step: the
#: parent is told it stopped early (``exhausted``, "its run was cancelled").
_OUTCOME = {"success": "completed", "completes_plus": "completed", "cancelled": "exhausted"}


def _fail_after_the_task_row(patch: pytest.MonkeyPatch) -> None:
    """The child's lock returns from the write that recorded the start's task
    row, then fails (a store outage on its release)."""
    original = Session.lock
    armed = [True]

    def lock(self: Session, callback: Any, *args: Any) -> Any:
        result = original(self, callback, *args)
        if (
            armed[0]
            and is_child(self)
            and any(
                entry.message.get("type") == "user" and entry.message.get("content") == "Matrix."
                for entry in self.thread().messages()
            )
        ):
            armed[0] = False
            raise OSError("store down")
        return result

    patch.setattr(Session, "lock", lock)


_RUN_OF = {
    "before_gate": "success",
    "run_started": "success",
    "after_b": "success",
    "own_run": "success",
    "tool_denied": "success",
    "at_run_finished": "success",
    "after_task_row": "success",
    "failing_undone": "undone",
    "failed_undone": "undone",
    "failing_undo_failed": "undo_failed",
    "failed_undo_failed": "undo_failed",
    "failing_kept": "fails_after",
    "failed_kept": "fails_after",
    "failing_write": "write_failed",
    "failed_write": "write_failed",
    "cancelled": "cancelled",
    "completes_plus": "completes_plus",
}

CELLS = [
    (scenario, phase, side)
    for scenario in SCENARIOS
    for phase in PHASES
    for side in SIDES
    if scenario in ONLY.get(phase, SCENARIOS)
]


@pytest.mark.parametrize(("scenario", "phase", "side"), CELLS)
def test_the_invariant_matrix(scenario: str, phase: str, side: str) -> None:
    found = observed(scenario, _RUN_OF[phase])
    assert phase in found["reads"], f"{phase} was never reached"
    got = found["reads"][phase][side]
    wrong: dict[str, Any] = {}
    for reader in READERS:
        symbol = EXPECTED[phase][side][reader]
        if symbol == "-":
            continue
        expected = found["values"][symbol][reader]
        if got[reader] != expected:
            wrong[reader] = {"symbol": symbol, "expected": expected, "got": got[reader]}
    assert wrong == {}


def test_the_table_covers_every_cell() -> None:
    for phase in PHASES:
        for side in SIDES:
            assert set(EXPECTED[phase][side]) == set(READERS)
            assert "?" not in EXPECTED[phase][side].values()


def test_two_starts_onto_one_child_in_one_parent_step_run_in_turn() -> None:
    """A second start in the SAME parent step: the two delegations run one
    after the other, each records its own task row, and the child's records
    are the second's."""
    world = World()
    world.calls(("worker", {"task": "First task."}), ("worker", {"task": "Second task."}))
    world.text("First answer.")
    world.text("Second answer.")
    world.text("Root done.")

    response = world.runtime.send(world.root(), "Go")

    assert response.finish_reason == "stop"
    following = str(world.requests[-1].messages)
    assert "First answer." in following and "Second answer." in following
    child = world.fresh().child("worker")
    tasks = [
        (entry.run_id, entry.message.get("content"))
        for entry in child.thread().messages()
        if entry.message.get("type") == "user"
    ]
    assert [content for _, content in tasks] == ["First task.", "Second task."]
    assert len({run_id for run_id, _ in tasks}) == 2
    run = child.run()
    assert isinstance(run, dict) and run["id"] == tasks[-1][0]
    assert not child.child_provisional()
    assert child.child_binding() is not None
