"""G-387 round 11, item 7: THE INVARIANT MATRIX.

Every child-state reader, through every phase of a runner start, read from
INSIDE the start's own run (the Session object that run uses) and from
OUTSIDE it (a fresh Session over the same stores, a decision through
``record_approval``). The expected values are written first, as symbols, from
the rules of BRIEF-R7..R11; each symbol is then resolved per scenario:

- ``P`` prior: what the reader returned before the start began.
- ``C`` committed: what the start's commit (b) wrote (its mode ``work``, its
  binding, refs, depth and ceiling, its run).
- ``A`` absent: no binding, depth 0, no ceiling.
- ``K`` kept: the last CONFIRMED run tree and plan ledger (round 10), which
  for a child never started is a pair of nones.
- ``E`` ended: the call the start's task row ended reads ended, nothing is
  pending, and a decision on it is refused ``approval_not_pending``.

Phases: before the gate (read while the gate reads its facts); after (b)
(the start's window, before its task row); during the start's own run (its
first model call); after the task row (the run finished); after a failed start
whose undo landed; after a failed start whose undo failed too. For the last
three, "inside" reads the Session object the run used, after its run ended:
its own view is cleared with the run.

Every future fix must keep this green.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from functools import cache
from typing import Any

import pytest
from test_run_starts import direct
from test_run_starts_r8 import is_child
from test_run_starts_r9 import fail_the_undo, raise_from_the_child_tools, start
from test_step_allowance import World
from test_step_allowance_r6 import dead_approval_child

from prism_harness import HarnessError, RunBudget, RunContext, Session, record_approval
from prism_harness.plans.service import Plans
from prism_harness.thread_rows import ended_approval_ids, pending_approval_ids


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


READERS = ("mode", "binding", "refs", "depth", "ceiling", "run", "approval", "offer", "decision")
PHASES = (
    "before_gate",
    "after_b",
    "own_run",
    "after_task_row",
    "failed_undone",
    "failed_undo_failed",
)
SCENARIOS = ("fresh", "confirmed", "parked")


def _row(**cells: str) -> dict[str, str]:
    return {reader: cells.get(reader, cells.get("rest", "?")) for reader in READERS}


#: THE EXPECTED TABLE, written before the fix. ``-``: not read from that side
#: (a decision is an entry point, so always outside).
EXPECTED: dict[str, dict[str, dict[str, str]]] = {
    "before_gate": {
        "inside": _row(rest="P", decision="-"),
        "outside": _row(rest="P"),
    },
    "after_b": {
        "inside": _row(rest="C", approval="P", decision="-"),
        "outside": _row(
            mode="P",
            binding="A",
            refs="K",
            depth="A",
            ceiling="A",
            run="P",
            approval="P",
            offer="P",
            decision="P",
        ),
    },
    "own_run": {
        "inside": _row(rest="C", approval="E", decision="-"),
        "outside": _row(rest="C", approval="E", decision="E"),
    },
    "after_task_row": {
        "inside": _row(rest="C", approval="E", decision="-"),
        "outside": _row(rest="C", approval="E", decision="E"),
    },
    "failed_undone": {
        "inside": _row(rest="P", decision="-"),
        "outside": _row(rest="P"),
    },
    "failed_undo_failed": {
        "inside": _row(
            mode="P",
            binding="A",
            refs="K",
            depth="A",
            ceiling="A",
            run="P",
            approval="P",
            offer="P",
            decision="-",
        ),
        "outside": _row(
            mode="P",
            binding="A",
            refs="K",
            depth="A",
            ceiling="A",
            run="P",
            approval="P",
            offer="P",
            decision="P",
        ),
    },
}


def table_markdown() -> str:
    """The expected table as markdown, for the PR body."""
    lines = [
        "| phase | side | " + " | ".join(READERS) + " |",
        "|---|---|" + "---|" * len(READERS),
    ]
    for phase in PHASES:
        for side in ("inside", "outside"):
            cells = EXPECTED[phase][side]
            lines.append(f"| {phase} | {side} | " + " | ".join(cells[r] for r in READERS) + " |")
    return "\n".join(lines)


# -- the readers -------------------------------------------------------------------


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


def read(world: World, session: Session, side: str, approval: str | None) -> dict[str, Any]:
    """Every reader, on ``session``."""
    run = session.run()
    rows = [entry.message for entry in world.child("worker").thread().messages()]
    return {
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
        "decision": _decision(world.child("worker"), approval) if side == "outside" else "-",
    }


# -- the scenarios -------------------------------------------------------------------


def _scenario(name: str) -> tuple[World, str | None]:
    """A world whose worker child is in the scenario's state, and the approval
    it is parked on (or None)."""
    world = World()
    approval: str | None = None
    if name == "confirmed":
        assert "Child answer." in start(world, "direct", "First.")
        world.child("worker").using_mode("free")
    elif name == "parked":
        approval = dead_approval_child(world)
        world.child("worker").using_mode("free")
    world.fresh()
    world.queue.clear()
    return world, approval


@cache
def observed(scenario: str, outcome: str) -> dict[str, Any]:
    """One runner start onto the scenario's child: ``success``, ``undone``
    (fails after (b), its undo lands) or ``undo_failed``. What every reader
    returned, by phase and side, and the values each symbol stands for."""
    world, approval = _scenario(scenario)
    outside = world.child("worker")
    prior = read(world, outside, "outside", approval)
    reads: dict[str, dict[str, dict[str, Any]]] = {}
    own: list[Session] = []
    committed: dict[str, Any] = {}

    def both(phase: str, session: Session) -> None:
        reads[phase] = {
            "inside": read(world, session, "inside", approval),
            "outside": read(world, world.child("worker"), "outside", approval),
        }

    original_hold = Session.child_live_hold
    original_context = Plans.context

    reading = outcome == "success"

    def at_the_gate(self: Session, *args: Any) -> bool:
        if reading and is_child(self) and "before_gate" not in reads:
            both("before_gate", self)
        return original_hold(self, *args)

    def in_the_window(self: Plans, session: Session, prompt: str, *args: Any) -> str:
        if is_child(session) and not own:
            own.append(session)
            raw = session._durable.get(f"{session.key()}:child")  # a store reader
            state = session._ephemeral.get(session._ephemeral_key()) or {}
            assert isinstance(raw, dict)
            if not reading:
                return original_context(self, session, prompt, *args)
            committed.update(
                {
                    "binding": raw.get("child_binding"),
                    "refs": _refs(raw.get("child_refs")),
                    "depth": raw.get("child_depth"),
                    "ceiling": raw.get("child_ceiling"),
                    "run": state["run"]["id"],
                }
            )
            both("after_b", session)
            world.on_request = lambda: both("own_run", own[0])
        return original_context(self, session, prompt, *args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Session, "child_live_hold", at_the_gate)
        patch.setattr(Plans, "context", in_the_window)
        if outcome != "success":
            raise_from_the_child_tools(patch, lambda: OSError("down"))
        if outcome == "undo_failed":
            fail_the_undo(patch)
        world.text("Child answer.")
        result = direct(world, world.root(), RunContext.root("host", RunBudget(1)), "Matrix.")
    assert result.outcome == ("completed" if outcome == "success" else "failed")
    finished = {"success": "after_task_row", "undone": "failed_undone"}.get(
        outcome, "failed_undo_failed"
    )
    both(finished, own[0])

    kept = prior["refs"] if prior["refs"] is not None else {"tree": None, "plan_ledger": None}
    values = {
        "P": prior,
        "A": {"binding": None, "depth": 0, "ceiling": None},
        "K": {"refs": kept},
        "E": {
            "approval": ((), approval is not None),
            "decision": "approval_not_pending",
        },
        "C": {
            **committed,
            "mode": "work",
            "offer": sorted(
                tool["name"] for tool in world.runtime.effective_tools(outside, "work")
            ),
        },
    }
    return {"reads": reads, "values": values}


_RUN_OF = {
    "before_gate": "success",
    "after_b": "success",
    "own_run": "success",
    "after_task_row": "success",
    "failed_undone": "undone",
    "failed_undo_failed": "undo_failed",
}


@pytest.mark.parametrize("side", ["inside", "outside"])
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_invariant_matrix(scenario: str, phase: str, side: str) -> None:
    found = observed(scenario, _RUN_OF[phase])
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
        for side in ("inside", "outside"):
            assert set(EXPECTED[phase][side]) == set(READERS)
            assert "?" not in EXPECTED[phase][side].values()
