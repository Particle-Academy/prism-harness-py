"""The durable half of a run tree's budget, for subagent children resumed
outside the runner.

A child that stops on an approval is resumed later on its own session, in a run
the runner did not start. That run must not get a fresh root budget: it
continues the run tree it belonged to. While the tree runs, its budget is the
in-memory :class:`~prism_harness.subagents.RunLedger` every run in it shares.
The first time a root run starts a child, the tree's allowance is written to
the durable store as a ledger record (the shape a shared plan budget keeps)
with the root run holding all of it; when the root run ends it settles what the
whole tree spent. A resumed child then holds what the tree has left before it
runs, settles what it spent when it ends, and is refused once the tree's
allowance is spent. A hold whose run never settled is charged in full once the
session lock's TTL has passed, so a crash can only over-charge the tree.

The tree a child belongs to is saved in the child session's durable half when
the runner starts it (:meth:`Session.child_refs`), so EVERY turn on the child
the runner did not start reads it: an answer to its approval, a new message
instead of one, or a run context the host built, which is narrowed to what the
tree has left. It also travels on the paused approval row (``harness_tree``),
read for a child paused before the durable record existed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from prism_harness.errors import HarnessError
from prism_harness.plans import budget
from prism_harness.stores.base import SessionStore
from prism_harness.subagents import RunContext
from prism_harness.thread_rows import paused_approval_row

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_harness.session import Session

__all__ = ["RunTrees", "refuse_spent_tree", "saved_refs", "saved_tree"]

_log = logging.getLogger("prism_harness.run_trees")

Tree = dict[str, str]


class RunTrees:
    """The root runs of one runtime, by run id, until they end."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        #: The tree each runner-started child run belongs to, until it ends.
        self._children: dict[str, Tree] = {}

    def context_kind(self, context: RunContext | None) -> str:
        """What kind of run context a turn was handed, for
        :func:`~prism_harness.plans.budget.route`: ``runner`` when it is a
        context of a run this runtime is running (its ledger is that run's),
        ``host`` for any other context."""
        if context is None:
            return "none"
        entry = self._runs.get(context.root_run_id)
        return "runner" if entry is not None and entry["ledger"] is context.ledger else "host"

    def root_run(
        self,
        session: Session,
        run_id: str,
        run: RunContext,
        context: RunContext | None,
        tree: Tree | None,
        hold_ttl: int,
    ) -> RunContext:
        """A run is starting. A turn on a child session the runner did not
        start (an answer to its approval, a new message, or a run context the
        host built) continues the tree it was saved with (``tree``): it holds
        what that tree has left, and its run is narrowed to the hold, so a host
        context can only lower what the tree allows. Any other root run opens a
        tree of its own, kept in memory until it starts a child. A run on the
        runner's own context already spends from its tree and is left alone.

        :raises HarnessError: ``run_not_permitted`` when that tree is spent.
        """
        kind = budget.route(self.context_kind(context), "", False, False, tree is not None, False)
        if kind["tree"] == "none":
            return run
        if kind["tree"] == "open":
            tree = None
        ledger = run.ledger
        entry: dict[str, Any] = {
            "session": session.key(),
            "tree": tree if tree is not None else {"session": session.key(), "run": run_id},
            "budget": run.budget,
            "ledger": ledger,
            "steps": ledger.steps,
            "cost": ledger.cost_usd,
            "unmetered": ledger.unmetered_runs,
            "held": False,
            "resumed": tree is not None,
        }
        if tree is None:
            self._runs[run_id] = entry
            return run

        own = run.budget
        turn_cost = (
            None
            if own.max_cost_usd is None
            else budget.micros(max(0.0, own.max_cost_usd - entry["cost"]))
        )
        now = int(time.time())

        def hold(state: dict[str, Any] | None) -> tuple[dict[str, Any] | None, Any]:
            if state is None:
                # The tree's record is gone: closed, never a fresh budget.
                return None, {"refused": budget.EXHAUSTED, "limit": "steps"}
            return budget.reserve(
                state, run_id, max(0, own.max_steps - entry["steps"]), turn_cost, now, hold_ttl
            )

        grant = _write(session.plan_store(), tree, hold)
        if "refused" in grant:
            raise HarnessError.run_not_permitted(f"This run may not proceed: {grant['limit']}.")
        narrowed = replace(
            run,
            budget=replace(
                own,
                max_steps=entry["steps"] + int(grant["steps"]),
                max_cost_usd=own.max_cost_usd
                if grant["cost_micros"] is None
                else entry["cost"] + grant["cost_micros"] / 1_000_000,
            ),
        )
        self._runs[run_id] = {**entry, "budget": narrowed.budget, "held": True}
        return narrowed

    def child_started(
        self, session: Session, run_id: str, context: RunContext, hold_ttl: int
    ) -> Tree | None:
        """A runner-started child is starting under a run tree: the tree it
        belongs to, opened in the durable store the first time its root starts
        a child. None when the tree's root is not a run of this runtime (a
        context the host built itself)."""
        root_run_id = context.root_run_id
        entry = self._runs.get(root_run_id)
        if entry is None:
            return None
        if not entry["held"]:
            own = entry["budget"]
            settings = {
                "mode": "shared",
                "max_steps": max(0, own.max_steps - entry["steps"]),
                "max_cost_micros": None
                if own.max_cost_usd is None
                else budget.micros(max(0.0, own.max_cost_usd - entry["cost"])),
            }
            now = int(time.time())

            def open_and_hold(state: dict[str, Any] | None) -> tuple[dict[str, Any] | None, Any]:
                current = state if state is not None else budget.open_ledger(root_run_id, settings)
                return budget.reserve(
                    current,
                    root_run_id,
                    max(1, int(settings["max_steps"])),
                    settings["max_cost_micros"],
                    now,
                    hold_ttl,
                )

            _write(session.plan_store(), entry["tree"], open_and_hold)
            entry["held"] = True
        tree: Tree = dict(entry["tree"])
        self._children[run_id] = tree
        return tree

    def tree_of(self, run_id: str) -> Tree | None:
        """The tree a child run's paused row carries: the one the runner started
        it in, or the one a resumed child continues."""
        if run_id in self._children:
            return dict(self._children[run_id])
        entry = self._runs.get(run_id)
        return dict(entry["tree"]) if entry is not None and entry["resumed"] else None

    def holds(self, run_id: str) -> bool:
        """Whether this run spends from a durable run tree record."""
        entry = self._runs.get(run_id)
        return entry is not None and entry["held"] is True

    def finished(self, session: Session, run_id: str) -> None:
        """A run is over: a root run that holds part of a tree settles what the
        tree spent under it."""
        self._children.pop(run_id, None)
        entry = self._runs.get(run_id)
        if entry is None or entry["session"] != session.key():
            return
        del self._runs[run_id]
        if not entry["held"]:
            return
        ledger = entry["ledger"]
        steps = ledger.steps - entry["steps"]
        cost = (
            None
            if ledger.unmetered_runs > entry["unmetered"]
            else budget.micros(max(0.0, ledger.cost_usd - entry["cost"]))
        )

        def settle(state: dict[str, Any] | None) -> tuple[dict[str, Any] | None, Any]:
            return (
                (None, None) if state is None else (budget.settle(state, run_id, steps, cost), None)
            )

        try:
            _write(session.plan_store(), entry["tree"], settle)
        except Exception as failure:  # noqa: BLE001 - the hold is charged in full once it expires
            _log.warning("A run tree budget update failed: %s.", type(failure).__name__)


def saved_tree(row: Mapping[str, Any] | None) -> Tree | None:
    """The run tree a paused row says its child belongs to, or None."""
    saved = row.get("harness_tree") if row is not None else None
    if (
        isinstance(saved, Mapping)
        and isinstance(saved.get("session"), str)
        and isinstance(saved.get("run"), str)
    ):
        return {"session": saved["session"], "run": saved["run"]}
    return None


def saved_refs(session: Session) -> tuple[Tree | None, dict[str, str] | None]:
    """The run tree and shared plan budget a child session was saved with: its
    durable record, or, for a child paused before that record existed, its
    paused row."""
    refs = session.child_refs()
    if refs is not None:
        return refs["tree"], refs["plan_ledger"]
    row = paused_approval_row([entry.message for entry in session.thread().messages()])
    return saved_tree(row), _saved_plan_ledger(row)


def _saved_plan_ledger(row: Mapping[str, Any] | None) -> dict[str, str] | None:
    saved = row.get("harness_plan_ledger") if row is not None else None
    if (
        isinstance(saved, Mapping)
        and isinstance(saved.get("session"), str)
        and isinstance(saved.get("plan"), str)
    ):
        return {"session": saved["session"], "plan": saved["plan"]}
    return None


def refuse_spent_tree(session: Session, now: int, hold_ttl: int) -> None:
    """Refuse, before anything is recorded, a decision on a child session whose
    run tree has nothing left: recorded, it would be consumed by a resume the
    tree then refuses, and the approved call would never run.

    :raises HarnessError: ``run_not_permitted``.
    """
    tree, _ = saved_refs(session)
    # A decision takes no context: it resumes as a turn with none.
    if tree is None or budget.route("none", "", False, False, True, False)["tree"] != "continue":
        return
    state = session.plan_store().get(_key(tree))
    if not isinstance(state, dict) or state.get("planId") != tree["run"]:
        raise HarnessError.run_not_permitted("This run may not proceed: steps.")
    _, grant = budget.reserve(state, "decision", 1, None, now, hold_ttl)
    if "refused" in grant:
        raise HarnessError.run_not_permitted(f"This run may not proceed: {grant['limit']}.")


def _write(
    store: SessionStore,
    tree: Tree,
    change: Callable[[dict[str, Any] | None], tuple[dict[str, Any] | None, Any]],
) -> Any:
    """One locked change to a tree's record."""
    key = _key(tree)

    def swap() -> Any:
        stored = store.get(key)
        state = stored if isinstance(stored, dict) and stored.get("planId") == tree["run"] else None
        following, result = change(state)
        if following is not None:
            store.put(key, following)
        return result

    return store.with_lock(key, swap, 10, 5)


def _key(tree: Tree) -> str:
    return f"{tree['session']}:run-tree:{tree['run']}"
