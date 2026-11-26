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
import weakref
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

__all__ = [
    "CONTEXT_KINDS",
    "RunTrees",
    "kept_refs",
    "membership",
    "refuse_spent_tree",
    "saved_refs",
    "saved_tree",
]

_log = logging.getLogger("prism_harness.run_trees")

Tree = dict[str, str]

#: The kinds of run context a turn can be handed.
CONTEXT_KINDS = ("none", "metadata", "runner", "host")


def membership(
    kind: str,
    session: str,
    run: str,
    parent: Mapping[str, Any] | None,
    saved_tree: Mapping[str, str] | None,
    saved_depth: int,
    context_depth: int,
    resumed_tree: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Which run tree a run belongs to (G-381): decided ONCE, when the run
    begins, and the one record every consumer reads. Pure, so every language
    answers it the same way (prism-parity harness-plans).

    - The runner's own context (``runner``) takes the recorded membership of
      the live parent run it was started from, verified by the runtime: the
      same tree, the same run it spends from, one level deeper. With no
      verified parent it is in no tree, at the depth the context states. The
      tree its session saved is never read: that is for turns the runner did
      not start.
    - Any other turn on a session that saved a tree continues that tree,
      spending from a hold of its own, at least as deep as it was started.
    - A host context on a session that saved none is in the host's tree.
    - No context, or a metadata-only one, opens a tree: its own, or, for a
      root resuming a run parked on approvals (``resumed_tree``), the tree
      that parked run began in, so a child it starts after the resume belongs
      to the same tree as the children it started before.

    A host context's own root and parent run ids are deliberately not inputs:
    they are the host's statement, not a fact this runtime holds.

    :raises ValueError: for an unknown context kind.
    """
    if kind not in CONTEXT_KINDS:
        raise ValueError(f"Unknown run context kind [{kind}].")
    saved_depth = max(0, saved_depth)
    context_depth = max(0, context_depth)
    if kind == "runner":
        if parent is None:
            return {"tree": "none", "spends": None, "depth": context_depth, "route": "none"}
        tree = parent["tree"]
        return {
            "tree": dict(tree) if isinstance(tree, Mapping) else tree,
            "spends": parent["spends"],
            "depth": max(0, parent["depth"]) + 1,
            "route": "none",
        }
    if saved_tree is not None:
        return {
            "tree": {"session": saved_tree["session"], "run": saved_tree["run"]},
            "spends": run,
            "depth": max(saved_depth, context_depth) if kind == "host" else saved_depth,
            "route": "continue",
        }
    if kind == "host":
        return {
            "tree": "host",
            "spends": None,
            "depth": max(saved_depth, context_depth),
            "route": "none",
        }
    return {
        "tree": {"session": session, "run": run}
        if resumed_tree is None
        else {"session": resumed_tree["session"], "run": resumed_tree["run"]},
        "spends": run,
        "depth": saved_depth,
        "route": "open",
    }


def kept_refs(
    membership: Mapping[str, Any],
    plan_ledger: Mapping[str, str] | None,
    saved_tree: Mapping[str, str] | None,
    saved_plan_ledger: Mapping[str, str] | None,
    saved_tree_cancelled: bool,
) -> dict[str, Any]:
    """The run tree and plan ledger the runner saves with a child (G-381): the
    tree it is in, with the plan ledger of the run it spends from. A child in
    the host's tree, or in none, has no durable tree to name, so what it saved
    before is kept, never replaced with nothing, unless a person cancelled that
    tree: starting the child again is then a new delegation."""
    tree = membership["tree"]
    if isinstance(tree, Mapping):
        return {
            "tree": {"session": tree["session"], "run": tree["run"]},
            "plan_ledger": None if plan_ledger is None else dict(plan_ledger),
        }
    if saved_tree_cancelled:
        return {"tree": None, "plan_ledger": None}
    return {
        "tree": None if saved_tree is None else dict(saved_tree),
        "plan_ledger": None if saved_plan_ledger is None else dict(saved_plan_ledger),
    }


class RunTrees:
    """The root runs of one runtime, by run id, until they end, and the run
    tree every live run belongs to."""

    def __init__(self) -> None:
        #: The budget entries of the runs that open or continue a tree.
        self._runs: dict[str, dict[str, Any]] = {}
        #: The membership of every live run, by run id, with its session key
        #: and its run ledger, recorded once when it begins (G-381).
        self.members: dict[str, dict[str, Any]] = {}
        #: The child contexts the subagent runner built, held weakly and by
        #: identity, and the run and session of the turn that started each.
        self._child_contexts: weakref.WeakValueDictionary[int, RunContext] = (
            weakref.WeakValueDictionary()
        )
        self._parents: dict[int, tuple[str | None, str | None]] = {}

    def starting_child(
        self,
        context: RunContext,
        parent_run_id: str | None = None,
        parent_session_key: str | None = None,
    ) -> RunContext:
        """The subagent runner is starting a child under ``context``, which it
        built, from the ACTIVE turn ``parent_run_id`` on the session
        ``parent_session_key`` (both None outside a turn). Marks it as the
        runner's own, so the child's binding is saved with its session even
        under a run tree the host built.

        Internal: called by the subagent runner only.
        """
        key = id(context)
        if self._child_contexts.get(key) is not context:
            self._child_contexts[key] = context
            weakref.finalize(context, self._parents.pop, key, None)
        self._parents[key] = (parent_run_id, parent_session_key)
        return context

    def binding_context_kind(self, context: RunContext | None) -> str:
        """The kind of context a turn was handed, for
        :func:`~prism_harness.subagents.binding_route`: the same kind as
        :meth:`context_kind`, so the binding and the budget routes read one
        context the same way."""
        return self.context_kind(context)

    def context_kind(self, context: RunContext | None) -> str:
        """What kind of run context a turn was handed: ``none`` without one,
        ``runner`` for the context the subagent runner built to start a child,
        ``host`` for any other. A context the host built is the host's,
        whatever ledger it carries."""
        if context is None:
            return "none"
        return "runner" if self._child_contexts.get(id(context)) is context else "host"

    def membership_of(
        self, session: Session, run_id: str, context: RunContext | None, prompt: str = ""
    ) -> dict[str, Any]:
        """The membership a run on ``session`` under ``context`` has:
        :func:`membership`, with the verified parent of a runner context and,
        for a bare turn resuming a parked run, the tree that run began in."""
        kind = self.context_kind(context)
        bare = kind in ("none", "metadata")
        tree, _ = saved_refs(session)
        return membership(
            kind,
            session.key(),
            run_id,
            self._verified_parent(context) if kind == "runner" else None,
            tree,
            session.child_depth(),
            0 if bare or context is None else context.depth,
            session.resumed_run_tree(prompt) if bare else None,
        )

    def _verified_parent(self, context: RunContext | None) -> dict[str, Any] | None:
        """The recorded membership of the run a runner context was started
        from, accepted only when that live run is on the session the mark names
        and its run ledger is the context's very ledger: proof the context came
        from that run's active turn."""
        if context is None or self._child_contexts.get(id(context)) is not context:
            return None
        parent_run, parent_session = self._parents.get(id(context), (None, None))
        member = None if parent_run is None else self.members.get(parent_run)
        if (
            member is None
            or member["session"] != parent_session
            or member["ledger"] is not context.ledger
        ):
            return None
        return {"tree": member["tree"], "spends": member["spends"], "depth": member["depth"]}

    def record(
        self, session: Session, run_id: str, member: Mapping[str, Any], run: RunContext
    ) -> None:
        """Record a live run's membership, until :meth:`finished`, with its
        own step ceiling (before any tree narrows it)."""
        self.members[run_id] = {
            **member,
            "session": session.key(),
            "ledger": run.ledger,
            "own_max_steps": run.budget.max_steps,
        }

    def member(self, run_id: str) -> dict[str, Any] | None:
        """The recorded membership of a live run, or None."""
        found = self.members.get(run_id)
        return None if found is None else dict(found)

    def root_run(
        self,
        session: Session,
        run_id: str,
        run: RunContext,
        member: Mapping[str, Any],
        hold_ttl: int,
        steps_left: int | None = None,
    ) -> RunContext:
        """A run is starting, in the tree its membership names. ``open``
        registers a budget entry of its own, kept in memory until it starts a
        child. ``continue`` (a turn on a child session the runner did not
        start: an answer, a new message, or a host context) holds what the
        saved tree has left, and its run is narrowed to the hold, so a host
        context can only lower what the tree allows. Any other run is left
        alone.

        :raises HarnessError: ``run_not_permitted`` when that tree is spent.
        """
        route = member["route"]
        if route not in ("open", "continue"):
            return run
        tree: Tree = {"session": member["tree"]["session"], "run": member["tree"]["run"]}
        # A root resuming its parked run is in the tree that run began, and
        # is held to it as any other turn continuing the tree, so the tree's
        # allowance holds across the pause: it holds what the tree's record
        # has left and is narrowed to it. When that run never delegated, the
        # record does not exist yet and is opened here, in the same locked
        # change, so a second resume of the same parked run that starts
        # meanwhile finds it and is bounded by it. An open run whose tree
        # already has a record (whatever its run id) is held to it too.
        # A run whose own step budget is already spent takes no step, so it
        # is left as it was: it still acts on the person's decisions, then
        # stops on its own budget, with the same outcome and message as
        # before.
        # "Own steps left" is ONE measure, the one the decision check read:
        # what the paused row says the parked run had left of its own budget
        # (its ledger, a runner child's steps included), when it says so.
        resumed = route == "open" and tree["run"] != run_id
        own_left = run.budget.max_steps - run.ledger.steps
        if route == "open" and steps_left is not None:
            own_left = min(own_left, steps_left)
        if route == "open" and (resumed or _record_of(session, tree) is not None):
            if own_left >= 1:
                route = "continue"
            elif run.budget.max_steps > run.ledger.steps:
                # Spent by that measure: it acts on the decisions and takes
                # no further step.
                run = replace(run, budget=replace(run.budget, max_steps=run.ledger.steps))
        ledger = run.ledger
        entry: dict[str, Any] = {
            "session": session.key(),
            "tree": tree,
            "budget": run.budget,
            "ledger": ledger,
            "steps": ledger.steps,
            "cost": ledger.cost_usd,
            "unmetered": ledger.unmetered_runs,
            "held": False,
        }
        if route == "open":
            self._runs[run_id] = entry
            return run

        own = run.budget
        turn_cost = (
            None
            if own.max_cost_usd is None
            else budget.micros(max(0.0, own.max_cost_usd - entry["cost"]))
        )
        now = int(time.time())

        left = max(0, own.max_steps - entry["steps"])

        def hold(state: dict[str, Any] | None) -> tuple[dict[str, Any] | None, Any]:
            if state is None:
                if not resumed:
                    # The tree's record is gone: closed, never a fresh budget.
                    return None, {"refused": budget.EXHAUSTED, "limit": "steps"}
                # A resumed root whose parked run never delegated opens its
                # tree's record, named by the tree's run, with what it has left.
                state = budget.open_ledger(
                    tree["run"],
                    {"mode": "shared", "max_steps": left, "max_cost_micros": turn_cost},
                )
            return budget.reserve(state, run_id, left, turn_cost, now, hold_ttl)

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
        self, session: Session, member: Mapping[str, Any], hold_ttl: int
    ) -> Tree | None:
        """A runner-started child is starting: the tree its membership names,
        opened in the durable store, for the budget entry of the run it spends
        from, the first time that run starts a child. None when it is in the
        host's tree or in none."""
        tree = member["tree"]
        if not isinstance(tree, Mapping):
            return None
        spends = member["spends"]
        entry = self._runs.get(spends) if isinstance(spends, str) else None
        if isinstance(spends, str) and entry is not None and not entry["held"]:
            own = entry["budget"]
            settings = {
                "mode": "shared",
                "max_steps": max(0, own.max_steps - entry["steps"]),
                "max_cost_micros": None
                if own.max_cost_usd is None
                else budget.micros(max(0.0, own.max_cost_usd - entry["cost"])),
            }
            now = int(time.time())
            holder: str = spends
            # The record is the tree's: named by the tree's run, which differs
            # from the run that holds from it for a resumed root.
            plan_id: str = tree["run"]

            def open_and_hold(state: dict[str, Any] | None) -> tuple[dict[str, Any] | None, Any]:
                # Named by the tree's run; held under the run that spends. A
                # resumed root holds from its tree in root_run, never here.
                current = state if state is not None else budget.open_ledger(plan_id, settings)
                return budget.reserve(
                    current,
                    holder,
                    max(1, int(settings["max_steps"])),
                    settings["max_cost_micros"],
                    now,
                    hold_ttl,
                )

            grant = _write(session.plan_store(), entry["tree"], open_and_hold)
            if "refused" in grant:
                # The tree cannot fund it: the delegation is refused rather
                # than run on a hold that was never taken.
                raise HarnessError.run_not_permitted(f"This run may not proceed: {grant['limit']}.")
            entry["held"] = True
        return {"session": tree["session"], "run": tree["run"]}

    def held_max_steps(self, run_id: str) -> int | None:
        """The step ceiling a run holding from its tree's record was narrowed
        to, or None for a run that holds nothing."""
        entry = self._runs.get(run_id)
        if entry is None or entry["held"] is not True:
            return None
        return int(entry["budget"].max_steps)

    def holds(self, run_id: str) -> bool:
        """Whether this run spends from a durable run tree record."""
        entry = self._runs.get(run_id)
        return entry is not None and entry["held"] is True

    def finished(self, session: Session, run_id: str) -> None:
        """A run is over: its membership is dropped, and a root run that holds
        part of a tree settles what the tree spent under it."""
        member = self.members.get(run_id)
        if member is not None and member["session"] == session.key():
            del self.members[run_id]
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
    saved, _ = saved_refs(session)
    # A decision takes no context: it resumes as a turn with none and no new
    # message, and that turn's would-be membership says whether it is held to
    # a tree's record.
    would_be = membership(
        "none",
        session.key(),
        "decision",
        None,
        saved,
        session.child_depth(),
        0,
        session.resumed_run_tree(""),
    )
    tree = would_be["tree"]
    if not isinstance(tree, dict):
        return
    state = _record_of(session, tree)
    if would_be["route"] == "open":
        # A root resuming its parked run is held to its tree's record only
        # when one exists, and only when its own step budget has steps left:
        # a resume that can take no step only acts on the decisions.
        if tree["run"] == "decision" or state is None:
            return
        row = paused_approval_row([entry.message for entry in session.thread().messages()])
        steps_left = row.get("harness_steps_left") if row is not None else None
        if type(steps_left) is int and steps_left < 1:
            return
    elif would_be["route"] != "continue":
        return
    if state is None:
        raise HarnessError.run_not_permitted("This run may not proceed: steps.")
    _, grant = budget.reserve(state, "decision", 1, None, now, hold_ttl)
    if "refused" in grant:
        raise HarnessError.run_not_permitted(f"This run may not proceed: {grant['limit']}.")


def _record_of(session: Session, tree: Tree) -> dict[str, Any] | None:
    """The durable budget record of a run tree, or None when it has none."""
    state = session.plan_store().get(_key(tree))
    return state if isinstance(state, dict) and state.get("planId") == tree["run"] else None


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
