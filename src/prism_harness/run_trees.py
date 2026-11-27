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
runs and settles what it spent when it ends. Once the tree's allowance is spent
it is granted no model step (never refused: it still acts on the decisions
recorded for it, G-387). A hold whose run never settled is charged in full once
the session lock's TTL has passed, so a crash can only over-charge the tree.

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

from prism_harness.plans import budget
from prism_harness.run_starts import StartRefused
from prism_harness.stores.base import SessionStore
from prism_harness.subagents import RunBudget, RunContext
from prism_harness.thread_rows import paused_approval_row

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_harness.session import Session

__all__ = [
    "CONTEXT_KINDS",
    "RunTrees",
    "kept_refs",
    "membership",
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
        #: What each of those contexts was built to start: the raw task, the
        #: child's mode and the host contexts whose stops the start obeys.
        self._starts: dict[int, dict[str, Any]] = {}

    def starting_child(
        self,
        context: RunContext,
        parent_run_id: str | None = None,
        parent_session_key: str | None = None,
        start: Mapping[str, Any] | None = None,
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
            weakref.finalize(context, self._starts.pop, key, None)
        self._parents[key] = (parent_run_id, parent_session_key)
        self._starts[key] = dict(start or {})
        return context

    def start_of(self, context: RunContext | None) -> dict[str, Any] | None:
        """What a runner context was built to start (``task``, ``mode``,
        ``hosts``), or None for any other context."""
        if context is None or self._child_contexts.get(id(context)) is not context:
            return None
        return dict(self._starts.get(id(context)) or {})

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
        """Record a live run's membership, until :meth:`finished`."""
        self.members[run_id] = {
            **member,
            "session": session.key(),
            "ledger": run.ledger,
        }

    def allot(
        self, run_id: str, allowance: Mapping[str, Any], begin_steps: int, run: RunBudget
    ) -> None:
        """Record a live run's step allowance (G-387), worked out once when it
        began, beside its membership: the allowance, where its ledger stood
        then, and the budget it was narrowed to. Every consumer of "how many
        steps this run may still take" reads this record."""
        member = self.members.get(run_id)
        if member is not None:
            member["allowance"] = dict(allowance)
            member["begin_steps"] = begin_steps
            member["budget"] = run

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
        own_left: int,
    ) -> tuple[RunContext, int | None]:
        """A run is starting, in the tree its membership names: the run, with
        its cost narrowed to what it holds, and the steps the tree's record
        grants it (``tree_left`` of its allowance), or None when no durable
        record bounds it.

        ``open`` registers a budget entry of its own, kept in memory until it
        starts a child. ``continue`` (a turn on a child session the runner did
        not start: an answer, a new message, or a host context) holds up to
        ``own_left`` steps from the saved tree's record, in one locked write.
        A root resuming its parked run is held to the record of the tree that
        run began in the same way; when that run never delegated, the record
        is opened here, in the same locked change, so a second resume of the
        same parked run finds it and is bounded by it.

        A tree whose steps or cost are spent, or a record that is gone, grants
        0 steps and is never a refusal here: a resume still acts on the
        decisions recorded for it, which makes no provider call (G-387). The
        runtime refuses a turn with a new prompt and nothing left.
        """
        route = member["route"]
        if route not in ("open", "continue"):
            return run, None
        tree: Tree = {"session": member["tree"]["session"], "run": member["tree"]["run"]}
        resumed = route == "open" and tree["run"] != run_id
        if route == "open" and (resumed or _record_of(session, tree) is not None):
            route = "continue"
        ledger = run.ledger
        entry: dict[str, Any] = {
            "session": session.key(),
            "tree": tree,
            "ledger": ledger,
            "steps": ledger.steps,
            "cost": ledger.cost_usd,
            "unmetered": ledger.unmetered_runs,
            "held": False,
        }
        if route == "open":
            self._runs[run_id] = entry
            return run, None

        own = run.budget
        turn_cost = (
            None
            if own.max_cost_usd is None
            else budget.micros(max(0.0, own.max_cost_usd - entry["cost"]))
        )
        now = int(time.time())
        asked = max(0, own_left)

        def hold(state: dict[str, Any] | None) -> tuple[dict[str, Any] | None, Any]:
            if state is None:
                if not resumed:
                    # The tree's record is gone: closed, never a fresh budget.
                    return None, {"steps": 0, "cost_micros": None}
                # A resumed root whose parked run never delegated opens its
                # tree's record, named by the tree's run, with what it has left.
                state = budget.open_ledger(
                    tree["run"],
                    {"mode": "shared", "max_steps": asked, "max_cost_micros": turn_cost},
                )
            if asked < 1:
                return None, {"steps": 0, "cost_micros": None}
            following, grant = budget.reserve(state, run_id, asked, turn_cost, now, hold_ttl)
            if "refused" in grant:
                # Spent, on steps or on cost: 0 steps and no hold.
                return None, {"steps": 0, "cost_micros": None}
            return following, grant

        grant = _write(session.plan_store(), tree, hold)
        narrowed = (
            run
            if grant["cost_micros"] is None
            else replace(
                run,
                budget=replace(own, max_cost_usd=entry["cost"] + grant["cost_micros"] / 1_000_000),
            )
        )
        # Held, whatever was granted: a run continuing a tree never opens a
        # second record for it, and settling a hold that was never taken
        # changes nothing.
        self._runs[run_id] = {**entry, "held": True}
        return narrowed, int(grant["steps"])

    def _start_hold(self, member: Mapping[str, Any]) -> dict[str, Any] | None:
        """The tree hold a runner start would take: the first time the run it
        spends from starts a child, that run's whole allowance is held from the
        tree's record. None when it is in the host's tree or in none, or the
        spending run already holds."""
        tree = member["tree"]
        if not isinstance(tree, Mapping):
            return None
        spends = member["spends"]
        entry = self._runs.get(spends) if isinstance(spends, str) else None
        if not isinstance(spends, str) or entry is None or entry["held"]:
            return None
        spender = self.members.get(spends) or {}
        allotted = spender.get("allowance")
        steps = int(allotted["allowance"]) if isinstance(allotted, Mapping) else 0
        narrowed = spender.get("budget")
        cost = (
            None
            if not isinstance(narrowed, RunBudget) or narrowed.max_cost_usd is None
            else budget.micros(max(0.0, narrowed.max_cost_usd - entry["cost"]))
        )
        return {
            "entry": entry,
            "holder": spends,
            # The record is the tree's: named by the tree's run, which differs
            # from the run that holds from it for a resumed root.
            "plan_id": tree["run"],
            "settings": {"mode": "shared", "max_steps": steps, "max_cost_micros": cost},
            "asked": max(1, steps),
            "cost": cost,
        }

    def start_grant(self, session: Session, member: Mapping[str, Any], hold_ttl: int) -> int | None:
        """The steps the tree's record would grant a runner start NOW, read
        without a write (G-387 start gate): 0 when it cannot fund it, None when
        no tree hold applies to the start."""
        hold = self._start_hold(member)
        if hold is None:
            return None
        state = _record_of(session, hold["entry"]["tree"])
        current = (
            state if state is not None else budget.open_ledger(hold["plan_id"], hold["settings"])
        )
        _, grant = budget.reserve(
            current, hold["holder"], hold["asked"], hold["cost"], int(time.time()), hold_ttl
        )
        return 0 if "refused" in grant else int(grant["steps"])

    def child_started(
        self,
        session: Session,
        member: Mapping[str, Any],
        hold_ttl: int,
        expected: int | None = None,
        run_id: str | None = None,
    ) -> Tree | None:
        """A runner-started child is starting: the tree its membership names,
        opened in the durable store, for the budget entry of the run it spends
        from, the first time that run starts a child. None when it is in the
        host's tree or in none.

        The record's step allowance is the one the spending run recorded when
        it began (:meth:`allot`), held in full by that run until it settles.
        A conditional write: when the record grants fewer than ``expected``
        steps (what :meth:`start_grant` read), nothing is stored and the start
        is refused (:class:`~prism_harness.run_starts.StartRefused`, ``tree``).

        The hold records ``run_id``, the start that took it: only that start
        releases it (:meth:`release_start`) or hands it over (:meth:`committed`).
        """
        tree = member["tree"]
        if not isinstance(tree, Mapping):
            return None
        hold = self._start_hold(member)
        if hold is not None:
            now = int(time.time())
            entry = hold["entry"]
            created = [False]

            def open_and_hold(state: dict[str, Any] | None) -> tuple[dict[str, Any] | None, Any]:
                # Named by the tree's run; held under the run that spends. A
                # resumed root holds from its tree in root_run, never here.
                created[0] = state is None
                current = (
                    state
                    if state is not None
                    else budget.open_ledger(hold["plan_id"], hold["settings"])
                )
                following, grant = budget.reserve(
                    current, hold["holder"], hold["asked"], hold["cost"], now, hold_ttl
                )
                if "refused" in grant or (expected is not None and int(grant["steps"]) < expected):
                    # A race since the read shrank the grant: nothing stored.
                    return None, None
                return following, grant

            grant = _write(session.plan_store(), entry["tree"], open_and_hold)
            if grant is None:
                raise StartRefused("tree", "exhausted")
            entry["held"] = True
            entry["start_hold"] = {"created": created[0], "by": run_id}
        return {"session": tree["session"], "run": tree["run"]}

    def release_start(
        self, session: Session, member: Mapping[str, Any], run_id: str | None = None
    ) -> None:
        """Undo the tree hold a runner start took in :meth:`child_started`,
        when a later part of its commit failed: the record goes back to what it
        was (removed, when the start opened it), and the spending run holds
        nothing, so the next child it starts takes the hold again. Only a hold
        THIS start (``run_id``) took is released, and only once: a re-entrant
        start that failed never releases the hold of the start it re-entered."""
        spends = member.get("spends")
        entry = self._runs.get(spends) if isinstance(spends, str) else None
        if entry is None or not isinstance(entry.get("start_hold"), dict):
            return
        if entry["start_hold"].get("by") != run_id:
            return
        taken = entry.pop("start_hold")
        entry["held"] = False
        holder = str(spends)
        store = session.plan_store()
        key = _key(entry["tree"])
        plan_id = entry["tree"]["run"]

        def undo() -> None:
            stored = store.get(key)
            if not isinstance(stored, dict) or stored.get("planId") != plan_id:
                return
            holds = dict(stored.get("holds") or {})
            holds.pop(holder, None)
            if taken["created"] and not holds and stored.get("steps") == 0:
                store.forget(key)
                return
            store.put(key, {**stored, "holds": holds})

        try:
            store.with_lock(key, undo, 10, 5)
        except Exception as failure:  # noqa: BLE001 - the hold is charged in full once it expires
            _log.warning("A run tree hold release failed: %s.", _label(failure))

    def committed(self, member: Mapping[str, Any], run_id: str | None = None) -> None:
        """A runner start's commit finished (its task is recorded): the tree
        hold it took is the spending run's from now on, settled when that run
        ends. Only the start that took the hold (``run_id``) hands it over."""
        spends = member.get("spends")
        entry = self._runs.get(spends) if isinstance(spends, str) else None
        taken = entry.get("start_hold") if entry is not None else None
        if entry is not None and isinstance(taken, dict) and taken.get("by") == run_id:
            entry.pop("start_hold", None)

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
            _log.warning("A run tree budget update failed: %s.", _label(failure))


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


def _label(failure: BaseException) -> str:
    """A class named by the identifier rule (G-387 rounds 10 and 11): never a
    forged name carrying a path. Imported here, not at the top: the failures
    module reads the plan refusals."""
    from prism_harness.failures import class_label

    return class_label(failure)
