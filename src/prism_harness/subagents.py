"""Budgets, ledgers, and the nested agents a run may call."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

__all__ = ["MAX_DEPTH", "RunBudget", "RunContext", "RunLedger", "Subagent", "subagent_from_config"]

#: How deep a run tree may nest.
#:
#: Budgets alone do bound a cycle -- mode A calling B calling A terminates when
#: the steps run out -- but they do not bound it CHEAPLY, and they do not bound
#: the child's address, which grows by ``::sub::<name>`` at every level. A tree
#: deep enough would produce session keys long enough to truncate in a store
#: with a fixed-width key and collide two distinct children onto one
#: conversation.
#:
#: Depth is also the honest limit to state: nobody debugs a six-deep agent tree,
#: and a config that produced one is a mistake worth reporting rather than
#: executing.
MAX_DEPTH = 4


@dataclass(frozen=True)
class RunBudget:
    """What a run is ALLOWED TO SPEND.

    ``max_steps`` alone was never a budget. It bounds ITERATIONS, and twenty
    steps each calling an expensive tool sits comfortably inside it -- so a run
    could respect its declared limit and still cost more than anyone intended.
    Cost and wall-clock are the two a person actually cares about when they say
    "bounded".
    """

    max_steps: int
    max_cost_usd: float | None = None
    max_seconds: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any], default_steps: int = 8) -> RunBudget:
        return cls(
            max_steps=_int_or(config.get("max_steps"), default_steps),
            max_cost_usd=_float_or_none(config.get("max_cost_usd")),
            max_seconds=_int_or_none(config.get("max_seconds")),
        )

    def nested_within(self, parent: RunBudget, ledger: RunLedger) -> RunBudget:
        """The budget a CHILD actually gets, as a TREE-ABSOLUTE ceiling.

        BUDGETS NEST; THEY DO NOT RESET. A resetting budget is not a budget: a
        parent limited to 8 steps that may spawn subagents each entitled to a
        fresh 8 has no bound at all -- it has a bound per node in a tree whose
        width it also controls, which is unbounded spend wearing a limit's
        clothing.

        ABSOLUTE, NOT REMAINING -- and this is a FIX, not a port. The reference
        computes the child's budget as what REMAINS
        (``min(declared, parent.max_steps - ledger.steps)``) and then
        ``exhaustion()`` compares the ledger's CUMULATIVE steps against it.
        Those two are in different units, and the result is that a child is
        refused the moment its parent has spent anything: parent 8, ledger 7,
        child declares 2 gives a budget of 1, and ``7 >= 1`` is immediately
        exhausted -- so the child gets ZERO steps while the tree genuinely has
        one left. Verified against the reference's own arithmetic.

        Expressing the child's budget as a ceiling THE TREE MAY REACH makes both
        halves the same unit and gives the right answer in every case.

        Recorded in the envelope's port gaps register as a divergence the
        reference should adopt.
        """
        return RunBudget(
            max_steps=min(parent.max_steps, ledger.steps + self.max_steps),
            max_cost_usd=_lesser(
                parent.max_cost_usd,
                None if self.max_cost_usd is None else ledger.cost_usd + self.max_cost_usd,
            ),
            max_seconds=_int_or_none(
                _lesser(
                    None if parent.max_seconds is None else float(parent.max_seconds),
                    None
                    if self.max_seconds is None
                    else int(ledger.elapsed_seconds()) + self.max_seconds,
                )
            ),
        )


class RunLedger:
    """What a run TREE has actually spent, and whether it has been cancelled.

    SHARED BY REFERENCE from a parent to every descendant, which is the whole
    point: budgets nest, and nesting is only real if the child's spend lands in
    the same account the parent is measured against. A per-run ledger would let
    each node report itself within budget while the tree went far past it.

    Mutable on purpose, and the only mutable thing here. Spend is a running
    total; modelling it immutably would mean threading a new instance back up
    through every return, and the one place that must not be missed is the
    failure path.
    """

    def __init__(self, root_run_id: str, started_at: float | None = None) -> None:
        self.root_run_id = root_run_id
        self._started_at = time.monotonic() if started_at is None else started_at
        self._steps = 0
        self._cost_usd = 0.0
        self._unmetered_runs = 0
        self._cancelled = False
        self._cancel_reason: str | None = None

    @classmethod
    def start(cls, root_run_id: str) -> RunLedger:
        return cls(root_run_id)

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def cost_usd(self) -> float:
        return self._cost_usd

    @property
    def unmetered_runs(self) -> int:
        return self._unmetered_runs

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancel_reason(self) -> str | None:
        return self._cancel_reason

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def record_steps(self, steps: int) -> None:
        self._steps += max(0, steps)

    def record_cost(self, usd: float | None) -> None:
        """Charge a run's cost to the tree.

        NONE IS NOT ZERO. A provider's reported cost is nullable because not
        every provider reports one, and folding that into ``+= 0.0`` would leave
        a cost budget that can never trip -- enforced in the documentation,
        absent at runtime, and indistinguishable from a tree that genuinely
        spent nothing. Counted separately so :meth:`exhaustion` can say the cap
        is UNENFORCEABLE instead of quietly failing open.
        """
        if usd is None:
            self._unmetered_runs += 1
            return

        self._cost_usd += max(0.0, usd)

    def cancel(self, reason: str = "cancelled") -> None:
        """Stop the tree.

        COOPERATIVE rather than pre-emptive: a running tool cannot be
        interrupted, and pretending otherwise would be the more dangerous lie. A
        half-executed tool is precisely the state the durability layer exists to
        protect, so the in-flight call is allowed to finish and the NEXT step is
        refused.
        """
        self._cancelled = True
        self._cancel_reason = reason

    def remaining_cost(self, budget: RunBudget) -> float | None:
        if budget.max_cost_usd is None:
            return None
        return max(0.0, budget.max_cost_usd - self._cost_usd)

    def remaining_seconds(self, budget: RunBudget) -> int | None:
        if budget.max_seconds is None:
            return None
        return max(0, int(budget.max_seconds - self.elapsed_seconds()))

    def exhaustion(self, budget: RunBudget) -> str | None:
        """Why the tree may not spend again -- or None when it may.

        Returns a REASON rather than a bool. The states are genuinely different
        (cancelled / out of steps / out of money / out of time) and a caller
        that cannot tell them apart writes one message for four causes, which is
        the collapse this ecosystem keeps finding. See prism-parity decision
        0020.
        """
        if self._cancelled:
            return self._cancel_reason or "cancelled"

        if self._steps >= budget.max_steps:
            return f"step budget exhausted ({self._steps} of {budget.max_steps} used)"

        if budget.max_cost_usd is not None and self._unmetered_runs > 0:
            # Failing CLOSED. A cost cap the provider gives us no numbers to
            # enforce is not a cap, and continuing would spend without limit
            # under a budget the operator believes is holding.
            return (
                f"cost budget cannot be enforced: {self._unmetered_runs} run(s) reported no "
                f"cost, so spend against the {budget.max_cost_usd:.4f} USD cap is unknown"
            )

        if budget.max_cost_usd is not None and self._cost_usd >= budget.max_cost_usd:
            return (
                f"cost budget exhausted ({self._cost_usd:.4f} of "
                f"{budget.max_cost_usd:.4f} USD used)"
            )

        if budget.max_seconds is not None and self.elapsed_seconds() >= budget.max_seconds:
            return (
                f"time budget exhausted ({int(self.elapsed_seconds())}s of "
                f"{budget.max_seconds}s used)"
            )

        return None


@dataclass(frozen=True)
class Subagent:
    """A nested agent a parent run may call, and the AUTHORITY IT GETS.

    The authority is DECLARED rather than inherited. A subagent that ran with
    whatever its parent happened to hold would make "narrowed toolset" a
    description instead of a constraint -- and the narrowing is the entire
    reason to reach for a subagent rather than another turn of the parent.
    """

    name: str
    description: str
    mode: str
    budget: RunBudget
    #: The scope suffix the child's own session and thread live under.
    #: Deterministic, so a cold worker resuming the tree lands on the same child
    #: conversation instead of starting a fresh one. Defaults to the name.
    scope_suffix: str | None = None

    def scope_under(self, parent_scope: str) -> str:
        """The scope the child session resolves under.

        A DIFFERENT scope from the parent, which is what keeps this from
        deadlocking. A session's lock is taken on its address, and a nested run
        asking for the parent's address inside the parent's own lock would wait
        for a lock it is already holding. Giving the child its own address
        removes the contention rather than making the lock reentrant -- which
        would let a child mutate parent state mid-run, the precise thing the
        lock is for.
        """
        return f"{parent_scope}::sub::{self.scope_suffix or self.name}"


def subagent_from_config(name: str, config: dict[str, Any]) -> Subagent:
    description = config.get("description")
    mode = config.get("mode")
    scope = config.get("scope")

    return Subagent(
        name=name,
        description=description
        if isinstance(description, str) and description
        else f"Run the [{name}] subagent and return its result.",
        mode=mode if isinstance(mode, str) and mode else name,
        budget=RunBudget.from_config(config),
        scope_suffix=scope if isinstance(scope, str) else None,
    )


@dataclass(frozen=True)
class RunContext:
    """Where a run sits in its tree, and what the tree has left to spend.

    A run with NO context is a root: that is the ordinary case and stays free of
    all of this.
    """

    ledger: RunLedger
    budget: RunBudget
    parent_run_id: str | None = None
    depth: int = 0

    @classmethod
    def root(cls, run_id: str, budget: RunBudget) -> RunContext:
        return cls(ledger=RunLedger.start(run_id), budget=budget)

    @property
    def root_run_id(self) -> str:
        return self.ledger.root_run_id

    def is_child(self) -> bool:
        return self.parent_run_id is not None

    def for_child(self, subagent: Subagent, parent_run_id: str) -> RunContext:
        """The context a child inherits: SAME LEDGER, narrowed budget.

        Same ledger by reference is the load-bearing part. A child with its own
        ledger would let every node report itself inside budget while the tree
        spent without limit.
        """
        return RunContext(
            ledger=self.ledger,
            budget=subagent.budget.nested_within(self.budget, self.ledger),
            parent_run_id=parent_run_id,
            depth=self.depth + 1,
        )

    def too_deep(self) -> bool:
        return self.depth >= MAX_DEPTH


def _lesser(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _int_or(value: Any, fallback: int) -> int:
    return (
        int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback
    )


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
