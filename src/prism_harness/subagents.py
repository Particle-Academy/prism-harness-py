"""Budgets, ledgers, and the nested agents a run may call."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from prism_harness.errors import HarnessError

__all__ = [
    "MAX_DEPTH",
    "ChildBinding",
    "RunBudget",
    "RunContext",
    "RunLedger",
    "Subagent",
    "subagent_from_config",
]

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
    max_output_tokens_per_call: int | None = None

    def __post_init__(self) -> None:
        # A step cap that is not a whole number is no cap: NaN compares false
        # with everything and an infinity never runs out, so either would
        # remove the limit it claims to set (G-387). Refused, never coerced.
        steps = self.max_steps
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("max_steps must be a whole number.")
        if steps < 0:
            raise ValueError("max_steps must be zero or more.")
        output = self.max_output_tokens_per_call
        if output is not None and (
            isinstance(output, bool) or not isinstance(output, int) or output < 1
        ):
            raise ValueError("Output ceilings must be positive integers or null.")
        # A time or cost cap that is not a finite number is no cap either: NaN
        # compares false with everything, an infinity never runs out, and a
        # negative one reads as spent by one path and as no limit by another.
        # Refused where the host builds the budget, never coerced (G-387).
        for name in ("max_cost_usd", "max_seconds"):
            if not _limit_ok(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number, zero or more, or null.")

    @classmethod
    def from_config(cls, config: dict[str, Any], default_steps: int = 8) -> RunBudget:
        output = config.get("max_output_tokens_per_call")
        if output is not None and (
            isinstance(output, bool) or not isinstance(output, int) or output < 1
        ):
            raise ValueError("Output ceilings must be positive integers or null.")
        steps = config.get("max_steps")
        if steps is None:
            steps = default_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            # A declaration that is not a whole number is refused, never read
            # as the default: a typo must not raise the cap it meant to set.
            raise ValueError("A subagent's max_steps must be a whole number, zero or more.")
        for name in ("max_cost_usd", "max_seconds"):
            # Checked before conversion: int() of an infinity raises an
            # OverflowError, and a declaration that cannot be read is refused
            # the same way however it fails.
            if not _limit_ok(config.get(name)):
                raise ValueError(f"A subagent's {name} must be a finite number, zero or more.")
        return cls(
            max_steps=steps,
            max_cost_usd=_float_or_none(config.get("max_cost_usd")),
            max_seconds=_int_or_none(config.get("max_seconds")),
            max_output_tokens_per_call=output,
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
            max_output_tokens_per_call=(
                parent.max_output_tokens_per_call
                if self.max_output_tokens_per_call is None
                else min(
                    parent.max_output_tokens_per_call or self.max_output_tokens_per_call,
                    self.max_output_tokens_per_call,
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
        # Spend by agent label, inside this one account. A label is not a
        # second ledger: every charge also lands in the tree's total above.
        self._agent_cost: dict[str, float] = {}
        self._agent_unmetered: dict[str, int] = {}

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

    def agent_cost_usd(self, agent: str) -> float:
        """What calls labelled with ``agent`` have spent in this tree."""
        return self._agent_cost.get(agent, 0.0)

    def agent_unmetered_runs(self, agent: str) -> int:
        return self._agent_unmetered.get(agent, 0)

    def carry_agent_spend(self, agent: str, usd: float, unmetered_runs: int) -> None:
        """Carry what ``agent`` spent before an approval pause into this tree.

        A resumed run is the same run, so its agent's ceiling is measured
        against what it spent before the pause as well. Charged to the agent's
        account only, never to the tree total: that spend was already reported
        by the run that paused, and a total here would count it twice.
        """
        self._agent_cost[agent] = self._agent_cost.get(agent, 0.0) + max(0.0, usd)
        if unmetered_runs > 0:
            self._agent_unmetered[agent] = self._agent_unmetered.get(agent, 0) + unmetered_runs

    def record_cost(self, usd: float | None, agent: str | None = None) -> None:
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
            if agent is not None:
                self._agent_unmetered[agent] = self._agent_unmetered.get(agent, 0) + 1
            return

        self._cost_usd += max(0.0, usd)
        if agent is not None:
            self._agent_cost[agent] = self._agent_cost.get(agent, 0.0) + max(0.0, usd)

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
        code = self.exhaustion_code(budget)
        if code is None:
            return None
        if code == "cancelled":
            # The host's own reason, for the host's own response and logs. The
            # subagent runner never passes it on: it reads exhaustion_code().
            return self._cancel_reason or "cancelled"

        if code == "steps":
            return f"step budget exhausted ({self._steps} of {budget.max_steps} used)"

        if code == "cost_unenforceable" and not _finite_limit(budget.max_cost_usd):
            return "cost budget cannot be enforced: its cap is not a finite number"

        if code == "time" and not _finite_limit(budget.max_seconds):
            return "time budget cannot be enforced: its limit is not a finite number"

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

        return code

    def exhaustion_code(self, budget: RunBudget) -> str | None:
        """Why the tree may not spend again, as a fixed code (``cancelled``,
        ``steps``, ``cost_unenforceable``, ``cost``, ``time``), or None when
        it may. Never carries text a host chose: what a model is told about a
        stop is chosen from this code (G-387).

        A time or cost limit that is not a finite number (forged past the
        constructor) is read as spent, never as no limit."""
        if self._cancelled:
            return "cancelled"
        steps: Any = budget.max_steps
        if not _finite_limit(steps) or self._steps >= steps:
            return "steps"
        cap: Any = budget.max_cost_usd
        if cap is not None and not _finite_limit(cap):
            return "cost_unenforceable"
        if cap is not None and self._unmetered_runs > 0:
            return "cost_unenforceable"
        if cap is not None and self._cost_usd >= cap:
            return "cost"
        limit: Any = budget.max_seconds
        if limit is not None and (not _finite_limit(limit) or self.elapsed_seconds() >= limit):
            return "time"
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
    #: The provider and model the child runs on, above any mode (G-321).
    provider: str | None = None
    model: str | None = None
    #: The named agent the child runs as. Only an agent the parent mode lists in
    #: its ``agents`` may be named here: authority is declared, never inherited.
    agent: str | None = None
    #: Put the parent's SESSION override in the child's session slot. Off by
    #: default: the override belongs to the parent's session, not its tree.
    inherit_session_model: bool = False

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
        provider=_nonempty_or_none(config.get("provider")),
        model=_nonempty_or_none(config.get("model")),
        agent=_nonempty_or_none(config.get("agent")),
        inherit_session_model=config.get("inherit_session_model") is True,
    )


def binding_route(
    context: str, supplied_binding: bool, saved_binding: bool, linked: bool
) -> dict[str, Any]:
    """Which binding a turn runs on, and whether it is saved with the session.

    The one rule every language's runtime asks (the reference's
    ``ChildBinding::route``), pinned by the harness-plans corpus:

    - ``runner``: the subagent runner's own context starts the child; its
      binding is used and saved with the child's session.
    - ``saved``: any other turn on a session with a saved binding uses it. A
      binding on a context the host supplied never replaces it.
    - ``refused``: a session the runner started as a child (``linked``) with
      no saved binding is refused (``child_binding_missing``) before any
      request.
    - ``host``: a host context's binding on a session that was never a child
      is used for that turn only; it is never saved.
    - ``root``: otherwise the root rules.

    :param context: none, metadata, runner or host, as ``RunTrees.context_kind``
    :raises ValueError: for an unknown context kind
    """
    if context not in ("none", "metadata", "runner", "host"):
        raise ValueError("Unknown run context kind.")
    if context == "runner" and supplied_binding:
        return {"binding": "runner", "save": True}
    if saved_binding:
        binding = "saved"
    elif linked:
        binding = "refused"
    else:
        binding = "host" if supplied_binding else "root"
    return {"binding": binding, "save": False}


@dataclass(frozen=True)
class ChildBinding:
    """What a subagent child's provider and model resolve from, beside its own mode.

    Attached by the runner, so a child resolves, field by field:

    1. the parent's session override -- only when the subagent asked for it;
    2. the subagent declaration;
    3. the child's own mode;
    4. the parent's MODE -- only when the child runs in the parent's agent;
    5. the child agent's default, then the global default.

    The child session's own override slot is not consulted: the runner owns
    that address.
    """

    session_provider: str | None = None
    session_model: str | None = None
    declared_provider: str | None = None
    declared_model: str | None = None
    parent_mode_provider: str | None = None
    parent_mode_model: str | None = None

    def to_row(self) -> dict[str, dict[str, str | None]]:
        """The ``harness_binding`` a paused child's approval row carries."""
        return {
            "session": {"provider": self.session_provider, "model": self.session_model},
            "declared": {"provider": self.declared_provider, "model": self.declared_model},
            "parent_mode": {
                "provider": self.parent_mode_provider,
                "model": self.parent_mode_model,
            },
        }

    @classmethod
    def from_row(cls, value: object) -> ChildBinding | None:
        """The binding a paused row stored, or None when absent or malformed.

        Malformed is ignored rather than refused: the run then resolves as a
        root run does, which is what it did before bindings were stored.
        """
        if not isinstance(value, Mapping):
            return None
        fields: list[str | None] = []
        for part in ("session", "declared", "parent_mode"):
            entry = value.get(part)
            if not isinstance(entry, Mapping):
                return None
            for key in ("provider", "model"):
                if key not in entry:
                    return None
                chosen = entry[key]
                if chosen is not None and not isinstance(chosen, str):
                    return None
                fields.append(chosen)
        return cls(*fields)


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
    #: The named agent spending through this context, as ``harness_agent``.
    agent: str | None = None
    #: That agent's ``spend_ceiling``, evaluated against THIS ledger.
    spend_ceiling: float | None = None
    #: Set by the runner for a subagent child; None resolves like a root run.
    binding: ChildBinding | None = None

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

    def with_binding(self, binding: ChildBinding) -> RunContext:
        return replace(self, binding=binding)

    def agent_stop(self) -> HarnessError | None:
        """Why this context's agent may not call a model again, or None.

        Checked before every model call. Unmetered first: a ceiling the
        provider reports no cost against cannot be enforced, and continuing
        would spend without limit under a cap the operator believes holds.
        """
        if self.agent is None or self.spend_ceiling is None:
            return None

        unmetered = self.ledger.agent_unmetered_runs(self.agent)
        if unmetered > 0:
            return HarnessError.budget_unpriced(self.agent, unmetered)

        spent = self.ledger.agent_cost_usd(self.agent)
        if spent >= self.spend_ceiling:
            return HarnessError.agent_spend_ceiling_exceeded(self.agent, spent, self.spend_ceiling)

        return None

    def too_deep(self) -> bool:
        return self.depth >= MAX_DEPTH


def _nonempty_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _lesser(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _limit_ok(value: Any) -> bool:
    """Whether a time or cost limit is usable: absent, or a finite number of
    zero or more. A limit that is PRESENT but anything else (text such as
    ``"NAN"``, ``"INF"``, ``"0,50"`` or a quoted ``"60"``, a boolean, a list) is
    a configuration error, never read as "no limit" (G-387 round 9)."""
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _finite_limit(value: Any) -> bool:
    """Whether a limit read at run time is a finite number. One forged past the
    constructor (NaN, an infinity, a non-number) is read as spent."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
