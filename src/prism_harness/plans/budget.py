"""How a plan's steps are budgeted, and the plan's shared ledger (G-366).

The app's developer chooses, in the plans config's ``budget`` or per plan
through host code (:meth:`PlanHandle.use_budget`); never the model, never a
per-call option:

- ``per_step`` (default): each step's turn gets the per-turn budget, as it
  always has.
- ``shared``: every turn that serves the plan also spends from ONE ledger with
  ``max_steps`` model steps and, optionally, ``max_cost_usd``. Each turn still
  keeps its own per-turn limits; the ledger bounds the total.
- ``none``: no plan-level budget. Per-turn limits, the plan's own step and
  advance limits and agent spend ceilings still apply.

Pure state in, state out, the same as the PHP reference's PlanBudget (pinned by
the ``harness-plans`` corpus). :class:`Plans` keeps the state in the plan store
and changes it in one locked write per change.

The ledger is a hold-and-settle account. Before a turn, the steps and cost it
may spend are HELD; after it, the hold is replaced by what it spent. Two turns
can therefore never both be granted the same remainder. A hold that outlives
the hold TTL (a turn that died without settling) is charged in full: an unknown
spend counts as spent, never as free.

Costs are kept in whole micro-USD so every language adds them exactly.
"""

from __future__ import annotations

import math
from typing import Any

from prism_harness.plans.refusal import PlanRefused

__all__ = [
    "EXHAUSTED",
    "MODES",
    "PlanBudgetInvalid",
    "exhaustion",
    "expire",
    "for_plan",
    "micros",
    "narrow",
    "open_ledger",
    "reserve",
    "settings",
    "settle",
]

MODES = ("per_step", "shared", "none")

#: The code a turn and a plan stop with when the shared ledger is spent.
EXHAUSTED = "plan_budget_exhausted"

STEPS_CEILING = 10000
COST_CEILING_USD = 1000000
_KEYS = ("mode", "max_steps", "max_cost_usd")

Settings = dict[str, Any]
State = dict[str, Any]


class PlanBudgetInvalid(TypeError):
    """An invalid budget setting. Raised at configuration, before anything runs."""


def _lower(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def micros(usd: float) -> int:
    """USD as whole micro-USD, rounded half up like the reference."""
    return math.floor(usd * 1_000_000 + 0.5)


def settings(config: object, source: str = "plans.budget", complete: bool = True) -> Settings:
    """Validate a budget setting. Raises :class:`PlanBudgetInvalid` for an
    unknown key or mode, a limit out of range, or ``shared`` without
    ``max_steps``."""
    if config is None:
        return {"mode": "per_step", "max_steps": None, "max_cost_micros": None}
    if not isinstance(config, dict):
        raise PlanBudgetInvalid(f"{source} must be a map.")
    for key in config:
        if key not in _KEYS:
            raise PlanBudgetInvalid(f"{source} has an unknown setting.")
    mode = config.get("mode")
    mode = "per_step" if mode is None else mode
    if not isinstance(mode, str) or mode not in MODES:
        raise PlanBudgetInvalid(f"{source}.mode must be per_step, shared or none.")
    steps = config.get("max_steps")
    if steps is not None and (
        isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or steps > STEPS_CEILING
    ):
        raise PlanBudgetInvalid(
            f"{source}.max_steps must be a whole number between 1 and {STEPS_CEILING}, or null."
        )
    cost = config.get("max_cost_usd")
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0
        or cost > COST_CEILING_USD
    ):
        raise PlanBudgetInvalid(
            f"{source}.max_cost_usd must be a number of USD between 0 and {COST_CEILING_USD}, or null."
        )
    if complete and mode == "shared" and steps is None:
        raise PlanBudgetInvalid(f"{source}.max_steps is required when the mode is shared.")
    return {
        "mode": mode,
        "max_steps": steps,
        "max_cost_micros": None if cost is None else micros(float(cost)),
    }


def for_plan(configured: Settings, requested: object) -> Settings:
    """The setting for one plan, chosen by host code: its mode, with each limit
    the lower of the host's and the configured one."""
    asked = settings(requested, "the plan budget", False)
    chosen = {
        "mode": asked["mode"],
        "max_steps": _lower(configured["max_steps"], asked["max_steps"]),
        "max_cost_micros": _lower(configured["max_cost_micros"], asked["max_cost_micros"]),
    }
    if chosen["mode"] == "shared" and chosen["max_steps"] is None:
        raise PlanBudgetInvalid("the plan budget.max_steps is required when the mode is shared.")
    return chosen


def open_ledger(plan_id: str, chosen: Settings) -> State:
    """A plan's budget record, fixed from a setting."""
    return {
        "planId": plan_id,
        "mode": chosen["mode"],
        "max_steps": chosen["max_steps"],
        "max_cost_micros": chosen["max_cost_micros"],
        "steps": 0,
        "cost_micros": 0,
        "unmetered": 0,
        "holds": {},
    }


def narrow(state: State, chosen: Settings) -> State:
    """A later host choice on a fixed record: the mode cannot change, and a
    limit can only be lowered."""
    if chosen["mode"] != state["mode"]:
        raise PlanRefused("plan.budget_fixed")
    return {
        **state,
        "max_steps": _lower(state["max_steps"], chosen["max_steps"]),
        "max_cost_micros": _lower(state["max_cost_micros"], chosen["max_cost_micros"]),
    }


def exhaustion(state: State) -> str | None:
    """Which limit is spent, or None. Holds do not count: they are spend that
    may yet be returned."""
    if state["mode"] != "shared":
        return None
    if state["steps"] >= (state["max_steps"] or 0):
        return "steps"
    if state["max_cost_micros"] is not None:
        # A cost a provider did not report cannot be held to a cap: closed.
        if state["unmetered"] > 0:
            return "cost_unmetered"
        if state["cost_micros"] >= state["max_cost_micros"]:
            return "cost"
    return None


def reserve(
    state: State, hold_id: str, steps: int, cost_micros: int | None, now: int, hold_ttl: int
) -> tuple[State, dict[str, Any]]:
    """Hold what one turn may spend: at most ``steps`` steps and ``cost_micros``
    (None: no per-turn cost cap), never more than the ledger has left after
    every other hold. Expired holds are charged in full first.

    Returns the state and either the grant (``steps``, ``cost_micros``) or the
    refusal (``refused`` with the EXHAUSTED code and the ``limit``)."""
    state = expire(state, now, hold_ttl)
    held_steps = sum(hold["steps"] for hold in state["holds"].values())
    held_cost = sum(hold["cost_micros"] for hold in state["holds"].values())
    limit = exhaustion(state)
    left_steps = (state["max_steps"] or 0) - state["steps"] - held_steps
    left_cost = (
        None
        if state["max_cost_micros"] is None
        else state["max_cost_micros"] - state["cost_micros"] - held_cost
    )
    if limit is None and left_steps <= 0:
        limit = "steps"
    if limit is None and left_cost is not None and left_cost <= 0:
        limit = "cost"
    if limit is not None or steps < 1:
        return state, {"refused": EXHAUSTED, "limit": limit or "steps"}
    grant_steps = min(steps, left_steps)
    grant_cost = (
        None
        if left_cost is None
        else (left_cost if cost_micros is None else min(cost_micros, left_cost))
    )
    holds = {
        **state["holds"],
        hold_id: {"steps": grant_steps, "cost_micros": grant_cost or 0, "at": now},
    }
    return {**state, "holds": dict(sorted(holds.items()))}, {
        "steps": grant_steps,
        "cost_micros": grant_cost,
    }


def settle(state: State, hold_id: str, steps: int, cost_micros: int | None) -> State:
    """Replace a turn's hold with what it spent. ``cost_micros`` None: the
    provider reported no cost. A hold already gone changes nothing."""
    if hold_id not in state["holds"]:
        return state
    holds = {key: hold for key, hold in state["holds"].items() if key != hold_id}
    return {
        **state,
        "holds": holds,
        "steps": state["steps"] + max(0, steps),
        "unmetered": state["unmetered"] + (1 if cost_micros is None else 0),
        "cost_micros": state["cost_micros"] + (0 if cost_micros is None else max(0, cost_micros)),
    }


def expire(state: State, now: int, hold_ttl: int) -> State:
    """Charge every hold older than the TTL in full."""
    for hold_id, hold in list(state["holds"].items()):
        if now - hold["at"] >= hold_ttl:
            holds = {key: value for key, value in state["holds"].items() if key != hold_id}
            state = {
                **state,
                "holds": holds,
                "steps": state["steps"] + hold["steps"],
                "cost_micros": state["cost_micros"] + hold["cost_micros"],
            }
    return state
