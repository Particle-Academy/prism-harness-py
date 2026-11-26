"""Whether a runner start may start: one gate, asked once (G-387).

A runner start (the subagent runner starting a child) used to check its stop
conditions at several places and write durable state between them, so a start
refused late had already re-bound the child. Now every fact the start depends
on is READ first, :func:`gate` decides once, and only a start it lets through
writes anything: the tree hold, then the child's records, then the run.

Pure, so every language answers it the same way (prism-parity harness-plans,
op ``start_gate``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from prism_harness.errors import HarnessError

__all__ = [
    "BLANK_CODEPOINTS",
    "COSTS",
    "PENDING",
    "REASONS",
    "START_REFUSED",
    "RunStarts",
    "StartRefused",
    "gate",
    "is_blank",
]

#: The codepoints a blank task consists of, spelled identically in every
#: language: never the language's own whitespace rule, which differs between
#: languages and versions.
BLANK_CODEPOINTS = frozenset(
    [
        0x09,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x20,
        0x85,
        0xA0,
        0x1680,
        *range(0x2000, 0x200E),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x2060,
        0x3000,
        0xFEFF,
    ]
)

#: What a start's cost limits say: ``ok``, ``spent`` (any cost limit that
#: would stop the child is spent) or ``unenforceable`` (one cannot be metered).
COSTS = ("ok", "spent", "unenforceable")

#: The child's pending approval: ``none``, ``answerable`` (a decision on it
#: would be accepted) or ``unanswerable``.
PENDING = ("none", "answerable", "unanswerable")

_MAX_SAFE_INTEGER = 2**53 - 1


def is_blank(task: str) -> bool:
    """Whether a task consists only of :data:`BLANK_CODEPOINTS` (or nothing)."""
    return all(ord(character) in BLANK_CODEPOINTS for character in task)


def _nat(value: Any) -> int:
    """A whole number above 0, or 0: anything else (a fraction, a negative,
    NaN, an infinity, a string, a boolean, nothing) cannot fund a step."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if 0 < value <= _MAX_SAFE_INTEGER else 0
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        whole = int(value)
        return whole if 0 < whole <= _MAX_SAFE_INTEGER else 0
    return 0


def _bounds(facts: Mapping[str, Any], name: str) -> bool:
    """Whether a grant applies: present and not None. A missing grant is never
    read as "no grant applies"; it is read as nothing granted."""
    return name not in facts or facts[name] is not None


def gate(facts: Mapping[str, Any]) -> dict[str, Any]:
    """The one decision a runner start makes before it writes anything.

    Facts: ``task`` (the raw value), ``too_deep``, ``cancelled``,
    ``time_spent``, ``steps_left`` (what the start's own allowance would give),
    ``cost`` (:data:`COSTS`), ``child_busy`` (a live hold on the child's
    delegation count), ``child_pending`` (:data:`PENDING`), ``tree_left`` and
    ``plan_left`` (what the tree and plan records would grant now; None when
    none applies).

    Asked in one fixed order, so every language gives the same reason when
    several stops hold: ``{"gate": {"refused": code, "outcome": outcome}}``, or
    ``{"gate": {"start": {"steps": n}}}`` with the least of what the start's
    allowance, the tree and the plan grant. ``{"invalid": True}`` for an
    unknown cost or pending state.
    """
    if facts.get("cost") not in COSTS or facts.get("child_pending") not in PENDING:
        return {"invalid": True}

    def refuse(code: str, outcome: str) -> dict[str, Any]:
        return {"gate": {"refused": code, "outcome": outcome}}

    task = facts.get("task")
    if not isinstance(task, str):
        return refuse("task_invalid", "denied")
    if is_blank(task):
        return refuse("task_blank", "denied")
    if facts.get("too_deep") is True:
        return refuse("too_deep", "denied")
    if facts.get("cancelled") is True:
        return refuse("cancelled", "cancelled")
    if facts.get("time_spent") is True:
        return refuse("time", "exhausted")
    if _nat(facts.get("steps_left")) < 1:
        return refuse("steps", "exhausted")
    if facts["cost"] == "spent":
        return refuse("cost", "exhausted")
    if facts["cost"] == "unenforceable":
        return refuse("cost_unenforceable", "exhausted")
    if facts.get("child_busy") is True:
        return refuse("child_busy", "denied")
    if facts["child_pending"] == "answerable":
        return refuse("child_waiting", "denied")
    tree, plan = _bounds(facts, "tree_left"), _bounds(facts, "plan_left")
    if tree and _nat(facts.get("tree_left")) < 1:
        return refuse("tree", "exhausted")
    if plan and _nat(facts.get("plan_left")) < 1:
        return refuse("plan", "exhausted")
    steps = _nat(facts.get("steps_left"))
    if tree:
        steps = min(steps, _nat(facts.get("tree_left")))
    if plan:
        steps = min(steps, _nat(facts.get("plan_left")))
    return {"gate": {"start": {"steps": steps}}}


#: The code of a runner start the gate (or its commit) refused.
START_REFUSED = "run_start_refused"

#: What the parent model is told for each refusal: harness-written, never a
#: host's or a store's text.
REASONS = {
    "task_invalid": "the task is not text",
    "task_blank": "the task is empty",
    "too_deep": "subagent nesting reached the maximum depth of {depth}",
    "cancelled": "its run was cancelled",
    "time": "time budget exhausted",
    "steps": "step budget exhausted",
    "cost": "cost budget exhausted",
    "cost_unenforceable": "cost budget cannot be enforced",
    "child_busy": "a turn on it is still running",
    "child_waiting": "it is waiting on a person's approval",
    "tree": "its run tree has no steps left",
    "plan": "its plan's shared budget has no steps left",
}


class StartRefused(HarnessError):
    """A runner start refused by :func:`gate`, or by its commit on a race since
    the read, having written nothing. ``refusal`` is the gate's code and
    ``outcome`` what the runner frames.

    Internal: raised by the runtime, framed by the subagent runner."""

    def __init__(self, refusal: str, outcome: str) -> None:
        super().__init__(START_REFUSED, f"This runner start was refused ({refusal}).")
        self.refusal = refusal
        self.outcome = outcome


class RunStarts:
    """The runner-start gate, as the other languages name it
    (``RunStarts.gate``)."""

    gate = staticmethod(gate)
    is_blank = staticmethod(is_blank)
