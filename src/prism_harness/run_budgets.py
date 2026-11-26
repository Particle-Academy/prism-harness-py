"""How many model steps a run may take: one authority (G-387).

G-381 gave a run's tree membership one authority. This does the same for its
step allowance: the facts it is worked out from are persisted, never
re-derived, and it is worked out once.

The facts: the steps a parked run used, children included, on the row it
parks on as ``harness_steps_used``; a child's ceiling, saved with its binding
when the runner starts it, and the steps the child used since, counted durably
beside it. Then :func:`allowance` turns them into
the run's allowance ONCE, when the run begins. The runtime records it beside
the run's membership, narrows the run to it, and every consumer reads that.

Acting on a decision a person recorded costs no model step, so a step budget
never refuses a decision: an allowance of 0 means the run acts on its decisions
and takes no model step (``acts_without_step``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = ["KINDS", "MAX_SAFE_INTEGER", "RunBudgets", "allowance", "fail_closed", "steps_used"]

#: The kinds of run context a turn can be handed (the same as
#: ``run_trees.CONTEXT_KINDS``).
KINDS = ("none", "metadata", "runner", "host")

#: The largest whole number every language reads exactly (JavaScript's
#: ``Number.MAX_SAFE_INTEGER``).
MAX_SAFE_INTEGER = 2**53 - 1


def _nat(value: Any) -> int | None:
    """A whole number, at least 0; None stays None. Anything that is not a
    finite number (NaN, an infinity, a string, a boolean) is 0: an input that
    cannot be measured never removes a cap, it closes it."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if isinstance(value, float) and not math.isfinite(value):
        return 0
    whole: int = math.trunc(value)
    return max(0, whole)


def allowance(facts: Mapping[str, Any]) -> dict[str, Any]:
    """The step allowance of a run, from the facts the runtime holds when it
    begins. Pure, so every language answers it the same way (prism-parity
    harness-plans, op ``step_allowance``).

    - ``context``: ``none``, ``metadata``, ``runner`` (the subagent runner's
      own context starting a child) or ``host`` (any other context).
    - ``mode_steps``: the resolved mode's ``max_steps``.
    - ``saved_ceiling``: the ceiling saved with a child's binding when the
      runner started it, or None.
    - ``context_left``: what a runner or host context has left
      (``budget.max_steps - ledger.steps``); None without one.
    - ``steps_ceiling``: the ``max_steps_ceiling`` setting, or None. Absolute
      on the run's whole count, under every context.
    - ``used``: the steps the parked run this turn resumes had used, children
      included (0 for a turn that resumes nothing; a runner start is a new
      run, so always 0 there).
    - ``tree_left`` / ``plan_left``: the grants of the durable run-tree record
      and the plan ledger, or None when none bounds the run.
    - ``took``: steps taken since the run began (for the values at a park).

    ``{"invalid": True}`` for an unknown kind, or a runner or host context
    without ``context_left``.
    """
    kind = facts.get("context")
    if kind not in KINDS:
        return {"invalid": True}
    if kind in ("runner", "host") and facts.get("context_left") is None:
        return {"invalid": True}
    mode = _nat(facts.get("mode_steps")) or 0
    saved = _nat(facts.get("saved_ceiling"))
    context_left = _nat(facts.get("context_left"))
    steps_ceiling = _nat(facts.get("steps_ceiling"))
    used = 0 if kind == "runner" else (_nat(facts.get("used")) or 0)
    tree = _nat(facts.get("tree_left"))
    plan = _nat(facts.get("plan_left"))
    took = _nat(facts.get("took")) or 0

    if kind == "runner":
        assert context_left is not None
        ceiling = context_left
    elif saved is not None:
        ceiling = saved
    elif kind == "host":
        assert context_left is not None
        ceiling = used + context_left
    else:
        ceiling = mode
    if steps_ceiling is not None:
        ceiling = min(ceiling, steps_ceiling)
    own_left = max(0, ceiling - used)
    if kind == "host":
        assert context_left is not None
        own_left = min(own_left, context_left)
    left = own_left
    if tree is not None:
        left = min(left, tree)
    if plan is not None:
        left = min(left, plan)
    taken = min(took, left)
    return {
        "allowance": {
            "ceiling": ceiling,
            "used": used,
            "own_left": own_left,
            "tree_left": tree,
            "plan_left": plan,
            "allowance": left,
            "acts_without_step": left == 0,
            "used_at_park": used + taken,
            "left_at_park": left - taken,
        }
    }


def fail_closed(result: Mapping[str, Any]) -> dict[str, Any]:
    """The allowance of a run whose parked row states a steps-used count that is
    not a fact the harness could have written: used is the ceiling, so the run
    acts on its decisions and takes no model step."""
    found = dict(result["allowance"])
    found.update(
        used=found["ceiling"],
        own_left=0,
        allowance=0,
        acts_without_step=True,
        used_at_park=found["ceiling"],
        left_at_park=0,
    )
    return {"allowance": found}


def steps_used(row: Mapping[str, Any] | None) -> int | str:
    """What a paused row says the run parked on it had used (``harness_steps_used``):
    the count, ``"absent"`` for a row written before the field existed, or
    ``"forged"`` for a value that is not a whole number from 0 to
    :data:`MAX_SAFE_INTEGER`."""
    if row is None or "harness_steps_used" not in row:
        return "absent"
    value = row["harness_steps_used"]
    if type(value) is int and 0 <= value <= MAX_SAFE_INTEGER:
        return value
    return "forged"


class RunBudgets:
    """The run step allowance, as the other languages name it
    (``RunBudgets.allowance``)."""

    allowance = staticmethod(allowance)
    fail_closed = staticmethod(fail_closed)
    steps_used = staticmethod(steps_used)
