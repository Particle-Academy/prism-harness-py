"""What the harness says about an exception it did not write (G-387, G-385).

An exception's message can carry text from a provider, a store, a host or a
tool: a prompt fragment, an operator's note, a file path. The harness has no
content-capture setting of its own (that gate lives in the telemetry bridges,
which read what the harness emits), so it never writes such text anywhere it
keeps or emits: a run's state, ``RunFailed``, or what a parent model is told.
It writes the exception's class name, a fixed description, and the harness
error code where one exists, and nothing else.
"""

from __future__ import annotations

import re

from prism_harness.errors import ErrorCode, HarnessError
from prism_harness.plans.budget import EXHAUSTED as PLAN_BUDGET_EXHAUSTED
from prism_harness.plans.refusal import MESSAGES as PLAN_REFUSALS
from prism_harness.run_starts import START_REFUSED

__all__ = ["HARNESS_CODES", "class_label", "failure_of", "harness_code", "named"]

#: What a class with no identifier-shaped name in its whole lineage is called.
UNNAMED = "an unnamed exception"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")

#: Every code the harness itself raises. A code outside this set is not one the
#: harness wrote, and is never repeated.
HARNESS_CODES = frozenset(
    {code.value for code in ErrorCode}
    | set(PLAN_REFUSALS)
    | {"plan.disabled", PLAN_BUDGET_EXHAUSTED, START_REFUSED, "child_turn_live"}
)


def harness_code(error: BaseException) -> str | None:
    """A harness error's code, read ONCE and guarded: None for any other
    exception, for a code getter that raises, and for a code the harness never
    raises (a host's own subclass can return anything from ``code``)."""
    if not isinstance(error, HarnessError):
        return None
    try:
        code = error.code
    except Exception:  # noqa: BLE001 - a getter that raises names no code
        return None
    return code if type(code) is str and code in HARNESS_CODES else None


def class_label(error: BaseException) -> str:
    """The exception's class as the harness names it (G-387 round 10): an
    identifier-shaped name only (ASCII letters, digits and ``_``, at most 128).
    A class whose ``__name__`` is anything else (set to a path, a dotted
    module path, text with spaces or line breaks) is named by the first
    ancestor that has one, else a fixed text. Never a file path or a line."""
    try:
        lineage = list(type(error).__mro__)
    except Exception:  # noqa: BLE001 - a lineage that cannot be read names nothing
        return UNNAMED
    for klass in lineage:
        try:
            name = klass.__name__
        except Exception:  # noqa: BLE001, S112 - a name that cannot be read is skipped
            continue
        if type(name) is str and _IDENTIFIER.fullmatch(name):
            return name
    return UNNAMED


def named(error: BaseException, code: str | None = None) -> str:
    """An exception named by its class and, for a harness error, its code:
    never its message. ``code`` is one already read with
    :func:`harness_code`, so a getter is never read twice."""
    name = class_label(error)
    found = code if code is not None else harness_code(error)
    return f"{name} ({found})" if found is not None else name


def failure_of(error: BaseException) -> str:
    """What a failed run records as its failure, in its run state and in
    ``RunFailed``: the exception's class, the harness code where there is one,
    and a fixed description."""
    return f"{named(error)}: the run failed."
