"""What happened during a run, for anything that wants to watch."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "HarnessEvent",
    "HarnessEvents",
    "HarnessListener",
    "PlanUpdated",
    "RunCancelRequested",
    "RunCancelled",
    "RunFailed",
    "RunFinished",
    "RunStarted",
    "ToolApprovalAsked",
    "ToolApproved",
    "ToolAutoAllowed",
    "ToolDenied",
]


@dataclass(frozen=True)
class RunStarted:
    type: str = field(default="run.started", init=False)
    run_id: str = ""
    session_key: str = ""
    mode: str = ""
    provider: str = ""
    model: str = ""
    #: The root of the tree this run belongs to; equal to ``run_id`` for a root.
    root_run_id: str = ""
    depth: int = 0
    at: str = ""
    #: The named agent this run spends as (G-321), or None in the legacy
    #: namespace -- where :meth:`HarnessEvents.to_dict` omits it, so a legacy
    #: event keeps its shape. Unrelated to agent-integrations' presence id.
    harness_agent: str | None = None


@dataclass(frozen=True)
class RunFinished:
    type: str = field(default="run.finished", init=False)
    run_id: str = ""
    session_key: str = ""
    finish_reason: str = ""
    #: NAMES only, in call order.
    tool_calls: tuple[str, ...] = ()
    steps: int = 0
    cost_usd: float | None = None
    at: str = ""
    #: The named agent this run spends as (G-321), or None in the legacy
    #: namespace -- where :meth:`HarnessEvents.to_dict` omits it, so a legacy
    #: event keeps its shape. Unrelated to agent-integrations' presence id.
    harness_agent: str | None = None


@dataclass(frozen=True)
class RunFailed:
    type: str = field(default="run.failed", init=False)
    run_id: str = ""
    session_key: str = ""
    #: Why it stopped: a budget reason, a cancellation, or a provider failure.
    failure: str = ""
    steps: int = 0
    at: str = ""
    #: The named agent this run spends as (G-321), or None in the legacy
    #: namespace -- where :meth:`HarnessEvents.to_dict` omits it, so a legacy
    #: event keeps its shape. Unrelated to agent-integrations' presence id.
    harness_agent: str | None = None


@dataclass(frozen=True)
class RunCancelRequested:
    """A person asked a run to stop (G-285), emitted the moment the durable
    flag is written, by the process that wrote it. ``state`` is ``running``
    (the run stops at its next check) or ``parked`` (it was waiting on an
    approval and is settled now, with :class:`RunCancelled` right after)."""

    type: str = field(default="run.cancel_requested", init=False)
    run_id: str = ""
    session_key: str = ""
    parent_run_id: str | None = None
    root_run_id: str | None = None
    state: str = "running"
    at: str = ""
    harness_agent: str | None = None


@dataclass(frozen=True)
class RunCancelled:
    """A run stopped because a person cancelled it, or the run tree it belongs
    to (G-285). Emitted INSTEAD of ``run.finished`` or ``run.failed``, never
    beside them."""

    type: str = field(default="run.cancelled", init=False)
    run_id: str = ""
    session_key: str = ""
    parent_run_id: str | None = None
    root_run_id: str | None = None
    at: str = ""
    harness_agent: str | None = None


@dataclass(frozen=True)
class PlanUpdated:
    """A plan was written (G-360): drafted, decided, settled, answered, resumed,
    cancelled or swept, inside a run or outside one.

    COUNTS AND IDS ONLY, and a fixed size. A client that sees a new revision
    fetches the plan from the host (``runtime.plan(session).current()``). No
    title, instruction, note, question or answer travels here. ``run_id`` is the
    run that made the write, or '' for a host call made outside a run.
    """

    type: str = field(default="plan.updated", init=False)
    run_id: str = ""
    session_key: str = ""
    plan_id: str = ""
    revision: int = 0
    status: str = ""
    holding_approval_id: str | None = None
    #: ``{total, todo, running, held, waiting, done, failed, skipped}``.
    counts: dict[str, int] = field(default_factory=dict)
    harness_agent: str | None = None


@dataclass(frozen=True)
class _ToolDecision:
    """One decision about a call to a tool that needs approval (G-326).

    Carries the arguments' DIGEST, never the arguments or the result: the
    sha256 of their RFC 8785 canonical JSON, identical in every language.
    ``policy_revision`` is the host policy's revision when the decision was made.
    """

    run_id: str = ""
    session_key: str = ""
    policy_revision: str = ""
    server: str | None = None
    tool: str = ""
    args_digest: str = ""
    #: ``policy`` or ``person``.
    decided_by: str = ""
    harness_agent: str | None = None


@dataclass(frozen=True)
class ToolAutoAllowed(_ToolDecision):
    """The host policy let the call run without asking."""

    type: str = field(default="tool_auto_allowed", init=False)


@dataclass(frozen=True)
class ToolApprovalAsked(_ToolDecision):
    """The call is waiting on a person."""

    type: str = field(default="tool_approval_asked", init=False)


@dataclass(frozen=True)
class ToolApproved(_ToolDecision):
    """A person approved the call; it runs now."""

    type: str = field(default="tool_approved", init=False)


@dataclass(frozen=True)
class ToolDenied(_ToolDecision):
    """Not run: denied or unanswered (``person``), or revoked before it began (``policy``)."""

    type: str = field(default="tool_denied", init=False)


@dataclass(frozen=True)
class ToolApprovalExpired(_ToolDecision):
    """Not run: the approval was stored by 0.5.2 or earlier without a marker, on
    a call stored as no arguments, which cannot be told apart from a garbled
    call. The model is told to call again. ``args_digest`` is of no arguments.
    """

    type: str = field(default="tool_approval_expired", init=False)


@dataclass(frozen=True)
class PlanBudgetExhausted:
    """A plan stopped because its shared budget is spent (G-366). Emitted once,
    when advance() blocks the plan for it; resume() refuses such a plan. Counts
    only; no plan text."""

    type: str = field(default="plan.budget_exhausted", init=False)
    session_key: str = ""
    plan_id: str = ""
    #: Always ``plan_budget_exhausted``, the code all three languages use.
    code: str = "plan_budget_exhausted"
    #: Which limit: ``steps``, ``cost`` or ``cost_unmetered``.
    limit: str = ""
    steps: int = 0
    cost_usd: float = 0.0
    harness_agent: str | None = None


HarnessEvent = (
    RunStarted
    | RunFinished
    | RunFailed
    | RunCancelRequested
    | RunCancelled
    | ToolAutoAllowed
    | ToolApprovalAsked
    | ToolApproved
    | ToolDenied
    | ToolApprovalExpired
    | PlanUpdated
    | PlanBudgetExhausted
)
HarnessListener = Callable[[HarnessEvent], None]


class HarnessEvents:
    """A plain listener list rather than an event bus: this package has no
    framework to hang one off, and a consumer that wants queues or broadcasting
    already has somewhere better to put them. The point is that the harness
    EMITS at the right moments, not that it owns the delivery mechanism.

    NO PROMPTS, NO TOOL ARGUMENTS, and that boundary is the same one
    ``Session.complete_run()`` holds: tool NAMES are what an operator needs to
    audit a guardrail and are not PII, while arguments are --
    ``prism-opentelemetry`` already carries those behind an opt-in capture gate
    with a length cap, and emitting them here ungated would quietly undo that
    decision for anyone who installed both.
    """

    def __init__(self) -> None:
        self._listeners: list[HarnessListener] = []

    def listen(self, listener: HarnessListener) -> Callable[[], None]:
        """Returns an unsubscribe callable, so a caller can stop listening."""
        self._listeners.append(listener)

        def stop() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return stop

    def emit(self, event: HarnessEvent) -> None:
        """Deliver an event to every listener.

        A RAISING LISTENER MUST NOT BREAK THE RUN. Telemetry that takes down the
        thing it observes is worse than no telemetry, and a listener is by
        definition somebody else's code. Failures are collected and warned about
        after every listener has had its turn.
        """
        failures: list[BaseException] = []

        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as error:  # noqa: BLE001 - a listener is someone else's code
                failures.append(error)

        for failure in failures:
            warnings.warn(
                f"A prism-harness event listener raised while handling [{event.type}]: {failure}",
                stacklevel=2,
            )

    @staticmethod
    def to_dict(event: HarnessEvent) -> dict[str, Any]:
        """For a consumer that wants to persist or forward the event as data."""
        data = {"type": event.type, **asdict(event)}
        if isinstance(event, _ToolDecision):
            return data
        if data.get("harness_agent") is None:
            data.pop("harness_agent", None)
        return data
