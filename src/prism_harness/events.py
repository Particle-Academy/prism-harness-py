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
    "RunFailed",
    "RunFinished",
    "RunStarted",
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


@dataclass(frozen=True)
class RunFailed:
    type: str = field(default="run.failed", init=False)
    run_id: str = ""
    session_key: str = ""
    #: Why it stopped: a budget reason, a cancellation, or a provider failure.
    failure: str = ""
    steps: int = 0
    at: str = ""


HarnessEvent = RunStarted | RunFinished | RunFailed
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
        return {"type": event.type, **asdict(event)}
