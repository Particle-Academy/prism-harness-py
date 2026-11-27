""" "Interrupted" is an answer in the replay only, never a stored outcome (G-410).

A call the harness recorded as started, with no result, is shown to the model
as "Tool call was interrupted, it may have run". That answer is built for the
request and never stored: the call's real result, when it lands, is its one
stored outcome. Two workers share one session through a host lock driver that
honours ``ttl_seconds`` (a lease that lapses); the approved call outlives the
lease, and a second worker resumes while it runs. Entered at the public seam:
``PrismHarness`` drivers, ``AgentRuntime.send`` and ``record_approval``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import Any, TypeVar

import pytest
from test_runtime import GUARDED, a_session, guarded_runtime, once

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    LlmRequest,
    LlmToolCall,
    MemorySessionStore,
    Participant,
    PrismHarness,
    Session,
    ToolRegistry,
    record_approval,
)
from prism_harness.thread import Thread

T = TypeVar("T")
INTERRUPTED = "Tool call was interrupted, it may have run"


class LeaseLocks(MemorySessionStore):
    """A host lock driver whose lease lapses after ``ttl_seconds`` on a clock."""

    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self._clock = clock
        self._held: dict[str, float] = {}

    def with_lock(
        self,
        key: str,
        callback: Callable[[], T],
        ttl_seconds: float = 10,
        wait_seconds: float = 5,
    ) -> T:
        until = self._held.get(key)
        if until is not None and self._clock[0] < until:
            raise HarnessError.session_locked(key, wait_seconds)
        self._held[key] = self._clock[0] + ttl_seconds
        try:
            return callback()
        finally:
            self._held.pop(key, None)


def outcomes(session: Session, call_id: str) -> list[str]:
    return [
        e["result"]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result"
        for e in m.message["tool_results"]
        if e["tool_call_id"] == call_id
    ]


class Echo:
    def __init__(self, during: Callable[[], None] | None = None) -> None:
        self.runs = 0
        self._during = during

    @property
    def name(self) -> str:
        return "echo"

    def handle(self, args: dict[str, Any]) -> Any:
        self.runs += 1
        if self._during is not None:
            self._during()
        return f"echo:{args.get('value', '')}"


class Shout:
    @property
    def name(self) -> str:
        return "shout"

    def handle(self, args: dict[str, Any]) -> Any:
        return "shout"


def test_a_second_worker_never_stores_interrupted_for_a_call_still_running() -> None:
    clock = [0.0]
    locks = LeaseLocks(clock)
    directory = tempfile.mkdtemp(prefix="prism-harness-lease-")

    def worker_session() -> Session:
        harness = PrismHarness(
            drivers={"memory": lambda: locks, "files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "memory", "durable": "files"},
        )
        session = harness.for_(Participant("User", 1)).session("support")
        session.using_mode("guarded").using_provider("anthropic").using_model("m")
        return session

    second_requests: list[LlmRequest] = []
    second_echo = Echo()

    def second_worker() -> None:
        clock[0] += 1000  # the call outlives the first worker's lease
        AgentRuntime(
            client=once([], second_requests),
            modes=GUARDED,
            tools=ToolRegistry().register(second_echo).register(Shout()),
        ).send(worker_session(), "")

    first = worker_session()
    runtime = AgentRuntime(
        client=once([LlmToolCall("p", "echo", {"value": "P"})]),
        modes=GUARDED,
        tools=ToolRegistry().register(Echo(second_worker)).register(Shout()),
    )
    approval = runtime.send(first, "go").pending_approvals[0].id
    record_approval(first, approval, True)
    runtime.send(first, "")

    assert second_echo.runs == 0
    # The second worker was shown the call as interrupted, and stored nothing.
    assert any(
        e["tool_call_id"] == "p" and e["result"] == INTERRUPTED
        for request in second_requests
        for m in request.messages
        if m["type"] == "tool_result"
        for e in m["tool_results"]
    )
    # The real result is the call's one stored outcome.
    assert outcomes(worker_session(), "p") == ["echo:P"]


def test_a_call_whose_result_write_failed_stores_no_outcome_and_replays_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})], requests), counts)
    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)

    native = Thread.record
    failed: list[bool] = []

    def record(self: Thread, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if not failed and any(
            m.get("type") == "tool_result"
            and any(e.get("result") == "echo:P" for e in m.get("tool_results", []))
            for m in messages
        ):
            failed.append(True)
            raise OSError("the store refused the write")
        return native(self, messages, *args, **kwargs)

    monkeypatch.setattr(Thread, "record", record)
    with pytest.raises(OSError):
        runtime.send(session, "")
    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {"echo": 1}
    assert outcomes(session, "p") == []
    replayed = [
        [e["result"] for m in r.messages if m["type"] == "tool_result" for e in m["tool_results"]]
        for r in requests[1:]
    ]
    assert replayed and all(results == [INTERRUPTED] for results in replayed)
