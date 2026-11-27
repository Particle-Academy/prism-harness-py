"""G-354: one approval, one answer, and every reader agrees which answer counts.

Two decisions for the same approval id (a double click, two reviewers, a retry
racing the original) must not both be recorded. And a thread that already holds
two -- written before this was enforced -- is read the same way everywhere:
the FIRST recorded decision is the one that counts, except that ANY DENY IS
FINAL (G-401): a deny recorded after an approve revokes it.
"""

from __future__ import annotations

import socket
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    Session,
    ToolRegistry,
    record_approval,
)
from prism_harness.thread_rows import pending_approval_ids, tool_result_row

MODES = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo"],
                "max_steps": 4,
                "requires_approval": ["echo"],
            },
        },
    }
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


class Echo:
    def __init__(self, ran: list[str]) -> None:
        self._ran = ran

    @property
    def name(self) -> str:
        return "echo"

    def handle(self, args: dict[str, Any]) -> Any:
        self._ran.append(str(args.get("value")))
        return "echoed"


class Rig:
    def __init__(self) -> None:
        directory = tempfile.mkdtemp(prefix="prism-harness-decisions-")
        harness = PrismHarness(
            drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "memory", "durable": "files"},
        )
        self.session: Session = harness.for_(Participant("User", 1)).session("decisions")
        self.requests: list[LlmRequest] = []
        self.ran: list[str] = []
        self.agent = AgentRuntime(
            client=self._client(),
            modes=MODES,
            tools=ToolRegistry().register(Echo(self.ran)),
            config={"provider": "anthropic", "model": "m"},
        )

    def _client(self) -> Callable[[LlmRequest], LlmResponse]:
        def client(request: LlmRequest) -> LlmResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return LlmResponse(
                    text="",
                    finish_reason="tool_calls",
                    tool_calls=[LlmToolCall(id="c1", name="echo", arguments={"value": "x"})],
                )
            return LlmResponse(text="done", finish_reason="stop")

        return client

    def pause(self) -> str:
        return self.agent.send(self.session, "go").pending_approvals[0].id

    def decisions(self) -> list[dict[str, Any]]:
        return [
            decision
            for entry in self.session.thread().messages()
            if entry.message.get("type") == "tool_result"
            for decision in entry.message.get("tool_approval_responses", [])
        ]


def test_two_concurrent_decisions_on_one_id_record_exactly_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig()
    approval = rig.pause()

    # Hold each decision between its pending check and its write, so both
    # checks happen before either write unless the two are serialised.
    barrier = threading.Barrier(2, timeout=2)
    original = pending_approval_ids

    def held(rows: Sequence[Any]) -> list[str]:
        pending = original(rows)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return pending

    monkeypatch.setattr("prism_harness.runtime.pending_approval_ids", held)

    outcomes: dict[bool, str] = {}

    def decide(approved: bool) -> None:
        try:
            record_approval(rig.session, approval, approved, None if approved else "no")
            outcomes[approved] = "recorded"
        except HarnessError as error:
            outcomes[approved] = error.code

    threads = [threading.Thread(target=decide, args=(answer,)) for answer in (False, True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert sorted(outcomes.values()) == ["approval_not_pending", "recorded"]
    assert len(rig.decisions()) == 1

    monkeypatch.setattr("prism_harness.runtime.pending_approval_ids", original)
    rig.agent.send(rig.session, "")
    recorded = rig.decisions()[0]["approved"]
    assert rig.ran == (["x"] if recorded else [])


def _record(rig: Rig, decisions: list[dict[str, Any]], separate: bool) -> None:
    run = rig.session.run()
    run_id = run["id"] if run else None
    if separate:
        for decision in decisions:
            rig.session.thread().record([tool_result_row([], [decision])], run_id)
    else:
        rig.session.thread().record([tool_result_row([], decisions)], run_id)


@pytest.mark.parametrize("separate", [True, False], ids=["two rows", "one row"])
def test_a_deny_recorded_before_an_approve_stands(separate: bool) -> None:
    rig = Rig()
    approval = rig.pause()
    _record(
        rig,
        [
            {"approval_id": approval, "approved": False, "reason": "no"},
            {"approval_id": approval, "approved": True, "reason": None},
        ],
        separate,
    )

    rig.agent.send(rig.session, "")

    assert rig.ran == []
    # The provider replay view still folds a later decision over an earlier one
    # (the thread rows corpus pins that); it decides nothing.
    assert "no" in str(rig.requests[-1].messages)


def test_a_deny_recorded_after_an_approve_revokes_it() -> None:
    """ANY DENY IS FINAL (G-401, the recommended rule, G-387 round 15): a
    deny recorded after an approve that has not been acted on revokes it.
    This pinned "the approve stands" before round 15."""
    rig = Rig()
    approval = rig.pause()
    _record(
        rig,
        [
            {"approval_id": approval, "approved": True, "reason": None},
            {"approval_id": approval, "approved": False, "reason": "no"},
        ],
        True,
    )

    rig.agent.send(rig.session, "")

    assert rig.ran == []
