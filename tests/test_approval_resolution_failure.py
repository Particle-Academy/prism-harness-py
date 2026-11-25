"""A tool that fails while approvals are resolved never re-runs an earlier call.

When a resumed run acts on the person's decisions, every approved call runs
once and the results are recorded together. A tool whose result cannot even be
written down (here, a value that is not JSON) must be recorded as a failed
result like any other tool failure. If it escaped, nothing would be recorded,
and the next resume would run the approved calls before it a second time.

Driven through ``send`` and ``decide`` with a scripted model; the network is
refused and every store is in memory.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_plan_budget import DurableMemoryStore

from prism_harness import (
    AgentRuntime,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    ToolRegistry,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


class Tool:
    def __init__(self, name: str, result: object, ran: list[str]) -> None:
        self._name = name
        self._result = result
        self._ran = ran

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        self._ran.append(self._name)
        return self._result


def test_an_unrecordable_tool_result_never_re_runs_the_earlier_approved_call() -> None:
    ran: list[str] = []
    queue: list[LlmResponse] = [
        LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[
                LlmToolCall(id="call_1", name="transfer", arguments={"to": "a"}),
                LlmToolCall(id="call_2", name="odd", arguments={}),
            ],
        ),
        LlmResponse(text="Done.", finish_reason="stop"),
        LlmResponse(text="Done again.", finish_reason="stop"),
    ]

    def client(request: LlmRequest) -> LlmResponse:
        return queue.pop(0)

    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "durable": DurableMemoryStore},
        stores={"ephemeral": "memory", "durable": "durable"},
    )
    runtime = AgentRuntime(
        client=client,
        modes=ModeRegistry(
            {
                "default": "chat",
                "modes": {
                    "chat": {
                        "system_prompt": "Act.",
                        "tools": ["transfer", "odd"],
                        "requires_approval": ["transfer", "odd"],
                        "max_steps": 6,
                    }
                },
            }
        ),
        tools=ToolRegistry()
        .register(Tool("transfer", "Sent", ran))
        .register(Tool("odd", object(), ran)),
        config={"provider": "openai", "model": "root-model"},
    )
    session = harness.for_(Participant("User", "ada")).session("chat")
    parked = runtime.send(session, "Go")
    for approval in parked.pending_approvals:
        runtime.decide(session, approval.id, True)

    for _ in range(2):
        try:
            runtime.send(session, "")
        except TypeError:
            pass

    assert ran.count("transfer") == 1
