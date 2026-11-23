"""What a turn holding a long argument list costs."""

from __future__ import annotations

import json
import socket
import tempfile
import tracemalloc
from collections.abc import Callable
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
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
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _peak(action: Callable[[], Any]) -> tuple[int, Any]:
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        result = action()

        return tracemalloc.get_traced_memory()[1] - before, result
    finally:
        tracemalloc.stop()


def _turn(arguments: Any) -> tuple[str, int, int]:
    directory = tempfile.mkdtemp(prefix="prism-arguments-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = (
        harness.for_(Participant("User", 1))
        .session("long")
        .using_mode("chat")
        .using_provider("synthetic")
        .using_model("offline")
    )
    replies = [
        LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall("call-1", "lookup", arguments)],
        ),
        LlmResponse(text="done", finish_reason="stop"),
    ]
    model_calls = 0
    tool_runs = 0

    class Lookup:
        name = "lookup"

        def handle(self, _args: dict[str, Any]) -> str:
            nonlocal tool_runs
            tool_runs += 1
            return "ran"

    def client(_request: LlmRequest) -> LlmResponse:
        nonlocal model_calls
        model_calls += 1
        return replies.pop(0)

    tools = ToolRegistry()
    tools.register(Lookup())
    runtime = AgentRuntime(
        client=client,
        tools=tools,
        modes=ModeRegistry(
            {"default": "chat", "modes": {"chat": {"tools": ["lookup"], "max_steps": 4}}}
        ),
    )
    finish = runtime.send(session, "Look it up.").finish_reason

    return finish, model_calls, tool_runs


def test_completes_a_turn_in_a_bounded_multiple_of_reading_the_list() -> None:
    text = '{"a":[' + ",".join(["0"] * 100_000) + "]}"
    reading, _ = _peak(lambda: json.loads(text))
    turn, outcome = _peak(lambda: _turn(text))

    assert outcome == ("stop", 2, 1)
    assert turn / reading < 9
