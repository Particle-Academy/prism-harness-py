"""A tool call whose arguments are not an object, as the loop handles it."""

from __future__ import annotations

import json
import tempfile
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


def _run(arguments: Any, requires_approval: list[str] | None = None) -> dict[str, Any]:
    directory = tempfile.mkdtemp(prefix="prism-malformed-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = (
        harness.for_(Participant("User", 1))
        .session("malformed")
        .using_mode("chat")
        .using_provider("synthetic")
        .using_model("offline")
    )

    received: list[Any] = []
    requests: list[LlmRequest] = []
    responses = [
        LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall("call-1", "lookup", arguments)],
        ),
        LlmResponse(text="done", finish_reason="stop"),
    ]

    class Lookup:
        name = "lookup"

        def handle(self, args: dict[str, Any]) -> str:
            received.append(args)
            return "ran"

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        assert responses, "Unexpected extra model call"
        return responses.pop(0)

    tools = ToolRegistry()
    tools.register(Lookup())
    runtime = AgentRuntime(
        client=client,
        tools=tools,
        modes=ModeRegistry(
            {
                "default": "chat",
                "modes": {
                    "chat": {
                        "tools": ["lookup"],
                        "max_steps": 4,
                        "requires_approval": requires_approval or [],
                    }
                },
            }
        ),
    )

    response = None
    crash = None
    try:
        response = runtime.send(session, "Look it up.")
    except (TypeError, ValueError) as error:
        crash = f"{type(error).__name__}: {error}"

    rows = [entry.message for entry in session.thread().messages()]

    return {
        "crash": crash,
        "received": received,
        "requests": requests,
        "response": response,
        "rows": rows,
    }


@pytest.mark.parametrize("arguments", ['{"city": ', [1, 2], 5], ids=["text", "list", "number"])
def test_reports_it_as_a_tool_error_without_running_the_tool(arguments: Any) -> None:
    outcome = _run(arguments)

    assert outcome["crash"] is None
    assert outcome["received"] == []
    assert outcome["response"].finish_reason == "stop"
    assert len(outcome["requests"]) == 2
    assert "Not run: the arguments for lookup are not a JSON object." in json.dumps(
        outcome["requests"][1].messages
    )

    assistant = next(row for row in outcome["rows"] if row["type"] == "assistant")
    assert assistant["tool_calls"][0]["arguments"] == {}


def test_does_not_ask_a_person_to_approve_it() -> None:
    outcome = _run('{"city": ', ["lookup"])

    assert outcome["crash"] is None
    assert outcome["response"].pending_approvals == []
    assert outcome["response"].finish_reason == "stop"
    assert outcome["received"] == []


def test_runs_a_well_formed_call_as_before() -> None:
    outcome = _run({"city": "Paris"})

    assert outcome["crash"] is None
    assert outcome["received"] == [{"city": "Paris"}]
    assert outcome["response"].finish_reason == "stop"
