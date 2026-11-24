"""A tool call whose arguments are not an object, as the loop handles it."""

from __future__ import annotations

import json
import socket
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
    record_approval,
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


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client below is a fake; opening a socket is a bug."""

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.mark.parametrize(
    "arguments",
    ["", "0", "null", "[]", "{}", None, [], {}],
    ids=["empty", "zero-text", "null-text", "list-text", "object-text", "none", "list", "object"],
)
def test_runs_an_empty_form_as_no_arguments(arguments: Any, no_network: None) -> None:
    outcome = _run(arguments)

    assert outcome["crash"] is None
    assert len(outcome["requests"]) == 2
    assert outcome["received"] == [{}]


def test_reads_json_text_naming_values(no_network: None) -> None:
    assert _run('{"city":"Paris"}')["received"] == [{"city": "Paris"}]


@pytest.mark.parametrize(
    "arguments",
    [
        {"0": "x"},
        '{"0":"x"}',
        '{"amount":1e999}',
        '{"amount":1' + "0" * 400 + "}",
        '{"amount":"5","\\u0000a":1}',
        {"a": {"\x00b": 1}},
        '{"amount":1\x0100}',
        "nu\x01ll",
        True,
        0,
        False,
        '{"a":' * 501 + "1" + "}" * 501,
    ],
    ids=[
        "integer-key",
        "integer-key-text",
        "past-float-range",
        "integer-past-float-range",
        "nul-key",
        "nested-nul-key",
        "control-in-number",
        "control-in-literal",
        "true",
        "zero",
        "false",
        "depth-501",
    ],
)
def test_reports_an_unusable_value_without_running(arguments: Any, no_network: None) -> None:
    outcome = _run(arguments)

    assert outcome["crash"] is None
    assert outcome["received"] == []
    assert len(outcome["requests"]) == 2
    assert "Not run: the arguments for lookup are not a JSON object." in json.dumps(
        outcome["requests"][1].messages
    )

    assistant = next(row for row in outcome["rows"] if row["type"] == "assistant")
    assert assistant["tool_calls"][0]["arguments"] == {}


def test_keeps_a_control_character_and_a_long_digit_key(no_network: None) -> None:
    key = "7" * 5000
    outcome = _run('{"note":"a\x01b","' + key + '":1}')

    assert outcome["crash"] is None
    assert outcome["received"] == [{"note": "a\x01b", key: 1}]


def _resume_stored(stored: Any) -> dict[str, Any]:
    directory = tempfile.mkdtemp(prefix="prism-stored-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = (
        harness.for_(Participant("User", 1))
        .session("stored")
        .using_mode("chat")
        .using_provider("synthetic")
        .using_model("offline")
    )
    session.thread().record(
        [
            {
                "type": "user",
                "content": "Look it up.",
                "additional_content": [],
                "additional_attributes": {},
            },
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": stored,
                        "result_id": None,
                        "reasoning_id": None,
                        "reasoning_summary": None,
                    }
                ],
                "additional_content": {},
                "tool_approval_requests": [
                    {"approval_id": "apr-1", "tool_call_id": "call-1", "arguments_format": 1}
                ],
            },
        ]
    )

    received: list[Any] = []
    requests: list[LlmRequest] = []

    class Lookup:
        name = "lookup"

        def handle(self, args: dict[str, Any]) -> str:
            received.append(args)
            return "ran"

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="done", finish_reason="stop")

    tools = ToolRegistry()
    tools.register(Lookup())
    runtime = AgentRuntime(
        client=client,
        tools=tools,
        modes=ModeRegistry(
            {
                "default": "chat",
                "modes": {
                    "chat": {"tools": ["lookup"], "max_steps": 4, "requires_approval": ["lookup"]}
                },
            }
        ),
    )

    record_approval(session, "apr-1", True)
    runtime.send(session, "")

    return {"received": received, "requests": requests}


@pytest.mark.parametrize(
    "stored",
    [
        {str(index): char for index, char in enumerate('{"city":"Par')},
        '{"city":"Par',
        5,
        True,
        [1, 2],
        {"0": "x"},
    ],
    ids=["split-text", "text", "number", "true", "list", "integer-key"],
)
def test_refuses_an_approved_stored_call(stored: Any, no_network: None) -> None:
    outcome = _resume_stored(stored)

    assert outcome["received"] == []
    assert "Not run: the arguments for lookup are not a JSON object." in json.dumps(
        outcome["requests"][0].messages
    )


@pytest.mark.parametrize(
    ("stored", "expected"),
    [({"city": "Paris"}, {"city": "Paris"}), ({}, {}), (None, {})],
    ids=["named", "none", "null"],
)
def test_still_runs_an_approved_stored_call(
    stored: Any, expected: dict[str, Any], no_network: None
) -> None:
    assert _resume_stored(stored)["received"] == [expected]


@pytest.mark.parametrize(
    "arguments",
    [
        '{"0":"prod"}',
        '["prod"]',
        '{"0":"prod","note":"x"}',
        '{"note":"x","1":"prod"}',
        '{"-1":"prod"}',
        {"0": "prod", "note": "x"},
    ],
)
def test_runs_nothing_for_a_key_that_could_bind_by_position(
    arguments: Any, no_network: None
) -> None:
    outcome = _run(arguments, ["lookup"])

    assert outcome["crash"] is None
    assert outcome["received"] == []
    assert outcome["response"].pending_approvals == []


@pytest.mark.parametrize(
    "arguments",
    ['{"00":"prod"}', '{"1.0":"prod"}', '{"-0":"prod"}', '{"+1":"prod"}', '{" 1":"prod"}'],
)
def test_hands_a_numeric_looking_key_over_by_name(arguments: str, no_network: None) -> None:
    assert _run(arguments)["received"] == [json.loads(arguments)]
