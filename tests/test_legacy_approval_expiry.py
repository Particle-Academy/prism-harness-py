"""Approvals stored without a marker on a call stored as no arguments expire.

prism-ai-harness 0.5.1 and earlier, fed by prism-ai-core 0.10.0 and earlier,
stored a gated call whose arguments arrived as a bare number or boolean as
``{}``, the same as a real call with no arguments, and wrote no marker. Such an
approval is expired at resume: the call does not run, and the model is told to
call again.

The rows in fixtures/legacy-approvals-0.5.1.json were written by the released
prism-ai-harness 0.5.1, from what the released prism-ai-core 0.10.0 parsed.
"""

from __future__ import annotations

import json
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessEvent,
    HarnessEvents,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    Session,
    ToolApprovalExpired,
    ToolRegistry,
    record_approval,
)

LEGACY: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy-approvals-0.5.1.json").read_text(encoding="utf-8")
)
EXPIRED = "has expired"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def _session() -> Session:
    directory = tempfile.mkdtemp(prefix="prism-g351-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    return (
        harness.for_(Participant("User", 1))
        .session("stored")
        .using_mode("chat")
        .using_provider("synthetic")
        .using_model("offline")
    )


class _Run:
    def __init__(self, responses: list[LlmResponse]) -> None:
        self.received: list[Any] = []
        self.requests: list[LlmRequest] = []
        self.seen: list[HarnessEvent] = []
        events = HarnessEvents()
        events.listen(self.seen.append)
        received = self.received

        class Lookup:
            name = "lookup"

            def handle(self, args: dict[str, Any]) -> str:
                received.append(args)
                return "ran"

        def client(request: LlmRequest) -> LlmResponse:
            self.requests.append(request)
            return responses.pop(0) if responses else LlmResponse(text="done", finish_reason="stop")

        self.agent = AgentRuntime(
            client=client,
            tools=ToolRegistry().register(Lookup()),
            events=events,
            modes=ModeRegistry(
                {
                    "default": "chat",
                    "modes": {
                        "chat": {
                            "tools": ["lookup"],
                            "max_steps": 4,
                            "requires_approval": ["lookup"],
                        }
                    },
                }
            ),
        )


def _resume(
    rows: list[dict[str, Any]],
    approval_id: str,
    approved: bool = True,
    responses: list[LlmResponse] | None = None,
) -> tuple[_Run, Any, list[dict[str, Any]]]:
    session = _session()
    session.thread().record(rows)
    run = _Run(responses or [])
    record_approval(session, approval_id, approved)
    response = run.agent.send(session, "")
    stored = [entry.message for entry in session.thread().messages()]
    return run, response, stored


def _stored_row(arguments: Any, request: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"type": "user", "content": "Look it up."},
        {
            "type": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "lookup",
                    "arguments": arguments,
                    "result_id": None,
                    "reasoning_id": None,
                    "reasoning_summary": None,
                }
            ],
            "additional_content": {},
            "tool_approval_requests": [
                {"approval_id": "apr-1", "tool_call_id": "call-1", **request}
            ],
            "harness_tool_names": ["lookup"],
        },
    ]


@pytest.mark.parametrize("form", ["mistral_wire_number", "anthropic_wire_false"])
def test_expires_the_row_the_released_version_wrote(form: str) -> None:
    run, response, stored = _resume(LEGACY[form]["rows"], LEGACY[form]["approval_id"])

    assert run.received == []
    assert response.finish_reason == "stop"
    sent = json.dumps(run.requests[0].messages)
    assert EXPIRED in sent
    assert "Call the tool again" in sent

    results = [
        entry
        for row in stored
        if row.get("type") == "tool_result"
        for entry in row.get("tool_results", [])
    ]
    assert len(results) == 1
    assert results[0]["args"] == {}
    assert EXPIRED in results[0]["result"]

    types = [event.type for event in run.seen]
    assert "tool_approved" not in types
    expired = [event for event in run.seen if event.type == "tool_approval_expired"]
    assert len(expired) == 1
    event = expired[0]
    assert isinstance(event, ToolApprovalExpired)
    assert not hasattr(event, "arguments")
    assert event.tool == "lookup"
    assert event.decided_by == "policy"


def test_asks_the_person_again_when_the_model_calls_again() -> None:
    legacy = LEGACY["mistral_wire_number"]
    run, response, stored = _resume(
        legacy["rows"],
        legacy["approval_id"],
        responses=[
            LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("call-2", "lookup", {})],
            )
        ],
    )

    assert run.received == []
    assert response.finish_reason == "awaiting_approval"
    assert len(response.pending_approvals) == 1
    assistant = [row for row in stored if row.get("type") == "assistant"][-1]
    assert assistant["tool_approval_requests"][0]["tool_call_id"] == "call-2"
    assert assistant["tool_approval_requests"][0].get("arguments_format") == 1


@pytest.mark.parametrize("marker", ["1", 0, 99, True, None, 1.0])
def test_treats_a_marker_it_does_not_recognise_as_none(marker: object) -> None:
    run, _response, _stored = _resume(_stored_row({}, {"arguments_format": marker}), "apr-1")

    assert run.received == []
    assert EXPIRED in json.dumps(run.requests[0].messages)


def test_leaves_a_denied_one_denied() -> None:
    legacy = LEGACY["anthropic_wire_false"]
    run, _response, _stored = _resume(legacy["rows"], legacy["approval_id"], approved=False)

    assert run.received == []
    assert EXPIRED not in json.dumps(run.requests[0].messages)
    assert "tool_denied" in [event.type for event in run.seen]


def test_runs_a_marked_approval_on_a_call_with_no_arguments() -> None:
    session = _session()
    first = _Run(
        [
            LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("call-1", "lookup", {})],
            )
        ]
    )
    paused = first.agent.send(session, "Look it up.")
    assistant = next(
        entry.message
        for entry in session.thread().messages()
        if entry.message.get("type") == "assistant"
    )
    assert assistant["tool_approval_requests"][0].get("arguments_format") == 1

    second = _Run([])
    record_approval(session, paused.pending_approvals[0].id, True)
    second.agent.send(session, "")

    assert second.received == [{}]
    assert "tool_approved" in [event.type for event in second.seen]


def test_runs_an_unmarked_approval_on_a_call_with_named_arguments() -> None:
    run, _response, _stored = _resume(_stored_row({"city": "Paris"}, {}), "apr-1")

    assert run.received == [{"city": "Paris"}]


def test_runs_a_marked_approval_on_a_call_id_an_older_unmarked_request_used() -> None:
    # Some providers number calls per response, so ids repeat across turns.
    # Only the turn being resumed decides whether an approval is ambiguous.
    session = _session()
    session.thread().record(
        [
            *_stored_row({}, {}),
            {
                "type": "tool_result",
                "tool_results": [
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "lookup",
                        "args": {},
                        "result": "done before",
                        "tool_call_result_id": None,
                        "artifacts": [],
                    }
                ],
                "tool_approval_responses": [
                    {"approval_id": "apr-1", "approved": False, "reason": None}
                ],
            },
            {
                "type": "assistant",
                "content": "ok",
                "tool_calls": [],
                "additional_content": {},
                "tool_approval_requests": [],
            },
        ]
    )
    first = _Run(
        [
            LlmResponse(
                text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("call-1", "lookup", {})]
            )
        ]
    )
    paused = first.agent.send(session, "Again.")
    assert len(paused.pending_approvals) == 1

    second = _Run([])
    record_approval(session, paused.pending_approvals[0].id, True)
    second.agent.send(session, "")

    assert second.received == [{}]
    assert "tool_approval_expired" not in [event.type for event in second.seen]
