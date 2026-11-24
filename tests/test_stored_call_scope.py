"""A resume reads only the turn it resumes.

Providers that number calls per response reuse call ids. A call stored with
unusable arguments in an OLDER turn must not refuse a later, valid call that
happens to carry the same id, any more than an older turn's approval request may
decide a newer one.
"""

from __future__ import annotations

import socket
import tempfile
from collections.abc import Iterator
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
from prism_harness.thread_rows import unusable_stored_calls

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


OLD_TURN: list[dict[str, Any]] = [
    {"type": "user", "content": "earlier"},
    {
        "type": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "name": "echo",
                "arguments": "[1, 2",
                "result_id": None,
                "reasoning_id": None,
                "reasoning_summary": None,
            }
        ],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "call_1",
                "tool_name": "echo",
                "args": {},
                "result": "Not run: the arguments for echo are not a JSON object.",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    },
    {"type": "assistant", "content": "Sorry.", "tool_calls": [], "additional_content": {}},
]


def test_an_older_unusable_call_with_a_reused_id_does_not_refuse_a_later_approved_call() -> None:
    directory = tempfile.mkdtemp(prefix="prism-harness-n7-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("n7")
    session.thread().record(OLD_TURN, None)
    ran: list[str] = []
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        if len(requests) == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall(id="call_1", name="echo", arguments={"value": "now"})],
            )
        return LlmResponse(text="done", finish_reason="stop")

    agent = AgentRuntime(
        client=client,
        modes=MODES,
        tools=ToolRegistry().register(Echo(ran)),
        config={"provider": "anthropic", "model": "m"},
    )

    pending = agent.send(session, "again").pending_approvals[0]
    record_approval(session, pending.id, True)
    agent.send(session, "")

    assert ran == ["now"]


def test_only_the_last_tool_calling_turn_is_read() -> None:
    later = {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"value": "ok"}}],
        "tool_approval_requests": [],
    }
    assert unusable_stored_calls(OLD_TURN) == {"call_1"}
    assert unusable_stored_calls([*OLD_TURN, {"type": "user", "content": "x"}, later]) == set()
