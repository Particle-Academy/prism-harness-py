"""The PHP round-13 review's items, checked in Python (G-387 round 14, items 6-8).

- G-405: a STRUCTURED turn parked for approval carries NO document. The fake
  is Anthropic-shaped: the provider keeps the model's text (here a JSON text
  with a bad type, and its parse) on the step that also calls a gated tool.
  An OpenAI-shaped fake that drops the text on a tool-call step hides it.
- G-406: the runtime's own in-run recording is scoped to exactly that record
  call. A host row recorded INSIDE the runtime's write window (here from the
  store, re-entering on the same thread) naming the run id is outside, like
  any caller's.
- Item 8: a new message denies approved calls that have not run yet, even
  when that turn then fails.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from test_repeated_call_ids import Recording, park, rows
from test_structured import a_session, scripted

from prism_harness import (
    AgentRuntime,
    HarnessError,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    ModeRegistry,
    StructuredAgentResponse,
    ToolRegistry,
    record_approval,
)
from prism_harness.stores.file import FileSessionStore
from prism_harness.thread_rows import SUPERSEDED, SUPERSEDED_RESULT, pending_approval_ids


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


# -- G-405: a parked structured turn has no document ------------------------------

GATED = ModeRegistry(
    {
        "default": "chat",
        "modes": {
            "chat": {
                "system_prompt": "Be brief.",
                "tools": ["echo", "lookup"],
                "max_steps": 4,
                "requires_approval": ["echo"],
            }
        },
    }
)

SCHEMA: dict[str, Any] = {
    "name": "answer",
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}

#: Valid JSON, the wrong type for the schema: never a document a caller may see.
BAD = {"n": "not a number"}


class Tool:
    def __init__(self, name: str) -> None:
        self.name = name

    def handle(self, args: dict[str, Any]) -> Any:
        return f"{self.name}:ran"


def runtime(client: Callable[[LlmRequest], LlmResponse]) -> AgentRuntime:
    return AgentRuntime(
        client=client,
        modes=GATED,
        tools=ToolRegistry().register(Tool("echo")).register(Tool("lookup")),
    )


def anthropic_step(*calls: LlmToolCall, structured: Any = BAD) -> LlmResponse:
    """A tool-call step as an Anthropic-shaped provider returns it: the text
    block the model wrote is kept beside the tool_use blocks."""
    return LlmResponse(
        text=json.dumps(BAD),
        finish_reason="tool_calls",
        tool_calls=list(calls),
        structured=structured,
    )


@pytest.mark.parametrize("parsed", ["with_parse", "text_only"])
def test_a_structured_first_step_that_parks_has_no_document(parsed: str) -> None:
    session = a_session()
    client = scripted(
        [anthropic_step(LlmToolCall("c1", "echo", {}), structured=BAD if parsed == "with_parse" else None)]
    )

    response = runtime(client).send_structured(session, "Answer.", SCHEMA)

    assert isinstance(response, StructuredAgentResponse)
    assert response.finish_reason == "awaiting_approval"
    assert response.structured is None
    assert len(response.pending_approvals) == 1


def test_a_structured_resume_that_parks_again_has_no_document() -> None:
    """The step before the park ran an ungated call and wrote a bad document."""
    session = a_session()
    client = scripted(
        [
            anthropic_step(LlmToolCall("l1", "lookup", {})),
            anthropic_step(LlmToolCall("c1", "echo", {})),
        ]
    )

    response = runtime(client).send_structured(session, "Answer.", SCHEMA)

    assert response.finish_reason == "awaiting_approval"
    assert response.structured is None


def test_a_structured_turn_parked_then_resumed_returns_only_a_validated_document() -> None:
    session = a_session()
    client = scripted(
        [
            anthropic_step(LlmToolCall("c1", "echo", {})),
            LlmResponse(text=json.dumps({"n": 3}), finish_reason="stop", structured={"n": 3}),
        ]
    )
    agent = runtime(client)
    parked = agent.send_structured(session, "Answer.", SCHEMA)
    assert parked.structured is None
    record_approval(session, parked.pending_approvals[0].id, True)

    done = agent.send_structured(session, "", SCHEMA)

    assert done.structured == {"n": 3}


# -- G-406: the in-run marker covers only the runtime's own record call -----------


def test_a_host_row_recorded_inside_the_runtimes_write_window_is_outside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step of a gated call and an ungated one parks: the runtime writes the
    step, runs the ungated call, then writes its result (its own in-run
    record). A host row recorded re-entrantly from inside that write, naming
    the parked run's id, supersedes the step like any host row."""
    world = Recording()
    original = FileSessionStore.with_lock
    armed = [True]
    seen: list[Any] = []

    def with_lock(self: FileSessionStore, key: str, callback: Any, *args: Any) -> Any:
        if armed[0] and key.endswith(":thread"):
            stored = self.get(key) or {}
            entries = [e for e in stored.get("messages") or [] if isinstance(e, dict)]
            last = entries[-1] if entries else None
            if (
                last is not None
                and last["message"].get("type") == "assistant"
                and last["message"].get("tool_approval_requests")
            ):
                armed[0] = False
                host = world.root()
                try:
                    host.thread().record(
                        [{"type": "user", "content": "From inside."}], last["run_id"]
                    )
                    seen.append("recorded")
                except HarnessError as error:
                    seen.append(error.code)
        return original(self, key, callback, *args)

    monkeypatch.setattr(FileSessionStore, "with_lock", with_lock)
    _, approvals = park(world, ("c1", "echo", {"n": 1}), ("c2", "lookup", {"n": 2}))
    monkeypatch.setattr(FileSessionStore, "with_lock", original)

    assert seen == ["recorded"]
    stored = rows(world.fresh().root())
    assert [row[SUPERSEDED] for row in stored if SUPERSEDED in row] == [
        {"approval_ids": approvals}
    ]
    assert pending_approval_ids(stored) == []
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().root(), approvals[0], True)
    assert caught.value.code == "approval_not_pending"


# -- item 8: a new message's denials stand when its turn fails --------------------


def test_a_new_message_denies_approved_unrun_calls_even_when_its_turn_fails() -> None:
    world = Recording()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)

    def down() -> None:
        raise OSError("provider down")

    world.on_request = down
    with pytest.raises(OSError):
        world.fresh().runtime.send(world.root(), "Something else.")
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    results = [
        entry["result"]
        for row in rows(world.root())
        if row.get("type") == "tool_result"
        for entry in row["tool_results"]
        if entry["tool_call_id"] == "c1"
    ]
    assert results == [SUPERSEDED_RESULT]
