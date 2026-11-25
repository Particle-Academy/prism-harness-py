"""What a host call policy sees, and what happens when a decision cannot be made.

A call policy is asked with the arguments of each call before it runs.
Arguments that are not a usable object never reach it as if they were real, and
arguments that mean "none" reach it as an empty object. When the decision itself
fails, the call does not run.
"""

from __future__ import annotations

import contextlib
import socket
import tempfile
import warnings
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessEvents,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    Session,
    ToolAuthorizer,
    ToolRegistry,
    record_approval,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client below is a fake; opening a socket is a bug."""

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


class Lookup:
    name = "lookup"

    def __init__(self, ran: list[Any]) -> None:
        self._ran = ran

    def handle(self, args: dict[str, Any]) -> str:
        self._ran.append(args)
        return "ran"


def _world(
    arguments: Any,
    *,
    authorizer: ToolAuthorizer | None = None,
    events: HarnessEvents | None = None,
    requires_approval: list[str] | None = None,
) -> tuple[AgentRuntime, Session, list[Any]]:
    directory = tempfile.mkdtemp(prefix="prism-call-decisions-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = (
        harness.for_(Participant("User", 1))
        .session("decisions")
        .using_mode("chat")
        .using_provider("synthetic")
        .using_model("offline")
    )
    ran: list[Any] = []
    responses = [
        LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall("call-1", "lookup", arguments)],
        ),
        LlmResponse(text="done", finish_reason="stop"),
    ]

    def client(_request: LlmRequest) -> LlmResponse:
        return responses.pop(0) if responses else LlmResponse(text="done", finish_reason="stop")

    runtime = AgentRuntime(
        client=client,
        tools=ToolRegistry().register(Lookup(ran)),
        authorizer=authorizer,
        events=events,
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
    return runtime, session, ran


def _send(runtime: AgentRuntime, session: Session, prompt: str) -> None:
    # Refusing by raising is one way of not running the call.
    with contextlib.suppress(Exception):
        runtime.send(session, prompt)


@pytest.mark.parametrize(
    "arguments",
    ['{"path": "/tmp/', ["/tmp"], 5, {0: "/tmp"}, '{"amount":1e999}'],
    ids=["unparsed text", "a list", "a number", "an integer key", "past the float range"],
)
def test_a_call_policy_is_never_asked_about_unusable_arguments(arguments: Any) -> None:
    asked: list[Any] = []

    def call(_session: Session, _tool: Any, args: dict[str, Any]) -> bool:
        asked.append(args)
        return True

    runtime, session, ran = _world(arguments, authorizer=ToolAuthorizer(enabled=True, call=call))
    _send(runtime, session, "Look it up.")

    assert asked == []
    assert ran == []


@pytest.mark.parametrize(
    "arguments", ["", None, {}], ids=["an empty string", "None", "an empty dict"]
)
def test_a_call_policy_is_asked_about_empty_forms_as_no_arguments(arguments: Any) -> None:
    asked: list[Any] = []

    def call(_session: Session, _tool: Any, args: dict[str, Any]) -> bool:
        asked.append(args)
        return True

    runtime, session, ran = _world(arguments, authorizer=ToolAuthorizer(enabled=True, call=call))
    runtime.send(session, "Look it up.")

    assert asked == [{}]
    assert ran == [{}]


def test_a_call_policy_that_raises_never_runs_the_call() -> None:
    def call(_session: Session, _tool: Any, _args: dict[str, Any]) -> bool:
        raise RuntimeError("check failed")

    runtime, session, ran = _world(
        {"city": "Paris"}, authorizer=ToolAuthorizer(enabled=True, call=call)
    )
    _send(runtime, session, "go")

    assert ran == []


def test_a_call_policy_that_raises_at_resume_never_runs_the_approved_call() -> None:
    failing = False

    def call(_session: Session, _tool: Any, _args: dict[str, Any]) -> bool:
        if failing:
            raise RuntimeError("check failed")
        return True

    runtime, session, ran = _world(
        {"city": "Paris"},
        authorizer=ToolAuthorizer(enabled=True, call=call),
        requires_approval=["lookup"],
    )
    paused = runtime.send(session, "go")
    assert paused.finish_reason == "awaiting_approval"
    record_approval(session, paused.pending_approvals[0].id, True)
    failing = True
    _send(runtime, session, "")

    assert ran == []


def test_a_raising_event_listener_still_pauses_for_a_person() -> None:
    events = HarnessEvents()

    def listener(_event: Any) -> None:
        raise RuntimeError("listener failed")

    events.listen(listener)
    runtime, session, ran = _world({"city": "Paris"}, events=events, requires_approval=["lookup"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = runtime.send(session, "go")

    assert response.finish_reason == "awaiting_approval"
    assert ran == []
