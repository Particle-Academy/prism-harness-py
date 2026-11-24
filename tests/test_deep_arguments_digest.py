"""A tool call nested as deep as the harness accepts must not wedge the session.

The approval path digests a call's arguments for its decision events. The
harness accepts arguments nested up to just under ``_MAX_ARGUMENT_DEPTH``
levels, so the digest must handle every depth it accepts: a digest that raises
on one call would raise again on every later ``send()`` while that call waits,
and the conversation could never move on.
"""

from __future__ import annotations

import socket
import tempfile
from collections.abc import Callable, Iterator
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
    ToolRegistry,
    record_approval,
)
from prism_harness.tool_arguments import _MAX_ARGUMENT_DEPTH, usable_arguments
from prism_harness.tool_policy import args_digest, canonical_json

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


def nested(depth: int, inner: Any = "x") -> dict[str, Any]:
    value: Any = inner
    for _ in range(depth - 1):
        value = {"a": value}
    return {"a": value}


class Echo:
    def __init__(self, ran: list[dict[str, Any]]) -> None:
        self._ran = ran

    @property
    def name(self) -> str:
        return "echo"

    def handle(self, args: dict[str, Any]) -> Any:
        self._ran.append(args)
        return "echoed"


class Rig:
    def __init__(self, arguments: dict[str, Any]) -> None:
        # The durable thread on files, as a host stores it.
        directory = tempfile.mkdtemp(prefix="prism-harness-deep-")
        harness = PrismHarness(
            drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "memory", "durable": "files"},
        )
        self.session: Session = harness.for_(Participant("User", 1)).session("deep")
        self.requests: list[LlmRequest] = []
        self.ran: list[dict[str, Any]] = []
        self.events: list[Any] = []
        bus = HarnessEvents()
        bus.listen(self.events.append)
        self.agent = AgentRuntime(
            client=self._client(arguments),
            modes=MODES,
            tools=ToolRegistry().register(Echo(self.ran)),
            events=bus,
            config={"provider": "anthropic", "model": "m"},
        )

    def _client(self, arguments: dict[str, Any]) -> Callable[[LlmRequest], LlmResponse]:
        def client(request: LlmRequest) -> LlmResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return LlmResponse(
                    text="",
                    finish_reason="tool_calls",
                    tool_calls=[LlmToolCall(id="c1", name="echo", arguments=arguments)],
                )
            return LlmResponse(text=f"answer {len(self.requests)}", finish_reason="stop")

        return client

    def kinds(self) -> list[str]:
        return [event.type for event in self.events if event.type.startswith("tool_")]


DEEPEST = _MAX_ARGUMENT_DEPTH - 1


def test_the_deepest_accepted_arguments_are_accepted() -> None:
    assert usable_arguments(nested(DEEPEST)) is not None


@pytest.mark.parametrize("shape", ["object", "list"])
def test_the_digest_handles_every_accepted_depth(shape: str) -> None:
    value: Any = nested(DEEPEST) if shape == "object" else {"a": _list(DEEPEST - 1)}
    assert usable_arguments(value) is not None

    digest = args_digest(value)

    assert digest.startswith("sha256:")
    assert canonical_json(value).count("[" if shape == "list" else "{") == DEEPEST


def _list(depth: int) -> Any:
    value: Any = "x"
    for _ in range(depth):
        value = [value]
    return value


def test_the_digest_is_unchanged_for_ordinary_arguments() -> None:
    # RFC 8785: UTF-16 key order, no whitespace, ECMAScript numbers.
    assert canonical_json({"b": [1, 2.5, None, True], "a": {"é": "x", "z": 1e21}}) == (
        '{"a":{"z":1e+21,"é":"x"},"b":[1,2.5,null,true]}'
    )
    assert canonical_json([]) == "[]"
    assert canonical_json({}) == "{}"
    assert canonical_json([[], {}]) == "[[],{}]"


def test_a_denied_deep_call_never_runs_and_the_session_moves_on() -> None:
    rig = Rig(nested(DEEPEST))

    first = rig.agent.send(rig.session, "go")
    assert first.finish_reason == "awaiting_approval"
    assert rig.kinds() == ["tool_approval_asked"]

    record_approval(rig.session, first.pending_approvals[0].id, False, "no")
    resumed = rig.agent.send(rig.session, "")
    assert resumed.finish_reason == "stop"
    assert rig.kinds() == ["tool_approval_asked", "tool_denied"]

    later = rig.agent.send(rig.session, "and now?")
    assert later.finish_reason == "stop"
    assert rig.ran == []


def test_an_approved_deep_call_runs_once() -> None:
    rig = Rig(nested(DEEPEST))

    first = rig.agent.send(rig.session, "go")
    record_approval(rig.session, first.pending_approvals[0].id, True)
    rig.agent.send(rig.session, "")

    assert len(rig.ran) == 1
    assert rig.kinds() == ["tool_approval_asked", "tool_approved"]
