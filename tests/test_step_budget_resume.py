"""The step limit holds across approval pauses.

Mirrors prism-harness-ts/test/step-budget-resume.test.ts. A run that pauses for
a person and is resumed is the SAME run, so every resume continues the same
step count rather than starting a new one. Entered through the public send()
and record_approval(), with a fake provider that always asks for a gated tool.
"""

from __future__ import annotations

import socket
import tempfile
from collections.abc import Callable
from typing import Any

import pytest

from prism_harness import (
    AgentResponse,
    AgentRuntime,
    FileSessionStore,
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


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider below is a fake; opening a socket is a bug."""

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


MAX_STEPS = 3
APPROVALS = 6

MODES = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo"],
                "max_steps": MAX_STEPS,
                "requires_approval": ["echo"],
            },
        },
    }
)


class Model:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: LlmRequest) -> LlmResponse:
        self.calls += 1
        return LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(f"c{self.calls}", "echo", {"n": self.calls})],
        )


class Echo:
    name = "echo"

    def __init__(self, ran: list[int] | None = None) -> None:
        self._ran = ran

    def handle(self, args: dict[str, Any]) -> Any:
        if self._ran is not None:
            self._ran.append(int(args["n"]))
        return "ok"


def runtime(model: Model, ran: list[int] | None = None) -> AgentRuntime:
    return AgentRuntime(client=model, modes=MODES, tools=ToolRegistry().register(Echo(ran)))


def persisted() -> Callable[[], Session]:
    """A fresh harness every time: only what is on disk survives."""
    directory = tempfile.mkdtemp(prefix="prism-harness-step-budget-")

    def open_session() -> Session:
        harness = PrismHarness(
            drivers={"files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "files", "durable": "files"},
        )
        session = harness.for_(Participant("User", 1)).session("support")
        session.using_provider("anthropic").using_model("claude-sonnet-5-5")
        return session

    return open_session


def one_session() -> Callable[[], Session]:
    directory = tempfile.mkdtemp(prefix="prism-harness-step-budget-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("support")
    session.using_provider("anthropic").using_model("claude-sonnet-5-5")
    return lambda: session


def approve_repeatedly(
    open_session: Callable[[], Session], agent: Callable[[], AgentRuntime]
) -> AgentResponse:
    response = agent().send(open_session(), "go")
    for _ in range(APPROVALS):
        if response.finish_reason != "awaiting_approval":
            break
        record_approval(open_session(), response.pending_approvals[0].id, True)
        response = agent().send(open_session(), "")
    return response


def test_never_takes_more_than_max_steps_across_approvals_in_one_runtime() -> None:
    model = Model()
    shared = runtime(model)
    response = approve_repeatedly(one_session(), lambda: shared)

    assert model.calls == MAX_STEPS
    assert response.finish_reason == "budget_exhausted"
    assert response.steps == MAX_STEPS


def test_never_takes_more_than_max_steps_across_approvals_resumed_from_persisted_state() -> None:
    model = Model()
    # A new runtime and a reloaded session per resume: a different worker.
    response = approve_repeatedly(persisted(), lambda: runtime(model))

    assert model.calls == MAX_STEPS
    assert response.finish_reason == "budget_exhausted"


def test_still_runs_the_last_approved_call_when_the_budget_is_spent() -> None:
    model = Model()
    ran: list[int] = []
    shared = runtime(model, ran)
    approve_repeatedly(one_session(), lambda: shared)

    assert ran == [1, 2, 3]
    assert model.calls == MAX_STEPS


def test_gives_a_new_prompt_after_a_finished_run_a_fresh_budget() -> None:
    model = Model()
    shared = runtime(model)
    open_session = one_session()
    response = approve_repeatedly(open_session, lambda: shared)
    assert response.finish_reason == "budget_exhausted"

    following = shared.send(open_session(), "Another task")
    assert following.finish_reason == "awaiting_approval"
    assert model.calls == MAX_STEPS + 1
