"""G-354: a decision for an approval that is not pending is refused.

A decision is an answer to a question the run asked. An approval id the run
never issued, or one already answered, is not an answer to anything. Recorded
while a call was pending, it was written to the thread and the resume closed
the real call as "No approval response provided"; recorded while nothing was
pending, it was written all the same, and the resume it invites is a model
turn with no new user message.
"""

from __future__ import annotations

import json
import socket
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
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

MODES = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo", "shout"],
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


class CountingTool:
    def __init__(self, name: str, counts: dict[str, int]) -> None:
        self._name = name
        self._counts = counts

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        self._counts[self._name] = self._counts.get(self._name, 0) + 1
        return f"{self._name}:{args.get('value', '')}"


class Rig:
    def __init__(self, tool_calls: list[LlmToolCall]) -> None:
        directory = tempfile.mkdtemp(prefix="prism-harness-g354-")
        harness = PrismHarness(
            drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "memory", "durable": "files"},
        )
        self.session: Session = harness.for_(Participant("User", 1)).session("support")
        self.session.using_provider("anthropic").using_model("claude-sonnet-5-5")
        self.requests: list[LlmRequest] = []
        self.counts: dict[str, int] = {}
        tools = (
            ToolRegistry()
            .register(CountingTool("echo", self.counts))
            .register(CountingTool("shout", self.counts))
        )
        self.agent = AgentRuntime(client=self._client(tool_calls), modes=MODES, tools=tools)

    def _client(self, tool_calls: list[LlmToolCall]) -> Callable[[LlmRequest], LlmResponse]:
        def client(request: LlmRequest) -> LlmResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return LlmResponse(text="", finish_reason="tool_calls", tool_calls=tool_calls)
            return LlmResponse(
                text=f"Finished after {len(self.requests) - 1}.", finish_reason="stop"
            )

        return client

    def rows(self) -> str:
        return json.dumps(
            [entry.message for entry in self.session.thread().messages()], sort_keys=True
        )


def gated() -> Rig:
    return Rig([LlmToolCall(id="c1", name="echo", arguments={"value": "x"})])


def refused(session: Session, approval_id: str, approved: bool = True) -> None:
    with pytest.raises(HarnessError) as caught:
        record_approval(session, approval_id, approved)
    assert caught.value.code == "approval_not_pending"
    assert approval_id in caught.value.message


def test_refuses_a_made_up_id_while_a_call_is_pending_and_the_real_one_still_works() -> None:
    rig = gated()
    first = rig.agent.send(rig.session, "go")
    before = rig.rows()

    refused(rig.session, "approval-never-issued")

    assert len(rig.requests) == 1
    assert rig.rows() == before
    assert rig.counts == {}

    record_approval(rig.session, first.pending_approvals[0].id, True)
    resumed = rig.agent.send(rig.session, "")

    assert rig.counts == {"echo": 1}
    assert resumed.text == "Finished after 1."
    assert "No approval response provided" not in json.dumps(rig.requests[1].messages, default=str)


def test_refuses_a_made_up_id_while_nothing_is_pending() -> None:
    rig = Rig([LlmToolCall(id="c1", name="shout", arguments={"value": "y"})])
    rig.agent.send(rig.session, "go")
    before = rig.rows()
    assert len(rig.requests) == 2
    assert rig.counts == {"shout": 1}

    refused(rig.session, "approval-never-issued")

    assert len(rig.requests) == 2
    assert rig.rows() == before
    assert rig.counts == {"shout": 1}


def test_refuses_a_decision_on_a_session_that_has_never_had_a_turn() -> None:
    rig = gated()

    refused(rig.session, "approval-never-issued", approved=False)

    assert rig.requests == []
    assert rig.rows() == "[]"


def test_the_real_id_works_exactly_as_before() -> None:
    rig = gated()
    first = rig.agent.send(rig.session, "go")

    record_approval(rig.session, first.pending_approvals[0].id, True)
    rig.agent.send(rig.session, "")

    assert rig.counts == {"echo": 1}
    assert len(rig.requests) == 2


def test_refuses_a_repeat_of_an_id_already_answered() -> None:
    rig = gated()
    first = rig.agent.send(rig.session, "go")
    record_approval(rig.session, first.pending_approvals[0].id, True)
    rig.agent.send(rig.session, "")
    before = rig.rows()

    refused(rig.session, first.pending_approvals[0].id)

    assert len(rig.requests) == 2
    assert rig.rows() == before
    assert rig.counts == {"echo": 1}


def test_refuses_a_repeat_recorded_before_the_run_resumed() -> None:
    # A denial and then an approval for the same call: the second would
    # overturn a decision the person already made.
    rig = gated()
    first = rig.agent.send(rig.session, "go")
    record_approval(rig.session, first.pending_approvals[0].id, False, "no")
    before = rig.rows()

    refused(rig.session, first.pending_approvals[0].id)

    assert len(rig.requests) == 1
    assert rig.rows() == before
