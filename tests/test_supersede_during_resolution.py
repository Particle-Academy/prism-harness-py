"""A superseding write while a step is being resolved (G-404).

A resume resolves the paused step under the session lock and runs its
approved calls there. A row that supersedes the step (a host's user row, or a
new message) written while an approved call runs must not give that call a
second outcome, and must not run it again. Every case is deterministic: the
superseding write is made by the tool itself, during its own execution.
Entered at the public seam; the model is scripted.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import pytest
from test_runtime import a_session, once

from prism_harness import (
    AgentRuntime,
    HarnessError,
    LlmToolCall,
    ModeRegistry,
    Session,
    ToolRegistry,
    record_approval,
)
from prism_harness.thread import Thread

REPLACED = "Tool call was not run because a later step replaced it"

MODES = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo"],
                "max_steps": 8,
                "requires_approval": ["echo"],
            }
        },
    }
)


class WritingTool:
    """Echo, which makes a superseding write while it runs."""

    def __init__(self, during: Callable[[], object]) -> None:
        self.runs = 0
        self._during = during

    @property
    def name(self) -> str:
        return "echo"

    def handle(self, args: dict[str, Any]) -> Any:
        self.runs += 1
        if self.runs == 1:
            self._during()
        return f"echo:{args.get('value', '')}"


def outcomes(session: Session) -> list[str]:
    return [
        entry["result"]
        for message in session.thread().messages()
        if message.message["type"] == "tool_result"
        for entry in message.message["tool_results"]
        if entry["args"].get("value") == "P"
    ]


def a_runtime(tool: WritingTool) -> AgentRuntime:
    return AgentRuntime(
        client=once([LlmToolCall("p", "echo", {"value": "P"})]),
        modes=MODES,
        tools=ToolRegistry().register(tool),
        # The superseding send made from inside the tool must not wait.
        config={"lock_wait": 0},
    )


@pytest.mark.parametrize("kind", ["user", "assistant"])
def test_a_host_row_recorded_while_the_approved_call_runs_leaves_it_one_outcome(
    kind: str,
) -> None:
    session = a_session("guarded")
    row = (
        {"type": "user", "content": "Hurry."}
        if kind == "user"
        else {"type": "assistant", "content": "Working on it."}
    )
    tool = WritingTool(lambda: session.thread().record([row]))
    runtime = a_runtime(tool)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    for _ in range(3):
        runtime.send(session, "")

    assert tool.runs == 1
    assert set(outcomes(session)) == {"echo:P"}
    assert REPLACED not in outcomes(session)


def test_a_new_message_sent_while_the_approved_call_runs_is_refused_and_runsnullcontext() -> None:
    session = a_session("guarded")
    refused: list[str] = []
    runtime_box: list[AgentRuntime] = []

    def new_message() -> None:
        try:
            runtime_box[0].send(session, "Another thing")
        except HarnessError as error:
            refused.append(error.code)

    tool = WritingTool(new_message)
    runtime = a_runtime(tool)
    runtime_box.append(runtime)

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    for _ in range(3):
        runtime.send(session, "")

    assert refused == ["session_locked"]
    assert tool.runs == 1
    assert set(outcomes(session)) == {"echo:P"}
    assert REPLACED not in outcomes(session)


@pytest.mark.parametrize("second_write", ["hook", "fails"])
def test_a_decision_around_a_new_messages_writes_runs_nothing(
    second_write: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A new message resolves the paused step FIRST, under the session lock
    # (an undecided call is refused), and only then records the user row.
    # A decision recorded between those two writes, or after the user row's
    # write failed, finds the call answered.
    session = a_session("guarded")
    tool = WritingTool(lambda: None)
    runtime = a_runtime(tool)
    approval = runtime.send(session, "go").pending_approvals[0].id

    native = Thread.record

    def record(
        self: Thread, messages: Any, run_id: str | None = None, origin: object = None
    ) -> Any:
        if any(m.get("type") == "user" and m.get("content") == "Another" for m in messages):
            if second_write == "fails":
                raise RuntimeError("the store refused the write")
            record_approval(session, approval, True)
        return native(self, messages, run_id, origin)

    monkeypatch.setattr(Thread, "record", record)
    with pytest.raises(RuntimeError) if second_write == "fails" else nullcontext():
        runtime.send(session, "Another")
    monkeypatch.setattr(Thread, "record", native)
    if second_write == "fails":
        record_approval(session, approval, True)
    for _ in range(3):
        runtime.send(session, "")

    assert tool.runs == 0
    assert outcomes(session) == ["No approval response provided"]
