"""Every request sent to the model answers each step's tool calls right after
the step (G-392).

Providers require each tool call to be answered by the turn right after the
step that made it. A host may record a user row or an assistant text row while
a call runs; the result the harness then records still belongs right after its
own step in the history sent to the model. Entered at the public seam; the
model is scripted and every request it receives is checked.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_runtime import GUARDED, CountingTool, a_session

from prism_harness import (
    AgentRuntime,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    Session,
    ToolRegistry,
    record_approval,
)

ROWS: dict[str, dict[str, Any]] = {
    "user": {"type": "user", "content": "Meanwhile, something else."},
    "assistant_text": {"type": "assistant", "content": "Working on it."},
}


def unanswered(request: LlmRequest) -> list[str]:
    """Tool calls the turn right after their step does not answer."""
    missing: list[str] = []
    messages = request.messages
    for index, message in enumerate(messages):
        if message.get("type") != "assistant" or not message.get("tool_calls"):
            continue
        after = messages[index + 1] if index + 1 < len(messages) else {}
        answered = (
            {e["tool_call_id"] for e in after.get("tool_results", [])}
            if after.get("type") == "tool_result"
            else set()
        )
        missing += [c["id"] for c in message["tool_calls"] if c["id"] not in answered]
    return missing


def a_rig(
    session: Session, tool: str, row: dict[str, Any]
) -> tuple[AgentRuntime, list[LlmRequest]]:
    requests: list[LlmRequest] = []
    state = {"n": 0}

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        state["n"] += 1
        if state["n"] == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("c1", tool, {"value": "a"})],
            )
        return LlmResponse(text="Done.", finish_reason="stop")

    class Writing(CountingTool):
        def handle(self, args: dict[str, Any]) -> Any:
            session.thread().record([row])  # the host, while the call runs
            return super().handle(args)

    counts: dict[str, int] = {}
    other = "shout" if tool == "echo" else "echo"
    tools = ToolRegistry().register(Writing(tool, counts)).register(CountingTool(other, counts))
    return AgentRuntime(client=client, modes=GUARDED, tools=tools), requests


@pytest.mark.parametrize("row", sorted(ROWS))
@pytest.mark.parametrize("tool", ["echo", "shout"], ids=["approved-call", "ungated-call"])
def test_a_row_recorded_while_a_call_runs_leaves_each_step_answered_right_after_it(
    row: str, tool: str
) -> None:
    session = a_session("guarded")
    runtime, requests = a_rig(session, tool, ROWS[row])

    first = runtime.send(session, "go")
    for pending in first.pending_approvals:
        record_approval(session, pending.id, True)
    runtime.send(session, "")
    runtime.send(session, "Next")
    runtime.send(session, "")

    assert len(requests) >= 3
    for number, request in enumerate(requests):
        assert unanswered(request) == [], (number, request.messages)
