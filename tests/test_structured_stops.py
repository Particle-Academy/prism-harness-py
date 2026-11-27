"""A structured turn that stops before its answer (G-395).

``send_structured`` returns a ``StructuredAgentResponse`` on every path. A run
that stops for a person, or runs out of steps, has no document yet: it says so
with ``structured`` None and the same stop fields ``send`` returns, and never
fails on the way out. Entered at the public seam; the model is scripted.
"""

from __future__ import annotations

from typing import Any

from test_runtime import a_session, guarded_runtime, once

from prism_harness import (
    AgentRuntime,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    ModeRegistry,
    StructuredAgentResponse,
    ToolRegistry,
    record_approval,
)

SCHEMA: dict[str, Any] = {
    "name": "answer",
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


def test_a_structured_turn_that_stops_for_approval_returns_the_pending_approvals() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("c1", "echo", {"value": "x"})]), counts)

    response = runtime.send_structured(session, "go", SCHEMA)

    assert isinstance(response, StructuredAgentResponse)
    assert response.finish_reason == "awaiting_approval"
    assert response.structured is None
    [pending] = response.pending_approvals
    assert (pending.tool_call_id, pending.tool) == ("c1", "echo")
    assert counts == {}

    # The approved call runs once on the structured resume, which then answers.
    record_approval(session, pending.id, True)

    def answer(request: LlmRequest) -> LlmResponse:
        return LlmResponse(text='{"ok": true}', finish_reason="stop", structured={"ok": True})

    resumed = guarded_runtime(answer, counts).send_structured(session, "", SCHEMA)

    assert counts == {"echo": 1}
    assert isinstance(resumed, StructuredAgentResponse)
    assert resumed.structured == {"ok": True}


class Shout:
    @property
    def name(self) -> str:
        return "shout"

    def handle(self, args: dict[str, Any]) -> Any:
        return "shouted"


def test_a_structured_turn_that_runs_out_of_steps_says_so() -> None:
    session = a_session("busy")

    def keeps_calling(_request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("c", "shout", {})]
        )

    runtime = AgentRuntime(
        client=keeps_calling,
        modes=ModeRegistry(
            {
                "default": "busy",
                "modes": {"busy": {"system_prompt": "Go.", "tools": ["shout"], "max_steps": 2}},
            }
        ),
        tools=ToolRegistry().register(Shout()),
    )

    response = runtime.send_structured(session, "go", SCHEMA)

    assert isinstance(response, StructuredAgentResponse)
    assert response.finish_reason == "budget_exhausted"
    assert response.stopped_because is not None
    assert response.structured is None
