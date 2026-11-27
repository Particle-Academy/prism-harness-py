"""Two pins on the 0.5 line (G-405, G-406).

G-405: a structured turn that parks for an approval hands back no document.
The model's text on a tool-calling step is not an answer, so it is never
checked against the schema or returned as ``structured``.

G-406: a host's user row recorded while the runtime is writing the parked step,
under the run's own id, still supersedes the step. This runtime consults no run
id for that. Entered at the public seam; the model is scripted.
"""

from __future__ import annotations

from typing import Any

from test_runtime import a_session, guarded_runtime, once

from prism_harness import (
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    StructuredAgentResponse,
    record_approval,
)
from prism_harness.thread import Thread

SCHEMA: dict[str, Any] = {
    "name": "profile",
    "type": "object",
    "properties": {"title": {"type": "string"}, "role": {"type": "string", "enum": ["user"]}},
    "required": ["title"],
}
REPLACED = "Tool call was not run because a later step replaced it"


def test_a_structured_turn_parked_for_approval_hands_back_no_document() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    document = {"title": 12345, "role": "admin"}

    def client(_request: LlmRequest) -> LlmResponse:
        # Anthropic-shaped: text and a tool call in one step.
        return LlmResponse(
            text='{"title": 12345, "role": "admin"}',
            finish_reason="tool_use",
            tool_calls=[LlmToolCall("toolu_1", "echo", {"value": "P"})],
            structured=document,
        )

    response = guarded_runtime(client, counts).send_structured(session, "go", SCHEMA)

    assert isinstance(response, StructuredAgentResponse)
    assert response.finish_reason == "awaiting_approval"
    assert response.structured is None
    assert counts == {}
    assistant = next(
        m.message for m in session.thread().messages() if m.message["type"] == "assistant"
    )
    assert "structured" not in assistant["additional_content"]


def test_a_user_row_recorded_during_the_parked_step_write_under_the_run_id_supersedes(
    monkeypatch: Any,
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)
    native = Thread.record
    fired: list[str | None] = []

    def record(self: Thread, messages: Any, run_id: str | None = None) -> Any:
        written = native(self, messages, run_id)
        if not fired and any(m.get("tool_approval_requests") for m in messages):
            fired.append(run_id)
            # The public API, naming the run that is writing the step.
            native(self, [{"type": "user", "content": "Hurry."}], run_id)
        return written

    monkeypatch.setattr(Thread, "record", record)
    approval = runtime.send(session, "go").pending_approvals[0].id
    monkeypatch.setattr(Thread, "record", native)
    record_approval(session, approval, True)
    for _ in range(3):
        runtime.send(session, "")

    assert fired and fired[0] is not None
    assert counts == {}
    results = [
        entry["result"]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result"
        for entry in m.message["tool_results"]
    ]
    assert results.count(REPLACED) == 1
