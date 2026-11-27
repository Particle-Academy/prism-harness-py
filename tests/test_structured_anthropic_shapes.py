"""A structured step with text and a gated call, in the two shapes a client
built on prism-py's Anthropic mapping hands over (G-405).

prism-py maps a Messages body whose content holds a text block (here a
document that misses the schema) and a ``tool_use`` block as follows:

- ``stop_reason: "tool_use"``: refused with ``tool_loop_not_supported``, so the
  client raises and the step never reaches the harness;
- ``stop_reason: "end_turn"``: finish ``stop``, the tool call kept, and the text
  parsed as the document.

In neither case does the harness hand back the document or drop the gated
call. Entered at the public seam; the model is scripted.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_runtime import a_session, guarded_runtime

from prism_harness import (
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    StructuredAgentResponse,
)

SCHEMA: dict[str, Any] = {
    "name": "profile",
    "type": "object",
    "properties": {"title": {"type": "string"}, "role": {"type": "string", "enum": ["user"]}},
    "required": ["title"],
}
TEXT = '{"title": 12345, "role": "admin"}'


class ToolLoopNotSupported(Exception):
    """What a prism-py client raises for ``stop_reason: "tool_use"``."""


def test_an_end_turn_step_with_a_gated_call_parks_and_returns_no_document() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}

    def client(_request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            text=TEXT,
            finish_reason="stop",
            tool_calls=[LlmToolCall("toolu_1", "echo", {"value": "P"})],
            structured={"title": 12345, "role": "admin"},
        )

    response = guarded_runtime(client, counts).send_structured(session, "go", SCHEMA)

    assert isinstance(response, StructuredAgentResponse)
    assert response.finish_reason == "awaiting_approval"
    assert response.structured is None
    [pending] = response.pending_approvals
    assert (pending.tool_call_id, pending.tool) == ("toolu_1", "echo")
    assert counts == {}


def test_a_tool_use_step_the_client_refuses_returns_no_document_and_no_call() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}

    def client(_request: LlmRequest) -> LlmResponse:
        raise ToolLoopNotSupported("tool_loop_not_supported")

    with pytest.raises(ToolLoopNotSupported):
        guarded_runtime(client, counts).send_structured(session, "go", SCHEMA)

    rows = [entry.message for entry in session.thread().messages()]
    assert not any(row.get("tool_calls") for row in rows)
    assert counts == {}
    run = session.run()
    assert run is not None and run["status"] == "failed"
