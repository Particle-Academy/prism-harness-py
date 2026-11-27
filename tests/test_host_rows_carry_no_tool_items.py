"""A host's assistant row without tool calls sends no tool call (G-415).

An assistant row can carry provider-native content in ``additional_content``:
Anthropic ``content_blocks`` and OpenAI ``output_items``, which a client built
on prism-py replays verbatim. A host may record an assistant row through the
public ``Thread.record`` with no ``tool_calls`` but with that content copied
from a stored step. In the history sent to the model, such a row carries no
tool item (``tool_use``, ``server_tool_use``, ``mcp_tool_use``, or any
``*_call`` output item): a tool call nobody answers would be refused by the
provider. Entered at the public seam; every request the model receives is
checked.
"""

from __future__ import annotations

from typing import Any

from test_runtime import a_session, guarded_runtime

from prism_harness import LlmRequest, LlmResponse

NATIVE: dict[str, Any] = {
    "content_blocks": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"value": "a"}},
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}},
    ],
    "output_items": [
        {"type": "message", "content": [{"type": "output_text", "text": "Checking."}]},
        {"type": "function_call", "call_id": "call_1", "name": "echo", "arguments": "{}"},
        {"type": "custom_tool_call", "call_id": "call_2", "name": "echo", "input": ""},
    ],
    "stop_reason": "tool_use",
}


def tool_items(message: dict[str, Any]) -> list[str]:
    extra = message.get("additional_content") or {}
    blocks = [str(b.get("type")) for b in extra.get("content_blocks", [])]
    items = [str(i.get("type")) for i in extra.get("output_items", [])]
    return [t for t in blocks if t.endswith("tool_use")] + [t for t in items if t.endswith("_call")]


def test_a_host_assistant_row_without_tool_calls_sends_no_tool_item() -> None:
    session = a_session("guarded")
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="Done.", finish_reason="stop")

    runtime = guarded_runtime(client, {})
    runtime.send(session, "go")
    session.thread().record(
        [{"type": "assistant", "content": "Checking.", "additional_content": NATIVE}]
    )
    runtime.send(session, "Next")
    runtime.send(session, "")

    for request in requests[1:]:
        host = [
            m
            for m in request.messages
            if m["type"] == "assistant" and m.get("content") == "Checking."
        ]
        assert host, request.messages
        assert all(tool_items(m) == [] for m in host)
        # Content that is not the row's own is not sent at all: the row is
        # sent as its text.
        assert all(m["additional_content"] == {} for m in host)
