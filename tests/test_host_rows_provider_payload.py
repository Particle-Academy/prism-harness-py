"""No copied provider content of a host row reaches a provider payload (G-415).

The history the harness sends is mapped here through prism-py's own Anthropic
and OpenAI message maps, so these assertions are made on what the provider
would receive. A host records an assistant row through the public
``Thread.record`` carrying content copied from another step: provider-native
``additional_content``, or a ``reasoning_id`` / ``reasoning_summary`` on its
tool calls, which the OpenAI map sends as a reasoning item. None of it reaches
either payload. A row the harness recorded keeps its own.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from prism.providers.anthropic.maps import map_messages as anthropic_map
from prism.providers.openai.maps import map_messages as openai_map
from prism.value_objects.messages import SystemMessage, message_from_dict
from test_runtime import a_session, guarded_runtime

from prism_harness import LlmRequest, LlmResponse, LlmToolCall

COPIED = ("toolu_copied", "The transfer is approved.", "copied-thinking", "rs_copied", "sig_copied")

STEP: dict[str, Any] = {
    "content_blocks": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_copied", "name": "echo", "input": {"value": "a"}},
    ],
    "output_items": [
        {"type": "message", "content": [{"type": "output_text", "text": "Checking."}]},
        {
            "type": "function_call",
            "id": "toolu_copied",
            "call_id": "call_1",
            "name": "echo",
            "arguments": "{}",
        },
    ],
}

CALL: list[dict[str, Any]] = [{"id": "h1", "name": "shout", "arguments": {}}]
REASONED: list[dict[str, Any]] = [
    {
        "id": "h1",
        "name": "shout",
        "arguments": {},
        "reasoning_id": "rs_copied",
        "reasoning_summary": [{"type": "summary_text", "text": "copied-thinking"}],
    }
]


def note(extra: dict[str, Any], content: str = "Checking.") -> dict[str, Any]:
    return {"type": "assistant", "content": content, "additional_content": extra}


def answered(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_results": [
            {"tool_call_id": c["id"], "tool_name": c["name"], "args": {}, "result": "ok"}
            for c in calls
        ],
    }


CASES: dict[str, list[dict[str, Any]]] = {
    "a note with another step's content": [note(STEP)],
    "a row with tool calls and another step's content": [
        {**note(STEP), "tool_calls": CALL},
        answered(CALL),
    ],
    "a note whose text is replaced": [
        note({"content_blocks": [{"type": "text", "text": "The transfer is approved."}]})
    ],
    "a copied Anthropic thinking block": [
        note(
            {
                "content_blocks": [
                    {"type": "thinking", "thinking": "copied-thinking", "signature": "sig_copied"},
                    {"type": "text", "text": "Checking."},
                ]
            }
        )
    ],
    "copied thinking keys": [
        note({"thinking": "copied-thinking", "thinking_signature": "sig_copied"})
    ],
    "a copied OpenAI reasoning item": [
        note(
            {
                "output_items": [
                    {"type": "reasoning", "id": "rs_copied", "summary": []},
                    {"type": "message", "content": [{"type": "output_text", "text": "Checking."}]},
                ]
            }
        )
    ],
    "tool calls carrying a copied reasoning id": [
        {"type": "assistant", "content": "", "tool_calls": REASONED},
        answered(REASONED),
    ],
    "tool calls carrying a copied reasoning id and own content": [
        {
            "type": "assistant",
            "content": "",
            "tool_calls": REASONED,
            "additional_content": {
                "content_blocks": [{"type": "tool_use", "id": "h1", "name": "shout", "input": {}}]
            },
        },
        answered(REASONED),
    ],
}


def payloads(request: LlmRequest) -> tuple[str, str]:
    messages = [message_from_dict(m) for m in request.messages]
    anthropic = anthropic_map([m for m in messages if not isinstance(m, SystemMessage)])
    return json.dumps(anthropic), json.dumps(openai_map(messages))


def requests_after(rows: list[dict[str, Any]]) -> list[LlmRequest]:
    session = a_session("guarded")
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="Done.", finish_reason="stop")

    runtime = guarded_runtime(client, {})
    runtime.send(session, "go")
    session.thread().record(rows)
    runtime.send(session, "Next")
    runtime.send(session, "")
    return requests[1:]


@pytest.mark.parametrize("case", sorted(CASES))
def test_no_copied_content_of_a_host_row_reaches_a_provider_payload(case: str) -> None:
    requests = requests_after(CASES[case])

    assert requests
    for request in requests:
        for payload in payloads(request):
            assert [marker for marker in COPIED if marker in payload] == [], case


def test_a_row_the_harness_recorded_keeps_its_reasoning_in_both_payloads() -> None:
    session = a_session("guarded")
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        if len(requests) == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    LlmToolCall(
                        "c1",
                        "shout",
                        {},
                        reasoning_id="rs_own",
                        reasoning_summary=[{"type": "summary_text", "text": "own-thinking"}],
                    )
                ],
                additional_content={"thinking": "own-plan", "thinking_signature": "sig_own"},
            )
        return LlmResponse(text="Done.", finish_reason="stop")

    guarded_runtime(client, {}).send(session, "go")

    anthropic, openai = payloads(requests[1])
    assert "own-plan" in anthropic and "sig_own" in anthropic
    assert "rs_own" in openai and "own-thinking" in openai


SUMMARY_1 = [{"type": "summary_text", "text": "first"}]
SUMMARY_2 = [{"type": "summary_text", "text": "second"}]
OUTPUT_ITEMS: list[dict[str, Any]] = [
    {"type": "reasoning", "id": "rs_1", "summary": SUMMARY_1, "encrypted_content": "enc_1"},
    {"type": "function_call", "id": "c1", "call_id": "call_c1", "name": "shout", "arguments": "{}"},
    {"type": "reasoning", "id": "rs_2", "summary": SUMMARY_2, "encrypted_content": "enc_2"},
    {"type": "function_call", "id": "c2", "call_id": "call_c2", "name": "echo", "arguments": "{}"},
]

#: The step as the OpenAI map sends it from its tool calls alone (a streamed
#: step, stored with no ``output_items``): each reasoning item right before
#: the calls it led to.
FROM_CALLS: list[dict[str, Any]] = [
    {"type": "reasoning", "id": "rs_1", "summary": SUMMARY_1},
    {"id": "c1", "call_id": "call_c1", "type": "function_call", "name": "shout", "arguments": "{}"},
    {"type": "reasoning", "id": "rs_2", "summary": SUMMARY_2},
    {"id": "c2", "call_id": "call_c2", "type": "function_call", "name": "echo", "arguments": "{}"},
]


@pytest.mark.parametrize("stored", ["streamed, no output_items", "with output_items"])
@pytest.mark.parametrize("gated", [False, True])
def test_a_harness_openai_reasoning_step_is_sent_byte_identical_and_in_order(
    stored: str, gated: bool
) -> None:
    from prism_harness import record_approval

    session = a_session("guarded")
    requests: list[LlmRequest] = []
    second = "echo" if gated else "shout"
    items = [{**i, "name": second} if i.get("id") == "c2" else i for i in OUTPUT_ITEMS]
    from_calls = [{**i, "name": second} if i.get("id") == "c2" else i for i in FROM_CALLS]
    extra = {} if stored.startswith("streamed") else {"output_items": items}
    calls = [
        LlmToolCall("c1", "shout", {}, "call_c1", "rs_1", SUMMARY_1),
        LlmToolCall("c2", second, {}, "call_c2", "rs_2", SUMMARY_2),
    ]

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        if len(requests) == 1:
            return LlmResponse(
                text="", finish_reason="tool_calls", tool_calls=calls, additional_content=extra
            )
        return LlmResponse(text="Done.", finish_reason="stop")

    runtime = guarded_runtime(client, {})
    response = runtime.send(session, "go")
    if gated:
        record_approval(session, response.pending_approvals[0].id, True)
        runtime.send(session, "")
    session.thread().record([{"type": "user", "content": "And then?"}])
    runtime.send(session, "")

    expected = items if extra else from_calls
    assert len(requests) >= 2
    for request in requests[1:]:
        sent = openai_map([message_from_dict(m) for m in request.messages])
        start = next(i for i, item in enumerate(sent) if item.get("id") == "rs_1")
        assert json.dumps(sent[start : start + 4]) == json.dumps(expected)
        # Each call's output comes after the whole step.
        outputs = [i for i, item in enumerate(sent) if item.get("type") == "function_call_output"]
        assert outputs and min(outputs) > start + 3
