"""A result counts only for the calls of the step whose segment it sits in
(G-392).

A step's segment is the tool result rows after it, up to the next step with
tool calls. Inside a segment a result ends the first unanswered call with the
same parsed arguments, then the same id and tool name, then the same id. It
never ends another step's call, so an earlier step's unanswered gated calls
get the replaced-step denial, and a call whose own result was recorded with
different arguments never runs again. Entered at the public seam over stored
rows; the model is scripted.
"""

from __future__ import annotations

from typing import Any

from test_runtime import a_session, guarded_runtime, once

from prism_harness import Session, record_approval

REPLACED = "Tool call was not run because a later step replaced it"


def step(call_id: str, value: str, approval: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "name": "echo", "arguments": {"value": value}}],
        "additional_content": {},
        "tool_approval_requests": [
            {"approval_id": approval, "tool_call_id": call_id, "arguments_format": 1}
        ],
        "harness_tool_names": ["echo", "shout"],
    }


def result(call_id: str, value: str, text: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": call_id,
                "tool_name": "echo",
                "args": {"value": value},
                "result": text,
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    }


def texts(session: Session) -> list[str]:
    return [
        entry["result"]
        for message in session.thread().messages()
        if message.message["type"] == "tool_result"
        for entry in message.message["tool_results"]
    ]


def test_a_newest_calls_result_stored_with_other_arguments_still_ends_it() -> None:
    # The newest call ran; its result was stored with arguments that differ
    # from its own (lossy or redacted) and equal an earlier call's.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step("a", "redacted", "apr_1"),
            {"type": "user", "content": "again"},
            step("a", "X", "apr_2"),
            result("a", "redacted", "echo:X"),
        ]
    )
    record_approval(session, "apr_2", True)
    runtime = guarded_runtime(once([]), counts)

    for _ in range(3):
        runtime.send(session, "")

    assert counts == {}
    assert texts(session).count(REPLACED) == 1


def test_a_result_in_a_later_segment_never_ends_an_earlier_steps_call() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step("a", "P", "apr_1"),
            step("b", "L", "apr_2"),
            result("a", "P", "echo:P"),
        ]
    )
    record_approval(session, "apr_1", True)
    runtime = guarded_runtime(once([]), counts)

    for _ in range(3):
        runtime.send(session, "")

    found = texts(session)
    assert found.count(REPLACED) == 1
    assert "No approval response provided" in found
    assert counts == {}
