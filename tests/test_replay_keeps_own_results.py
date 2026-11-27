"""The replay keeps each call's own result (G-392).

The thread view folds a step's consecutive tool result rows into one, and a
later row's result for the SAME call replaces an earlier one. The same call is
the same call id, tool name and arguments (the n-th such result of a row
replacing the n-th of an earlier row), never the call id alone: two calls of
one step may share an id. Entered at the public seam; the model is scripted
and its requests are captured.
"""

from __future__ import annotations

from typing import Any

from test_runtime import a_session, guarded_runtime, once

from prism_harness import LlmRequest, LlmToolCall, record_approval

REPEATED = "Tool call was not run because another call in this step has the same id"


def replayed_results(request: LlmRequest) -> list[tuple[str, Any]]:
    folded = [m for m in request.messages if m["type"] == "tool_result"]
    return [
        (entry["result"], entry["args"].get("value"))
        for row in folded
        for entry in row["tool_results"]
    ]


def test_two_calls_sharing_an_id_each_keep_their_result_in_the_replay() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    calls = [LlmToolCall("w", "echo", {"value": "P"}), LlmToolCall("w", "echo", {"value": "R"})]
    runtime = guarded_runtime(once(calls, requests), counts)

    first = runtime.send(session, "go")
    for pending in first.pending_approvals:
        record_approval(session, pending.id, True)
    runtime.send(session, "")

    assert counts == {}
    assert sorted(replayed_results(requests[-1])) == [(REPEATED, "P"), (REPEATED, "R")]


def test_a_later_row_still_replaces_the_same_calls_earlier_result() -> None:
    session = a_session("guarded")
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(once([], requests), {})
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "name": "shout", "arguments": {"value": "S"}}],
            },
            {
                "type": "tool_result",
                "tool_results": [
                    {
                        "tool_call_id": "a",
                        "tool_name": "shout",
                        "args": {"value": "S"},
                        "result": "one",
                    }
                ],
            },
            {
                "type": "tool_result",
                "tool_results": [
                    {
                        "tool_call_id": "a",
                        "tool_name": "shout",
                        "args": {"value": "S"},
                        "result": "two",
                    }
                ],
            },
        ]
    )

    runtime.send(session, "Next")

    assert replayed_results(requests[-1]) == [("two", "S")]
