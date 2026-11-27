"""Every gated call ends with exactly ONE outcome across steps (G-392, G-397).

A result row counts ONLY for the step whose segment it sits in (the rows
after that step's assistant row, up to the next step with tool calls), never
for another step's call. Entered at the public seam (``send``,
``record_approval``, ``Thread.record``); the model is scripted.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import LlmToolCall, Session, record_approval

REPLACED = "Tool call was not run because a later step replaced it"


def result_row(call_id: str, name: str, value: str, result: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": call_id,
                "tool_name": name,
                "args": {"value": value},
                "result": result,
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    }


def later_step(call_id: str, name: str, value: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "name": name, "arguments": {"value": value}}],
        "tool_approval_requests": [],
    }


def outcomes(session: Session) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for entry in session.thread().messages():
        if entry.message["type"] == "tool_result":
            for result in entry.message["tool_results"]:
                found.setdefault(result["args"].get("value"), set()).add(result["result"])
    return found


@pytest.mark.parametrize("segment", ["earlier", "later"])
@pytest.mark.parametrize("later_tool", ["shout", "echo"])
def test_a_result_counts_for_the_step_whose_segment_it_sits_in(
    segment: str, later_tool: str
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("a", "echo", {"value": "P"})]), counts)

    first = runtime.send(session, "go")
    recorded = result_row("a", "echo", "P", "recorded:P")
    if segment == "earlier":
        session.thread().record([recorded, later_step("a", later_tool, "L")])
    else:
        session.thread().record([later_step("a", later_tool, "L"), recorded])
    record_approval(session, first.pending_approvals[0].id, True)
    for _ in range(3):
        runtime.send(session, "")

    found = outcomes(session)
    assert "echo" not in counts
    if segment == "earlier":
        # In the parked step's segment: that call's only outcome.
        assert found["P"] == {"recorded:P"}
        if later_tool == "echo":
            assert found["L"] == {"No approval response provided"}
    else:
        # In the later step's segment: the later call's (same id), never the
        # parked call's, which gets the replaced-step denial.
        assert found["P"] == {"recorded:P", REPLACED}
        assert "L" not in found


def test_a_replaced_steps_open_call_gets_one_denial_beside_a_later_steps_result() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("a", "echo", {"value": "P"})]), counts)

    runtime.send(session, "go")
    session.thread().record(
        [later_step("a", "shout", "L"), result_row("a", "shout", "L", "shout:L")]
    )
    for _ in range(3):
        runtime.send(session, "")

    found = outcomes(session)
    assert found["P"] == {REPLACED}
    assert found["L"] == {"shout:L"}
    assert counts == {}
