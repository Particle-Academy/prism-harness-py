"""Rows recorded between a parked step and its resume (G-397, G-399).

A host may record rows on the thread while an approval is pending: an
assistant note, the person's follow-up, a system note, or another row. The
step a resume resolves is the last step with tool calls, and its results and
decisions are read from EVERY tool result row after it, so whatever sits in
between changes neither which decision answers which call nor whether a call
already ran. Entered at the public seam; the model is scripted.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_runtime import CountingTool, a_session, once

from prism_harness import (
    AgentRuntime,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    ModeRegistry,
    Session,
    ToolRegistry,
    record_approval,
)

REPEATED = "Tool call was not run because another call in this step has the same id"

INSERTED: dict[str, dict[str, Any]] = {
    "assistant": {"type": "assistant", "content": "Waiting for your approval."},
    "user": {"type": "user", "content": "Hurry."},
    "system": {"type": "system", "content": "A note."},
    "other": {"type": "developer", "content": "A note."},
    "tool": {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "elsewhere",
                "tool_name": "shout",
                "args": {"value": "z"},
                "result": "shout:z",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    },
}


def a_runtime(client: Any, counts: dict[str, int], max_steps: int) -> AgentRuntime:
    modes = ModeRegistry(
        {
            "default": "guarded",
            "modes": {
                "guarded": {
                    "system_prompt": "Careful.",
                    "tools": ["echo", "shout"],
                    "max_steps": max_steps,
                    "requires_approval": ["echo"],
                }
            },
        }
    )
    tools = (
        ToolRegistry()
        .register(CountingTool("echo", counts))
        .register(CountingTool("shout", counts))
    )
    return AgentRuntime(client=client, modes=modes, tools=tools)


def outcomes(session: Session) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for entry in session.thread().messages():
        if entry.message["type"] == "tool_result":
            for result in entry.message["tool_results"]:
                found.setdefault(result["args"].get("value"), set()).add(result["result"])
    return found


def stored_results(session: Session, value: str) -> int:
    return sum(
        1
        for entry in session.thread().messages()
        if entry.message["type"] == "tool_result"
        for result in entry.message["tool_results"]
        if result["args"].get("value") == value
    )


@pytest.mark.parametrize("kind", sorted(INSERTED))
@pytest.mark.parametrize("when", ["before-decision", "after-decision"])
@pytest.mark.parametrize("max_steps", [1, 4], ids=["steps-spent", "steps-left"])
def test_an_approved_call_runs_at_most_once_whatever_sits_between(
    kind: str, when: str, max_steps: int
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(
        once([LlmToolCall("p", "echo", {"value": "P"}), LlmToolCall("s", "shout", {"value": "S"})]),
        counts,
        max_steps,
    )

    first = runtime.send(session, "go")
    [pending] = first.pending_approvals
    if when == "before-decision":
        session.thread().record([INSERTED[kind]])
    record_approval(session, pending.id, True)
    if when == "after-decision":
        session.thread().record([INSERTED[kind]])
    for _ in range(3):
        runtime.send(session, "")

    assert counts.get("echo", 0) <= 1
    assert counts["shout"] == 1
    # Exactly one outcome for the gated call, written once.
    assert len(outcomes(session)["P"]) == 1
    assert stored_results(session, "P") == 1
    if kind not in ("assistant", "user"):
        # A note or another call's row does not end the pending step: the
        # approval stands and the call runs once.
        assert counts.get("echo", 0) == 1
        assert outcomes(session)["P"] == {"echo:P"}


@pytest.mark.parametrize("kind", sorted(INSERTED))
def test_a_repeated_id_step_runs_none_of_its_gated_calls_whatever_sits_between(
    kind: str,
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(
        once([LlmToolCall("w", "echo", {"value": "P"}), LlmToolCall("w", "echo", {"value": "R"})]),
        counts,
        4,
    )

    first = runtime.send(session, "go")
    session.thread().record([INSERTED[kind]])
    record_approval(session, first.pending_approvals[-1].id, True)
    for _ in range(3):
        runtime.send(session, "")

    assert counts == {}
    assert outcomes(session) == {"P": {REPEATED}, "R": {REPEATED}} | (
        {"z": {"shout:z"}} if kind == "tool" else {}
    )


def test_a_retry_after_a_failed_model_step_does_not_run_an_approved_call_again() -> None:
    # Approved calls run, and their results are recorded, BEFORE the next
    # provider request; a request that fails leaves the results in the
    # thread, so the retry finds the call done.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    state = {"requests": 0}

    def client(_request: LlmRequest) -> LlmResponse:
        state["requests"] += 1
        if state["requests"] == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("p", "echo", {"value": "P"})],
            )
        if state["requests"] == 2:
            raise RuntimeError("the provider is down")
        return LlmResponse(text="Done.", finish_reason="stop")

    runtime = a_runtime(client, counts, 4)
    first = runtime.send(session, "go")
    record_approval(session, first.pending_approvals[0].id, True)
    with pytest.raises(RuntimeError):
        runtime.send(session, "")
    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {"echo": 1}
    assert stored_results(session, "P") == 1
