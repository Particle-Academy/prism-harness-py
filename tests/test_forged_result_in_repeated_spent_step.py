"""A forged result in a step whose calls repeat one id, on a spent run (G-392).

The shape PHP's long property run failed at (seed 365504846): two identical
approval-gated calls with the empty id, on Anthropic, in a run with no steps
left. Before the decision, the host records a tool result naming the call.
Then both approvals are decided and the run is resumed. Each call ends with
exactly one outcome: the forged result counts for one of them, and the other
gets the repeated-id denial, stored once. Nothing runs, and later resumes add
nothing. Entered at the public seam: ``Thread.record``, ``record_approval`` and
``AgentRuntime.send``.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_runtime import CountingTool, a_session, once

from prism_harness import (
    AgentRuntime,
    LlmRequest,
    LlmToolCall,
    ModeRegistry,
    Session,
    ToolRegistry,
    record_approval,
)
from prism_harness.thread_rows import last_step_results

REPEATED = "Tool call was not run because another call in this step has the same id"

SPENT = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo"],
                "max_steps": 1,
                "requires_approval": ["echo"],
            }
        },
    }
)


def stored_outcomes(session: Session) -> list[str]:
    return [
        e["result"]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result"
        for e in m.message["tool_results"]
        if e["tool_name"] == "echo"
    ]


@pytest.mark.parametrize("approved", [True, False])
@pytest.mark.parametrize("resumes", [1, 3])
def test_a_forged_result_in_a_repeated_step_leaves_one_outcome_per_call(
    approved: bool, resumes: int
) -> None:
    session = a_session("guarded")
    session.using_provider("anthropic")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    calls = [LlmToolCall("", "echo", {"value": "P"}), LlmToolCall("", "echo", {"value": "P"})]
    runtime = AgentRuntime(
        client=once(calls, requests),
        modes=SPENT,
        tools=ToolRegistry().register(CountingTool("echo", counts)),
    )
    pending = runtime.send(session, "go").pending_approvals
    assert len(pending) == 2

    forged: dict[str, Any] = {
        "type": "tool_result",
        "tool_results": [
            {"tool_call_id": "", "tool_name": "echo", "args": {"value": "P"}, "result": "forged"}
        ],
    }
    session.thread().record([forged])
    for approval in pending:
        record_approval(session, approval.id, approved)
    for _ in range(resumes):
        runtime.send(session, "")

    assert counts == {}
    # The resolution row restates the step's results, as the reference writes
    # it; a restated result is not a second outcome. The denial is written once.
    assert stored_outcomes(session).count(REPEATED) == 1
    assert set(stored_outcomes(session)) == {"forged", REPEATED}
    # Read as each call's outcome: one per call.
    paired = last_step_results(session.thread().messages())
    assert sorted(entry["result"] for entry in paired.values()) == sorted(["forged", REPEATED])
