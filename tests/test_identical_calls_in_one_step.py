"""Two identical gated calls in one step each end with exactly one outcome
(G-392).

Identical means the same call id, tool and arguments: a provider that gives
every call to one function the same id, asked for one tool twice with the same
arguments. The step repeats an id, so neither runs; each gets its own denial,
once, and later resumes write nothing more. Entered at the public seam; the
model is scripted.
"""

from __future__ import annotations

import pytest
from test_runtime import CountingTool, a_session, once

from prism_harness import (
    AgentRuntime,
    LlmToolCall,
    ModeRegistry,
    Session,
    ToolRegistry,
    record_approval,
)

REPEATED = "Tool call was not run because another call in this step has the same id"


def a_runtime(counts: dict[str, int], max_steps: int) -> AgentRuntime:
    modes = ModeRegistry(
        {
            "default": "guarded",
            "modes": {
                "guarded": {
                    "system_prompt": "Careful.",
                    "tools": ["echo"],
                    "max_steps": max_steps,
                    "requires_approval": ["echo"],
                }
            },
        }
    )
    return AgentRuntime(
        client=once([LlmToolCall("dup", "echo", {"value": "a"})] * 2),
        modes=modes,
        tools=ToolRegistry().register(CountingTool("echo", counts)),
    )


def results(session: Session) -> list[list[str]]:
    """Each tool result row's results, row by row."""
    return [
        [entry["result"] for entry in m.message["tool_results"]]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result" and m.message["tool_results"]
    ]


@pytest.mark.parametrize("max_steps", [1, 4], ids=["steps-spent", "steps-left"])
def test_two_identical_gated_calls_each_get_one_denial_once(max_steps: int) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(counts, max_steps)

    first = runtime.send(session, "go")
    for pending in first.pending_approvals:
        record_approval(session, pending.id, True)
    for _ in range(4):
        runtime.send(session, "")

    assert counts == {}
    assert results(session) == [[REPEATED, REPEATED]]
