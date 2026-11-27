"""An approval belongs to the step that FIRST asked for it (G-408).

A host may record a copy of the paused step that keeps its approval requests:
an exact copy, one whose tool calls are not all objects, or one holding a NEW
call under the old call's id. A later row that repeats an approval id asks for
nothing, so a decision for that approval never runs a call of the copy: the
copy's calls have no request of their own, and the paused step, replaced by
the copy, runs nothing either. Entered at the public seam; the model is
scripted.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from test_runtime import CountingTool, a_session

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

NEW = {"id": "c1", "name": "echo", "arguments": {"value": "NEW"}}
COPIES: dict[str, Any] = {
    "mixed": lambda calls: ["x", NEW],
    "junk-then-new": lambda calls: [1, None, NEW],
    "new-only": lambda calls: [NEW],
    "exact": lambda calls: calls,
    "old-and-new": lambda calls: [*calls, NEW],
}


def a_runtime(ran: list[str], max_steps: int, model_fails: bool) -> AgentRuntime:
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
    state = {"requests": 0}

    def client(_request: LlmRequest) -> LlmResponse:
        state["requests"] += 1
        if state["requests"] == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("c1", "echo", {"value": "OLD"})],
            )
        if model_fails:
            raise RuntimeError("the provider is down")
        return LlmResponse(text="Done.", finish_reason="stop")

    class Echo(CountingTool):
        def handle(self, args: dict[str, Any]) -> Any:
            ran.append(str(args.get("value")))
            return f"ran:{args.get('value')}"

    counts: dict[str, int] = {}
    return AgentRuntime(
        client=client,
        modes=modes,
        tools=ToolRegistry().register(Echo("echo", counts)).register(CountingTool("shout", counts)),
    )


def resume(session: Session, runtime: AgentRuntime, times: int) -> None:
    for _ in range(times):
        try:
            runtime.send(session, "")
        except RuntimeError:
            pass  # the provider failing after the step was resolved


@pytest.mark.parametrize("shape", sorted(COPIES))
@pytest.mark.parametrize("when", ["before-approval", "after-approval", "after-result"])
@pytest.mark.parametrize("resend", [False, True], ids=["once", "decision-resent"])
@pytest.mark.parametrize(
    ("max_steps", "model_fails"),
    [(1, False), (6, False), (6, True)],
    ids=["steps-spent", "steps-left", "model-fails-after"],
)
def test_a_copy_keeping_the_approval_request_runs_nothing_under_that_approval(
    shape: str, when: str, resend: bool, max_steps: int, model_fails: bool
) -> None:
    session = a_session("guarded")
    ran: list[str] = []
    runtime = a_runtime(ran, max_steps, model_fails)

    approval = runtime.send(session, "go").pending_approvals[0].id
    if when in ("after-approval", "after-result"):
        record_approval(session, approval, True)
    if when == "after-result":
        resume(session, runtime, 1)  # the paused call runs, once
    parked = next(
        m.message
        for m in session.thread().messages()
        if m.message["type"] == "assistant" and m.message.get("tool_calls")
    )
    duplicate = copy.deepcopy(parked)
    duplicate["tool_calls"] = COPIES[shape](copy.deepcopy(parked["tool_calls"]))
    session.thread().record([duplicate])
    if when == "before-approval" or resend:
        record_approval(session, approval, True)
    resume(session, runtime, 3)

    # The approval was the paused step's; the copy's calls have none. Only a
    # paused call resumed before the copy existed has run, once.
    assert ran == (["OLD"] if when == "after-result" else [])
    results = [
        e["result"]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result"
        for e in m.message["tool_results"]
    ]
    assert [t for t in results if t.startswith("ran:")] == (
        ["ran:OLD"] if when == "after-result" else []
    )
