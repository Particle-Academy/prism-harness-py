"""A step with tool calls written while an approved call runs (G-392, G-404).

The writer may be the host (``Thread.record``) or another run on the same
session (its own loop records its step). Resolution writes the results it
produced marked with the step they answer, so wherever that row lands they
count for that step: the call that ran keeps its own result, it is never
recorded as replaced, and the later step's call is never credited with it; it
gets its own outcome.
Deterministic: the write is made by the tool itself, while it runs. Entered at
the public seam; the model is scripted and its requests are captured.
"""

from __future__ import annotations

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

REPLACED = "Tool call was not run because a later step replaced it"
MODES = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo"],
                "max_steps": 8,
                "requires_approval": ["echo"],
            }
        },
    }
)


class Writing(CountingTool):
    def __init__(self, session: Session, later_id: str, run_id: str | None, counts: Any) -> None:
        super().__init__("echo", counts)
        self._session, self._later_id, self._run_id = session, later_id, run_id

    def handle(self, args: dict[str, Any]) -> Any:
        if args.get("value") == "a":
            self._session.thread().record(
                [
                    {
                        "type": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": self._later_id, "name": "echo", "arguments": {"value": "b"}}
                        ],
                        "tool_approval_requests": [
                            {
                                "approval_id": "apr_later",
                                "tool_call_id": self._later_id,
                                "arguments_format": 1,
                            }
                        ],
                        "harness_tool_names": ["echo"],
                    }
                ],
                self._run_id,
            )
        return super().handle(args)


def outcomes(session: Session) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for m in session.thread().messages():
        if m.message["type"] == "tool_result":
            for entry in m.message["tool_results"]:
                found.setdefault(entry["args"].get("value"), []).append(entry["result"])
    return found


@pytest.mark.parametrize("later_id", ["c1", "z9"], ids=["same-id", "other-id"])
@pytest.mark.parametrize("writer", ["host", "another-run"])
@pytest.mark.parametrize("later_decided", [False, True], ids=["later-undecided", "later-approved"])
def test_the_call_that_ran_keeps_its_result_and_the_later_call_its_own(
    later_id: str, writer: str, later_decided: bool
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    state = {"asked": False}

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        if not state["asked"]:
            state["asked"] = True
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("c1", "echo", {"value": "a"})],
            )
        return LlmResponse(text="Done.", finish_reason="stop")

    tool = Writing(session, later_id, None if writer == "host" else "run-other", counts)
    runtime = AgentRuntime(client=client, modes=MODES, tools=ToolRegistry().register(tool))

    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    runtime.send(session, "")
    if later_decided:
        record_approval(session, "apr_later", True)
    for _ in range(3):
        runtime.send(session, "")

    found = outcomes(session)
    assert set(found["a"]) == {"echo:a"}
    assert counts == {"echo": 1}
    # The later step is followed by the resuming run's answer, so it is
    # replaced: its call gets the one replaced-step denial, and a decision for
    # it runs nothing.
    assert found["b"] == [REPLACED]
    # The replay sends each result as its own call's.
    last = requests[-1]
    replayed = [
        (entry["tool_call_id"], entry["args"].get("value"), entry["result"])
        for message in last.messages
        if message["type"] == "tool_result"
        for entry in message["tool_results"]
    ]
    assert ("c1", "a", "echo:a") in replayed
    assert not any(value == "b" and result == "echo:a" for _, value, result in replayed)
