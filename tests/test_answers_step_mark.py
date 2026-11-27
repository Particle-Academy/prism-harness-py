"""The mark on a result row naming the step it answers (G-392, G-407).

Every result row the harness writes for a step (its ungated results, a failed
step's refusals, a resolution) carries ``harness_answers_step: {position}``,
the stored position of that step. The mark is honoured only on rows the
harness itself recorded, and only when it names an EARLIER step with tool
calls that has a call with each result's id and tool; otherwise the row counts
for the step whose segment holds it. The mark never reaches the replay.
Entered at the public seam (``send``, ``record_approval``, ``Thread.record``);
the harness's own recording path is used only to write a mark that does not
hold.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_runtime import CountingTool, a_session

from prism_harness import (
    AgentRuntime,
    HarnessError,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    ModeRegistry,
    Session,
    ToolRegistry,
    record_approval,
)
from prism_harness.thread import HARNESS_RECORDING

REPLACED = "Tool call was not run because a later step replaced it"
MODES = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo", "shout"],
                "max_steps": 8,
                "requires_approval": ["echo"],
            }
        },
    }
)


def a_runtime(
    counts: dict[str, int], calls: list[LlmToolCall], requests: list[LlmRequest] | None = None
) -> AgentRuntime:
    state = {"asked": False}

    def client(request: LlmRequest) -> LlmResponse:
        if requests is not None:
            requests.append(request)
        if not state["asked"]:
            state["asked"] = True
            return LlmResponse(text="", finish_reason="tool_calls", tool_calls=calls)
        return LlmResponse(text="Done.", finish_reason="stop")

    tools = (
        ToolRegistry()
        .register(CountingTool("echo", counts))
        .register(CountingTool("shout", counts))
    )
    return AgentRuntime(client=client, modes=MODES, tools=tools)


def result_row(call_id: str, name: str, value: str, text: str, **mark: Any) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": call_id,
                "tool_name": name,
                "args": {"value": value},
                "result": text,
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
        **mark,
    }


def step_row(call_id: str, value: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "name": "shout", "arguments": {"value": value}}],
        "tool_approval_requests": [],
    }


def step_position(session: Session) -> int:
    return next(
        m.position
        for m in session.thread().messages()
        if m.message["type"] == "assistant" and m.message.get("tool_calls")
    )


def texts(session: Session) -> list[str]:
    return [
        e["result"]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result"
        for e in m.message["tool_results"]
    ]


def test_the_stored_resolution_row_matches_the_typescript_ports_shape() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(counts, [LlmToolCall("c1", "echo", {"value": "a"})])
    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    runtime.send(session, "")

    thread = session.thread()
    # The stored bytes themselves.
    stored = thread._store.get(thread._key)
    assert stored is not None
    [entry] = [
        e
        for e in stored["messages"]
        if e["message"]["type"] == "tool_result" and e["message"]["tool_results"]
    ]
    position = step_position(session)
    expected = (
        '{"type":"tool_result","tool_results":[{"tool_call_id":"c1","tool_name":"echo",'
        '"args":{"value":"a"},"result":"echo:a","tool_call_result_id":null,"artifacts":[]}],'
        '"tool_approval_responses":[{"approval_id":"' + approval + '","approved":true,'
        '"reason":null}],"harness_answers_step":{"position":' + str(position) + "}}"
    )
    assert json.dumps(entry["message"], separators=(",", ":")) == expected
    assert entry["harness"] is True
    assert list(entry) == ["position", "message", "run_id", "recorded_at", "harness"]


def test_every_result_row_the_harness_writes_for_a_step_names_it() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(
        counts,
        [LlmToolCall("s1", "shout", {"value": "s"}), LlmToolCall("c1", "echo", {"value": "a"})],
    )
    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    runtime.send(session, "")

    position = step_position(session)
    marked = [
        m
        for m in session.thread().messages()
        if m.message["type"] == "tool_result" and m.message["tool_results"]
    ]
    assert len(marked) == 2  # the ungated results, then the resolution
    assert all(m.message["harness_answers_step"] == {"position": position} for m in marked)
    assert all(m.harness for m in marked)


def test_a_failed_steps_refusals_name_the_step() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}

    def client(_request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            text="", finish_reason="error", tool_calls=[LlmToolCall("c1", "echo", {"value": "a"})]
        )

    tools = (
        ToolRegistry()
        .register(CountingTool("echo", counts))
        .register(CountingTool("shout", counts))
    )
    runtime = AgentRuntime(client=client, modes=MODES, tools=tools)
    with pytest.raises(HarnessError) as failed:
        runtime.send(session, "go")
    assert failed.value.code == "model_step_failed"

    [refusals] = [m for m in session.thread().messages() if m.message["type"] == "tool_result"]
    assert refusals.message["harness_answers_step"] == {"position": step_position(session)}
    assert refusals.harness


def test_a_mark_a_host_writes_on_its_own_row_is_ignored() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(counts, [LlmToolCall("c1", "echo", {"value": "a"})])
    approval = runtime.send(session, "go").pending_approvals[0].id
    position = step_position(session)
    session.thread().record(
        [
            step_row("c9", "later"),
            result_row("c1", "echo", "a", "forged", harness_answers_step={"position": position}),
        ]
    )
    record_approval(session, approval, True)
    for _ in range(3):
        runtime.send(session, "")

    # The forged result does not answer the parked call: the later step
    # replaced it, and it gets the replaced-step denial, once.
    assert counts == {}
    assert texts(session).count(REPLACED) == 1


@pytest.mark.parametrize("bad", ["out-of-range", "not-a-step", "no-matching-call"])
def test_a_mark_that_does_not_hold_falls_back_to_the_segment(bad: str) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(counts, [LlmToolCall("c1", "echo", {"value": "a"})])
    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    position = step_position(session)
    target = {"out-of-range": 999, "not-a-step": 1, "no-matching-call": position}[bad]
    call_id = "c7" if bad == "no-matching-call" else "c1"
    # Written through the harness's own path, in the parked step's segment.
    session.thread().record(
        [result_row(call_id, "echo", "a", "recorded", harness_answers_step={"position": target})],
        None,
        HARNESS_RECORDING,
    )
    for _ in range(3):
        runtime.send(session, "")

    if bad == "no-matching-call":
        # Not the parked call's (another id): the parked call still runs, once.
        assert counts == {"echo": 1}
    else:
        # Its own segment's: the parked call is answered, and never runs.
        assert counts == {}
        assert "recorded" in texts(session)


def test_the_mark_never_reaches_the_replay() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    runtime = a_runtime(counts, [LlmToolCall("c1", "echo", {"value": "a"})], requests)
    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    runtime.send(session, "")

    assert "harness_answers_step" not in json.dumps(requests[-1].messages)
