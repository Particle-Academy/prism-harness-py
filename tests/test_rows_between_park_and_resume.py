"""Rows recorded between a parked step and its resume (G-397, G-399).

A host may record rows on the thread while an approval is pending.

- A user row or an assistant row (an answer in text, or a later step with
  tool calls) SUPERSEDES the parked step: each of its gated calls with no
  result gets ONE denial, "Tool call was not run because a later step
  replaced it", and nothing runs, whatever was decided before or after.
- Any other row (a system note, an empty tool result row, another call's
  result) does not: the step is still the one a resume resolves, and its
  results and decisions are read from EVERY tool result row after it.
- A result recorded for a gated call is that call's one outcome.

Entered at the public seam; the model is scripted.
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
REPLACED = "Tool call was not run because a later step replaced it"


def a_result_row(*entries: tuple[str, str, str, str]) -> dict[str, Any]:
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
            for call_id, name, value, result in entries
        ],
        "tool_approval_responses": [],
    }


INSERTED: dict[str, dict[str, Any]] = {
    "assistant_text": {"type": "assistant", "content": "Waiting for your approval."},
    "assistant_call": {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": "p", "name": "shout", "arguments": {"value": "Z"}}],
        "tool_approval_requests": [],
    },
    "user": {"type": "user", "content": "Hurry."},
    "system": {"type": "system", "content": "A note."},
    "other": {"type": "developer", "content": "A note."},
    "tool_empty": a_result_row(),
    "tool_other": a_result_row(("elsewhere", "shout", "z", "shout:z")),
}

#: The rows after which the parked step is replaced.
SUPERSEDES = ("assistant_call", "assistant_text", "user")


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


def outcomes(session: Session) -> dict[str, list[str]]:
    """Every stored result by the call's ``value``, as stored, never folded."""
    found: dict[str, list[str]] = {}
    for entry in session.thread().messages():
        if entry.message["type"] == "tool_result":
            for result in entry.message["tool_results"]:
                found.setdefault(result["args"].get("value"), []).append(result["result"])
    return found


@pytest.mark.parametrize("kind", sorted(INSERTED))
@pytest.mark.parametrize("when", ["before-decision", "after-decision"])
@pytest.mark.parametrize("decision", [True, False, None], ids=["approve", "deny", "undecided"])
@pytest.mark.parametrize("max_steps", [1, 4], ids=["steps-spent", "steps-left"])
def test_a_parked_call_gets_one_outcome_and_runs_at_most_once_whatever_sits_between(
    kind: str, when: str, decision: bool | None, max_steps: int
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
    if decision is not None:
        record_approval(session, pending.id, decision, None if decision else "no")
    if when == "after-decision":
        session.thread().record([INSERTED[kind]])
    for _ in range(3):
        runtime.send(session, "")
    # A decision made afterwards changes nothing.
    record_approval(session, pending.id, True)
    runtime.send(session, "")

    assert counts.get("shout") == 1
    if kind in SUPERSEDES:
        expected = REPLACED
    elif decision is None:
        expected = "No approval response provided"
    else:
        expected = "echo:P" if decision else "no"
    assert outcomes(session)["P"] == [expected]
    assert counts.get("echo", 0) == (1 if expected == "echo:P" else 0)


@pytest.mark.parametrize("kind", sorted(INSERTED))
def test_a_repeated_id_step_runs_none_of_its_gated_calls_whatever_sits_between(kind: str) -> None:
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
    denial = REPLACED if kind in SUPERSEDES else REPEATED
    found = outcomes(session)
    assert (found["P"], found["R"]) == ([denial], [denial])


@pytest.mark.parametrize("decision", [True, False, None], ids=["approve", "deny", "undecided"])
def test_a_result_recorded_for_a_parked_call_is_its_only_outcome(decision: bool | None) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(
        once([LlmToolCall("p", "echo", {"value": "P"}), LlmToolCall("q", "echo", {"value": "Q"})]),
        counts,
        4,
    )

    first = runtime.send(session, "go")
    session.thread().record([a_result_row(("p", "echo", "P", "forged:P"))])
    for pending in first.pending_approvals:
        if decision is not None:
            record_approval(session, pending.id, decision, None if decision else "no")
    for _ in range(3):
        runtime.send(session, "")

    found = outcomes(session)
    assert set(found["P"]) == {"forged:P"}
    assert counts.get("echo", 0) == (1 if decision else 0)


def test_an_approved_call_that_is_refused_is_not_reported_as_called() -> None:
    # The run reports the tools it invoked. An approved call refused instead
    # of run (here: its tool is not offered to the resuming run) is not one.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts, 4)

    first = runtime.send(session, "go")
    record_approval(session, first.pending_approvals[0].id, True)
    resumed = runtime.send(session, "", ["shout"])

    assert counts == {}
    assert outcomes(session) == {"P": ["Not run: echo is not available to this run."]}
    assert resumed.tool_calls == []
    run = session.run()
    assert run is not None and run["tool_calls"] == []


def test_a_retry_after_a_failed_model_step_does_not_run_an_approved_call_again() -> None:
    # Approved calls run, and their results are recorded, BEFORE the next
    # provider request; a request that fails leaves the results in the
    # thread, so the retry finds the call done. (The G-398 shape does not
    # apply to this runtime.)
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
    assert outcomes(session)["P"] == ["echo:P"]


# -- who recorded the row ------------------------------------------------------
#
# A text or user row supersedes the parked step when it is recorded OUTSIDE the
# run that owns the step. In this runtime every such row recorded after the
# park and before the step is resolved is outside it: the run that parked
# returned and writes nothing more, and a resuming run resolves the step FIRST
# (inside the session lock), before it records the person's prompt or the
# model's answer. So no run id is consulted; a row cannot be made "inside" by
# carrying the parking run's id, which Thread.record accepts from any caller.


@pytest.mark.parametrize("shape", ["text", "structured"])
def test_the_runtimes_own_answer_after_resolving_does_not_supersede_the_step(shape: str) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts, 4)
    schema = {
        "name": "answer",
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }

    def answers(request: LlmRequest) -> LlmResponse:
        if request.schema is not None:
            return LlmResponse(text='{"ok": true}', finish_reason="stop", structured={"ok": True})
        return LlmResponse(text="Done.", finish_reason="stop")

    first = runtime.send(session, "go")
    record_approval(session, first.pending_approvals[0].id, True)
    resuming = a_runtime(answers, counts, 4)
    for _ in range(3):
        if shape == "structured":
            resuming.send_structured(session, "", schema)
        else:
            resuming.send(session, "")

    assert counts == {"echo": 1}
    assert outcomes(session) == {"P": ["echo:P"]}
    # The answer rows are the resuming runs' own.
    answers_recorded = [
        entry
        for entry in session.thread().messages()
        if entry.message["type"] == "assistant" and not entry.message.get("tool_calls")
    ]
    assert len(answers_recorded) == 3
    assert all(entry.run_id is not None for entry in answers_recorded)


@pytest.mark.parametrize("kind", ["assistant_text", "user"])
@pytest.mark.parametrize("run", ["parking-run", "no-run", "invented-run"])
def test_a_text_or_user_row_recorded_by_a_host_supersedes_whatever_run_id_it_carries(
    kind: str, run: str
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = a_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts, 4)

    first = runtime.send(session, "go")
    parked = session.run()
    assert parked is not None
    # Thread.record is public and takes any run id, the parking run's included.
    run_id = {"parking-run": parked["id"], "no-run": None, "invented-run": "run-x"}[run]
    session.thread().record([INSERTED[kind]], run_id)
    record_approval(session, first.pending_approvals[0].id, True)
    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {}
    assert outcomes(session) == {"P": [REPLACED]}
