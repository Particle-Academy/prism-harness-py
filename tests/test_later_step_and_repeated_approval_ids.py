"""Two more approval pairing rules (G-392).

1. An earlier parked step followed by a later tool-call step: each gated call
   of the earlier step with no result of its own gets one denial, "Tool call
   was not run because a later step replaced it", written once. A decision
   for the earlier step never runs anything.
2. A step whose approval requests repeat an APPROVAL id runs none of its
   gated calls; each gets the repeated-id denial.

Entered at the public seam (``send``, ``record_approval``) over steps stored
as another writer left them; the model is scripted.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import Session, record_approval

REPEATED = "Tool call was not run because another call in this step has the same id"
REPLACED = "Tool call was not run because a later step replaced it"


def call(call_id: str, name: str, value: str) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": {"value": value}}


def step(calls: list[dict[str, Any]], requests: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "content": "",
        "tool_calls": calls,
        "additional_content": {},
        "tool_approval_requests": [
            {"approval_id": approval, "tool_call_id": call_id, "arguments_format": 1}
            for approval, call_id in requests
        ],
        "harness_tool_names": ["echo", "shout"],
    }


def outcomes(session: Session) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for entry in session.thread().messages():
        if entry.message["type"] == "tool_result":
            for result in entry.message["tool_results"]:
                found.setdefault(result["args"].get("value"), []).append(result["result"])
    return found


@pytest.mark.parametrize("decided", [True, False], ids=["approved", "undecided"])
def test_an_earlier_parked_step_replaced_by_a_later_one_is_denied_once(decided: bool) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step([call("a", "echo", "old"), call("b", "shout", "old-ungated")], [("apr_old", "a")]),
            {"type": "user", "content": "never mind"},
            step([call("c", "echo", "new")], [("apr_new", "c")]),
        ]
    )
    if decided:
        record_approval(session, "apr_old", True)
    record_approval(session, "apr_new", True)
    runtime = guarded_runtime(once([]), counts)

    for _ in range(3):
        runtime.send(session, "")
    record_approval(session, "apr_old", True)
    runtime.send(session, "")

    assert counts == {"echo": 1}
    found = outcomes(session)
    assert found["old"] == [REPLACED]
    assert found["new"] == ["echo:new"]
    # An ungated call is nobody's to deny.
    assert "old-ungated" not in found


def test_an_earlier_step_whose_calls_all_have_results_is_left_alone() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step([call("a", "echo", "old")], [("apr_old", "a")]),
            {
                "type": "tool_result",
                "tool_results": [
                    {
                        "tool_call_id": "a",
                        "tool_name": "echo",
                        "args": {"value": "old"},
                        "result": "echo:old",
                        "tool_call_result_id": None,
                        "artifacts": [],
                    }
                ],
                "tool_approval_responses": [
                    {"approval_id": "apr_old", "approved": True, "reason": None}
                ],
            },
            step([call("a", "echo", "new")], [("apr_new", "a")]),
        ]
    )
    record_approval(session, "apr_new", False, "no")
    runtime = guarded_runtime(once([]), counts)

    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {}
    assert outcomes(session) == {"old": ["echo:old"], "new": ["no"]}


@pytest.mark.parametrize("decisions", [(True, True), (True, False), (False, True)])
def test_a_step_whose_requests_repeat_an_approval_id_runs_none_of_its_gated_calls(
    decisions: tuple[bool, bool],
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step(
                [call("a", "echo", "A"), call("b", "echo", "B")],
                [("apr_same", "a"), ("apr_same", "b")],
            ),
        ]
    )
    for approved in decisions:
        record_approval(session, "apr_same", approved, None if approved else "no")
    runtime = guarded_runtime(once([]), counts)

    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {}
    assert outcomes(session) == {"A": [REPEATED], "B": [REPEATED]}
