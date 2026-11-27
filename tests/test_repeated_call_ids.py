"""Each tool call is decided by its own approval (G-392).

A step is one assistant message with tool calls and the approval requests
issued for it. When a tool call id repeats among the step's calls (an empty id
counts), or a ``tool_call_id`` repeats among its approval requests, the step is
ambiguous: every gated call in it that has no result yet is denied and never
runs, whatever was decided, each with its own result. A step whose ids are all
unique is decided call by call, as before.

Some providers reuse an id within one response, so two calls can arrive under
one id. Every test enters at the public seam (``send``, ``record_approval``);
the model is scripted and the network is refused.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import LlmToolCall, Session, record_approval

REPEATED = "Tool call was not run because another call in this step has the same id"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def results_after_park(session: Session) -> list[tuple[str, str, Any]]:
    """Every stored result written after the parked step, as stored (never the
    provider view, which folds results by call id)."""
    stored = [entry.message for entry in session.thread().messages()]
    index = max(
        i for i, row in enumerate(stored) if row["type"] == "assistant" and row["tool_calls"]
    )
    return [
        (entry["tool_call_id"], entry["result"], entry["args"].get("value"))
        for row in stored[index + 1 :]
        if row["type"] == "tool_result"
        for entry in row["tool_results"]
    ]


def calls_of(*calls: tuple[str, str, str]) -> list[LlmToolCall]:
    return [LlmToolCall(call_id, name, {"value": value}) for call_id, name, value in calls]


SAME_ID = (("dup", "echo", "a"), ("dup", "echo", "b"))
EMPTY_ID = (("", "echo", "a"), ("", "echo", "b"))


@pytest.mark.parametrize("calls", [SAME_ID, EMPTY_ID], ids=["same-id", "empty-id"])
@pytest.mark.parametrize(
    "decisions",
    [(False, True), (True, False), (True, True), (False, False)],
    ids=["deny1-approve2", "approve1-deny2", "approve-both", "deny-both"],
)
def test_two_gated_calls_under_one_id_never_run(
    calls: tuple[tuple[str, str, str], ...], decisions: tuple[bool, bool]
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once(calls_of(*calls)), counts)

    first = runtime.send(session, "go")
    approvals = [pending.id for pending in first.pending_approvals]
    assert len(set(approvals)) == 2
    for approval, approved in zip(approvals, decisions, strict=True):
        record_approval(session, approval, approved, None if approved else "no")
    runtime.send(session, "")

    assert counts == {}
    call_id = calls[0][0]
    assert results_after_park(session) == [(call_id, REPEATED, "a"), (call_id, REPEATED, "b")]
    # A later decision and resume run nothing either.
    for approval in approvals:
        record_approval(session, approval, True)
    runtime.send(session, "")
    assert counts == {}


def test_an_undecided_ambiguous_step_is_denied_with_the_same_reason() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once(calls_of(*SAME_ID)), counts)

    runtime.send(session, "go")
    runtime.send(session, "")

    assert counts == {}
    assert [result for _, result, _ in results_after_park(session)] == [REPEATED, REPEATED]


def test_an_ungated_call_in_an_ambiguous_step_keeps_running() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(
        once(calls_of(("dup", "echo", "a"), ("dup", "echo", "b"), ("solo", "shout", "c"))), counts
    )

    first = runtime.send(session, "go")
    assert counts == {"shout": 1}
    for pending in first.pending_approvals:
        record_approval(session, pending.id, True)
    runtime.send(session, "")

    assert counts == {"shout": 1}
    assert [r for r in results_after_park(session) if r[1] == REPEATED] == [
        ("dup", REPEATED, "a"),
        ("dup", REPEATED, "b"),
    ]


def test_an_ungated_call_sharing_the_gated_calls_id_never_lets_it_run() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once(calls_of(("", "shout", "a"), ("", "echo", "b"))), counts)

    first = runtime.send(session, "go")
    assert counts == {"shout": 1}
    for pending in first.pending_approvals:
        record_approval(session, pending.id, True)
    runtime.send(session, "")

    assert "echo" not in counts


def test_unique_ids_are_still_decided_call_by_call() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(
        once(calls_of(("a", "echo", "1"), ("b", "echo", "2"), ("c", "echo", "3"))), counts
    )

    first = runtime.send(session, "go")
    approvals = [pending.id for pending in first.pending_approvals]
    record_approval(session, approvals[0], False, "not this one")
    record_approval(session, approvals[1], True)
    runtime.send(session, "")

    assert counts == {"echo": 1}
    assert results_after_park(session) == [
        ("a", "not this one", "1"),
        ("b", "echo:2", "2"),
        ("c", "No approval response provided", "3"),
    ]


def outcomes_by_value(session: Session) -> dict[str, list[str]]:
    """Every stored result, by the call's ``value`` argument, never by call id."""
    found: dict[str, list[str]] = {}
    for entry in session.thread().messages():
        if entry.message["type"] == "tool_result":
            for result in entry.message["tool_results"]:
                found.setdefault(result["args"].get("value"), []).append(result["result"])
    return found


@pytest.mark.parametrize("gated_first", [True, False], ids=["gated-first", "ungated-first"])
@pytest.mark.parametrize("decision", [True, False, None], ids=["approve", "deny", "undecided"])
def test_a_gated_call_sharing_an_ungated_calls_id_gets_its_own_denial(
    gated_first: bool, decision: bool | None
) -> None:
    # The ungated call's result is not the gated call's: results are matched
    # to calls by position, so the gated call still gets exactly one outcome.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    calls = (("", "echo", "a"), ("", "shout", "b"))
    runtime = guarded_runtime(once(calls_of(*(calls if gated_first else calls[::-1]))), counts)

    first = runtime.send(session, "go")
    assert counts == {"shout": 1}
    [pending] = first.pending_approvals
    if decision is not None:
        record_approval(session, pending.id, decision, None if decision else "no")
    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {"shout": 1}
    outcomes = {value: set(texts) for value, texts in outcomes_by_value(session).items()}
    assert outcomes == {"a": {REPEATED}, "b": {"shout:b"}}


def test_a_stored_gated_call_sharing_a_finished_calls_id_gets_its_own_denial() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "", "name": "echo", "arguments": {"value": "a"}},
                    {"id": "", "name": "shout", "arguments": {"value": "b"}},
                ],
                "additional_content": {},
                "tool_approval_requests": [],
                "harness_tool_names": ["echo", "shout"],
            },
            {
                "type": "tool_result",
                "tool_results": [
                    {
                        "tool_call_id": "",
                        "tool_name": "shout",
                        "args": {"value": "b"},
                        "result": "shout:b",
                        "tool_call_result_id": None,
                        "artifacts": [],
                    }
                ],
                "tool_approval_responses": [],
            },
        ]
    )
    runtime = guarded_runtime(once([]), counts)

    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {}
    outcomes = {value: set(texts) for value, texts in outcomes_by_value(session).items()}
    assert outcomes == {"a": {REPEATED}, "b": {"shout:b"}}


@pytest.mark.parametrize("name", ["echo", "", "renamed"], ids=["named", "nameless", "other-name"])
def test_a_result_stored_without_the_calls_tool_name_still_ends_the_call(name: str) -> None:
    # Pairing by position must never read a finished call as unanswered: a
    # result whose tool name is missing or different still ends the call with
    # its id, and the approved call does not run again.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"value": "x"}}],
                "additional_content": {},
            },
            {
                "type": "tool_approval_request",
                "approvals": [{"id": "c1", "tool": "echo", "arguments": {"value": "x"}}],
            },
            {"type": "tool_approval_response", "approval_id": "c1", "approved": True},
            {"type": "tool_result", "tool_call_id": "c1", "name": name, "result": "echo:x"},
        ]
    )

    guarded_runtime(once([]), counts).send(session, "")

    assert counts == {}
