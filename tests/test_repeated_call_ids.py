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
