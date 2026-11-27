"""The replay keeps each call's own result, ONCE (G-403 and item 5, G-387 round 14).

The thread view folds a step's consecutive tool result rows into one, and a
later row's result for the SAME call replaces an earlier one. The same call is
the same call id, tool name and arguments (the n-th such result of a row
replacing the n-th of an earlier row), never the call id alone: two calls of
one step may share an id. The fold is PHP main's (``ToolResultRuns``,
prism-harness#141) and the 0.5.x line's (prism-harness-py#75).

And each result is emitted EXACTLY ONCE across the whole replay: a row that
repeats a result an earlier row of the same step already holds (the
resolution row repeats the results the step already had) adds no second
copy, even when another row sits between them.

Entered at the public seam; the model is scripted and its requests are
captured.
"""

from __future__ import annotations

import json
import socket
from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import LlmRequest, LlmToolCall, record_approval

REPEATED = "Tool call was not run because another call in this step has the same id"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


def replayed_results(request: LlmRequest) -> list[tuple[str, str, Any]]:
    folded = [m for m in request.messages if m["type"] == "tool_result"]
    return [
        (entry["tool_name"], entry["result"], entry["args"].get("value"))
        for row in folded
        for entry in row["tool_results"]
    ]


def occurrences(request: LlmRequest) -> Counter[str]:
    """Each replayed result by id + tool name + canonical arguments."""
    return Counter(
        json.dumps(
            [entry["tool_call_id"], entry["tool_name"], entry["args"]],
            sort_keys=True,
            separators=(",", ":"),
        )
        for message in request.messages
        if message["type"] == "tool_result"
        for entry in message["tool_results"]
    )


def _resume_all(calls: list[LlmToolCall]) -> tuple[list[LlmRequest], dict[str, int]]:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(once(calls, requests), counts)

    first = runtime.send(session, "go")
    for pending in first.pending_approvals:
        record_approval(session, pending.id, True)
    runtime.send(session, "")
    return requests, counts


def test_two_gated_calls_sharing_an_id_each_keep_their_result_in_the_replay() -> None:
    requests, counts = _resume_all(
        [LlmToolCall("w", "echo", {"value": "P"}), LlmToolCall("w", "echo", {"value": "R"})]
    )

    assert counts == {}
    assert sorted(replayed_results(requests[-1])) == [
        ("echo", REPEATED, "P"),
        ("echo", REPEATED, "R"),
    ]


@pytest.mark.parametrize("gated_first", [True, False], ids=["gated-first", "ungated-first"])
def test_an_ungated_and_a_gated_call_sharing_an_id_each_keep_their_result(
    gated_first: bool,
) -> None:
    calls = [LlmToolCall("w", "shout", {"value": "S"}), LlmToolCall("w", "echo", {"value": "E"})]
    requests, counts = _resume_all(calls[::-1] if gated_first else calls)

    # The ungated call ran, and the model is still shown its real output.
    assert counts == {"shout": 1}
    assert sorted(replayed_results(requests[-1])) == [
        ("echo", REPEATED, "E"),
        ("shout", "shout:S", "S"),
    ]


def test_two_ungated_calls_sharing_an_id_each_keep_their_result() -> None:
    requests, counts = _resume_all(
        [
            LlmToolCall("w", "shout", {"value": "S"}),
            LlmToolCall("w", "shout", {"value": "T"}),
            LlmToolCall("g", "echo", {"value": "E"}),
        ]
    )

    # The step repeats an id, so its gated call never runs (G-392).
    assert counts == {"shout": 2}
    assert sorted(replayed_results(requests[-1])) == [
        ("echo", REPEATED, "E"),
        ("shout", "shout:S", "S"),
        ("shout", "shout:T", "T"),
    ]


def test_a_later_row_still_replaces_the_same_calls_earlier_result() -> None:
    session = a_session("guarded")
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(once([], requests), {})
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "name": "shout", "arguments": {"value": "S"}}],
            },
            {
                "type": "tool_result",
                "tool_results": [
                    {"tool_call_id": "a", "tool_name": "shout", "args": {"value": "S"}, "result": "one"}
                ],
            },
            {
                "type": "tool_result",
                "tool_results": [
                    {"tool_call_id": "a", "tool_name": "shout", "args": {"value": "S"}, "result": "two"}
                ],
            },
        ]
    )

    runtime.send(session, "Next")

    assert replayed_results(requests[-1]) == [("shout", "two", "S")]


# -- item 5: each result emitted once across the whole replay ---------------------

NOTE = {"type": "system", "content": "A host note."}


@pytest.mark.parametrize("note", ["before_the_decision", "after_the_decision"])
def test_a_resumed_steps_results_are_replayed_once_with_a_row_between(note: str) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    calls = [LlmToolCall("a", "shout", {"value": "S"}), LlmToolCall("b", "echo", {"value": "E"})]
    runtime = guarded_runtime(once(calls, requests), counts)
    first = runtime.send(session, "go")
    if note == "before_the_decision":
        session.thread().record([dict(NOTE)])
    record_approval(session, first.pending_approvals[0].id, True)
    if note == "after_the_decision":
        session.thread().record([dict(NOTE)])

    runtime.send(session, "")

    assert counts == {"shout": 1, "echo": 1}
    seen = occurrences(requests[-1])
    assert set(seen.values()) == {1}, seen
    assert len(seen) == 2


def test_a_new_message_on_a_parked_step_replays_each_result_once() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    calls = [LlmToolCall("a", "shout", {"value": "S"}), LlmToolCall("b", "echo", {"value": "E"})]
    runtime = guarded_runtime(once(calls, requests), counts)
    first = runtime.send(session, "go")
    record_approval(session, first.pending_approvals[0].id, True)

    runtime.send(session, "Something else.")

    seen = occurrences(requests[-1])
    assert set(seen.values()) == {1}, seen
    assert len(seen) == 2
