"""Results are matched to calls with both sides' arguments in the same parsed
form (G-392).

Another writer (the PHP reference) may store a call's arguments, and a
result's, as a JSON string. A result is credited to the call it answers
whichever form either side was stored in, a result in the newest step's
segment that matches one of its calls is never credited to an earlier step,
and no call runs twice. Entered at the public seam over stored rows; the model
is scripted.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import Session, record_approval

FORMS = ["map", "string"]


def stored(value: dict[str, Any], form: str) -> Any:
    return json.dumps(value) if form == "string" else value


def step(call_id: str, arguments: Any, approval: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "name": "echo", "arguments": arguments}],
        "additional_content": {},
        "tool_approval_requests": [
            {"approval_id": approval, "tool_call_id": call_id, "arguments_format": 1}
        ],
        "harness_tool_names": ["echo", "shout"],
    }


def result(call_id: str, arguments: Any, text: str, approval: str | None) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": call_id,
                "tool_name": "echo",
                "args": arguments,
                "result": text,
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": (
            [{"approval_id": approval, "approved": True, "reason": None}] if approval else []
        ),
    }


def results(session: Session) -> list[str]:
    return [
        entry["result"]
        for message in session.thread().messages()
        if message.message["type"] == "tool_result"
        for entry in message.message["tool_results"]
    ]


@pytest.mark.parametrize("call_form", FORMS)
@pytest.mark.parametrize("result_form", FORMS)
@pytest.mark.parametrize("arguments", [{"value": "X"}, {}], ids=["named", "empty"])
def test_a_later_call_with_an_earlier_calls_id_and_arguments_runs_once(
    call_form: str, result_form: str, arguments: dict[str, Any]
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step("a", stored(arguments, call_form), "apr_1"),
            result("a", stored(arguments, result_form), "echo:first", "apr_1"),
            {"type": "user", "content": "again"},
            step("a", stored(arguments, call_form), "apr_2"),
        ]
    )
    record_approval(session, "apr_2", True)
    runtime = guarded_runtime(once([]), counts)

    for _ in range(3):
        runtime.send(session, "")

    assert counts == {"echo": 1}
    assert results(session).count("echo:first") == 1
    assert sum(text.startswith("echo:") for text in results(session)) == 2


@pytest.mark.parametrize("call_form", FORMS)
@pytest.mark.parametrize("result_form", FORMS)
def test_a_result_stored_in_either_form_ends_its_own_call(call_form: str, result_form: str) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step("a", stored({"value": "X"}, call_form), "apr_1"),
            result("a", stored({"value": "X"}, result_form), "echo:X", "apr_1"),
        ]
    )
    runtime = guarded_runtime(once([]), counts)

    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {}
    assert results(session).count("echo:X") >= 1
    assert "Tool call was not run because a later step replaced it" not in results(session)


@pytest.mark.parametrize("call_form", FORMS)
@pytest.mark.parametrize("result_form", FORMS)
def test_a_newest_steps_result_is_never_credited_to_an_earlier_open_call(
    call_form: str, result_form: str
) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step("a", {}, "apr_1"),
            {"type": "user", "content": "again"},
            step("a", stored({"value": "X"}, call_form), "apr_2"),
            result("a", stored({"value": "X"}, result_form), "echo:X", "apr_2"),
        ]
    )
    runtime = guarded_runtime(once([]), counts)

    for _ in range(3):
        runtime.send(session, "")

    assert counts == {}
    texts = results(session)
    assert texts.count("Tool call was not run because a later step replaced it") == 1
    assert texts.count("echo:X") >= 1


@pytest.mark.parametrize("form", FORMS)
def test_two_steps_with_the_same_call_and_the_same_result_each_keep_their_own(form: str) -> None:
    # The same call, answered the same way, in two steps: each step's result
    # is its own, not a restatement of the other's, and neither call runs.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    session.thread().record(
        [
            {"type": "user", "content": "go"},
            step("a", stored({"value": "X"}, form), "apr_1"),
            result("a", stored({"value": "X"}, form), "echo:X", "apr_1"),
            {"type": "user", "content": "again"},
            step("a", stored({"value": "X"}, form), "apr_2"),
            result("a", stored({"value": "X"}, form), "echo:X", "apr_2"),
        ]
    )
    runtime = guarded_runtime(once([]), counts)

    for _ in range(3):
        runtime.send(session, "")

    assert counts == {}
    assert set(results(session)) == {"echo:X"}
