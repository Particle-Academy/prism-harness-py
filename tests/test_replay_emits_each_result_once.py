"""The replay emits each call's result EXACTLY ONCE, in its own segment (G-392).

A step's results may be stored in tool result rows that are not consecutive:
a host may record a note between them, or a new message may follow the
decisions. The history sent to the model holds each result once, in the
step's segment. Entered at the public seam; the model is scripted and its
requests are captured.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest
from test_runtime import a_session, guarded_runtime, once

from prism_harness import LlmRequest, LlmToolCall, record_approval


def emitted(request: LlmRequest) -> Counter[tuple[int, str, str, str]]:
    """Each replayed result by (segment, call id, tool, arguments)."""
    seen: Counter[tuple[int, str, str, str]] = Counter()
    segment = 0
    for message in request.messages:
        if message["type"] == "assistant" and message.get("tool_calls"):
            segment += 1
        if message["type"] == "tool_result":
            for entry in message["tool_results"]:
                seen[
                    (
                        segment,
                        entry["tool_call_id"],
                        entry["tool_name"],
                        json.dumps(entry["args"], sort_keys=True),
                    )
                ] += 1
    return seen


def calls() -> list[LlmToolCall]:
    return [LlmToolCall("s", "shout", {"value": "S"}), LlmToolCall("p", "echo", {"value": "P"})]


NOTES: dict[str, dict[str, Any]] = {
    "system": {"type": "system", "content": "A note."},
    "other": {"type": "developer", "content": "A note."},
}


@pytest.mark.parametrize("note", sorted(NOTES))
@pytest.mark.parametrize("when", ["before-decision", "after-decision"])
def test_a_note_between_a_steps_result_rows_never_repeats_a_result(note: str, when: str) -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(once(calls(), requests), counts)

    first = runtime.send(session, "go")
    if when == "before-decision":
        session.thread().record([NOTES[note]])
    record_approval(session, first.pending_approvals[0].id, True)
    if when == "after-decision":
        session.thread().record([NOTES[note]])
    runtime.send(session, "")
    runtime.send(session, "Next")

    assert counts == {"shout": 1, "echo": 1}
    for request in requests:
        assert max(emitted(request).values(), default=0) <= 1, emitted(request)


def test_a_new_message_after_the_decisions_never_repeats_a_result() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(once(calls(), requests), counts)

    first = runtime.send(session, "go")
    record_approval(session, first.pending_approvals[0].id, True)
    runtime.send(session, "A new message")
    runtime.send(session, "Another")

    assert counts == {"shout": 1, "echo": 1}
    for request in requests:
        assert max(emitted(request).values(), default=0) <= 1, emitted(request)
