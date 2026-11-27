"""The write-ahead started row (G-410).

Just before an approved call runs, the harness records one row naming it,
in the bytes every language stores (the harness-thread-rows corpus case
thr-0022). A started call with no result reads "Tool call was interrupted,
it may have run" and never runs again; the row never reaches the replay, and
a host's forged started row changes nothing. The decision that stands at the
moment the call starts decides whether it starts. Entered at the public seam;
the model is scripted.
"""

from __future__ import annotations

import json
from typing import Any

from test_runtime import GUARDED, CountingTool, a_session, guarded_runtime, once

from prism_harness import (
    AgentRuntime,
    LlmRequest,
    LlmToolCall,
    Session,
    ToolRegistry,
    record_approval,
)
from prism_harness.thread_rows import started_row

INTERRUPTED = "Tool call was interrupted, it may have run"

#: thr-0022's stored bytes, as the PHP reference and the TypeScript port write them.
THR_0022 = (
    '[{"type":"tool_result","tool_results":[],"tool_approval_responses":[],'
    '"harness_started":{"tool_call_id":"call_paris","tool_name":"weather","approval_id":"apr_1"},'
    '"harness_answers_step":{"position":2}}]'
)


def test_the_started_row_is_byte_identical_to_the_shared_fixture() -> None:
    row = started_row("call_paris", "weather", "apr_1", 2)
    assert json.dumps([row], ensure_ascii=False, separators=(",", ":")) == THR_0022


def stored(session: Session) -> list[dict[str, Any]]:
    thread = session.thread()
    # The stored entries themselves.
    data = thread._store.get(thread._key)
    assert data is not None
    entries: list[dict[str, Any]] = data["messages"]
    return entries


def test_the_harness_records_the_started_row_before_the_call_runs() -> None:
    session = a_session("guarded")
    seen_before_run: list[bool] = []
    state: dict[str, str] = {}

    class Watching:
        @property
        def name(self) -> str:
            return "echo"

        def handle(self, args: dict[str, Any]) -> Any:
            seen_before_run.append(
                any(
                    e["message"].get("harness_started", {}).get("approval_id") == state["approval"]
                    for e in stored(session)
                )
            )
            return "echo:P"

    runtime = AgentRuntime(
        client=once([LlmToolCall("p", "echo", {"value": "P"})]),
        modes=GUARDED,
        tools=ToolRegistry().register(Watching()).register(CountingTool("shout", {})),
    )
    approval = runtime.send(session, "go").pending_approvals[0].id
    state["approval"] = approval
    record_approval(session, approval, True)
    position = next(
        m.position
        for m in session.thread().messages()
        if m.message["type"] == "assistant" and m.message.get("tool_calls")
    )
    runtime.send(session, "")

    [entry] = [e for e in stored(session) if "harness_started" in e["message"]]
    assert entry["harness"] is True
    assert json.dumps(entry["message"], separators=(",", ":")) == (
        '{"type":"tool_result","tool_results":[],"tool_approval_responses":[],'
        '"harness_started":{"tool_call_id":"p","tool_name":"echo","approval_id":"'
        + approval
        + '"},"harness_answers_step":{"position":'
        + str(position)
        + "}}"
    )
    assert seen_before_run == [True]


def test_the_started_row_never_reaches_the_replay() -> None:
    session = a_session("guarded")
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})], requests), {})
    approval = runtime.send(session, "go").pending_approvals[0].id
    record_approval(session, approval, True)
    runtime.send(session, "")

    assert "harness_started" not in json.dumps(requests[-1].messages)


def test_a_started_row_a_host_records_changes_nothing() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("p", "echo", {"value": "P"})]), counts)
    approval = runtime.send(session, "go").pending_approvals[0].id
    position = next(
        m.position
        for m in session.thread().messages()
        if m.message["type"] == "assistant" and m.message.get("tool_calls")
    )
    session.thread().record([started_row("p", "echo", approval, position)])
    record_approval(session, approval, True)
    runtime.send(session, "")
    runtime.send(session, "")

    assert counts == {"echo": 1}
    results = [
        e["result"]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result"
        for e in m.message["tool_results"]
    ]
    assert INTERRUPTED not in results


NOT_OPEN = "Not run: this call is not the one that was approved, or it is no longer open."


def test_a_denial_recorded_after_the_resume_read_its_history_refuses_the_call() -> None:
    session = a_session("guarded")
    ran: list[str] = []
    approvals: dict[str, str] = {}

    class Denying:
        @property
        def name(self) -> str:
            return "echo"

        def handle(self, args: dict[str, Any]) -> Any:
            ran.append(str(args["value"]))
            if args["value"] == "A":
                # The person denies the second call while the first runs.
                record_approval(session, approvals["B"], False, "changed my mind")
            return f"echo:{args['value']}"

    runtime = AgentRuntime(
        client=once(
            [LlmToolCall("a", "echo", {"value": "A"}), LlmToolCall("b", "echo", {"value": "B"})]
        ),
        modes=GUARDED,
        tools=ToolRegistry().register(Denying()).register(CountingTool("shout", {})),
    )
    first = runtime.send(session, "go")
    for pending in first.pending_approvals:
        approvals[str(pending.arguments["value"])] = pending.id
        record_approval(session, pending.id, True)
    runtime.send(session, "")
    runtime.send(session, "")

    assert ran == ["A"]
    results = {
        e["args"]["value"]: e["result"]
        for m in session.thread().messages()
        if m.message["type"] == "tool_result"
        for e in m.message["tool_results"]
    }
    assert results == {"A": "echo:A", "B": NOT_OPEN}
