"""The run loop. Mirrors prism-harness-ts/test/runtime.test.ts."""

from __future__ import annotations

import json
import re
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from prism_harness import (
    MAX_DEPTH,
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    RunBudget,
    RunContext,
    Session,
    Subagent,
    ToolAuthorizer,
    ToolRegistry,
    record_approval,
)

MODES = ModeRegistry(
    {
        "default": "chat",
        "modes": {
            "chat": {"system_prompt": "Be brief.", "tools": ["echo"], "max_steps": 4},
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo"],
                "max_steps": 4,
                "requires_approval": ["echo"],
            },
        },
    }
)


class EchoTool:
    def __init__(self, on_call: Callable[[], None] | None = None, explode: bool = False) -> None:
        self._on_call = on_call
        self._explode = explode

    @property
    def name(self) -> str:
        return "echo"

    def handle(self, args: dict[str, Any]) -> Any:
        if self._on_call is not None:
            self._on_call()
        if self._explode:
            raise RuntimeError("the tool exploded")
        return f"echoed:{args.get('value', '')}"


def scripted(responses: list[LlmResponse]) -> Callable[[LlmRequest], LlmResponse]:
    """Each call returns the next response, then repeats the last."""
    state = {"call": 0}

    def client(_request: LlmRequest) -> LlmResponse:
        response = responses[min(state["call"], len(responses) - 1)]
        state["call"] += 1
        return response

    return client


def a_session(mode: str = "chat") -> Session:
    directory = tempfile.mkdtemp(prefix="prism-harness-runtime-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    session = harness.for_(Participant("User", 1)).session("support")
    session.using_mode(mode).using_provider("anthropic").using_model("claude-sonnet-4-5")
    return session


def a_runtime(
    client: Callable[[LlmRequest], LlmResponse],
    tools: ToolRegistry | None = None,
    authorizer: ToolAuthorizer | None = None,
    events: HarnessEvents | None = None,
) -> AgentRuntime:
    return AgentRuntime(
        client=client,
        modes=MODES,
        tools=tools or ToolRegistry().register(EchoTool()),
        authorizer=authorizer,
        events=events,
    )


# -- a plain turn ------------------------------------------------------------


def test_returns_the_text_and_records_both_messages() -> None:
    session = a_session()
    client = scripted([LlmResponse(text="Hello.", finish_reason="stop")])

    response = a_runtime(client).send(session, "Hi")

    assert response.text == "Hello."
    assert response.steps == 1
    assert [m.message["type"] for m in session.thread().messages()] == ["user", "assistant"]


def test_marks_the_run_completed_with_the_tools_it_reached_for() -> None:
    session = a_session()
    client = scripted(
        [
            LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("c1", "echo", {"value": "x"})],
            ),
            LlmResponse(text="Done.", finish_reason="stop"),
        ]
    )

    response = a_runtime(client).send(session, "Use the tool")

    assert response.tool_calls == ["echo"]
    run = session.run()
    assert run is not None
    assert run["status"] == "completed"
    assert run["tool_calls"] == ["echo"]


def test_does_not_record_a_user_message_for_an_empty_prompt() -> None:
    # An empty prompt is how a run resumes after an approval: the conversation
    # already holds the request and the decision, and a new prompt there would
    # be a second instruction competing with the one the tool call came from.
    session = a_session()
    a_runtime(scripted([LlmResponse(text="ok", finish_reason="stop")])).send(session, "")

    assert [m.message["type"] for m in session.thread().messages()] == ["assistant"]


# -- what the next step is sent back (G-58) ------------------------------------


def test_records_call_arguments_ids_and_provider_state_and_sends_them_on_the_next_step() -> None:
    # The next request is built from the thread. Recorded with an id and a name
    # only, a client had no input to send for the tool_use it was replaying, and
    # nowhere to find the thinking signature Anthropic requires with it.
    session = a_session()
    requests: list[LlmRequest] = []
    responses = [
        LlmResponse(
            text="Checking.",
            finish_reason="tool_calls",
            tool_calls=[
                LlmToolCall("fc_1", "echo", {"value": "x"}, result_id="call_1", reasoning_id="rs_1")
            ],
            additional_content={"thinking": "Use the tool.", "thinking_signature": "sig-1"},
        ),
        LlmResponse(text="Done.", finish_reason="stop"),
    ]

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return responses[len(requests) - 1]

    a_runtime(client).send(session, "Use the tool")

    assert requests[1].messages[1:] == [
        {
            "type": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "fc_1",
                    "name": "echo",
                    "arguments": {"value": "x"},
                    "result_id": "call_1",
                    "reasoning_id": "rs_1",
                    "reasoning_summary": None,
                }
            ],
            "additional_content": {"thinking": "Use the tool.", "thinking_signature": "sig-1"},
            "tool_approval_requests": [],
        },
        {
            "type": "tool_result",
            "tool_results": [
                {
                    "tool_call_id": "fc_1",
                    "tool_name": "echo",
                    "args": {"value": "x"},
                    "result": "echoed:x",
                    "tool_call_result_id": "call_1",
                    "artifacts": [],
                }
            ],
            "tool_approval_responses": [],
        },
    ]


def test_records_empty_provider_state_when_the_client_reports_none() -> None:
    session = a_session()

    a_runtime(scripted([LlmResponse(text="Hello.", finish_reason="stop")])).send(session, "Hi")

    assert session.thread().messages()[-1].message == {
        "type": "assistant",
        "content": "Hello.",
        "tool_calls": [],
        "additional_content": {},
        "tool_approval_requests": [],
    }


def test_records_all_of_a_steps_results_as_one_row() -> None:
    session = a_session()
    client = scripted(
        [
            LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    LlmToolCall("c1", "echo", {"value": "a"}),
                    LlmToolCall("c2", "echo", {"value": "b"}),
                ],
            ),
            LlmResponse(text="Done.", finish_reason="stop"),
        ]
    )

    a_runtime(client).send(session, "go")
    rows = [m.message for m in session.thread().messages()]

    assert [row["type"] for row in rows] == ["user", "assistant", "tool_result", "assistant"]
    assert [entry["result"] for entry in rows[2]["tool_results"]] == ["echoed:a", "echoed:b"]


# -- budgets -----------------------------------------------------------------


def test_stops_before_taking_a_step_it_cannot_afford() -> None:
    # Checking afterwards means the step that broke the limit has already been
    # paid for, which makes a budget a report rather than a control.
    session = a_session()
    calls = {"n": 0}

    def client(_request: LlmRequest) -> LlmResponse:
        calls["n"] += 1
        return LlmResponse(
            text="again",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(f"c{calls['n']}", "echo", {})],
        )

    context = RunContext.root("root", RunBudget(2))
    response = a_runtime(client).send(session, "go", context=context)

    assert calls["n"] == 2
    assert "step budget exhausted" in (response.stopped_because or "")
    assert response.finish_reason == "budget_exhausted"


def test_reports_a_cancellation_as_the_reason_it_stopped() -> None:
    session = a_session()
    context = RunContext.root("root", RunBudget(4))
    context.ledger.cancel("the user closed the tab")

    response = a_runtime(scripted([LlmResponse(text="never", finish_reason="stop")])).send(
        session, "go", context=context
    )

    assert response.stopped_because == "the user closed the tab"


def test_refuses_a_run_nested_past_the_depth_ceiling() -> None:
    session = a_session()
    context = RunContext.root("root", RunBudget(8))
    child = Subagent("r", "", "chat", RunBudget(8))
    for _ in range(MAX_DEPTH):
        context = context.for_child(child, "root")

    with pytest.raises(HarnessError) as caught:
        a_runtime(scripted([LlmResponse(text="x", finish_reason="stop")])).send(
            session, "go", context=context
        )

    assert caught.value.code == "run_not_permitted"


# -- approvals ---------------------------------------------------------------

GUARDED = ModeRegistry(
    {
        "default": "guarded",
        "modes": {
            "guarded": {
                "system_prompt": "Careful.",
                "tools": ["echo", "shout"],
                "max_steps": 4,
                "requires_approval": ["echo"],
            }
        },
    }
)


class CountingTool:
    def __init__(self, name: str, counts: dict[str, int]) -> None:
        self._name = name
        self._counts = counts

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        self._counts[self._name] = self._counts.get(self._name, 0) + 1
        return f"{self._name}:{args.get('value', '')}"


def guarded_runtime(
    client: Callable[[LlmRequest], LlmResponse], counts: dict[str, int]
) -> AgentRuntime:
    tools = (
        ToolRegistry()
        .register(CountingTool("echo", counts))
        .register(CountingTool("shout", counts))
    )
    return AgentRuntime(client=client, modes=GUARDED, tools=tools)


def once(
    tool_calls: list[LlmToolCall], requests: list[LlmRequest] | None = None
) -> Callable[[LlmRequest], LlmResponse]:
    """A model that asks for the calls ONCE; a real provider does not re-issue a call under the same id."""
    seen: list[LlmRequest] = requests if requests is not None else []

    def client(request: LlmRequest) -> LlmResponse:
        seen.append(request)
        if len(seen) == 1:
            return LlmResponse(text="", finish_reason="tool_calls", tool_calls=tool_calls)
        return LlmResponse(text=f"Finished after {len(seen) - 1}.", finish_reason="stop")

    return client


def test_stops_and_does_not_run_a_gated_tool_without_approval() -> None:
    # Failing closed is the only safe direction: an unanswered approval that
    # executed anyway is exactly what the mechanism exists to prevent.
    session = a_session("guarded")
    counts: dict[str, int] = {}

    response = guarded_runtime(once([LlmToolCall("c1", "echo", {"value": "x"})]), counts).send(
        session, "go"
    )

    assert counts == {}
    assert response.finish_reason == "awaiting_approval"
    [pending] = response.pending_approvals
    assert re.fullmatch(r"apr_[0-9a-f]{32}", pending.id)
    assert (pending.tool_call_id, pending.tool, pending.arguments) == ("c1", "echo", {"value": "x"})


def test_writes_the_request_onto_the_assistant_row() -> None:
    session = a_session("guarded")

    response = guarded_runtime(once([LlmToolCall("c1", "echo", {})]), {}).send(session, "go")

    assistant = next(
        m.message for m in session.thread().messages() if m.message["type"] == "assistant"
    )
    assert assistant["tool_approval_requests"] == [
        {"approval_id": response.pending_approvals[0].id, "tool_call_id": "c1"}
    ]


def test_runs_the_calls_that_need_nobody_before_stopping_for_the_rest() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}

    response = guarded_runtime(
        once(
            [LlmToolCall("c1", "echo", {"value": "x"}), LlmToolCall("c2", "shout", {"value": "y"})]
        ),
        counts,
    ).send(session, "go")

    assert counts == {"shout": 1}
    assert [p.tool_call_id for p in response.pending_approvals] == ["c1"]
    assert [m.message["type"] for m in session.thread().messages()] == [
        "user",
        "assistant",
        "tool_result",
    ]


def test_runs_an_approved_call_once_on_the_resumed_turn_without_asking_the_model_again() -> None:
    # A real provider asked again issues the call afresh under a new id, so a
    # decision recorded against the old id would never match.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(once([LlmToolCall("c1", "echo", {"value": "x"})], requests), counts)

    first = runtime.send(session, "go")
    record_approval(session, first.pending_approvals[0].id, True)
    resumed = runtime.send(session, "")

    assert counts == {"echo": 1}
    assert resumed.text == "Finished after 1."
    assert resumed.tool_calls == ["echo"]
    last = requests[1].messages[-1]
    assert last["type"] == "tool_result"
    assert [e["result"] for e in last["tool_results"]] == ["echo:x"]
    assert [d["approved"] for d in last["tool_approval_responses"]] == [True]


def test_sends_a_denied_call_the_reason_and_an_unanswered_one_a_refusal() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    requests: list[LlmRequest] = []
    runtime = guarded_runtime(
        once(
            [LlmToolCall("c1", "echo", {"value": "a"}), LlmToolCall("c2", "echo", {"value": "b"})],
            requests,
        ),
        counts,
    )

    first = runtime.send(session, "go")
    record_approval(session, first.pending_approvals[0].id, False, "not today")
    runtime.send(session, "")

    assert counts == {}
    assert [(e["tool_call_id"], e["result"]) for e in requests[1].messages[-1]["tool_results"]] == [
        ("c1", "not today"),
        ("c2", "No approval response provided"),
    ]


def test_runs_every_approved_call_when_all_decisions_are_recorded_first() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(
        once(
            [LlmToolCall("c1", "echo", {"value": "a"}), LlmToolCall("c2", "echo", {"value": "b"})]
        ),
        counts,
    )

    first = runtime.send(session, "go")
    for pending in first.pending_approvals:
        record_approval(session, pending.id, True)
    runtime.send(session, "")

    assert counts == {"echo": 2}


def test_never_runs_an_approved_call_again_once_it_has_a_result() -> None:
    # The decision stays in the thread, and every later turn reads it again.
    session = a_session("guarded")
    counts: dict[str, int] = {}
    runtime = guarded_runtime(once([LlmToolCall("c1", "echo", {"value": "x"})]), counts)

    first = runtime.send(session, "go")
    record_approval(session, first.pending_approvals[0].id, True)
    runtime.send(session, "")
    runtime.send(session, "And again?")
    runtime.send(session, "")

    assert counts == {"echo": 1}


def test_runs_an_approved_call_once_when_two_workers_resume_at_once() -> None:
    # Without the session lock both read the approved call with no result, and
    # both run it.
    directory = tempfile.mkdtemp(prefix="prism-harness-concurrent-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    counts: dict[str, int] = {}
    guard = threading.Lock()

    class SlowEcho:
        name = "echo"

        def handle(self, args: dict[str, Any]) -> Any:
            with guard:
                counts["echo"] = counts.get("echo", 0) + 1
            time.sleep(0.05)
            return "ran"

    def open_session() -> Session:
        session = harness.for_(Participant("User", 1)).session("support")
        session.using_mode("guarded")
        return session

    def client(request: LlmRequest) -> LlmResponse:
        if any(row["type"] == "assistant" for row in request.messages):
            return LlmResponse(text="Finished.", finish_reason="stop")
        return LlmResponse(
            text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("c1", "echo", {})]
        )

    def runtime() -> AgentRuntime:
        tools = ToolRegistry().register(SlowEcho()).register(CountingTool("shout", counts))
        return AgentRuntime(client=client, modes=GUARDED, tools=tools)

    first = runtime().send(open_session(), "go")
    record_approval(open_session(), first.pending_approvals[0].id, True)

    workers = [
        threading.Thread(target=lambda: runtime().send(open_session(), "")) for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert counts == {"echo": 1}


def test_records_an_approved_call_whose_tool_is_no_longer_offered_as_not_run() -> None:
    session = a_session("guarded")
    counts: dict[str, int] = {}
    first = guarded_runtime(once([LlmToolCall("c1", "echo", {})]), counts).send(session, "go")
    record_approval(session, first.pending_approvals[0].id, True)

    # This run is offered shout only.
    runtime = guarded_runtime(
        scripted([LlmResponse(text="Finished.", finish_reason="stop")]), counts
    )
    runtime.send(session, "", ["shout"])
    again = runtime.send(session, "Next", ["shout"])

    assert counts == {}
    assert again.text == "Finished."
    rows = [m.message for m in session.thread().messages() if m.message["type"] == "tool_result"]
    assert "Not run: echo is not available to this run." in json.dumps(rows)


def test_resumes_an_approval_recorded_by_0_1_0_in_the_rows_it_wrote() -> None:
    # 0.1.0 kept the request in its own row, keyed by the CALL id, and answered
    # it in a tool_approval_response row.
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
            {
                "type": "tool_approval_response",
                "approval_id": "c1",
                "approved": True,
                "reason": None,
            },
        ]
    )

    resumed = guarded_runtime(
        scripted([LlmResponse(text="Finished.", finish_reason="stop")]), counts
    ).send(session, "")

    assert counts == {"echo": 1}
    assert resumed.text == "Finished."


# -- tools -------------------------------------------------------------------


def test_records_a_failed_tool_as_a_result_rather_than_crashing_the_run() -> None:
    # The model can often recover, and losing the whole turn to one bad call is
    # worse than telling it what happened.
    session = a_session()
    tools = ToolRegistry().register(EchoTool(explode=True))
    client = scripted(
        [
            LlmResponse(
                text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("c1", "echo", {})]
            ),
            LlmResponse(text="Recovered.", finish_reason="stop"),
        ]
    )

    response = a_runtime(client, tools=tools).send(session, "go")

    assert response.text == "Recovered."
    result = next(m for m in session.thread().messages() if m.message["type"] == "tool_result")
    assert "exploded" in result.message["tool_results"][0]["result"]


def test_a_refused_call_propagates() -> None:
    session = a_session()
    authorizer = ToolAuthorizer(enabled=True, call=lambda _s, _t, _a: False)
    client = scripted(
        [
            LlmResponse(
                text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("c1", "echo", {})]
            )
        ]
    )

    with pytest.raises(HarnessError) as caught:
        a_runtime(client, authorizer=authorizer).send(session, "go")

    assert caught.value.code == "call_not_authorized"


# -- events and failures -----------------------------------------------------


def test_emits_started_and_finished_with_tool_names_only() -> None:
    session = a_session()
    events = HarnessEvents()
    seen: list[HarnessEvent] = []
    events.listen(seen.append)

    client = scripted(
        [
            LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall("c1", "echo", {"secret": "do-not-log"})],
            ),
            LlmResponse(text="done", finish_reason="stop", cost_usd=0.01),
        ]
    )

    a_runtime(client, events=events).send(session, "go")

    assert [event.type for event in seen] == ["run.started", "run.finished"]
    assert "do-not-log" not in json.dumps([HarnessEvents.to_dict(e) for e in seen])


def test_reports_a_none_cost_rather_than_pretending_the_tree_spent_nothing() -> None:
    session = a_session()
    events = HarnessEvents()
    seen: list[HarnessEvent] = []
    events.listen(seen.append)

    a_runtime(scripted([LlmResponse(text="done", finish_reason="stop")]), events=events).send(
        session, "go"
    )

    finished = next(e for e in seen if e.type == "run.finished")
    assert finished.cost_usd is None  # type: ignore[union-attr]


def test_marks_the_run_failed_and_emits_when_the_model_raises() -> None:
    session = a_session()
    events = HarnessEvents()
    seen: list[HarnessEvent] = []
    events.listen(seen.append)

    def client(_request: LlmRequest) -> LlmResponse:
        raise RuntimeError("the provider is down")

    with pytest.raises(RuntimeError, match="provider is down"):
        a_runtime(client, events=events).send(session, "go")

    run = session.run()
    assert run is not None
    assert run["status"] == "failed"
    assert [event.type for event in seen] == ["run.started", "run.failed"]
