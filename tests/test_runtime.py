"""The run loop. Mirrors prism-harness-ts/test/runtime.test.ts."""

from __future__ import annotations

import json
import tempfile
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


def test_stops_and_does_not_run_a_gated_tool_without_approval() -> None:
    # Failing closed is the only safe direction: an unanswered approval that
    # executed anyway is exactly what the mechanism exists to prevent.
    session = a_session("guarded")
    handled = {"n": 0}
    tools = ToolRegistry().register(
        EchoTool(on_call=lambda: handled.__setitem__("n", handled["n"] + 1))
    )

    response = a_runtime(
        scripted(
            [
                LlmResponse(
                    text="",
                    finish_reason="tool_calls",
                    tool_calls=[LlmToolCall("c1", "echo", {"value": "x"})],
                )
            ]
        ),
        tools=tools,
    ).send(session, "go")

    assert handled["n"] == 0
    assert response.finish_reason == "awaiting_approval"
    assert [p.tool for p in response.pending_approvals] == ["echo"]


def test_writes_the_request_to_the_thread() -> None:
    session = a_session("guarded")
    a_runtime(
        scripted(
            [
                LlmResponse(
                    text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("c1", "echo", {})]
                )
            ]
        )
    ).send(session, "go")

    assert "tool_approval_request" in [m.message["type"] for m in session.thread().messages()]


def test_runs_the_tool_once_the_approval_is_recorded_on_a_resumed_turn() -> None:
    # The approval a person grants this morning is a durable row, so the worker
    # that resumes tonight -- a different process, possibly after a deploy --
    # reads the same answer.
    session = a_session("guarded")
    handled = {"n": 0}
    tools = ToolRegistry().register(
        EchoTool(on_call=lambda: handled.__setitem__("n", handled["n"] + 1))
    )
    turn = {"n": 0}

    def client(_request: LlmRequest) -> LlmResponse:
        turn["n"] += 1
        if turn["n"] <= 2:
            return LlmResponse(
                text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("c1", "echo", {})]
            )
        return LlmResponse(text="Finished.", finish_reason="stop")

    runtime = a_runtime(client, tools=tools)

    runtime.send(session, "go")
    assert handled["n"] == 0

    record_approval(session, "c1", True)
    resumed = runtime.send(session, "")

    assert handled["n"] == 1
    assert resumed.text == "Finished."


def test_does_not_ask_twice_once_an_approval_is_answered() -> None:
    session = a_session("guarded")
    client = scripted(
        [
            LlmResponse(
                text="", finish_reason="tool_calls", tool_calls=[LlmToolCall("c1", "echo", {})]
            ),
            LlmResponse(text="done", finish_reason="stop"),
        ]
    )

    a_runtime(client).send(session, "go")
    record_approval(session, "c1", True)

    assert a_runtime(client).send(session, "").pending_approvals == []


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
    assert result.message["failed"] is True
    assert "exploded" in result.message["result"]


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
