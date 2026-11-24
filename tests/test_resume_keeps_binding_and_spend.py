"""A run paused for a person and resumed is the SAME run (G-321 E).

Its step count already carries across the pause. So must the two other things
that decide what it may spend: its named agent's spend so far, which its
``spend_ceiling`` is measured against, and -- for a subagent child resumed
outside the runner -- the provider and model binding the runner gave it.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from binding_world import World

from prism_harness import HarnessError, LlmRequest, LlmResponse, LlmToolCall, record_approval

GLOBAL = {"provider": "anthropic", "model": "global-model"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def gated_reply(cost: float | None) -> Callable[[World, LlmRequest], LlmResponse]:
    """The first call of any run asks for the gated tool; then it answers."""

    def reply(world: World, request: LlmRequest) -> LlmResponse:
        asked = [r for r in world.requests if r.system_prompt == request.system_prompt]
        if len(asked) == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall(id="g1", name="risky", arguments={})],
                cost_usd=cost,
            )
        return LlmResponse(text="answered", finish_reason="stop", cost_usd=cost)

    return reply


def researcher_alone(cost: float | None, ceiling: float) -> World:
    world = World(
        global_config=GLOBAL,
        agents={
            "researcher": {
                "model": "r-model",
                "spend_ceiling": ceiling,
                "modes": {"research": {"tools": ["risky"], "requires_approval": ["risky"]}},
            }
        },
        reply=gated_reply(cost),
    )
    world.add_tool("researcher", "risky")
    return world


def test_a_spend_ceiling_reached_before_a_pause_still_holds_after_the_resume() -> None:
    world = researcher_alone(0.05, 0.04)
    session = world.session("researcher", "research")

    paused = world.runtime.send(session, "go")
    assert paused.finish_reason == "awaiting_approval"
    record_approval(session, paused.pending_approvals[0].id, True)

    with pytest.raises(HarnessError) as refused:
        world.runtime.send(session, "")

    assert refused.value.code == "agent_spend_ceiling_exceeded"
    assert len(world.requests) == 1


def test_spend_under_the_ceiling_carries_and_the_resume_continues() -> None:
    world = researcher_alone(0.01, 0.04)
    session = world.session("researcher", "research")

    paused = world.runtime.send(session, "go")
    record_approval(session, paused.pending_approvals[0].id, True)
    resumed = world.runtime.send(session, "")

    assert resumed.text == "answered"
    assert len(world.requests) == 2


def test_an_unpriced_call_before_the_pause_still_refuses_after_it() -> None:
    world = researcher_alone(None, 0.04)
    session = world.session("researcher", "research")

    paused = world.runtime.send(session, "go")
    record_approval(session, paused.pending_approvals[0].id, True)

    with pytest.raises(HarnessError) as refused:
        world.runtime.send(session, "")

    assert refused.value.code == "budget_unpriced"


def _paused_row(world: World, session: Any) -> dict[str, Any]:
    return next(
        entry.message
        for entry in reversed(session.thread().messages())
        if entry.message.get("type") == "assistant" and entry.message.get("tool_approval_requests")
    )


@pytest.mark.parametrize(
    "stored",
    [
        "0.5",
        {"cost_usd": -1, "unmetered_runs": 0},
        {"cost_usd": float("nan"), "unmetered_runs": 0},
        {"cost_usd": 0.0, "unmetered_runs": -1},
        {"cost_usd": 0.0, "unmetered_runs": True},
        {"cost_usd": True, "unmetered_runs": 0},
        {"unmetered_runs": 0},
    ],
)
def test_a_malformed_carried_spend_fails_closed(stored: object) -> None:
    world = researcher_alone(0.01, 0.04)
    session = world.session("researcher", "research")
    paused = world.runtime.send(session, "go")
    assert _paused_row(world, session)["harness_agent_spend"] == {
        "cost_usd": 0.01,
        "unmetered_runs": 0,
    }

    # Rewrite the stored row the way a hand edit or a foreign writer would.
    rows = [entry.message for entry in session.thread().messages()]
    rows[-1] = {**rows[-1], "harness_agent_spend": stored}
    session.thread().clear()
    session.thread().record(rows, None)
    record_approval(session, paused.pending_approvals[0].id, True)

    with pytest.raises(HarnessError) as refused:
        world.runtime.send(session, "")

    assert refused.value.code == "budget_unpriced"


def test_a_row_from_an_older_version_carries_nothing() -> None:
    world = researcher_alone(0.05, 0.04)
    session = world.session("researcher", "research")
    paused = world.runtime.send(session, "go")

    rows = [entry.message for entry in session.thread().messages()]
    rows[-1] = {key: value for key, value in rows[-1].items() if key != "harness_agent_spend"}
    session.thread().clear()
    session.thread().record(rows, None)
    record_approval(session, paused.pending_approvals[0].id, True)

    assert world.runtime.send(session, "").text == "answered"


def delegating_world() -> World:
    world = World(
        global_config=GLOBAL,
        agents={
            "planner": {
                "model": "opus-like",
                "modes": {
                    "plan": {
                        "subagents": {
                            "research": {
                                "mode": "research",
                                "provider": "custom-cheap",
                                "model": "cheap",
                            }
                        }
                    },
                    "research": {"tools": ["risky"], "requires_approval": ["risky"]},
                },
            }
        },
    )

    def reply(world: World, request: LlmRequest) -> LlmResponse:
        if request.system_prompt == world.parent_prompt:
            parent = [r for r in world.requests if r.system_prompt == world.parent_prompt]
            if len(parent) == 1:
                return LlmResponse(
                    text="",
                    finish_reason="tool_calls",
                    tool_calls=[LlmToolCall(id="p1", name="research", arguments={"task": "t"})],
                )
            return LlmResponse(text="parent done", finish_reason="stop")
        return gated_reply(0.01)(world, request)

    world._reply = reply
    world.delegate = "research"
    world.offer_subagents("planner", "plan")
    world.add_tool("planner", "risky")
    return world


def test_a_child_resumed_outside_the_runner_keeps_its_bound_provider_and_model() -> None:
    world = delegating_world()
    parent = world.session("planner", "plan")
    world.send(parent)

    child_requests = world.child_requests()
    assert [(r.provider, r.model) for r in child_requests] == [("custom-cheap", "cheap")]

    child = (
        world.harness.for_agent("planner")
        .for_(parent.participant)
        .session("binding::sub::research")
    )
    stored = _paused_row(world, child)
    assert stored["harness_binding"] == {
        "session": {"provider": None, "model": None},
        "declared": {"provider": "custom-cheap", "model": "cheap"},
        "parent_mode": {"provider": None, "model": None},
    }
    record_approval(child, stored["tool_approval_requests"][0]["approval_id"], True)

    # Resumed directly, as a host's approval endpoint does: no RunContext.
    world.runtime.send(child, "")

    resumed = world.child_requests()[-1]
    assert (resumed.provider, resumed.model) == ("custom-cheap", "cheap")


def test_a_malformed_stored_binding_is_ignored() -> None:
    world = delegating_world()
    parent = world.session("planner", "plan")
    world.send(parent)
    child = (
        world.harness.for_agent("planner")
        .for_(parent.participant)
        .session("binding::sub::research")
    )
    rows = [entry.message for entry in child.thread().messages()]
    paused = rows[-1]
    rows[-1] = {**paused, "harness_binding": {"declared": {"provider": 7, "model": "cheap"}}}
    child.thread().clear()
    child.thread().record(rows, None)
    record_approval(child, paused["tool_approval_requests"][0]["approval_id"], True)

    world.runtime.send(child, "")

    resumed = world.child_requests()[-1]
    assert (resumed.provider, resumed.model) == ("anthropic", "opus-like")
