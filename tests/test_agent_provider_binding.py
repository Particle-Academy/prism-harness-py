"""G-321 E: several agents in one app, each on its own provider and model.

Every test enters where an application enters -- a session sent through the
runtime -- and asserts what the fake client was actually asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from binding_world import World

from prism_harness import HarnessError, HarnessEvents, LlmRequest, LlmResponse, LlmToolCall

GLOBAL = {"provider": "anthropic", "model": "global-model"}


def binding(request: LlmRequest) -> tuple[str, str]:
    return (request.provider, request.model)


def planner(plan: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"model": "agent-model", "modes": {"plan": plan, "research": {}}, **extra}


def test_a_mode_model_overrides_the_agent_model() -> None:
    world = World(global_config=GLOBAL, agents={"planner": planner({"model": "mode-model"})})

    world.send(world.session("planner", "plan"))

    assert [binding(r) for r in world.requests] == [("anthropic", "mode-model")]


def test_a_mode_provider_overrides_the_global_provider_in_the_legacy_namespace() -> None:
    world = World(global_config=GLOBAL, modes={"chat": {"provider": "custom-cheap"}})

    world.send(world.session(None, "chat"))

    assert [binding(r) for r in world.requests] == [("custom-cheap", "global-model")]


def test_a_session_override_beats_the_mode() -> None:
    world = World(global_config=GLOBAL, agents={"planner": planner({"model": "mode-model"})})

    world.send(world.session("planner", "plan", model="session-model"))

    assert [binding(r) for r in world.requests] == [("anthropic", "session-model")]


def delegating(plan_extra: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
    return {"subagents": {"research": {"mode": "research", **research}}, **plan_extra}


def test_a_subagent_runs_on_a_cheaper_model_under_a_parent_on_another() -> None:
    world = World(
        global_config=GLOBAL,
        agents={
            "planner": planner(
                delegating({"model": "opus-like"}, {"provider": "custom-cheap", "model": "cheap"})
            )
        },
    )
    world.delegate = "research"
    world.offer_subagents("planner", "plan")

    world.send(world.session("planner", "plan"))

    assert [binding(r) for r in world.parent_requests()] == [("anthropic", "opus-like")] * 2
    assert [binding(r) for r in world.child_requests()] == [("custom-cheap", "cheap")]


def test_the_parents_session_override_is_not_inherited_by_default() -> None:
    world = World(global_config=GLOBAL, agents={"planner": planner(delegating({}, {}))})
    world.delegate = "research"
    world.offer_subagents("planner", "plan")

    world.send(world.session("planner", "plan", provider="custom-cheap", model="session-model"))

    assert binding(world.parent_requests()[0]) == ("custom-cheap", "session-model")
    assert [binding(r) for r in world.child_requests()] == [("anthropic", "agent-model")]


def test_a_subagent_that_asks_inherits_the_parents_session_override() -> None:
    world = World(
        global_config=GLOBAL,
        agents={
            "planner": planner(delegating({}, {"model": "cheap", "inherit_session_model": True}))
        },
    )
    world.delegate = "research"
    world.offer_subagents("planner", "plan")

    world.send(world.session("planner", "plan", model="session-model"))

    assert [binding(r) for r in world.child_requests()] == [("anthropic", "session-model")]


def test_the_runner_no_longer_writes_an_override_onto_the_child_session() -> None:
    world = World(global_config=GLOBAL, agents={"planner": planner(delegating({}, {}))})
    world.delegate = "research"
    world.offer_subagents("planner", "plan")
    parent = world.session("planner", "plan", model="session-model")

    world.send(parent)

    child = (
        world.harness.for_agent("planner")
        .for_(parent.participant)
        .session("binding::sub::research")
    )
    assert child.model() is None
    assert child.provider() is None


def test_a_subagent_on_a_declared_agent_runs_on_that_agents_binding() -> None:
    world = World(
        global_config=GLOBAL,
        agents={
            "planner": planner(
                delegating(
                    {"model": "opus-like", "agents": ["researcher"]}, {"agent": "researcher"}
                )
            ),
            "researcher": {
                "provider": "custom-cheap",
                "model": "r-model",
                "modes": {"research": {}},
            },
        },
    )
    world.delegate = "research"
    world.offer_subagents("planner", "plan")

    world.send(world.session("planner", "plan"))

    assert [binding(r) for r in world.child_requests()] == [("custom-cheap", "r-model")]


@pytest.mark.parametrize(
    ("allowed", "named", "message"),
    [
        ([], "researcher", "which the mode does not list in its agents"),
        (["researcher"], " researcher", "malformed agent id"),
        (["researcher"], "researcher ", "malformed agent id"),
        (["researcher"], "Researcher", "malformed agent id"),
        (["Researcher"], "researcher", "malformed agents list"),
        ([" researcher"], "researcher", "malformed agents list"),
    ],
    ids=["undeclared", "padded-left", "padded-right", "mixed-case", "list-mixed", "list-padded"],
)
def test_an_agent_the_mode_does_not_declare_is_refused_before_any_call(
    allowed: list[str], named: str, message: str
) -> None:
    world = World(
        global_config=GLOBAL,
        agents={
            "planner": planner(delegating({"agents": allowed}, {"agent": named})),
            "researcher": {"model": "r-model", "modes": {"research": {}}},
        },
    )
    world.delegate = "research"
    world.offer_subagents("planner", "plan")

    with pytest.raises(HarnessError, match=message) as refused:
        world.send(world.session("planner", "plan"))

    assert refused.value.code == "mode_malformed"
    assert world.requests == []


def test_a_declared_agent_without_the_subagents_mode_is_refused() -> None:
    world = World(
        global_config=GLOBAL,
        agents={
            "planner": planner(delegating({"agents": ["researcher"]}, {"agent": "researcher"})),
            "researcher": {"modes": {"other": {}}},
        },
    )

    with pytest.raises(HarnessError, match=r"whose mode \[research\] is not configured"):
        world.send(world.session("planner", "plan"))

    assert world.requests == []


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ({"model": ""}, "malformed provider/model"),
        ({"provider": 3}, "malformed provider/model"),
        ({"inherit_session_model": "yes"}, "malformed inherit_session_model"),
    ],
)
def test_a_malformed_subagent_binding_is_refused(declaration: dict[str, Any], message: str) -> None:
    world = World(global_config=GLOBAL, agents={"planner": planner(delegating({}, declaration))})

    with pytest.raises(HarnessError, match=message):
        world.send(world.session("planner", "plan"))

    assert world.requests == []


def test_a_malformed_mode_binding_is_refused() -> None:
    world = World(global_config=GLOBAL, agents={"planner": planner({"model": ""})})

    with pytest.raises(HarnessError, match="malformed provider/model"):
        world.send(world.session("planner", "plan"))

    assert world.requests == []


@pytest.mark.parametrize("ceiling", [0, -1.0, float("inf"), float("nan"), True, "1"])
def test_a_malformed_spend_ceiling_is_refused(ceiling: object) -> None:
    with pytest.raises(ValueError, match="spend_ceiling must be a positive finite number"):
        World(global_config=GLOBAL, agents={"planner": planner({}, spend_ceiling=ceiling)})


def priced(cost: float | None) -> Callable[[World, LlmRequest], LlmResponse]:
    def reply(world: World, request: LlmRequest) -> LlmResponse:
        answer = World.default_reply(world, request)
        return LlmResponse(
            text=answer.text,
            finish_reason=answer.finish_reason,
            tool_calls=answer.tool_calls,
            cost_usd=cost,
        )

    return reply


def researcher_world(cost: float | None, ceiling: float) -> World:
    world = World(
        global_config=GLOBAL,
        agents={
            "planner": planner(
                delegating({"agents": ["researcher"]}, {"agent": "researcher"}),
            ),
            "researcher": {
                "model": "r-model",
                "spend_ceiling": ceiling,
                "modes": {"research": {"max_steps": 4, "tools": ["again"]}},
            },
        },
        reply=priced(cost),
    )
    world.delegate = "research"
    world.offer_subagents("planner", "plan")
    world.add_tool("researcher", "again")
    return world


def test_a_per_agent_ceiling_stops_only_that_agent() -> None:
    # The researcher's first call costs 0.05 against a 0.04 ceiling, so its
    # second is refused; the planner, which has no ceiling, carries on.
    def reply(world: World, request: LlmRequest) -> LlmResponse:
        if request.system_prompt != world.parent_prompt:
            return LlmResponse(
                text="more",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall(id=f"r{len(world.requests)}", name="again")],
                cost_usd=0.05,
            )
        return priced(0.05)(world, request)

    world = researcher_world(0.05, 0.04)
    world._reply = reply

    response = world.send(world.session("planner", "plan"))

    assert response.text == "answered"
    assert response.stopped_because is None
    assert len(world.child_requests()) == 1
    assert len(world.parent_requests()) == 2
    assert "spend ceiling" in str(world.parent_requests()[1].messages)
    # One ledger for the tree: the researcher's spend sits beside the planner's.
    assert world.context.ledger.cost_usd == pytest.approx(0.15)
    assert world.context.ledger.agent_cost_usd("researcher") == pytest.approx(0.05)


def test_a_ceilinged_agent_running_alone_is_refused_with_the_code() -> None:
    world = World(
        global_config=GLOBAL,
        agents={
            "researcher": {
                "model": "r-model",
                "spend_ceiling": 0.04,
                "modes": {"research": {"tools": ["again"]}},
            }
        },
        reply=lambda world, request: LlmResponse(
            text="more",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(id=f"c{len(world.requests)}", name="again")],
            cost_usd=0.05,
        ),
    )
    world.add_tool("researcher", "again")

    with pytest.raises(HarnessError) as refused:
        world.send(world.session("researcher", "research"))

    assert refused.value.code == "agent_spend_ceiling_exceeded"
    assert len(world.requests) == 1


def test_an_unpriced_provider_under_a_ceiling_is_refused() -> None:
    world = World(
        global_config=GLOBAL,
        agents={
            "researcher": {
                "model": "r-model",
                "spend_ceiling": 5.0,
                "modes": {"research": {"tools": ["again"]}},
            }
        },
        reply=lambda world, request: LlmResponse(
            text="more",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(id=f"c{len(world.requests)}", name="again")],
            cost_usd=None,
        ),
    )
    world.add_tool("researcher", "again")

    with pytest.raises(HarnessError) as refused:
        world.send(world.session("researcher", "research"))

    assert refused.value.code == "budget_unpriced"
    assert len(world.requests) == 1


def test_events_carry_harness_agent_for_a_named_agent() -> None:
    world = World(global_config=GLOBAL, agents={"planner": planner({})})

    world.send(world.session("planner", "plan"))

    assert [event.type for event in world.events] == ["run.started", "run.finished"]
    assert [event.harness_agent for event in world.events] == ["planner", "planner"]
    assert all(HarnessEvents.to_dict(e)["harness_agent"] == "planner" for e in world.events)


def test_a_refused_run_carries_harness_agent_on_its_failure() -> None:
    world = World(
        global_config=GLOBAL,
        agents={
            "researcher": {
                "model": "r-model",
                "spend_ceiling": 5.0,
                "modes": {"research": {"tools": ["again"]}},
            }
        },
        reply=lambda world, request: LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(id="c", name="again")],
        ),
    )
    world.add_tool("researcher", "again")

    with pytest.raises(HarnessError):
        world.send(world.session("researcher", "research"))

    assert [(e.type, e.harness_agent) for e in world.events] == [
        ("run.started", "researcher"),
        ("run.failed", "researcher"),
    ]


def test_legacy_events_keep_their_shape() -> None:
    world = World(global_config=GLOBAL, modes={"chat": {}})

    world.send(world.session(None, "chat"))

    assert all("harness_agent" not in HarnessEvents.to_dict(e) for e in world.events)
