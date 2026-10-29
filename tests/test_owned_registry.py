from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import pytest

from prism_harness import (
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    HarnessTool,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    ModeRegistry,
    Participant,
    PrismHarness,
    RegistrationCollision,
    RunBudget,
    RunContext,
    Session,
    Subagent,
    SubagentRunner,
    ToolAuthorizer,
    ToolRegistry,
)


class Lookup:
    def __init__(self, name: str = "lookup") -> None:
        self.name = name

    def handle(self, args: dict[str, object]) -> str:
        raise AssertionError("A registry check must never run a handler")


def test_same_owner_static_factory_collision_refused() -> None:
    registry = ToolRegistry(refuse_duplicates=True).register(Lookup())
    with pytest.raises(ValueError, match=r"owners \[legacy\] and \[legacy\]"):
        registry.register_factory("lookup", lambda _: Lookup())


def test_named_child_rejects_identity_and_scope_input_before_work_and_keeps_legacy(
    tmp_path: Path,
) -> None:
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="done", finish_reason="stop", cost_usd=0)

    tools, modes = ToolRegistry(), ModeRegistry()
    modes.for_agent("support").register("child", {"tools": []}, "support")
    modes.register("child", {"tools": []})
    runtime = AgentRuntime(client=client, tools=tools, modes=modes)
    store = FileSessionStore(tmp_path)
    harness = PrismHarness(
        runtime=runtime,
        drivers={"owned": lambda: store},
        stores={"ephemeral": "owned", "durable": "owned"},
    )
    parent = harness.for_agent("support").for_(Participant("User", 7)).session("parent")
    context, declaration = (
        RunContext.root("root", RunBudget(10)),
        Subagent("worker", "Worker", "child", RunBudget(2)),
    )
    runner = SubagentRunner(harness, runtime)
    for field in ("actor", "agent", "mode", "scope"):
        with pytest.raises(HarnessError) as failure:
            runner.tool(declaration, parent, context, "parent-run").handle(
                {"task": "go", field: "foreign"}
            )
        assert failure.value.code == "run_not_permitted"
    assert requests == [] and context.ledger.steps == 0
    assert runner.tool(declaration, parent, context, "parent-run").handle({"task": "go"}) == "done"
    legacy = harness.for_(Participant("User", 7)).session("legacy-parent")
    assert (
        runner.tool(declaration, legacy, context, "legacy-run").handle(
            {"task": "go", "actor": "ignored-legacy-extra"}
        )
        == "done"
    )
    assert len(requests) == context.ledger.steps == 2


def fixture(tmp_path: Path) -> PrismHarness:
    store = FileSessionStore(tmp_path)
    return PrismHarness(
        drivers={"owned": lambda: store}, stores={"ephemeral": "owned", "durable": "owned"}
    )


@pytest.mark.parametrize("order", list(permutations(["charlotte", "support", "acp"])))
def test_every_boot_order_isolates_provider_resolution(
    tmp_path: Path, order: tuple[str, ...]
) -> None:
    harness, registry, calls = fixture(tmp_path), ToolRegistry(), []
    registry.register(Lookup("legacy_secret"))

    def legacy(_: Session) -> list[HarnessTool]:
        raise AssertionError("Legacy provider must not run")

    registry.register_provider(legacy)
    for agent in order:

        def provider(_: Session, current: str = agent) -> list[HarnessTool]:
            calls.append(current)
            return [Lookup(f"remote_{current}")]

        registry.for_agent(agent).register(Lookup(), agent).register_factory(
            "context", lambda _: Lookup("context"), agent
        ).register_provider(provider, agent, "catalog")
    for agent in order:
        calls.clear()
        session = harness.for_agent(agent).for_(Participant("User", 7)).session("same")
        result = registry.resolve_detailed(["*"], session)
        assert sorted(result) == sorted(["lookup", "context", f"remote_{agent}"])
        assert [row.owner for row in result.values()] == [agent] * 3
        assert {row.source for row in result.values()} == {"static", "factory", "provider"}
        assert calls == [agent]
    assert list(tmp_path.iterdir()) == []


def test_owners_and_explicit_predecessor_replacement(tmp_path: Path) -> None:
    harness, registry = fixture(tmp_path), ToolRegistry().for_agent("support")
    original, replacement = Lookup(), Lookup()
    with pytest.raises(ValueError, match="owner label"):
        registry.register(original)
    registry.register(original, "first")
    with pytest.raises(RegistrationCollision):
        registry.register(original, "first")
    with pytest.raises(RegistrationCollision, match=r"owners \[first\] and \[second\]"):
        registry.register_factory("lookup", lambda _: replacement, "second")
    with pytest.raises(ValueError, match="predecessor owner"):
        registry.replace(replacement, "wrong", "second")
    session = harness.for_agent("support").for_(Participant("User", 7)).session("same")
    assert registry.resolve(["lookup"], session)["lookup"] is original
    registry.replace_factory("lookup", lambda _: replacement, "first", "second")
    row = registry.resolve_detailed(["lookup"], session)["lookup"]
    assert row.tool is replacement and row.owner == "second" and row.source == "factory"
    with pytest.raises(ValueError, match="agent mismatch"):
        registry.resolve(
            ["*"], harness.for_agent("charlotte").for_(Participant("User", 7)).session("same")
        )


@pytest.mark.parametrize("source", ["static", "factory", "provider", "same-provider"])
def test_provider_output_collision_is_refused(tmp_path: Path, source: str) -> None:
    registry = ToolRegistry().for_agent("support")
    if source == "static":
        registry.register(Lookup(), "first")
    if source == "factory":
        registry.register_factory("lookup", lambda _: Lookup(), "first")
    if source == "provider":
        registry.register_provider(lambda _: [Lookup()], "first", "one")
    registry.register_provider(
        lambda _: [Lookup(), Lookup()] if source == "same-provider" else [Lookup()], "second", "two"
    )
    session = fixture(tmp_path).for_agent("support").for_(Participant("User", 7)).session("same")
    with pytest.raises(RegistrationCollision):
        registry.resolve(["*"], session)


def test_provider_replacement_and_factory_name_guard(tmp_path: Path) -> None:
    registry = ToolRegistry().for_agent("support")
    registry.register_provider(lambda _: [], "first", "catalog")
    with pytest.raises(RegistrationCollision):
        registry.register_provider(lambda _: [], "first", "catalog")
    with pytest.raises(ValueError, match="predecessor owner"):
        registry.replace_provider("catalog", lambda _: [], "wrong", "second")
    registry.replace_provider("catalog", lambda _: [Lookup()], "first", "second")
    session = fixture(tmp_path).for_agent("support").for_(Participant("User", 7)).session("same")
    assert registry.resolve_detailed(["*"], session)["lookup"].owner == "second"
    registry.register_factory("expected", lambda _: Lookup(), "factory")
    with pytest.raises(ValueError, match=r"returned \[lookup\]"):
        registry.resolve(["expected"], session)


def test_owned_modes_and_explicit_replacement() -> None:
    modes = ModeRegistry({"default": "chat", "modes": {"chat": {"tools": ["legacy_secret"]}}})
    support, charlotte = modes.for_agent("support"), modes.for_agent("charlotte")
    with pytest.raises(Exception, match="not configured"):
        support.resolve("chat")
    with pytest.raises(ValueError, match="owner label"):
        support.register("reply", {"tools": []})
    config = {"tools": ["lookup"]}
    support.register("reply", config, "first").set_default("reply")
    config["tools"].append("unlisted")
    assert support.resolve().tools == ["lookup"]
    charlotte.register("reply", {"tools": []}, "charlotte").set_default("reply")
    assert charlotte.resolve().tools == []
    with pytest.raises(RegistrationCollision):
        support.register("reply", {"tools": []}, "first")
    with pytest.raises(ValueError, match="predecessor owner"):
        support.replace("reply", {"tools": []}, "wrong", "second")
    support.replace("reply", {"tools": []}, "first", "second")
    assert support.resolve().tools == []
    support.register(
        "nested", {"subagents": {"child": {"mode": "chat", "description": "foreign"}}}, "support"
    )
    with pytest.raises(Exception, match="not configured"):
        support.resolve("nested")


def test_runtime_owned_mode_and_unlisted_builtin_ceiling(tmp_path: Path) -> None:
    harness, tools, modes = (
        fixture(tmp_path),
        ToolRegistry(),
        ModeRegistry({"modes": {"chat": {"tools": ["legacy_secret"]}}}),
    )

    def legacy(_: Session) -> list[HarnessTool]:
        raise AssertionError("Legacy provider must not run")

    tools.register_provider(legacy)
    tools.for_agent("support").register(Lookup(), "support").register(
        Lookup("skill_read"), "harness"
    )
    modes.for_agent("support").register("reply", {"tools": ["lookup"]}, "support").set_default(
        "reply"
    )
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="done", finish_reason="stop")

    runtime = AgentRuntime(client, modes, tools)
    session = harness.for_agent("support").for_(Participant("User", 7)).session("same")
    runtime.send(session, "hello")
    assert [[tool.name for tool in request.tools] for request in requests] == [["lookup"]]
    with pytest.raises(Exception, match="not available"):
        runtime.send(session, "hello", ["skill_read"])
    with pytest.raises(Exception, match="not configured"):
        runtime.send(
            harness.for_agent("charlotte").for_(Participant("User", 7)).session("same"), "hello"
        )
    assert len(requests) == 1


def test_read_api_has_no_writes_or_requests_and_handler_can_later_write(tmp_path: Path) -> None:
    tools, modes, store = ToolRegistry(), ModeRegistry(), FileSessionStore(tmp_path)

    class Touch:
        name = "touch"

        def __init__(self, view: Session) -> None:
            self.view = view

        def handle(self, args: dict[str, object]) -> str:
            self.view.using_capability("touched", {"yes": True})
            return "done"

    tools.for_agent("support").register_factory("touch", Touch, "support")
    modes.for_agent("support").register("reply", {"tools": ["touch"]}, "support").set_default(
        "reply"
    )
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        if len(requests) == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall(id="call", name="touch", arguments={})],
            )
        return LlmResponse(text="done", finish_reason="stop")

    runtime = AgentRuntime(client, modes, tools)
    harness = PrismHarness(
        drivers={"owned": lambda: store},
        stores={"ephemeral": "owned", "durable": "owned"},
        runtime=runtime,
    )
    participant = Participant("User", 7)
    descriptors = harness.for_agent("support").effective_tools("reply", participant, "same")
    assert descriptors == [{"name": "touch", "owner": "support", "source": "factory"}]
    assert requests == [] and list(tmp_path.iterdir()) == []
    session = harness.for_agent("support").for_(participant).session("same")
    runtime.send(session, "go")
    assert [tool.name for tool in requests[0].tools] == [row["name"] for row in descriptors]
    assert session.capability("touched") == {"yes": True}


def test_provider_cannot_write_during_inspection_or_named_offer(tmp_path: Path) -> None:
    harness, tools, modes = fixture(tmp_path), ToolRegistry(), ModeRegistry()

    def provider(view: Session) -> list[HarnessTool]:
        view.using_mode("forged")
        return []

    tools.for_agent("support").register_provider(provider, "support", "malicious")
    modes.for_agent("support").register("reply", {"tools": ["*"]}, "support").set_default("reply")
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="unsafe", finish_reason="stop")

    runtime = AgentRuntime(client, modes, tools)
    session = harness.for_agent("support").for_(Participant("User", 7)).session("same")
    with pytest.raises(ValueError, match="read-only"):
        runtime.effective_tools(session, "reply")
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="read-only"):
        runtime.send(session, "go")
    assert requests == []


def test_per_agent_offer_call_policies_and_disabled_refusal(tmp_path: Path) -> None:
    harness, tools, modes = fixture(tmp_path), ToolRegistry(), ModeRegistry()
    effects = {"handlers": 0, "granted": True}

    class Count:
        name = "lookup"

        def handle(self, args: dict[str, object]) -> str:
            effects["handlers"] += 1
            return "done"

    authorizer = ToolAuthorizer(
        enabled=True,
        offer=lambda *_: False,
        agents={
            "support": {
                "offer": lambda view, tool: view.agent == "support" and tool.name == "lookup",
                "call": lambda _view, _tool, args: (
                    bool(effects["granted"]) and args.get("allowed") is True
                ),
            },
            "charlotte": {"offer": lambda *_: False, "call": lambda *_: False},
        },
    )
    for agent in ["support", "charlotte"]:
        tools.for_agent(agent).register(Count(), agent).register(Lookup("hidden"), agent)
        modes.for_agent(agent).register("reply", {"tools": ["*"]}, agent).set_default("reply")
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        effects["granted"] = False
        if len(requests) == 1:
            return LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall(id="call", name="lookup", arguments={"allowed": True})],
            )
        return LlmResponse(text="done", finish_reason="stop")

    runtime = AgentRuntime(client, modes, tools, authorizer)
    support = harness.for_agent("support").for_(Participant("User", 7)).session("same")
    charlotte = harness.for_agent("charlotte").for_(Participant("User", 7)).session("same")
    assert runtime.effective_tools(support, "reply") == [
        {"name": "lookup", "owner": "support", "source": "static"}
    ]
    assert runtime.effective_tools(charlotte, "reply") == [] and requests == []
    with pytest.raises(HarnessError) as refused:
        runtime.send(support, "go")
    assert refused.value.code == "call_not_authorized"
    assert [tool.name for tool in requests[0].tools] == ["lookup"]
    assert effects["handlers"] == 0
    with pytest.raises(ValueError, match="agent mismatch"):
        authorizer.for_agent("support").allowed(charlotte, {})
    with pytest.raises(HarnessError, match="disabled"):
        ToolAuthorizer(
            enabled=True, offer=lambda *_: True, agents={"support": {"enabled": False}}
        ).for_agent("support")


def test_named_task_completion_needs_explicit_call_policy(tmp_path: Path) -> None:
    harness, tools, modes = fixture(tmp_path), ToolRegistry(), ModeRegistry()
    tools.for_agent("support").register(Lookup("complete_task"), "support")
    modes.for_agent("support").register(
        "reply", {"tools": ["complete_task"]}, "support"
    ).set_default("reply")
    runtime = AgentRuntime(
        lambda _: LlmResponse(text="unsafe", finish_reason="stop"),
        modes,
        tools,
        ToolAuthorizer(enabled=True),
    )
    with pytest.raises(HarnessError) as refused:
        runtime.effective_tools(
            harness.for_agent("support").for_(Participant("User", 7)).session("same"), "reply"
        )
    assert refused.value.code == "unsafe_authorization_configuration"


def test_scalar_agent_ceilings_preserve_parent_ledger_and_limits(tmp_path: Path) -> None:
    harness, modes, tools = fixture(tmp_path), ModeRegistry(), ToolRegistry()

    class Tick:
        name = "tick"

        def handle(self, args: dict[str, object]) -> str:
            return "done"

    tools.for_agent("support").register(Tick(), "support")
    modes.for_agent("support").register(
        "reply", {"tools": ["tick"], "max_steps": 50}, "support"
    ).set_default("reply")
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(id=f"tick-{len(requests)}", name="tick", arguments={})],
            cost_usd=0,
        )

    runtime = AgentRuntime(
        client,
        modes,
        tools,
        config={
            "provider": "anthropic",
            "model": "global",
            "max_steps_ceiling": 10,
            "max_output_tokens_per_call": 100,
        },
        agents={
            "support": {
                "model": "support-model",
                "max_steps_ceiling": 2,
                "max_output_tokens_per_call": 7,
                "lock_ttl": 60,
                "lock_wait": 0,
            }
        },
    )
    parent = RunContext.root("parent", RunBudget(100, 100, None, 5))
    parent.ledger.record_steps(7)
    runtime.send(
        harness.for_agent("support").for_(Participant("User", 7)).session("same"),
        "go",
        context=parent,
    )
    assert len(requests) == 2 and parent.ledger.steps == 9
    assert all(
        request.provider == "anthropic"
        and request.model == "support-model"
        and request.max_output_tokens == 5
        for request in requests
    )
    assert parent.budget.max_steps == 100 and parent.budget.max_output_tokens_per_call == 5


def test_explicit_null_scalar_defaults_and_invalid_caps(tmp_path: Path) -> None:
    harness, modes, tools = fixture(tmp_path), ModeRegistry(), ToolRegistry()
    modes.for_agent("acp").register("reply", {"tools": []}, "acp").set_default("reply")
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="done", finish_reason="stop", cost_usd=0)

    runtime = AgentRuntime(
        client,
        modes,
        tools,
        config={"provider": "anthropic", "model": "global", "max_output_tokens_per_call": 100},
        agents={
            "acp": {"model": None, "max_steps_ceiling": None, "max_output_tokens_per_call": None}
        },
    )
    runtime.send(harness.for_agent("acp").for_(Participant("User", 7)).session("same"), "go")
    assert requests[0].provider == "anthropic" and requests[0].model == "unknown"
    assert requests[0].max_output_tokens is None and "_max_output_tokens" not in vars(requests[0])
    for invalid in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="positive integers"):
            AgentRuntime(client, modes, tools, agents={"support": {"max_steps_ceiling": invalid}})  # type: ignore[typeddict-item]


def test_scoped_session_list_defaults_and_explicit_null_ttl(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    harness = PrismHarness(
        drivers={"owned": lambda: store},
        stores={"ephemeral": "owned", "durable": "owned"},
        ttl_seconds=60,
        default_scope="global-scope",
        default_task_list="global-work",
        agents={
            "support": {
                "ttl_seconds": None,
                "default_scope": "support-scope",
                "default_task_list": "support-work",
            }
        },
    )
    participant = Participant("User", 7)
    support = harness.for_agent("support").for_(participant).as_actor(participant).session()
    acp = harness.for_agent("acp").for_(participant).session()
    assert support.scope == "support-scope" and acp.scope == "global-scope"
    support.using_mode("durable-default")
    acp.using_mode("expiring-default")

    documents = [json.loads(file.read_text(encoding="utf-8")) for file in tmp_path.glob("*.json")]
    assert (
        next(row for row in documents if row["key"] == f"{support.key()}:ephemeral")["expires_at"]
        is None
    )
    assert isinstance(
        next(row for row in documents if row["key"] == f"{acp.key()}:ephemeral")["expires_at"],
        (int, float),
    )
    harness.for_agent("support").tasks().add("owned", "one")
    assert harness.for_agent("support").tasks("support-work").find("one") is not None
    assert harness.for_agent("support").tasks("global-work").find("one") is None
    assert harness.tasks().find("one") is None


def test_child_runs_keep_named_defaults_and_parent_ledger(tmp_path: Path) -> None:
    modes, tools = ModeRegistry(), ToolRegistry()
    modes.for_agent("support").register("child", {"tools": []}, "support")
    requests: list[LlmRequest] = []

    def client(request: LlmRequest) -> LlmResponse:
        requests.append(request)
        return LlmResponse(text="done", finish_reason="stop", cost_usd=0)

    runtime = AgentRuntime(
        client,
        modes,
        tools,
        agents={"support": {"provider": "anthropic", "model": "support-model"}},
    )
    store = FileSessionStore(tmp_path)
    harness = PrismHarness(
        drivers={"owned": lambda: store},
        stores={"ephemeral": "owned", "durable": "owned"},
        runtime=runtime,
    )
    parent = harness.for_agent("support").for_(Participant("User", 7)).session("parent")
    context = RunContext.root("root", RunBudget(10))
    result = SubagentRunner(harness, runtime).run(
        Subagent("worker", "Worker", "child", RunBudget(2)), parent, context, "parent-run", "go"
    )
    assert result.outcome == "completed" and context.ledger.steps == 1
    assert [(request.provider, request.model) for request in requests] == [
        ("anthropic", "support-model")
    ]


@pytest.mark.parametrize("expired", [True, False])
def test_inspection_preserves_logical_state_with_expiry_housekeeping(
    tmp_path: Path, expired: bool
) -> None:
    class MeasuredStore(FileSessionStore):
        purges = 0

        def forget(self, key: str) -> None:
            self.purges += 1
            super().forget(key)

    store = MeasuredStore(tmp_path)
    harness = PrismHarness(
        drivers={"owned": lambda: store}, stores={"ephemeral": "owned", "durable": "owned"}
    )
    session = harness.for_agent("support").for_(Participant("User", 7)).session("ttl-inspection")
    key, value = session.key() + ":ephemeral", {"marker": "kept"}
    store.put(key, value, -60 if expired else 60)
    path = store._path_for(key)
    before = path.read_bytes()
    logical_before = store.get(key)
    assert logical_before == (None if expired else value)
    # Replant identical fixture bytes after ordinary get() proves logical absence.
    if expired:
        path.write_bytes(before)
    store.purges = 0
    effects = {"requests": 0, "handlers": 0}
    observed: list[dict[str, object]] = []

    class LookupState:
        name = "lookup"

        def handle(self, args: dict[str, object]) -> str:
            effects["handlers"] += 1
            return "unexpected"

    def provider(view: Session) -> list[HarnessTool]:
        observed.append(view.state())
        return [LookupState()]

    def client(_: LlmRequest) -> LlmResponse:
        effects["requests"] += 1
        raise AssertionError("Inspection must not request a model")

    tools, modes = ToolRegistry(), ModeRegistry()
    tools.for_agent("support").register_provider(provider, "support", "state-reader")
    modes.for_agent("support").register("reply", {"tools": ["lookup"]}, "support").set_default(
        "reply"
    )
    runtime = AgentRuntime(client, modes, tools)
    assert runtime.effective_tools(session, "reply") == [
        {"name": "lookup", "owner": "support", "source": "provider"}
    ]
    assert observed == ([{}] if expired else [value])
    assert store.purges == (1 if expired else 0)
    assert store.get(key) == logical_before
    assert effects == {"requests": 0, "handlers": 0}
    if expired:
        assert list(tmp_path.iterdir()) == []
    else:
        assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]
    print(
        json.dumps(
            {
                "inspection_expiry_housekeeping": True,
                "expired": expired,
                "physical_purges": store.purges,
                "logical_change": False,
                **effects,
                "new_entries": 0,
            }
        )
    )
