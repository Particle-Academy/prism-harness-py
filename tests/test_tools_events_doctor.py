"""Mirrors prism-harness-ts/test/tools-events-doctor.test.ts."""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from prism_harness import (
    FileSessionStore,
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    HarnessTool,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PrismHarness,
    RunFailed,
    RunStarted,
    Session,
    SessionStoreManager,
    ToolAuthorizer,
    ToolRegistry,
    diagnose,
)


class FakeTool:
    def __init__(self, name: str, result: Any = "ok") -> None:
        self._name = name
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        return self._result


def a_session() -> Session:
    directory = tempfile.mkdtemp(prefix="prism-harness-tools-")
    harness = PrismHarness(
        drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
        stores={"ephemeral": "memory", "durable": "files"},
    )
    return harness.for_(Participant("User", 1)).session("support")


# -- ToolRegistry ------------------------------------------------------------


def test_resolves_named_tools() -> None:
    registry = ToolRegistry().register(FakeTool("search")).register(FakeTool("write"))

    assert list(registry.resolve(["search"])) == ["search"]


def test_star_resolves_everything_it_can_produce() -> None:
    registry = ToolRegistry().register_many([FakeTool("a"), FakeTool("b")])

    assert sorted(registry.resolve(["*"])) == ["a", "b"]


def test_refuses_a_name_it_cannot_produce_rather_than_dropping_it() -> None:
    # A mode that lists a tool it cannot get is a misconfiguration, and dropping
    # it quietly would leave a run wondering why the model never called it.
    registry = ToolRegistry().register(FakeTool("search"))

    with pytest.raises(HarnessError) as caught:
        registry.resolve(["ghost"])

    assert caught.value.code == "tool_not_available"


def test_builds_a_factory_tool_with_the_session() -> None:
    registry = ToolRegistry().register_factory(
        "scoped", lambda session: FakeTool(f"scoped:{session.scope}")
    )

    assert registry.resolve(["scoped"], a_session())["scoped"].name == "scoped:support"


def test_takes_tools_from_a_provider() -> None:
    registry = ToolRegistry().register_provider(lambda _session: [FakeTool("mcp:read")])

    assert list(registry.resolve(["*"], a_session())) == ["mcp:read"]


# -- ToolAuthorizer ----------------------------------------------------------


def test_the_authorizer_is_off_by_default_and_offers_everything() -> None:
    authorizer = ToolAuthorizer()
    tools = ToolRegistry().register_many([FakeTool("a"), FakeTool("b")]).resolve(["*"])

    assert authorizer.enabled is False
    assert len(authorizer.allowed(a_session(), tools)) == 2


def test_refuses_a_policy_that_would_never_be_consulted() -> None:
    # Both at once is the one configuration not to leave in place: a defined
    # policy that is never consulted looks like a control to every reader and is
    # not one.
    with pytest.raises(HarnessError, match="never consulted"):
        ToolAuthorizer(enabled=False, offer=lambda _s, _t: True)


def test_filters_what_a_run_is_offered() -> None:
    authorizer = ToolAuthorizer(enabled=True, offer=lambda _s, tool: tool.name != "danger")
    tools = ToolRegistry().register_many([FakeTool("safe"), FakeTool("danger")]).resolve(["*"])

    assert [tool.name for tool in authorizer.allowed(a_session(), tools)] == ["safe"]


def test_asks_again_at_call_time_with_the_arguments() -> None:
    # At the moment the toolset is assembled the arguments do not exist yet, so
    # an offer policy can say "may use delete_file" and never "only under /tmp".
    authorizer = ToolAuthorizer(
        enabled=True,
        call=lambda _s, _t, args: str(args.get("path", "")).startswith("/tmp/"),
    )
    tools = ToolRegistry().register(FakeTool("delete_file")).resolve(["*"])
    wrapped = authorizer.allowed(a_session(), tools)[0]

    assert wrapped.handle({"path": "/tmp/scratch"}) == "ok"

    with pytest.raises(HarnessError) as caught:
        wrapped.handle({"path": "/etc/passwd"})

    assert caught.value.code == "call_not_authorized"


def test_a_refusal_is_raised_not_returned() -> None:
    # A refusal handed back as a tool result reads to the model as a failure it
    # might retry differently, and a denied action being retried is the opposite
    # of what a guard is for.
    authorizer = ToolAuthorizer(enabled=True, call=lambda _s, _t, _a: False)
    tools = ToolRegistry().register(FakeTool("x")).resolve(["*"])
    wrapped = authorizer.allowed(a_session(), tools)[0]

    with pytest.raises(HarnessError):
        wrapped.handle({})


# -- events ------------------------------------------------------------------


def test_delivers_to_every_listener_and_can_unsubscribe() -> None:
    events = HarnessEvents()
    seen: list[HarnessEvent] = []
    stop = events.listen(seen.append)

    started = RunStarted(run_id="r1", session_key="k", mode="chat", provider="anthropic", model="m")

    events.emit(started)
    stop()
    events.emit(started)

    assert len(seen) == 1


def test_a_raising_listener_does_not_break_the_run() -> None:
    # Telemetry that takes down the thing it observes is worse than no
    # telemetry, and a listener is by definition somebody else's code.
    events = HarnessEvents()
    seen: list[str] = []

    def explode(_event: HarnessEvent) -> None:
        raise RuntimeError("listener exploded")

    events.listen(explode)
    events.listen(lambda event: seen.append(event.type))

    with pytest.warns(UserWarning, match="listener exploded"):
        events.emit(RunFailed(run_id="r1", session_key="k", failure="nope", steps=1))

    assert seen == ["run.failed"]


# -- the doctor --------------------------------------------------------------


def test_reports_a_consistent_configuration() -> None:
    directory = tempfile.mkdtemp(prefix="prism-harness-doctor-")
    report = diagnose(
        stores=SessionStoreManager(
            drivers={"memory": MemorySessionStore, "files": lambda: FileSessionStore(directory)},
            stores={"ephemeral": "memory", "durable": "files"},
        ),
        modes=ModeRegistry({"modes": {"chat": {"tools": ["search"]}}, "default": "chat"}),
        tools=ToolRegistry().register(FakeTool("search")),
        authorizer=ToolAuthorizer(),
    )

    assert report.ok is True
    assert "consistent" in report.summary()


def test_catches_a_volatile_durable_store() -> None:
    report = diagnose(
        stores=SessionStoreManager(
            drivers={"memory": MemorySessionStore},
            stores={"ephemeral": "memory", "durable": "memory"},
        )
    )

    assert report.ok is False
    assert "VOLATILE" in report.summary()


def test_catches_a_broken_mode_nobody_has_entered_yet() -> None:
    # The whole reason this exists. That mode keeps its broken subagent
    # reference until someone switches to it, and the first person to find out
    # is a user mid-conversation.
    report = diagnose(
        modes=ModeRegistry(
            {
                "default": "chat",
                "modes": {"chat": {}, "broken": {"subagents": {"helper": {"mode": "ghost"}}}},
            }
        )
    )

    assert report.ok is False
    assert next(f for f in report.findings if f.check == "mode:broken").ok is False
    # The default still resolves, so only the unentered mode is reported.
    assert next(f for f in report.findings if f.check == "mode:chat").ok is True


def test_catches_a_mode_naming_a_tool_the_registry_cannot_produce() -> None:
    report = diagnose(
        modes=ModeRegistry({"default": "chat", "modes": {"chat": {"tools": ["ghost"]}}}),
        tools=ToolRegistry().register(FakeTool("search")),
    )

    assert report.ok is False
    assert "ghost" in report.summary()


def test_says_plainly_when_the_authorizer_is_off() -> None:
    report = diagnose(authorizer=ToolAuthorizer())

    assert "DISABLED" in report.summary()
    # Not a failure -- off is the default and a legitimate choice.
    assert report.ok is True


def test_reports_having_no_modes_at_all_as_a_problem() -> None:
    assert diagnose(modes=ModeRegistry({})).ok is False


def test_the_tool_protocol_accepts_a_plain_object() -> None:
    # Structural, not an import: anything with a name and a handler satisfies
    # it, which is what keeps this package at zero dependencies.
    tool: HarnessTool = FakeTool("anything")

    assert tool.name == "anything"
