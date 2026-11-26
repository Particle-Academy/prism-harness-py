"""A subagent child keeps its saved binding on turns the runner does not start.

The provider and model binding the runner gave a child is kept with the child
session's durable data, as prism-harness keeps it, not only on the row a paused
run stopped on. A new message, an answer, and a turn under a context the host
built all run on it. A child session with no saved binding is refused
(``child_binding_missing``) rather than run on the root rules, and a context
the host supplies never replaces the binding or run tree the child saved.

Driven through ``runtime.send``; the assertions read the provider and model off
the request the model client was actually sent. The network is refused and
every store is in memory.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from prism_harness import (
    AgentRuntime,
    ChildBinding,
    Durability,
    HarnessError,
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
    RunStarted,
    Session,
    SubagentRunner,
    ToolRegistry,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


class DurableMemoryStore(MemorySessionStore):
    """A memory store that claims durability. For tests only."""

    def durability(self) -> Durability:
        return Durability.DURABLE


class ExpiringStore(MemorySessionStore):
    """An ephemeral store a test can empty, as an expired TTL would."""

    keys: ClassVar[set[str]] = set()

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        ExpiringStore.keys.add(key)
        super().put(key, payload, ttl_seconds)

    def expire_all(self) -> None:
        for key in list(ExpiringStore.keys):
            self.forget(key)


class Weather:
    name = "weather"

    def handle(self, args: dict[str, Any]) -> Any:
        return "Sunny"


CHEAP = ("custom-cheap", "cheap-model")


class Rig:
    def __init__(self, chat_steps: int = 20, nested: bool = False) -> None:
        self.events = HarnessEvents()
        self.depths: list[int] = []
        self.events.listen(
            lambda event: self.depths.append(event.depth) if isinstance(event, RunStarted) else None
        )
        self.queue: list[LlmResponse] = []
        self.requests: list[LlmRequest] = []
        self._call = 0
        ExpiringStore.keys = set()
        self.harness = PrismHarness(
            drivers={"memory": ExpiringStore, "durable": DurableMemoryStore},
            stores={"ephemeral": "memory", "durable": "durable"},
        )
        modes: dict[str, Any] = {
            "chat": {
                "system_prompt": "You delegate.",
                "tools": ["research"],
                "max_steps": chat_steps,
                "subagents": {
                    "research": {
                        "mode": "research",
                        "provider": CHEAP[0],
                        "model": CHEAP[1],
                    }
                },
            },
            "research": {
                "system_prompt": "Research.",
                "tools": ["weather", "dig"] if nested else ["weather"],
                "requires_approval": ["weather"],
                "max_steps": 8,
                **({"subagents": {"dig": {"mode": "dig"}}} if nested else {}),
            },
            "dig": {"system_prompt": "Dig.", "tools": [], "max_steps": 4},
        }
        tools = ToolRegistry().register(Weather())

        def research(session: Session) -> Any:
            subagent = self.runtime.mode_for(session).subagents["research"]
            return SubagentRunner(self.harness, self.runtime).tool(
                subagent, session, RunContext.root("parent-run", RunBudget(100)), "parent-run"
            )

        tools.register_factory("research", research)

        def dig(session: Session) -> Any:
            subagent = self.runtime.mode_for(session).subagents["dig"]
            return SubagentRunner(self.harness, self.runtime).tool(
                subagent, session, RunContext.root("dig-run", RunBudget(100)), "dig-run"
            )

        if nested:
            tools.register_factory("dig", dig)
        self.runtime = AgentRuntime(
            client=self._client,
            modes=ModeRegistry({"default": "chat", "modes": modes}),
            tools=tools,
            config={"provider": "openai", "model": "root-model"},
            events=self.events,
        )

    def _client(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if not self.queue:
            raise AssertionError("The scripted model has no response left.")
        return self.queue.pop(0)

    def session(self, scope: str = "chat") -> Session:
        return self.harness.for_(Participant("User", "ada")).session(scope)

    def calls(self, *calls: tuple[str, dict[str, Any]]) -> None:
        tool_calls = []
        for name, arguments in calls:
            self._call += 1
            tool_calls.append(LlmToolCall(id=f"call_{self._call}", name=name, arguments=arguments))
        self.queue.append(LlmResponse(text="", finish_reason="tool_calls", tool_calls=tool_calls))

    def text(self, text: str = "Done.") -> None:
        self.queue.append(LlmResponse(text=text, finish_reason="stop"))

    def sent(self, start: int) -> list[tuple[str, str]]:
        return [(request.provider, request.model) for request in self.requests[start:]]


def started_child(pause: bool) -> tuple[Rig, Session]:
    """The parent delegates; the child answers in text, or stops on an approval."""
    rig = Rig()
    rig.calls(("research", {"task": "Find the weather."}))
    if pause:
        rig.calls(("weather", {"city": "Paris"}))
    else:
        rig.text("Child answer.")
    rig.text("Parent answer.")
    rig.runtime.send(rig.session(), "Go")
    assert rig.sent(1)[0] == CHEAP
    return rig, rig.session("chat::sub::research")


def test_a_new_message_to_a_finished_child_runs_on_its_binding() -> None:
    rig, child = started_child(False)
    before = len(rig.requests)

    rig.text("Again.")
    rig.runtime.send(child, "New question")

    assert rig.sent(before) == [CHEAP]


def test_a_new_message_to_a_paused_child_runs_on_its_binding() -> None:
    rig, child = started_child(True)
    before = len(rig.requests)

    rig.text("Again.")
    rig.runtime.send(child, "New question")

    assert rig.sent(before) == [CHEAP]


def test_the_binding_outlives_the_ephemeral_half() -> None:
    rig, child = started_child(False)
    before = len(rig.requests)
    store = rig.harness.ephemeral_store()
    assert isinstance(store, ExpiringStore)
    store.expire_all()

    rig.text("Again.")
    rig.runtime.send(child, "New question")

    assert rig.sent(before) == [CHEAP]


def test_a_host_built_context_runs_on_the_child_binding() -> None:
    rig, child = started_child(False)
    before = len(rig.requests)

    rig.text("Again.")
    rig.runtime.send(child, "New question", context=RunContext.root("host", RunBudget(100)))

    assert rig.sent(before) == [CHEAP]


def test_a_host_context_carrying_a_binding_never_replaces_the_saved_one() -> None:
    rig, child = started_child(False)
    refs = child.child_refs()
    assert refs is not None and refs["tree"] is not None
    before = len(rig.requests)
    forged = RunContext.root("forged", RunBudget(100)).with_binding(
        ChildBinding(declared_provider="forged-provider", declared_model="forged-model")
    )

    rig.text("Forged turn.")
    try:
        rig.runtime.send(child, "Forged question", context=forged)
    except HarnessError:
        pass
    rig.text("Again.")
    rig.runtime.send(child, "New question")

    assert all(seen == CHEAP for seen in rig.sent(before))
    assert child.child_refs() == refs


def test_a_child_with_no_saved_binding_is_refused_before_any_request() -> None:
    rig = Rig()
    child = rig.session("chat::sub::research")
    # A child written before the binding was kept on the session: its run
    # tree reference is saved, its binding is not.
    child.remember_child_refs({"session": "some-root", "run": "some-run"}, None)

    rig.text("Should not run.")
    with pytest.raises(HarnessError) as caught:
        rig.runtime.send(child, "New question")

    assert caught.value.code == "child_binding_missing"
    assert rig.requests == []


# The runner starts a child under a run tree the HOST built: the parent's turn
# was handed a RunContext the host made. The child's context is still the
# runner's own (it derived and bound it), so its binding is saved with the
# child session exactly as under a tree this runtime opened. Without that, the
# child's next turn would run on the default provider. The PHP reference does
# this (RunTrees::startingChild); the parity row is harness-plans hpl-0059.


def host_tree_child(pause: bool) -> tuple[Rig, Session]:
    rig = Rig()
    rig.calls(("research", {"task": "Find the weather."}))
    if pause:
        rig.calls(("weather", {"city": "Paris"}))
    else:
        rig.text("Child answer.")
    rig.text("Parent answer.")
    rig.runtime.send(rig.session(), "Go", context=RunContext.root("host_root", RunBudget(100)))
    assert rig.sent(1)[0] == CHEAP
    return rig, rig.session("chat::sub::research")


def test_a_child_started_under_a_host_tree_saves_its_binding() -> None:
    rig, child = host_tree_child(False)
    assert child.child_binding() is not None
    before = len(rig.requests)

    rig.text("Again.")
    rig.runtime.send(child, "New question")

    assert rig.sent(before) == [CHEAP]


def test_a_child_paused_under_a_host_tree_resumes_on_its_binding() -> None:
    rig, child = host_tree_child(True)
    before = len(rig.requests)

    rig.text("Again.")
    rig.runtime.send(child, "New question")

    assert rig.sent(before) == [CHEAP]


def test_a_host_tree_child_keeps_its_binding_after_the_ephemeral_half() -> None:
    rig, child = host_tree_child(False)
    before = len(rig.requests)
    store = rig.harness.ephemeral_store()
    assert isinstance(store, ExpiringStore)
    store.expire_all()

    rig.text("Again.")
    rig.runtime.send(child, "New question")

    assert rig.sent(before) == [CHEAP]


def test_a_binding_the_host_put_on_its_own_context_is_never_saved() -> None:
    rig = Rig()
    session = rig.session("plain")
    hosted = RunContext.root("host_root", RunBudget(100)).with_binding(
        ChildBinding(declared_provider=CHEAP[0], declared_model=CHEAP[1])
    )

    rig.text("Hosted.")
    rig.runtime.send(session, "Go", context=hosted)
    rig.text("Plain.")
    rig.runtime.send(session, "Again")

    assert rig.sent(0) == [CHEAP, ("openai", "root-model")]
    assert session.child_binding() is None


# A child the runner starts again, now under a run tree the host built, keeps
# the run tree reference it saved under an earlier tree: the runner's own
# context spends from the host's tree and is never held against the saved one,
# and the saved one is never replaced with nothing, so a later turn the runner
# does not start still continues it rather than a fresh root budget. A child's
# depth is saved with it, so a later turn on the child cannot reset its nesting.


def test_a_child_started_again_under_a_host_tree_keeps_its_saved_tree() -> None:
    rig = Rig(chat_steps=3)
    rig.calls(("research", {"task": "Find the weather."}))
    # The child answers: a child a person's approval is pending on is never
    # delegated into again (G-387).
    rig.text("Child answer.")
    rig.text("Parent answer.")
    rig.runtime.send(rig.session(), "Go")
    child = rig.session("chat::sub::research")
    refs = child.child_refs()
    assert refs is not None and refs["tree"] is not None
    before = len(rig.requests)

    rig.calls(("research", {"task": "Again."}))
    rig.text("Child again.")
    rig.text("Parent again.")
    rig.runtime.send(rig.session(), "Again", context=RunContext.root("host_root", RunBudget(100)))

    assert CHEAP in rig.sent(before)
    assert child.child_refs() == refs


def test_a_child_keeps_the_depth_the_runner_started_it_at() -> None:
    rig, child = started_child(False)
    assert child.child_depth() == 1
    start = len(rig.depths)

    rig.text("Again.")
    rig.runtime.send(child, "New question")
    rig.text("Hosted.")
    rig.runtime.send(child, "Hosted question", context=RunContext.root("host_root", RunBudget(100)))

    assert rig.depths[start:] == [1, 1]


def test_a_child_started_again_runs_at_the_depth_the_runner_gives_it() -> None:
    rig = Rig()
    child = rig.session("chat::sub::research")
    child.remember_child_binding(
        ChildBinding(declared_provider=CHEAP[0], declared_model=CHEAP[1]).to_row(), 3
    )

    rig.calls(("research", {"task": "Find the weather."}))
    rig.text("Child answer.")
    rig.text("Parent answer.")
    rig.runtime.send(rig.session(), "Go")

    assert rig.depths == [0, 1]
    assert child.child_depth() == 1


# A child resumed under a run context the host built continues the run tree it
# saved, and a grandchild it starts belongs to that tree too: the grandchild
# saves the same tree, so its own later turns continue it.
def test_a_grandchild_of_a_child_resumed_under_a_host_context_saves_the_tree() -> None:
    rig = Rig(nested=True)
    rig.calls(("research", {"task": "Find the weather."}))
    rig.calls(("weather", {"city": "Paris"}))
    rig.text("Parent answer.")
    rig.runtime.send(rig.session(), "Go")
    child = rig.session("chat::sub::research")
    refs = child.child_refs()
    assert refs is not None and refs["tree"] is not None

    rig.calls(("dig", {"task": "Dig."}))
    rig.text("Grandchild answer.")
    rig.text("Child answer.")
    rig.runtime.send(child, "Dig deeper", context=RunContext.root("host_answer", RunBudget(100)))

    grandchild = rig.session("chat::sub::research::sub::dig")
    grand = grandchild.child_refs()
    assert grand is not None and grand["tree"] == refs["tree"]
