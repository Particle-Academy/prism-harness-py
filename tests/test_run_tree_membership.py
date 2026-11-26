"""Which run tree a run belongs to is decided once, when it begins (G-381).

Nothing used to record it: each consumer re-derived it from a signal of its
own (the context's root run id, a scan for a live run holding the same ledger,
the subagent tool's construction-time parent run id, the saved depth). A
context the HOST builds is the host's statement, not a fact this runtime
holds, so it must never put a run into another session's tree. The runner's
own context joins the tree of the turn that is actually calling it, proven by
that turn's session key and ledger identity.

Driven through ``send`` and ``decide`` with a scripted model, as the other
run-tree tests are; the network is refused and every store is in memory.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any, ClassVar

import pytest
from test_plan_budget import DurableMemoryStore

from prism_harness import (
    MAX_DEPTH,
    AgentRuntime,
    ChildBinding,
    HarnessError,
    HarnessEvents,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    Plans,
    PrismHarness,
    RunBudget,
    RunContext,
    RunLedger,
    RunStarted,
    Session,
    SubagentRunner,
    ToolRegistry,
)
from prism_harness.plans import budget
from prism_harness.run_trees import RunTrees, membership
from prism_harness.runtime import current_turn_run
from prism_harness.thread_rows import pending_approval_ids


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


CHEAP = ChildBinding(declared_provider="custom-cheap", declared_model="cheap-model")


class Tool:
    def __init__(self, name: str, rig: Rig) -> None:
        self._name = name
        self._rig = rig

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        self._rig.ran.append(self._name)
        hook = self._rig.hooks.get(self._name)
        if hook is not None:
            hook()
        return "Done"


class Rig:
    """chat delegates to research, research to dig. ``relay`` and ``forge`` run
    whatever host code a test hooks onto them, inside the turn that calls them."""

    def __init__(
        self,
        plans: dict[str, Any] | None = None,
        chat_steps: int = 20,
        durable: type[MemorySessionStore] = DurableMemoryStore,
    ) -> None:
        self.queue: list[LlmResponse] = []
        self.requests: list[LlmRequest] = []
        self.ran: list[str] = []
        self.hooks: dict[str, Callable[[], None]] = {}
        self.depths: list[tuple[str, int]] = []
        #: Host code run once, at the start of the next model call.
        self.on_request: Callable[[], None] | None = None
        self._call = 0
        self.harness = PrismHarness(
            drivers={"memory": MemorySessionStore, "durable": durable},
            stores={"ephemeral": "memory", "durable": "durable"},
        )
        events = HarnessEvents()
        events.listen(
            lambda event: (
                self.depths.append((event.session_key, event.depth))
                if isinstance(event, RunStarted)
                else None
            )
        )
        modes: dict[str, Any] = {
            "chat": {
                "system_prompt": "You delegate.",
                "tools": ["research", "survey", "relay", "forge", "lookup", "weather"],
                "requires_approval": ["weather"],
                "max_steps": chat_steps,
                "plans": plans is not None,
                "subagents": {"research": {"mode": "research"}, "survey": {"mode": "research"}},
            },
            "research": {
                "system_prompt": "Research.",
                "tools": ["weather", "dig", "relay"],
                "requires_approval": ["weather"],
                "max_steps": 8,
                "subagents": {"dig": {"mode": "dig"}},
            },
            "dig": {
                "system_prompt": "Dig.",
                "tools": ["weather"],
                "requires_approval": ["weather"],
                "max_steps": 4,
            },
        }
        tools = ToolRegistry()
        for name in ("weather", "lookup", "relay", "forge"):
            tools.register(Tool(name, self))

        def delegate(name: str) -> Callable[[Session], Any]:
            def factory(session: Session) -> Any:
                subagent = self.runtime.mode_for(session).subagents[name]
                # The construction-time run id is the host's, never the turn's.
                return SubagentRunner(self.harness, self.runtime).tool(
                    subagent, session, RunContext.root("tool-run", RunBudget(100)), "tool-run"
                )

            return factory

        tools.register_factory("research", delegate("research"))
        tools.register_factory("dig", delegate("dig"))
        tools.register_factory("survey", delegate("survey"))
        self.runtime = AgentRuntime(
            client=self._client,
            modes=ModeRegistry({"default": "chat", "modes": modes}),
            tools=tools,
            events=events,
            config={"provider": "openai", "model": "root-model"},
            plans=None
            if plans is None
            else Plans({"enabled": True, **plans}, index_store=self.harness.durable_store()),
        )

    def _client(self, request: LlmRequest) -> LlmResponse:
        hook, self.on_request = self.on_request, None
        if hook is not None:
            hook()
        self.requests.append(request)
        if not self.queue:
            raise AssertionError("The scripted model has no response left.")
        return self.queue.pop(0)

    def session(self, who: str, scope: str = "chat") -> Session:
        return self.harness.for_(Participant("User", who)).session(scope)

    def calls(self, *calls: tuple[str, dict[str, Any]]) -> None:
        tool_calls = []
        for name, arguments in calls:
            self._call += 1
            tool_calls.append(LlmToolCall(id=f"call_{self._call}", name=name, arguments=arguments))
        self.queue.append(LlmResponse(text="", finish_reason="tool_calls", tool_calls=tool_calls))

    def text(self, text: str = "Done.") -> None:
        self.queue.append(LlmResponse(text=text, finish_reason="stop"))


def tree_record(session: Session, tree: dict[str, str]) -> Any:
    return deepcopy(session.plan_store().get(f"{tree['session']}:run-tree:{tree['run']}"))


def forged(kind: str, claimed_run: str) -> RunContext:
    """A context the host built that claims a live run of another session."""
    if kind == "root":
        return RunContext.root(claimed_run, RunBudget(50))
    return RunContext(
        ledger=RunLedger.start("host-own"), budget=RunBudget(50), parent_run_id=claimed_run
    )


# -- a host context never joins another session's tree ---------------------


@pytest.mark.parametrize("claim", ["root", "parent"])
def test_a_host_context_claiming_a_live_run_puts_no_grandchild_in_its_tree(claim: str) -> None:
    rig = Rig()
    ada, bob = rig.session("ada"), rig.session("bob")
    seen: dict[str, Any] = {}

    def relay() -> None:
        turn = current_turn_run(ada.key())
        assert turn is not None
        tree = {"session": ada.key(), "run": turn.root_run_id}
        seen["tree"], seen["before"] = tree, tree_record(ada, tree)
        rig.runtime.send(bob, "Go", context=forged(claim, turn.root_run_id))
        seen["after"] = tree_record(ada, tree)

    rig.hooks["relay"] = relay
    rig.calls(("research", {"task": "Look."}))  # ada: opens her tree
    rig.text("Child answer.")  # ada's child
    rig.calls(("relay", {}))  # ada: host code sends to bob
    rig.calls(("research", {"task": "Look."}))  # bob, under the forged context
    rig.text("Bob's child answer.")  # bob's child
    rig.text("Bob done.")  # bob
    rig.text("Ada done.")  # ada
    rig.runtime.send(ada, "Go")

    assert rig.session("ada", "chat::sub::research").child_refs() == {
        "tree": seen["tree"],
        "plan_ledger": None,
    }
    assert rig.session("bob", "chat::sub::research").child_refs() == {
        "tree": None,
        "plan_ledger": None,
    }
    assert seen["before"] is not None
    assert seen["after"] == seen["before"]


def test_a_direct_runner_call_with_a_forged_parent_run_joins_no_tree() -> None:
    rig = Rig()
    ada = rig.session("ada")
    seen: dict[str, Any] = {}

    def forge() -> None:
        turn = current_turn_run(ada.key())
        assert turn is not None
        live = turn.root_run_id
        subagent = rig.runtime.mode_for(ada).subagents["research"]
        result = SubagentRunner(rig.harness, rig.runtime).run(
            subagent, ada, RunContext.root(live, RunBudget(50)), live, "Look."
        )
        seen["outcome"], seen["tree"] = result.outcome, {"session": ada.key(), "run": live}

    rig.hooks["forge"] = forge
    rig.calls(("forge", {}))  # ada: host code drives the runner itself
    rig.text("Child answer.")  # the child
    rig.text("Ada done.")  # ada
    rig.runtime.send(ada, "Go")

    assert seen["outcome"] == "completed"
    assert rig.session("ada", "chat::sub::research").child_refs() == {
        "tree": None,
        "plan_ledger": None,
    }
    assert tree_record(ada, seen["tree"]) is None


# -- the runner's own context joins the calling turn's tree ----------------


def test_a_child_of_a_child_saves_the_roots_tree_two_deep() -> None:
    rig = Rig()
    ada = rig.session("ada")
    rig.calls(("research", {"task": "Look."}))  # ada
    rig.calls(("dig", {"task": "Dig."}))  # child
    rig.text("Grandchild answer.")  # grandchild
    rig.text("Child answer.")  # child
    rig.text("Ada done.")  # ada
    root = rig.runtime.send(ada, "Go").run_id

    grandchild = rig.session("ada", "chat::sub::research::sub::dig")
    refs = grandchild.child_refs()
    assert refs is not None and refs["tree"] == {"session": ada.key(), "run": root}
    assert grandchild.child_depth() == 2
    assert [depth for _, depth in rig.depths] == [0, 1, 2]


def test_a_grandchild_of_a_child_resumed_under_a_host_context_saves_its_tree_and_plan() -> None:
    rig = Rig({"budget": {"mode": "shared", "max_steps": 20}})
    ada = rig.session("ada")
    rig.calls(
        (
            "plan_draft",
            {
                "goal": "Plan the trip",
                "steps": [{"title": "Step 1", "instruction": "Do it.", "owner": "agent"}],
            },
        )
    )
    drafted = rig.runtime.send(ada, "Plan the trip").pending_approvals[0]
    rig.text("Plan approved.")
    rig.runtime.decide(ada, drafted.id, True)
    rig.runtime.send(ada, "")
    rig.calls(("research", {"task": "Find the weather."}))  # the step
    rig.calls(("weather", {"city": "Paris"}))  # child: pauses
    rig.calls(("plan_complete_step", {"summary": "Asked."}))  # the step
    rig.text("Step done.")
    rig.runtime.plan(ada).advance()
    child = rig.session("ada", "chat::sub::research")
    refs = child.child_refs()
    assert refs is not None and refs["tree"] is not None and refs["plan_ledger"] is not None

    rig.calls(("dig", {"task": "Dig."}))  # child, under a host context
    rig.text("Grandchild answer.")
    rig.text("Child answer.")
    rig.runtime.send(child, "Dig deeper", context=RunContext.root("host", RunBudget(100)))

    assert rig.session("ada", "chat::sub::research::sub::dig").child_refs() == refs


def test_one_host_context_on_two_sessions_saves_each_grandchild_its_own_parents_tree() -> None:
    rig = Rig()
    ada, bob = rig.session("ada"), rig.session("bob")
    for who in (ada, bob):
        rig.calls(("research", {"task": "Look."}))
        rig.text("Child answer.")
        rig.text("Done.")
        rig.runtime.send(who, "Go")
    ada_child = rig.session("ada", "chat::sub::research")
    bob_child = rig.session("bob", "chat::sub::research")
    ada_refs, bob_refs = ada_child.child_refs(), bob_child.child_refs()
    assert ada_refs is not None and bob_refs is not None
    assert ada_refs["tree"] is not None and ada_refs["tree"] != bob_refs["tree"]
    shared = RunContext.root("host-shared", RunBudget(100))

    def relay() -> None:
        rig.runtime.send(bob_child, "Dig", context=shared)

    rig.hooks["relay"] = relay
    rig.calls(("relay", {}))  # ada's child, under the shared context
    rig.calls(("dig", {"task": "Dig."}))  # bob's child, under the same context
    rig.text("Bob's grandchild answer.")
    rig.text("Bob's child answer.")
    rig.text("Ada's child answer.")
    rig.runtime.send(ada_child, "Relay", context=shared)

    grand = rig.session("bob", "chat::sub::research::sub::dig").child_refs()
    assert grand is not None and grand["tree"] == bob_refs["tree"]


# -- a child started again --------------------------------------------------


def test_a_child_started_again_from_a_new_tree_saves_it_at_the_runners_depth() -> None:
    rig = Rig()
    ada = rig.session("ada")
    child = rig.session("ada", "chat::sub::research")
    child.remember_child_binding(CHEAP.to_row(), 3)
    child.remember_child_refs({"session": "elsewhere", "run": "old-run"}, None)

    rig.calls(("research", {"task": "Look."}))
    rig.text("Child answer.")
    rig.text("Ada done.")
    root = rig.runtime.send(ada, "Go").run_id

    assert child.child_refs() == {"tree": {"session": ada.key(), "run": root}, "plan_ledger": None}
    assert child.child_depth() == 1
    assert [depth for _, depth in rig.depths] == [0, 1]

    # Started again under a tree the host built: the saved tree is kept.
    saved = child.child_refs()
    rig.calls(("research", {"task": "Again."}))
    rig.text("Child again.")
    rig.text("Ada again.")
    rig.runtime.send(ada, "Again", context=RunContext.root("host", RunBudget(100)))
    assert child.child_refs() == saved


# -- depth --------------------------------------------------------------------


def test_a_host_context_shallower_than_the_saved_depth_runs_at_the_saved_depth() -> None:
    rig = Rig()
    child = rig.session("ada", "chat::sub::research").using_mode("research")
    child.remember_child_binding(CHEAP.to_row(), MAX_DEPTH - 1)
    child.remember_child_refs(None, None)

    rig.calls(("dig", {"task": "Dig."}))  # one level deeper is the ceiling
    rig.text("Child answer.")
    rig.runtime.send(child, "Q", context=RunContext.root("host", RunBudget(100)))

    assert rig.depths == [(child.key(), MAX_DEPTH - 1)]
    assert rig.session("ada", "chat::sub::research::sub::dig").child_binding() is None
    assert len(rig.requests) == 2


def test_a_host_context_deeper_than_the_saved_depth_runs_deeper() -> None:
    rig = Rig()
    child = rig.session("ada", "chat::sub::research").using_mode("research")
    child.remember_child_binding(CHEAP.to_row(), 1)
    child.remember_child_refs(None, None)
    deeper = RunContext(ledger=RunLedger.start("host"), budget=RunBudget(100), depth=2)

    rig.text("Child answer.")
    rig.runtime.send(child, "Q", context=deeper)

    assert rig.depths == [(child.key(), 2)]


# -- a resumed root and a root that parks --------------------------------------


def test_a_resumed_root_delegates_in_the_tree_its_parked_run_began() -> None:
    rig = Rig()
    ada = rig.session("ada")
    rig.calls(("research", {"task": "Look."}))  # ada
    rig.calls(("weather", {"city": "Paris"}))  # her child: parks
    rig.calls(("weather", {"city": "Rome"}))  # ada: parks on her own approval
    parked = rig.runtime.send(ada, "Go")
    assert len(parked.pending_approvals) == 1
    first = parked.run_id
    child = rig.session("ada", "chat::sub::research").child_refs()
    assert child is not None and child["tree"] == {"session": ada.key(), "run": first}

    rig.runtime.decide(ada, parked.pending_approvals[0].id, True)
    rig.calls(("survey", {"task": "Survey."}))  # ada, resumed under a new run id
    rig.text("Survey answer.")  # the new child
    rig.text("Ada done.")  # ada
    resumed = rig.runtime.send(ada, "")
    assert resumed.run_id != first

    assert rig.session("ada", "chat::sub::survey").child_refs() == {
        "tree": {"session": ada.key(), "run": first},
        "plan_ledger": None,
    }


def test_a_top_level_session_bound_by_its_host_keeps_resuming_after_it_parks() -> None:
    rig = Rig()
    ada = rig.session("ada")
    hosted = RunContext.root("host", RunBudget(100)).with_binding(CHEAP)
    rig.calls(("weather", {"city": "Paris"}))
    approval = rig.runtime.send(ada, "Go", context=hosted).pending_approvals[0].id

    for city in ("Rome", "Oslo"):
        rig.runtime.decide(ada, approval, True)
        rig.calls(("weather", {"city": city}))  # resumed: parks again
        approval = rig.runtime.send(ada, "").pending_approvals[0].id

    rig.runtime.decide(ada, approval, True)
    rig.text("Resumed.")
    assert rig.runtime.send(ada, "").stopped_because is None
    rig.text("Again.")
    assert rig.runtime.send(ada, "New question").stopped_because is None
    assert rig.ran.count("weather") == 3


# A root that resumes into a tree that already has a budget record holds what
# the tree has left and is narrowed to it, like any other turn that continues
# the tree: the tree's allowance holds across the approval pause.


def resumed_root_delegates_again(chat_steps: int) -> tuple[Rig, Session, str, str]:
    """Ada delegates research (its child parks on weather) and parks on her
    own weather; her approval is decided and she resumes, delegates survey
    (it answers) and parks again. Returns the rig, the parked child, its
    approval id and Ada's first run id."""
    rig = Rig(chat_steps=chat_steps)
    ada = rig.session("ada")
    rig.calls(("research", {"task": "Look."}))  # ada
    rig.calls(("weather", {"city": "Paris"}))  # her child: parks
    rig.calls(("weather", {"city": "Rome"}))  # ada: parks on her own approval
    parked = rig.runtime.send(ada, "Go")
    first = parked.run_id

    rig.runtime.decide(ada, parked.pending_approvals[0].id, True)
    rig.calls(("survey", {"task": "Survey."}))  # ada, resumed: delegates again
    rig.text("Surveyed.")  # the new child answers
    rig.calls(("weather", {"city": "Oslo"}))  # ada parks again
    rig.text("Spare.")
    try:
        rig.runtime.send(ada, "")
    except HarnessError:
        pass

    child = rig.session("ada", "chat::sub::research")
    waiting = pending_approval_ids([entry.message for entry in child.thread().messages()])
    assert len(waiting) == 1
    return rig, child, waiting[0], first


def test_a_parked_subagent_keeps_what_its_tree_has_left_after_the_root_resumed() -> None:
    rig, child, approval, _ = resumed_root_delegates_again(20)
    ran = rig.ran.count("weather")

    rig.text("Child done.")
    rig.runtime.decide(child, approval, True)
    resumed = rig.runtime.send(child, "")

    assert rig.ran.count("weather") == ran + 1
    assert resumed.stopped_because is None


def test_a_resumed_root_is_held_to_its_tree_record_so_the_tree_never_overspends() -> None:
    rig, child, approval, first = resumed_root_delegates_again(6)
    ada = rig.session("ada")
    record = tree_record(ada, {"session": ada.key(), "run": first})
    assert isinstance(record, dict)
    assert record["steps"] <= record["max_steps"]
    assert record["steps"] == record["max_steps"]
    ran, requests = rig.ran.count("weather"), len(rig.requests)

    # Spent, the tree refuses the parked child before anything is recorded.
    with pytest.raises(HarnessError) as caught:
        rig.runtime.decide(child, approval, True)
    assert caught.value.code == "run_not_permitted"
    assert pending_approval_ids([entry.message for entry in child.thread().messages()]) == [
        approval
    ]
    assert rig.ran.count("weather") == ran
    assert len(rig.requests) == requests


# A resume the spent tree refuses still begins a run, and the session keeps the
# trees of its newest runs only. The run the thread is parked on is never
# dropped, so retrying the refused resume never escapes the tree.
def test_a_resume_of_a_spent_tree_is_refused_however_often_it_is_retried() -> None:
    rig, _, _, _ = resumed_root_delegates_again(6)
    ada = rig.session("ada")
    assert len(pending_approval_ids([entry.message for entry in ada.thread().messages()])) == 1
    ran, requests = len(rig.ran), len(rig.requests)
    refusals = 0
    for _ in range(23):
        try:
            rig.runtime.send(ada, "")
        except HarnessError as error:
            assert error.code == "run_not_permitted"
            refusals += 1

    assert refusals == 23
    assert len(rig.ran) == ran
    assert len(rig.requests) == requests


class RecordingStore(DurableMemoryStore):
    """A durable memory store that records every key it is asked to write."""

    puts: ClassVar[list[str]] = []

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        RecordingStore.puts.append(key)
        super().put(key, payload, ttl_seconds)


# A run's tree is kept under a key of its own: starting a run never rewrites
# the session's durable record, where a host keeps capabilities it may revoke
# from another process while the run starts.
def test_a_run_start_never_rewrites_the_session_durable_record() -> None:
    rig = Rig(durable=RecordingStore)
    ada = rig.session("ada")
    ada.using_capability("grant", {"on": True})
    RecordingStore.puts = []

    rig.text("Done.")
    rig.runtime.send(ada, "Go")

    assert f"{ada.key()}:durable" not in RecordingStore.puts
    assert f"{ada.key()}:run-trees" in RecordingStore.puts


def test_a_resumed_root_that_would_overspend_its_tree_stops_at_the_trees_allowance() -> None:
    rig = Rig(chat_steps=6)
    ada = rig.session("ada")
    rig.calls(("research", {"task": "Look."}))  # ada
    rig.calls(("weather", {"city": "Paris"}))  # her child: parks
    rig.calls(("weather", {"city": "Rome"}))  # ada: parks on her own approval
    parked = rig.runtime.send(ada, "Go")

    rig.runtime.decide(ada, parked.pending_approvals[0].id, True)
    rig.calls(("survey", {"task": "Survey."}))  # ada, resumed
    rig.text("Surveyed.")
    for q in ("a", "b", "c", "d"):
        rig.calls(("lookup", {"q": q}))  # ada would keep going
    rig.text("Done.")
    rig.runtime.send(ada, "")

    record = tree_record(ada, {"session": ada.key(), "run": parked.run_id})
    assert isinstance(record, dict)
    assert record["steps"] <= record["max_steps"]


# A parked root whose tree is spent: its decision is refused before anything
# is recorded, so the person's answer is never consumed by a resume the tree
# then refuses.
def test_a_parked_roots_decision_on_a_spent_tree_is_refused_before_it_is_recorded() -> None:
    rig, _, _, _ = resumed_root_delegates_again(6)
    ada = rig.session("ada")
    waiting = pending_approval_ids([entry.message for entry in ada.thread().messages()])
    assert len(waiting) == 1
    rows = len(ada.thread().messages())

    with pytest.raises(HarnessError) as caught:
        rig.runtime.decide(ada, waiting[0], True)

    assert caught.value.code == "run_not_permitted"
    assert pending_approval_ids([entry.message for entry in ada.thread().messages()]) == waiting
    assert len(ada.thread().messages()) == rows


# Two resumes of one parked root that never delegated, the second starting
# while the first waits on its model: both join the tree the parked run began
# in, and the tree's allowance bounds the two of them together.
def test_two_concurrent_resumes_of_a_parked_root_never_overspend_its_tree() -> None:
    rig = Rig(chat_steps=6)
    ada = rig.session("ada")
    rig.calls(("weather", {"city": "Paris"}))  # ada parks, delegating nothing
    parked = rig.runtime.send(ada, "Go")
    rig.runtime.decide(ada, parked.pending_approvals[0].id, True)
    seen: dict[str, Any] = {}

    def second_worker() -> None:
        scripted = [
            LlmResponse(
                text="",
                finish_reason="tool_calls",
                tool_calls=[LlmToolCall(id="w2_1", name="survey", arguments={"task": "S."})],
            ),
            LlmResponse(text="Surveyed by the second worker.", finish_reason="stop"),
            LlmResponse(text="Second worker done.", finish_reason="stop"),
        ]
        rig.queue[0:0] = scripted
        before = len(rig.queue)
        try:
            seen["second"] = rig.runtime.send(ada, "").stopped_because
        except HarnessError as error:
            seen["second"] = error.code
        unused = len(scripted) - (before - len(rig.queue))
        del rig.queue[0 : max(0, unused)]

    rig.on_request = second_worker
    rig.calls(("survey", {"task": "Survey."}))  # the first resume
    rig.text("Surveyed.")
    for q in ("a", "b", "c"):
        rig.calls(("lookup", {"q": q}))
    rig.text("First worker done.")
    rig.runtime.send(ada, "")

    record = tree_record(ada, {"session": ada.key(), "run": parked.run_id})
    assert isinstance(record, dict)
    assert record["steps"] <= record["max_steps"]
    assert seen["second"] == "run_not_permitted"


class HookedStore(DurableMemoryStore):
    """A durable memory store that runs a hook before a write to a watched key."""

    watched: ClassVar[set[str]] = set()
    hook: ClassVar[Callable[[], None] | None] = None

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        hook = HookedStore.hook
        if key in HookedStore.watched and hook is not None:
            HookedStore.hook = None
            hook()
        super().put(key, payload, ttl_seconds)


# The runner saving a child's binding and run tree never rewrites the child's
# durable record, where the host keeps capabilities it may revoke from another
# process while the parent delegates.
def test_a_capability_revoked_while_the_runner_saves_a_child_stays_revoked() -> None:
    HookedStore.hook = None
    rig = Rig(durable=HookedStore)
    ada = rig.session("ada")
    rig.calls(("research", {"task": "One."}))
    rig.text("Child one.")
    rig.text("Ada one.")
    rig.runtime.send(ada, "One")
    child = rig.session("ada", "chat::sub::research")
    child.using_capability("grant", {"on": True})

    def revoke() -> None:
        # Another worker, between the runner's read and its write.
        rig.session("ada", "chat::sub::research").forget_capability("grant")

    HookedStore.watched = {f"{child.key()}:durable", f"{child.key()}:child"}
    HookedStore.hook = revoke
    rig.calls(("research", {"task": "Two."}))
    rig.text("Child two.")
    rig.text("Ada two.")
    rig.runtime.send(ada, "Two")

    assert HookedStore.hook is None
    assert rig.session("ada", "chat::sub::research").capability("grant") is None


# A root that delegates and then parks on its last step: its own budget, the
# tree's ledger with its child's steps on it, is spent. The decision check and
# the resume read that one measure, so the decision is never accepted and then
# refused on every resume: the resume acts on it and stops on its own budget.
def test_a_root_that_delegated_and_parked_on_its_last_step_acts_on_its_decision() -> None:
    rig = Rig(chat_steps=3)
    ada = rig.session("ada")
    rig.calls(("survey", {"task": "S"}))  # ada
    rig.text("Child answer.")  # her child
    rig.calls(("weather", {"city": "Paris"}))  # ada parks on her last step
    parked = rig.runtime.send(ada, "Go")
    assert len(parked.pending_approvals) == 1
    requests = len(rig.requests)

    rig.runtime.decide(ada, parked.pending_approvals[0].id, True)
    resumed = rig.runtime.send(ada, "")

    assert rig.ran.count("weather") == 1
    assert resumed.stopped_because is not None
    assert len(rig.requests) == requests
    record = tree_record(ada, {"session": ada.key(), "run": parked.run_id})
    assert isinstance(record, dict)
    assert record["steps"] <= record["max_steps"]


# No public seam reaches a refused grant in child_started: a fresh root's tree
# is named by its own new run id, so it has no record before its first
# delegation, and a root with no step left is stopped by the runner's own
# exhaustion check before a child is spawned. Pinned one level down: an open
# root whose tree record appears, already spent, before it delegates.
def test_child_started_refuses_the_delegation_when_the_tree_cannot_fund_it() -> None:
    rig = Rig()
    ada = rig.session("ada")
    trees = RunTrees()
    member = membership("none", ada.key(), "run_root", None, None, 0, 0)
    assert member["route"] == "open"
    trees.root_run(ada, "run_root", RunContext.root("run_root", RunBudget(5)), member, 300)
    spent = budget.open_ledger(
        "run_root", {"mode": "shared", "max_steps": 1, "max_cost_micros": None}
    )
    ada.plan_store().put(f"{ada.key()}:run-tree:run_root", {**spent, "steps": 1})

    with pytest.raises(HarnessError) as caught:
        trees.child_started(ada, member, 300)

    assert caught.value.code == "run_not_permitted"
