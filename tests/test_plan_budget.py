"""G-366: a plan's budget is the app's choice (Wish, q_fb920047).

One shared budget for all of a plan's steps, the per-turn budget per step (the
default), or no plan-level budget. Ports prism-harness's
tests/PlanBudgetTest.php, driven through ``send``, ``decide`` and
``plan(session).advance()`` with a scripted model; the network is refused and
every store is in memory.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from prism_harness import (
    AgentRuntime,
    Durability,
    HarnessError,
    HarnessEvent,
    HarnessEvents,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    MemorySessionStore,
    ModeRegistry,
    Participant,
    PlanBudgetExhausted,
    PlanBudgetInvalid,
    PrismHarness,
    RunBudget,
    RunContext,
    Session,
    ToolRegistry,
)
from prism_harness.plans import PlanRefused, Plans, machine


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


class Counted:
    def __init__(self, name: str, result: str, rig: Rig) -> None:
        self._name = name
        self._result = result
        self._rig = rig

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        self._rig.ran.append(self._name)
        if self._name == "lookup" and self._rig.on_lookup is not None:
            self._rig.on_lookup()
        return self._result


class Rig:
    def __init__(self, plans: dict[str, Any] | None = None, mode: dict[str, Any] | None = None) -> None:
        self.queue: list[LlmResponse] = []
        self.requests: list[LlmRequest] = []
        self.ran: list[str] = []
        self.events: list[HarnessEvent] = []
        self.on_lookup: Callable[[], None] | None = None
        self._call = 0
        self._plans_config = {"enabled": True, **(plans or {})}
        self._mode = mode or {}
        self.harness = PrismHarness(
            drivers={"memory": MemorySessionStore, "durable": DurableMemoryStore},
            stores={"ephemeral": "memory", "durable": "durable"},
        )
        self.restart()
        self.session: Session = self.harness.for_(Participant("User", "ada")).session("chat")

    def restart(self) -> None:
        """A new process: new plans and runtime over the same stores."""
        events = HarnessEvents()
        events.listen(self.events.append)
        chat: dict[str, Any] = {
            "system_prompt": "Be brief.",
            "tools": ["weather", "lookup"],
            "max_steps": 8,
            "requires_approval": ["weather"],
            "plans": True,
            **self._mode,
        }
        self.runtime = AgentRuntime(
            client=self._client,
            modes=ModeRegistry({"default": "chat", "modes": {"chat": chat}}),
            tools=ToolRegistry()
            .register(Counted("weather", "Sunny", self))
            .register(Counted("lookup", "Found", self)),
            events=events,
            config={"provider": "openai", "model": "gpt-4o"},
            plans=Plans(self._plans_config, index_store=self.harness.durable_store()),
        )

    def _client(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if not self.queue:
            raise AssertionError("The scripted model has no response left.")
        return self.queue.pop(0)

    def calls(self, *calls: tuple[str, dict[str, Any]]) -> None:
        tool_calls = []
        for name, arguments in calls:
            self._call += 1
            tool_calls.append(LlmToolCall(id=f"call_{self._call}", name=name, arguments=arguments))
        self.queue.append(LlmResponse(text="", finish_reason="tool_calls", tool_calls=tool_calls))

    def text(self, text: str = "Done.") -> None:
        self.queue.append(LlmResponse(text=text, finish_reason="stop"))

    def step(self, lookups: int, summary: str) -> None:
        """One step's turn: lookups, then the report, then text."""
        for i in range(lookups):
            self.calls(("lookup", {"q": f"q{i}"}))
        self.calls(("plan_complete_step", {"summary": summary}))
        self.text("Step done.")

    def active(self, count: int = 3) -> None:
        self.calls(("plan_draft", {"goal": "Plan the trip", "steps": steps(count)}))
        pending = self.runtime.send(self.session, "Plan the trip").pending_approvals[0]
        self.text("Plan approved.")
        self.runtime.decide(self.session, pending.id, True)
        self.runtime.send(self.session, "")

    def advance(self, context: RunContext | None = None) -> Any:
        return self.runtime.plan(self.session).advance(context)

    def budget(self) -> dict[str, Any]:
        found = self.runtime.plan(self.session).budget()
        assert found is not None
        return found

    def plan(self) -> dict[str, Any]:
        snapshot = self.runtime.plan(self.session).current()
        return snapshot.to_dict() if snapshot is not None else {}

    def lookups(self) -> int:
        return self.ran.count("lookup")


def steps(count: int) -> list[dict[str, Any]]:
    return [
        {"title": f"Step {i}", "instruction": f"Do part {i} of the work.", "owner": "agent"}
        for i in range(1, count + 1)
    ]


def test_per_step_is_the_default_and_records_nothing() -> None:
    rig = Rig()
    rig.active(2)
    before = len(rig.requests)
    rig.step(3, "one")
    assert rig.advance().status == "ran"
    rig.step(3, "two")
    assert rig.advance().status == "ran"
    assert rig.advance().status == "done"

    assert len(rig.requests) - before == 10
    assert rig.lookups() == 6
    assert rig.budget()["mode"] == "per_step"
    assert rig.budget()["steps"] == 0
    assert rig.harness.durable_store().get(f"{rig.session.key()}:plan-budget") is None


def test_per_step_ignores_configured_limits() -> None:
    rig = Rig({"budget": {"mode": "per_step", "max_steps": 1}})
    rig.active(1)
    rig.step(2, "one")
    rig.advance()
    assert rig.lookups() == 2
    assert rig.advance().status == "done"


def test_none_keeps_the_per_turn_step_limit_and_the_advance_limit() -> None:
    rig = Rig({"budget": {"mode": "none"}, "max_advances": 2}, {"max_steps": 2})
    rig.active(3)
    before = len(rig.requests)
    rig.step(2, "one")
    rig.advance()
    assert len(rig.requests) - before == 2
    assert rig.lookups() == 2

    rig.queue.clear()
    rig.step(0, "two")
    rig.advance()
    outcome = rig.advance()
    assert outcome.status == "blocked"
    assert outcome.reason == "budget"


def test_shared_budget_spent_across_several_steps_stops_the_plan_once() -> None:
    rig = Rig({"budget": {"mode": "shared", "max_steps": 5}})
    rig.active(3)
    before = len(rig.requests)
    rig.step(1, "one")
    rig.advance()
    rig.step(3, "two")
    rig.advance()

    assert len(rig.requests) - before == 5
    assert rig.lookups() == 3
    assert rig.budget()["steps"] == 5

    outcome = rig.advance()
    assert outcome.status == "blocked"
    assert outcome.reason == "plan_budget_exhausted"
    plan = rig.plan()
    assert plan["status"] == "blocked"
    assert plan["blockedReason"] == "plan_budget_exhausted"
    assert plan["steps"][0]["state"] == "done"
    exhausted = [event for event in rig.events if isinstance(event, PlanBudgetExhausted)]
    assert len(exhausted) == 1
    assert exhausted[0].code == "plan_budget_exhausted"
    assert exhausted[0].limit == "steps"
    assert exhausted[0].steps == 5

    with pytest.raises(PlanRefused) as caught:
        rig.runtime.plan(rig.session).resume("user:ada")
    assert caught.value.code == "plan.not_resumable"
    assert rig.advance().status == "not_active"
    assert len(rig.requests) - before == 5
    assert len([event for event in rig.events if isinstance(event, PlanBudgetExhausted)]) == 1


def test_shared_budget_carries_across_an_approval_pause_and_a_restart() -> None:
    rig = Rig({"budget": {"mode": "shared", "max_steps": 4}})
    rig.active(2)
    before = len(rig.requests)
    rig.calls(("weather", {"city": "Paris"}))
    rig.advance()
    held = rig.plan()["steps"][0]
    assert held["state"] == "held"
    assert rig.budget()["steps"] == 1

    rig.restart()
    for q in ("a", "b", "c", "d"):
        rig.calls(("lookup", {"q": q}))
    rig.runtime.decide(rig.session, held["approvalIds"][0], True)
    rig.runtime.send(rig.session, "")

    assert rig.ran.count("weather") == 1
    assert rig.lookups() == 3
    assert len(rig.requests) - before == 4
    assert rig.budget()["steps"] == 4

    rig.queue.clear()
    outcome = rig.advance()
    assert outcome.status == "blocked"
    assert outcome.reason == "plan_budget_exhausted"
    assert len(rig.requests) - before == 4


def test_only_one_of_two_concurrent_advances_runs_and_is_charged() -> None:
    rig = Rig({"budget": {"mode": "shared", "max_steps": 4}})
    rig.active(2)
    second: list[str] = []

    def advance_again() -> None:
        try:
            rig.advance()
            second.append("ran")
        except HarnessError as error:
            second.append(error.code)

    rig.on_lookup = advance_again
    rig.step(1, "one")
    rig.advance()

    assert second == ["session_locked"]
    assert rig.budget()["steps"] == 3


def test_two_turns_are_never_granted_the_same_remainder() -> None:
    rig = Rig({"budget": {"mode": "shared", "max_steps": 10}})
    rig.active(1)
    plans = rig.runtime.plans
    plan = plans.load(rig.session)
    assert plan is not None
    claimed = machine.claim(plan, plan["steps"][0]["id"], "worker", plans.now() + 300, plans.now())
    plans.write(rig.session, claimed, plan["revision"])

    def turn(run_id: str) -> RunContext:
        return plans.plan_turn(rig.session, run_id, RunContext.root(run_id, RunBudget(8)), None, 300)

    assert turn("run_a").budget.max_steps == 8
    assert turn("run_b").budget.max_steps == 2
    with pytest.raises(HarnessError) as caught:
        turn("run_c")
    assert caught.value.code == "plan_budget_exhausted"


def test_model_budget_fields_and_a_per_call_run_tree_cannot_change_the_budget() -> None:
    rig = Rig({"budget": {"mode": "shared", "max_steps": 3}})
    rig.calls(
        (
            "plan_draft",
            {"goal": "Plan", "steps": steps(1), "budget": {"mode": "none", "max_steps": 999}},
        )
    )
    rig.text("ok")
    rig.runtime.send(rig.session, "Plan")
    assert rig.runtime.plan(rig.session).current() is None

    rig.active(2)
    before = len(rig.requests)
    rig.text("unused")
    with pytest.raises(PlanRefused) as caught:
        rig.advance(RunContext.root("host", RunBudget(100)))
    assert caught.value.code == "plan.budget_context"
    assert len(rig.requests) == before
    assert rig.budget()["mode"] == "shared"
    assert rig.budget()["max_steps"] == 3


def test_host_code_chooses_a_plan_budget_and_can_only_lower_it() -> None:
    rig = Rig({"budget": {"mode": "per_step", "max_steps": 50}})
    rig.active(2)
    handle = rig.runtime.plan(rig.session)
    handle.use_budget({"mode": "shared", "max_steps": 2})
    assert rig.budget()["mode"] == "shared"
    assert rig.budget()["max_steps"] == 2
    handle.use_budget({"mode": "shared", "max_steps": 40})
    assert rig.budget()["max_steps"] == 2
    with pytest.raises(PlanRefused) as caught:
        handle.use_budget({"mode": "none"})
    assert caught.value.code == "plan.budget_fixed"

    rig.step(2, "one")
    rig.advance()
    assert rig.lookups() == 2
    assert rig.advance().reason == "plan_budget_exhausted"


@pytest.mark.parametrize(
    "budget",
    [
        {"mode": "unlimited"},
        {"mode": "shared"},
        {"mode": "shared", "max_steps": 0},
        {"mode": "shared", "max_steps": "5"},
        {"mode": "shared", "max_steps": True},
        {"mode": "shared", "max_steps": 5, "max_cost_usd": -1},
        {"mode": "shared", "max_steps": 5, "per_call": True},
        "shared",
    ],
)
def test_an_invalid_plan_budget_fails_when_the_host_configures_it(budget: object) -> None:
    with pytest.raises(PlanBudgetInvalid):
        Plans({"enabled": True, "budget": budget})
