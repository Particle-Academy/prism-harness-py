"""One authority for a run's step allowance (G-387).

A run's allowance is worked out ONCE, when it begins, from facts the harness
wrote itself: the steps the parked run it resumes had used (children
included, ``harness_steps_used`` on the row it parked on) and, for a subagent
child, the ceiling saved with its binding when the runner started it. Every
consumer reads that one allowance.

Each test enters where a real caller enters (``send``, ``send_structured``,
``decide``, ``record_approval``, ``SubagentRunner.run``, a host ``RunContext``)
and most of them through FRESH harness and runtime objects over one file store,
so only what is on disk survives between turns. The model is scripted; the
network is refused.
"""

from __future__ import annotations

import socket
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from prism_harness import (
    AgentResponse,
    AgentRuntime,
    FileSessionStore,
    HarnessError,
    LlmRequest,
    LlmResponse,
    LlmToolCall,
    ModeRegistry,
    Participant,
    PrismHarness,
    RunBudget,
    RunContext,
    Session,
    StructuredAgentResponse,
    SubagentRunner,
    ToolRegistry,
    record_approval,
)
from prism_harness.runtime import RuntimeConfig
from prism_harness.thread_rows import paused_approval_row, pending_approval_ids


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def modes(lead_steps: int = 8) -> ModeRegistry:
    return ModeRegistry(
        {
            "default": "lead",
            "modes": {
                "lead": {
                    "system_prompt": "Lead.",
                    "tools": ["echo", "lookup", "worker", "alpha", "beta", "big", "deep", "gated"],
                    "requires_approval": ["echo", "gated"],
                    "max_steps": lead_steps,
                    "subagents": {
                        "worker": {"mode": "work", "max_steps": 2},
                        "alpha": {"mode": "work", "max_steps": 2},
                        "beta": {"mode": "work", "max_steps": 2},
                        "big": {"mode": "work", "max_steps": 4},
                        "deep": {"mode": "free", "max_steps": 3},
                        "gated": {"mode": "free", "max_steps": 3},
                    },
                },
                "work": {
                    "system_prompt": "Work.",
                    "tools": ["echo"],
                    "requires_approval": ["echo"],
                    "max_steps": 8,
                },
                "free": {"system_prompt": "Free.", "tools": ["lookup"], "max_steps": 8},
            },
        }
    )


class World:
    """Fresh harness and runtime objects for every call, over one file store:
    a different worker each time, with only what is on disk carried over."""

    def __init__(self, lead_steps: int = 8, config: RuntimeConfig | None = None) -> None:
        self.directory = tempfile.mkdtemp(prefix="prism-harness-g387-")
        self.lead_steps = lead_steps
        self.config: RuntimeConfig = config or {}
        self.queue: list[LlmResponse] = []
        self.requests: list[LlmRequest] = []
        self.ran: list[str] = []
        self._call = 0
        self.harness = self._harness()
        self.runtime = self._runtime()

    def _harness(self) -> PrismHarness:
        return PrismHarness(
            drivers={"files": lambda: FileSessionStore(self.directory)},
            stores={"ephemeral": "files", "durable": "files"},
        )

    def _runtime(self) -> AgentRuntime:
        world = self

        class Tool:
            def __init__(self, name: str) -> None:
                self.name = name

            def handle(self, args: dict[str, Any]) -> Any:
                world.ran.append(self.name)
                return "ok"

        tools = ToolRegistry().register(Tool("echo")).register(Tool("lookup"))

        def delegate(name: str) -> Callable[[Session], Any]:
            def factory(session: Session) -> Any:
                subagent = self.runtime.mode_for(session).subagents[name]
                return SubagentRunner(self.harness, self.runtime).tool(
                    subagent, session, RunContext.root("tool-run", RunBudget(100)), "tool-run"
                )

            return factory

        for name in ("worker", "alpha", "beta", "big", "deep", "gated"):
            tools.register_factory(name, delegate(name))
        return AgentRuntime(
            client=self._client,
            modes=modes(self.lead_steps),
            tools=tools,
            config={"provider": "openai", "model": "root-model", **self.config},
        )

    def fresh(self) -> World:
        """A new worker: new harness and runtime objects, the same disk."""
        self.harness = self._harness()
        self.runtime = self._runtime()
        return self

    def _client(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self.queue:
            return self.queue.pop(0)
        # By default every step asks for the gated tool again.
        self._call += 1
        return LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[LlmToolCall(f"auto_{self._call}", "echo", {"n": self._call})],
        )

    def calls(self, *calls: tuple[str, dict[str, Any]]) -> None:
        tool_calls = []
        for name, arguments in calls:
            self._call += 1
            tool_calls.append(LlmToolCall(id=f"call_{self._call}", name=name, arguments=arguments))
        self.queue.append(LlmResponse(text="", finish_reason="tool_calls", tool_calls=tool_calls))

    def text(self, text: str = "Done.") -> None:
        self.queue.append(LlmResponse(text=text, finish_reason="stop"))

    def root(self) -> Session:
        session = self.harness.for_(Participant("User", "ada")).session("support")
        session.using_mode("lead")
        return session

    def child(self, name: str = "worker") -> Session:
        return self.harness.for_(Participant("User", "ada")).session(f"support::sub::{name}")

    def pending(self, session: Session) -> list[str]:
        return pending_approval_ids([entry.message for entry in session.thread().messages()])

    def run_child_in_runner(self, name: str = "worker", budget: int = 8) -> None:
        """The runner starts a child directly, which parks on an approval
        after one step."""
        parent = self.root()
        subagent = self.runtime.mode_for(parent).subagents[name]
        SubagentRunner(self.harness, self.runtime).run(
            subagent, parent, RunContext.root("root", RunBudget(budget)), "root", "go"
        )
        assert len(self.requests) == 1
        assert len(self.pending(self.child(name))) == 1


def approve_and_resume(
    world: World,
    session: Callable[[], Session],
    times: int,
    context: Callable[[], RunContext | None] = lambda: None,
) -> AgentResponse | None:
    """Approve what the session waits on and resume it, up to ``times`` times,
    each through fresh objects."""
    response: AgentResponse | None = None
    for _ in range(times):
        waiting = world.fresh().pending(session())
        if not waiting:
            break
        record_approval(session(), waiting[0], True)
        response = world.fresh().runtime.send(session(), "", context=context())
        if response.finish_reason != "awaiting_approval":
            break
    return response


def rewrite_paused_row(session: Session, change: Callable[[dict[str, Any]], None]) -> None:
    """Change the row the session is parked on, as a store writer could."""
    key = f"{session.key()}:thread"
    harness_store = session._durable  # a test standing in for a store writer
    stored = harness_store.get(key)
    assert isinstance(stored, dict)
    for entry in reversed(stored["messages"]):
        if entry["message"].get("tool_approval_requests"):
            change(entry["message"])
            break
    harness_store.put(key, stored)


# -- D1: a child's declared ceiling holds on every turn the runner did not start


@pytest.mark.parametrize("how", ["none", "host"])
def test_a_child_declared_two_steps_takes_at_most_one_more_outside_the_runner(how: str) -> None:
    world = World()
    world.run_child_in_runner()

    approve_and_resume(
        world,
        world.child,
        6,
        (lambda: RunContext.root("host", RunBudget(10))) if how == "host" else (lambda: None),
    )

    # Declared 2, one taken in the runner: at most one more, however resumed.
    assert len(world.requests) <= 2
    assert world.ran.count("echo") >= 1


def test_a_host_context_larger_than_the_saved_ceiling_never_raises_it() -> None:
    world = World()
    world.run_child_in_runner("big")  # declared 4, one taken

    approve_and_resume(
        world, lambda: world.child("big"), 8, lambda: RunContext.root("h", RunBudget(50))
    )

    assert len(world.requests) == 4


def test_a_host_context_smaller_than_the_saved_ceiling_lowers_it() -> None:
    world = World()
    world.run_child_in_runner("big")  # declared 4, one taken: 3 of its own left

    approve_and_resume(
        world, lambda: world.child("big"), 1, lambda: RunContext.root("h", RunBudget(1))
    )

    assert len(world.requests) == 2


def test_a_runner_start_overwrites_the_saved_ceiling() -> None:
    world = World()
    world.run_child_in_runner("big")  # declared 4
    # The runner starts it again from a parent with one step left: its ceiling
    # is now 1, and a later turn outside the runner is held to that.
    parent = world.fresh().root()
    subagent = world.runtime.mode_for(parent).subagents["big"]
    ledger_context = RunContext.root("root2", RunBudget(1))
    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, ledger_context, "root2", "again"
    )
    assert len(world.requests) == 2

    approve_and_resume(world, lambda: world.child("big"), 6)

    assert len(world.requests) == 2


def test_a_runner_start_is_never_read_as_a_resume_of_the_parked_run() -> None:
    world = World()
    world.run_child_in_runner("big")
    # The parked row says the run used every step there is.
    rewrite_paused_row(world.child("big"), lambda row: row.update(harness_steps_used=999))
    parent = world.fresh().root()
    subagent = world.runtime.mode_for(parent).subagents["big"]

    SubagentRunner(world.harness, world.runtime).run(
        subagent, parent, RunContext.root("root2", RunBudget(8)), "root2", "a new task"
    )

    # A runner start is a new run: it takes model steps of its own.
    assert len(world.requests) == 2


# -- the tree a sibling spent: the decision is acted on, with no model step


@pytest.mark.parametrize("path", ["send", "structured"])
def test_a_child_whose_sibling_spent_the_tree_acts_on_its_approved_call(path: str) -> None:
    world = World(lead_steps=5)
    world.calls(("alpha", {"task": "A"}), ("beta", {"task": "B"}))  # root step 1
    world.calls(("echo", {"who": "alpha"}))  # alpha parks
    world.calls(("echo", {"who": "beta"}))  # beta parks
    world.text("Root done.")  # root step 2: the tree has spent 4 of 5
    world.runtime.send(world.root(), "Go")
    assert len(world.requests) == 4

    # alpha, resumed outside the runner, takes the tree's last step.
    approve_and_resume(world, lambda: world.child("alpha"), 1)
    assert len(world.requests) == 5
    assert world.ran.count("echo") == 1

    # beta's decision is accepted, and its approved call RUNS, with no model step.
    beta = world.fresh().child("beta")
    record_approval(beta, world.pending(beta)[0], True)
    fresh = world.fresh()
    if path == "send":
        response: AgentResponse = fresh.runtime.send(fresh.child("beta"), "")
    else:
        response = fresh.runtime.send_structured(fresh.child("beta"), "", SCHEMA)
    assert response.finish_reason == "budget_exhausted"
    assert world.ran.count("echo") == 2
    assert len(world.requests) == 5
    assert world.pending(world.child("beta")) == []


def test_a_decision_on_a_child_whose_tree_cost_is_spent_is_still_refused() -> None:
    world = World(lead_steps=5)
    world.calls(("alpha", {"task": "A"}))
    world.calls(("echo", {"who": "alpha"}))
    world.text("Root done.")
    world.runtime.send(world.root(), "Go")
    alpha = world.child("alpha")
    record = alpha.child_refs()
    assert record is not None and record["tree"] is not None
    tree = record["tree"]
    key = f"{tree['session']}:run-tree:{tree['run']}"
    store = alpha.plan_store()
    state = store.get(key)
    assert isinstance(state, dict)
    store.put(key, {**state, "max_cost_micros": 10, "cost_micros": 10})

    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().child("alpha"), world.pending(alpha)[0], True)

    assert caught.value.code == "run_not_permitted"
    assert world.ran.count("echo") == 0


# -- re-park: the ceiling holds in total across many approvals


def test_a_root_that_spent_five_of_eight_through_a_child_gets_at_most_three_more() -> None:
    world = World(lead_steps=8)
    world.calls(("deep", {"task": "Dig."}))  # root step 1
    for q in ("a", "b", "c"):
        world.calls(("lookup", {"q": q}))  # the child's three steps
    world.calls(("echo", {"n": 0}))  # root step 2 parks: 5 of 8 used
    parked = world.runtime.send(world.root(), "Go")
    assert parked.finish_reason == "awaiting_approval"
    assert len(world.requests) == 5

    approve_and_resume(world, world.root, 6)

    assert len(world.requests) == 8


def test_a_child_resumed_six_times_outside_the_runner_never_passes_its_ceiling() -> None:
    world = World()
    world.run_child_in_runner("big")

    approve_and_resume(world, lambda: world.child("big"), 6)

    assert len(world.requests) == 4


# -- which ceiling binds: the mode, the declaration, max_steps_ceiling (D2)


def test_the_mode_ceiling_binds_a_root() -> None:
    world = World(lead_steps=3)
    world.runtime.send(world.root(), "Go")
    approve_and_resume(world, world.root, 6)
    assert len(world.requests) == 3


@pytest.mark.parametrize("how", ["none", "host"])
def test_max_steps_ceiling_is_absolute_on_every_resume(how: str) -> None:
    world = World(lead_steps=8, config={"max_steps_ceiling": 2})
    world.runtime.send(world.root(), "Go")
    assert len(world.requests) == 1

    approve_and_resume(
        world,
        world.root,
        6,
        (lambda: RunContext.root("host", RunBudget(10))) if how == "host" else (lambda: None),
    )

    assert len(world.requests) == 2


def test_max_steps_ceiling_binds_a_child_below_its_declaration() -> None:
    world = World(config={"max_steps_ceiling": 2})
    world.run_child_in_runner("big")  # declared 4, ceiling 2

    approve_and_resume(world, lambda: world.child("big"), 6)

    assert len(world.requests) == 2


# -- spent on its last step: the decision runs on send and structured alike


@pytest.mark.parametrize("path", ["send", "structured"])
def test_a_root_spent_on_its_last_step_runs_its_approved_call(path: str) -> None:
    world = World(lead_steps=1)
    parked = world.runtime.send(world.root(), "Go")
    record_approval(world.fresh().root(), parked.pending_approvals[0].id, True)
    fresh = world.fresh()
    response = (
        fresh.runtime.send(fresh.root(), "")
        if path == "send"
        else fresh.runtime.send_structured(fresh.root(), "", SCHEMA)
    )
    assert response.finish_reason == "budget_exhausted"
    assert world.ran == ["echo"]
    assert len(world.requests) == 1


# -- the parked row's count is a fact the harness wrote, or the run fails closed


@pytest.mark.parametrize("forged", [-1, "0", 1.5, True, None, 2**53, [1]])
def test_a_garbage_steps_used_on_the_parked_row_fails_closed(forged: object) -> None:
    world = World(lead_steps=8)
    parked = world.runtime.send(world.root(), "Go")
    rewrite_paused_row(world.root(), lambda row: row.update(harness_steps_used=forged))
    record_approval(world.fresh().root(), parked.pending_approvals[0].id, True)

    response = world.fresh().runtime.send(world.root(), "")

    assert response.finish_reason == "budget_exhausted"
    assert world.ran == ["echo"]  # acted on, with no model step
    assert len(world.requests) == 1


def test_a_parked_row_writes_the_steps_used_and_left() -> None:
    world = World(lead_steps=8)
    world.calls(("deep", {"task": "Dig."}))
    for q in ("a", "b", "c"):
        world.calls(("lookup", {"q": q}))
    world.calls(("echo", {"n": 0}))
    world.runtime.send(world.root(), "Go")
    row = paused_approval_row([entry.message for entry in world.root().thread().messages()])
    assert row is not None
    assert row["harness_steps_used"] == 5
    assert row["harness_steps_left"] == 3


def test_a_forged_low_steps_used_on_a_child_is_still_bounded_by_its_tree() -> None:
    world = World(lead_steps=4)
    world.calls(("big", {"task": "B"}))  # root step 1
    world.calls(("echo", {"who": "big"}))  # big parks: 2 of the tree's 4
    world.text("Root done.")  # root step 2: 3 of 4
    world.runtime.send(world.root(), "Go")
    rewrite_paused_row(world.child("big"), lambda row: row.update(harness_steps_used=0))

    approve_and_resume(world, lambda: world.child("big"), 6)

    # The forged count cannot reach past what the tree has left: one step.
    assert len(world.requests) == 4


def test_a_legacy_root_row_without_steps_used_falls_back_to_its_transcript() -> None:
    world = World(lead_steps=3)
    world.runtime.send(world.root(), "Go")

    def legacy(row: dict[str, Any]) -> None:
        row.pop("harness_steps_used", None)
        row.pop("harness_steps_left", None)

    for _ in range(6):
        waiting = world.fresh().pending(world.root())
        if not waiting:
            break
        rewrite_paused_row(world.root(), legacy)
        record_approval(world.root(), waiting[0], True)
        if world.fresh().runtime.send(world.root(), "").finish_reason != "awaiting_approval":
            break

    assert len(world.requests) == 3


def test_a_structured_run_parks_and_resumes_within_its_ceiling() -> None:
    world = World(lead_steps=3)
    response: AgentResponse = world.runtime.send_structured(world.root(), "Go", SCHEMA)
    for _ in range(6):
        assert isinstance(response, StructuredAgentResponse)
        if response.finish_reason != "awaiting_approval":
            break
        record_approval(world.fresh().root(), response.pending_approvals[0].id, True)
        response = world.fresh().runtime.send_structured(world.root(), "", SCHEMA)

    assert isinstance(response, StructuredAgentResponse)
    assert response.finish_reason == "budget_exhausted"
    assert response.structured is None
    assert world.ran == ["echo", "echo", "echo"]
    assert len(world.requests) == 3


def test_a_child_row_without_steps_used_fails_closed() -> None:
    world = World()
    world.run_child_in_runner("big")  # saved ceiling 4: its rows always carry the count
    rewrite_paused_row(world.child("big"), lambda row: row.pop("harness_steps_used"))

    approve_and_resume(world, lambda: world.child("big"), 6)

    # Not read as a laxer legacy measure: no model step, the approval acted on.
    assert len(world.requests) == 1
    assert world.ran == ["echo"]


def test_an_approved_subagent_call_on_a_spent_run_starts_no_model_step() -> None:
    world = World(lead_steps=1)
    world.calls(("gated", {"task": "Dig."}))  # the root's only step: a gated subagent
    parked = world.runtime.send(world.root(), "Go")
    assert parked.finish_reason == "awaiting_approval"
    record_approval(world.fresh().root(), parked.pending_approvals[0].id, True)

    response = world.fresh().runtime.send(world.root(), "")

    # Acted on: the runner refuses the child on the turn's spent allowance,
    # before it can call a model.
    assert response.finish_reason == "budget_exhausted"
    assert len(world.requests) == 1
    assert world.ran == []
