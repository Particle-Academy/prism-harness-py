"""G-285: a durable, cross-process cancel.

Ports prism-harness's tests/DurableCancelTest.php, case for case, through the
public API with a scripted model; the network is refused before the first
request. The run executes in the "worker" process; the cancel comes from a
"web" process: its own ``PrismHarness``, its own ``FileSessionStore`` instances
and its own ``AgentRuntime``, over the same temporary directory, as a web
request stopping a queued run would. Nothing in memory is shared between the
two, and the web process reads the run id from the store, never from the
worker.
"""

from __future__ import annotations

import socket
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import prism_harness
from prism_harness import (
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
    SubagentRunner,
    ToolRegistry,
    record_approval,
)
from prism_harness.plans import Plans

REFUSAL = "Not run: the run was cancelled."
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}
WATCHED = ("run.cancel_requested", "run.cancelled", "run.finished", "run.failed")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


class Counted:
    def __init__(self, name: str, result: str, world: World) -> None:
        self._name, self._result, self._world = name, result, world

    @property
    def name(self) -> str:
        return self._name

    def handle(self, args: dict[str, Any]) -> Any:
        self._world.ran[self._name] += 1
        during = self._world.during.get(self._name)
        if during is not None:
            during()
        return self._result


class EndingStore(FileSessionStore):
    """The worker's file store, with a hook just before a run's end is written."""

    def __init__(self, directory: str, world: World) -> None:
        super().__init__(directory)
        self._world = world

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        run = payload.get("run")
        hook = self._world.before_end
        if (
            hook is not None
            and key.endswith(":ephemeral")
            and isinstance(run, dict)
            and run.get("status") == "completed"
        ):
            self._world.before_end = None
            hook()
        super().put(key, payload, ttl_seconds)


class World:
    """One durable backing (a temp directory), and the processes over it."""

    def __init__(self, *, tree: bool = False, plans: bool = False) -> None:
        self.directory = tempfile.mkdtemp(prefix="prism-harness-g285-")
        self.queue: list[LlmResponse] = []
        self.requests = 0
        self.on_request: Callable[[int], None] | None = None
        self.ran = {"search": 0, "weather": 0}
        self.during: dict[str, Callable[[], None]] = {}
        self.events: list[HarnessEvent] = []
        self.before_end: Callable[[], None] | None = None
        self._call = 0
        self._tree = tree
        self._plans = plans
        self.worker_runtime, self._worker_harness = self._process(worker=True)

    def _process(self, worker: bool = False) -> tuple[AgentRuntime, PrismHarness]:
        """A process: its own harness, stores, runtime and event listeners."""
        directory = self.directory
        harness = PrismHarness(
            drivers={
                "files": (lambda: EndingStore(directory, self))
                if worker
                else (lambda: FileSessionStore(directory))
            },
            stores={"ephemeral": "files", "durable": "files"},
        )
        events = HarnessEvents()
        events.listen(self._record)
        chat: dict[str, Any] = {
            "system_prompt": "Be brief.",
            "tools": ["search", "weather"],
            "max_steps": 8,
        }
        modes: dict[str, Any] = {
            "chat": chat,
            "guarded": {**chat, "requires_approval": ["weather"]},
        }
        tools = (
            ToolRegistry()
            .register(Counted("search", "Found it", self))
            .register(Counted("weather", "Sunny", self))
        )
        holder: list[AgentRuntime] = []
        if self._tree:
            chat["tools"] = ["run_op"]
            chat["subagents"] = {"run_op": {"mode": "op_runner", "max_steps": 3}}
            modes["op_runner"] = {
                "system_prompt": "You run one Op.",
                "tools": ["search"],
                "max_steps": 4,
            }

            def run_op(session: Session) -> Any:
                subagent = holder[0].mode_for(session).subagents["run_op"]
                parent = (session.run() or {}).get("id", "parent-run")
                return SubagentRunner(harness, holder[0]).tool(
                    subagent, session, RunContext.root(parent, RunBudget(8)), parent
                )

            tools.register_factory("run_op", run_op)
        if self._plans:
            chat["plans"] = True
        runtime = AgentRuntime(
            client=self._client,
            modes=ModeRegistry({"default": "chat", "modes": modes}),
            tools=tools,
            events=events,
            config={"provider": "openai", "model": "gpt-4o"},
            plans=Plans({"enabled": self._plans}, index_store=harness.durable_store())
            if self._plans
            else None,
        )
        holder.append(runtime)
        return runtime, harness

    def _record(self, event: HarnessEvent) -> None:
        if event.type in WATCHED:
            self.events.append(event)

    def _client(self, _request: LlmRequest) -> LlmResponse:
        self.requests += 1
        if self.on_request is not None:
            self.on_request(self.requests)
        if not self.queue:
            raise AssertionError("The scripted model has no response left.")
        return self.queue.pop(0)

    # -- scripting ---------------------------------------------------------

    def calls(self, *calls: tuple[str, dict[str, Any]]) -> None:
        tool_calls = []
        for name, arguments in calls:
            self._call += 1
            tool_calls.append(LlmToolCall(id=f"call_{self._call}", name=name, arguments=arguments))
        self.queue.append(LlmResponse(text="", finish_reason="tool_calls", tool_calls=tool_calls))

    def text(self, text: str) -> None:
        self.queue.append(LlmResponse(text=text, finish_reason="stop", structured={"ok": True}))

    # -- the two processes -------------------------------------------------

    def worker(self, name: str = "ada") -> Session:
        """The queue worker's session: the one that runs."""
        return self._worker_harness.for_(Participant("User", name)).session("chat")

    def web(self, name: str = "ada") -> tuple[AgentRuntime, Session]:
        """Another process for the same address: new stores, nothing cached."""
        runtime, harness = self._process()
        return runtime, harness.for_(Participant("User", name)).session("chat")

    def web_session(self, name: str = "ada") -> Session:
        return self.web(name)[1]

    def run_id(self, name: str = "ada") -> str:
        """The run id as another process reads it: from the store."""
        run = self.web_session(name).run()
        assert run is not None
        run_id = run.get("id")
        assert isinstance(run_id, str)
        return run_id

    def cancel(self, run_id: str, name: str = "ada") -> str:
        """Red by assertion on a build without G-285, not by a missing method."""
        runtime, session = self.web(name)
        assert hasattr(session, "cancel"), "Session.cancel() is G-285"
        assert hasattr(runtime, "cancel"), "AgentRuntime.cancel() is G-285"
        outcome = runtime.cancel(session, run_id)
        enum = getattr(prism_harness, "CancelOutcome", None)
        assert enum is not None, "prism_harness.CancelOutcome is G-285"
        assert isinstance(outcome, enum)
        return str(outcome.value)

    def turn(self, path: str, prompt: str = "go", name: str = "ada") -> HarnessError | None:
        """Run the worker's turn on a path, returning the error it ended with."""
        try:
            if path == "send":
                self.worker_runtime.send(self.worker(name), prompt)
            else:
                self.worker_runtime.send_structured(self.worker(name), prompt, SCHEMA)
        except HarnessError as error:
            return error
        return None

    def names(self) -> list[str]:
        return [event.type for event in self.events]

    def thread_text(self, name: str = "ada") -> str:
        return repr([entry.message for entry in self.web_session(name).thread().messages()])

    def status(self, name: str = "ada") -> object:
        run = self.web_session(name).run()
        return None if run is None else run.get("status")


def approval_status(session: Session, approval_id: str) -> str:
    assert hasattr(session, "approval_status"), "Session.approval_status() is G-285"
    status = session.approval_status(approval_id)
    assert isinstance(status, str)
    return status


PATHS = ["send", "send_structured"]


@pytest.mark.parametrize("path", PATHS)
def test_stops_a_run_from_another_process_at_the_next_tool_and_lets_the_running_tool_finish(
    path: str,
) -> None:
    world = World()
    outcome: list[str] = []
    world.during["search"] = lambda: outcome.append(world.cancel(world.run_id()))
    world.calls(("search", {"query": "tigers"}), ("weather", {"city": "Detroit"}))
    world.text("This step must never be asked for.")

    refused = world.turn(path)

    assert outcome == ["ok"]
    assert refused is not None and refused.code == "run_cancelled"
    # The in-flight call finished; the next call in the same step did not run.
    assert world.ran == {"search": 1, "weather": 0}
    # And no next model step was requested.
    assert world.requests == 1
    assert world.status() == "cancelled"
    assert REFUSAL in world.thread_text()
    assert world.names() == ["run.cancel_requested", "run.cancelled"]
    requested, cancelled = world.events
    web = world.web_session()
    run = web.run()
    assert run is not None
    assert getattr(requested, "run_id", None) == getattr(cancelled, "run_id", None) == run["id"]
    assert getattr(requested, "root_run_id", None) == run["id"]
    assert getattr(requested, "parent_run_id", "unset") is None
    assert getattr(requested, "session_key", None) == web.key()
    assert getattr(cancelled, "session_key", None) == web.key()
    assert getattr(requested, "state", None) == "running"


@pytest.mark.parametrize("path", PATHS)
def test_stops_between_steps_when_the_cancel_lands_while_the_model_is_thinking(
    path: str,
) -> None:
    world = World()
    outcome: list[str] = []

    def on_request(request: int) -> None:
        if request == 1:
            outcome.append(world.cancel(world.run_id()))

    world.on_request = on_request
    world.calls(("search", {"query": "tigers"}), ("weather", {"city": "Detroit"}))
    world.text("This step must never be asked for.")

    refused = world.turn(path)

    assert outcome == ["ok"]
    assert refused is not None and refused.code == "run_cancelled"
    assert world.ran == {"search": 0, "weather": 0}
    assert world.requests == 1
    assert world.status() == "cancelled"
    assert world.names() == ["run.cancel_requested", "run.cancelled"]


@pytest.mark.parametrize("path", PATHS)
def test_ends_a_run_cancelled_when_the_cancel_races_its_last_step(path: str) -> None:
    world = World()
    outcome: list[str] = []

    def on_request(request: int) -> None:
        if request == 2:
            outcome.append(world.cancel(world.run_id()))

    world.on_request = on_request
    world.calls(("search", {"query": "tigers"}))
    world.text("All done.")

    refused = world.turn(path)

    # The last step had already been asked for, so its answer is kept; but a
    # cancel that returned ok is never followed by a run that reads completed.
    assert outcome == ["ok"]
    assert refused is not None and refused.code == "run_cancelled"
    assert world.status() == "cancelled"
    # The STORED record says so, not only the read of it.
    stored = world.web_session().state()["run"]
    assert stored["status"] == "cancelled"
    assert stored["ended_as"] == "completed"
    assert "All done." in world.thread_text()
    assert world.requests == 2
    assert world.names() == ["run.cancel_requested", "run.cancelled"]


def test_a_cancel_racing_the_terminal_write_never_returns_ok_for_a_completed_run() -> None:
    # The web process cancels at the moment the worker is writing the run's
    # end, from another thread. The cancel lock serialises the two: either the
    # cancel lands first and the run ends cancelled, or the run ends first and
    # the cancel is not-running. Never ok for a run that reads completed.
    world = World()
    outcome: list[str] = []
    threads: list[threading.Thread] = []

    def before_end() -> None:
        run_id = world.run_id()
        thread = threading.Thread(target=lambda: outcome.append(world.cancel(run_id)))
        thread.start()
        threads.append(thread)
        # Long enough for an unserialised cancel to read "running" and write.
        thread.join(1.0)

    world.before_end = before_end
    world.text("Done.")

    world.turn("send")
    for thread in threads:
        thread.join(20)

    assert len(outcome) == 1
    assert (outcome[0], world.status()) in {("ok", "cancelled"), ("not-running", "completed")}


def test_refuses_to_cancel_another_sessions_run_and_that_run_is_untouched() -> None:
    world = World()
    # Bob has a session, and a run of his own, so an id that is not his reads
    # as what it is: not one of his runs.
    world.text("Bob is done.")
    assert world.turn("send", "mine", "bob") is None
    world.events.clear()
    outcome: list[str] = []

    def during() -> None:
        outcome.append(world.cancel(world.run_id("ada"), "bob"))
        # A session with no run at all touches nothing either.
        outcome.append(world.cancel(world.run_id("ada"), "cy"))

    world.during["search"] = during
    world.calls(("search", {"query": "tigers"}), ("weather", {"city": "Detroit"}))
    world.text("Both done.")

    assert world.turn("send") is None
    assert outcome == ["refused", "no-turn"]
    assert world.ran == {"search": 1, "weather": 1}
    assert world.status("ada") == "completed"
    assert world.status("bob") == "completed"
    assert world.names() == ["run.finished"]


def test_refuses_a_forged_or_malformed_run_id_whatever_the_session_holds() -> None:
    world = World()
    outcome: list[str] = []

    def during() -> None:
        real = world.run_id()
        for forged in [
            str(uuid.uuid4()),
            "",
            real[:9],
            "../" + real,
            real.upper(),
            real + "\n",
            real + " ",
            " " + real,
        ]:
            outcome.append(world.cancel(forged))

    world.during["search"] = during
    world.calls(("search", {"query": "tigers"}))
    world.text("Done.")

    assert world.turn("send") is None
    assert set(outcome) == {"refused"}
    assert len(outcome) == 8
    assert world.status() == "completed"
    assert world.names() == ["run.finished"]


def test_answers_no_turn_not_running_and_unsupported_without_changing_anything() -> None:
    world = World()
    assert world.cancel(str(uuid.uuid4())) == "no-turn"

    world.text("First.")
    world.text("Second.")
    assert world.turn("send", "one") is None
    first = world.run_id()
    assert world.turn("send", "two") is None
    second = world.run_id()

    assert world.cancel(second) == "not-running"
    # An earlier run of this session is its own, and over.
    assert world.cancel(first) == "not-running"
    # A process-local lock cannot hold a cancel another process can see.
    local = Session(
        Participant("User", "ada"),
        "chat",
        MemorySessionStore(),
        FileSessionStore(world.directory),
    )
    assert hasattr(local, "cancel"), "Session.cancel() is G-285"
    assert local.cancel(second).value == "unsupported"
    assert world.status() == "completed"
    assert world.names() == ["run.finished", "run.finished"]


def test_cancels_twice_harmlessly_one_request_event_one_end_and_then_not_running() -> None:
    world = World()
    outcome: list[str] = []

    def during() -> None:
        outcome.append(world.cancel(world.run_id()))
        outcome.append(world.cancel(world.run_id()))

    world.during["search"] = during
    world.calls(("search", {"query": "tigers"}))
    world.text("Never.")

    refused = world.turn("send")
    assert refused is not None and refused.code == "run_cancelled"
    assert outcome == ["ok", "ok"]
    assert world.cancel(world.run_id()) == "not-running"
    assert world.names() == ["run.cancel_requested", "run.cancelled"]


def test_settles_a_run_parked_on_an_approval_as_cancelled_and_it_never_resumes() -> None:
    world = World()
    world.worker().using_mode("guarded")
    world.calls(("search", {"query": "tigers"}), ("weather", {"city": "Detroit"}))
    pending = world.worker_runtime.send(world.worker(), "go").pending_approvals[0]
    run_id = world.run_id()
    assert approval_status(world.web_session(), pending.id) == "pending"
    assert approval_status(world.web_session(), "apr_" + uuid.uuid4().hex) == "unknown"
    world.events.clear()

    assert world.cancel(run_id) == "ok"
    assert world.names() == ["run.cancel_requested", "run.cancelled"]
    assert getattr(world.events[0], "state", None) == "parked"
    assert getattr(world.events[1], "run_id", None) == run_id
    assert approval_status(world.web_session(), pending.id) == "cancelled"
    assert world.status() == "cancelled"
    assert world.cancel(run_id) == "not-running"

    requests = world.requests
    world.text("A resumed run must never be asked for.")
    tries: list[Callable[[], object]] = [
        lambda: record_approval(world.worker(), pending.id, True),
        lambda: record_approval(world.worker(), pending.id, False),
        lambda: world.worker_runtime.decide(world.worker(), pending.id, True),
        lambda: world.worker_runtime.decide(world.worker(), pending.id, False, "no"),
    ]
    for attempt in tries:
        with pytest.raises(HarnessError) as refused:
            attempt()
        assert refused.value.code == "approval_not_pending"
    for path in PATHS:
        resumed = world.turn(path, "")
        assert resumed is not None and resumed.code == "run_cancelled"
    assert world.requests == requests
    assert world.ran == {"search": 1, "weather": 0}
    assert approval_status(world.web_session(), pending.id) == "cancelled"
    assert world.names() == ["run.cancel_requested", "run.cancelled"]


def test_stops_the_whole_run_tree_when_the_root_is_cancelled_during_a_subagent() -> None:
    world = World(tree=True)
    outcome: list[str] = []
    # The person stops the ROOT run, from another process, while the child works.
    world.during["search"] = lambda: outcome.append(world.cancel(world.run_id()))
    world.calls(("run_op", {"task": "Run the billing Op"}))
    world.calls(("search", {"query": "billing"}))
    world.text("Never asked for.")

    refused = world.turn("send")

    assert refused is not None and refused.code == "run_cancelled"
    assert outcome == ["ok"]
    assert world.ran["search"] == 1
    # Parent step, child step; neither took another.
    assert world.requests == 2
    assert world.status() == "cancelled"
    assert world.names() == ["run.cancel_requested", "run.cancelled", "run.cancelled"]
    # The child ended cancelled too, under its own id, in the root's tree.
    child, root = world.events[1], world.events[2]
    assert getattr(child, "run_id", None) != getattr(root, "run_id", None)
    assert getattr(child, "root_run_id", None) == getattr(root, "run_id", None)
    assert getattr(child, "parent_run_id", None) == getattr(root, "run_id", None)
    assert "cancelled" in world.thread_text()


def test_blocks_a_plan_whose_step_turn_was_cancelled_and_the_next_advance_runs_nothing() -> None:
    world = World(plans=True)
    steps = [
        {"title": f"Step {i}", "instruction": f"Do part {i} of the work.", "owner": "agent"}
        for i in (1, 2)
    ]
    world.calls(("plan_draft", {"goal": "Plan the trip", "steps": steps}))
    pending = world.worker_runtime.send(world.worker(), "Plan the trip").pending_approvals[0]
    world.text("Plan approved.")
    world.worker_runtime.decide(world.worker(), pending.id, True)
    world.worker_runtime.send(world.worker(), "")
    current = world.worker_runtime.plan(world.worker()).current()
    assert current is not None and current.status == "active"

    outcome: list[str] = []
    asked = world.requests + 1

    def on_request(request: int) -> None:
        if request == asked:
            outcome.append(world.cancel(world.run_id()))

    world.on_request = on_request
    world.calls(("plan_complete_step", {"summary": "Booked the flight"}))
    world.text("Never asked for.")

    advanced = world.worker_runtime.plan(world.worker()).advance()
    snapshot = world.worker_runtime.plan(world.worker()).current()
    assert snapshot is not None
    plan = snapshot.to_dict()

    assert outcome == ["ok"]
    assert advanced.status == "blocked"
    assert advanced.reason == "run_cancelled"
    assert plan["status"] == "blocked"
    assert plan["blockedReason"] == "run_cancelled"
    assert plan["steps"][0]["state"] == "todo"
    assert world.requests == asked

    # Blocked plans do not advance: nothing runs until the host resumes it.
    again = world.worker_runtime.plan(world.worker()).advance()
    assert not again.ran()
    assert again.plan_status == "blocked"
    assert world.requests == asked


def test_keeps_the_tool_calls_it_guards_identical_in_every_other_respect() -> None:
    world = World()
    world.calls(("search", {"query": "tigers"}), ("weather", {"city": "Detroit"}))
    world.text("Both done.")

    assert world.turn("send") is None
    assert world.ran == {"search": 1, "weather": 1}
    rows = [entry.message for entry in world.web_session().thread().messages()]
    outputs = {
        result["tool_call_id"]: result["result"]
        for row in rows
        if row["type"] == "tool_result"
        for result in row["tool_results"]
    }
    assert outputs == {"call_1": "Found it", "call_2": "Sunny"}
    assert REFUSAL not in repr(rows)
    assert world.status() == "completed"
    assert world.names() == ["run.finished"]
