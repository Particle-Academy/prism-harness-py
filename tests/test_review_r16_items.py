"""G-387 round 17: the round-16 review findings, at the public seam.

Every test enters where a real caller enters (``AgentRuntime.send``,
``record_approval``, ``Session.thread().record``, an event listener) with a
scripted model and the network refused. Interleavings are forced by a hook
inside the running tool or around a store write, never by timing.

1. Every SENT provider request answers every call of every step it holds,
   exactly once, in that step's own result run (directly after the step),
   whatever was written while the call ran, for approved AND ungated calls.
2. One trust signal: the nested-park stop takes a step's owner only from a
   row the harness wrote, never from a caller-passed run id; the Python-only
   run-state keys on a host step change nothing; a key named after a mark
   inside a tool's arguments or result is never a mark.
3. Nested parks and ungated outcomes: a run never reports an approval that
   is no longer pending; an ungated call replaced before it started never
   runs and keeps its one outcome; a failed started-row write leaves every
   call one honest outcome.
4. A result the replay synthesises carries its call's own provider ids.
5. The readers' index never goes stale on a list edited in place.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_concurrent_supersede import Hooked, approved_and_parked, outcomes
from test_repeated_call_ids import park, say

import prism_harness.runtime as runtime_module
from prism_harness import HarnessError, HarnessEvents, record_approval
from prism_harness.events import ToolApprovalAsked
from prism_harness.stores.file import FileSessionStore
from prism_harness.thread_rows import (
    INTERRUPTED_RESULT,
    START_DENIED,
    STARTED,
    SUPERSEDED_RESULT,
    acted_at,
    pending_approval_ids,
    placed_view,
    superseded_approval_ids,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


def _key(call_id: object, name: object, args: object) -> str:
    """The key a provider pairs a result to its call by: the call id."""
    del name, args
    return repr(call_id)


def misplaced(messages: list[dict[str, Any]]) -> list[str]:
    """Every way a provider request breaks the one rule real providers hold
    it to (Anthropic: the tool_result blocks directly after the tool_use;
    OpenAI: the tool messages directly after the assistant's tool_calls):
    each call of a step answered EXACTLY ONCE in the run of tool result rows
    directly after it. Empty means the request is valid."""
    problems: list[str] = []
    index = 0
    while index < len(messages):
        row = messages[index]
        index += 1
        if row.get("type") != "assistant" or not row.get("tool_calls"):
            continue
        needed: dict[str, int] = {}
        for call in row["tool_calls"]:
            key = _key(call.get("id"), call.get("name"), call.get("arguments"))
            needed[key] = needed.get(key, 0) + 1
        got: dict[str, int] = {}
        while index < len(messages) and messages[index].get("type") == "tool_result":
            for entry in messages[index].get("tool_results", []):
                key = _key(entry.get("tool_call_id"), entry.get("tool_name"), entry.get("args"))
                got[key] = got.get(key, 0) + 1
            index += 1
        for key, count in needed.items():
            if got.get(key, 0) != count:
                problems.append(f"{key} answered {got.get(key, 0)}x in its own run")
        for key in got:
            if key not in needed:
                problems.append(f"{key} answers no call of its step")
    return problems


class Both(Hooked):
    """The Hooked world whose ungated ``lookup`` also runs ``during``."""

    def _runtime(self) -> Any:
        runtime = super()._runtime()
        world = self

        class Lookup:
            name = "lookup"

            def handle(self, args: dict[str, Any]) -> Any:
                world.ran_with.append(("lookup", dict(args)))
                hook, world.during = world.during, None
                if hook is not None:
                    hook()
                return f"lookup:{args.get('n')}"

        runtime._tools.register(Lookup())
        return runtime


USER = {"type": "user", "content": "Host row."}
TEXT = {
    "type": "assistant",
    "content": "Host text.",
    "tool_calls": [],
    "additional_content": {},
    "tool_approval_requests": [],
}
SYSTEM = {"type": "system", "content": "A host note."}

CONCURRENT = ("user", "text", "user_text", "system", "nested_plain", "nested_park")


def _concurrent(world: Both, action: str) -> None:
    """``action``, done WHILE the call runs, from inside the tool."""
    if action in ("nested_plain", "nested_park"):
        if action == "nested_plain":
            world.text("Inner answer.")
        else:
            say(world, ("h9", "echo", {"n": 9}))
        world.queue.insert(0, world.queue.pop())  # the nested send's model step
        world.runtime.send(world.root(), "Inner message.")
        return
    rows: dict[str, list[dict[str, Any]]] = {
        "user": [USER],
        "text": [TEXT],
        "user_text": [USER, TEXT],
        "system": [SYSTEM],
    }
    world.root().thread().record([dict(row) for row in rows[action]])


@pytest.mark.parametrize("gating", ["approved", "ungated"])
@pytest.mark.parametrize("action", CONCURRENT)
def test_every_sent_request_answers_every_call_in_its_own_run(gating: str, action: str) -> None:
    """Item 1: a row written while the call runs (a host user row, a host
    text row, both, a system row, a nested send answered with plain text, a
    nested park) never leaves a call's result after it: every request a
    provider is SENT, now and on every later turn, and the stored rows as a
    provider would be sent them, answer each call once in its own run."""
    world = Both()
    world.during = lambda: _concurrent(world, action)
    try:
        if gating == "approved":
            approved_and_parked(world)
            world.text("Done.")
            world.fresh().runtime.send(world.root(), "")
        else:
            say(world, ("c1", "lookup", {"n": 1}))
            world.text("Done.")
            world.runtime.send(world.root(), "Go")
    except HarnessError as error:
        # Only the nested park stops the outer run (parked_elsewhere).
        assert action == "nested_park" and "parked_elsewhere" in str(error)
    world.text("Later answer.")
    world.fresh().runtime.send(world.root(), "Later.")
    assert world.requests
    for number, request in enumerate(world.requests):
        assert misplaced(list(request.messages)) == [], (number, request.messages)
    assert misplaced(placed_view(world.fresh().root().thread().rows())) == []
    ran = "echo" if gating == "approved" else "lookup"
    assert outcomes(world.root(), 1) == [f"{ran}:1"]


# -- 2. one trust signal -----------------------------------------------------------


def _outer_run(world: Both) -> str | None:
    return runtime_module.current_turn_run_id(world.root().key())


@pytest.mark.parametrize("which_run", ["none", "outer", "other"])
@pytest.mark.parametrize("gating", ["approved", "ungated"])
def test_a_host_step_mid_run_stops_the_run_whatever_run_id_it_carries(
    which_run: str, gating: str
) -> None:
    """Item 2 (Py run21 F1): while a call runs, the tool records a NEW gated
    step on its own session, under no run id, the RUNNING turn's own id, or
    another's. A run id never says who wrote a row: in all three the outer
    run stops before another model step (parked_elsewhere), the new step
    stays pending and answerable, and no request leaves a call unanswered."""
    world = Both()
    world.config = {"lock_wait": 0}
    world.fresh()
    seen: dict[str, Any] = {}

    def during() -> None:
        outer = _outer_run(world)
        seen["outer"] = outer
        run = {"none": None, "outer": outer, "other": "run_other"}[which_run]
        world.root().thread().record(
            [
                {
                    "type": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "h9", "name": "echo", "arguments": {"n": 9}}],
                    "tool_approval_requests": [{"approval_id": "apr_h9", "tool_call_id": "h9"}],
                }
            ],
            run,
        )

    if gating == "approved":
        approved_and_parked(world)
    else:
        say(world, ("c1", "lookup", {"n": 1}))
    world.during = during
    world.text("Done.")
    before = len(world.requests)
    with pytest.raises(HarnessError) as stopped:
        if gating == "approved":
            world.fresh().runtime.send(world.root(), "")
        else:
            world.runtime.send(world.root(), "Go")
    assert "parked_elsewhere" in str(stopped.value)
    assert seen["outer"] is not None
    # The outer run asked the provider nothing after the host's step.
    assert len(world.requests) == before + (0 if gating == "approved" else 1)
    rows = world.fresh().root().thread().rows()
    assert pending_approval_ids(rows) == ["apr_h9"]
    assert outcomes(world.root(), 9) == []
    record_approval(world.fresh().root(), "apr_h9", True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert outcomes(world.root(), 9) == ["echo:9"]
    for request in world.requests:
        assert misplaced(list(request.messages)) == []


MARK_NAMES = (
    "harness",
    "harness_replaced",
    "harness_replaced_step",
    "harness_superseded",
    "harness_ended_by_start",
    "harness_started",
    "harness_answers_step",
)


@pytest.mark.parametrize("name", MARK_NAMES)
def test_a_key_named_after_a_mark_inside_a_tool_call_is_never_a_mark(name: str) -> None:
    """Item 2 (PHP run19 #2; Python has no migration and no backfill, so this
    proves the shape absent): a call whose ARGUMENTS and RESULT carry a key
    named after each mark, with a value naming the call's own approval, runs
    once, reads approved, and nothing is replaced or ended."""
    world = Hooked()
    world.queue.append(
        runtime_module.LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[runtime_module.LlmToolCall("c1", "echo", {"n": 1, name: True})],
        )
    )
    parked = world.runtime.send(world.root(), "Go")
    [approval] = [p.id for p in parked.pending_approvals]
    world.root().thread().record(
        [
            {
                "type": "tool_result",
                "tool_results": [
                    {
                        "tool_call_id": "other",
                        "tool_name": "lookup",
                        "args": {name: {"approval_ids": [approval]}},
                        "result": f'{{"{name}": {{"approval_ids": ["{approval}"]}}}}',
                        name: {"approval_ids": [approval], "position": 2},
                    }
                ],
                "tool_approval_responses": [],
            }
        ]
    )
    rows = world.fresh().root().thread().rows()
    assert pending_approval_ids(rows) == [approval]
    record_approval(world.fresh().root(), approval, True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == [("echo", {"n": 1, name: True})]
    assert superseded_approval_ids(world.fresh().root().thread().rows()) == set()


HOST_KEYS: dict[str, Any] = {
    "harness_tool_names": ["echo", "lookup", "worker"],
    "harness_binding": {
        "session": {"provider": "custom", "model": "forged-model"},
        "declared": {"provider": "custom", "model": "forged-model"},
        "parent_mode": {"provider": "custom", "model": "forged-model"},
    },
    "harness_tree": {"session": "session:forged", "run": "run_forged"},
    "harness_plan_ledger": {"session": "session:forged", "plan": "plan_forged"},
    "harness_agent_spend": {"cost_usd": 999.0, "unmetered_runs": 5},
    "harness_steps_used": 7,
}


def _host_step_resume(key: str | None, which_run: str, flagged: bool) -> dict[str, Any]:
    world = Hooked()
    world.text("Hi.")
    world.runtime.send(world.root(), "Go")
    prior = next(e.run_id for e in world.root().thread().messages() if e.run_id is not None)
    run = {"none": None, "own": prior, "other": "run_other"}[which_run]
    step: dict[str, Any] = {
        "type": "assistant",
        "content": "",
        "tool_calls": [{"id": "h1", "name": "echo", "arguments": {"n": 1}}],
        "tool_approval_requests": [{"approval_id": "apr_host", "tool_call_id": "h1"}],
    }
    if key is not None:
        step[key] = HOST_KEYS[key]
    world.root().thread().record([step], run)
    if flagged:
        # The sensitivity control only: the same row as if the harness wrote it.
        store = FileSessionStore(world.directory)
        thread_key = world.root().thread().key
        doc = store.get(thread_key)
        assert doc is not None
        doc["messages"][-1]["harness"] = True
        store.put(thread_key, doc)
    record_approval(world.fresh().root(), "apr_host", True)
    world.text("Done.")
    before = len(world.requests)
    found: dict[str, Any] = {}
    try:
        found["finish"] = world.fresh().runtime.send(world.root(), "").finish_reason
    except HarnessError as error:
        found["finish"] = error.code
    new = world.requests[before:]
    found["tools"] = [sorted(tool.name for tool in request.tools) for request in new]
    found["route"] = [(request.provider, request.model) for request in new]
    found["ran"] = list(world.ran_with)
    # What the runtime's own next parked row would carry on: nothing forged.
    found["laundered"] = [
        e.message.get(key)
        for e in world.fresh().root().thread().messages()
        if key is not None and e.written_by_harness and e.message.get(key) == HOST_KEYS[key]
    ]
    return found


@pytest.mark.parametrize("which_run", ["none", "own", "other"])
@pytest.mark.parametrize("key", list(HOST_KEYS))
def test_a_run_state_key_on_a_host_step_changes_nothing(key: str, which_run: str) -> None:
    """Item 2 (TS run20 F2, confirmed for Python): each run-state key on a
    HOST step, with no run id, the session's prior run id, or another's,
    resumes exactly as the same step without it (the tools offered, the
    provider and model, what ran, the finish), and is laundered onto no row
    the harness writes. The flagged control shows the key would act."""
    control = _host_step_resume(None, which_run, False)
    host = _host_step_resume(key, which_run, False)
    assert host == {**control, "laundered": []}
    if key == "harness_tool_names":
        forced = _host_step_resume(key, which_run, True)
        assert forced["tools"] != control["tools"]


# -- 3. nested parks and ungated outcomes --------------------------------------------


def test_a_nested_park_from_an_ungated_call_beside_a_gated_one_reports_no_dead_approval() -> None:
    """Item 3 (TS run20 F3a, Py run21 F3): one step holds an ungated call and
    a gated one; the ungated call's own nested send parks a NEWER step. The
    outer run never reports the gated call's approval (the newer step ended
    it): it stops with parked_elsewhere, and the newer step stays pending
    and answerable."""
    world = Both()
    world.config = {"lock_wait": 0}
    world.fresh()
    inner: dict[str, Any] = {}

    def during() -> None:
        say(world, ("h9", "echo", {"n": 9}))
        world.queue.insert(0, world.queue.pop())
        parked = world.runtime.send(world.root(), "Inner.")
        inner["pending"] = [p.id for p in parked.pending_approvals]

    world.during = during
    say(world, ("u1", "lookup", {"n": 1}), ("c2", "echo", {"n": 2}))
    with pytest.raises(HarnessError) as stopped:
        world.runtime.send(world.root(), "Go")
    assert "parked_elsewhere" in str(stopped.value)
    rows = world.fresh().root().thread().rows()
    assert pending_approval_ids(rows) == inner["pending"]
    assert outcomes(world.root(), 1) == ["lookup:1"]
    assert len(outcomes(world.root(), 2)) == 1
    record_approval(world.fresh().root(), inner["pending"][0], True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert outcomes(world.root(), 9) == ["echo:9"]
    for request in world.requests:
        assert misplaced(list(request.messages)) == []


@pytest.mark.parametrize("row", ["user", "assistant_text"])
def test_an_ungated_call_replaced_before_it_started_never_runs(row: str) -> None:
    """Item 3 (Py run21 F2): a listener records a row when the gated call's
    approval is asked, before the step's ungated calls start; that row
    replaces the step. The ungated call keeps its ONE outcome (the replaced
    text) and never runs, and the run reports no approval that is no longer
    pending."""
    world = Hooked()
    bus = HarnessEvents()
    fired: list[int] = []

    def listener(event: object) -> None:
        if isinstance(event, ToolApprovalAsked) and not fired:
            fired.append(1)
            message = (
                {"type": "user", "content": "notified"}
                if row == "user"
                else {
                    "type": "assistant",
                    "content": "Waiting for your approval.",
                    "tool_calls": [],
                }
            )
            world.root().thread().record([message])

    bus.listen(listener)
    world.runtime._events = bus
    say(world, ("u1", "lookup", {"n": 5}), ("c1", "echo", {"n": 1}))
    try:
        response = world.runtime.send(world.root(), "Go")
        reported = [p.id for p in response.pending_approvals]
    except HarnessError as error:
        assert "parked_elsewhere" not in str(error)
        reported = []
    rows = world.fresh().root().thread().rows()
    assert set(reported) <= set(pending_approval_ids(rows))
    assert world.ran_with == []
    assert outcomes(world.root(), 5) == [SUPERSEDED_RESULT]
    assert outcomes(world.root(), 1) == [SUPERSEDED_RESULT]
    world.text("Fine.")
    world.fresh().runtime.send(world.root(), "Next.")
    for request in world.requests:
        assert misplaced(list(request.messages)) == []


def _failing_started_write(monkeypatch: pytest.MonkeyPatch, writes: int = 1) -> None:
    """The ungated batch's started-row write fails, once; with ``writes`` 2,
    the next write of the turn (its not-started answer) fails too."""
    original = FileSessionStore.put
    state = {"left": writes, "failing": False}

    def put(self: FileSessionStore, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if state["left"] and key.endswith(":thread") and isinstance(value, dict):
            last = (value.get("messages") or [{}])[-1].get("message", {})
            started = (
                isinstance(last.get(STARTED), dict) and last[STARTED].get("approval_id") is None
            )
            if started or state["failing"]:
                state["left"] -= 1
                state["failing"] = True
                raise OSError("disk full (test)")
        return original(self, key, value, *args, **kwargs)

    monkeypatch.setattr(FileSessionStore, "put", put)


@pytest.mark.parametrize("writes", [1, 2], ids=["started_write", "and_the_answer"])
@pytest.mark.parametrize("then", ["approve", "new_message"])
def test_a_failed_ungated_started_write_leaves_one_honest_outcome(
    monkeypatch: pytest.MonkeyPatch, then: str, writes: int
) -> None:
    """Item 3 (TS run20 F3b, also Python): the started-row write for the
    step's ungated call fails, so it never runs. It gets ONE honest outcome,
    never "interrupted" (it did not start) and never a result it did not
    produce, and every later provider request answers every call."""
    world = Hooked()
    say(world, ("u1", "lookup", {"n": 5}), ("c1", "echo", {"n": 1}))
    _failing_started_write(monkeypatch, writes)
    with pytest.raises(OSError):
        world.runtime.send(world.root(), "Go")
    assert world.ran_with == []
    if then == "approve":
        for approval in pending_approval_ids(world.fresh().root().thread().rows()):
            record_approval(world.fresh().root(), approval, True)
        world.text("Again.")
        world.fresh().runtime.send(world.root(), "")
        assert outcomes(world.root(), 1) == ["echo:1"]
    else:
        world.text("Fine.")
        world.fresh().runtime.send(world.root(), "Something else.")
    assert ("lookup", {"n": 5}) not in world.ran_with
    [outcome] = outcomes(world.root(), 5)
    assert outcome not in (INTERRUPTED_RESULT, "lookup:5")
    world.text("Later.")
    world.fresh().runtime.send(world.root(), "Later.")
    for request in world.requests:
        assert misplaced(list(request.messages)) == []


# -- 4. synthesised results carry their call's own ids -------------------------------


def test_a_synthesised_result_carries_its_calls_own_provider_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 4 (PHP run19 #4; Python has no journal): every result the replay
    gives a call that has none (interrupted, replaced) carries the call's own
    id and its own provider result id, never None for a call that has one,
    so each provider map (OpenAI chat and responses, Anthropic) is sent the
    id its own tool call carries."""
    world = Hooked()
    world.queue.append(
        runtime_module.LlmResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=[
                runtime_module.LlmToolCall("call_u1", "lookup", {"n": 5}, result_id="fc_u1"),
                runtime_module.LlmToolCall("toolu_c1", "echo", {"n": 1}, result_id="fc_c1"),
            ],
        )
    )
    original = FileSessionStore.put
    state = {"armed": True}

    def put(self: FileSessionStore, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if state["armed"] and key.endswith(":thread") and isinstance(value, dict):
            last = (value.get("messages") or [{}])[-1].get("message", {})
            if last.get("type") == "tool_result" and any(
                entry.get("tool_name") == "lookup" and entry.get("result") == "lookup:5"
                for entry in last.get("tool_results", [])
            ):
                state["armed"] = False
                raise OSError("disk full (test)")
        return original(self, key, value, *args, **kwargs)

    monkeypatch.setattr(FileSessionStore, "put", put)
    with pytest.raises(OSError):
        world.runtime.send(world.root(), "Go")
    world.text("Fine.")
    world.fresh().runtime.send(world.root(), "Something else.")
    sent = world.requests[-1].messages
    step = next(m for m in sent if m["type"] == "assistant" and m["tool_calls"])
    ids = {call["id"]: call["result_id"] for call in step["tool_calls"]}
    answered = {
        entry["tool_call_id"]: entry["tool_call_result_id"]
        for m in sent
        if m["type"] == "tool_result"
        for entry in m["tool_results"]
    }
    assert answered == ids == {"call_u1": "fc_u1", "toolu_c1": "fc_c1"}
    assert misplaced(list(sent)) == []


# -- 5. the readers' index ------------------------------------------------------------


def test_the_readers_never_read_a_stale_index_after_an_in_place_edit() -> None:
    """Item 5 (Py run21 I1): a list a host edits IN PLACE at the same length
    reads exactly as a fresh copy of it does, through every reader."""
    world = Hooked()
    _, [approval] = park(world, ("c1", "echo", {"n": 1}))
    rows = world.fresh().root().thread().rows()
    rows.append({"type": "user", "content": "filler"})
    assert pending_approval_ids(rows) == pending_approval_ids(list(rows))
    assert acted_at(rows) == {}
    rows[-1] = {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": "c1",
                "tool_name": "echo",
                "args": {"n": 1},
                "result": "host",
                "tool_call_result_id": None,
                "artifacts": [],
            }
        ],
        "tool_approval_responses": [],
    }
    fresh = list(rows)
    assert acted_at(rows) == acted_at(fresh) == {approval: len(rows) - 1}
    assert pending_approval_ids(rows) == pending_approval_ids(fresh)
    assert placed_view(rows) == placed_view(fresh)


def test_a_call_whose_start_a_standing_denial_refused_says_it_was_denied() -> None:
    """Item 5 (Py run21 I3): three approved calls; while c1 runs a host
    stores a raw denial of c3. c3 never runs, and its one outcome says it was
    denied before it started (the reference's text), not "not the one that
    was approved"."""
    world = Hooked()
    _, approvals = park(
        world, ("c1", "echo", {"n": 1}), ("c2", "echo", {"n": 2}), ("c3", "echo", {"n": 3})
    )
    for approval in approvals:
        record_approval(world.fresh().root(), approval, True)

    def deny() -> None:
        world.root().thread().record(
            [
                {
                    "type": "tool_result",
                    "tool_results": [],
                    "tool_approval_responses": [
                        {"approval_id": approvals[2], "approved": False, "reason": "raw"}
                    ],
                }
            ]
        )

    world.during = deny
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == [("echo", {"n": 1}), ("echo", {"n": 2})]
    assert outcomes(world.root(), 3) == [START_DENIED]
