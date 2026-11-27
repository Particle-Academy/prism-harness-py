"""The round-14 review items that apply to every language (G-387 round 15).

- Z2 (PHP run17 #1): the ended-by-start channel is the runtime's alone. The
  public ``Thread.record`` takes no such argument, and a host row carrying
  the start's ended mark neither skips the replaced denials of a row that
  supersedes the step nor ends a call that ran.
- Z3 (PHP run17 #2): a thread with no session around it (no participant,
  no session lock) takes a superseding row as before, with its denial.
"""

from __future__ import annotations

import inspect
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_repeated_call_ids import Recording, park, rows

from prism_harness import record_approval
from prism_harness.stores.memory import MemorySessionStore
from prism_harness.thread import Thread
from prism_harness.thread_rows import (
    ENDED_BY_START,
    SUPERSEDED,
    SUPERSEDED_RESULT,
    pending_approval_ids,
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


def _outcomes(stored: list[dict[str, Any]], call_id: str = "c1") -> list[str]:
    return [
        entry["result"]
        for row in stored
        if row.get("type") == "tool_result"
        for entry in row.get("tool_results") or []
        if entry.get("tool_call_id") == call_id
    ]


# -- Z2: the ended-by-start channel is runtime-only ------------------------------


def test_z2_the_public_record_has_no_ended_by_start_channel() -> None:
    parameters = list(inspect.signature(Thread.record).parameters)
    assert parameters == ["self", "messages", "run_id"]
    thread = Thread(MemorySessionStore(), "k:thread")
    with pytest.raises(TypeError):
        thread.record([{"type": "user", "content": "x"}], None, ended_by_start=["a"])  # type: ignore[call-arg]
    assert thread.messages() == []


def test_z2_a_host_row_carrying_the_ended_mark_never_skips_the_replaced_denial() -> None:
    world = Recording()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    forged = {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [],
        ENDED_BY_START: {"approval_ids": approvals},
    }
    world.fresh().root().thread().record([forged, {"type": "user", "content": "Next."}])

    stored = rows(world.root())
    assert [row["type"] for row in stored][-3:] == ["tool_result", "tool_result", "user"]
    assert stored[-2].get(SUPERSEDED) == {"approval_ids": approvals}
    assert _outcomes(stored) == [SUPERSEDED_RESULT]
    assert pending_approval_ids(stored) == []
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == []
    assert _outcomes(rows(world.root())) == [SUPERSEDED_RESULT]


# -- Z3: a thread with no session takes rows as before ---------------------------


def test_z3_a_thread_with_no_session_takes_a_superseding_row_with_its_denial() -> None:
    thread = Thread(MemorySessionStore(), "shared:admin-review:thread")
    thread.record(
        [
            {"type": "user", "content": "Go"},
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"n": 1}}],
                "additional_content": {},
                "tool_approval_requests": [{"approval_id": "apr_1", "tool_call_id": "c1"}],
            },
        ]
    )
    recorded = thread.record([{"type": "user", "content": "Next."}])

    stored = [entry.message for entry in thread.messages()]
    assert [row["type"] for row in stored] == ["user", "assistant", "tool_result", "user"]
    assert [entry.message["type"] for entry in recorded] == ["tool_result", "user"]
    assert _outcomes(stored) == [SUPERSEDED_RESULT]
    assert superseded_approval_ids(stored) == {"apr_1"}


# -- PHP Info 6a: a parked structured turn carries no raw step documents ---------


def test_a_parked_structured_response_carries_no_raw_steps() -> None:
    """PHP's parked structured response still carried each raw provider step
    (with the model's unchecked document); Python's response has no step
    list at all: ``steps`` is a count, and the parked document is absent."""
    from dataclasses import fields

    from test_review_r13_php_items import SCHEMA, anthropic_step, runtime
    from test_structured import a_session, scripted

    from prism_harness import LlmToolCall, StructuredAgentResponse

    assert {field.name for field in fields(StructuredAgentResponse)} == {
        "run_id",
        "text",
        "steps",
        "tool_calls",
        "finish_reason",
        "pending_approvals",
        "stopped_because",
        "stopped_code",
        "structured",
    }
    session = a_session()
    response = runtime(scripted([anthropic_step(LlmToolCall("c1", "echo", {}))])).send_structured(
        session, "Answer.", SCHEMA
    )
    assert isinstance(response, StructuredAgentResponse)
    assert response.finish_reason == "awaiting_approval"
    assert response.structured is None
    assert type(response.steps) is int


# -- the PHP 0.26.10 audit's directed tests (a) and (d) ----------------------------


def test_a_a_denial_then_a_locked_resume_then_an_approve_never_runs_the_call() -> None:
    """(a) A DENIAL is stored; a resume is then refused ``session_locked``
    (another worker holds the session); then an APPROVE arrives. The call
    never runs and keeps its one outcome, the denial."""
    import threading

    from prism_harness import HarnessError

    world = Recording()
    world.config = {"lock_wait": 0}
    world.fresh()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], False, "no")
    holding, release = threading.Event(), threading.Event()

    def hold(_: Any) -> None:
        holding.set()
        release.wait(60)

    holder = threading.Thread(target=lambda: world.fresh().root().lock(hold, 60, 5), daemon=True)
    holder.start()
    assert holding.wait(60)
    with pytest.raises(HarnessError) as locked:
        world.fresh().runtime.send(world.root(), "")
    assert locked.value.code == "session_locked"
    release.set()
    holder.join(60)
    try:
        record_approval(world.fresh().root(), approvals[0], True)
    except HarnessError as error:
        assert error.code == "approval_not_pending"
    for _ in range(2):
        world.text("Done.")
        world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    assert _outcomes(rows(world.root())) == ["no"]


def test_d_an_approval_then_a_deny_before_the_turn_runs_is_denied() -> None:
    """(d) An APPROVAL is stored, the turn never runs, then a DENY arrives:
    any deny is final (G-401, the recommended option), so the call is denied,
    never runs, and has one outcome."""
    from prism_harness.plans.thread_facts import approval_decisions

    world = Recording()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    record_approval(world.fresh().root(), approvals[0], False, "changed my mind")
    assert approval_decisions(rows(world.root())) == {approvals[0]: False}
    for _ in range(2):
        world.text("Done.")
        world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    assert _outcomes(rows(world.root())) == ["changed my mind"]
    # And an approve after it is never recorded over it.
    try:
        record_approval(world.fresh().root(), approvals[0], True)
    except Exception as error:  # noqa: BLE001 - refused is as good as ignored
        assert getattr(error, "code", None) == "approval_not_pending"
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == []
