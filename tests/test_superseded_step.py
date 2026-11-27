"""DECISION B (G-387 round 13): a superseded parked step denies its open
gated calls, once.

A parked step is SUPERSEDED by any later assistant row with tool calls, or by
an assistant text row or a user row recorded OUTSIDE the runs that own it (the
run that recorded it, and every run that recorded one of its results): no run
id, or another run. Each of its calls with no result then gets ONE denial,
"Tool call was not run because a later step replaced it", on a tool result row marked
``harness_superseded``; nothing of it runs, and a later decision is refused.
The harness's own rows in the runs resolving it never supersede, and a host
that hands the step's own run id to the public ``Thread.record`` gains
nothing: its row still ends the step for every decision reader (G-397), and
nothing of the step runs.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_repeated_call_ids import Recording, park, rows

from prism_harness import HarnessError, Session, record_approval
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.thread_rows import (
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
    yield


SUPERSEDING: dict[str, dict[str, Any]] = {
    "assistant_text": {
        "type": "assistant",
        "content": "An aside.",
        "tool_calls": [],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    "assistant_step": {
        "type": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "later",
                "name": "echo",
                "arguments": {"n": 7},
                "result_id": None,
                "reasoning_id": None,
                "reasoning_summary": None,
            }
        ],
        "additional_content": {},
        "tool_approval_requests": [],
    },
    "user": {"type": "user", "content": "Something else."},
}

STEPS = {
    "unique": (("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}), ("c", "lookup", {"n": 3})),
    "repeated": (("dup", "echo", {"n": 1}), ("dup", "echo", {"n": 2})),
}


def parked_run_id(session: Session) -> str:
    for entry in reversed(session.thread().messages()):
        if entry.message.get("tool_approval_requests"):
            assert entry.run_id is not None
            return entry.run_id
    raise AssertionError("nothing parked")


def outcomes_of(session: Session, n: int) -> list[str]:
    return [
        entry["result"]
        for row in rows(session)
        if row.get("type") == "tool_result"
        for entry in row["tool_results"]
        if entry["args"].get("n") == n
    ]


def marks(session: Session) -> list[Any]:
    return [row[SUPERSEDED] for row in rows(session) if SUPERSEDED in row]


@pytest.mark.parametrize("step", sorted(STEPS))
@pytest.mark.parametrize("run", ["none", "another"])
@pytest.mark.parametrize("kind", sorted(SUPERSEDING))
def test_a_host_recorded_row_supersedes_the_parked_step(kind: str, run: str, step: str) -> None:
    world = Recording()
    _, approvals = park(world, *STEPS[step])
    # A person approves the first call while the step is answerable.
    record_approval(world.fresh().root(), approvals[0], True)
    host = world.fresh().root()
    run_id = {"none": None, "another": "host-run"}[run]

    host.thread().record([dict(SUPERSEDING[kind])], run_id)

    stored = rows(world.fresh().root())
    marked = [row for row in stored if SUPERSEDED in row]
    assert [row[SUPERSEDED] for row in marked] == [{"approval_ids": approvals}]
    # As read: the harness's own flag beside the stored keys (G-387 round
    # 16), set on the denial the harness wrote for the host's row.
    assert list(marked[0]) == [
        "type",
        "tool_results",
        "tool_approval_responses",
        SUPERSEDED,
        "harness",
    ]
    # Placed right before the row that superseded the step.
    at = stored.index(marked[0])
    assert stored[at + 1]["type"] == SUPERSEDING[kind]["type"]
    assert superseded_approval_ids(stored) == set(approvals)
    assert approval_decisions(stored) == {}
    assert pending_approval_ids(stored) == []
    # A later decision is refused, and no resume runs anything of the step.
    for approval in approvals:
        with pytest.raises(HarnessError) as caught:
            record_approval(world.fresh().root(), approval, True)
        assert caught.value.code == "approval_not_pending"
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    world.text("Again.")
    world.fresh().runtime.send(world.root(), "")
    assert not any(name == "echo" for name, _ in world.ran_with)
    # Exactly ONE outcome per gated call of the parked step.
    assert outcomes_of(world.root(), 1) == [SUPERSEDED_RESULT]
    assert outcomes_of(world.root(), 2) == [SUPERSEDED_RESULT]


def test_the_harness_own_rows_in_its_run_do_not_supersede() -> None:
    """A resume answers the step and then writes its own text answer in its
    run: no denial, and the approved call ran once."""
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == [("echo", {"n": 1})]
    assert marks(world.root()) == []


@pytest.mark.parametrize("step", sorted(STEPS))
@pytest.mark.parametrize("kind", ["assistant_text", "user"])
def test_a_host_row_carrying_the_steps_own_run_id_supersedes_it(kind: str, step: str) -> None:
    """A run id a CALLER passes is always outside, the step's own included:
    only the runtime's own in-run recording is inside. The host's row
    supersedes the step: one denial per open call, every decision refused,
    nothing runs."""
    world = Recording()
    _, approvals = park(world, *STEPS[step])
    record_approval(world.fresh().root(), approvals[0], True)
    host = world.fresh().root()

    host.thread().record([dict(SUPERSEDING[kind])], parked_run_id(host))

    assert marks(world.root()) == [{"approval_ids": approvals}]
    for approval in approvals:
        with pytest.raises(HarnessError) as caught:
            record_approval(world.fresh().root(), approval, True)
        assert caught.value.code == "approval_not_pending"
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    world.text("Again.")
    world.fresh().runtime.send(world.root(), "")
    assert not any(name == "echo" for name, _ in world.ran_with)
    assert outcomes_of(world.root(), 1) == [SUPERSEDED_RESULT]
    assert outcomes_of(world.root(), 2) == [SUPERSEDED_RESULT]


@pytest.mark.parametrize(("run", "supersedes"), [("r0", False), ("r1", False), ("r2", True)])
def test_the_runtime_own_row_is_inside_only_in_an_owning_run(run: str, supersedes: bool) -> None:
    """The runtime's own recording (the private _record_in_run) is inside
    when its run owns the step (it recorded the step, r0, or one of its
    results, r1)."""
    from prism_harness.thread import _record_in_run

    world = Recording()
    session = _stored_step(world, [("apr_a", "a"), ("apr_b", "b")], "a")

    _record_in_run(session.thread(), [{"type": "user", "content": "Later."}], run)

    assert marks(world.root()) == ([{"approval_ids": ["apr_b"]}] if supersedes else [])


def test_a_later_step_with_tool_calls_supersedes_even_under_the_steps_own_run_id() -> None:
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}))
    root = world.fresh().root()

    root.thread().record([dict(SUPERSEDING["assistant_step"])], parked_run_id(root))

    assert marks(world.root()) == [{"approval_ids": approvals}]


def test_a_call_with_a_result_is_not_denied_again() -> None:
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))
    record_approval(world.fresh().root(), approvals[0], True)
    record_approval(world.fresh().root(), approvals[1], False, "no")
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    world.fresh().root().thread().record([dict(SUPERSEDING["user"])])

    assert marks(world.root()) == []
    assert outcomes_of(world.root(), 1) == ["echo:1"]


@pytest.mark.parametrize("decided", [False, True], ids=["undecided", "approved"])
def test_a_new_message_supersedes_the_parked_step(decided: bool) -> None:
    """A new message through the harness is a new run: the parked step's open
    gated calls are denied once before its turn, even one a person approved,
    and a later decision is refused."""
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}), ("b", "echo", {"n": 2}))
    if decided:
        record_approval(world.fresh().root(), approvals[0], True)
    world.text("Answered.")

    world.fresh().runtime.send(world.root(), "Something else.")

    assert not any(name == "echo" for name, _ in world.ran_with)
    assert marks(world.root()) == [{"approval_ids": approvals}]
    assert outcomes_of(world.root(), 1) == [SUPERSEDED_RESULT]
    assert outcomes_of(world.root(), 2) == [SUPERSEDED_RESULT]
    for approval in approvals:
        with pytest.raises(HarnessError):
            record_approval(world.fresh().root(), approval, True)


# -- the release lines' rules (G-387 round 13) ----------------------------------------


def _stored_step(world: Recording, requests: list[tuple[str, str]], prior: str | None) -> Session:
    """A step as another writer left it: echo calls a (n=1) and b (n=2), the
    requests given, and (optionally) a result for one call recorded under the
    run ``r1``."""
    from prism_harness.thread_rows import (
        ToolCallInput,
        assistant_row,
        tool_result_entry,
        tool_result_row,
    )

    session = world.root()
    calls = [ToolCallInput("a", "echo", {"n": 1}), ToolCallInput("b", "echo", {"n": 2})]
    step = assistant_row(
        "",
        calls,
        {},
        [
            {"approval_id": approval, "tool_call_id": call, "arguments_format": 1}
            for approval, call in requests
        ],
    )
    step["harness_tool_names"] = ["echo", "lookup"]
    session.thread().record([{"type": "user", "content": "go"}, step], "r0")
    if prior is not None:
        call = calls[0] if prior == "a" else calls[1]
        session.thread().record([tool_result_row([tool_result_entry(call, "prior")])], "r1")
    return session


@pytest.mark.parametrize("run", ["r0", "r1", "r2", None])
def test_a_host_text_or_user_row_supersedes_whatever_run_id_it_carries(run: str | None) -> None:
    """A run id a caller passes is outside, even an owning run's (the run
    that recorded the step, r0, or one of its results, r1)."""
    supersedes = True
    world = Recording()
    session = _stored_step(world, [("apr_a", "a"), ("apr_b", "b")], "a")

    session.thread().record([{"type": "user", "content": "Later."}], run)

    assert marks(world.root()) == ([{"approval_ids": ["apr_b"]}] if supersedes else [])
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert not any(name == "echo" for name, _ in world.ran_with)
    assert len(outcomes_of(world.root(), 2)) == 1


def test_a_repeated_approval_id_across_distinct_calls_fails_closed() -> None:
    world = Recording()
    _stored_step(world, [("apr_same", "a"), ("apr_same", "b")], None)

    record_approval(world.fresh().root(), "apr_same", True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    repeated = "Tool call was not run because another call in this step has the same id"
    assert outcomes_of(world.root(), 1) == [repeated]
    assert outcomes_of(world.root(), 2) == [repeated]
    decisions = [
        response
        for row in rows(world.root())
        if row.get("type") == "tool_result"
        for response in row.get("tool_approval_responses", [])
    ]
    assert decisions and all(response["approved"] is False for response in decisions)


@pytest.mark.parametrize("how", ["record_approval", "decide"])
def test_a_second_decision_on_a_decided_approval_stays_refused(how: str) -> None:
    """G-401: an approval answered once is answered; a second answer, either
    way, is refused with nothing written, and only the first one counts."""
    world = Recording()
    _, approvals = park(world, ("a", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], False, "no")
    before = rows(world.fresh().root())

    with pytest.raises(HarnessError) as caught:
        if how == "record_approval":
            record_approval(world.fresh().root(), approvals[0], True)
        else:
            world.fresh().runtime.decide(world.root(), approvals[0], True)

    assert caught.value.code == "approval_not_pending"
    assert rows(world.fresh().root()) == before
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    assert world.ran_with == []
    assert outcomes_of(world.root(), 1) == ["no"]
