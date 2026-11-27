"""G-408 (G-387 round 15): a stored copy of the paused step whose
``tool_calls`` are NOT objects.

An app may store a copy of the parked assistant step with its calls as a
JSON string, a number, null, a list of strings or a list of JSON-string
calls, keeping (P24) or dropping (P22) its approval requests. Such a row is
never THE paused step, for ANY reader: the decision readers, the resolver,
the supersede and the replay all select a step with the one rule, an
assistant row whose ``tool_calls`` is a list holding at least one object
(:func:`~prism_harness.thread_rows.is_tool_step`; the TS 0.11.x #92 rule the
PHP reference adopted). And an approval belongs only to the step that FIRST
asked for it, so a copy that keeps the old request asks for nothing.

Each copy is recorded after the approval, through the public ``record`` (a
host) and through the runtime's own private in-run write, in a mode with
steps left and in a spent one, then the session is resumed three times. The
approved call never runs twice, it has exactly ONE outcome, and no resume
writes an extra row.
"""

from __future__ import annotations

import copy
import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from test_repeated_call_ids import Recording, park, rows

from prism_harness import HarnessError, record_approval
from prism_harness.thread import _record_in_run
from prism_harness.thread_rows import (
    decided_unresolved_ids,
    paused_approval_row,
    pending_approval_ids,
    requested_step,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


SHAPES: dict[str, Callable[[list[dict[str, Any]]], Any]] = {
    "json_string": lambda calls: '[{"id": "c1", "name": "echo", "arguments": {"n": 1}}]',
    "number": lambda calls: 7,
    "null": lambda calls: None,
    "strings": lambda calls: ["x"],
    "json_string_calls": lambda calls: [
        '{"id": "c1", "name": "echo", "arguments": {"n": 1}}' for _ in calls
    ],
}


def _copy(parked: dict[str, Any], shape: str, requests: bool) -> dict[str, Any]:
    duplicate = copy.deepcopy(parked)
    duplicate["tool_calls"] = SHAPES[shape](parked["tool_calls"])
    if not requests:
        duplicate["tool_approval_requests"] = []
    return duplicate


def _outcomes(stored: list[dict[str, Any]]) -> list[str]:
    return [
        entry["result"]
        for row in stored
        if row.get("type") == "tool_result"
        for entry in row.get("tool_results") or []
        if isinstance(entry, dict) and entry.get("tool_call_id") == "c1"
    ]


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("requests", [True, False], ids=["P24-requests", "P22-no-requests"])
@pytest.mark.parametrize("steps", [4, 1], ids=["steps-left", "spent"])
@pytest.mark.parametrize("route", ["host", "in_run"])
def test_a_non_object_copy_of_the_paused_step_is_never_the_step(
    shape: str, requests: bool, steps: int, route: str
) -> None:
    world = Recording()
    world.lead_steps = steps
    world.fresh()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    parked = next(row for row in rows(world.root()) if row.get("type") == "assistant")
    duplicate = _copy(parked, shape, requests)
    thread = world.fresh().root().thread()
    if route == "host":
        thread.record([duplicate])
    else:
        _record_in_run(thread, [duplicate], "run_copy")

    stored = rows(world.root())
    # Every reader agrees: the copy is not a step, so nothing is answerable
    # through it and nothing reads as decided-but-unresolved through it.
    assert requested_step(stored) is None
    assert paused_approval_row(stored) is None
    assert pending_approval_ids(stored) == []
    assert decided_unresolved_ids(stored) == []
    with pytest.raises(HarnessError) as caught:
        record_approval(world.fresh().root(), approvals[0], True)
    assert caught.value.code == "approval_not_pending"

    counts: list[int] = []
    for _ in range(3):
        world.text("Done.")
        world.fresh().runtime.send(world.root(), "")
        counts.append(len([row for row in rows(world.root()) if row.get("type") == "tool_result"]))

    assert world.ran_with == []
    assert len(_outcomes(rows(world.root()))) == 1
    # No resume writes another result row.
    assert counts == [counts[0]] * 3


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_non_object_copy_recorded_after_the_result_changes_nothing(shape: str) -> None:
    world = Recording()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    world.text("Done.")
    world.fresh().runtime.send(world.root(), "")
    parked = next(row for row in rows(world.root()) if row.get("type") == "assistant")
    world.fresh().root().thread().record([_copy(parked, shape, True)])

    for _ in range(3):
        world.text("Done.")
        world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == [("echo", {"n": 1})]
    assert _outcomes(rows(world.root())) == ["echo:1"]


def test_a_list_holding_an_object_call_is_a_step_by_its_objects() -> None:
    """The TS 0.11.x #92 rule (the PHP reference's, REVISION 2): a list
    holding at least one object is a step, its calls exactly those objects;
    a list with none replays as a text row with no calls."""
    from prism_harness.thread_rows import is_tool_step, step_calls, thread_view

    call = {"id": "c1", "name": "e", "arguments": {}}
    mixed = {"type": "assistant", "content": "", "tool_calls": ["x", call, None]}
    assert is_tool_step(mixed)
    assert step_calls(mixed) == [call]
    assert [c["id"] for c in thread_view([mixed])[0]["tool_calls"]] == ["c1"]
    for junk in (["x"], [], "[]", None, 7, {"id": "c1"}):
        row = {"type": "assistant", "content": "", "tool_calls": junk}
        assert not is_tool_step(row)
        assert thread_view([row])[0]["tool_calls"] == []
    assert not is_tool_step({"type": "user", "tool_calls": [call]})


@pytest.mark.parametrize("route", ["host", "in_run"])
@pytest.mark.parametrize("calls", ["mixed", "objects"])
def test_a_copy_with_a_new_call_keeping_the_old_request_never_runs_the_new_call(
    route: str, calls: str
) -> None:
    """A copy of the parked step holding a NEW call (mixed with non-objects,
    or all objects) that keeps the OLD approval request, then that approval
    recorded again after it: an approval belongs only to the step that FIRST
    asked for it, so the new call is never run under it (TS 0.11.x #92)."""
    world = Recording()
    world.lead_steps = 1
    world.fresh()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    record_approval(world.fresh().root(), approvals[0], True)
    parked = next(row for row in rows(world.root()) if row.get("type") == "assistant")
    new_call = {"id": "c1", "name": "echo", "arguments": {"n": 9}}
    duplicate = copy.deepcopy(parked)
    duplicate["tool_calls"] = ["x", new_call, None] if calls == "mixed" else [new_call]
    again = {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [
            {"approval_id": approvals[0], "approved": True, "reason": None}
        ],
    }
    thread = world.fresh().root().thread()
    if route == "host":
        thread.record([duplicate])
    else:
        _record_in_run(thread, [duplicate], "run_copy")
    thread.record([again])
    stored = rows(world.root())
    assert pending_approval_ids(stored) == []
    assert decided_unresolved_ids(stored) == []

    for _ in range(3):
        world.text("Done.")
        world.fresh().runtime.send(world.root(), "")

    assert world.ran_with == []
    assert _outcomes(rows(world.root())) in (
        ["Tool call was not run because a later step replaced it"],
        [
            "Tool call was not run because a later step replaced it",
            "No approval response provided",
        ],
    )
    with pytest.raises(HarnessError):
        record_approval(world.fresh().root(), approvals[0], True)
