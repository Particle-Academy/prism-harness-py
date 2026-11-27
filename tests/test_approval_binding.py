"""An approval binds to the exact call the person was shown (G-387 round 15).

When a step parks, each approval request records, as its LAST key, the
call's tool name and the digest of its canonical arguments
(``harness_call``: ``{tool_name, args_digest}``; ``sha256:`` and the hex
SHA-256 of the RFC 8785 canonical JSON, the PHP reference's form). A call
runs only when its request's binding matches it; otherwise it is denied,
fail-closed, with exactly one outcome. And an approval belongs only to the
step that FIRST asked for it, so one approval never runs more than one call.

Each shape is entered at the public seam (``send``, ``record_approval``,
``Thread.record``) or as a store writer would leave the thread.
"""

from __future__ import annotations

import copy
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_repeated_call_ids import Recording, park, rows
from test_step_allowance import rewrite_paused_row

from prism_harness import HarnessError, record_approval
from prism_harness.thread_rows import (
    CALL_BINDING,
    NOT_THE_APPROVED_CALL,
    binding_digest,
    pending_approval_ids,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


def outcomes(world: Recording, n: int) -> list[str]:
    return [
        entry["result"]
        for row in rows(world.root())
        if row.get("type") == "tool_result"
        for entry in row.get("tool_results") or []
        if (entry.get("args") or {}).get("n") == n
    ]


def resume(world: Recording, times: int = 3) -> None:
    for _ in range(times):
        world.text("Done.")
        world.fresh().runtime.send(world.root(), "")


def test_the_binding_is_the_reference_digest_and_the_last_key() -> None:
    assert (
        binding_digest({"city": "Paris"})
        == "sha256:6e1e312d537bc71b5410b0599f5a508142149e13174c6ee0d1671658845bc67d"
    )
    assert binding_digest('{"city":"Paris"}') == binding_digest({"city": "Paris"})
    assert binding_digest(7) == binding_digest({}) == binding_digest(None)
    world = Recording()
    park(world, ("c1", "echo", {"n": 1}))
    (request,) = next(row for row in rows(world.root()) if row["type"] == "assistant")[
        "tool_approval_requests"
    ]
    assert list(request)[-1] == CALL_BINDING
    assert request[CALL_BINDING] == {"tool_name": "echo", "args_digest": binding_digest({"n": 1})}


def test_a_host_step_gets_its_binding_computed_never_taken_as_given() -> None:
    world = Recording()
    forged = {"tool_name": "echo", "args_digest": "sha256:" + "0" * 64}
    world.root().thread().record(
        [
            {"type": "user", "content": "Go"},
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"n": 1}}],
                "tool_approval_requests": [
                    {"approval_id": "apr_1", "tool_call_id": "c1", CALL_BINDING: forged}
                ],
            },
        ]
    )
    # Stored computed, never as given; and read as absent on a row the
    # harness did not write (G-387 round 16): it binds its own call by
    # position, which is the same call.
    stored = [entry.message for entry in world.root().thread().messages()]
    (request,) = stored[1]["tool_approval_requests"]
    assert request[CALL_BINDING] == {"tool_name": "echo", "args_digest": binding_digest({"n": 1})}
    (read,) = rows(world.root())[1]["tool_approval_requests"]
    assert CALL_BINDING not in read


@pytest.mark.parametrize("field", ["arguments", "name"])
def test_a_parked_call_rewritten_after_the_park_never_runs(field: str) -> None:
    """A store writer changes the parked call (its arguments, or its tool)
    under the same id: the binding no longer matches, so the call is denied
    with one outcome, whenever the person approves."""
    world = Recording()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))

    def alter(row: dict[str, Any]) -> None:
        if field == "arguments":
            row["tool_calls"][0]["arguments"] = {"n": 2}
        else:
            row["tool_calls"][0]["name"] = "lookup"

    rewrite_paused_row(world.root(), alter)
    record_approval(world.fresh().root(), approvals[0], True)
    resume(world)

    assert world.ran_with == []
    results = [
        entry["result"]
        for row in rows(world.root())
        if row.get("type") == "tool_result"
        for entry in row.get("tool_results") or []
    ]
    assert results == [NOT_THE_APPROVED_CALL]


@pytest.mark.parametrize("when", ["before_the_approval", "after_the_approval"])
@pytest.mark.parametrize("held", ["altered", "both"])
def test_a_copy_altering_the_args_under_the_kept_request_never_runs(when: str, held: str) -> None:
    """A host records a copy of the parked step that keeps the request but
    changes the arguments under the same call id (alone, or beside the
    original call), before or after the approval: neither call runs, and
    the approval is never accepted for the copy."""
    world = Recording()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    parked = next(row for row in rows(world.root()) if row["type"] == "assistant")
    altered = copy.deepcopy(parked)
    original = copy.deepcopy(parked["tool_calls"][0])
    changed = {**copy.deepcopy(original), "arguments": {"n": 2}}
    altered["tool_calls"] = [changed] if held == "altered" else [original, changed]

    if when == "after_the_approval":
        record_approval(world.fresh().root(), approvals[0], True)
    world.fresh().root().thread().record([altered])
    assert pending_approval_ids(rows(world.root())) == []
    if when == "before_the_approval":
        with pytest.raises(HarnessError) as caught:
            record_approval(world.fresh().root(), approvals[0], True)
        assert caught.value.code == "approval_not_pending"
    resume(world)

    assert world.ran_with == []
    # The copy's calls: at most one outcome each, never a run.
    assert len(outcomes(world, 2)) <= 1
    assert all(not text.startswith("echo:") for text in outcomes(world, 2))
    assert len(outcomes(world, 1)) == (1 if held == "altered" else 2)


def test_a_request_without_a_binding_binds_its_own_call_by_position() -> None:
    """The upgrade rule (rows written before the binding): a request with no
    ``harness_call`` binds the call it is paired with by position."""
    world = Recording()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))
    rewrite_paused_row(
        world.root(),
        lambda row: row["tool_approval_requests"][0].pop(CALL_BINDING),
    )
    record_approval(world.fresh().root(), approvals[0], True)
    resume(world)

    assert world.ran_with == [("echo", {"n": 1})]
    assert outcomes(world, 1) == ["echo:1"]


@pytest.mark.parametrize("malformed", [None, "x", {}, {"tool_name": "echo"}, {"args_digest": 7}])
def test_a_malformed_binding_never_binds(malformed: object) -> None:
    world = Recording()
    _, approvals = park(world, ("c1", "echo", {"n": 1}))

    def spoil(row: dict[str, Any]) -> None:
        row["tool_approval_requests"][0][CALL_BINDING] = malformed

    rewrite_paused_row(world.root(), spoil)
    record_approval(world.fresh().root(), approvals[0], True)
    resume(world)

    assert world.ran_with == []
    assert outcomes(world, 1) == [NOT_THE_APPROVED_CALL]
