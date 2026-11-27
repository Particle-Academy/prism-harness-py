"""The round-13 review's Info items (G-387 round 14, item 3).

- No public way to mark a row as the runtime's own in-run recording: the
  helper is private.
- A 0.5.x ``harness_replaced_step.position`` is a thread POSITION
  (``entry.position``), never a list index: a thread whose positions do not
  run 1..n still reads its replaced step.
- The "ran at once" pairing heuristic skips every harness mark (ended by a
  start, superseded, replaced step): a denial row written right after the
  step never reads as calls the step ran.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_upgrade_from_0_5 import FIXTURE, Upgrade, load_again

import prism_harness.thread as thread_module
from prism_harness import HarnessError, Session, record_approval
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.thread_rows import (
    REPLACED_STEP,
    SUPERSEDED,
    SUPERSEDED_RESULT,
    step_requests,
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


def test_the_in_run_recording_helper_is_not_public() -> None:
    assert not hasattr(thread_module, "record_in_run")
    assert "record_in_run" not in dir(thread_module)
    assert all(not name.startswith("record") for name in thread_module.__all__)


# -- the 0.5.x position is a thread position --------------------------------------

SHIFT = 10


def load_shifted(world: Upgrade, name: str) -> Session:
    """The 0.5.x thread with every position (and the mark's) moved by SHIFT,
    as a thread whose earlier rows were pruned would hold them."""
    session = load_again(world)
    entries = []
    for entry in FIXTURE["threads"][name]:
        message = json.loads(json.dumps(entry["message"]))
        mark = message.get(REPLACED_STEP)
        if isinstance(mark, dict):
            mark["position"] = mark["position"] + SHIFT
        entries.append(
            {
                "position": entry["position"] + SHIFT,
                "message": message,
                "run_id": entry["run_id"],
                "recorded_at": "2026-11-27T00:00:00+00:00",
            }
        )
    session._durable.put(f"{session.key()}:thread", {"messages": entries})
    return session


def denials_for_the_parked_step(session: Session) -> int:
    """Replaced-step denials attributed to the parked step (position 2+SHIFT):
    in its own segment, or marked for it."""
    count = 0
    step: int | None = None
    for entry in session.thread().messages():
        row = entry.message
        if row.get("type") == "assistant" and row.get("tool_calls"):
            step = entry.position
            continue
        if row.get("type") != "tool_result":
            continue
        mark = row.get(REPLACED_STEP)
        owner = mark.get("position") if isinstance(mark, dict) else step
        if owner == 2 + SHIFT:
            count += sum(1 for result in row["tool_results"] if result["result"] == SUPERSEDED_RESULT)
    return count


@pytest.mark.parametrize("name", ["user-resumed", "later_step-resumed"])
def test_a_replaced_step_mark_is_read_by_thread_position(name: str) -> None:
    world = Upgrade()
    load_shifted(world, name)
    stored = [entry.message for entry in load_again(world).thread().messages()]
    approvals = [request["approval_id"] for request in stored[1]["tool_approval_requests"]]

    # The mark is read: both approvals read as replaced, none as approved.
    assert superseded_approval_ids(stored) == set(approvals)

    world.text("Done.")
    world.fresh()
    world.runtime.send(load_again(world), "")
    for approval in approvals:
        with pytest.raises(HarnessError):
            record_approval(load_again(world), approval, True)

    # One denial per gated call, never a second written by main.
    assert denials_for_the_parked_step(load_again(world)) == 2
    assert world.ran == []


# -- the pairing heuristic skips every harness mark -------------------------------


def _step(*calls: tuple[str, str]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "name": "echo", "arguments": {"token": token}}
            for call_id, token in calls
        ],
        "additional_content": {},
        "tool_approval_requests": [
            {"approval_id": f"apr_{call_id}", "tool_call_id": call_id} for call_id, _ in calls
        ],
    }


def _denial(calls: list[tuple[str, str]], mark: tuple[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_results": [
            {
                "tool_call_id": call_id,
                "tool_name": "echo",
                "args": {"token": token},
                "result": SUPERSEDED_RESULT,
                "tool_call_result_id": None,
                "artifacts": [],
            }
            for call_id, token in calls
        ],
        "tool_approval_responses": [],
        mark[0]: mark[1],
    }


@pytest.mark.parametrize(
    "mark",
    [
        (SUPERSEDED, {"approval_ids": ["apr_a", "apr_b"]}),
        (REPLACED_STEP, {"position": 2}),
        ("harness_ended_by_start", {"approval_ids": ["apr_a", "apr_b"]}),
    ],
    ids=["superseded", "replaced_step", "ended_by_start"],
)
def test_a_harness_denial_right_after_the_step_is_never_read_as_calls_it_ran(
    mark: tuple[str, Any],
) -> None:
    calls = [("a", "t0"), ("b", "t1")]
    rows = [{"type": "user", "content": "go"}, _step(*calls), _denial(calls, mark)]

    pairs = step_requests(rows, 1)

    # Every request keeps its own call: the denial row is not "ran at once".
    assert [(request["approval_id"], call) for request, call in pairs] == [
        ("apr_a", 0),
        ("apr_b", 1),
    ]


def test_a_0_5_x_denial_right_after_the_step_still_names_its_approvals() -> None:
    """The 0.5.x replaced-step row written straight after the step it
    answers (no other result between): its approvals read as replaced, and
    a decision recorded after it never reads as approved for a plan."""
    calls = [("a", "t0"), ("b", "t1")]
    later = {**_step(("z", "t9")), "tool_approval_requests": []}
    decision = {
        "type": "tool_result",
        "tool_results": [],
        "tool_approval_responses": [{"approval_id": "apr_a", "approved": True, "reason": None}],
    }
    rows = [
        {"type": "user", "content": "go"},
        _step(*calls),
        _denial(calls, (REPLACED_STEP, {"position": 2})),
        decision,
        later,
    ]

    assert superseded_approval_ids(rows) == {"apr_a", "apr_b"}
    assert approval_decisions(rows).get("apr_a") is not True


def test_the_fixture_file_is_the_one_the_upgrade_tests_read() -> None:
    path = Path(__file__).parent / "fixtures" / "py-0.5.x-replaced-step-threads.json"
    assert json.loads(path.read_text(encoding="utf-8")) == FIXTURE
