"""A 0.27 main reads the threads its released line (0.5.x) wrote (G-387 round
13, DECIDED).

The 0.5.x line answers an earlier step a later one replaced with a row marked
``harness_replaced_step: {position}``, written when it resumes, after the
later rows. The fixture holds threads that line's own runtime recorded (see
its ``_note``): a parked step of two gated calls, the first approved, then a
host's user row or a later assistant step with tool calls, before and after
that line resumed it. Main resumes each, twice, with every approval decided
again late: each gated call ends with EXACTLY ONE outcome, the replaced-step
denial, no call is denied twice, and nothing runs.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_step_allowance import World

from prism_harness import HarnessError, ModeRegistry, Participant, Session, record_approval
from prism_harness.plans.thread_facts import approval_decisions
from prism_harness.thread_rows import SUPERSEDED_RESULT

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "py-0.5.x-replaced-step-threads.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


class Upgrade(World):
    """The World with echo and shout tools that record every run."""

    def _runtime(self) -> Any:
        runtime = super()._runtime()
        world = self

        class Tool:
            def __init__(self, name: str) -> None:
                self.name = name

            def handle(self, args: dict[str, Any]) -> Any:
                world.ran.append(f"{self.name}:{args.get('token')}")
                return "ran"

        runtime._tools.register(Tool("shout"))
        runtime._modes = _modes()
        return runtime


def _modes() -> ModeRegistry:
    return ModeRegistry(
        {
            "default": "guarded",
            "modes": {
                "guarded": {
                    "system_prompt": "Careful.",
                    "tools": ["echo", "shout"],
                    "max_steps": 8,
                    "requires_approval": ["echo"],
                }
            },
        }
    )


def load(world: Upgrade, name: str) -> Session:
    """The 0.5.x line's thread, byte for byte, in a session main resumes."""
    session = load_again(world)
    key = f"{session.key()}:thread"
    session._durable.put(  # a test standing in for the store the 0.5.x line wrote
        key,
        {
            "messages": [
                {
                    "position": entry["position"],
                    "message": entry["message"],
                    "run_id": entry["run_id"],
                    "recorded_at": "2026-11-27T00:00:00+00:00",
                }
                for entry in FIXTURE["threads"][name]
            ]
        },
    )
    return session


def outcomes(session: Session) -> dict[str, list[str]]:
    """Every stored result for the parked step's calls (tokens t0, t1),
    attributed to the step it answers: recorded in that step's segment, or
    marked for it."""
    rows = [entry.message for entry in session.thread().messages()]
    found: dict[str, list[str]] = {}
    step = 0
    for position, row in enumerate(rows, start=1):
        if row.get("type") == "assistant" and row.get("tool_calls"):
            step = position
            continue
        if row.get("type") != "tool_result":
            continue
        mark = row.get("harness_replaced_step")
        owner = mark.get("position") if isinstance(mark, dict) else step
        if owner != 2:  # the parked step is the row at position 2
            continue
        for result in row["tool_results"]:
            token = str(result["args"].get("token"))
            found.setdefault(token, []).append(result["result"])
    return found


@pytest.mark.parametrize("name", sorted(FIXTURE["threads"]))
def test_main_resumes_a_0_5_x_thread_with_one_outcome_per_gated_call(name: str) -> None:
    world = Upgrade()
    session = load(world, name)
    stored = [entry.message for entry in session.thread().messages()]
    approvals = [request["approval_id"] for request in stored[1]["tool_approval_requests"]]

    for _ in range(2):
        world.text("Done.")
        world.fresh()
        world.runtime.send(load_again(world), "")
        for approval in approvals:
            try:
                record_approval(load_again(world), approval, True)
            except HarnessError as error:
                assert error.code == "approval_not_pending"

    found = outcomes(load_again(world))
    assert {token: len(set(texts)) for token, texts in found.items()} == {"t0": 1, "t1": 1}
    assert found["t0"][0] == SUPERSEDED_RESULT
    assert found["t1"][0] == SUPERSEDED_RESULT
    # No call is denied twice: one replaced-step row per call, in all.
    assert sum(texts.count(SUPERSEDED_RESULT) for texts in found.values()) == 2
    assert world.ran == []
    rows = [entry.message for entry in load_again(world).thread().messages()]
    assert all(approval_decisions(rows).get(approval) is not True for approval in approvals)


def load_again(world: Upgrade) -> Session:
    return world.harness.for_(Participant("User", "ada")).session("upgrade")


@pytest.mark.parametrize("name", sorted(FIXTURE["threads"]))
def test_a_0_5_x_replaced_step_is_never_approvable_and_denied_exactly_once(name: str) -> None:
    """The PHP 0.26.10 audit's upgrade rule, in the 0.5.x line's own bytes: a
    step that line replaced (by a user row, or by a later step) is never
    approvable on main, before or after a resume, and each of its gated calls
    is denied exactly once (counted, never as a set), and nothing runs."""
    world = Upgrade()
    session = load(world, name)
    stored = [entry.message for entry in session.thread().messages()]
    approvals = [request["approval_id"] for request in stored[1]["tool_approval_requests"]]

    for _ in range(3):
        for approval in approvals:
            with pytest.raises(HarnessError) as caught:
                record_approval(load_again(world), approval, True)
            assert caught.value.code == "approval_not_pending"
        world.text("Done.")
        world.fresh()
        world.runtime.send(load_again(world), "")

    found = outcomes(load_again(world))
    assert found == {"t0": [SUPERSEDED_RESULT], "t1": [SUPERSEDED_RESULT]}
    assert world.ran == []
