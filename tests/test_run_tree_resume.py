"""A child resumed outside the runner continues its run tree's budget.

A subagent child that stops on an approval is resumed later on its own session,
outside the runner. That run continues the run tree it was started in: it
spends what the tree has left, never a fresh root budget. A spent tree grants
no model step and refuses nothing: an approval is acted on with no model step
(G-387). Ports
prism-harness's tests/RunTreeResumeTest.php, driven through ``send``,
``record_approval`` and ``runtime.decide`` with a scripted model; the network
is refused and every store is in memory.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest
from test_plan_budget import Rig, pending

from prism_harness import (
    Participant,
    RunBudget,
    RunContext,
    Session,
    record_approval,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def paused_child(tree_steps: int) -> tuple[Rig, Session, str]:
    """The parent turn (a tree of ``tree_steps`` model steps) delegates; the
    child pauses on weather after one step; the parent answers. Three tree
    steps."""
    rig = Rig(mode={"max_steps": tree_steps, "plans": False}, subagents=True)
    rig.calls(("research", {"task": "Find the weather."}))  # parent
    rig.calls(("weather", {"city": "Paris"}))  # child: pauses
    rig.text("Asked.")  # parent
    rig.runtime.send(rig.session, "Go")

    child = rig.harness.for_(Participant("User", "ada")).session("chat::sub::research")
    waiting = pending(rig, child)
    assert len(waiting) == 1
    return rig, child, waiting[0]


def test_a_resumed_child_runs_on_what_its_tree_has_left_not_a_fresh_root_budget() -> None:
    rig, child, approval = paused_child(4)
    before = len(rig.requests)

    # The child would take three more steps; the tree has one left.
    rig.calls(("lookup", {"q": "a"}))
    rig.calls(("lookup", {"q": "b"}))
    rig.text("Child answer.")
    record_approval(child, approval, True)
    rig.runtime.send(child, "")

    assert rig.ran.count("weather") == 1
    assert len(rig.requests) - before == 1


@pytest.mark.parametrize("path", ["decide", "record_approval"])
def test_an_approval_for_a_child_whose_run_tree_is_spent_is_acted_on_with_no_model_step(
    path: str,
) -> None:
    rig, child, approval = paused_child(3)
    before = len(rig.requests)

    # G-387: acting on a decision costs no model step, so a spent tree never
    # refuses it; the resume runs the approved call and takes no step.
    if path == "decide":
        rig.runtime.decide(child, approval, True)
    else:
        record_approval(child, approval, True)
    resumed = rig.runtime.send(child, "")

    assert resumed.finish_reason == "budget_exhausted"
    assert pending(rig, child) == []
    assert rig.ran.count("weather") == 1
    assert len(rig.requests) == before


def test_a_resumed_child_is_charged_to_its_tree_once() -> None:
    rig, child, approval = paused_child(5)

    # Resumed: one step, which pauses on weather again (tree 4 of 5).
    rig.calls(("weather", {"city": "Rome"}))
    record_approval(child, approval, True)
    second = rig.runtime.send(child, "").pending_approvals[0].id
    before = len(rig.requests)

    # One left: the child takes one of the three it would.
    rig.calls(("lookup", {"q": "a"}))
    rig.calls(("lookup", {"q": "b"}))
    rig.text("Child answer.")
    record_approval(child, second, True)
    rig.runtime.send(child, "")

    assert len(rig.requests) - before == 1


# A turn on a child session the runner did not start never gets a fresh root
# budget by any route: not a new message instead of an answer, and not a run
# context the host hands in (which may only narrow what the tree has left).


def test_a_new_message_to_a_child_session_runs_on_what_its_tree_has_left() -> None:
    rig, child, _ = paused_child(4)
    before = len(rig.requests)

    rig.calls(("lookup", {"q": "a"}))
    rig.calls(("lookup", {"q": "b"}))
    rig.text("Child answer.")
    rig.runtime.send(child, "New question")

    assert len(rig.requests) - before == 1


def test_a_new_message_to_a_child_session_whose_tree_is_spent_takes_no_model_step() -> None:
    rig, child, _ = paused_child(3)
    before = len(rig.requests)

    # A spent tree grants 0 steps (G-387), not a refusal.
    response = rig.runtime.send(child, "New question")

    assert response.finish_reason == "budget_exhausted"
    assert len(rig.requests) == before


def test_a_host_context_on_a_resumed_child_is_narrowed_to_what_its_tree_has_left() -> None:
    rig, child, approval = paused_child(4)
    before = len(rig.requests)

    rig.calls(("lookup", {"q": "a"}))
    rig.calls(("lookup", {"q": "b"}))
    rig.text("Child answer.")
    record_approval(child, approval, True)
    rig.runtime.send(child, "", context=RunContext.root("host_root", RunBudget(100)))

    assert rig.ran.count("weather") == 1
    assert len(rig.requests) - before == 1


def test_a_host_context_on_a_child_whose_tree_is_spent_takes_no_model_step() -> None:
    rig, child, _ = paused_child(3)
    before = len(rig.requests)

    response = rig.runtime.send(
        child, "New question", context=RunContext.root("host_root", RunBudget(100))
    )

    assert response.finish_reason == "budget_exhausted"
    assert len(rig.requests) == before
