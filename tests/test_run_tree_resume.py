"""A child resumed outside the runner continues its run tree's budget.

A subagent child that stops on an approval is resumed later on its own session,
outside the runner. That run continues the run tree it was started in: it
spends what the tree has left, never a fresh root budget, and an approval the
spent tree could not run is refused with nothing recorded. Ports
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

from prism_harness import HarnessError, Participant, Session, record_approval


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
def test_an_approval_for_a_child_whose_run_tree_is_spent_is_refused(path: str) -> None:
    rig, child, approval = paused_child(3)
    before = len(rig.requests)

    with pytest.raises(HarnessError) as caught:
        if path == "decide":
            rig.runtime.decide(child, approval, True)
        else:
            record_approval(child, approval, True)
    assert caught.value.code == "run_not_permitted"
    assert pending(rig, child) == [approval]
    assert rig.ran.count("weather") == 0
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
