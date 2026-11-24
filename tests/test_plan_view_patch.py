"""G-360: a patch carries changed steps by id and cannot say where a new step
goes. A pause inserts the person's step BEFORE the current one, so a reader
applying a patch put it last. Any change to the list of ids is a snapshot."""

from __future__ import annotations

from typing import Any

from prism_harness.plans import PlanIds, PlanLimits, machine, view


def _plan() -> tuple[dict[str, Any], PlanIds, PlanLimits]:
    limits = PlanLimits()
    counter = [0]

    def mint(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]:026d}"

    ids = PlanIds(mint)
    steps = [
        {"title": "One", "instruction": "Do one.", "owner": "agent"},
        {"title": "Two", "instruction": "Do two.", "owner": "agent"},
    ]
    drafted = machine.propose(
        machine.check_draft({"goal": "g", "steps": steps}, limits), "d", ids, 0, None, limits
    )
    return machine.activate(drafted, "d", 0, None), ids, limits


def test_patches_only_the_changed_steps_while_the_list_is_the_same() -> None:
    plan, _, _ = _plan()
    before = view.steps(plan)
    claimed = machine.claim(plan, plan["steps"][0]["id"], "w", 100, 0)
    update = view.update(claimed, before)
    assert update["snapshot"] is False
    assert [step["id"] for step in update["steps"]] == [claimed["steps"][0]["id"]]


def test_sends_a_snapshot_in_order_when_a_step_is_inserted_before_the_current_one() -> None:
    plan, ids, limits = _plan()
    claimed = machine.claim(plan, plan["steps"][0]["id"], "w", 100, 0)
    before = view.steps(claimed)
    paused, _ = machine.pause(
        claimed, {"kind": "input", "question": "Which?"}, None, ids, limits, 0, None
    )
    update = view.update(paused, before)
    assert update["snapshot"] is True
    assert [step["owner"] for step in update["steps"]] == ["human", "agent", "agent"]
