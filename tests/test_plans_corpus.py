"""The cross-language plans corpus (G-360, prism-parity suites/harness-plans).

Ports prism-harness's tests/PlanCorpusTest.php: each case runs its ops through
the pure plan machine and view, and the outcome is compared, as JSON text so a
list and a map never compare equal by accident, with what the PHP reference
recorded. With HARNESS_PLANS_RECORD set to a path, the outcomes are written
there instead of compared.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from prism_harness import HarnessError, ToolPolicy, binding_route
from prism_harness.plans import (
    PlanBudgetInvalid,
    PlanIds,
    PlanLimits,
    PlanRefused,
    budget,
    machine,
    view,
)
from prism_harness.plans import text as plan_text

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "harness-plans.json").read_text(encoding="utf-8")
)
CASES: list[dict[str, Any]] = CORPUS["cases"]
RUN = "run_case"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A test tried to use the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    given = case.get("limits") or {}
    limits = PlanLimits(
        max_steps=given.get("max_steps", 20),
        max_attempts=given.get("max_attempts", 3),
        max_pauses_per_step=given.get("max_pauses_per_step", 1),
        max_changes=given.get("max_changes", 5),
        max_advances=given.get("max_advances", 60),
        context_max_chars=given.get("context_max_chars", 2000),
        context_window=given.get("context_window", 8),
        event_max_bytes=given.get("event_max_bytes", 8192),
    )
    counter = [0]

    def mint(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]:026d}"

    ids = PlanIds(mint)
    state: dict[str, Any] = {"plan": None, "marked": None, "ledger": None, "saved_binding": False}
    results: list[dict[str, Any]] = []
    for op in case["ops"]:
        now = op.get("now", case.get("now", 1767225600))
        try:
            if op["op"] == "binding_route":
                results.append(plans_binding_op(op, state))
            elif str(op["op"]).startswith("budget_"):
                results.append(plans_budget_op(op, state, now))
            else:
                results.append(plans_op(op, state, ids, limits, now))
        except PlanRefused as refused:
            results.append({"refused": refused.code})
    plan = state["plan"]
    out: dict[str, Any] = {
        "results": results,
        "final": None if plan is None else plans_final(plan, limits),
    }
    if state["ledger"] is not None:
        out["budget"] = plans_budget_final(state["ledger"])
    return out


def plans_binding_op(op: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Which binding a turn runs on, and whether it is saved (binding_route).
    Whether a binding is saved carries from op to op, as it does on a session."""
    saved = bool(op.get("saved_binding", state["saved_binding"]))
    try:
        route = binding_route(op["context"], op["supplied_binding"], saved, op["linked"])
    except ValueError:
        return {"invalid": True}
    state["saved_binding"] = saved or route["save"]
    return {"route": route}


def plans_budget_op(op: dict[str, Any], state: dict[str, Any], now: int) -> dict[str, Any]:
    """The plan budget ops (G-366): settings and the shared ledger, pure."""
    kind = op["op"]
    try:
        if kind == "budget_settings":
            return {"settings": budget.settings(op["config"])}
        if kind == "budget_for_plan":
            return {"settings": budget.for_plan(budget.settings(op["configured"]), op["requested"])}
        if kind == "budget_open":
            state["ledger"] = budget.open_ledger("plan", budget.settings(op["config"]))
            return {"ok": True}
        if kind == "budget_narrow":
            state["ledger"] = budget.narrow(
                state["ledger"], budget.for_plan(budget.settings(None), op["requested"])
            )
            return {"ok": True}
        if kind == "budget_reserve":
            state["ledger"], grant = budget.reserve(
                state["ledger"], op["hold"], op["steps"], op["cost_micros"], now, op.get("ttl", 300)
            )
            return grant if "refused" in grant else {"grant": grant}
        if kind == "budget_settle":
            state["ledger"] = budget.settle(
                state["ledger"], op["hold"], op["steps"], op["cost_micros"]
            )
            return {"ok": True}
        if kind == "budget_exhaustion":
            return {"exhaustion": budget.exhaustion(state["ledger"])}
        if kind == "budget_route":
            return {
                "route": budget.route(
                    op["context"],
                    op["prompt"],
                    op["advancing"],
                    op["step_approval"],
                    op["saved_tree"],
                    op["saved_plan_ledger"],
                )
            }
    except PlanBudgetInvalid:
        return {"invalid": True}
    raise AssertionError(f"Unknown op {kind}")


def plans_budget_final(ledger: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": ledger["mode"],
        "max_steps": ledger["max_steps"],
        "max_cost_micros": ledger["max_cost_micros"],
        "steps": ledger["steps"],
        "cost_micros": ledger["cost_micros"],
        "unmetered": ledger["unmetered"],
        "holds": [
            {
                "id": key,
                "steps": hold["steps"],
                "cost_micros": hold["cost_micros"],
                "at": hold["at"],
            }
            for key, hold in ledger["holds"].items()
        ],
        "exhaustion": budget.exhaustion(ledger),
    }


def plans_op(
    op: dict[str, Any], state: dict[str, Any], ids: PlanIds, limits: PlanLimits, now: int
) -> dict[str, Any]:
    def needs() -> dict[str, Any]:
        if state["plan"] is None:
            raise PlanRefused("plan.no_plan")
        plan: dict[str, Any] = state["plan"]
        return plan

    kind = op["op"]
    if kind == "draft":
        draft = machine.check_draft(op["args"], limits, op.get("authorized", True))
        if machine.is_live(state["plan"]):
            raise PlanRefused("plan.exists")
        state["plan"] = machine.propose(draft, op["digest"], ids, now, RUN, limits)
        return {"ok": True}
    if kind == "activate":
        state["plan"] = machine.activate(needs(), op["digest"], now, RUN)
        return {"ok": True}
    if kind == "reconcile":
        decided_in = op["decided"]
        decided = (
            {str(key): value for key, value in decided_in.items()}
            if isinstance(decided_in, dict)
            else {}
        )
        state["plan"] = machine.reconcile(
            needs(), op["pending"], decided, now, None, op.get("new_turn", False)
        )
        return {"ok": True}
    if kind == "prepare":
        exhausted = state["ledger"] is not None and budget.exhaustion(state["ledger"]) is not None
        state["plan"], outcome = machine.prepare(
            needs(), op["run_failed"], limits, now, None, exhausted
        )
        return {"outcome": outcome}
    if kind == "claim":
        current = needs()
        head = machine.head_index(current)
        if head is None:
            raise PlanRefused("plan.conflict")
        state["plan"] = machine.claim(
            current, current["steps"][head]["id"], "worker", op["until"], now
        )
        return {"ok": True}
    if kind == "report":
        state["plan"] = machine.report(needs(), op["summary"], now, RUN)
        return {"ok": True}
    if kind == "pause":
        state["plan"], code = machine.pause(
            needs(), op["args"], op["digest"], ids, limits, now, RUN, op.get("authorized", True)
        )
        return {"ok": True} if code is None else {"refused": code}
    if kind == "answer":
        current = needs()
        steps = current["steps"]
        index = op["step"]
        step_id = steps[index]["id"] if 0 <= index < len(steps) else "ps_missing"
        state["plan"] = machine.answer(current, step_id, op["text"], op["choice"], op["actor"], now)
        return {"ok": True}
    if kind == "change":
        current = needs()
        args = dict(op["args"])
        if args.get("base_revision") == "current":
            args["base_revision"] = current["revision"]
        args["steps"] = [
            {"id": current["steps"][entry["ref"]]["id"]}
            if isinstance(entry, dict) and "ref" in entry
            else entry
            for entry in args["steps"]
        ]
        state["plan"] = machine.propose_change(
            current, args, op["digest"], ids, limits, now, RUN, op.get("authorized", True)
        )
        return {"ok": True}
    if kind == "apply_change":
        state["plan"] = machine.apply_change(needs(), op["digest"], now, RUN)
        return {"ok": True}
    if kind == "resume":
        state["plan"] = machine.resume(needs(), ids, limits, now)
        return {"ok": True}
    if kind == "cancel":
        state["plan"] = machine.cancel(needs(), now)
        return {"ok": True}
    if kind == "mark":
        state["marked"] = view.steps(needs())
        return {"ok": True}
    if kind == "update_since_mark":
        return {"update": view.update(needs(), state["marked"])}
    if kind == "stalled":
        return {"stalled": machine.stalled(needs(), now, op["stale_after"])}
    if kind == "policy":
        try:
            ToolPolicy.from_json({"revision": "r1", "allow": [{"tool": op["name"]}], "honor": True})
        except HarnessError as error:
            if error.code != "tool_policy_refused":
                raise
            return {"refused": True}
        return {"refused": False}
    if kind == "render":
        return {"rendered": plan_text.render(op["value"], op["max"])}
    raise AssertionError(f"Unknown op {kind}")


def plans_final(plan: dict[str, Any], limits: PlanLimits) -> dict[str, Any]:
    steps = [
        {
            "id": step["id"],
            "position": step["position"],
            "title": step["title"],
            "owner": step["owner"],
            "interaction": step["interaction"],
            "state": step["state"],
            "attempts": step["attempts"],
            "pauses": step["pauses"],
            "approvalIds": step["approvalIds"],
            "approvedIds": step["approvedIds"],
            "supersededApprovalIds": step["supersededApprovalIds"],
            "completedBy": step["completedBy"],
            "failure": step["failure"],
            "retryOf": step["retryOf"],
            "summary": step["summary"],
            "answer": step["answer"],
        }
        for step in plan["steps"]
    ]
    change = plan["pendingChange"]
    return {
        "status": plan["status"],
        "revision": plan["revision"],
        "holdingApprovalId": plan["holdingApprovalId"],
        "blockedReason": plan["blockedReason"],
        "counters": plan["counters"],
        "pendingChange": None
        if change is None
        else {
            "approvalId": change["approvalId"],
            "order": change["order"],
            "newSteps": [step["id"] for step in change["newSteps"]],
        },
        "steps": steps,
        "update": view.update(plan, None),
        "counts": view.counts(plan),
        "context": view.context(plan, limits),
        "first_event_bytes": view.first_event_bytes(plan),
    }


def _text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


RECORD = os.environ.get("HARNESS_PLANS_RECORD", "")

if RECORD:

    def test_records_the_plans_corpus() -> None:
        out = {case["id"]: run_case(case) for case in CASES}
        Path(RECORD).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        assert out

else:

    def test_the_corpus_has_every_case() -> None:
        assert len(CASES) == 58

    @pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
    def test_every_plans_corpus_row_as_the_reference_recorded_it(case: dict[str, Any]) -> None:
        assert _text(run_case(case)) == _text(case["expected"])
