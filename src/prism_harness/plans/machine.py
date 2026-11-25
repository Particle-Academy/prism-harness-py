"""Every transition a plan can make, as pure functions over the stored plan (G-360).

No I/O, no clock, no randomness: the caller passes the time, the id generator and
the facts read from the thread, so the same inputs give the same plan in PHP,
TypeScript and Python. The ``harness-plans`` parity corpus drives these functions
directly. A line-for-line port of the reference's ``PlanMachine``.

A PLAN GRANTS NOTHING. Nothing here runs a tool, decides an approval or touches
the task list. Approvals are core approvals, decided through the runtime; this
only reads which ids are pending or answered, and those ids come from the
persisted thread, never from a tool argument.

Refusals raise :class:`PlanRefused` with a constant message.

Every function takes a plan and returns a new one; the argument is never
changed. The reference's arrays are values, so a refusal raised half way
through a transition leaves the caller's plan as it was, and so does this.

The tool arguments arrive as decoded JSON. The reference decodes JSON objects
to PHP arrays, where an empty object and an empty list are the same thing, so
``{}`` is read as an empty list here wherever the reference would.
"""

from __future__ import annotations

import calendar
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from prism_harness.plans import text as plan_text
from prism_harness.plans import view
from prism_harness.plans.ids import PlanIds, is_step_id
from prism_harness.plans.limits import PlanLimits
from prism_harness.plans.refusal import PlanRefused

__all__ = [
    "CHANGE",
    "COMPLETE",
    "DRAFT",
    "LIVE",
    "PAUSE",
    "TERMINAL",
    "TOOLS",
    "activate",
    "answer",
    "apply_change",
    "cancel",
    "check_draft",
    "check_step_input",
    "claim",
    "head_index",
    "is_live",
    "iso",
    "pause",
    "prepare",
    "propose",
    "propose_change",
    "reconcile",
    "report",
    "resume",
    "stalled",
]

Plan = dict[str, Any]
Step = dict[str, Any]

DRAFT = "plan_draft"
COMPLETE = "plan_complete_step"
PAUSE = "plan_pause_for_person"
CHANGE = "plan_propose_change"

#: Every plan tool name. A pending approval for one of these never holds an agent step.
TOOLS = (DRAFT, COMPLETE, PAUSE, CHANGE)

LIVE = ("proposed", "active", "waiting", "blocked")
TERMINAL = ("done", "failed", "skipped")


# ---------------------------------------------------------------------------
# PHP array semantics, for model-written arguments
# ---------------------------------------------------------------------------


def _is_array(value: object) -> bool:
    return isinstance(value, (dict, list))


def _is_list(value: object) -> bool:
    """``array_is_list``: a list, or a map keyed 0..n-1 in order (``{}`` included)."""
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return [str(key) for key in value] == [str(i) for i in range(len(value))]
    return False


def _values(value: object) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    return []


def _keys(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(i) for i in range(len(value))]
    if isinstance(value, dict):
        return [str(key) for key in value]
    return []


def _has(value: object, key: str) -> bool:
    return key in _keys(value)


def _get(value: object, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _empty(value: object) -> bool:
    return _is_array(value) and len(value) == 0  # type: ignore[arg-type]


def _extra_keys(value: object, allowed: Sequence[str]) -> bool:
    return any(key not in allowed for key in _keys(value))


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


def check_draft(args: object, limits: PlanLimits, authorized: bool = True) -> dict[str, Any]:
    """Check a draft at the tool boundary. Refused, never truncated."""
    if not authorized:
        # The host's call policy refused this call: nothing is checked,
        # recorded or proposed, and the live-plan slot stays free.
        raise PlanRefused("plan.not_authorized")
    if (
        not _is_array(args)
        or not _has(args, "goal")
        or not _has(args, "steps")
        or _extra_keys(args, ("goal", "steps"))
    ):
        raise PlanRefused("plan.shape")
    steps = _get(args, "steps")
    if not _is_array(steps) or (not _empty(steps) and not _is_list(steps)):
        raise PlanRefused("plan.shape")
    goal = _get(args, "goal")
    if not plan_text.acceptable(goal, plan_text.GOAL):
        raise PlanRefused("plan.text")
    count = len(steps)
    if count < 1 or count > limits.max_steps:
        raise PlanRefused("plan.bounds")
    return {"goal": goal, "steps": [check_step_input(step) for step in _values(steps)]}


def check_step_input(step: object) -> dict[str, Any]:
    """One step as the model wrote it: ``{title, instruction, owner, interaction?, question?, options?}``.

    A person's step is a question (``interaction: input``); a person's approval
    mid-plan is a core approval, raised with plan_pause_for_person.
    """
    keys = ("title", "instruction", "owner", "interaction", "question", "options")
    if not _is_array(step) or (not _empty(step) and _is_list(step)) or _extra_keys(step, keys):
        raise PlanRefused("plan.shape")
    owner = _get(step, "owner")
    interaction = _get(step, "interaction")
    question = _get(step, "question")
    options = _get(step, "options")
    instruction = _get(step, "instruction")
    if owner == "agent" and isinstance(owner, str):
        if interaction is not None or question is not None or options is not None:
            raise PlanRefused("plan.shape")
    elif owner == "human" and isinstance(owner, str):
        if interaction != "input" or not isinstance(interaction, str) or question is None:
            raise PlanRefused("plan.shape")
    else:
        raise PlanRefused("plan.shape")
    if options is not None and (
        not _is_array(options)
        or (not _empty(options) and not _is_list(options))
        or len(options) > plan_text.OPTIONS
    ):
        raise PlanRefused("plan.shape")
    title = _get(step, "title")
    if not plan_text.acceptable(title, plan_text.TITLE):
        raise PlanRefused("plan.text")
    if owner == "agent" and not plan_text.acceptable(instruction, plan_text.INSTRUCTION):
        raise PlanRefused("plan.text")
    if (
        owner == "human"
        and instruction is not None
        and not plan_text.acceptable(instruction, plan_text.INSTRUCTION)
    ):
        raise PlanRefused("plan.text")
    if question is not None and not plan_text.acceptable(question, plan_text.QUESTION):
        raise PlanRefused("plan.text")
    for option in _values(options):
        if not plan_text.acceptable(option, plan_text.OPTION):
            raise PlanRefused("plan.text")

    return {
        "title": title,
        "instruction": instruction,
        "owner": owner,
        "interaction": "input" if owner == "human" else None,
        "question": question,
        "options": _values(options),
    }


def propose(
    draft: Mapping[str, Any],
    digest: str,
    ids: PlanIds,
    now: int,
    run_id: str | None,
    limits: PlanLimits,
) -> Plan:
    """A new plan from a checked draft: ``proposed``, revision 1."""
    plan: Plan = {
        "id": ids.plan(),
        "anchorRunId": run_id,
        "revision": 1,
        "status": "proposed",
        "goal": draft["goal"],
        "holdingApprovalId": None,
        "pendingChange": None,
        "blockedReason": None,
        "draftDigest": digest,
        "counters": {"advances": 0, "changes": 0, "settledByTool": 0, "settledWithoutTool": 0},
        "lastRunId": run_id,
        "createdAt": iso(now),
        "updatedAt": iso(now),
        "lastAdvancedAt": None,
        "steps": [],
    }
    for given in draft["steps"]:
        plan["steps"].append(_new_step(ids.step(), given))
    plan = _renumber(plan)
    if view.first_event_bytes(plan) > limits.event_max_bytes:
        raise PlanRefused("plan.size")
    return plan


def activate(plan: Plan, digest: str, now: int, run_id: str | None) -> Plan:
    """The person approved the draft: the approved call ran with its stored
    arguments, and they must be the arguments that were proposed."""
    if plan["status"] != "proposed" or plan["draftDigest"] != digest:
        raise PlanRefused("plan.not_waiting")
    plan = deepcopy(plan)
    plan["status"] = "active"
    plan["holdingApprovalId"] = None
    return _written(plan, now, run_id)


def _reject(plan: Plan, reason: str | None) -> Plan:
    plan["status"] = "rejected"
    plan["blockedReason"] = reason
    plan["holdingApprovalId"] = None
    for i, step in enumerate(list(plan["steps"])):
        if step["state"] not in TERMINAL:
            plan["steps"][i] = _closed(step, "skipped", "host", "rejected")
    return plan


# ---------------------------------------------------------------------------
# Reading the thread: held, superseded, moved, decided
# ---------------------------------------------------------------------------


def reconcile(
    plan: Plan,
    pending: Sequence[Mapping[str, Any]],
    decided: Mapping[str, bool],
    now: int,
    run_id: str | None,
    new_turn: bool = False,
) -> Plan:
    """Bring the plan in line with the thread (G-360 section 1.4).

    ``pending`` lists every approval pending at the transcript tail, with the
    tool it was asked for and the digest of its stored arguments. ``decided``
    maps every answered approval id to whether it was approved. Both are read
    from the persisted thread; nothing here accepts an id from a tool argument
    or from model text.

    ``new_turn``: a turn with a new user message is starting, and every pending
    approval is about to end.
    """
    if plan["status"] not in LIVE:
        return plan
    original = plan
    plan = deepcopy(plan)
    pending_ids = [entry["id"] for entry in pending]

    known = _bound_ids(plan)

    def find(name: str, digest: str | None) -> str | None:
        for entry in pending:
            if entry["name"] == name and entry["digest"] == digest and entry["id"] not in known:
                known.append(entry["id"])
                found: str = entry["id"]
                return found
        return None

    # The draft.
    if plan["status"] == "proposed":
        if plan["holdingApprovalId"] is None:
            plan["holdingApprovalId"] = find(DRAFT, plan["draftDigest"])
        holding = plan["holdingApprovalId"]
        if holding is None:
            plan = _reject(plan, "approval_withdrawn")
        elif holding in decided:
            if decided[holding] is False:
                plan = _reject(plan, None)
        elif holding not in pending_ids:
            plan = _reject(plan, "approval_withdrawn")

    # A proposed change.
    if isinstance(plan["pendingChange"], dict):
        change = plan["pendingChange"]
        if change["approvalId"] is None:
            change["approvalId"] = find(CHANGE, change["digest"])
        plan["pendingChange"] = change
        approval = change["approvalId"]
        if (
            approval is None
            or (approval in decided and decided[approval] is False)
            or (approval not in decided and approval not in pending_ids)
        ):
            plan["pendingChange"] = None
            if plan["status"] == "waiting":
                plan["status"] = "active"

    # A person's approval step, raised with plan_pause_for_person.
    for i, snapshot in enumerate(list(plan["steps"])):
        step = deepcopy(snapshot)
        if step["owner"] != "human" or step["interaction"] != "approve" or step["state"] != "held":
            continue
        if step["approvalIds"] == []:
            bound = find(PAUSE, step["pauseDigest"])
            if bound is None:
                plan["steps"][i] = _closed(step, "failed", "host", "approval_withdrawn")
                plan = _block(plan, "approval_withdrawn")
                continue
            step["approvalIds"] = [bound]
            plan["steps"][i] = step
        approval = step["approvalIds"][0]
        if approval in decided:
            yes = decided[approval]
            step["approvedIds"] = [approval] if yes else []
            plan["steps"][i] = _closed(
                step, "done" if yes else "skipped", "person", None if yes else "declined"
            )
        elif approval not in pending_ids:
            step["supersededApprovalIds"] = [*step["supersededApprovalIds"], approval]
            step["approvalIds"] = []
            plan["steps"][i] = _closed(step, "failed", "host", "approval_withdrawn")
            plan = _block(plan, "approval_withdrawn")

    # The agent's head step and the approvals its turn staged. Not at the start
    # of a turn with a new message: that turn may stage the card the hold moves
    # to, so the hold is judged when the turn is over.
    head = head_index(plan)
    if not new_turn and head is not None and plan["status"] in ("active", "waiting"):
        step = plan["steps"][head]
        if step["owner"] == "agent" and step["state"] in ("running", "held"):
            plan = _reconcile_agent_step(plan, head, pending, pending_ids, decided, list(known))

    plan = _normalised(plan)

    return original if plan == original else _written(plan, now, run_id)


def _reconcile_agent_step(
    plan: Plan,
    index: int,
    pending: Sequence[Mapping[str, Any]],
    pending_ids: Sequence[str],
    decided: Mapping[str, bool],
    known: Sequence[str],
) -> Plan:
    step = deepcopy(plan["steps"][index])
    live = [approval for approval in step["approvalIds"] if approval not in step["approvedIds"]]

    for approval in live:
        if approval in decided and decided[approval] is False:
            # A person refused work this step needed: never shown as done.
            plan["steps"][index] = _closed(step, "failed", "person", "approval_refused")
            return plan
    still_pending: list[str] = []
    superseded: list[str] = []
    for approval in live:
        if approval in decided:
            step["approvedIds"].append(approval)
        elif approval in pending_ids:
            still_pending.append(approval)
        else:
            superseded.append(approval)
    fresh = [
        entry["id"] for entry in pending if entry["name"] not in TOOLS and entry["id"] not in known
    ]
    step["approvalIds"] = [
        *[approval for approval in step["approvalIds"] if approval not in superseded],
        *fresh,
    ]
    step["supersededApprovalIds"] = [*step["supersededApprovalIds"], *superseded]

    if still_pending or fresh:
        step["state"] = "held"
    elif superseded:
        # The person's card was lost and nothing replaced it.
        plan["steps"][index] = _closed(step, "failed", "host", "approval_withdrawn")
        return _block(plan, "approval_withdrawn")
    else:
        step["state"] = "running"
    plan["steps"][index] = step
    return plan


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


#: The reason a plan blocks when a person cancelled its step's run (G-285).
RUN_CANCELLED = "run_cancelled"


def prepare(
    plan: Plan,
    run_failed: bool,
    limits: PlanLimits,
    now: int,
    run_id: str | None,
    budget_exhausted: bool = False,
    run_cancelled: bool = False,
) -> tuple[Plan, list[str | None]]:
    """Settle the previous head from persisted state, check the limits, and
    choose what happens next. Returns the plan and an outcome: ``[run, stepId]``,
    ``[waiting, stepId]``, ``[blocked, reason]``, ``[done, None]`` or
    ``[not_active, status]``."""
    if plan["status"] != "active":
        return plan, ["not_active", plan["status"]]
    original = plan
    # G-285: the head step's own turn was cancelled by a person. The step goes
    # back to todo and the plan stops, once: the next advance does not quietly
    # start a new run the person just asked to stop. resume() continues it.
    head = head_index(plan)
    cancelled_turn = (
        run_cancelled
        and head is not None
        and plan["steps"][head]["owner"] == "agent"
        and plan["steps"][head]["state"] == "running"
    )
    plan = _settle(deepcopy(plan), run_failed or run_cancelled, limits)
    if cancelled_turn and plan["status"] == "active":
        plan = _block(plan, RUN_CANCELLED)

    head = head_index(plan)
    outcome: list[str | None]
    if plan["status"] == "blocked":
        outcome = ["blocked", plan["blockedReason"]]
    elif head is None:
        plan = _normalised(plan)
        outcome = ["done", None]
    else:
        step = plan["steps"][head]
        if step["owner"] == "human" and step["state"] == "todo":
            step = {**step, "state": "waiting_input"}
            plan["steps"][head] = step
            outcome = ["waiting", step["id"]]
        elif budget_exhausted and step["owner"] == "agent" and step["state"] == "held":
            # G-366: a step held on an approval whose resume the spent shared
            # budget could not run stops the plan, once, instead of waiting on a
            # decision that would be refused.
            plan = _block(plan, "plan_budget_exhausted")
            outcome = ["blocked", "plan_budget_exhausted"]
        elif step["state"] != "todo":
            outcome = ["waiting", step["id"]]
        elif plan["counters"]["advances"] >= limits.max_advances:
            plan = _block(plan, "budget")
            outcome = ["blocked", "budget"]
        elif budget_exhausted:
            # G-366: the plan's shared budget is spent. It is only ever
            # lowered, so resume() refuses this as it refuses "budget".
            plan = _block(plan, "plan_budget_exhausted")
            outcome = ["blocked", "plan_budget_exhausted"]
        else:
            outcome = ["run", step["id"]]
    plan = _normalised(plan)

    return (original if plan == original else _written(plan, now, run_id)), outcome


def _settle(plan: Plan, run_failed: bool, limits: PlanLimits) -> Plan:
    head = head_index(plan)
    if head is None:
        return plan
    step = deepcopy(plan["steps"][head])
    if step["owner"] != "agent":
        return plan
    if step["state"] == "running":
        if isinstance(step["report"], dict) and not run_failed:
            step["summary"] = step["report"]["summary"]
            plan["steps"][head] = _closed(step, "done", "agent-report", None)
            plan["counters"]["settledByTool"] += 1
            return plan
        step["state"] = "todo"
        step["claim"] = None
        step["report"] = None
        step["lastSettle"] = "run_failed" if run_failed else "no_report"
        if not run_failed:
            plan["counters"]["settledWithoutTool"] += 1
        plan["steps"][head] = step
    step = plan["steps"][head]
    if step["state"] == "todo" and step["attempts"] >= limits.max_attempts:
        reason = "no_completion" if step["lastSettle"] == "no_report" else "attempts_exhausted"
        plan["steps"][head] = _closed(step, "failed", "host", reason)
        plan = _block(plan, reason)
    return plan


def claim(plan: Plan, step_id: str, worker: str, until: int, now: int) -> Plan:
    """Claim the head step for one turn."""
    head = head_index(plan)
    if (
        plan["status"] != "active"
        or head is None
        or plan["steps"][head]["id"] != step_id
        or plan["steps"][head]["state"] != "todo"
    ):
        raise PlanRefused("plan.conflict")
    plan = deepcopy(plan)
    step = plan["steps"][head]
    step["state"] = "running"
    step["claim"] = {"worker": worker, "until": until}
    step["attempts"] += 1
    step["report"] = None
    plan["counters"]["advances"] += 1
    plan["lastAdvancedAt"] = iso(now)
    return _written(plan, now, None)


def report(plan: Plan, summary: object, now: int, run_id: str | None) -> Plan:
    """plan_complete_step: the agent reports its running step done. It does NOT
    advance; the next advance settles it."""
    head = head_index(plan)
    if (
        plan["status"] not in ("active", "waiting")
        or head is None
        or plan["steps"][head]["owner"] != "agent"
        or plan["steps"][head]["state"] != "running"
    ):
        raise PlanRefused("plan.no_running_step")
    if not plan_text.acceptable(summary, plan_text.SUMMARY):
        raise PlanRefused("plan.text")
    plan = deepcopy(plan)
    plan["steps"][head]["report"] = {"summary": summary}
    return _written(plan, now, run_id)


def pause(
    plan: Plan,
    args: object,
    digest: str | None,
    ids: PlanIds,
    limits: PlanLimits,
    now: int,
    run_id: str | None,
    authorized: bool = True,
) -> tuple[Plan, str | None]:
    """plan_pause_for_person: insert a person's step before the agent's running
    step. A second pause on one step blocks the plan, and the blocked plan is
    returned with the refusal code so it can be written."""
    if not authorized:
        # The host's call policy refused this call: nothing is checked,
        # recorded or proposed, and the live-plan slot stays free.
        raise PlanRefused("plan.not_authorized")
    kind = _get(args, "kind")
    if (
        not _is_array(args)
        or _extra_keys(args, ("kind", "question", "options"))
        or not isinstance(kind, str)
        or kind not in ("approve", "input")
    ):
        raise PlanRefused("plan.shape")
    options = _get(args, "options")
    if options is not None and (
        not _is_array(options)
        or (not _empty(options) and not _is_list(options))
        or len(options) > plan_text.OPTIONS
        or (kind == "approve" and not _empty(options))
    ):
        raise PlanRefused("plan.shape")
    question = _get(args, "question")
    if not plan_text.acceptable(question, plan_text.TITLE):
        raise PlanRefused("plan.text")
    for option in _values(options):
        if not plan_text.acceptable(option, plan_text.OPTION):
            raise PlanRefused("plan.text")
    head = head_index(plan)
    if (
        plan["status"] != "active"
        or head is None
        or plan["steps"][head]["owner"] != "agent"
        or plan["steps"][head]["state"] != "running"
    ):
        raise PlanRefused("plan.no_running_step")
    plan = deepcopy(plan)
    step = plan["steps"][head]
    if step["pauses"] >= limits.max_pauses_per_step:
        plan["steps"][head] = _closed(step, "failed", "host", "pause_limit")
        plan = _block(plan, "pause_limit")
        return _written(_normalised(plan), now, run_id), "plan.pause_limit"
    step["pauses"] += 1
    step["state"] = "todo"
    step["claim"] = None
    step["report"] = None
    step["attempts"] = max(0, step["attempts"] - 1)

    human = _new_step(
        ids.step(),
        {
            "title": question,
            "instruction": None,
            "owner": "human",
            "interaction": kind,
            "question": question,
            "options": _values(options),
        },
    )
    human["state"] = "held" if kind == "approve" else "waiting_input"
    human["pauseDigest"] = digest if kind == "approve" else None
    plan["steps"].insert(head, human)
    if len(plan["steps"]) > limits.max_steps:
        raise PlanRefused("plan.bounds")

    return _written(_normalised(_renumber(plan)), now, run_id), None


def answer(
    plan: Plan, step_id: object, text: object, choice: object, actor: object, now: int
) -> Plan:
    """A person answers a question step. The actor is the host's, never the
    model's. An answer is data for the next turn; it releases no approval."""
    if not isinstance(actor, str) or actor.strip(" \t\n\r\0\x0b") == "":
        raise PlanRefused("plan.no_actor")
    index = None
    for i, step in enumerate(plan["steps"]):
        if step["id"] == step_id and isinstance(step_id, str):
            index = i
    if index is None or plan["status"] not in ("active", "waiting"):
        raise PlanRefused("plan.not_answerable")
    step = deepcopy(plan["steps"][index])
    if (
        step["owner"] != "human"
        or step["interaction"] != "input"
        or step["state"] != "waiting_input"
    ):
        raise PlanRefused("plan.not_answerable")
    if text is None:
        text = ""
    if (
        not isinstance(text, str)
        or not plan_text.valid(text)
        or plan_text.length(text) > plan_text.ANSWER
        or (choice is not None and not (isinstance(choice, str) and choice in step["options"]))
        or (text == "" and choice is None)
    ):
        raise PlanRefused("plan.answer_invalid")
    step["answer"] = {"text": text, "choice": choice, "actor": actor, "at": iso(now)}
    plan = deepcopy(plan)
    plan["steps"][index] = _closed(step, "done", "person", None)
    return _written(_normalised(plan), now, None)


def propose_change(
    plan: Plan,
    args: object,
    digest: str,
    ids: PlanIds,
    limits: PlanLimits,
    now: int,
    run_id: str | None,
    authorized: bool = True,
) -> Plan:
    """plan_propose_change: a new list for the steps that can still be replaced.

    Only ``todo``, agent-owned steps can be replaced; every other step must
    appear unchanged, by id, in its order, with every finished step first. A
    stale revision, an unknown id and a step that cannot be replaced get the
    same refusal.
    """
    if not authorized:
        # The host's call policy refused this call: nothing is checked,
        # recorded or proposed, and the live-plan slot stays free.
        raise PlanRefused("plan.not_authorized")
    given = _get(args, "steps")
    if (
        not _is_array(args)
        or _extra_keys(args, ("reason", "base_revision", "steps"))
        or not _is_array(given)
        or (not _empty(given) and not _is_list(given))
    ):
        raise PlanRefused("plan.shape")
    base = _get(args, "base_revision")
    if plan["status"] != "active" or type(base) is not int or base != plan["revision"]:
        raise PlanRefused("plan.change_refused")
    if plan["counters"]["changes"] >= limits.max_changes:
        raise PlanRefused("plan.change_limit")
    reason = _get(args, "reason")
    if not plan_text.acceptable(reason, plan_text.REASON):
        raise PlanRefused("plan.text")
    entries = _values(given)
    if len(entries) < 1 or len(entries) > limits.max_steps:
        raise PlanRefused("plan.bounds")

    existing: dict[str, Step] = {step["id"]: step for step in plan["steps"]}
    order: list[str] = []
    new: dict[str, Step] = {}
    for entry in entries:
        if isinstance(entry, dict) and "id" in entry and entry["id"] is None:
            entry = {key: value for key, value in entry.items() if key != "id"}
        if isinstance(entry, dict) and "id" in entry:
            rest = [value for key, value in entry.items() if key != "id" and value is not None]
            kept = entry["id"]
            if rest or not is_step_id(kept) or kept not in existing or kept in order:
                raise PlanRefused("plan.change_refused")
            order.append(kept)
            continue
        step = _new_step(ids.step(), check_step_input(entry))
        new[step["id"]] = step
        order.append(step["id"])

    fixed = [step["id"] for step in plan["steps"] if not _replaceable(step)]
    # Every fixed step is kept, in its order.
    if [step_id for step_id in order if step_id in fixed] != fixed:
        raise PlanRefused("plan.change_refused")
    # Finished steps come first; a fixed head stays the head.
    seen_open = False
    for step_id in order:
        terminal = step_id in existing and existing[step_id]["state"] in TERMINAL
        if terminal and seen_open:
            raise PlanRefused("plan.change_refused")
        if not terminal:
            if not seen_open:
                head = head_index(plan)
                if (
                    head is not None
                    and not _replaceable(plan["steps"][head])
                    and plan["steps"][head]["id"] != step_id
                ):
                    raise PlanRefused("plan.change_refused")
            seen_open = True
    replaced = [step_id for step_id in existing if step_id not in order]
    if len(order) + len(replaced) > limits.max_steps:
        raise PlanRefused("plan.bounds")

    plan = deepcopy(plan)
    plan["pendingChange"] = {
        "approvalId": None,
        "baseRevision": plan["revision"],
        "reason": reason,
        "digest": digest,
        "order": order,
        "newSteps": list(new.values()),
    }
    plan["status"] = "waiting"
    plan["counters"]["changes"] += 1
    if view.first_event_bytes(plan) > limits.event_max_bytes:
        raise PlanRefused("plan.size")

    return _written(plan, now, run_id)


def apply_change(plan: Plan, digest: str, now: int, run_id: str | None) -> Plan:
    """The person approved the change: the approved call ran with its stored
    arguments, which must be the proposed ones."""
    change = plan["pendingChange"]
    if plan["status"] != "waiting" or not isinstance(change, dict) or change["digest"] != digest:
        raise PlanRefused("plan.not_waiting")
    plan = deepcopy(plan)
    change = plan["pendingChange"]
    by_id: dict[str, Step] = {}
    for step in [*plan["steps"], *change["newSteps"]]:
        by_id[step["id"]] = step
    steps = [by_id[step_id] for step_id in change["order"]]
    for step in plan["steps"]:
        if step["id"] not in change["order"]:
            steps.append(_closed(step, "skipped", "host", "replaced"))
    plan["steps"] = steps
    plan["pendingChange"] = None
    plan["status"] = "active"

    return _written(_normalised(_renumber(plan)), now, run_id)


def resume(plan: Plan, ids: PlanIds, limits: PlanLimits, now: int) -> Plan:
    """The host unblocks a plan. A step that failed for lack of a person's card
    or of attempts gets a retry step after it; a step that paused too often is
    not replayed; a plan that ran out of advances stays blocked."""
    if plan["status"] != "blocked" or plan["blockedReason"] in ("budget", "plan_budget_exhausted"):
        raise PlanRefused("plan.not_resumable")
    plan = deepcopy(plan)
    reason = plan["blockedReason"]
    if reason in ("attempts_exhausted", "no_completion", "approval_withdrawn"):
        failed = None
        for i, step in enumerate(plan["steps"]):
            if step["owner"] == "agent" and step["state"] == "failed" and step["failure"] == reason:
                failed = i
        if failed is not None:
            if len(plan["steps"]) + 1 > limits.max_steps:
                raise PlanRefused("plan.bounds")
            old = plan["steps"][failed]
            retry = _new_step(
                ids.step(),
                {
                    "title": old["title"],
                    "instruction": old["instruction"],
                    "owner": "agent",
                    "interaction": None,
                    "question": None,
                    "options": [],
                },
            )
            retry["retryOf"] = old["id"]
            plan["steps"].insert(failed + 1, retry)
    plan["status"] = "active"
    plan["blockedReason"] = None

    return _written(_normalised(_renumber(plan)), now, None)


def cancel(plan: Plan, now: int) -> Plan:
    if plan["status"] not in LIVE:
        raise PlanRefused("plan.not_live")
    plan = deepcopy(plan)
    for i, step in enumerate(list(plan["steps"])):
        if step["state"] not in TERMINAL:
            plan["steps"][i] = _closed(step, "skipped", "host", "cancelled")
    plan["pendingChange"] = None
    plan["holdingApprovalId"] = None
    plan["status"] = "cancelled"
    return _written(plan, now, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def head_index(plan: Mapping[str, Any]) -> int | None:
    for i, step in enumerate(plan["steps"]):
        if step["state"] not in TERMINAL:
            return i
    return None


def stalled(plan: Mapping[str, Any], now: int, stale_after: int) -> bool:
    """Whether the reaper should hand this plan back to the host: it is active,
    its head is the agent's step with no live claim (todo, or running on a lease
    that ran out), and it has not moved for ``stale_after`` seconds. A step
    waiting on a person is never stalled."""
    head = head_index(plan)
    if plan["status"] != "active" or head is None:
        return False
    step = plan["steps"][head]
    claim_live = isinstance(step["claim"], dict) and step["claim"]["until"] > now
    idle = (step["owner"] == "agent" and step["state"] == "todo") or (
        step["state"] == "running" and not claim_live
    )
    moved = plan.get("lastAdvancedAt")
    last = _parse_iso(str(moved if moved is not None else plan.get("updatedAt")))
    return idle and (last is None or now - last >= stale_after)


def is_live(plan: object) -> bool:
    return isinstance(plan, dict) and plan.get("status") in LIVE


def iso(seconds: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def _parse_iso(value: str) -> int | None:
    try:
        return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def _new_step(step_id: str, given: Mapping[str, Any]) -> Step:
    return {
        "id": step_id,
        "position": 0,
        "title": given["title"],
        "instruction": given["instruction"],
        "owner": given["owner"],
        "interaction": given["interaction"],
        "question": given["question"],
        "options": list(given["options"]),
        "state": "todo",
        "approvalIds": [],
        "approvedIds": [],
        "supersededApprovalIds": [],
        "pauseDigest": None,
        "claim": None,
        "attempts": 0,
        "pauses": 0,
        "lastSettle": None,
        "report": None,
        "summary": None,
        "answer": None,
        "completedBy": None,
        "failure": None,
        "retryOf": None,
    }


def _replaceable(step: Mapping[str, Any]) -> bool:
    return bool(step["owner"] == "agent" and step["state"] == "todo")


def _closed(step: Mapping[str, Any], state: str, by: str, failure: str | None) -> Step:
    closed = deepcopy(dict(step))
    closed["state"] = state
    closed["completedBy"] = by
    closed["failure"] = failure
    closed["claim"] = None
    closed["report"] = None
    return closed


def _block(plan: Plan, reason: str) -> Plan:
    plan["status"] = "blocked"
    plan["blockedReason"] = reason
    return plan


def _renumber(plan: Plan) -> Plan:
    for i, step in enumerate(plan["steps"]):
        step["position"] = i + 1
    return plan


def _normalised(plan: Plan) -> Plan:
    """Finish an active plan whose steps are all closed, and point
    holdingApprovalId at the one approval that is actually holding it."""
    if (
        plan["status"] in ("active", "waiting")
        and plan["pendingChange"] is None
        and head_index(plan) is None
    ):
        plan["status"] = "done"
    if plan["status"] == "proposed":
        return plan
    holding = None
    change = plan["pendingChange"]
    if isinstance(change, dict) and change["approvalId"] is not None:
        holding = change["approvalId"]
    else:
        head = head_index(plan)
        if head is not None and plan["steps"][head]["state"] == "held":
            step = plan["steps"][head]
            open_ids = [a for a in step["approvalIds"] if a not in step["approvedIds"]]
            holding = open_ids[0] if open_ids else None
    plan["holdingApprovalId"] = holding if plan["status"] in ("active", "waiting") else None
    return plan


def _bound_ids(plan: Mapping[str, Any]) -> list[str]:
    """Every approval id the plan is already bound to."""
    ids: list[str] = []
    if isinstance(plan["holdingApprovalId"], str):
        ids.append(plan["holdingApprovalId"])
    change = plan["pendingChange"]
    if isinstance(change, dict) and isinstance(change["approvalId"], str):
        ids.append(change["approvalId"])
    for step in plan["steps"]:
        ids.extend(step["approvalIds"])
        ids.extend(step["supersededApprovalIds"])
    return list(dict.fromkeys(ids))


def _written(plan: Plan, now: int, run_id: str | None) -> Plan:
    plan["revision"] += 1
    plan["updatedAt"] = iso(now)
    plan["lastRunId"] = run_id
    return plan


#: For a caller that transforms a plan with one of the functions above.
Transition = Callable[[Plan], Plan]
