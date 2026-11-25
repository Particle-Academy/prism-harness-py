"""One session's plan, for the host (G-360)."""

from __future__ import annotations

import secrets
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prism_harness.errors import HarnessError
from prism_harness.plans import budget, machine, view
from prism_harness.plans.refusal import PlanRefused

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_harness.plans.service import Plans
    from prism_harness.runtime import AgentRuntime
    from prism_harness.session import Session
    from prism_harness.subagents import RunContext

__all__ = ["AdvanceOutcome", "PlanHandle", "PlanSnapshot"]


@dataclass(frozen=True)
class AdvanceOutcome:
    """What one :meth:`PlanHandle.advance` did.

    ``ran``: one turn ran on ``step_id``, as run ``run_id``. Call advance()
    again to settle it and run the next step.
    ``waiting``: the head step waits on a person (an approval or an answer).
    ``blocked``: the plan stopped on a limit; ``reason`` says which.
    ``done``: every step is closed.
    ``not_active``: there is no active plan; ``plan_status`` says what there is.
    """

    status: str
    plan_status: str | None
    step_id: str | None = None
    run_id: str | None = None
    reason: str | None = None

    def ran(self) -> bool:
        return self.status == "ran"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "planStatus": self.plan_status,
            "stepId": self.step_id,
            "runId": self.run_id,
            "reason": self.reason,
        }


class PlanSnapshot:
    """The whole plan as stored, for the host's fetch-on-change endpoint.

    This is the one place the instruction, the step notes, the questions and the
    answers are available. Events carry ids, states and counts only. Whoever
    serves this to a client decides who may read it.
    """

    def __init__(self, plan: dict[str, Any]) -> None:
        self._plan = deepcopy(plan)

    @property
    def id(self) -> str:
        plan_id: str = self._plan["id"]
        return plan_id

    @property
    def status(self) -> str:
        status: str = self._plan["status"]
        return status

    @property
    def revision(self) -> int:
        revision: int = self._plan["revision"]
        return revision

    @property
    def holding_approval_id(self) -> str | None:
        holding: str | None = self._plan["holdingApprovalId"]
        return holding

    @property
    def blocked_reason(self) -> str | None:
        reason: str | None = self._plan["blockedReason"]
        return reason

    def steps(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._plan["steps"]))

    def counts(self) -> dict[str, int]:
        return view.counts(self._plan)

    def to_dict(self) -> dict[str, Any]:
        """The plan, less the argument digests it binds approvals with."""
        plan = deepcopy(self._plan)
        plan.pop("draftDigest", None)
        if isinstance(plan.get("pendingChange"), dict):
            plan["pendingChange"].pop("digest", None)
        for step in plan["steps"]:
            step.pop("pauseDigest", None)
        return plan


class PlanHandle:
    """One session's plan, for the host.

    :meth:`advance` moves the plan at most ONE step: it settles the previous
    step from what the thread and the run record say, checks the limits, and
    runs one turn on the next step. Settling never trusts a return value and
    never reads the model's prose: a step is done only when the agent called
    plan_complete_step and the turn ended cleanly with nothing pending.

    :meth:`answer_step`, :meth:`resume` and :meth:`cancel` are host calls. WHO
    MAY MAKE THEM IS THE APPLICATION'S DECISION, as with approvals: authorize
    first, and name the person as ``actor``. The actor is never taken from the
    model.
    """

    def __init__(self, session: Session, plans: Plans, runtime: AgentRuntime) -> None:
        self._session = session
        self._plans = plans
        self._runtime = runtime

    def current(self) -> PlanSnapshot | None:
        """The whole plan, the target of fetch-on-change; None when there is none."""
        plan = self._plans.load(self._session)
        return None if plan is None else PlanSnapshot(plan)

    def context(self) -> str:
        """The plan context block as the model would read it now; '' when there is no live plan."""
        plan = self._plans.load(self._session)
        return "" if plan is None else view.context(plan, self._plans.limits())

    def advance(self, context: RunContext | None = None) -> AdvanceOutcome:
        """Settle, choose, and run at most one turn, under the session's plan lock.

        A second advance while one holds the lock raises ``HarnessError`` with
        code ``session_locked`` and does nothing.
        """
        release = self._session.hold_plan_lock(self._lock_ttl())
        try:
            outcome, step = self._prepare()
            if step is None:
                return outcome
            # This turn, and only this one, is the step's own (G-366).
            serving = self._plans.serving_step(self._session)
            try:
                response = self._runtime.send(self._session, _prompt(step), context=context)
            except HarnessError as error:
                stopped = (
                    self._stopped_on_budget()
                    if error.code in ("run_not_permitted", budget.EXHAUSTED)
                    else None
                )
                if stopped is None:
                    raise
                return stopped
            finally:
                serving()
            return AdvanceOutcome("ran", "active", step["id"], response.run_id)
        finally:
            release()

    def continue_plan(self, context: RunContext | None = None) -> AdvanceOutcome:
        """An alias of :meth:`advance`, the name the request used."""
        return self.advance(context)

    def use_budget(self, requested: dict[str, Any]) -> None:
        """Choose this plan's budget (G-366), in host code: ``shared`` with
        ``max_steps`` and optionally ``max_cost_usd``, ``per_step``, or ``none``.
        Each limit is the lower of this and the configured one. The first
        choice, or the plan's first shared turn, fixes the budget: after that
        the mode cannot change (``plan.budget_fixed``) and a limit can only be
        lowered. Never reachable from a tool or the model."""
        plan = self._plans.load(self._session)
        if plan is None or not machine.is_live(plan):
            raise PlanRefused("plan.no_plan")
        self._plans.use_budget(self._session, plan["id"], requested)

    def budget(self) -> dict[str, Any] | None:
        """The plan's budget as it stands: mode, limits and what a shared budget
        has spent. None when there is no plan."""
        plan = self._plans.load(self._session)
        if plan is None:
            return None
        chosen = self._plans.budget_for(self._session, plan["id"])
        record = self._plans.budget_record(self._session, plan["id"]) or {}
        cost = chosen["max_cost_micros"]
        return {
            "mode": chosen["mode"],
            "max_steps": chosen["max_steps"],
            "max_cost_usd": None if cost is None else cost / 1_000_000,
            "steps": int(record.get("steps", 0)),
            "cost_usd": int(record.get("cost_micros", 0)) / 1_000_000,
        }

    def answer_step(self, step_id: str, text: str, choice: str | None, actor: str) -> PlanSnapshot:
        """A person answers a question step. The answer is data the agent reads on
        its next turn; it is never an approval and releases nothing held."""
        plan = self._plans.mutate(
            self._session,
            lambda current: machine.answer(
                current, step_id, text, choice, actor, self._plans.now()
            ),
        )
        if plan is None:
            raise PlanRefused("plan.not_answerable")
        return PlanSnapshot(plan)

    def resume(self, actor: str) -> PlanSnapshot:
        """The host unblocks the plan. See :func:`machine.resume` for what is retried."""
        _actor(actor)
        plan = self._plans.mutate(
            self._session,
            lambda current: machine.resume(
                current, self._plans.ids(), self._plans.limits(), self._plans.now()
            ),
        )
        if plan is None:
            raise PlanRefused("plan.not_resumable")
        return PlanSnapshot(plan)

    def cancel(self, actor: str) -> PlanSnapshot:
        """The host stops the plan: every open step is skipped."""
        _actor(actor)
        plan = self._plans.mutate(
            self._session, lambda current: machine.cancel(current, self._plans.now())
        )
        if plan is None:
            raise PlanRefused("plan.not_live")
        return PlanSnapshot(plan)

    def _prepare(self) -> tuple[AdvanceOutcome, dict[str, Any] | None]:
        plan = self._plans.reconciled(self._session)
        if plan is None:
            return AdvanceOutcome("not_active", None), None
        run = self._session.run()
        run_failed = isinstance(run, dict) and run.get("status") == "failed"
        now = self._plans.now()
        spent = (
            self._plans.budget_exhaustion(self._session, plan["id"], self._lock_ttl())
            if plan["status"] == "active"
            else None
        )
        following, (kind, value) = machine.prepare(
            plan, run_failed, self._plans.limits(), now, None, spent is not None
        )
        if following != plan:
            self._plans.write(self._session, following, plan["revision"])
        if kind == "blocked" and value == budget.EXHAUSTED:
            self._plans.announce_budget_exhausted(self._session, plan["id"], str(spent))
        if kind != "run":
            status = following["status"]
            if kind == "waiting":
                return AdvanceOutcome("waiting", status, value), None
            if kind == "blocked":
                return AdvanceOutcome("blocked", status, reason=value), None
            if kind == "done":
                return AdvanceOutcome("done", status), None
            return AdvanceOutcome("not_active", status), None
        claimed = machine.claim(
            following,
            str(value),
            "advance_" + secrets.token_hex(6),
            now + self._lock_ttl(),
            now,
        )
        self._plans.write(self._session, claimed, following["revision"])
        for step in claimed["steps"]:
            if step["id"] == value:
                return AdvanceOutcome("ran", "active", value), step
        raise PlanRefused("plan.conflict")

    def _stopped_on_budget(self) -> AdvanceOutcome | None:
        """A turn refused because the plan's shared budget is spent stops the
        plan cleanly; None when the budget is not what stopped it."""
        plan = self._plans.load(self._session)
        if (
            plan is None
            or plan["status"] != "active"
            or self._plans.budget_exhaustion(self._session, plan["id"], self._lock_ttl()) is None
        ):
            return None
        return self._prepare()[0]

    def _lock_ttl(self) -> int:
        ttl = self._runtime.lock_ttl(self._session)
        return ttl if ttl > 0 else 300


def _prompt(step: dict[str, Any]) -> str:
    # The host's fixed words. The step's own instruction reaches the model only
    # through the fenced plan context, never as the user turn.
    return f"Continue the plan. Work on step {int(step['position'])}."


def _actor(actor: object) -> None:
    if not isinstance(actor, str) or actor.strip(" \t\n\r\0\x0b") == "":
        raise PlanRefused("plan.no_actor")
