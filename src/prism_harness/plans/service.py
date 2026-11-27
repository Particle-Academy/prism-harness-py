"""Agent plans (G-360): the store, the agent's tools, and the hooks the runtime
calls at the moments a plan has to look at the thread.

OFF UNLESS BOTH SWITCHES ARE ON: the host's ``enabled`` and the mode's ``plans``
flag. Off, no plan tool is offered, no context is added and
``runtime.plan(session)`` raises :class:`PlansDisabled`.

A person approves a plan before it runs. That is on by default, and only the
host's config (``require_approval``) turns it off: never a per-call option,
never anything the model can send.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from prism_harness.events import HarnessEvents, PlanBudgetExhausted, PlanUpdated
from prism_harness.plans import budget, machine, view
from prism_harness.plans.ids import PlanIds
from prism_harness.plans.limits import PlanLimits
from prism_harness.plans.refusal import PlanRefused, PlansDisabled
from prism_harness.plans.thread_facts import approval_decisions, pending_approval_calls
from prism_harness.run_trees import saved_refs
from prism_harness.stores.base import SessionStore
from prism_harness.thread_rows import paused_approval_row

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_harness.modes import ModeRegistry
    from prism_harness.plans.handle import PlanHandle
    from prism_harness.plans.tools import PlanTool
    from prism_harness.runtime import AgentRuntime
    from prism_harness.session import Session
    from prism_harness.subagents import RunContext

__all__ = ["HEARTBEAT_KEY", "INDEX_KEY", "Plans"]

INDEX_KEY = "prism-harness:plans:live"
HEARTBEAT_KEY = "prism-harness:plans:heartbeat"

_log = logging.getLogger("prism_harness.plans")

Plan = dict[str, Any]

#: The shared plan ledger the plan turn running in this context spends from,
#: so a subagent child that turn starts can carry it on its paused row (G-366).
_ACTIVE_LEDGER: ContextVar[dict[str, str] | None] = ContextVar(
    "prism_harness_plan_ledger", default=None
)


class Plans:
    """The host's plan settings and the plan store, shared by every session.

    ``config`` takes the keys the reference reads from ``prism-harness.plans``::

        {"enabled": True, "require_approval": True, "max_steps": 20,
         "max_attempts": 3, "max_pauses_per_step": 1, "max_changes": 5,
         "max_advances": 60, "context": {"max_chars": 2000, "window": 8},
         "events": {"max_bytes": 8192}, "reaper": {"stale_after": 300}}

    A value outside its range, or of the wrong type, is refused here, when the
    host builds it, rather than on the first plan.

    ``index_store`` is the DURABLE store the sessions use (for a
    :class:`~prism_harness.PrismHarness`, ``harness.durable_store()``). The
    reaper reads the live plans from it and writes its heartbeat there. Without
    it, :meth:`sweep` finds nothing and the doctor's reaper check fails.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        index_store: SessionStore | None = None,
        ids: PlanIds | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._config = deepcopy(dict(config or {}))
        enabled = self._config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("plans.enabled must be true or false.")
        self._enabled = enabled
        self._limits = PlanLimits.from_config(self._config)
        # G-366: an invalid plan budget fails here, before anything runs.
        self._budget = budget.settings(self._config.get("budget"))
        #: Turns spending from a shared plan budget, by run id, until they settle.
        self._budget_turns: dict[str, dict[str, Any]] = {}
        #: Sessions whose advance() is running the turn of the step it claimed.
        self._advancing: dict[str, int] = {}
        self._index_store = index_store
        self._ids = ids
        self._clock = clock
        self._modes: ModeRegistry | None = None
        self._events: HarnessEvents | None = None
        self._reap_dispatcher: Callable[[str, str], None] | None = None

    def _attach(self, modes: ModeRegistry, events: HarnessEvents | None) -> None:
        """Called by the runtime that owns these plans."""
        self._modes = modes
        self._events = events

    # -- settings --------------------------------------------------------

    def enabled(self) -> bool:
        return self._enabled

    def limits(self) -> PlanLimits:
        return self._limits

    def enabled_for(self, session: Session, mode: str | None = None) -> bool:
        """Whether this session's current mode runs with plans."""
        if not self._enabled or self._modes is None:
            return False
        try:
            return (
                self._modes.for_agent(session.agent)
                .resolve(mode if mode is not None else session.mode())
                .plans
                is True
            )
        except Exception:  # noqa: BLE001 - a mode that does not resolve has no plans
            return False

    def handle(self, session: Session, runtime: AgentRuntime) -> PlanHandle:
        from prism_harness.plans.handle import PlanHandle

        if not self.enabled_for(session):
            raise PlansDisabled()
        return PlanHandle(session, self, runtime)

    def ids(self) -> PlanIds:
        return self._ids if self._ids is not None else PlanIds()

    def use_clock(self, clock: Callable[[], int] | None) -> None:
        """Pin the clock, in tests."""
        self._clock = clock

    def now(self) -> int:
        return self._clock() if self._clock is not None else int(time.time())

    # -- store -----------------------------------------------------------

    def load(self, session: Session) -> Plan | None:
        payload = session.plan_store().get(_key(session.key()))
        plan = payload.get("plan") if isinstance(payload, dict) else None
        return plan if isinstance(plan, dict) else None

    def write(self, session: Session, plan: Plan, expected_revision: int | None) -> None:
        """Write a plan, compare-and-set on its revision. Refused with
        ``plan.conflict`` when the stored revision is not the one read."""
        store = session.plan_store()
        key = _key(session.key())

        def swap() -> None:
            stored = store.get(key)
            current = stored.get("plan") if isinstance(stored, dict) else None
            revision = current.get("revision") if isinstance(current, dict) else None
            if revision != expected_revision and not (
                expected_revision is None
                and isinstance(current, dict)
                and not machine.is_live(current)
            ):
                raise PlanRefused("plan.conflict")
            store.put(key, {"plan": plan})

        store.with_lock(key, swap, 10, 5)
        self._indexed(session.key(), plan)
        self._announce(session, plan)

    def mutate(self, session: Session, change: Callable[[Plan], Plan]) -> Plan | None:
        """Load, transform, and write when the transform changed anything."""
        plan = self.load(session)
        if plan is None:
            return None
        following = change(plan)
        if following != plan:
            self.write(session, following, plan["revision"])
        return following

    def reconciled(self, session: Session, superseding: bool = False) -> Plan | None:
        """The plan brought in line with the thread, written when it moved."""
        plan = self.load(session)
        if not machine.is_live(plan):
            return plan
        rows = [entry.message for entry in session.thread().messages()]
        pending = [] if superseding else pending_approval_calls(rows)
        decided = approval_decisions(rows)
        return self.mutate(
            session,
            lambda current: machine.reconcile(
                current, pending, decided, self.now(), self.run_id(session), superseding
            ),
        )

    def _indexed(self, session_key: str, plan: Plan) -> None:
        store = self._index_store
        if store is None:
            return

        def index() -> None:
            live = dict(store.get(INDEX_KEY) or {})
            if machine.is_live(plan):
                live[session_key] = plan["id"]
            else:
                live.pop(session_key, None)
            store.put(INDEX_KEY, live)

        store.with_lock(INDEX_KEY, index, 10, 5)

    def _announce(self, session: Session, plan: Plan) -> None:
        if self._events is None:
            return
        last_run = plan.get("lastRunId")
        self._events.emit(
            PlanUpdated(
                run_id=last_run if isinstance(last_run, str) else "",
                session_key=session.key(),
                plan_id=plan["id"],
                revision=plan["revision"],
                status=plan["status"],
                holding_approval_id=plan["holdingApprovalId"],
                counts=view.counts(plan),
                harness_agent=session.agent,
            )
        )

    # -- runtime hooks ---------------------------------------------------

    def after_turn(self, session: Session) -> None:
        """After the runtime records a turn: bind the approvals the turn staged,
        and notice the ones a new message superseded."""
        self._settle_turn(session)
        if not self._enabled:
            return
        try:
            self.reconciled(session)
        except Exception as failure:  # noqa: BLE001 - the next advance reconciles again
            _report_quietly(failure)

    def decided(self, session: Session) -> None:
        """After decisions are recorded, before the run resumes."""
        if not self._enabled:
            return
        try:
            self.reconciled(session)
        except Exception as failure:  # noqa: BLE001 - the decision is recorded either way
            _report_quietly(failure)

    def context(self, session: Session, prompt: str, mode: str | None = None) -> str:
        """The plan context for the turn about to start. A turn with a new user
        message ends every pending approval, so the plan is reconciled as if
        nothing were pending before it is rendered. ``mode`` is the mode the
        run resolved (a runner start's is not the session's until its task
        row exists)."""
        if not self.enabled_for(session, mode):
            return ""
        try:
            plan = self.reconciled(session, superseding=prompt != "")
        except Exception as failure:  # noqa: BLE001 - a turn without the block still runs
            _report_quietly(failure)
            return ""
        return "" if plan is None else view.context(plan, self._limits)

    def tools(self, session: Session, mode: str | None) -> list[PlanTool]:
        """The plan tools for a run in this session, or none when plans are off."""
        from prism_harness.plans.tools import plan_tools

        if not self.enabled_for(session, mode):
            return []
        return plan_tools(self, session)

    def run_id(self, session: Session) -> str | None:
        run = session.run()
        if (
            isinstance(run, dict)
            and isinstance(run.get("id"), str)
            and run.get("status") == "running"
        ):
            run_id: str = run["id"]
            return run_id
        return None

    # -- plan budget (G-366) ---------------------------------------------

    def budget_settings(self) -> dict[str, Any]:
        """The configured plan budget, validated when these plans were built."""
        return dict(self._budget)

    def budget_record(self, session: Session, plan_id: str) -> dict[str, Any] | None:
        """The budget record of this plan, or None when none was fixed yet."""
        record = session.plan_store().get(_budget_key(session.key()))
        return record if isinstance(record, dict) and record.get("planId") == plan_id else None

    def budget_for(self, session: Session, plan_id: str) -> dict[str, Any]:
        """The mode and limits this plan runs under: the fixed record, else config."""
        record = self.budget_record(session, plan_id)
        if record is None:
            return self.budget_settings()
        return {
            "mode": record["mode"],
            "max_steps": record["max_steps"],
            "max_cost_micros": record["max_cost_micros"],
        }

    def use_budget(self, session: Session, plan_id: str, requested: object) -> dict[str, Any]:
        """Host code chooses this plan's budget. The first choice fixes it; after
        that the mode cannot change and a limit can only be lowered."""
        chosen = budget.for_plan(self._budget, requested)
        return self._budget_write(
            session, plan_id, chosen, lambda state: (budget.narrow(state, chosen), None)
        )[0]

    def budget_exhaustion(self, session: Session, plan_id: str, hold_ttl: int) -> str | None:
        """Which shared limit this plan has spent, or None."""
        record = self.budget_record(session, plan_id)
        if record is None:
            return None
        return budget.exhaustion(budget.expire(record, self.now(), hold_ttl))

    def announce_budget_exhausted(self, session: Session, plan_id: str, limit: str) -> None:
        """Announce that a plan stopped on its shared budget."""
        if self._events is None:
            return
        record = self.budget_record(session, plan_id) or budget.open_ledger(
            plan_id, self.budget_for(session, plan_id)
        )
        self._events.emit(
            PlanBudgetExhausted(
                session_key=session.key(),
                plan_id=plan_id,
                limit=limit,
                steps=record["steps"],
                cost_usd=record["cost_micros"] / 1_000_000,
                harness_agent=session.agent,
            )
        )

    def serving_shared(self, session: Session, prompt: str = "") -> bool:
        """Whether this turn on this session spends from a shared plan budget."""
        return self._ledger_for(session, prompt, None) is not None

    def serving_step(self, session: Session) -> Callable[[], None]:
        """Mark the turn advance() runs for the step it claimed as the plan's
        own, until the returned release is called."""
        key = session.key()
        self._advancing[key] = self._advancing.get(key, 0) + 1
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            left = self._advancing.get(key, 1) - 1
            if left <= 0:
                self._advancing.pop(key, None)
            else:
                self._advancing[key] = left

        return release

    def ledger_of(self, run_id: str) -> dict[str, str] | None:
        """The shared plan ledger a run spends from, as the session key the plan
        belongs to and the plan id; None when it spends from none."""
        turn = self._budget_turns.get(run_id)
        return None if turn is None else {"session": turn["ledger"], "plan": turn["plan"]}

    def active_ledger(self) -> dict[str, str] | None:
        """The shared plan ledger of the plan turn running in this context, for a
        subagent child it starts to carry; None outside one."""
        ledger = _ACTIVE_LEDGER.get()
        return None if ledger is None else dict(ledger)

    def _ledger_for(
        self,
        session: Session,
        prompt: str,
        context: RunContext | None,
        kind: str = "none",
        mode: str | None = None,
    ) -> dict[str, Any] | None:
        """The shared ledger a turn spends from, as :func:`budget.route`
        decides: its own plan's when the turn is the plan's own; else, for a
        turn on a subagent child the runner did not start (an answer, a new
        message, or a host context, which the ledger narrows), the ledger it was
        saved with. ``refused`` marks the plan's own turn handed a run tree of
        its own. ``kind`` is the context's kind, from the runtime's run trees;
        ``mode`` the mode the turn runs in, when the session does not say it
        yet (a runner start, read before anything is written)."""
        if not self._enabled:
            return None
        plan = self.load(session) if self.enabled_for(session, mode) else None
        step = _active_agent_step(plan)
        advancing = step is not None and session.key() in self._advancing
        step_approval = step is not None and bool(
            set(step["approvalIds"]) & set(_paused_ids(_rows(session)))
        )
        return _routed_ledger(
            session, plan, kind, prompt, advancing, step_approval, self.budget_for
        )

    def plan_turn(
        self,
        session: Session,
        run_id: str,
        run: RunContext,
        context: RunContext | None,
        hold_ttl: int,
        prompt: str = "",
        kind: str | None = None,
        steps: int | None = None,
    ) -> tuple[RunContext, int | None]:
        """The run a turn executes under, and the steps the plan's ledger grants
        it (``plan_left`` of its allowance, G-387), when the turn is one of the
        plan's own and its budget is shared: the per-turn cost narrowed to what
        the ledger holds for it, and up to ``steps`` steps held (by default,
        what the run's own budget has left). Any other run is returned
        unchanged, with None.

        A ledger whose steps or cost are spent grants 0 steps; it does not
        refuse the turn here: acting on recorded decisions makes no provider
        call (G-387). The runtime refuses a turn with a new prompt and nothing
        left, with ``plan_budget_exhausted`` when this grant is what is 0.

        The plan's own turns are the one advance() runs for the step it
        claimed, the resume of an approval that step asked for, and any turn on
        a subagent child those turns started that the runner did not start (an
        answer, a new message, or a host context, which is narrowed to what the
        plan has left). Any other turn
        on the session (a person chatting while a step waits on them, or while
        a claim whose worker died is still live) spends nothing from the plan's
        budget and is never refused for it.

        The plan's own turn handed a run tree of its own cannot also spend
        from the plan's ledger, so it is refused rather than let around it.
        Called by the runtime once a run's own budget is known, with the kind
        of context the turn was handed (``none`` or ``host`` by default)."""
        if kind is None:
            kind = "none" if context is None else "host"
        target = self._ledger_for(session, prompt, context, kind)
        if target is None:
            return run, None
        if target["refused"]:
            raise PlanRefused("plan.budget_context")

        ledger = run.ledger
        own = run.budget
        base_steps = ledger.steps
        base_cost = ledger.cost_usd
        asked = max(0, own.max_steps - base_steps if steps is None else steps)
        turn_cost = (
            None
            if own.max_cost_usd is None
            else budget.micros(max(0.0, own.max_cost_usd - base_cost))
        )
        now = self.now()

        def hold(state: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            if asked < 1:
                return state, {"steps": 0, "cost_micros": None}
            following, grant = budget.reserve(state, run_id, asked, turn_cost, now, hold_ttl)
            if "refused" in grant:
                # Spent, on steps or on cost: 0 steps and no hold.
                return state, {"steps": 0, "cost_micros": None}
            return following, grant

        _, grant = self._budget_write_at(
            session.plan_store(), target["session"], target["plan"], self._budget, hold
        )
        narrowed = (
            run
            if grant["cost_micros"] is None
            else replace(
                run,
                budget=replace(own, max_cost_usd=base_cost + grant["cost_micros"] / 1_000_000),
            )
        )
        ledger_ref = {"session": target["session"], "plan": target["plan"]}
        self._budget_turns[run_id] = {
            "session": session.key(),
            "ledger": target["session"],
            "plan": target["plan"],
            "context": narrowed,
            "steps": base_steps,
            "cost": base_cost,
            "unmetered": ledger.unmetered_runs,
            "token": _ACTIVE_LEDGER.set(ledger_ref),
        }
        return narrowed, int(grant["steps"])

    def _settle_turn(self, session: Session) -> None:
        """Replace a finished turn's hold with what it spent."""
        run = session.run()
        run_id = run.get("id") if isinstance(run, dict) else None
        turn = self._budget_turns.get(run_id) if isinstance(run_id, str) else None
        if turn is None or turn["session"] != session.key():
            return
        assert isinstance(run_id, str)
        del self._budget_turns[run_id]
        try:
            _ACTIVE_LEDGER.reset(turn["token"])
        except ValueError:  # set in another context; nothing of this one to undo
            pass
        ledger = turn["context"].ledger
        steps = ledger.steps - turn["steps"]
        cost = (
            None
            if ledger.unmetered_runs > turn["unmetered"]
            else budget.micros(max(0.0, ledger.cost_usd - turn["cost"]))
        )
        try:
            self._budget_write_at(
                session.plan_store(),
                turn["ledger"],
                turn["plan"],
                self._budget,
                lambda state: (budget.settle(state, run_id, steps, cost), None),
            )
        except Exception as failure:  # noqa: BLE001 - the hold is charged in full once it expires
            _report_quietly(failure)

    def _budget_write(
        self,
        session: Session,
        plan_id: str,
        chosen: dict[str, Any],
        change: Callable[[dict[str, Any]], tuple[dict[str, Any], Any]],
    ) -> tuple[dict[str, Any], Any]:
        """One locked change to this plan's budget record, opening it from
        ``chosen`` when it is absent or belongs to an earlier plan."""
        return self._budget_write_at(session.plan_store(), session.key(), plan_id, chosen, change)

    def _budget_write_at(
        self,
        store: SessionStore,
        session_key: str,
        plan_id: str,
        chosen: dict[str, Any],
        change: Callable[[dict[str, Any]], tuple[dict[str, Any], Any]],
    ) -> tuple[dict[str, Any], Any]:
        """The same change to the budget record of the plan kept under another
        session key, in the same durable store (a subagent child's parent)."""
        key = _budget_key(session_key)

        def swap() -> tuple[dict[str, Any], Any]:
            stored = store.get(key)
            state = (
                stored
                if isinstance(stored, dict) and stored.get("planId") == plan_id
                else budget.open_ledger(plan_id, chosen)
            )
            following, result = change(state)
            store.put(key, following)
            return following, result

        return store.with_lock(key, swap, 10, 5)

    # -- reaper ----------------------------------------------------------

    def reap_using(self, dispatch: Callable[[str, str], None]) -> None:
        """The callback the reaper hands a stalled plan to: normally "queue the
        job that calls advance() for this session"."""
        self._reap_dispatcher = dispatch

    def reap(self) -> int:
        """Hand every stalled plan to the registered callback. What the
        reference's ``harness:plans:reap`` command does; schedule it every few
        minutes when plans are advanced by a queue."""
        if not self._enabled:
            return 0
        if self._reap_dispatcher is None:
            raise ValueError("No reaper callback is registered. Call Plans.reap_using() first.")
        return self.sweep(self._reap_dispatcher)

    def heartbeat(self) -> dict[str, int] | None:
        store = self._index_store
        beat = store.get(HEARTBEAT_KEY) if store is not None else None
        at = beat.get("at") if isinstance(beat, dict) else None
        return {"at": at} if isinstance(at, int) and not isinstance(at, bool) else None

    def sweep(self, dispatch: Callable[[str, str], None]) -> int:
        """Find active plans whose head step has no live claim and that have not
        moved for ``reaper.stale_after`` seconds, and hand each to ``dispatch``.
        Writes the heartbeat the doctor checks. Returns how many it handed on."""
        store = self._index_store
        if store is None:
            return 0
        now = self.now()
        handed = 0
        for session_key, plan_id in (store.get(INDEX_KEY) or {}).items():
            stored = store.get(_key(str(session_key)))
            plan = stored.get("plan") if isinstance(stored, dict) else None
            if not isinstance(plan, dict) or plan.get("id") != plan_id:
                continue
            if machine.stalled(plan, now, self._limits.stale_after):
                dispatch(str(session_key), str(plan_id))
                handed += 1
        store.put(HEARTBEAT_KEY, {"at": now})
        return handed


def _key(session_key: str) -> str:
    return f"{session_key}:plan"


def _budget_key(session_key: str) -> str:
    return f"{session_key}:plan-budget"


def _active_agent_step(plan: Plan | None) -> dict[str, Any] | None:
    """The plan's head step when it is the agent's and claimed for a turn or
    held on an approval that turn asked for; None otherwise."""
    if plan is None or plan.get("status") not in ("active", "waiting", "blocked"):
        return None
    head = machine.head_index(plan)
    if head is None:
        return None
    step: dict[str, Any] = plan["steps"][head]
    if step["owner"] != "agent" or step["state"] not in ("running", "held"):
        return None
    return step


def _rows(session: Session) -> list[dict[str, Any]]:
    return [entry.message for entry in session.thread().messages()]


def _paused_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every approval id the run paused at the transcript tail asked for,
    answered or not. A turn with no new message continues exactly these."""
    row = paused_approval_row(rows)
    requests = row.get("tool_approval_requests") if row is not None else None
    if not isinstance(requests, list):
        return []
    return [
        entry["approval_id"]
        for entry in requests
        if isinstance(entry, dict) and isinstance(entry.get("approval_id"), str)
    ]


def _shared_record(store: SessionStore, session_key: str, plan_id: str) -> dict[str, Any] | None:
    record = store.get(_budget_key(session_key))
    if (
        isinstance(record, dict)
        and record.get("planId") == plan_id
        and record.get("mode") == "shared"
    ):
        return record
    return None


def _child_ledger(session: Session) -> dict[str, str] | None:
    """The plan ledger a subagent child was saved with, while it is still that
    plan's shared ledger."""
    _, saved = saved_refs(session)
    if saved is None:
        return None
    if _shared_record(session.plan_store(), saved["session"], saved["plan"]) is None:
        return None
    return {"session": saved["session"], "plan": saved["plan"]}


def _routed_ledger(
    session: Session,
    plan: Plan | None,
    kind: str,
    prompt: str,
    advancing: bool,
    step_approval: bool,
    budget_for: Callable[[Session, str], dict[str, Any]],
) -> dict[str, Any] | None:
    tree, plan_ledger = saved_refs(session)
    route = budget.route(
        kind, prompt, advancing, step_approval, tree is not None, plan_ledger is not None
    )["plan"]
    if route in ("own", "refused") and plan is not None:
        if budget_for(session, plan["id"])["mode"] != "shared":
            return None
        return {"session": session.key(), "plan": plan["id"], "refused": route == "refused"}
    if route != "child":
        return None
    child = _child_ledger(session)
    return None if child is None else {**child, "refused": False}


def _report_quietly(failure: BaseException) -> None:
    """Report a failure by its class only. A store error's message can carry the
    data it was writing, which here would be plan text the model wrote, and
    plan text never goes into a log."""
    _log.warning("A plan update failed: %s.", type(failure).__name__)
