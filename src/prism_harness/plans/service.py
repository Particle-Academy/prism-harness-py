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
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from prism_harness.events import HarnessEvents, PlanUpdated
from prism_harness.plans import machine, view
from prism_harness.plans.ids import PlanIds
from prism_harness.plans.limits import PlanLimits
from prism_harness.plans.refusal import PlanRefused, PlansDisabled
from prism_harness.plans.thread_facts import approval_decisions, pending_approval_calls
from prism_harness.stores.base import SessionStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism_harness.modes import ModeRegistry
    from prism_harness.plans.handle import PlanHandle
    from prism_harness.plans.tools import PlanTool
    from prism_harness.runtime import AgentRuntime
    from prism_harness.session import Session

__all__ = ["HEARTBEAT_KEY", "INDEX_KEY", "Plans"]

INDEX_KEY = "prism-harness:plans:live"
HEARTBEAT_KEY = "prism-harness:plans:heartbeat"

_log = logging.getLogger("prism_harness.plans")

Plan = dict[str, Any]


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

    def context(self, session: Session, prompt: str) -> str:
        """The plan context for the turn about to start. A turn with a new user
        message ends every pending approval, so the plan is reconciled as if
        nothing were pending before it is rendered."""
        if not self.enabled_for(session):
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


def _report_quietly(failure: BaseException) -> None:
    """Report a failure by its class only. A store error's message can carry the
    data it was writing, which here would be plan text the model wrote, and
    plan text never goes into a log."""
    _log.warning("A plan update failed: %s.", type(failure).__name__)
