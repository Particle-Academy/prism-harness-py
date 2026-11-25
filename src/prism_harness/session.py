"""One participant's live runtime, reconstructed per turn."""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeVar

from prism_harness.address import ParticipantRef, SessionAddress
from prism_harness.address_index import (
    IndexedSessionStore,
    IndexLockScopeUnsupported,
    StoreAddressIndex,
    store_lock_rank,
)
from prism_harness.errors import HarnessError
from prism_harness.events import HarnessEvents, RunCancelled, RunCancelRequested
from prism_harness.stores.base import SessionStore
from prism_harness.tasks import DEFAULT_LEASE_SECONDS, StoreTaskSource
from prism_harness.thread import Thread, ThreadMessage
from prism_harness.thread_rows import pending_approval_ids

__all__ = ["CancelOutcome", "Participant", "Session"]

R = TypeVar("R")

#: How many run ids a session remembers, so cancel() can tell one of its own
#: earlier runs (``not-running``) from an id it never issued (``refused``). The
#: same bound as the PHP reference.
RUN_HISTORY = 20

#: The run id this port issues: a lowercase UUID4, matched in full. Nothing is
#: trimmed or folded first, so a padded, upper-cased or newline-terminated
#: copy of a real id is refused like any other made-up one.
_RUN_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")

_SCOPE_RANKS = {"process": 0, "host": 1, "distributed": 2}


class CancelOutcome(str, Enum):
    """What :meth:`Session.cancel` did (G-285). The values are the wire strings
    every port returns."""

    #: The cancel flag is set: the run stops at its next step or tool, or, if
    #: it was parked on an approval, is settled now.
    OK = "ok"
    #: The run belongs to this session and has already ended (finished,
    #: failed or cancelled). Nothing changed.
    NOT_RUNNING = "not-running"
    #: This session has no run at all. Nothing changed.
    NO_TURN = "no-turn"
    #: This session's stores cannot hold a cancel another process can see.
    #: Nothing changed.
    UNSUPPORTED = "unsupported"
    #: The run id is not one of this session's runs: malformed, forged, or
    #: another session's. Nothing changed.
    REFUSED = "refused"


@dataclass(frozen=True)
class Participant:
    """Who a session belongs to.

    A TYPE and an ID, not just an id: one application holds users and teams and
    bots, and ``7`` means a different participant in each. The reference gets
    the same pair from Eloquent's morph class and primary key.
    """

    type: str
    id: str | int


class Session:
    """RESOLVED, NEVER HELD.

    Nothing survives in memory between turns, so a fresh worker resolving the
    same address has to see the same active mode, the same model and the same
    conversation as whatever set them. Everything here reads through a store for
    that reason, rather than being an attribute that happens to be populated.

    State is split deliberately:

    - mode, model, provider and run bookkeeping are EPHEMERAL. Lose them and the
      next turn falls back to a default, which is a shrug.
    - the thread and stored capabilities are DURABLE.
    """

    def __init__(
        self,
        participant: Participant | ParticipantRef,
        scope: str,
        ephemeral: SessionStore,
        durable: SessionStore,
        ttl_seconds: float | None = None,
        agent: str | None = None,
        actor: object | None = None,
        address_index: StoreAddressIndex | None = None,
        max_addresses_per_participant: int | None = None,
    ) -> None:
        if isinstance(participant, ParticipantRef) and actor is None:
            raise ValueError("Typed participants require an explicit authorization actor.")
        self._participant = participant
        self._legacy_identity = agent is None and not isinstance(participant, ParticipantRef)
        self._legacy_scope = scope
        self._address = SessionAddress(participant, scope, agent)
        if address_index is None and (agent is not None or isinstance(participant, ParticipantRef)):
            address_index = StoreAddressIndex(durable, max_addresses_per_participant)
        self._actor = participant if actor is None else actor
        self._address_index = address_index
        self._raw_durable = durable
        self._ephemeral = (
            IndexedSessionStore(ephemeral, address_index, self._address, "ephemeral")
            if address_index is not None
            else ephemeral
        )
        self._durable = (
            IndexedSessionStore(durable, address_index, self._address, "durable")
            if address_index is not None
            else durable
        )
        self._ttl_seconds = ttl_seconds
        self._cached_state: dict[str, Any] | None = None
        self._resolution_only = False
        self._resolved_mode: str | None = None
        self._lock_store = self._ephemeral
        if address_index is not None and store_lock_rank(durable) > store_lock_rank(ephemeral):
            self._lock_store = self._durable

    @property
    def address(self) -> SessionAddress:
        return (
            SessionAddress(self._participant, self._legacy_scope)
            if self._legacy_identity
            else self._address
        )

    @property
    def participant(self) -> Participant | ParticipantRef:
        return self._participant

    @participant.setter
    def participant(self, value: Participant | ParticipantRef) -> None:
        if not self._legacy_identity or isinstance(value, ParticipantRef):
            raise AttributeError("Named and ParticipantRef session identity is immutable")
        self._participant = value

    @property
    def scope(self) -> str:
        return self._legacy_scope if self._legacy_identity else self._address.scope

    @scope.setter
    def scope(self, value: str) -> None:
        if not self._legacy_identity:
            raise AttributeError("Named and ParticipantRef session identity is immutable")
        self._legacy_scope = value

    @property
    def agent(self) -> str | None:
        return self._address.agent

    @property
    def actor(self) -> object:
        return self._actor

    def key(self) -> str:
        """The address this session is resolved by.

        Participant AND scope, because one participant holds several unrelated
        conversations at once and they must not collide.

        The type is HASHED rather than interpolated, with sha1 truncated to 12
        characters -- byte for byte what the PHP reference produces. A class
        name contains backslashes there, which make for awkward store keys and
        leak the application's namespace layout into something visible in
        tooling. Matching the reference exactly is what lets a PHP app and a
        Python agent share ONE store and resolve the same session.
        """
        return self.address.session_key()

    # -- the ephemeral half ------------------------------------------------

    def mode(self) -> str | None:
        if self._resolved_mode is not None:
            return self._resolved_mode
        return self._read_str("mode")

    def using_mode(self, mode: str) -> Session:
        return self._write("mode", mode)

    def model(self) -> str | None:
        return self._read_str("model")

    def using_model(self, model: str) -> Session:
        return self._write("model", model)

    def provider(self) -> str | None:
        return self._read_str("provider")

    def using_provider(self, provider: str) -> Session:
        return self._write("provider", provider)

    def state(self) -> dict[str, Any]:
        if self._cached_state is None:
            self._cached_state = self._ephemeral.get(self._ephemeral_key()) or {}
        return deepcopy(self._cached_state) if self._resolution_only else self._cached_state

    def forget(self) -> Session:
        self._assert_writable()
        """Drop the ephemeral half. THE CONVERSATION IS UNTOUCHED."""
        self._ephemeral.forget(self._ephemeral_key())
        self._cached_state = None
        return self

    # -- the durable half --------------------------------------------------

    def thread(self) -> Thread:
        self._assert_writable()
        store = (
            IndexedSessionStore(
                self._raw_durable, self._address_index, self.address, "durable", "thread"
            )
            if self._address_index is not None
            else self._durable
        )
        return Thread(store, f"{self.key()}:thread")

    def find_thread(self) -> Thread | None:
        thread = self.thread()
        return thread if thread.exists() else None

    def tasks(self, lease_seconds: float = DEFAULT_LEASE_SECONDS) -> StoreTaskSource:
        self._assert_writable()
        """This session's task list, in the DURABLE half.

        The same reasoning as :meth:`thread`, and the reason the address is the
        session key with a suffix: a restarted process resolves the same session,
        sees the same list, and finds any task its predecessor was holding either
        still leased or expired back to ``todo``. That is what "an agent that
        survives a reboot picks up where it left off" actually requires.

        Durable by construction -- the store manager already refused a volatile
        driver for this slot before the session existed, and
        :class:`~prism_harness.tasks.StoreTaskSource` checks again anyway,
        because a consumer can build one over any store it likes.
        """
        store = (
            IndexedSessionStore(
                self._raw_durable, self._address_index, self.address, "durable", "tasks"
            )
            if self._address_index is not None
            else self._durable
        )
        return StoreTaskSource(store, f"{self.key()}:tasks", lease_seconds=lease_seconds)

    def plan_store(self) -> SessionStore:
        """Where this session's agent plan is stored (G-360): the durable store,
        always. A half-finished plan that vanished on a deploy would read as
        finished. Internal; reach a plan through ``runtime.plan(session)``."""
        self._assert_writable()
        return self._durable

    def hold_plan_lock(self, ttl_seconds: float) -> Callable[[], None]:
        """Hold the plan driver's lock, so one advance runs at a time per session.

        A LEASE, not a callback lock, because an advance runs a whole model turn
        between taking it and giving it back: a record in the lock store, with a
        token and an expiry, written and removed under the store's own lock. A
        second holder while it stands raises ``HarnessError`` with code
        ``session_locked`` and does nothing. A holder that died releases it
        when ``ttl_seconds`` runs out. Returns the release, which removes only
        this holder's record and is safe to call twice.
        """
        self._assert_writable()
        store = self._lock_store
        key = f"{self.key()}:plan-advance"
        token = secrets.token_hex(16)

        def take() -> None:
            held = store.get(key)
            until = held.get("until") if isinstance(held, dict) else None
            if isinstance(until, (int, float)) and until > time.time():
                raise HarnessError.session_locked(key, 0)
            store.put(key, {"token": token, "until": time.time() + ttl_seconds}, ttl_seconds)

        store.with_lock(key, take, 10, 5)
        released = [False]

        def release() -> None:
            if released[0]:
                return
            released[0] = True

            def drop() -> None:
                held = store.get(key)
                if isinstance(held, dict) and held.get("token") == token:
                    store.forget(key)

            store.with_lock(key, drop, 10, 5)

        return release

    def capability(self, name: str) -> dict[str, Any] | None:
        stored = self._durable.get(self._durable_key()) or {}
        capabilities = stored.get("capabilities")
        capability = capabilities.get(name) if isinstance(capabilities, dict) else None
        return (
            (deepcopy(capability) if self._resolution_only else capability)
            if isinstance(capability, dict)
            else None
        )

    def child_refs(self) -> dict[str, dict[str, str] | None] | None:
        """The run tree and shared plan budget this session was last started
        under as a subagent child, or None for a session the runner never
        started (G-366). Every turn on the child that the runner did not start
        reads them, a new message as much as an answer to its approval, so no
        turn on it gets a fresh root budget.

        Internal: written by the runtime when the subagent runner starts a
        child run."""
        stored = self._durable.get(self._durable_key()) or {}
        refs = stored.get("child_refs")
        if not isinstance(refs, dict):
            return None
        return {
            "tree": _stored_pair(refs.get("tree"), "session", "run"),
            "plan_ledger": _stored_pair(refs.get("plan_ledger"), "session", "plan"),
        }

    def child_binding(self) -> dict[str, Any] | None:
        """The provider and model binding the subagent runner gave this
        session's child runs, as stored (``ChildBinding.to_row()``), or None
        for a session the runner never started or one started before the
        binding was kept here.

        Kept in the DURABLE half, beside the thread and its pending approvals:
        a child is answered or sent a new message long after the ephemeral half
        may have expired.

        Internal: read by the runtime for every turn the runner did not start."""
        stored = self._durable.get(self._durable_key()) or {}
        binding = stored.get("child_binding")
        return deepcopy(binding) if isinstance(binding, dict) else None

    def remember_child_binding(self, binding: dict[str, Any]) -> Session:
        """Internal: save the binding the runner gave a child run, in the
        session's durable half."""
        self._assert_writable()
        stored = self._durable.get(self._durable_key()) or {}
        if stored.get("child_binding") != binding:
            self._durable.put(self._durable_key(), {**stored, "child_binding": deepcopy(binding)})
        return self

    def remember_child_refs(
        self, tree: dict[str, str] | None, plan_ledger: dict[str, str] | None
    ) -> Session:
        """Internal: save the run tree and plan budget a runner-started child
        run belongs to, in the session's durable half."""
        self._assert_writable()
        refs = {
            "tree": None if tree is None else dict(tree),
            "plan_ledger": None if plan_ledger is None else dict(plan_ledger),
        }
        stored = self._durable.get(self._durable_key()) or {}
        if stored.get("child_refs") != refs:
            self._durable.put(self._durable_key(), {**stored, "child_refs": refs})
        return self

    def using_capability(self, name: str, state: dict[str, Any]) -> Session:
        return self._write_capabilities(lambda capabilities: {**capabilities, name: state})

    def forget_capability(self, name: str) -> Session:
        return self._write_capabilities(
            lambda capabilities: {k: v for k, v in capabilities.items() if k != name}
        )

    # -- runs ---------------------------------------------------------------

    def run(self) -> dict[str, Any] | None:
        """This session's latest run, or None when it has none.

        A run parked on an approval when a person cancelled it reads ``status``
        ``cancelled`` (G-285): the cancel is durable even though no process was
        running the parked run to write it.
        """
        run = self.state().get("run")
        if not isinstance(run, dict) or not isinstance(run.get("id"), str):
            return None
        if run.get("status") != "cancelled":
            flag = self._cancel_flag(run["id"])
            if flag is not None and flag.get("parked") is True:
                return {**run, "status": "cancelled"}
        return run

    def begin_run(
        self,
        run_id: str,
        mode: str,
        provider: str,
        model: str,
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
    ) -> Session:
        """Record a run starting.

        ``parent_run_id`` and ``root_run_id`` are the run's tags, kept so a
        cancel from another process carries them on its event (G-285). The
        run id also joins a bounded history, so cancel() can answer
        ``not-running`` for one of this session's earlier runs.
        """
        stored = self.state().get("run_history")
        history = (
            [entry for entry in stored if isinstance(entry, str)]
            if isinstance(stored, list)
            else []
        )
        history = list(dict.fromkeys([*history, run_id]))[-RUN_HISTORY:]
        self._write("run_history", history)
        return self._write(
            "run",
            {
                "id": run_id,
                "status": "running",
                "mode": mode,
                "provider": provider,
                "model": model,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "parent_run_id": parent_run_id,
                "root_run_id": root_run_id if root_run_id is not None else run_id,
            },
        )

    # -- cancel (G-285) -----------------------------------------------------

    def cancel(self, run_id: str, events: HarnessEvents | None = None) -> CancelOutcome:
        """Ask this session's run to stop (G-285), from any process.

        The flag is DURABLE: written to the session's durable store, so the
        process running the run sees it at its next check even when the caller
        is a different process. The run checks it before every model step and
        before every tool call. A tool call already executing is NOT
        interrupted: it finishes, its result is recorded, and the next step or
        tool is refused. The run then ends ``cancelled`` and never resumes.
        Cancelling a run stops the whole run tree it belongs to.

        A run parked on an approval is cancelled at once: its approvals read as
        settled-cancelled (:meth:`approval_status`) and deciding one is refused.

        ``events`` receives ``run.cancel_requested`` (and, for a parked run,
        ``run.cancelled``) once the flag is written; ``AgentRuntime.cancel``
        passes the runtime's own.

        WHO MAY CANCEL IS THE APPLICATION'S DECISION, as with approvals: the
        session is scoped to a participant, but "this user may stop THIS run"
        is a question only the host can answer. Authorize before calling.
        """
        self._assert_writable()
        if not self._cancel_supported():
            return CancelOutcome.UNSUPPORTED
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            return CancelOutcome.REFUSED

        announced: list[RunCancelRequested | RunCancelled] = []

        def decide() -> CancelOutcome:
            # Read fresh: the run is written by whichever process executes it.
            self._cached_state = None
            state = self.state()
            run = state.get("run")
            if not isinstance(run, dict) or not isinstance(run.get("id"), str):
                return CancelOutcome.NO_TURN
            if run["id"] != run_id:
                history = state.get("run_history")
                return (
                    CancelOutcome.NOT_RUNNING
                    if isinstance(history, list) and run_id in history
                    else CancelOutcome.REFUSED
                )
            if run.get("status") != "running":
                return CancelOutcome.NOT_RUNNING
            flag = self._cancel_flag(run_id)
            if flag is not None:
                # Twice is harmless: a running run stops once, and only the
                # first cancel announces it. A parked run is already settled.
                return CancelOutcome.NOT_RUNNING if flag.get("parked") is True else CancelOutcome.OK
            parked = self._parked_on(run_id)
            self._record_cancel(run_id, parked)
            at = datetime.now(timezone.utc).isoformat()
            parent, root = _tag(run, "parent_run_id"), _tag(run, "root_run_id")
            announced.append(
                RunCancelRequested(
                    run_id=run_id,
                    session_key=self.key(),
                    parent_run_id=parent,
                    root_run_id=root,
                    state="parked" if parked else "running",
                    at=at,
                    harness_agent=self.agent,
                )
            )
            if parked:
                # Parked on a person: no process is running it, so it is
                # settled here, and its approvals now read as cancelled.
                announced.append(
                    RunCancelled(
                        run_id=run_id,
                        session_key=self.key(),
                        parent_run_id=parent,
                        root_run_id=root,
                        at=at,
                        harness_agent=self.agent,
                    )
                )
            return CancelOutcome.OK

        outcome = self._under_cancel_lock(decide)
        if events is not None:
            for event in announced:
                events.emit(event)
        return outcome

    def approval_status(self, approval_id: str) -> str:
        """Where an approval stands (G-285): ``pending``, ``approved``,
        ``denied``, ``cancelled`` (its run was cancelled before anyone
        decided), ``superseded`` (never decided, and a later turn ended it) or
        ``unknown`` (this thread never issued it). An approval in a cancelled
        run never reads ``pending``."""
        entries = self.thread().messages()
        issued = _issuing_entry(entries, approval_id)
        if issued is None:
            return "unknown"
        index, run_id = issued
        decision = _first_decision(entries[index + 1 :], approval_id)
        if decision is not None:
            return "approved" if decision else "denied"
        if run_id is not None and self.cancel_requested_for(run_id):
            return "cancelled"
        rows = [entry.message for entry in entries]
        return "pending" if approval_id in pending_approval_ids(rows) else "superseded"

    def approval_run_cancelled(self, approval_id: str) -> bool:
        """Whether the run that asked for this approval was cancelled (G-285).
        Internal: a decision on such an approval is refused."""
        issued = _issuing_entry(self.thread().messages(), approval_id)
        return issued is not None and issued[1] is not None and self.cancel_requested_for(issued[1])

    def cancel_requested_for(self, run_id: str) -> bool:
        """Whether a person cancelled this run (G-285), read from the durable
        store. Internal: read by the runtime's cancel checks."""
        return self._cancel_flag(run_id) is not None

    def resumes_cancelled_run(self, prompt: str) -> bool:
        """Whether an empty-prompt turn would continue a run parked on
        approvals that a person cancelled (G-285). Such a turn is refused: a
        cancelled run never resumes. Internal."""
        if prompt != "":
            return False
        paused = _paused_entry(self.thread().messages())
        return (
            paused is not None
            and paused.run_id is not None
            and self.cancel_requested_for(paused.run_id)
        )

    def cancel_run(self, run_id: str) -> bool:
        """End a run as cancelled (G-285). Returns whether the end is this
        process's to announce: False when the cancel already settled it as a
        parked run, and announced it. Internal."""
        return self.end_run(run_id, "cancelled", {"failure": "cancelled"})[1]

    def end_run(self, run_id: str, status: str, extra: dict[str, Any]) -> tuple[str, bool]:
        """The run's terminal write, under the cancel lock (G-285). Internal.

        A cancel that set the flag before this point ends the run
        ``cancelled``, recording what it would otherwise have ended as in
        ``ended_as``, even when its last step had already finished: a cancel
        that returned ``ok`` is never followed by a run that reads
        ``completed``. Returns the status written and whether this process
        announces ``run.cancelled`` for it.
        """

        def write() -> tuple[str, bool]:
            self._cached_state = None
            flag = self._cancel_flag(run_id)
            final, details = status, extra
            if flag is not None and status != "cancelled":
                final, details = "cancelled", {"failure": "cancelled", "ended_as": status}
            self._finish_run(run_id, final, details)
            parked = flag is not None and flag.get("parked") is True
            return final, final == "cancelled" and not parked

        return self._under_cancel_lock(write)

    def complete_run(
        self, run_id: str, finish_reason: str, tool_calls: Sequence[str] = ()
    ) -> Session:
        """``tool_calls`` is the NAMES of the tools this run invoked, in order.

        NAMES ONLY, and that boundary is deliberate. "Which tools did this run
        reach for" is what an operator needs to audit a guardrail, and a tool
        name is not PII. ARGUMENTS are -- ``prism-opentelemetry`` already
        carries them behind an opt-in capture gate with a length cap, and
        recording them a second time here, ungated, would quietly undo that
        decision for everyone who installed both.
        """
        self.end_run(
            run_id, "completed", {"finish_reason": finish_reason, "tool_calls": list(tool_calls)}
        )
        return self

    def fail_run(self, run_id: str, failure: str) -> Session:
        self.end_run(run_id, "failed", {"failure": failure})
        return self

    def lock(
        self,
        callback: Callable[[Session], R],
        ttl_seconds: float = 10,
        wait_seconds: float = 5,
    ) -> R:
        """Run something that MUST NOT HAPPEN TWICE.

        Two workers can hold the same session at the same moment: a queued job
        finishing a run while the user sends another message is ordinary.
        Advance a run inside this, not outside it.
        """

        self._assert_writable()

        def run_it() -> R:
            # Re-read inside the lock. State written by whoever held it before
            # us is otherwise invisible to this instance, and acting on a stale
            # read is the thing the lock exists to prevent.
            self._cached_state = None
            return callback(self)

        return self._lock_store.with_lock(self.key(), run_it, ttl_seconds, wait_seconds)

    # -- internals ----------------------------------------------------------

    def resolve_read_only(self, mode: str, resolve: Callable[[Session], R]) -> R:
        """Refuse logical mutations; ordinary get() retains expired-entry cleanup."""
        view = copy(self)
        view._cached_state = None
        view._resolution_only, view._resolved_mode = True, mode
        try:
            return resolve(view)
        finally:
            view._resolution_only, view._cached_state = False, None

    def _assert_writable(self) -> None:
        if self._resolution_only:
            raise ValueError("Harness tool resolution is read-only.")

    def _ephemeral_key(self) -> str:
        return f"{self.key()}:ephemeral"

    def _durable_key(self) -> str:
        return f"{self.key()}:durable"

    def _read_str(self, key: str) -> str | None:
        value = self.state().get(key)
        return value if isinstance(value, str) else None

    def _write(self, key: str, value: Any) -> Session:
        self._assert_writable()
        state = {**self.state(), key: value}
        self._ephemeral.put(self._ephemeral_key(), state, self._ttl_seconds)
        self._cached_state = state
        return self

    def _cancel_key(self) -> str:
        return f"{self.key()}:cancel"

    def _cancel_flag(self, run_id: str) -> dict[str, Any] | None:
        record = self._durable.get(self._cancel_key()) or {}
        runs = record.get("runs")
        flag = runs.get(run_id) if isinstance(runs, dict) else None
        return flag if isinstance(flag, dict) else None

    def _record_cancel(self, run_id: str, parked: bool) -> None:
        record = self._durable.get(self._cancel_key()) or {}
        stored = record.get("runs")
        runs = dict(stored) if isinstance(stored, dict) else {}
        runs.pop(run_id, None)
        runs[run_id] = {"at": datetime.now(timezone.utc).isoformat(), "parked": parked}
        # Bounded: the newest flags are the ones a run or an approval can still ask about.
        kept = dict(list(runs.items())[-RUN_HISTORY:])
        self._durable.put(self._cancel_key(), {"runs": kept})

    def _parked_on(self, run_id: str) -> bool:
        """Whether the thread's tail is waiting on approvals this run asked for."""
        paused = _paused_entry(self.thread().messages())
        return paused is not None and paused.run_id == run_id

    def _cancel_supported(self) -> bool:
        """Whether the cancel lock is seen by every process these stores are
        shared with: the lock store's lock scope covers the sharing scope of
        both the ephemeral store (the run) and the durable store (the flag).
        A process-local lock over a host-shared store cannot serialise a
        cancel with a run in another process, so it is ``unsupported``."""
        try:
            lock = store_lock_rank(self._lock_store)
        except IndexLockScopeUnsupported:
            return False
        for store in (self._ephemeral, self._durable):
            sharing = getattr(store, "sharing_scope", None)
            scope = sharing() if callable(sharing) else None
            if not isinstance(scope, str) or scope not in _SCOPE_RANKS:
                return False
            if _SCOPE_RANKS[scope] > lock:
                return False
        return True

    def _under_cancel_lock(self, work: Callable[[], R]) -> R:
        """Run a cancel decision, or a run's end, under this session's cancel
        lock, so a cancel racing a run's last step cannot both return ``ok``
        and leave the run ``completed``: whichever takes the lock second sees
        the other. Its own key, never the session lock; the order is always
        session lock, then cancel lock."""
        if not self._cancel_supported():
            return work()
        return self._lock_store.with_lock(f"{self.key()}:cancel-lock", work, 10, 10)

    def _write_capabilities(self, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> Session:
        self._assert_writable()
        stored = self._durable.get(self._durable_key()) or {}
        capabilities = stored.get("capabilities")
        current = capabilities if isinstance(capabilities, dict) else {}
        self._durable.put(self._durable_key(), {**stored, "capabilities": mutate(current)})
        return self

    def _finish_run(self, run_id: str, status: str, extra: dict[str, Any]) -> Session:
        current = self.run()

        # A run that is not the one in flight does not overwrite it. A late
        # worker reporting on a superseded run would otherwise mark the live one
        # finished.
        if current is not None and current.get("id") != run_id:
            return self

        return self._write(
            "run",
            {
                **(current or {"id": run_id}),
                "id": run_id,
                "status": status,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **extra,
            },
        )


def _tag(run: dict[str, Any], key: str) -> str | None:
    value = run.get(key)
    return value if isinstance(value, str) else None


def _issuing_entry(
    entries: Sequence[ThreadMessage], approval_id: str
) -> tuple[int, str | None] | None:
    """The position of the stored row that asked for an approval id, newest
    first, and the run that recorded it; None when no row issued it."""
    for index in range(len(entries) - 1, -1, -1):
        message = entries[index].message
        if message.get("type") != "assistant":
            continue
        requests = message.get("tool_approval_requests")
        if isinstance(requests, list) and any(
            isinstance(request, dict) and request.get("approval_id") == approval_id
            for request in requests
        ):
            return index, entries[index].run_id
    return None


def _first_decision(entries: Sequence[ThreadMessage], approval_id: str) -> bool | None:
    """The first decision recorded for an approval id, or None."""
    for entry in entries:
        message = entry.message
        kind = message.get("type")
        responses: object
        if kind == "tool_result":
            responses = message.get("tool_approval_responses")
        elif kind == "tool_approval_response":
            responses = [message]
        else:
            continue
        for response in responses if isinstance(responses, list) else []:
            if isinstance(response, dict) and response.get("approval_id") == approval_id:
                return response.get("approved") is True
    return None


def _paused_entry(entries: Sequence[ThreadMessage]) -> ThreadMessage | None:
    """The stored entry whose approvals the transcript tail is waiting on, as
    :func:`~prism_harness.thread_rows.paused_approval_row` reads it."""
    for entry in reversed(entries):
        kind = entry.message.get("type")
        if kind == "user":
            return None
        if kind != "assistant":
            continue
        requests = entry.message.get("tool_approval_requests")
        return entry if isinstance(requests, list) and requests else None
    return None


def _stored_pair(value: object, first: str, second: str) -> dict[str, str] | None:
    """A saved run tree or plan budget reference; None when absent or malformed."""
    if not isinstance(value, dict):
        return None
    a, b = value.get(first), value.get(second)
    return {first: a, second: b} if isinstance(a, str) and a and isinstance(b, str) and b else None
