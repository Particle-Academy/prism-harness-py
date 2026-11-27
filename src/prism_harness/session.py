"""One participant's live runtime, reconstructed per turn."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from prism_harness.address import ParticipantRef, SessionAddress
from prism_harness.address_index import IndexedSessionStore, StoreAddressIndex, store_lock_rank
from prism_harness.errors import HarnessError
from prism_harness.stores.base import SessionStore
from prism_harness.tasks import DEFAULT_LEASE_SECONDS, StoreTaskSource
from prism_harness.thread import Thread

__all__ = ["Participant", "Session"]

#: How many runs' trees a session keeps in its durable half (G-381).
_RUN_TREE_HISTORY = 20

R = TypeVar("R")


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
        if self.child_provisional():
            # A runner start that has not recorded its task: still a child
            # the runner started, under no tree and no plan budget, so every
            # turn on it is refused as a child with no binding.
            return {"tree": None, "plan_ledger": None}
        refs = self._child_field("child_refs")
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
        binding = self._child_field("child_binding")
        return deepcopy(binding) if isinstance(binding, dict) else None

    def child_depth(self) -> int:
        """The depth the subagent runner started this session's child run at,
        or 0 for a session the runner never started. Every later turn on the
        child runs at least this deep, so a resumed child cannot reset its
        nesting.

        Internal: read by the runtime."""
        depth = self._child_field("child_depth")
        return depth if type(depth) is int and depth > 0 else 0

    def child_ceiling(self) -> int | None:
        """The step ceiling the subagent runner started this session's child
        run with (G-387): the least of its declared ``max_steps``, what its
        parent had left and ``max_steps_ceiling``. Re-applied on every turn on
        the child the runner did not start, so a declaration holds however the
        child is resumed. None for a session the runner never started, or one
        started before the ceiling was saved.

        Internal: read by the runtime."""
        state, ceiling = self.child_ceiling_state()
        return ceiling if state == "valid" else None

    def child_ceiling_state(self) -> tuple[str, int | None]:
        """Whether this session saved a step ceiling (G-387): ``("valid", n)``
        for a whole number from 0 to 2^53-1, ``("absent", None)`` when none was
        ever saved, and ``("garbage", None)`` for any other value, which the
        runtime fails closed on rather than read as "never saved".

        Internal: read by the runtime."""
        present, ceiling = self._child_field_raw("child_ceiling")
        if not present:
            return "absent", None
        if type(ceiling) is int and 0 <= ceiling <= _MAX_SAFE_INTEGER:
            return "valid", ceiling
        return "garbage", None

    def claim_child_steps(
        self,
        run_id: str,
        now: int,
        hold_ttl: int,
        reset: bool,
        decide: Callable[[int | None], int],
    ) -> int | None:
        """Claim steps from the child's delegation budget, in ONE locked write
        (G-387): the steps this child has used since the runner last started
        it, counted durably beside its saved ceiling, plus every live hold.

        ``decide`` is handed that count (None when the stored count is
        missing or not one the harness wrote, or ``run_id`` already holds or
        was charged: the run then fails closed) and returns how many steps
        this run holds; the hold is kept under ``run_id`` until
        :meth:`settle_child_steps`. A hold older than ``hold_ttl`` is charged
        in full first and remembered per run, so a run that never settles can
        only over-charge the delegation, and one that settles late replaces
        that charge with what it took. ``reset`` (a runner start) is the only
        thing that writes a fresh count. Returns the count it decided on.

        Internal: called by the runtime when a run begins."""
        self._assert_writable()
        key = self._child_key()
        found: list[int | None] = [None]

        def change() -> None:
            stored = self._durable.get(key)
            current = dict(stored) if isinstance(stored, dict) else {}
            if reset:
                # A new delegation never resets under a turn still spending on
                # the child: refused here, in the same locked write, so no
                # check-then-act window lets a reset wipe a live hold.
                standing = _child_steps(current.get("child_steps"))
                if standing is not None and any(
                    now - hold["at"] < hold_ttl for hold in standing["holds"].values()
                ):
                    raise HarnessError(CHILD_TURN_LIVE, "A turn on this child is still running.")
            record: dict[str, Any] | None = (
                {"steps": 0, "holds": {}, "charged": {}}
                if reset
                else _child_steps(current.get("child_steps"))
                if self._confirmed(current)
                else None
            )
            if record is None or run_id in record["holds"] or run_id in record["charged"]:
                decide(None)
                return
            for hold_id, hold in list(record["holds"].items()):
                if now - hold["at"] >= hold_ttl:
                    del record["holds"][hold_id]
                    record["steps"] += hold["steps"]
                    record["charged"][hold_id] = hold["steps"]
            counted = record["steps"] + sum(hold["steps"] for hold in record["holds"].values())
            found[0] = counted
            taken = max(0, decide(counted))
            if taken > 0:
                record["holds"][run_id] = {"steps": taken, "at": now}
            self._durable.put(key, {**current, "child_steps": record})

        self._durable.with_lock(key, change, 10, 5)
        return found[0]

    def settle_child_steps(self, run_id: str, took: int) -> None:
        """Replace a run's charge on the child's delegation budget with the
        steps it took (G-387): its live hold, or the full charge its hold
        became when it expired, so a late settle is counted once. A record that
        is not one the harness wrote is left as it is, never repaired into a
        permissive one.

        Internal: called by the runtime when a run ends, parks or fails."""
        self._assert_writable()
        key = self._child_key()

        def change() -> None:
            stored = self._durable.get(key)
            current = dict(stored) if isinstance(stored, dict) else {}
            record = _child_steps(current.get("child_steps"))
            if record is None:
                return
            if run_id in record["holds"]:
                del record["holds"][run_id]
            elif run_id in record["charged"]:
                record["steps"] = max(0, record["steps"] - record["charged"].pop(run_id))
            else:
                return
            record["steps"] += max(0, took)
            self._durable.put(key, {**current, "child_steps": record})

        self._durable.with_lock(key, change, 10, 5)

    def child_live_hold(self, now: int, hold_ttl: int) -> bool:
        """Whether a turn on this child holds steps from its delegation budget
        right now (a hold younger than ``hold_ttl``). Read only.

        Internal: read by the subagent runner before it starts the child."""
        stored = self._durable.get(self._child_key())
        record = _child_steps(stored.get("child_steps")) if isinstance(stored, dict) else None
        return record is not None and any(
            now - hold["at"] < hold_ttl for hold in record["holds"].values()
        )

    def remember_child_binding(
        self, binding: dict[str, Any], depth: int = 0, ceiling: int | None = None
    ) -> Session:
        """Internal: save the binding the runner gave a child run, the depth
        it started at and its step ceiling, in the session's durable half. A
        runner start always overwrites what an earlier start saved."""
        self._assert_writable()
        depth = depth if type(depth) is int and depth > 0 else 0
        ceiling = ceiling if type(ceiling) is int and ceiling >= 0 else None
        if (
            self._child_field("child_binding") != binding
            or self.child_depth() != depth
            or self.child_ceiling() != ceiling
        ):
            self._write_child(
                {"child_binding": deepcopy(binding), "child_depth": depth, "child_ceiling": ceiling}
            )
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
        if self._child_field("child_refs") != refs:
            self._write_child({"child_refs": refs})
        return self

    def start_child(
        self,
        run_id: str,
        now: int,
        hold_ttl: int,
        steps: int,
        binding: dict[str, Any],
        depth: int,
        ceiling: int | None,
        refs: dict[str, dict[str, str] | None],
        mode: str | None = None,
        recheck: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """A runner start's child records, in ONE locked write (G-387 start
        gate): a fresh delegation count with this run holding ``steps``, the
        binding, the depth, the step ceiling, the run tree and plan budget
        references, and the session ``mode`` the child runs in. Refused
        (``child_turn_live``) with nothing written while a turn on the child
        holds steps from its delegation, re-checked here, in the same write, so
        a resume claiming since the gate read it is never wiped. ``recheck``
        runs inside the same lock before anything is written; whatever it
        raises refuses the write (the runtime re-reads the child's pending
        approvals there).

        Everything it writes is marked ``provisional`` with ``run_id``: every
        reader treats it as absent until that run's task row is in the thread
        (:meth:`child_provisional`), so a start that fails before then needs
        no second write to stay harmless.

        Returns what :meth:`undo_child_start` needs to put the child back: the
        records as they were before this write, and what it wrote.

        Internal: called by the runtime once the start gate let a start through
        and its tree hold is taken."""
        self._assert_writable()
        key = self._child_key()
        depth = depth if type(depth) is int and depth > 0 else 0
        ceiling = ceiling if type(ceiling) is int and ceiling >= 0 else None
        taken = max(0, steps)
        handle: dict[str, Any] = {}

        def change() -> None:
            stored = self._durable.get(key)
            current = dict(stored) if isinstance(stored, dict) else {}
            standing = _child_steps(current.get("child_steps"))
            if standing is not None and any(
                now - hold["at"] < hold_ttl for hold in standing["holds"].values()
            ):
                raise HarnessError(CHILD_TURN_LIVE, "A turn on this child is still running.")
            if recheck is not None:
                recheck()
            legacy = self._durable.get(self._durable_key()) or {}
            for name in ("child_binding", "child_depth", "child_ceiling", "child_refs"):
                if name not in current and name in legacy:
                    current[name] = deepcopy(legacy[name])
            written = {
                **current,
                "child_steps": {
                    "steps": 0,
                    "holds": {run_id: {"steps": taken, "at": now}} if taken > 0 else {},
                    "charged": {},
                },
                # Everything here is ABSENT to every reader until this run's
                # task row is in the thread (G-387 round 9).
                _PROVISIONAL: run_id,
                "child_binding": deepcopy(binding),
                "child_depth": depth,
                "child_ceiling": ceiling,
                "child_refs": {
                    "tree": None if refs.get("tree") is None else dict(refs["tree"] or {}),
                    "plan_ledger": None
                    if refs.get("plan_ledger") is None
                    else dict(refs["plan_ledger"] or {}),
                },
            }
            state = self._ephemeral.get(self._ephemeral_key())
            before = dict(state) if isinstance(state, dict) else {}
            self._durable.put(key, written)
            if mode is not None:
                try:
                    following = {**before, "mode": mode}
                    self._ephemeral.put(self._ephemeral_key(), following, self._ttl_seconds)
                    self._cached_state = following
                except BaseException:
                    # One write: the child's records go back with the mode.
                    if isinstance(stored, dict):
                        self._durable.put(key, stored)
                    else:
                        self._durable.forget(key)
                    raise
            handle.update(
                {
                    "prior": deepcopy(stored) if isinstance(stored, dict) else None,
                    "written": deepcopy(written),
                    "mode_present": "mode" in before,
                    "mode_prior": before.get("mode"),
                    "mode_written": mode,
                }
            )

        self._durable.with_lock(key, change, 10, 5)
        return handle

    def undo_child_start(self, run_id: str, handle: Mapping[str, Any]) -> None:
        """Put a child back as it was before :meth:`start_child`, when the
        start failed before its task was recorded (G-387). Housekeeping only:
        what :meth:`start_child` wrote is provisional, absent to every reader
        until a task row confirms it, so whether this lands changes nothing.
        In ONE locked write,
        and only what is still exactly what that write left. A record another
        writer changed since is never clobbered; only this run's own hold is
        removed from it. The same for the session mode.

        Internal: called by the runtime."""
        self._assert_writable()
        key = self._child_key()

        def change() -> None:
            stored = self._durable.get(key)
            if stored == handle.get("written"):
                prior = handle.get("prior")
                if isinstance(prior, dict):
                    self._durable.put(key, deepcopy(prior))
                else:
                    self._durable.forget(key)
            elif isinstance(stored, dict):
                record = _child_steps(stored.get("child_steps"))
                if record is not None and run_id in record["holds"]:
                    del record["holds"][run_id]
                    self._durable.put(key, {**stored, "child_steps": record})
            written_mode = handle.get("mode_written")
            if written_mode is None:
                return
            state = self._ephemeral.get(self._ephemeral_key())
            current = dict(state) if isinstance(state, dict) else {}
            if current.get("mode") != written_mode:
                return
            if handle.get("mode_present"):
                current["mode"] = handle.get("mode_prior")
            else:
                current.pop("mode", None)
            self._ephemeral.put(self._ephemeral_key(), current, self._ttl_seconds)
            self._cached_state = current

        self._durable.with_lock(key, change, 10, 5)

    def _child_key(self) -> str:
        """What the subagent runner saves about this session as a child (its
        binding, depth and run tree references) lives under a key of its own,
        never in the session's durable record, where a host keeps state
        (capabilities) another process may change while the runner writes."""
        return f"{self.key()}:child"

    def child_provisional(self) -> bool:
        """Whether this session's child records were written by a runner start
        whose task row is not in the thread (G-387 round 9). Every reader then
        treats them as ABSENT: no binding, no depth, no saved ceiling, no
        delegation count, no run tree or plan references, while the session
        still reads as a child the runner started, so the fail-closed rules
        for a linked child with no binding apply. A mark that is not a run id
        never confirms. Records with no mark (written before the mark existed,
        or confirmed since) read as they are.

        Internal: read by the runtime."""
        stored = self._durable.get(self._child_key())
        return isinstance(stored, dict) and not self._confirmed(stored)

    def _confirmed(self, stored: Mapping[str, Any]) -> bool:
        """Whether a record carrying a ``provisional`` run id is confirmed:
        that run's task row (a user row recorded under its id) is in the
        thread. Confirmation is only ever READ from the thread, never written,
        so no failed write can leave an unconfirmed start in force."""
        if _PROVISIONAL not in stored:
            return True
        run_id = stored[_PROVISIONAL]
        return type(run_id) is str and run_id != "" and run_id in self._task_runs()

    def _task_runs(self) -> set[str]:
        """The run ids the thread holds a task row (a user row) for. Read
        only: never allocates the thread."""
        store = (
            IndexedSessionStore(
                self._raw_durable, self._address_index, self.address, "durable", "thread"
            )
            if self._address_index is not None
            else self._durable
        )
        return {
            entry.run_id
            for entry in Thread(store, f"{self.key()}:thread").messages()
            if entry.run_id is not None and entry.message.get("type") == "user"
        }

    def confirm_child_start(self, run_id: str) -> None:
        """Housekeeping once a runner start's task row is recorded: drop the
        ``provisional`` mark from the child's records and its run state. The
        task row is the authority; whether this write lands changes nothing.

        Internal: called by the runtime, best effort."""
        self._assert_writable()
        key = self._child_key()

        def change() -> None:
            stored = self._durable.get(key)
            if isinstance(stored, dict) and stored.get(_PROVISIONAL) == run_id:
                self._durable.put(
                    key, {name: value for name, value in stored.items() if name != _PROVISIONAL}
                )

        self._durable.with_lock(key, change, 10, 5)
        run = self.state().get("run")
        if isinstance(run, dict) and run.get("id") == run_id and _PROVISIONAL in run:
            self._write(
                "run",
                {
                    name: value
                    for name, value in run.items()
                    if name not in (_PROVISIONAL, _PREVIOUS)
                },
            )

    def _child_field(self, name: str) -> Any:
        """A saved child field: from the child key, or, for a child saved
        before that key existed, from the session's durable record. A
        provisional record (:meth:`child_provisional`) holds nothing."""
        stored = self._durable.get(self._child_key())
        if isinstance(stored, dict) and not self._confirmed(stored):
            return None
        if isinstance(stored, dict) and name in stored:
            return stored[name]
        legacy = self._durable.get(self._durable_key()) or {}
        return legacy.get(name)

    def _child_field_raw(self, name: str) -> tuple[bool, Any]:
        """Whether a child field was saved at all, and its value. A
        provisional record (:meth:`child_provisional`) saved nothing."""
        stored = self._durable.get(self._child_key())
        if isinstance(stored, dict) and not self._confirmed(stored):
            return False, None
        if isinstance(stored, dict) and name in stored:
            return True, stored[name]
        legacy = self._durable.get(self._durable_key()) or {}
        return (name in legacy), legacy.get(name)

    def _write_child(self, fields: dict[str, Any]) -> None:
        """One locked change to the child key. A field never written there yet
        is carried over from the legacy record, so the key is complete once
        written."""
        key = self._child_key()

        def change() -> None:
            stored = self._durable.get(key)
            current = dict(stored) if isinstance(stored, dict) else {}
            legacy = self._durable.get(self._durable_key()) or {}
            for name in ("child_binding", "child_depth", "child_ceiling", "child_refs"):
                if name not in current and name in legacy:
                    current[name] = deepcopy(legacy[name])
            self._durable.put(key, {**current, **fields})

        self._durable.with_lock(key, change, 10, 5)

    def using_capability(self, name: str, state: dict[str, Any]) -> Session:
        return self._write_capabilities(lambda capabilities: {**capabilities, name: state})

    def forget_capability(self, name: str) -> Session:
        return self._write_capabilities(
            lambda capabilities: {k: v for k, v in capabilities.items() if k != name}
        )

    # -- runs ---------------------------------------------------------------

    def run(self) -> dict[str, Any] | None:
        """The session's current run. A run a runner start began is ABSENT
        until its task row is in the thread (G-387 round 9): the run it
        replaced is read instead, so a start that failed before its task row
        never stands in for the run the child is parked on."""
        run = self.state().get("run")
        if not isinstance(run, dict) or not isinstance(run.get("id"), str):
            return None
        if _PROVISIONAL in run and not (
            run.get(_PROVISIONAL) == run["id"] and run["id"] in self._task_runs()
        ):
            previous = run.get(_PREVIOUS)
            return (
                previous
                if isinstance(previous, dict)
                and isinstance(previous.get("id"), str)
                and _PROVISIONAL not in previous
                else None
            )
        return run

    def begin_run(
        self,
        run_id: str,
        mode: str,
        provider: str,
        model: str,
        tree: dict[str, str] | str = "none",
        depth: int = 0,
        provisional: bool = False,
    ) -> Session:
        """``tree`` and ``depth`` are the run's membership (G-381): the run tree
        it belongs to (``{session, run}``, ``host`` or ``none``) and how deep it
        is, decided once when the run begins. Also kept durably, by run id,
        so the root that resumes a run parked on approvals stays in the tree
        that run began in, whatever became of the ephemeral half
        (:meth:`resumed_run_tree`).

        ``provisional`` (a runner start): the run state and the run-tree
        history entry are absent to every reader until the run's task row is
        in the thread (G-387 round 9); :meth:`run` reads the run it replaced
        meanwhile."""
        self._assert_writable()
        member: dict[str, Any] = {
            "tree": dict(tree) if isinstance(tree, dict) else tree,
            "depth": depth,
        }
        marks: dict[str, Any] = {}
        if provisional:
            member[_PROVISIONAL] = run_id
            marks = {_PROVISIONAL: run_id, _PREVIOUS: deepcopy(self.run())}
        self._remember_run_tree(run_id, member)
        return self._write(
            "run",
            {
                **marks,
                "id": run_id,
                "status": "running",
                "mode": mode,
                "provider": provider,
                "model": model,
                "tree": dict(tree) if isinstance(tree, dict) else tree,
                "depth": depth,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def resumed_run_tree(self, prompt: str) -> dict[str, str] | None:
        """The durable run tree the run parked on this session's approvals
        began in (G-381), for a turn with no new message that resumes it; None
        for a new message, a session parked on nothing, or a run in the host's
        tree, in none, or no longer remembered.

        Internal: read by the runtime when a run begins."""
        if prompt != "":
            return None
        for entry in reversed(self.thread().messages()):
            row = entry.message
            kind = row.get("type")
            if kind == "user":
                return None
            if kind != "assistant":
                continue
            requests = row.get("tool_approval_requests")
            if not isinstance(requests, list) or not any(
                isinstance(request, dict) for request in requests
            ):
                return None
            return self._run_tree(entry.run_id) if entry.run_id else None
        return None

    def _parked_run_ids(self) -> set[str]:
        """The runs that issued the approvals this session's thread is parked
        on: the paused row's run, or none."""
        for entry in reversed(self.thread().messages()):
            kind = entry.message.get("type")
            if kind == "user":
                return set()
            if kind != "assistant":
                continue
            requests = entry.message.get("tool_approval_requests")
            paused = isinstance(requests, list) and any(isinstance(r, dict) for r in requests)
            return {entry.run_id} if paused and entry.run_id else set()
        return set()

    def _run_trees_key(self) -> str:
        """The run trees live under a key of their own, never in the session's
        durable record, where a host keeps state (capabilities, the child
        binding) another process may change while a run starts."""
        return f"{self.key()}:run-trees"

    def _remember_run_tree(self, run_id: str, member: dict[str, Any]) -> None:
        """Keep a run's tree under the run-trees key, in one locked change:
        the newest runs by an explicit sequence, and always the run the thread
        is parked on, so a retried resume never forgets the tree it belongs
        to."""
        key = self._run_trees_key()
        parked = self._parked_run_ids()

        def change() -> None:
            stored = self._durable.get(key) or {}
            runs = stored.get("runs") if isinstance(stored.get("runs"), dict) else {}
            assert isinstance(runs, dict)
            # An earlier runner start's entry whose task row never landed is
            # absent (G-387 round 9): dropped, never kept in place of a run
            # the history should hold.
            tasks: set[str] | None = None
            for held, entry in list(runs.items()):
                if isinstance(entry, dict) and _PROVISIONAL in entry:
                    tasks = self._task_runs() if tasks is None else tasks
                    if not (entry.get(_PROVISIONAL) == held and held in tasks):
                        runs = {name: kept for name, kept in runs.items() if name != held}
            last = stored.get("seq")
            seq = last + 1 if type(last) is int else 1
            runs = {**runs, run_id: {**member, "seq": seq}}

            def order(item: tuple[str, Any]) -> int:
                at = item[1].get("seq") if isinstance(item[1], dict) else None
                return at if type(at) is int else 0

            kept: dict[str, Any] = {}
            for held, entry in sorted(runs.items(), key=order, reverse=True):
                if len(kept) < _RUN_TREE_HISTORY or held in parked:
                    kept[held] = entry
            self._durable.put(key, {"seq": seq, "runs": kept})

        self._durable.with_lock(key, change, 10, 5)

    def _run_tree(self, run_id: str) -> dict[str, str] | None:
        stored = self._durable.get(self._run_trees_key()) or {}
        runs = stored.get("runs")
        recorded = runs.get(run_id) if isinstance(runs, dict) else None
        if (
            isinstance(recorded, dict)
            and _PROVISIONAL in recorded
            and not (recorded.get(_PROVISIONAL) == run_id and run_id in self._task_runs())
        ):
            return None
        tree = recorded.get("tree") if isinstance(recorded, dict) else None
        return _stored_pair(tree, "session", "run")

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
        return self._finish_run(
            run_id, "completed", {"finish_reason": finish_reason, "tool_calls": list(tool_calls)}
        )

    def fail_run(self, run_id: str, failure: str) -> Session:
        return self._finish_run(run_id, "failed", {"failure": failure})

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


def _stored_pair(value: object, first: str, second: str) -> dict[str, str] | None:
    """A saved run tree or plan budget reference; None when absent or malformed."""
    if not isinstance(value, dict):
        return None
    a, b = value.get(first), value.get(second)
    return {first: a, second: b} if isinstance(a, str) and a and isinstance(b, str) and b else None


_MAX_SAFE_INTEGER = 2**53 - 1

#: The mark a runner start puts on everything it writes before its task row
#: (G-387 round 9): the start's run id. Absent to every reader until that run's
#: task row is in the thread.
_PROVISIONAL = "provisional"

#: Beside a provisional run state: the run it replaced, read meanwhile.
_PREVIOUS = "previous"

#: The code a runner start is refused with when a turn on the child still
#: holds steps from its delegation count.
CHILD_TURN_LIVE = "child_turn_live"


def _whole(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_SAFE_INTEGER


def _child_steps(value: object) -> dict[str, Any] | None:
    """A child's durable step count as the harness writes it, copied, or None
    for anything else: missing, or any field (``steps``, the ``holds`` and
    ``charged`` containers, every hold's ``steps`` and ``at``, every charge)
    that is not a whole number from 0 to 2^53-1."""
    if not isinstance(value, dict) or not _whole(value.get("steps")):
        return None
    holds, charged = value.get("holds"), value.get("charged")
    if not isinstance(holds, dict) or not isinstance(charged, dict):
        return None
    copied: dict[str, dict[str, int]] = {}
    for hold_id, hold in holds.items():
        if (
            not isinstance(hold_id, str)
            or not isinstance(hold, dict)
            or not _whole(hold.get("steps"))
            or not _whole(hold.get("at"))
        ):
            return None
        copied[hold_id] = {"steps": hold["steps"], "at": hold["at"]}
    charges: dict[str, int] = {}
    for hold_id, steps in charged.items():
        if not isinstance(hold_id, str) or not _whole(steps):
            return None
        charges[hold_id] = steps
    return {"steps": value["steps"], "holds": copied, "charged": charges}
