"""One participant's live runtime, reconstructed per turn."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from prism_harness.address import ParticipantRef, SessionAddress
from prism_harness.address_index import IndexedSessionStore, StoreAddressIndex, store_lock_rank
from prism_harness.stores.base import SessionStore
from prism_harness.tasks import DEFAULT_LEASE_SECONDS, StoreTaskSource
from prism_harness.thread import Thread

__all__ = ["Participant", "Session"]

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

    def capability(self, name: str) -> dict[str, Any] | None:
        stored = self._durable.get(self._durable_key()) or {}
        capabilities = stored.get("capabilities")
        capability = capabilities.get(name) if isinstance(capabilities, dict) else None
        return (
            (deepcopy(capability) if self._resolution_only else capability)
            if isinstance(capability, dict)
            else None
        )

    def using_capability(self, name: str, state: dict[str, Any]) -> Session:
        return self._write_capabilities(lambda capabilities: {**capabilities, name: state})

    def forget_capability(self, name: str) -> Session:
        return self._write_capabilities(
            lambda capabilities: {k: v for k, v in capabilities.items() if k != name}
        )

    # -- runs ---------------------------------------------------------------

    def run(self) -> dict[str, Any] | None:
        run = self.state().get("run")
        return run if isinstance(run, dict) and isinstance(run.get("id"), str) else None

    def begin_run(self, run_id: str, mode: str, provider: str, model: str) -> Session:
        return self._write(
            "run",
            {
                "id": run_id,
                "status": "running",
                "mode": mode,
                "provider": provider,
                "model": model,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

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
