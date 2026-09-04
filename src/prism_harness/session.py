"""One participant's live runtime, reconstructed per turn."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

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
        participant: Participant,
        scope: str,
        ephemeral: SessionStore,
        durable: SessionStore,
        ttl_seconds: float | None = None,
    ) -> None:
        self.participant = participant
        self.scope = scope
        self._ephemeral = ephemeral
        self._durable = durable
        self._ttl_seconds = ttl_seconds
        self._cached_state: dict[str, Any] | None = None

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
        digest = hashlib.sha1(self.participant.type.encode("utf-8")).hexdigest()[:12]
        return f"session:{digest}:{self.participant.id}:{self.scope}"

    # -- the ephemeral half ------------------------------------------------

    def mode(self) -> str | None:
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
        return self._cached_state

    def forget(self) -> Session:
        """Drop the ephemeral half. THE CONVERSATION IS UNTOUCHED."""
        self._ephemeral.forget(self._ephemeral_key())
        self._cached_state = None
        return self

    # -- the durable half --------------------------------------------------

    def thread(self) -> Thread:
        return Thread(self._durable, f"{self.key()}:thread")

    def tasks(self, lease_seconds: float = DEFAULT_LEASE_SECONDS) -> StoreTaskSource:
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
        return StoreTaskSource(self._durable, f"{self.key()}:tasks", lease_seconds=lease_seconds)

    def capability(self, name: str) -> dict[str, Any] | None:
        stored = self._durable.get(self._durable_key()) or {}
        capabilities = stored.get("capabilities")
        capability = capabilities.get(name) if isinstance(capabilities, dict) else None
        return capability if isinstance(capability, dict) else None

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

        def run_it() -> R:
            # Re-read inside the lock. State written by whoever held it before
            # us is otherwise invisible to this instance, and acting on a stale
            # read is the thing the lock exists to prevent.
            self._cached_state = None
            return callback(self)

        return self._ephemeral.with_lock(self.key(), run_it, ttl_seconds, wait_seconds)

    # -- internals ----------------------------------------------------------

    def _ephemeral_key(self) -> str:
        return f"{self.key()}:ephemeral"

    def _durable_key(self) -> str:
        return f"{self.key()}:durable"

    def _read_str(self, key: str) -> str | None:
        value = self.state().get(key)
        return value if isinstance(value, str) else None

    def _write(self, key: str, value: Any) -> Session:
        state = {**self.state(), key: value}
        self._ephemeral.put(self._ephemeral_key(), state, self._ttl_seconds)
        self._cached_state = state
        return self

    def _write_capabilities(self, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> Session:
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
