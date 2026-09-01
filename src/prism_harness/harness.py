"""The entry point: ``harness.for_(participant).session('support')``."""

from __future__ import annotations

from collections.abc import Mapping

from prism_harness.session import Participant, Session
from prism_harness.stores.base import SessionStore
from prism_harness.stores.manager import SessionStoreManager, StoreFactory
from prism_harness.stores.memory import MemorySessionStore

__all__ = ["PendingSession", "PrismHarness"]


class PrismHarness:
    """Mirrors the reference's ``PrismHarness::for($user)->session('support')``,
    including the two-step shape -- the participant is chosen first and the
    scope second, because one participant holds several unrelated conversations
    and the pair is what addresses one of them.

    ``for`` is a Python keyword, so the method is ``for_``. Forced rather than
    chosen, like ``Media.as_`` in ``prism-ai``.

    **The default is deliberately not usable for durable state.** With no
    drivers configured both slots resolve to an in-memory store, and the manager
    then REFUSES it for the durable slot -- because it reports itself volatile
    and the durable slot holds approvals a human has not answered yet.
    Constructing a harness therefore works, and asking it for durable state
    fails loudly with a message that names the fix.

    That is not an oversight to be smoothed over. A package that silently
    accepted an in-memory durable store would pass every test in one process and
    lose a half-executed action the first time it was deployed on two.
    """

    def __init__(
        self,
        drivers: Mapping[str, StoreFactory] | None = None,
        stores: Mapping[str, str] | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._stores = SessionStoreManager(
            drivers=drivers if drivers is not None else {"default": MemorySessionStore},
            stores=stores,
        )

    def for_(self, participant: Participant) -> PendingSession:
        """Bind to a participant. Returns a builder, because the scope comes next."""
        return PendingSession(participant, self._stores, self._ttl_seconds)

    def ephemeral_store(self) -> SessionStore:
        return self._stores.ephemeral()

    def durable_store(self) -> SessionStore:
        return self._stores.durable()


class PendingSession:
    def __init__(
        self,
        participant: Participant,
        stores: SessionStoreManager,
        ttl_seconds: float | None,
    ) -> None:
        self._participant = participant
        self._stores = stores
        self._ttl_seconds = ttl_seconds

    def session(self, scope: str) -> Session:
        """Resolve the session for a scope.

        The stores are resolved HERE, which is what makes the volatile-durable
        guard fire when a session is opened rather than at some later moment
        when an approval needs saving.
        """
        return Session(
            participant=self._participant,
            scope=scope,
            ephemeral=self._stores.ephemeral(),
            durable=self._stores.durable(),
            ttl_seconds=self._ttl_seconds,
        )
