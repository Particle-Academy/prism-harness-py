"""Where session state lives between turns, and how durable that is."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol, TypeVar

__all__ = ["Durability", "SessionStore", "T"]

T = TypeVar("T")


class Durability(str, Enum):
    """Whether a store's contents survive a deploy.

    This is the distinction the whole state layer turns on. An in-memory dict or
    a Redis used as a cache is the natural home for live session state -- and a
    cache is disposable by definition. Something has to say which of the two a
    configured store actually is, because the package cannot detect it and
    guessing wrong loses a half-executed agent action rather than a cheap value.
    """

    #: Contents may vanish at any time -- a flush, an eviction, a deploy. Only
    #: safe for state whose loss degrades to a default: the active mode, the
    #: selected model, run bookkeeping.
    VOLATILE = "volatile"

    #: Contents survive until deliberately removed. Required for anything whose
    #: loss is a correctness failure rather than an inconvenience -- a pending
    #: tool approval is a half-executed action waiting on a human, and it has to
    #: outlive the request, the worker, and a deploy.
    DURABLE = "durable"

    def is_durable(self) -> bool:
        return self is Durability.DURABLE


class SessionStore(Protocol):
    """The store contract.

    A server handles a request and moves on, so a session cannot be an object
    held in memory the way a single-process agent's is -- it has to be
    reconstructed from a store every time. This is that store.

    ONE DELIBERATE DIVERGENCE FROM ``prism-harness-ts``: this contract is
    SYNCHRONOUS. The TypeScript port is async because Node's filesystem API is,
    not because the operations are slow; Python's is not, the PHP reference is
    synchronous too, and a caller who needs this off the event loop can wrap it
    in ``asyncio.to_thread`` -- which is exactly what ``prism-workspace-py``
    does for the same reason. Forcing async here would make every consumer of a
    plain WSGI application write ``asyncio.run`` around a dictionary lookup.
    """

    def get(self, key: str) -> dict[str, Any] | None:
        """None when nothing is stored."""
        ...

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        """``ttl_seconds`` of None keeps the payload until it is removed."""
        ...

    def forget(self, key: str) -> None: ...

    def with_lock(
        self,
        key: str,
        callback: Callable[[], T],
        ttl_seconds: float = 10,
        wait_seconds: float = 5,
    ) -> T:
        """Run the callback while holding an EXCLUSIVE lock on the key.

        Two workers can resolve the same session at the same moment -- a queued
        job finishing a run while the user sends another message is ordinary,
        not exotic. Whatever must not happen twice goes in here.

        Returns the callback's value. Raises :class:`HarnessError` with code
        ``session_locked`` if the lock cannot be acquired within
        ``wait_seconds``, rather than running the callback anyway.
        """
        ...

    def durability(self) -> Durability:
        """Whether this store's contents survive a deploy.

        Read by the manager when a slot is resolved: a store that reports itself
        volatile is refused for durable state instead of silently accepting it.
        """
        ...
