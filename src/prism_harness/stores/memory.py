"""State in this process's memory. Volatile, and it says so."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prism_harness.errors import HarnessError
from prism_harness.stores.base import Durability, T

__all__ = ["MemorySessionStore"]


@dataclass
class _Entry:
    payload: dict[str, Any]
    expires_at: float | None


class MemorySessionStore:
    """The right home for the ephemeral slot in a test or a single-process tool,
    and REFUSED for the durable slot by the store manager -- not as a
    technicality: the contents do not outlive the process, and the durable slot
    holds approvals a human has not answered yet.

    The lock is real but PROCESS-LOCAL. Two workers cannot see each other's
    locks, which is the whole reason a deployment uses something else; this
    serialises callers within one process and claims nothing beyond that.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, threading.Lock] = {}
        # Guards the two dicts themselves. Without it, two threads creating the
        # lock for the same key would each build one and neither would exclude
        # the other -- the check-then-act race the lock exists to prevent, one
        # level up.
        self._guard = threading.Lock()

    def durability(self) -> Durability:
        return Durability.VOLATILE

    def get(self, key: str) -> dict[str, Any] | None:
        with self._guard:
            entry = self._entries.get(key)

            if entry is None:
                return None

            if entry.expires_at is not None and entry.expires_at <= time.monotonic():
                del self._entries[key]
                return None

            # A copy, so a caller mutating what it read cannot reach into the
            # store. Only this driver could ever get that wrong, which is why
            # the tests check it on every driver.
            return copy.deepcopy(entry.payload)

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        with self._guard:
            self._entries[key] = _Entry(
                payload=copy.deepcopy(payload),
                expires_at=None if ttl_seconds is None else time.monotonic() + ttl_seconds,
            )

    def forget(self, key: str) -> None:
        with self._guard:
            self._entries.pop(key, None)

    def with_lock(
        self,
        key: str,
        callback: Callable[[], T],
        ttl_seconds: float = 10,
        wait_seconds: float = 5,
    ) -> T:
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())

        if not lock.acquire(timeout=wait_seconds):
            raise HarnessError.session_locked(key, wait_seconds)

        try:
            return callback()
        finally:
            lock.release()
