"""Resolves the two state slots, and refuses a configuration that would lose work."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from prism_harness.errors import HarnessError
from prism_harness.stores.base import SessionStore

__all__ = ["SLOT_DURABLE", "SLOT_EPHEMERAL", "SessionStoreManager", "StoreFactory"]

SLOT_EPHEMERAL = "ephemeral"
SLOT_DURABLE = "durable"

StoreFactory = Callable[[], SessionStore]


class SessionStoreManager:
    """State is split into two named slots rather than one store, because the
    halves have genuinely different requirements:

    - ``ephemeral`` -- active mode, selected model, run bookkeeping. Losing it
      degrades to a default.
    - ``durable`` -- threads and stored capabilities. Losing it is a
      correctness failure, not a cache miss.

    The guard is the point of this class: a driver that reports itself volatile
    is refused for the durable slot AT RESOLVE TIME. Accepting it and finding
    out later is exactly the failure the package was written to avoid.
    """

    def __init__(
        self,
        drivers: Mapping[str, StoreFactory],
        stores: Mapping[str, str] | None = None,
    ) -> None:
        self._drivers = dict(drivers)
        slots = dict(stores or {})
        self._slots = {
            SLOT_EPHEMERAL: slots.get(SLOT_EPHEMERAL, "default"),
            SLOT_DURABLE: slots.get(SLOT_DURABLE, "default"),
        }
        self._resolved: dict[str, SessionStore] = {}

    def ephemeral(self) -> SessionStore:
        return self.slot(SLOT_EPHEMERAL)

    def durable(self) -> SessionStore:
        return self.slot(SLOT_DURABLE)

    def slot(self, slot: str) -> SessionStore:
        cached = self._resolved.get(slot)
        if cached is not None:
            return cached

        name = self._slots.get(slot, "default")
        factory = self._drivers.get(name)

        if factory is None:
            raise HarnessError.unknown_store_driver(slot, name)

        store = factory()

        # Checked HERE rather than at construction, so it fires in the same
        # place whether the store is configured up front, swapped in a test, or
        # changed at runtime -- and so a misconfiguration cannot lie dormant
        # until the first approval needs saving.
        if slot == SLOT_DURABLE and not store.durability().is_durable():
            raise HarnessError.volatile_durable_store(slot, name)

        self._resolved[slot] = store
        return store
