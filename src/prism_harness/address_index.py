"""Fixed pages and a small recoverable commit head; no participant-wide ceiling."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

from prism_harness.address import ParticipantRef, SessionAddress
from prism_harness.errors import ErrorCode, HarnessError
from prism_harness.stores.base import Durability, SessionStore, T

if TYPE_CHECKING:
    from prism_harness.session import Participant


class IndexLockScopeUnsupported(RuntimeError):
    code = "index_lock_scope_unsupported"

    def __init__(self) -> None:
        super().__init__(
            "Indexed namespaces require declared sharing and lock scopes, with the lock covering the sharing scope."
        )


def store_lock_rank(store: SessionStore) -> int:
    sharing, locking = getattr(store, "sharing_scope", None), getattr(store, "lock_scope", None)
    ranks = {"process": 0, "host": 1, "distributed": 2}
    if not callable(sharing) or not callable(locking):
        raise IndexLockScopeUnsupported()
    shared, locked = sharing(), locking()
    if (
        not isinstance(shared, str)
        or not isinstance(locked, str)
        or shared not in ranks
        or locked not in ranks
        or ranks[locked] < ranks[shared]
    ):
        raise IndexLockScopeUnsupported()
    return ranks[locked]


class IndexRecoveryRequired(RuntimeError):
    code = "index_recovery_required"

    def __init__(self) -> None:
        super().__init__(
            "Recover the interrupted address append before reading a complete index or planning erasure."
        )


class AddressLimitExceeded(RuntimeError):
    code = "address_limit_exceeded"

    def __init__(self) -> None:
        super().__init__(
            "The configured max_addresses_per_participant limit was reached; no resource was written."
        )


def address_limit(value: int | None) -> int | None:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("max_addresses_per_participant must be a positive integer or null.")
        if value < 1:
            raise ValueError("max_addresses_per_participant must be a positive integer or null.")
    return value


class AddressIndexPage(TypedDict):
    entries: list[dict[str, Any]]
    next_cursor: int | None


class StoreAddressIndex:
    PAGE_SIZE = 100

    def __init__(
        self, store: SessionStore, max_addresses_per_participant: int | None = None
    ) -> None:
        if store.durability() is not Durability.DURABLE:
            raise HarnessError(
                ErrorCode.UNSAFE_STATE_CONFIGURATION, "The address index requires a durable store."
            )
        store_lock_rank(store)
        self._store = store
        self._max_addresses = address_limit(max_addresses_per_participant)

    @staticmethod
    def _root(type: str | None, id: str | None) -> str:
        data = json.dumps([type, id], separators=(",", ":"), ensure_ascii=False)
        return "harness-address-index:v1:" + hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def _head(value: dict[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {"version": 1, "count": 0, "pages": 0, "pending": None}
        count, pending = value.get("count"), value.get("pending")
        if (
            value.get("version") != 1
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or value.get("pages")
            != (count + StoreAddressIndex.PAGE_SIZE - 1) // StoreAddressIndex.PAGE_SIZE
            or (pending is not None and not isinstance(pending, dict))
            or "pending" not in value
        ):
            raise ValueError("Invalid harness address index head.")
        return value

    def _page_ids(self, root: str, number: int) -> list[str]:
        page = self._store.get(f"{root}:page:{number}")
        if page is None:
            return []
        ids = page.get("ids")
        if (
            page.get("version") != 1
            or not isinstance(ids, list)
            or len(ids) > self.PAGE_SIZE
            or any(
                not isinstance(id, str) or re.fullmatch(r"[a-f0-9]{64}", id) is None for id in ids
            )
        ):
            raise ValueError("Invalid harness address index page.")
        return ids

    def _recover(self, root: str, head: dict[str, Any]) -> dict[str, Any]:
        pending = head["pending"]
        if pending is None:
            return head
        if (
            not isinstance(pending, dict)
            or not isinstance(pending.get("hash"), str)
            or re.fullmatch(r"[a-f0-9]{64}", pending["hash"]) is None
            or pending.get("position") != head["count"]
            or not isinstance(pending.get("entry"), dict)
            or pending["entry"].get("position") != head["count"]
        ):
            raise ValueError("Invalid harness address append journal.")
        number, offset = divmod(head["count"], self.PAGE_SIZE)
        ids = self._page_ids(root, number)
        if len(ids) == offset:
            ids.append(pending["hash"])
            self._store.put(f"{root}:page:{number}", {"version": 1, "ids": ids})
        elif len(ids) != offset + 1 or ids[offset] != pending["hash"]:
            raise ValueError("Conflicting harness address append journal.")
        self._store.put(f"{root}:entry:{pending['hash']}", pending["entry"])
        committed = {"version": 1, "count": head["count"] + 1, "pages": number + 1, "pending": None}
        # Visibility/commit point; no resource data is written until this succeeds.
        self._store.put(f"{root}:head", committed)
        return committed

    def record(self, address: SessionAddress, slot: str, kind: str, key: str) -> None:
        if slot not in ("ephemeral", "durable") or kind not in ("state", "lock", "tasks", "thread"):
            raise ValueError("Unknown harness resource slot or kind.")
        root = self._root(address.participant_type, address.participant_id)

        def write() -> None:
            head = self._recover(root, self._head(self._store.get(f"{root}:head")))
            hash = address.index_key()
            identity = {
                "agent": address.agent,
                "participant_type": address.participant_type,
                "participant_id": address.participant_id,
                "scope": address.scope,
            }
            existing = self._store.get(f"{root}:entry:{hash}")
            if existing is not None:
                position = existing.get("position")
                if (
                    existing.get("address") != identity
                    or isinstance(position, bool)
                    or not isinstance(position, int)
                    or not 0 <= position < head["count"]
                    or not isinstance(existing.get("resources"), list)
                ):
                    raise ValueError("Invalid harness address index entry.")
                ids = self._page_ids(root, position // self.PAGE_SIZE)
                if len(ids) <= position % self.PAGE_SIZE or ids[position % self.PAGE_SIZE] != hash:
                    raise ValueError("Unlinked harness address index entry.")
            resources = list(existing["resources"]) if existing is not None else []
            resource = {"slot": slot, "kind": kind, "key": key}
            if resource in resources:
                return
            resources.append(resource)
            entry = {
                "address": identity,
                "resources": resources,
                "position": head["count"] if existing is None else existing["position"],
            }
            if existing is not None:
                self._store.put(f"{root}:entry:{hash}", entry)
                return
            if self._max_addresses is not None and head["count"] >= self._max_addresses:
                raise AddressLimitExceeded()
            prepared = {
                **head,
                "pending": {"hash": hash, "position": head["count"], "entry": entry},
            }
            self._store.put(f"{root}:head", prepared)
            self._recover(root, prepared)

        self._store.with_lock(root, write)

    def recover(self, participant: Participant | ParticipantRef | None) -> None:
        root = self._root(
            None if participant is None else participant.type,
            None if participant is None else str(participant.id),
        )
        self._store.with_lock(
            root, lambda: self._recover(root, self._head(self._store.get(f"{root}:head")))
        )

    def page(
        self,
        participant: Participant | ParticipantRef | None,
        cursor: int = 0,
        limit: int = 100,
    ) -> AddressIndexPage:
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("Invalid address index page cursor or limit.")
        root = self._root(
            None if participant is None else participant.type,
            None if participant is None else str(participant.id),
        )
        head = self._head(self._store.get(f"{root}:head"))
        if head["pending"] is not None:
            raise IndexRecoveryRequired()
        end, entries = min(cursor + limit, head["count"]), []
        page_number, ids = -1, []
        for position in range(cursor, end):
            number, offset = divmod(position, self.PAGE_SIZE)
            if number != page_number:
                ids, page_number = self._page_ids(root, number), number
            if len(ids) <= offset:
                raise ValueError("Missing committed harness address page member.")
            hash = ids[offset]
            entry = self._store.get(f"{root}:entry:{hash}")
            if entry is None or entry.get("position") != position:
                raise ValueError("Missing committed harness address index entry.")
            entries.append({"address_key": hash, **entry})
        return {"entries": entries, "next_cursor": end if end < head["count"] else None}


class IndexedSessionStore:
    def __init__(
        self,
        inner: SessionStore,
        index: StoreAddressIndex,
        address: SessionAddress,
        slot: str,
        kind: str = "state",
    ) -> None:
        self._inner, self._index, self._address, self._slot, self._kind = (
            inner,
            index,
            address,
            slot,
            kind,
        )

    def get(self, key: str) -> dict[str, Any] | None:
        return self._inner.get(key)

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        self._index.record(self._address, self._slot, self._kind, key)
        self._inner.put(key, payload, ttl_seconds)

    def forget(self, key: str) -> None:
        self._inner.forget(key)

    def durability(self) -> Durability:
        return self._inner.durability()

    def lock_scope(self) -> str:
        scope = getattr(self._inner, "lock_scope", None)
        return str(scope()) if callable(scope) else "unsupported"

    def sharing_scope(self) -> str:
        scope = getattr(self._inner, "sharing_scope", None)
        return str(scope()) if callable(scope) else "unsupported"

    def with_lock(
        self,
        key: str,
        callback: Callable[[], T],
        ttl_seconds: float = 10,
        wait_seconds: float = 5,
    ) -> T:
        self._index.record(self._address, self._slot, "lock", key)
        return self._inner.with_lock(key, callback, ttl_seconds, wait_seconds)
