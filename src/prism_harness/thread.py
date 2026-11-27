"""The stored conversation a session is bound to."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from prism_harness.errors import HarnessError
from prism_harness.stores.base import SessionStore
from prism_harness.thread_rows import superseding_denials

__all__ = ["Thread", "ThreadMessage"]


@dataclass(frozen=True)
class ThreadMessage:
    #: Position in the conversation, from 1. Assigned by the thread, never by
    #: the caller.
    position: int
    #: The serialised message -- whatever ``prism-ai-core``'s ``message_from_dict``
    #: can rebuild.
    message: dict[str, Any]
    #: The run that produced it, when there was one.
    run_id: str | None
    recorded_at: str


class Thread:
    """DURABLE by construction: it lives in the durable slot, so it is the one
    thing here a flushed cache cannot take away. The PHP reference keeps it as
    Eloquent rows for the same reason.

    **Position is assigned inside a lock.** :meth:`record` takes the lock before
    reading the current length. Two turns landing concurrently would otherwise
    both read position 4 and both write position 5, and the conversation would
    silently lose a message -- the race the reference tracks as
    prism-harness#2. Doing the read and the write inside one lock is the fix,
    not a retry afterwards.
    """

    def __init__(self, store: SessionStore, key: str) -> None:
        self._store = store
        self._key = key

    def exists(self) -> bool:
        """Read only: no allocation, lock or empty transcript write."""
        return self._store.get(self._key) is not None

    def messages(self) -> list[ThreadMessage]:
        stored = self._store.get(self._key)
        if stored is None:
            return []

        entries = stored.get("messages")
        if not isinstance(entries, list):
            return []

        return [self._to_message(entry) for entry in entries if isinstance(entry, dict)]

    def count(self) -> int:
        return len(self.messages())

    def record(
        self, messages: Sequence[dict[str, Any]], run_id: str | None = None
    ) -> list[ThreadMessage]:
        """Append messages, in order, and return them with their positions.

        Read-and-write inside ONE lock. See the class docstring.

        A row that SUPERSEDES the paused step (DECISION B, G-387 round 13) is
        recorded after one denial for each of that step's calls that has no
        result: any assistant row with tool calls, and an assistant text row
        or a user row recorded outside the runs that own the step (see
        :func:`~prism_harness.thread_rows.superseding_denials`). Nothing of the
        step runs after it, and no decision is accepted for it.
        """
        if not messages:
            return []

        def append() -> list[ThreadMessage]:
            stored = self._store.get(self._key) or {}
            existing = [
                entry for entry in (stored.get("messages") or []) if isinstance(entry, dict)
            ]
            recorded_at = datetime.now(timezone.utc).isoformat()
            batch = superseding_denials(
                [
                    (
                        entry["message"],
                        entry.get("run_id") if isinstance(entry.get("run_id"), str) else None,
                    )
                    for entry in existing
                    if isinstance(entry.get("message"), dict)
                ],
                list(messages),
                run_id,
            )

            appended = [
                {
                    "position": len(existing) + index + 1,
                    "message": message,
                    "run_id": run_id,
                    "recorded_at": recorded_at,
                }
                for index, message in enumerate(batch)
            ]

            self._store.put(self._key, {**stored, "messages": [*existing, *appended]})

            return [self._to_message(entry) for entry in appended]

        return self._store.with_lock(self._key, append)

    def supersede_paused_step(self) -> None:
        """Supersede the paused step now (DECISION B): one denial for each of
        its gated calls that has no result, recorded with no run id. What a
        new message does before its turn reads the thread: nothing of the
        step runs, whatever was decided on it, and no decision is accepted
        for it afterwards.

        Internal: called by the runtime when a new message starts a turn."""

        def append() -> None:
            stored = self._store.get(self._key) or {}
            existing = [
                entry for entry in (stored.get("messages") or []) if isinstance(entry, dict)
            ]
            rows = [
                (
                    entry["message"],
                    entry.get("run_id") if isinstance(entry.get("run_id"), str) else None,
                )
                for entry in existing
                if isinstance(entry.get("message"), dict)
            ]
            batch = superseding_denials(rows, [{"type": "user"}], None)
            if len(batch) == 1:
                return
            denial = {
                "position": len(existing) + 1,
                "message": batch[0],
                "run_id": None,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            self._store.put(self._key, {**stored, "messages": [*existing, denial]})

        self._store.with_lock(self._key, append)

    def clear(self) -> None:
        """Forget the conversation.

        Deliberately separate from :meth:`Session.forget`, which drops only the
        ephemeral half. Losing a thread is not a cache miss and must never be a
        side effect of clearing session state.
        """

        def empty() -> None:
            stored = self._store.get(self._key) or {}
            self._store.put(self._key, {**stored, "messages": []})

        self._store.with_lock(self._key, empty)

    @staticmethod
    def _to_message(entry: dict[str, Any]) -> ThreadMessage:
        message = entry.get("message")

        if not isinstance(message, dict):
            raise HarnessError.unmappable_content(
                f"a stored thread entry has no message object (position {entry.get('position')})"
            )

        position = entry.get("position")
        run_id = entry.get("run_id")
        recorded_at = entry.get("recorded_at")

        return ThreadMessage(
            position=position if isinstance(position, int) else 0,
            message=message,
            run_id=run_id if isinstance(run_id, str) else None,
            recorded_at=recorded_at if isinstance(recorded_at, str) else "",
        )
