"""The stored conversation a session is bound to."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from prism_harness.errors import HarnessError
from prism_harness.stores.base import SessionStore
from prism_harness.thread_rows import REPLACED_STEP, superseding_denials

__all__ = ["Thread", "ThreadMessage"]

T = TypeVar("T")

#: The field on the stored thread document (beside ``messages``, never a
#: row) that holds a resolution's CLAIM on the paused step while the resolver
#: runs its approved calls (G-404, G-387 round 14): ``{token, until}``, the
#: claim's private token and when it lapses (epoch seconds). While it stands
#: the thread takes no row but the resolver's own answer, so a superseding
#: row can never deny a call that is running, whoever records it.
CLAIM = "harness_resolving"


def _record_in_run(
    thread: Thread,
    messages: Sequence[dict[str, Any]],
    run_id: str,
    release: str | None = None,
) -> list[ThreadMessage]:
    """Record ``messages`` as the runtime's own rows of run ``run_id``.

    Internal, and scoped to exactly this one record call (G-406): inside-ness
    travels as an argument of this write, never as ambient state that a host
    write made during it (from a store, a listener) could share. With
    ``release``, the same write ends the resolution claim that token holds."""
    return thread._append(messages, run_id, run_id, release)


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

    @property
    def key(self) -> str:
        return self._key

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

    def rows(self) -> list[dict[str, Any]]:
        """The stored messages as every reader takes them, one per message.

        A 0.5.x ``harness_replaced_step`` mark names a thread POSITION
        (``entry.position``); the readers take it as the row's place in this
        list, so it is mapped here (G-387 round 14): a mark naming a position
        no row has names nothing. Every other row is the stored one."""
        entries = self.messages()
        return _indexed([entry.message for entry in entries], [e.position for e in entries])

    def count(self) -> int:
        return len(self.messages())

    def record(
        self, messages: Sequence[dict[str, Any]], run_id: str | None = None
    ) -> list[ThreadMessage]:
        """Append messages, in order, and return them with their positions.

        Read-and-write inside ONE lock. See the class docstring.

        A row that SUPERSEDES the paused step (DECISION B, G-387 round 13) is
        recorded after one denial for each of that step's calls that has no
        result, IN THE SAME WRITE: any assistant row with tool calls, and an
        assistant text row or a user row recorded outside the runs that own
        the step (see :func:`~prism_harness.thread_rows.superseding_denials`).
        A run id passed here is always outside. Nothing of the step runs after
        it, and no decision is accepted for it.

        While a resume is running the step's approved calls (G-404) the
        thread takes no row at all: :class:`~prism_harness.errors.HarnessError`
        ``session_locked``, with nothing written. Record it once the resume
        has answered the step.
        """
        return self._append(messages, run_id, None)

    def _append(
        self,
        messages: Sequence[dict[str, Any]],
        run_id: str | None,
        inside: str | None,
        release: str | None = None,
    ) -> list[ThreadMessage]:
        if not messages:
            return []
        batch_in = list(messages)

        def append() -> list[ThreadMessage]:
            stored = self._store.get(self._key) or {}
            claim = stored.get(CLAIM)
            held = claim.get("token") if isinstance(claim, dict) else None
            if _standing(claim) and (release is None or held != release):
                raise HarnessError.session_locked(self._key, 0)
            existing = [
                entry for entry in (stored.get("messages") or []) if isinstance(entry, dict)
            ]
            listed = [entry for entry in existing if isinstance(entry.get("message"), dict)]
            positions = [_position(entry) for entry in listed]
            # Read in index terms (see rows()); written as given.
            readable = _indexed([entry["message"] for entry in listed], positions)
            as_read = _indexed(batch_in, positions)
            original = {id(read): given for read, given in zip(as_read, batch_in, strict=True)}
            batch = [
                original.get(id(row), row)
                for row in superseding_denials(
                    [
                        (
                            row,
                            entry.get("run_id") if isinstance(entry.get("run_id"), str) else None,
                        )
                        for row, entry in zip(readable, listed, strict=True)
                    ],
                    as_read,
                    run_id,
                    inside,
                )
            ]
            recorded_at = datetime.now(timezone.utc).isoformat()
            last = max(
                [len(existing), *(_position(entry) for entry in existing)],
                default=0,
            )
            appended = [
                {
                    "position": last + index + 1,
                    "message": message,
                    "run_id": run_id,
                    "recorded_at": recorded_at,
                }
                for index, message in enumerate(batch)
            ]
            document = {**stored, "messages": [*existing, *appended]}
            if release is not None and held == release:
                document.pop(CLAIM, None)
            self._store.put(self._key, document)

            return [self._to_message(entry) for entry in appended]

        return self._store.with_lock(self._key, append)

    def _claim(
        self,
        token: str,
        seconds: float,
        work: Callable[[list[dict[str, Any]]], T | None],
    ) -> T | None:
        """Internal (G-404): read ``work`` from the rows inside the thread lock,
        and when it finds something to resolve, claim the thread for ``token``
        in the same lock, for ``seconds``. A superseding row recorded after
        the read is refused until the claim is released; one recorded before
        it is in the rows ``work`` read."""

        def claim() -> T | None:
            stored = self._store.get(self._key) or {}
            if _standing(stored.get(CLAIM)):
                raise HarnessError.session_locked(self._key, 0)
            found = work(self.rows())
            if found is not None:
                self._store.put(
                    self._key,
                    {**stored, CLAIM: {"token": token, "until": time.time() + max(seconds, 1)}},
                )
            return found

        return self._store.with_lock(self._key, claim)

    def _release(self, token: str) -> None:
        """Internal (G-404): end ``token``'s claim, if it still stands."""

        def release() -> None:
            stored = self._store.get(self._key) or {}
            claim = stored.get(CLAIM)
            if isinstance(claim, dict) and claim.get("token") == token:
                rest = dict(stored)
                rest.pop(CLAIM, None)
                self._store.put(self._key, rest)

        self._store.with_lock(self._key, release)

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


def _standing(claim: object) -> bool:
    """Whether a stored claim still holds: the shape the runtime writes, not
    yet lapsed. A claim a crashed resolver left lapses with its lock's ttl."""
    if not isinstance(claim, Mapping):
        return False
    until = claim.get("until")
    return (
        isinstance(claim.get("token"), str)
        and isinstance(until, (int, float))
        and not isinstance(until, bool)
        and until > time.time()
    )


def _position(entry: Mapping[str, Any]) -> int:
    position = entry.get("position")
    return position if type(position) is int else 0


def _indexed(rows: Sequence[dict[str, Any]], positions: Sequence[int]) -> list[dict[str, Any]]:
    """``rows`` with each 0.5.x replaced-step mark's thread position mapped
    to its row's place among ``rows`` (from 1), or to 0 when no row of those
    holds that position. Unchanged rows are the same objects."""
    place: dict[int, int] = {}
    for index, position in enumerate(positions):
        place.setdefault(position, index + 1)
    out: list[dict[str, Any]] = []
    for row in rows:
        mark = row.get(REPLACED_STEP)
        if isinstance(mark, dict) and type(mark.get("position")) is int:
            row = {**row, REPLACED_STEP: {**mark, "position": place.get(mark["position"], 0)}}
        out.append(row)
    return out
