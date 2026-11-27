"""The stored conversation a session is bound to."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from prism_harness.errors import HarnessError
from prism_harness.stores.base import SessionStore
from prism_harness.thread_rows import (
    ANSWERS_STEP,
    CALL_BINDING,
    HARNESS,
    REPLACED_STEP,
    bound_step,
    superseding_denials,
)

__all__ = ["Thread", "ThreadMessage"]

#: The flag on a stored thread ENTRY (beside ``run_id``, never inside the
#: message) saying the harness itself wrote the row, through its private
#: in-run write (G-406, G-387 round 15). Only such a row's marks are
#: honoured (:data:`~prism_harness.thread_rows.ANSWERS_STEP`), and nothing a
#: caller records can set it.
WRITTEN_BY_HARNESS = "harness"

#: What a guarded write's guard is handed (the thread's rows, as read, and
#: each row's stored position) and returns: the rows to write, or None to
#: write nothing.
Guard = Callable[[list[dict[str, Any]], list[int]], Sequence[dict[str, Any]] | None]


def _record_in_run(
    thread: Thread,
    messages: Sequence[dict[str, Any]],
    run_id: str,
    guard: Guard | None = None,
) -> list[ThreadMessage]:
    """Record ``messages`` as the runtime's own rows of run ``run_id``.

    Internal, and scoped to exactly this one record call (G-406): inside-ness
    travels as an argument of this write, never as ambient state that a host
    write made during it (from a store, a listener) could share. Each row is
    stored as written by the harness, so its marks are honoured.

    With ``guard``, the rows written are not ``messages`` but what ``guard``
    returns when handed the thread's rows (as :meth:`Thread.rows` reads them)
    INSIDE this write's lock, and nothing is written when it returns None: a
    check and the write it allows are one step, so no row recorded in between
    is missed (G-387 round 15)."""
    return thread._append(messages, run_id, run_id, guard)


@dataclass(frozen=True)
class ThreadMessage:
    #: Position in the conversation, from 1. Assigned by the thread, never by
    #: the caller.
    position: int
    #: The serialised message -- whatever ``prism-ai-core``'s ``message_from_dict``
    #: can rebuild.
    message: dict[str, Any]
    #: The run that produced it, when there was one. Says who wrote the row,
    #: never whether its marks count (G-387 round 16): see
    #: :attr:`written_by_harness`.
    run_id: str | None
    recorded_at: str
    #: Whether the harness itself wrote the row, through its private in-run
    #: write (or as the denial a superseding write made it write): the only
    #: rows whose harness marks count. Never set by a caller.
    written_by_harness: bool = False


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

        A 0.5.x ``harness_replaced_step`` mark and a ``harness_answers_step``
        mark name a thread POSITION (``entry.position``); the readers take it
        as the row's place in this list, so it is mapped here (G-387 rounds 14
        and 15): a mark naming a position no row has names nothing. A tool
        result row the harness itself wrote reads with ``"harness": true``
        (:data:`~prism_harness.thread_rows.HARNESS`), and no other row does,
        whatever its stored message holds: only the harness's marks count.
        Every other row is the stored one."""
        stored = self._store.get(self._key)
        entries = stored.get("messages") if stored is not None else None
        listed = [
            entry
            for entry in (entries if isinstance(entries, list) else [])
            if isinstance(entry, dict) and isinstance(entry.get("message"), dict)
        ]
        return _readable(listed)

    def snapshot(self) -> tuple[list[dict[str, Any]], list[ThreadMessage]]:
        """:meth:`rows` and :meth:`messages` from ONE read of the store (G-387
        round 16): the rows every reader takes, and the stored entries beside
        them (positions, run ids), index for index."""
        stored = self._store.get(self._key)
        entries = stored.get("messages") if stored is not None else None
        listed = [
            entry
            for entry in (entries if isinstance(entries, list) else [])
            if isinstance(entry, dict) and isinstance(entry.get("message"), dict)
        ]
        return _readable(listed), [self._to_message(entry) for entry in listed]

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

        A call the harness is RUNNING (its started row recorded, its result
        not yet; G-387 round 15) is answered-in-progress: it gets no denial,
        and its result pairs to it wherever it lands. So a row recorded here
        never waits on, and is never refused by, a resume in progress.
        """
        return self._append(messages, run_id, None)

    def _append(
        self,
        messages: Sequence[dict[str, Any]],
        run_id: str | None,
        inside: str | None,
        guard: Guard | None = None,
    ) -> list[ThreadMessage]:
        if not messages and guard is None:
            return []

        def append() -> list[ThreadMessage]:
            stored = self._store.get(self._key) or {}
            existing = [
                entry for entry in (stored.get("messages") or []) if isinstance(entry, dict)
            ]
            listed = [entry for entry in existing if isinstance(entry.get("message"), dict)]
            positions = [_position(entry) for entry in listed]
            # Read in index terms (see rows()); written as given.
            readable = _readable(listed)
            if guard is None:
                given = list(messages)
            else:
                allowed = guard(readable, positions)
                if allowed is None:
                    return []
                given = list(allowed)
            # Every stored step's approval requests carry their binding (G-387
            # round 15), whoever records it: computed here, never taken as
            # given.
            batch_in = [bound_step(row) for row in given]
            if not batch_in:
                return []
            as_read = [_flagged(row, inside is not None) for row in _indexed(batch_in, positions)]
            original = {id(read): given for read, given in zip(as_read, batch_in, strict=True)}
            # Each row with whether the HARNESS wrote it: the runtime's own
            # rows, and the denials a superseding write makes the harness
            # write whoever triggered it (G-387 round 16): the only rows
            # whose marks count. A caller's own rows never.
            batch = [
                (original[id(row)], inside is not None) if id(row) in original else (row, True)
                for row in superseding_denials(
                    [(row, _owner_run(entry)) for row, entry in zip(readable, listed, strict=True)],
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
                    # The harness's own write, and only that.
                    **({WRITTEN_BY_HARNESS: True} if harness else {}),
                }
                for index, (message, harness) in enumerate(batch)
            ]
            document = {**stored, "messages": [*existing, *appended]}
            self._store.put(self._key, document)

            return [self._to_message(entry) for entry in appended]

        return self._store.with_lock(self._key, append)

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
            written_by_harness=entry.get(WRITTEN_BY_HARNESS) is True,
        )


def _owner_run(entry: Mapping[str, Any]) -> str | None:
    """The run that may OWN a step through this entry (run-aware supersede):
    its run id when the harness wrote it, else None (G-387 round 16): a run id
    a caller passed never makes its row one of a run's own."""
    run_id = entry.get("run_id")
    return run_id if isinstance(run_id, str) and entry.get(WRITTEN_BY_HARNESS) is True else None


#: The harness's own keys on an assistant row: read as ABSENT on a row the
#: harness did not write (G-387 round 16, ONE TRUST SIGNAL), as the
#: reference reads them.
_STEP_MARKS = ("harness_steps_used", "harness_steps_left")


def _readable(listed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Stored entries as every reader takes their rows: positions mapped
    (:func:`_indexed`), and each tool result row flagged
    :data:`~prism_harness.thread_rows.HARNESS` exactly when the harness wrote
    it -- a flag no stored message can carry for itself. On an assistant row
    the harness did not write, its own keys (``harness_steps_used``,
    ``harness_steps_left`` and each request's binding,
    :data:`~prism_harness.thread_rows.CALL_BINDING`) read as absent: a request
    with no binding binds its own step's call by position."""
    rows = _indexed([entry["message"] for entry in listed], [_position(e) for e in listed])
    return [
        _flagged(row, entry.get(WRITTEN_BY_HARNESS) is True)
        for row, entry in zip(rows, listed, strict=True)
    ]


def _flagged(row: dict[str, Any], harness: bool) -> dict[str, Any]:
    """A tool result row with the read-only harness flag set exactly to
    ``harness`` (any copy of the key a message stored dropped); an assistant
    row the harness did not write without its own keys; every other row
    unchanged. Unchanged rows are the same objects."""
    if row.get("type") == "assistant" and not harness:
        return _unmarked_step(row)
    if row.get("type") != "tool_result" or (not harness and HARNESS not in row):
        return row
    flagged = {key: value for key, value in row.items() if key != HARNESS}
    if harness:
        flagged[HARNESS] = True
    return flagged


def _unmarked_step(row: dict[str, Any]) -> dict[str, Any]:
    """An assistant row read without the harness's own keys."""
    requests = row.get("tool_approval_requests")
    bound = isinstance(requests, list) and any(
        isinstance(request, dict) and CALL_BINDING in request for request in requests
    )
    if not bound and not any(key in row for key in _STEP_MARKS):
        return row
    read = {key: value for key, value in row.items() if key not in _STEP_MARKS}
    if bound and isinstance(requests, list):
        read["tool_approval_requests"] = [
            {key: value for key, value in request.items() if key != CALL_BINDING}
            if isinstance(request, dict)
            else request
            for request in requests
        ]
    return read


def _position(entry: Mapping[str, Any]) -> int:
    position = entry.get("position")
    return position if type(position) is int else 0


def _indexed(rows: Sequence[dict[str, Any]], positions: Sequence[int]) -> list[dict[str, Any]]:
    """``rows`` with each replaced-step and answers-step mark's thread
    position mapped to its row's place among ``rows`` (from 1), or to 0 when
    no row of those holds that position. Unchanged rows are the same
    objects."""
    place: dict[int, int] = {}
    for index, position in enumerate(positions):
        place.setdefault(position, index + 1)
    out: list[dict[str, Any]] = []
    for row in rows:
        for key in (REPLACED_STEP, ANSWERS_STEP):
            mark = row.get(key)
            if isinstance(mark, dict) and type(mark.get("position")) is int:
                row = {**row, key: {**mark, "position": place.get(mark["position"], 0)}}
        out.append(row)
    return out
