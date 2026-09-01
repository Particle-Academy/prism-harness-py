"""State on disk, as one JSON file per key. Durable, and it says so."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prism_harness.errors import HarnessError
from prism_harness.stores.base import Durability, T

__all__ = ["FileSessionStore"]

#: Errnos that all mean "the lock is already held".
#:
#: POSIX says EEXIST. WINDOWS DOES NOT, reliably: deleting a file another handle
#: still has open leaves it in a pending-delete state, and a create attempt
#: against that raises PermissionError instead. That window is not rare -- it is
#: exactly the moment one caller releases the lock while the next is trying to
#: take it, which is the common case under contention rather than an edge.
#:
#: Treating those as "held, retry" is also the safe direction to be wrong in: a
#: genuine permission problem retries until the wait expires and then reports
#: `session_locked`, which is survivable. Treating them as "not held" would hand
#: the same lock to two callers.
_HELD = (FileExistsError, PermissionError)


class FileSessionStore:
    """The default durable driver for a port with no database and no dependencies.

    A real deployment points the durable slot at a database instead; this exists
    so the package WORKS ON INSTALL rather than requiring infrastructure before
    the first session can be opened -- the same reason the PHP reference
    defaults both slots to ``database``.

    Two properties that matter more than speed:

    **Writes are atomic.** A payload is written to a temporary file and renamed
    over the target, because ``os.replace`` is atomic on both POSIX and Windows.
    A partial write here is a corrupted thread, and a process killed mid-write
    is ordinary rather than exotic.

    **The lock is cross-process.** It is an exclusive-create lockfile
    (``O_CREAT | O_EXCL``), the one primitive that is atomic on every filesystem
    worth supporting -- unlike an "is it there?" check followed by a create,
    which has a window between the two. Two workers on one machine genuinely
    exclude each other. Two workers on different machines over a network
    filesystem do NOT reliably, and no file lock can promise that; use a
    database or Redis there.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def durability(self) -> Durability:
        return Durability.DURABLE

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self._path_for(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

        try:
            document = json.loads(raw)
        except json.JSONDecodeError as error:
            raise HarnessError.unmappable_content(
                f"the stored payload for [{key}] is not valid JSON"
            ) from error

        if not isinstance(document, dict):
            raise HarnessError.unmappable_content(
                f"the stored payload for [{key}] is not an object"
            )

        expires_at = document.get("expires_at")
        if isinstance(expires_at, (int, float)) and expires_at <= time.time():
            self.forget(key)
            return None

        payload = document.get("payload")
        return payload if isinstance(payload, dict) else None

    def put(self, key: str, payload: dict[str, Any], ttl_seconds: float | None = None) -> None:
        target = self._path_for(key)
        temporary = target.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
        document = json.dumps(
            {
                "key": key,
                "payload": payload,
                "expires_at": None if ttl_seconds is None else time.time() + ttl_seconds,
            }
        )

        self.directory.mkdir(parents=True, exist_ok=True)
        temporary.write_text(document, encoding="utf-8")

        try:
            # Atomic on both POSIX and Windows. A reader sees the whole old
            # payload or the whole new one, never half of either.
            os.replace(temporary, target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise

    def forget(self, key: str) -> None:
        self._path_for(key).unlink(missing_ok=True)

    def with_lock(
        self,
        key: str,
        callback: Callable[[], T],
        ttl_seconds: float = 10,
        wait_seconds: float = 5,
    ) -> T:
        lock_path = self._path_for(key).with_suffix(".json.lock")
        deadline = time.monotonic() + wait_seconds

        while True:
            try:
                # O_EXCL fails if the file exists. Exclusive CREATION is atomic;
                # a check-then-create is not, and the window between the two is
                # exactly where two workers both decide they hold the lock.
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except _HELD:
                # A lock whose TTL has passed belonged to a process that died
                # holding it. Left alone it would wedge the key forever, which
                # is worse than the small race in reclaiming it -- and the
                # reclaim is itself a create-exclusive, so only one waiter wins.
                if self._expired(lock_path):
                    try:
                        lock_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue

                if time.monotonic() >= deadline:
                    raise HarnessError.session_locked(key, wait_seconds) from None

                time.sleep(0.025)
                continue

            try:
                with os.fdopen(descriptor, "w") as handle:
                    handle.write(str(time.time() + ttl_seconds))

                return callback()
            finally:
                self._release(lock_path)

    @staticmethod
    def _release(lock_path: Path) -> None:
        """Give the lock up, and NEVER leave it held on the way out.

        Deleting is the normal path, but on Windows it can fail: another waiter
        attempting to create the same path holds a transient handle, and the
        unlink then raises PermissionError. Swallowing that -- which is what
        this did first -- LEAKS THE LOCK. The waiter then sees a lockfile whose
        TTL is still in the future, has no way to know its holder is gone, and
        blocks for the whole wait before failing. That is not theoretical: it is
        what two threads recording a message concurrently hit on the first run.

        So if the file cannot be removed, it is rewritten with an ALREADY-PAST
        expiry. Any waiter then reclaims it on its next attempt, which is the
        same path a genuinely dead holder takes.
        """
        for _ in range(5):
            try:
                lock_path.unlink(missing_ok=True)
                return
            except OSError:
                time.sleep(0.005)

        try:
            lock_path.write_text("0", encoding="utf-8")
        except OSError:
            # Nothing left to try. The TTL is the backstop, and it is why the
            # lockfile carries one at all.
            pass

    @staticmethod
    def _expired(lock_path: Path) -> bool:
        try:
            return float(lock_path.read_text(encoding="utf-8")) <= time.time()
        except (OSError, ValueError):
            # Gone or half-written between the failed create and this read:
            # treat it as not expired and let the next attempt take it cleanly.
            return False

    def _path_for(self, key: str) -> Path:
        """One file per key, named by a digest.

        A session key contains colons, which are legal on POSIX and not on
        Windows, and it is long enough to run into path ceilings once a scope is
        appended. The digest sidesteps both, and the key itself is stored INSIDE
        the file so the mapping stays inspectable.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.json"
