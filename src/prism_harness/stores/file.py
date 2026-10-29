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
from prism_harness.stores._file_owner_lock import pending_file_owner_releases, with_file_owner_lock
from prism_harness.stores.base import Durability, T

__all__ = ["FileSessionStore"]


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

    **The lock is cross-process.** Decision 0022 atomically publishes a complete
    owner directory. TTL never proves death. Local filesystems on one host only;
    SMB/NFS are unsupported. Stop all old workers before upgrading and inspect
    legacy timestamp locks explicitly. Use a database or Redis across hosts.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def durability(self) -> Durability:
        return Durability.DURABLE

    def lock_scope(self) -> str:
        return "host"

    def sharing_scope(self) -> str:
        """Local filesystem on one host. SMB/NFS sharing is unsupported."""
        return "host"

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
        return with_file_owner_lock(
            self._path_for(key).with_suffix(".json.lock"), key, callback, ttl_seconds, wait_seconds
        )

    def pending_releases(self) -> list[dict[str, str]]:
        """Digest/nonce snapshots for this directory; len() is the pending count."""
        return pending_file_owner_releases(self.directory)

    def _path_for(self, key: str) -> Path:
        """One file per key, named by a digest.

        A session key contains colons, which are legal on POSIX and not on
        Windows, and it is long enough to run into path ceilings once a scope is
        appended. The digest sidesteps both, and the key itself is stored INSIDE
        the file so the mapping stays inspectable.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.json"
