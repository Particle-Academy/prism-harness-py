"""Nonce-bound local directory locks. Decision 0022 is the protocol authority."""

from __future__ import annotations

import errno
import json
import math
import os
import re
import secrets
import stat
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

from prism_harness.errors import HarnessError
from prism_harness.stores._file_lock_identity import (
    LockIdentity,
    native_identity,
    pid_is_dead,
    same_identity,
)
from prism_harness.stores._file_lock_read import read_owner_bytes
from prism_harness.stores.base import T


class FileLockOwner(TypedDict):
    version: int
    pid: int
    identity: LockIdentity
    nonce: str
    released: bool
    expires_at: int


@dataclass
class _ReleaseProgress:
    lock: Path
    owner: FileLockOwner
    unlinked: bool = False
    gate: threading.RLock = field(default_factory=threading.RLock, repr=False)


_pending: dict[str, _ReleaseProgress] = {}
_pending_gate = threading.RLock()
_pending_pid = os.getpid()


def _reset_pending_after_fork() -> None:
    global _pending, _pending_gate, _pending_pid
    _pending, _pending_gate, _pending_pid = {}, threading.RLock(), os.getpid()


def pending_file_owner_releases(directory: Path) -> list[dict[str, str]]:
    """Snapshot only: no I/O, retries, raw keys, paths or PID exposure."""
    if _pending_pid != os.getpid():
        return []
    parent = directory.absolute()
    with _pending_gate:
        rows = [
            {
                "key_digest": progress.lock.name.removesuffix(".json.lock"),
                "nonce": progress.owner["nonce"],
            }
            for progress in _pending.values()
            if progress.lock.parent == parent
        ]
    return sorted(rows, key=lambda row: (row["key_digest"], row["nonce"]))


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_pending_after_fork)


_OWNER = re.compile(r"owner-([0-9a-f]{32})\.json")
_HELD = {
    errno.EEXIST,
    errno.ENOTEMPTY,
    errno.EISDIR,
    errno.ENOTDIR,
    errno.EPERM,
    errno.EACCES,
    errno.EBUSY,
}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate owner field")
        result[key] = value
    return result


def _no_float(value: str) -> None:
    raise ValueError("owner numbers must be integers")


def _bounded(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value.encode("utf-8")) <= 256


def parse_owner(raw: bytes, filename: str) -> FileLockOwner | None:
    if len(raw) > 4096 or not raw.endswith(b"\n"):
        return None
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_float=_no_float,
            parse_constant=_no_float,
        )
        match = _OWNER.fullmatch(filename)
        if (
            match is None
            or not isinstance(value, dict)
            or set(value) != {"version", "pid", "identity", "nonce", "released", "expires_at"}
        ):
            return None
        if (
            type(value["version"]) is not int
            or value["version"] != 2
            or type(value["pid"]) is not int
            or not 1 <= value["pid"] <= 4294967295
            or type(value["released"]) is not bool
            or type(value["expires_at"]) is not int
            or abs(value["expires_at"]) > 9007199254740991
            or value["nonce"] != match[1]
        ):
            return None
        identity = value["identity"]
        if (
            not isinstance(identity, dict)
            or set(identity) != {"platform", "boot_id", "pid_namespace"}
            or identity["platform"] not in ("linux", "darwin", "win32")
            or not _bounded(identity["boot_id"])
            or not (identity["pid_namespace"] is None or _bounded(identity["pid_namespace"]))
        ):
            return None
        return cast(FileLockOwner, value)
    except (ValueError, UnicodeError):
        return None


def _name(owner: FileLockOwner) -> str:
    return "owner-" + owner["nonce"] + ".json"


def _payload(owner: FileLockOwner) -> bytes:
    return (json.dumps(owner, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def clean_owner(
    lock: Path,
    owner: FileLockOwner,
    temporary: bool = False,
    progress: _ReleaseProgress | None = None,
) -> None:
    """Missing observed nonce never grants authority to remove canonical dir."""
    if temporary:
        (lock / ("owner-" + owner["nonce"] + ".release.tmp")).unlink(missing_ok=True)
    for attempt in range(5):
        if progress is not None and progress.unlinked:
            break
        try:
            (lock / _name(owner)).unlink()
            if progress is not None:
                progress.unlinked = True
            break
        except FileNotFoundError:
            return
        except OSError as error:
            if attempt == 4 or error.errno not in _HELD:
                raise
            time.sleep(0.005)
    for attempt in range(5):
        try:
            lock.rmdir()  # nonrecursive; populated replacement protects ABA
            return
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST):
                return
            if attempt == 4 or error.errno not in _HELD:
                raise
            time.sleep(0.005)


def _inspect(lock: Path, local: LockIdentity) -> str:
    try:
        info = lock.lstat()
        if stat.S_ISLNK(info.st_mode) or int(getattr(info, "st_file_attributes", 0)) & 0x400:
            return "held"
        if stat.S_ISREG(info.st_mode):
            return "legacy"
        if not stat.S_ISDIR(info.st_mode):
            return "held"
        entries = list(lock.iterdir())
        owners = [path for path in entries if _OWNER.fullmatch(path.name)]
        if len(owners) != 1:
            return "held"
        path = owners[0]
        release_name = path.name.replace(".json", ".release.tmp")
        if any(entry.name not in (path.name, release_name) for entry in entries):
            return "held"
        temporary = any(entry.name == release_name for entry in entries)
        for entry in entries:
            details = entry.lstat()
            if not stat.S_ISREG(details.st_mode) or details.st_size > 4096:
                return "held"
        owner = parse_owner(read_owner_bytes(path), path.name)
        # Do not consult a foreign or unknown PID realm, even if released.
        if owner is None or not same_identity(owner["identity"], local):
            return "held"
        if not owner["released"] and not pid_is_dead(owner["pid"]):
            return "held"
    except OSError as error:
        if error.errno == errno.ENOENT:
            return "retry"
        if error.errno in _HELD:
            return "held"
        raise
    clean_owner(lock, owner, temporary)
    return "retry"


def _release(progress: _ReleaseProgress) -> None:
    lock, owner = progress.lock, progress.owner
    if progress.unlinked:
        clean_owner(lock, owner, progress=progress)
        return
    path = lock / _name(owner)
    try:
        observed = parse_owner(read_owner_bytes(path), path.name)
        if observed is None or observed["nonce"] != owner["nonce"]:
            raise RuntimeError("File lock owner changed before release")
        if observed["released"]:
            clean_owner(lock, owner, temporary=True, progress=progress)
            return
    except FileNotFoundError:
        return
    released: FileLockOwner = {**owner, "released": True}
    temporary = lock / ("owner-" + owner["nonce"] + ".release.tmp")
    # close the complete UTF-8 record before publishing released:true.
    # Only this exact unreleased live owner may rewrite its known nonce temp.
    with temporary.open("wb") as stream:
        stream.write(_payload(released))
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            break
        except OSError as error:
            if attempt == 4 or error.errno not in _HELD:
                raise
            time.sleep(0.005)
    clean_owner(lock, owner, progress=progress)


def _flush_pending() -> None:
    if _pending_pid != os.getpid():
        _reset_pending_after_fork()
    with _pending_gate:
        pending = list(_pending.values())
    for progress in pending:
        with progress.gate:
            nonce = progress.owner["nonce"]
            with _pending_gate:
                if _pending.get(nonce) is not progress:
                    continue
            _release(progress)
            with _pending_gate:
                if _pending.get(nonce) is progress:
                    del _pending[nonce]


def with_file_owner_lock(
    path: Path, key: str, callback: Callable[[], T], ttl: float, wait: float
) -> T:
    if not math.isfinite(ttl) or not math.isfinite(wait) or wait < 0:
        raise ValueError("Invalid file lock duration")
    expires = math.floor(time.time() + ttl)
    if abs(expires) > 9007199254740991:
        raise ValueError("Invalid file lock duration")
    _flush_pending()
    lock = path.absolute()
    try:
        existing = lock.lstat()
        if stat.S_ISREG(existing.st_mode):
            raise HarnessError(
                "session_locked",
                "legacy_lock_requires_maintenance: "
                + str(lock)
                + "; stop_old_workers_then_inspect_and_archive",
            )
    except FileNotFoundError:
        pass
    identity = native_identity()
    if identity is None:
        raise HarnessError("session_locked", "file_lock_identity_unavailable: " + str(lock))
    owner: FileLockOwner = {
        "version": 2,
        "pid": os.getpid(),
        "identity": identity,
        "nonce": secrets.token_hex(16),
        "released": False,
        "expires_at": expires,
    }
    candidate = Path(str(lock) + ".candidate-" + owner["nonce"])
    if candidate.parent != lock.parent or candidate.name == lock.name:
        raise RuntimeError("Lock publication must stay inside its parent")
    deadline = time.monotonic() + wait
    candidate.mkdir()
    published = False
    try:
        with (candidate / _name(owner)).open("xb") as stream:
            stream.write(_payload(owner))
        while True:
            _flush_pending()
            try:
                existing = lock.lstat()
                if (
                    stat.S_ISLNK(existing.st_mode)
                    or int(getattr(existing, "st_file_attributes", 0)) & 0x400
                ):
                    raise HarnessError.session_locked(key, wait)
                if stat.S_ISREG(existing.st_mode):
                    raise HarnessError(
                        "session_locked",
                        "legacy_lock_requires_maintenance: "
                        + str(lock)
                        + "; stop_old_workers_then_inspect_and_archive",
                    )
                if not stat.S_ISDIR(existing.st_mode):
                    raise HarnessError.session_locked(key, wait)
            except FileNotFoundError:
                pass
            try:
                os.rename(candidate, lock)
                published = True
                break
            except OSError as error:
                if error.errno not in _HELD:
                    raise
            state = _inspect(lock, identity)
            if state == "legacy":
                raise HarnessError(
                    "session_locked",
                    "legacy_lock_requires_maintenance: "
                    + str(lock)
                    + "; stop_old_workers_then_inspect_and_archive",
                )
            if state == "retry" and time.monotonic() < deadline:
                continue
            if time.monotonic() >= deadline:
                raise HarnessError.session_locked(key, wait)
            time.sleep(0.025)
    finally:
        if not published:
            clean_owner(candidate, owner)
    # This callback is deliberately outside the acquisition error/retry catch.
    try:
        return callback()
    finally:
        progress = _ReleaseProgress(lock, owner)
        try:
            _release(progress)
        except BaseException:
            with _pending_gate:
                _pending[owner["nonce"]] = progress
            warnings.warn(
                "file_lock_release_pending "
                + json.dumps(
                    {"key_digest": lock.name.removesuffix(".json.lock"), "nonce": owner["nonce"]},
                    separators=(",", ":"),
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            raise
