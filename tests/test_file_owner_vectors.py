"""Production decision0022 record, refusal and nonce-cleanup vectors."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

import pytest

from prism_harness import FileSessionStore, HarnessError
from prism_harness.stores import _file_owner_lock as locks
from prism_harness.stores._file_lock_identity import native_identity


def test_canonical_reparse_metadata_is_refused_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from typing import cast

    store, lock = fixture(tmp_path)
    lock.mkdir()
    original = Path.lstat

    def reparse(path: Path) -> os.stat_result:
        result = original(path)
        if path == lock:
            return cast(
                os.stat_result, SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
            )
        return result

    def forbidden(source: object, target: object) -> None:
        raise AssertionError("unknown canonical reparse metadata must not authorize rename")

    monkeypatch.setattr(Path, "lstat", reparse)
    monkeypatch.setattr(os, "rename", forbidden)
    with pytest.raises(HarnessError):
        store.with_lock("k", lambda: pytest.fail("must not enter"), wait_seconds=0)
    assert lock.is_dir() and list(lock.iterdir()) == []


def owner() -> locks.FileLockOwner:
    identity = native_identity()
    assert identity is not None and identity["pid_namespace"] is not None
    return {
        "version": 2,
        "pid": os.getpid(),
        "identity": identity,
        "nonce": secrets.token_hex(16),
        "released": False,
        "expires_at": 0,
    }


def name(value: locks.FileLockOwner) -> str:
    return "owner-" + value["nonce"] + ".json"


def raw(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")


def fixture(tmp_path: Path) -> tuple[FileSessionStore, Path]:
    store = FileSessionStore(tmp_path)
    return store, store._path_for("k").with_suffix(".json.lock")


def plant(lock: Path, value: locks.FileLockOwner) -> None:
    lock.mkdir()
    (lock / name(value)).write_bytes(raw(value))


def test_closed_record_parser_refuses_schema_and_encoding_violations() -> None:
    valid = owner()
    filename = name(valid)
    assert locks.parse_owner(raw(valid), filename) == valid
    bad: list[dict[str, Any]] = [
        {**valid, "pid": True},
        {**valid, "pid": 0},
        {**valid, "pid": 4294967296},
        {**valid, "version": True},
        {**valid, "released": 0},
        {**valid, "expires_at": 0.1},
        {**valid, "unknown": True},
        {**valid, "identity": None},
        {**valid, "identity": {**valid["identity"], "unknown": 1}},
        {**valid, "identity": {**valid["identity"], "platform": ["linux"]}},
        {**valid, "identity": {**valid["identity"], "boot_id": ""}},
        {**valid, "identity": {**valid["identity"], "boot_id": "\ud800"}},
        {**valid, "identity": {**valid["identity"], "boot_id": "é" * 129}},
        {**valid, "identity": {**valid["identity"], "pid_namespace": ""}},
        {**valid, "nonce": "a" * 32},
    ]
    for value in bad:
        assert locks.parse_owner(raw(value), filename) is None
    for data in [
        raw(valid)[:-1],
        b'{"version":2,"pid":\n',
        raw(valid).replace(b'"version":2', b'"version":2,"version":2'),
        raw(valid).replace(b'"version":2', b'"version":2.0'),
        b"\xef\xbb\xbf" + raw(valid),
        b"\xff" + raw(valid),
        b" " * 4097,
    ]:
        assert locks.parse_owner(data, filename) is None


@pytest.mark.parametrize("kind", ["boot", "realm", "platform", "foreign-released", "unknown"])
def test_foreign_identity_never_probes_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    store, lock = fixture(tmp_path)
    value = owner()
    if kind == "boot":
        value["identity"]["boot_id"] = "different-boot"
    if kind in ("realm", "foreign-released"):
        value["identity"]["pid_namespace"] = "foreign-realm"
    if kind == "platform":
        value["identity"]["platform"] = "linux" if os.name == "nt" else "win32"
    if kind == "foreign-released":
        value["released"] = True
    if kind == "unknown":
        value["identity"]["pid_namespace"] = None
        value["released"] = True
    plant(lock, value)
    before = (lock / name(value)).read_bytes()

    def forbidden(pid: int) -> bool:
        raise AssertionError("Foreign PID must never be probed")

    monkeypatch.setattr(locks, "pid_is_dead", forbidden)
    with pytest.raises(HarnessError) as failure:
        store.with_lock("k", lambda: pytest.fail("must not enter"), wait_seconds=0)
    assert failure.value.code == "session_locked"
    assert (lock / name(value)).read_bytes() == before


def test_live_past_ttl_held_and_explicitly_released_owner_reclaimed(tmp_path: Path) -> None:
    store, lock = fixture(tmp_path)
    value = owner()
    plant(lock, value)
    with pytest.raises(HarnessError):
        store.with_lock("k", lambda: 1, wait_seconds=0)
    value["released"] = True
    (lock / name(value)).write_bytes(raw(value))
    assert store.with_lock("k", lambda: 7, wait_seconds=0.1) == 7
    assert not lock.exists()


def test_unknown_liveness_keeps_matching_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, lock = fixture(tmp_path)
    value = owner()
    plant(lock, value)
    monkeypatch.setattr(locks, "pid_is_dead", lambda pid: False)
    with pytest.raises(HarnessError):
        store.with_lock("k", lambda: 1, wait_seconds=0)
    assert (lock / name(value)).read_bytes() == raw(value)


def test_unknown_entry_prevents_released_owner_cleanup(tmp_path: Path) -> None:
    store, lock = fixture(tmp_path)
    value = owner()
    value["released"] = True
    plant(lock, value)
    (lock / "unexpected").write_text("keep")
    with pytest.raises(HarnessError):
        store.with_lock("k", lambda: 1, wait_seconds=0)
    assert sorted(path.name for path in lock.iterdir()) == sorted([name(value), "unexpected"])


def test_missing_observed_nonce_does_not_remove_empty_canonical(tmp_path: Path) -> None:
    _, lock = fixture(tmp_path)
    lock.mkdir()
    locks.clean_owner(lock, owner())
    assert lock.is_dir()


def test_stale_cleanup_preserves_populated_replacement(tmp_path: Path) -> None:
    _, lock = fixture(tmp_path)
    old, replacement = owner(), owner()
    plant(lock, old)
    (lock / name(old)).unlink()
    lock.rmdir()
    plant(lock, replacement)
    locks.clean_owner(lock, old)
    assert (lock / name(replacement)).read_bytes() == raw(replacement)


def test_native_empty_directory_publication_semantics(tmp_path: Path) -> None:
    store, lock = fixture(tmp_path)
    lock.mkdir()
    if os.name == "nt":
        with pytest.raises(HarnessError):
            store.with_lock("k", lambda: 1, wait_seconds=0)
        assert list(lock.iterdir()) == []
    else:
        assert store.with_lock("k", lambda: 1, wait_seconds=0) == 1
