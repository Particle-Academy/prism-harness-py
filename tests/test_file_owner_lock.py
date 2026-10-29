"""Production owner-proof lock boundary, decision 0022."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from prism_harness import FileSessionStore, HarnessError


def test_publishes_complete_v2_owner_directory_before_callback(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    lock = store._path_for("k").with_suffix(".json.lock")

    def inspect() -> None:
        assert lock.is_dir()
        files = list(lock.iterdir())
        assert len(files) == 1
        raw = files[0].read_bytes()
        assert raw.endswith(b"\n")
        owner = json.loads(raw)
        assert set(owner) == {"version", "pid", "identity", "nonce", "released", "expires_at"}
        assert owner["version"] == 2 and owner["pid"] == os.getpid()
        assert owner["released"] is False
        assert len(owner["nonce"]) == 32 and all(
            value in "0123456789abcdef" for value in owner["nonce"]
        )
        assert files[0].name == "owner-" + owner["nonce"] + ".json"
        assert type(owner["expires_at"]) is int
        assert owner["identity"]["platform"] in ("linux", "darwin", "win32")
        assert isinstance(owner["identity"]["boot_id"], str)
        assert isinstance(owner["identity"]["pid_namespace"], str)

    store.with_lock("k", inspect)


def test_expired_legacy_lock_requires_explicit_maintenance(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    lock = store._path_for("k").with_suffix(".json.lock")
    lock.write_text("1\n", encoding="utf-8")
    ran: list[bool] = []
    with pytest.raises(HarnessError, match="legacy_lock_requires_maintenance") as failure:
        store.with_lock("k", lambda: ran.append(True), wait_seconds=0)
    assert failure.value.code == "session_locked"
    assert str(lock) in str(failure.value)
    assert "stop_old_workers_then_inspect_and_archive" in str(failure.value)
    assert ran == [] and lock.read_text(encoding="utf-8") == "1\n"


def test_legacy_diagnostic_does_not_require_native_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prism_harness.stores import _file_owner_lock as locks

    store = FileSessionStore(tmp_path)
    lock = store._path_for("k").with_suffix(".json.lock")
    lock.write_text("1\n", encoding="utf-8")
    calls: list[bool] = []

    def unavailable() -> None:
        calls.append(True)

    monkeypatch.setattr(locks, "native_identity", unavailable)
    with pytest.raises(HarnessError, match="legacy_lock_requires_maintenance"):
        store.with_lock("k", lambda: pytest.fail("legacy callback"), wait_seconds=0)
    assert calls == [] and lock.read_text(encoding="utf-8") == "1\n"


def test_propagates_callback_filesystem_errors_without_retry(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    original = FileExistsError("callback failure")
    callbacks: list[bool] = []

    def fail() -> None:
        callbacks.append(True)
        raise original

    with pytest.raises(FileExistsError) as failure:
        store.with_lock("k", fail, wait_seconds=0.12)
    assert failure.value is original and callbacks == [True]
