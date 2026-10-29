"""Production release publication retry and per-process identity cache controls."""

from __future__ import annotations

import errno
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from prism_harness import FileSessionStore
from prism_harness.stores import _file_lock_identity as identity
from prism_harness.stores._file_lock_read import shared_owner_reader


def test_release_retries_transient_replace_without_repeating_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSessionStore(tmp_path)
    original = os.replace
    callbacks: list[bool] = []
    attempts: list[bytes] = []

    def transient(source: Any, target: Any) -> None:
        if str(source).endswith(".release.tmp"):
            attempts.append(Path(source).read_bytes())
            assert json.loads(attempts[-1])["released"] is True
            if len(attempts) < 3:
                raise PermissionError(errno.EACCES, "owned transient reader")
        original(source, target)

    monkeypatch.setattr(os, "replace", transient)
    assert store.with_lock("k", lambda: callbacks.append(True)) is None
    assert callbacks == [True] and len(attempts) == 3
    assert attempts[0] == attempts[1] == attempts[2]
    assert list(tmp_path.iterdir()) == []


def test_release_publish_exhaustion_is_observable_and_not_reclaimable_by_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSessionStore(tmp_path)
    original = os.replace
    callbacks: list[bool] = []

    def blocked(source: Any, target: Any) -> None:
        if str(source).endswith(".release.tmp"):
            raise PermissionError(errno.EACCES, "owned persistent reader")
        original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", blocked)
        with pytest.raises(PermissionError):
            store.with_lock("k", lambda: callbacks.append(True), ttl_seconds=-1)
        assert callbacks == [True]
        lock = store._path_for("k").with_suffix(".json.lock")
        files = sorted(lock.iterdir())
        assert len(files) == 2
        owner = next(path for path in files if path.suffix == ".json")
        temporary = next(path for path in files if path.suffix == ".tmp")
        assert json.loads(owner.read_bytes())["released"] is False
        assert json.loads(temporary.read_bytes())["released"] is True
        with pytest.raises(PermissionError):
            store.with_lock(
                "k", lambda: pytest.fail("pending failure must remain observable"), wait_seconds=0
            )
    assert store.with_lock("other-key", lambda: "flushed") == "flushed"
    assert store.with_lock("k", lambda: 7) == 7
    assert callbacks == [True]


def test_identity_cache_is_not_inherited_as_proof_in_another_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = identity.native_identity()
    assert native is not None and native["pid_namespace"] is not None
    stale = native.copy()
    stale["pid_namespace"] = "synthetic-parent-only-realm"
    monkeypatch.setattr(identity, "_established", stale)
    monkeypatch.setattr(identity, "_established_pid", os.getpid() + 1)
    assert identity.native_identity() == native


def test_pending_cleanup_retains_successful_unlink_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSessionStore(tmp_path)
    lock = store._path_for("k").with_suffix(".json.lock")
    original = Path.rmdir

    def blocked(path: Path) -> None:
        if path == lock:
            raise PermissionError(errno.EACCES, "owned cleanup denial")
        original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rmdir", blocked)
        with pytest.raises(PermissionError):
            store.with_lock("k", lambda: 7)
        assert lock.is_dir() and list(lock.iterdir()) == []
    # This is the recorded successful unlink, not a new grant from a missing file.
    assert store.with_lock("other-key", lambda: 9) == 9
    assert not lock.exists()


def test_native_shared_reader_replacement_and_pending_completion_after_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSessionStore(tmp_path)
    original = os.replace
    attempts, failures = 0, 0

    def measured(source: Any, target: Any) -> None:
        nonlocal attempts, failures
        if str(source).endswith(".release.tmp"):
            attempts += 1
            try:
                original(source, target)
            except OSError:
                failures += 1
                raise
        else:
            original(source, target)

    monkeypatch.setattr(os, "replace", measured)
    with ExitStack() as readers:

        def hold_shared_reader() -> int:
            lock = store._path_for("k").with_suffix(".json.lock")
            path = next(path for path in lock.iterdir() if path.suffix == ".json")
            if os.name == "nt":
                readers.enter_context(shared_owner_reader(path))
            else:
                readers.enter_context(path.open("rb"))
            return 7

        if os.name == "nt":
            with pytest.raises(PermissionError):
                store.with_lock("k", hold_shared_reader)
            assert attempts > 0 and failures == attempts
        else:
            assert store.with_lock("k", hold_shared_reader) == 7
            assert attempts == 1 and failures == 0
        print(
            json.dumps(
                {
                    "native_reader": "share-read-write-delete",
                    "platform": os.name,
                    "attempts": attempts,
                    "failures": failures,
                }
            )
        )
    assert store.with_lock("other-key", lambda: 9) == 9
    assert not store._path_for("k").with_suffix(".json.lock").exists()


def test_pending_diagnostic_contains_digest_and_nonce_without_raw_key_or_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import warnings

    store = FileSessionStore(tmp_path)
    key = "private:participant:do-not-log"
    original = os.replace

    def blocked(source: Any, target: Any) -> None:
        if str(source).endswith(".release.tmp"):
            raise PermissionError(errno.EACCES, "owned denial")
        original(source, target)

    with monkeypatch.context() as patch, warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        patch.setattr(os, "replace", blocked)
        with pytest.raises(PermissionError):
            store.with_lock(key, lambda: 1)
    messages = [str(item.message) for item in emitted]
    assert len(messages) == 1 and "file_lock_release_pending" in messages[0]
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    assert digest in messages[0] and key not in messages[0] and str(tmp_path) not in messages[0]
    body = json.loads(messages[0].split(" ", 1)[1])
    assert set(body) == {"key_digest", "nonce"} and len(body["nonce"]) == 32
    assert store.with_lock("other-key", lambda: 7) == 7
