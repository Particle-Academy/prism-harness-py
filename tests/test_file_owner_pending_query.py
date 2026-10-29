"""Two concurrent failed releases remain visible independent of warning filters."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import threading
import warnings
from pathlib import Path
from typing import Any

import pytest

from prism_harness import FileSessionStore


def test_two_pending_releases_visible_through_store_without_io_or_raw_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSessionStore(tmp_path / "owned")
    other = FileSessionStore(tmp_path / "other")
    keys = ["private:pending:A", "private:pending:B"]
    barrier = threading.Barrier(2)
    gate = threading.Lock()
    callbacks: list[str] = []
    failures: list[BaseException] = []
    attempts: list[str] = []
    original = os.replace

    def blocked(source: Any, target: Any) -> None:
        if str(source).endswith(".release.tmp"):
            with gate:
                attempts.append(str(source))
            raise PermissionError(errno.EACCES, "owned denial")
        original(source, target)

    def run(key: str) -> None:
        def callback() -> None:
            with gate:
                callbacks.append(key)
            barrier.wait(timeout=10)

        try:
            store.with_lock(key, callback)
        except (OSError, threading.BrokenBarrierError) as error:
            with gate:
                failures.append(error)

    with monkeypatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(os, "replace", blocked)
        threads = [threading.Thread(target=run, args=(key,)) for key in keys]
        for worker in threads:
            worker.start()
        for worker in threads:
            worker.join(timeout=15)
        assert all(not worker.is_alive() for worker in threads)
        assert sorted(callbacks) == sorted(keys)
        assert len(failures) == 2 and all(isinstance(error, PermissionError) for error in failures)
        before = len(attempts)
        rows = store.pending_releases()
        assert len(rows) == 2
        assert FileSessionStore(tmp_path / "owned").pending_releases() == rows
        assert other.pending_releases() == []
        assert {row["key_digest"] for row in rows} == {
            hashlib.sha256(key.encode()).hexdigest()[:32] for key in keys
        }
        assert all(set(row) == {"key_digest", "nonce"} and len(row["nonce"]) == 32 for row in rows)
        assert "private:pending:" not in json.dumps(rows) and str(tmp_path) not in json.dumps(rows)
        rows[0]["nonce"] = "mutated-copy"
        assert all(row["nonce"] != "mutated-copy" for row in store.pending_releases())
        assert len(attempts) == before, "querying pending releases must not retry or write"
    assert store.with_lock("next-operation", lambda: 7) == 7
    assert store.pending_releases() == []
