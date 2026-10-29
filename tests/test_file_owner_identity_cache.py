"""Definitive identity is cached; transient failures retry after a bounded interval."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from prism_harness.stores import _file_lock_identity as identity

_HOST: identity.LockIdentity = {
    "platform": "win32",
    "boot_id": "cim-utc-ticks:638000000000000000",
    "pid_namespace": "win32:kernel",
}


@pytest.fixture(autouse=True)
def preserve_identity_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "_established",
        "_established_pid",
        "_identity_gate",
        "_retry_at",
        "_definitive",
        "_duration_ms",
    ):
        monkeypatch.setattr(identity, name, getattr(identity, name))


def test_transient_failure_retries_after_interval_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    calls: list[int] = []

    def read() -> identity.LockIdentity | None:
        calls.append(os.getpid())
        time.sleep(0.01)
        return None if len(calls) == 1 else _HOST.copy()

    monkeypatch.setattr(identity, "_established_pid", None)
    monkeypatch.setattr(identity, "_read_native_identity", read)
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    assert identity.file_lock_identity_status()["sampled"] is False and not calls
    assert identity.native_identity() is None
    now[0] = 59.999
    assert identity.native_identity() is None and len(calls) == 1
    assert identity.file_lock_identity_status()["code"] == "file_lock_identity_unavailable"
    now[0] = 60.0
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(lambda _: identity.native_identity(), range(8))) == [_HOST] * 8
    assert len(calls) == 2
    now[0] = 99999999
    assert identity.native_identity() == _HOST and len(calls) == 2
    assert identity.file_lock_identity_status()["definitive"] is True


def test_positive_container_result_never_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    calls: list[int] = []

    def read() -> identity.LockIdentity:
        calls.append(1)
        return {**_HOST, "pid_namespace": None}

    monkeypatch.setattr(identity, "_established_pid", None)
    monkeypatch.setattr(identity, "_read_native_identity", read)
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    assert identity.native_identity() == {**_HOST, "pid_namespace": None}
    now[0] = 99999999
    assert identity.native_identity() == {**_HOST, "pid_namespace": None} and len(calls) == 1
    assert identity.file_lock_identity_status()["definitive"] is True


@pytest.mark.parametrize("first", [None, {**_HOST, "pid_namespace": None}, _HOST])
def test_identity_including_unknown_is_sampled_once_under_concurrency(
    first: identity.LockIdentity | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def read() -> identity.LockIdentity | None:
        calls.append(os.getpid())
        time.sleep(0.01)
        return first

    monkeypatch.setattr(identity, "_established", None)
    monkeypatch.setattr(identity, "_established_pid", None)
    monkeypatch.setattr(identity, "_identity_gate", threading.Lock())
    monkeypatch.setattr(identity, "_read_native_identity", read)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: identity.native_identity(), range(8)))
    assert results == [first] * 8
    assert calls == [os.getpid()]
    monkeypatch.setattr(identity, "_read_native_identity", lambda: _HOST)
    assert identity.native_identity() == first
    if results[0] is not None:
        results[0]["pid_namespace"] = "caller-mutated-copy"
        assert identity.native_identity() == first


def test_fork_reset_discards_parent_cache_and_locked_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    parent_gate = threading.Lock()
    parent_gate.acquire()
    monkeypatch.setattr(identity, "_identity_gate", parent_gate)
    monkeypatch.setattr(identity, "_established", _HOST)
    monkeypatch.setattr(identity, "_established_pid", os.getpid())
    monkeypatch.setattr(identity, "_read_native_identity", lambda: None)
    try:
        identity._reset_identity_after_fork()
        assert identity._identity_gate is not parent_gate
        assert identity.native_identity() is None
    finally:
        parent_gate.release()
