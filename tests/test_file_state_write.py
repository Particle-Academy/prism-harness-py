"""G163: real target readers must not turn transient contention into lost state."""

from __future__ import annotations

import errno
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from prism_harness import FileSessionStore


@contextmanager
def held_reader(target: Path, sharing: str) -> Iterator[subprocess.Popen[str]]:
    with subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "fixtures/file_state_reader.py"),
            str(target),
            sharing,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        stream = child.stdout
        assert stream is not None
        ready: queue.Queue[str] = queue.Queue()
        threading.Thread(target=lambda: ready.put(stream.readline()), daemon=True).start()
        try:
            assert ready.get(timeout=5).strip() == "ready"
            yield child
        finally:
            close_reader(child)


def close_reader(child: subprocess.Popen[str]) -> None:
    if child.poll() is None:
        assert child.stdin is not None
        child.stdin.close()
        try:
            assert child.wait(timeout=5) == 0
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
            raise
    assert child.returncode == 0


@pytest.mark.parametrize("sharing", ["ordinary", "share-delete"])
def test_write_waits_for_native_reader_then_publishes_whole_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sharing: str,
) -> None:
    store = FileSessionStore(tmp_path)
    store.put("k", {"revision": 0})
    target = store._path_for("k")
    replace = os.replace
    blocked = threading.Event()
    errors: list[PermissionError] = []

    def observed_replace(source: Path, destination: Path) -> None:
        try:
            replace(source, destination)
        except PermissionError as failure:
            errors.append(failure)
            blocked.set()
            raise

    monkeypatch.setattr(os, "replace", observed_replace)
    with held_reader(target, sharing) as reader, ThreadPoolExecutor(max_workers=1) as pool:
        write = pool.submit(store.put, "k", {"revision": 1, "body": "界" * 70000})
        if sys.platform == "win32":
            assert blocked.wait(timeout=3), "must observe a real native permission failure"
            assert not write.done()
            close_reader(reader)
        # On POSIX this must complete while the reader still has its old handle.
        write.result(timeout=3)
    assert bool(errors) is (sys.platform == "win32")
    assert store.get("k") == {"revision": 1, "body": "界" * 70000}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("sharing", ["ordinary", "share-delete"])
def test_permanent_native_reader_preserves_original_error_old_state_and_no_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sharing: str,
) -> None:
    store = FileSessionStore(tmp_path)
    store.put("k", {"revision": 0})
    target = store._path_for("k")
    original_bytes = target.read_bytes()
    replace = os.replace
    errors: list[PermissionError] = []

    def observed_replace(source: Path, destination: Path) -> None:
        try:
            replace(source, destination)
        except PermissionError as failure:
            errors.append(failure)
            raise

    monkeypatch.setattr(os, "replace", observed_replace)
    with held_reader(target, sharing):
        started = time.monotonic()
        if sys.platform == "win32":
            with pytest.raises(PermissionError) as caught:
                store.put("k", {"revision": 1})
            elapsed = time.monotonic() - started
            # A lower bound only: the retry budget itself is pinned in-process,
            # with an injected clock, by
            # test_the_windows_replace_retry_budget_is_its_deadline_not_the_wall_clock.
            # An upper wall-clock bound measured a loaded runner, not the code.
            assert elapsed >= 0.70
            assert caught.value is errors[0]
            assert getattr(caught.value, "winerror", None) == 5
            assert target.read_bytes() == original_bytes
        else:
            store.put("k", {"revision": 1})
            assert not errors
    assert list(tmp_path.iterdir()) == [target]
    assert store.get("k") == {"revision": 0 if sys.platform == "win32" else 1}


@pytest.mark.parametrize(
    "failure",
    [
        OSError(errno.EIO, "synthetic IO"),
        PermissionError(errno.EACCES, "synthetic POSIX permission"),
    ],
)
def test_non_windows_errors_are_not_retried_or_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: OSError,
) -> None:
    store = FileSessionStore(tmp_path)
    store.put("k", {"revision": 0})
    calls = 0

    def fail(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError) as caught:
        store.put("k", {"revision": 1})
    assert caught.value is failure
    assert calls == 1
    assert list(tmp_path.iterdir()) == [store._path_for("k")]


def test_windows_non_permission_error_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSessionStore(tmp_path)
    store.put("k", {"revision": 0})
    failure = OSError(errno.EIO, "synthetic IO")
    calls = 0

    def fail(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError) as caught:
        store.put("k", {"revision": 1})
    assert caught.value is failure
    assert calls == 1
    assert list(tmp_path.iterdir()) == [store._path_for("k")]


def test_state_reader_is_closed_before_json_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSessionStore(tmp_path)
    store.put("k", {"revision": 1})
    target = store._path_for("k")
    replacement = tmp_path / "replacement"
    replacement.write_text(json.dumps({"key": "k", "payload": {"revision": 2}}), encoding="utf-8")
    decode = json.loads

    def decoding_after_read(raw: str) -> Any:
        # A live Windows read handle would make this native operation fail.
        os.replace(replacement, target)
        return decode(raw)

    with monkeypatch.context() as patch:
        patch.setattr(json, "loads", decoding_after_read)
        assert store.get("k") == {"revision": 1}
    assert store.get("k") == {"revision": 2}


def test_the_windows_replace_retry_budget_is_its_deadline_not_the_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic: the retry loop's 0.75 s deadline, read from an injected
    clock that only the loop's own sleeps advance, with a stubbed rename that
    always fails. The original error propagates, the loop sleeps exactly its
    budget in steps of at most 10 ms, and it never sleeps past the deadline."""
    from prism_harness.stores import _file_state_io as io

    now = [100.0]
    slept: list[float] = []
    attempts: list[int] = []
    first = PermissionError(errno.EACCES, "held")

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    def replace(source: Path, destination: Path) -> None:
        attempts.append(1)
        raise first if len(attempts) == 1 else PermissionError(errno.EACCES, "still held")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(PermissionError) as caught:
        io.replace_state_file(tmp_path / "a.tmp", tmp_path / "a")

    assert caught.value is first
    assert all(0 < seconds <= 0.01 for seconds in slept)
    assert sum(slept) == pytest.approx(0.75)
    assert now[0] - 100.0 <= 0.75 + 1e-9
    # One first attempt, then one per sleep that left time on the clock.
    assert len(attempts) in (len(slept), len(slept) + 1)
    assert 75 <= len(slept) <= 76


def test_the_windows_replace_retry_stops_at_the_first_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prism_harness.stores import _file_state_io as io

    now = [0.0]
    attempts: list[int] = []

    def replace(source: Path, destination: Path) -> None:
        attempts.append(1)
        if len(attempts) < 4:
            raise PermissionError(errno.EACCES, "held")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(os, "replace", replace)

    io.replace_state_file(tmp_path / "a.tmp", tmp_path / "a")

    assert len(attempts) == 4
    assert now[0] == pytest.approx(0.03)
