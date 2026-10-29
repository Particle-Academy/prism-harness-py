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
            assert 0.70 <= elapsed < 2.0
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
