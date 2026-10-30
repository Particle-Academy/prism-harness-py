from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import pytest

from prism_harness import FileSessionStore, HarnessError
from prism_harness.stores import _file_owner_lock as ownership
from prism_harness.stores._file_lock_identity import native_identity


@pytest.mark.parametrize("action", ["release", "crash"])
@pytest.mark.parametrize("blocked_command_path", [False, True])
def test_real_unknown_worker_excludes_and_preserves_release_or_crash(
    action: str, blocked_command_path: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known = native_identity()
    assert known is not None and known["pid_namespace"] is not None
    nonce = secrets.token_hex(16)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"root": str(tmp_path), "nonce": nonce, "action": action}))
    if blocked_command_path:
        # Unrelated namespace entries must not become the process command channel.
        (tmp_path / "command.json").mkdir()
    script = Path(__file__).parent / "fixtures/file_owner_unknown_holder.py"
    child = subprocess.Popen(
        [sys.executable, str(script), str(spec)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    commanded = False

    def command() -> None:
        assert child.stdin is not None
        child.stdin.write(json.dumps({"nonce": nonce, "action": action}) + "\n")
        child.stdin.flush()

    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready.json").exists():
            assert child.poll() is None, "Owned unknown worker exited before acquisition"
            assert time.monotonic() < deadline, "Owned unknown worker setup timed out"
            time.sleep(0.01)
        ready = json.loads((tmp_path / "ready.json").read_text())
        # Windows venv executables can be redirectors with a separate launcher PID.
        assert set(ready) == {"nonce", "pid"} and ready["nonce"] == nonce
        assert type(ready["pid"]) is int and ready["pid"] > 0 and ready["pid"] != os.getpid()
        store = FileSessionStore(tmp_path / "store")
        lock = store._path_for("unknown-key").with_suffix(".json.lock")
        names = list(lock.iterdir())
        assert len(names) == 1
        before = names[0].read_bytes()
        owner = json.loads(before)
        assert owner["pid"] == ready["pid"] and owner["released"] is False
        assert owner["identity"] == {
            "platform": sys.platform,
            "boot_id": "unknown",
            "pid_namespace": None,
        }

        def forbidden_probe(pid: int) -> bool:
            raise AssertionError("Unknown owner must not authorize a PID probe")

        monkeypatch.setattr(ownership, "pid_is_dead", forbidden_probe)
        calls = []
        with pytest.raises(HarnessError) as caught:
            store.with_lock("unknown-key", lambda: calls.append(1), wait_seconds=0)
        assert caught.value.code == "session_locked" and not calls and child.poll() is None
        assert names[0].read_bytes() == before
        command()
        commanded = True
        _, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        if action == "crash":
            time.sleep(1.1)
            assert owner["expires_at"] <= int(time.time())
            with pytest.raises(HarnessError) as caught:
                store.with_lock("unknown-key", lambda: calls.append(1), wait_seconds=0)
            assert caught.value.code == "session_locked" and not calls
            assert names[0].read_bytes() == before
        else:
            assert not lock.exists()
            store.with_lock("unknown-key", lambda: calls.append(1), wait_seconds=0)
            assert calls == [1]
    finally:
        if not commanded and child.poll() is None:
            command()
        child.communicate(timeout=30)


@pytest.mark.parametrize("wire", ["", "{}\n", '{"nonce":"wrong","action":"crash"}\n'])
def test_unknown_holder_refuses_eof_malformed_and_foreign_pipe_commands(
    wire: str, tmp_path: Path
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"root": str(tmp_path), "nonce": secrets.token_hex(16), "action": "crash"})
    )
    script = Path(__file__).parent / "fixtures/file_owner_unknown_holder.py"
    child = subprocess.Popen(
        [sys.executable, str(script), str(spec)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    _, stderr = child.communicate(input=wire, timeout=30)
    assert (tmp_path / "ready.json").exists(), stderr
    assert child.returncode != 0, "An invalid command must not execute the requested crash"
    assert "JSONDecodeError" in stderr or "AssertionError" in stderr


def test_unknown_identity_acquires_releases_and_excludes_another_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSessionStore(tmp_path)
    lock = tmp_path / (hashlib.sha256(b"k").hexdigest()[:32] + ".json.lock")
    monkeypatch.setattr(ownership, "native_identity", lambda: None)

    def forbidden_probe(pid: int) -> bool:
        raise AssertionError("Unknown local identity cannot probe a PID")

    monkeypatch.setattr(ownership, "pid_is_dead", forbidden_probe)
    calls = []

    def callback() -> None:
        calls.append(1)
        files = list(lock.iterdir())
        assert len(files) == 1
        owner = json.loads(files[0].read_bytes())
        assert owner["pid"] == os.getpid() and not owner["released"]
        assert owner["identity"] == {
            "platform": sys.platform,
            "boot_id": "unknown",
            "pid_namespace": None,
        }
        with pytest.raises(HarnessError) as caught:
            FileSessionStore(tmp_path).with_lock(
                "k", lambda: pytest.fail("Contender entered"), wait_seconds=0
            )
        assert caught.value.code == "session_locked"

    with pytest.warns(RuntimeWarning, match="file_lock_identity_unavailable"):
        store.with_lock("k", callback)
    assert calls == [1] and not lock.exists()
    with pytest.warns(RuntimeWarning, match="file_lock_identity_unavailable"):
        assert FileSessionStore(tmp_path).with_lock("k", lambda: 7) == 7


@pytest.mark.parametrize("released", [False, True])
def test_unknown_owner_never_authorizes_reclaim(
    released: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / (hashlib.sha256(b"k").hexdigest()[:32] + ".json.lock")
    lock.mkdir()
    nonce = secrets.token_hex(16)
    path = lock / f"owner-{nonce}.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "pid": os.getpid(),
                "identity": {"platform": sys.platform, "boot_id": "unknown", "pid_namespace": None},
                "nonce": nonce,
                "released": released,
                "expires_at": 0,
            }
        )
        + "\n"
    )
    before = path.read_bytes()

    def forbidden_probe(pid: int) -> bool:
        raise AssertionError("Unknown owner cannot authorize a probe")

    monkeypatch.setattr(ownership, "pid_is_dead", forbidden_probe)
    with pytest.raises(HarnessError) as caught:
        FileSessionStore(tmp_path).with_lock(
            "k", lambda: pytest.fail("Unknown owner reclaimed"), wait_seconds=0
        )
    assert caught.value.code == "session_locked" and path.read_bytes() == before


def test_unknown_prober_never_reclaims_a_known_released_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = native_identity()
    assert known is not None and known["pid_namespace"] is not None
    lock = tmp_path / (hashlib.sha256(b"k").hexdigest()[:32] + ".json.lock")
    lock.mkdir()
    nonce = secrets.token_hex(16)
    path = lock / f"owner-{nonce}.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "pid": os.getpid(),
                "identity": known,
                "nonce": nonce,
                "released": True,
                "expires_at": 0,
            }
        )
        + "\n"
    )
    before = path.read_bytes()
    monkeypatch.setattr(ownership, "native_identity", lambda: None)

    def forbidden_probe(pid: int) -> bool:
        raise AssertionError("Unknown prober cannot probe")

    monkeypatch.setattr(ownership, "pid_is_dead", forbidden_probe)
    with (
        pytest.warns(RuntimeWarning, match="file_lock_identity_unavailable"),
        pytest.raises(HarnessError) as caught,
    ):
        FileSessionStore(tmp_path).with_lock(
            "k", lambda: pytest.fail("Unknown prober reclaimed"), wait_seconds=0
        )
    assert caught.value.code == "session_locked" and path.read_bytes() == before
