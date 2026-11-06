"""Native production owner protocol, also executed under owned Windows Jobs."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import pytest

from prism_harness import FileSessionStore, HarnessError
from prism_harness.stores import _file_owner_lock as locks
from prism_harness.stores._file_lock_identity import (
    file_lock_identity_status,
    native_identity,
    pid_is_dead,
)


@pytest.mark.parametrize("blocked_control_paths", [False, True])
def test_native_independent_live_dead_and_foreign_unknown_production_rows(
    blocked_control_paths: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = file_lock_identity_status()
    started = time.monotonic()
    local = native_identity()
    print(
        json.dumps(
            {
                "native_identity_setup": True,
                "platform": sys.platform,
                "elapsed_ms": (time.monotonic() - started) * 1000,
                "before": before,
                "status": file_lock_identity_status(),
            }
        )
    )
    assert local is not None and local["pid_namespace"] is not None
    nonce = secrets.token_hex(16)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"root": str(tmp_path), "nonce": nonce}), encoding="utf-8")
    if blocked_control_paths:
        for name in ("ready.tmp", "ready.json", "crash.tmp", "crash.json"):
            (tmp_path / name).mkdir()
    script = Path(__file__).parent / "fixtures" / "file_owner_native_holder.py"
    child = subprocess.Popen(
        [sys.executable, str(script), str(spec)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    store = FileSessionStore(tmp_path / "store")
    canonical = store._path_for("native-key").with_suffix(".json.lock")
    probes = 0
    native_probe = pid_is_dead

    def counted(pid: int) -> bool:
        nonlocal probes
        probes += 1
        return native_probe(pid)

    monkeypatch.setattr(locks, "pid_is_dead", counted)
    stopped = False

    def stop() -> None:
        nonlocal stopped
        if not stopped and child.poll() is None:
            assert child.stdin is not None
            child.stdin.write(json.dumps({"nonce": nonce, "action": "crash"}) + "\n")
            child.stdin.flush()
            stopped = True

    readiness: Queue[str | OSError | UnicodeError] = Queue(maxsize=1)

    def read_ready() -> None:
        assert child.stdout is not None
        try:
            readiness.put(child.stdout.readline())
        except (OSError, UnicodeError) as error:
            readiness.put(error)

    Thread(target=read_ready, daemon=True).start()
    try:
        try:
            raw = readiness.get(timeout=20)
        except Empty as error:
            raise RuntimeError("Owned native holder setup timed out") from error
        if isinstance(raw, (OSError, UnicodeError)):
            raise raw
        ready: dict[str, Any] = json.loads(raw)
        assert ready["nonce"] == nonce and ready["identity"] == local
        owner_path = next(canonical.iterdir())
        old_bytes = owner_path.read_bytes()
        old = json.loads(old_bytes)
        assert old["pid"] == ready["pid"] and old["released"] is False
        time.sleep(1.1)
        assert old["expires_at"] <= int(time.time())
        probes = 0
        callbacks: list[bool] = []
        with pytest.raises(HarnessError):
            store.with_lock("native-key", lambda: callbacks.append(True), wait_seconds=0)
        assert callbacks == [] and probes == 1 and child.poll() is None
        assert owner_path.read_bytes() == old_bytes
        live = {"result": "held", "probes": probes}
        stop()
        _, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr
        assert owner_path.read_bytes() == old_bytes
        probes = 0
        replacement_nonce: list[str] = []

        def recovered() -> None:
            callbacks.append(True)
            replacement = json.loads(next(canonical.iterdir()).read_bytes())
            assert replacement["pid"] == os.getpid() and replacement["identity"] == local
            assert replacement["nonce"] != old["nonce"]
            replacement_nonce.append(replacement["nonce"])

        store.with_lock("native-key", recovered, wait_seconds=0.5)
        assert callbacks == [True] and probes == 1 and not canonical.exists()
        dead = {
            "result": "nonce-bound-reclaimed",
            "probes": probes,
            "replacement_nonce": replacement_nonce[0],
        }
        held = []
        for row in ("foreign-realm", "foreign-realm-released", "both-unknown"):
            owner_identity = local.copy()
            owner_identity["pid_namespace"] = (
                None if row == "both-unknown" else str(local["pid_namespace"]) + ":foreign"
            )
            fixture_nonce = secrets.token_hex(16)
            name = "owner-" + fixture_nonce + ".json"
            canonical.mkdir()
            data = (
                json.dumps(
                    {
                        "version": 2,
                        "pid": os.getpid(),
                        "identity": owner_identity,
                        "nonce": fixture_nonce,
                        "released": row != "foreign-realm",
                        "expires_at": 0,
                    }
                )
                + "\n"
            ).encode()
            (canonical / name).write_bytes(data)
            probes = 0
            with monkeypatch.context() as fixture:
                if row == "both-unknown":
                    unknown = local.copy()
                    unknown["pid_namespace"] = None
                    fixture.setattr(locks, "native_identity", unknown.copy)
                with pytest.raises(HarnessError):
                    store.with_lock(
                        "native-key", lambda: pytest.fail("foreign callback"), wait_seconds=0
                    )
            assert probes == 0 and (canonical / name).read_bytes() == data
            held.append(
                {"id": row, "result": "held-unknown", "probes": 0, "directory_unchanged": True}
            )
            (canonical / name).unlink()
            canonical.rmdir()
        proof = {
            "backend": "py",
            "production": True,
            "identity": local,
            "holder_identity": ready["identity"],
            "holder_pid": ready["pid"],
            "prober_pid": os.getpid(),
            "live": live,
            "dead": dead,
            "held": held,
            "container_path": "unmeasured; no container created",
        }
        report = os.environ.get("PRISM_FILE_OWNER_NATIVE_REPORT")
        if report:
            with Path(report).open("x", encoding="utf-8") as stream:
                json.dump(proof, stream)
        print(json.dumps(proof))
    finally:
        stop()
        child.communicate(timeout=30)


@pytest.mark.parametrize("command", ["valid", "eof", "malformed", "foreign", "wrong-action"])
def test_native_holder_pipe_commands_ignore_denied_control_reads(
    command: str, tmp_path: Path
) -> None:
    nonce = secrets.token_hex(16)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"root": str(tmp_path), "nonce": nonce}), encoding="utf-8")
    script = Path(__file__).parent / "fixtures" / "file_owner_native_holder.py"
    # The published failure was a read denied by the OS. Inject that exact fault
    # only on old control paths; real owner-record reads/writes remain native.
    wrapper = (
        "import runpy, sys; from pathlib import Path; from unittest.mock import patch; "
        "original = Path.read_text; "
        "blocked = {'ready.json', 'crash.json'}; "
        "\ndef read(self, *args, **kwargs):\n"
        " if self.name in blocked: raise PermissionError(13, 'Denied control read', str(self))\n"
        " return original(self, *args, **kwargs)\n"
        "with patch.object(Path, 'read_text', read):\n"
        " sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    wires = {
        "valid": json.dumps({"nonce": nonce, "action": "crash"}) + "\n",
        "eof": "",
        "malformed": "{}\n",
        "foreign": json.dumps({"nonce": "wrong", "action": "crash"}) + "\n",
        "wrong-action": json.dumps({"nonce": nonce, "action": "release"}) + "\n",
    }
    child = subprocess.Popen(
        [sys.executable, "-c", wrapper, str(script), str(spec)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    stdout, stderr = child.communicate(input=wires[command], timeout=10)
    ready = json.loads(stdout)
    assert ready["nonce"] == nonce and ready["identity"] == native_identity()
    canonical = (
        FileSessionStore(tmp_path / "store")._path_for("native-key").with_suffix(".json.lock")
    )
    if command == "valid":
        assert child.returncode == 0, stderr
        owner = json.loads(next(canonical.iterdir()).read_bytes())
        assert owner["pid"] == ready["pid"] and owner["released"] is False
    else:
        assert child.returncode != 0, "Invalid input must not execute the deliberate crash"
        assert "JSONDecodeError" in stderr or "AssertionError" in stderr
        assert not canonical.exists(), "Rejected control input must unwind the holder"
