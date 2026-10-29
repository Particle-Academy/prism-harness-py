"""Native production owner protocol, also executed under owned Windows Jobs."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from prism_harness import FileSessionStore, HarnessError
from prism_harness.stores import _file_owner_lock as locks
from prism_harness.stores._file_lock_identity import native_identity, pid_is_dead


def test_native_independent_live_dead_and_foreign_unknown_production_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = native_identity()
    assert local is not None and local["pid_namespace"] is not None
    nonce = secrets.token_hex(16)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"root": str(tmp_path), "nonce": nonce}), encoding="utf-8")
    script = Path(__file__).parent / "fixtures" / "file_owner_native_holder.py"
    child = subprocess.Popen(
        [sys.executable, str(script), str(spec)],
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
        if not stopped:
            (tmp_path / "crash.tmp").write_text(json.dumps({"nonce": nonce}), encoding="utf-8")
            os.replace(tmp_path / "crash.tmp", tmp_path / "crash.json")
            stopped = True

    try:
        deadline = time.monotonic() + 20
        ready: dict[str, Any] | None = None
        while ready is None:
            try:
                ready = json.loads((tmp_path / "ready.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Owned native holder setup timed out")
            if ready is None:
                time.sleep(0.01)
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
