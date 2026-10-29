"""Native held-reader release proof: owner stays alive throughout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO

import pytest

from prism_harness import FileSessionStore


def test_reader_closes_and_pending_release_completes_while_owner_stays_alive(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path)
    reader: BinaryIO | None = None
    callbacks: list[bool] = []
    owner_pid = os.getpid()

    def hold_reader() -> None:
        nonlocal reader
        callbacks.append(True)
        lock = store._path_for("k").with_suffix(".json.lock")
        path = next(path for path in lock.iterdir() if path.suffix == ".json")
        reader = path.open("rb")  # deliberate Windows reader WITHOUT delete sharing
        assert reader.read(1) == b"{"

    try:
        if os.name == "nt":
            with pytest.raises(PermissionError):
                store.with_lock("k", hold_reader, ttl_seconds=-1)
        else:
            store.with_lock("k", hold_reader, ttl_seconds=-1)
        assert callbacks == [True] and os.getpid() == owner_pid
    finally:
        if reader is not None:
            reader.close()
    # The owner's next lock operation must finish its failed release first.
    assert store.with_lock("other-key", lambda: "owner-alive") == "owner-alive"
    script = (
        "import json,os,sys; from prism_harness import FileSessionStore; "
        "store=FileSessionStore(sys.argv[1]); count=[]\n"
        "try:\n"
        " store.with_lock('k',lambda:count.append(1),wait_seconds=0.1)\n"
        " print(json.dumps({'callbacks':len(count),'code':None,'pid':os.getpid()}))\n"
        "except Exception as e: print(json.dumps({'callbacks':len(count),'code':e.code,'pid':os.getpid()}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    import json

    contender = json.loads(result.stdout)
    assert contender["callbacks"] == 1 and contender["code"] is None
    assert contender["pid"] != owner_pid and os.getpid() == owner_pid
