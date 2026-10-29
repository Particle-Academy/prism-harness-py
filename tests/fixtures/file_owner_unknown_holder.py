"""Metadata fault injection only; real production store/lock runs in this process."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

from prism_harness import FileSessionStore
from prism_harness.stores import _file_owner_lock as ownership

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert spec["action"] in ("release", "crash")
root = Path(spec["root"])


def forbidden_probe(pid: int) -> bool:
    raise AssertionError("Unknown identity must not probe a PID")


def hold() -> None:
    (root / "ready.tmp").write_text(
        json.dumps({"nonce": spec["nonce"], "pid": os.getpid()}), encoding="utf-8"
    )
    os.replace(root / "ready.tmp", root / "ready.json")
    deadline = time.monotonic() + 25
    while True:
        try:
            command = json.loads((root / "command.json").read_text(encoding="utf-8"))
            assert command == {"nonce": spec["nonce"], "action": spec["action"]}
            if command["action"] == "crash":
                os._exit(0)
            return
        except FileNotFoundError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("Owned unknown holder barrier timed out")
        time.sleep(0.01)


with (
    patch.object(ownership, "native_identity", return_value=None),
    patch.object(ownership, "pid_is_dead", side_effect=forbidden_probe),
):
    FileSessionStore(root / "store").with_lock("unknown-key", hold, ttl_seconds=1)
