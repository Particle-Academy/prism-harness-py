"""Owned worker deliberately exits without release after an authenticated barrier."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from prism_harness import FileSessionStore
from prism_harness.stores._file_lock_identity import native_identity

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(spec["root"])
identity = native_identity()
assert identity is not None and identity["pid_namespace"] is not None
store = FileSessionStore(root / "store")


def hold() -> None:
    (root / "ready.tmp").write_text(
        json.dumps({"nonce": spec["nonce"], "pid": os.getpid(), "identity": identity}),
        encoding="utf-8",
    )
    os.replace(root / "ready.tmp", root / "ready.json")
    deadline = time.monotonic() + 25
    while True:
        try:
            message = json.loads((root / "crash.json").read_text(encoding="utf-8"))
            assert message["nonce"] == spec["nonce"]
            os._exit(0)
        except FileNotFoundError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("Owned holder barrier timed out")
        time.sleep(0.01)


store.with_lock("native-key", hold, ttl_seconds=1)
