"""Owned worker deliberately exits without release after an authenticated barrier."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from prism_harness import FileSessionStore
from prism_harness.stores._file_lock_identity import native_identity

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(spec["root"])
identity = native_identity()
assert identity is not None and identity["pid_namespace"] is not None
store = FileSessionStore(root / "store")


def hold() -> None:
    print(
        json.dumps({"nonce": spec["nonce"], "pid": os.getpid(), "identity": identity}),
        flush=True,
    )
    # An unrelated filesystem sharing denial must not interrupt the real holder.
    commands: Queue[str | OSError | UnicodeError] = Queue(maxsize=1)

    def read_command() -> None:
        try:
            commands.put(sys.stdin.readline())
        except (OSError, UnicodeError) as error:
            commands.put(error)

    Thread(target=read_command, daemon=True).start()
    try:
        raw = commands.get(timeout=25)
    except Empty as error:
        raise RuntimeError("Owned holder barrier timed out") from error
    if isinstance(raw, (OSError, UnicodeError)):
        raise raw
    message = json.loads(raw)
    assert message == {"nonce": spec["nonce"], "action": "crash"}
    os._exit(0)


store.with_lock("native-key", hold, ttl_seconds=1)
