"""Metadata fault injection only; real production store/lock runs in this process."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
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

    # Commands travel over the already-owned pipe. A filesystem sharing denial
    # on an unrelated command path must not make the holder abandon its lock.
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
        raise RuntimeError("Owned unknown holder barrier timed out") from error
    if isinstance(raw, (OSError, UnicodeError)):
        raise raw
    command = json.loads(raw)
    assert command == {"nonce": spec["nonce"], "action": spec["action"]}
    if command["action"] == "crash":
        os._exit(0)


with (
    patch.object(ownership, "native_identity", return_value=None),
    patch.object(ownership, "pid_is_dead", side_effect=forbidden_probe),
):
    FileSessionStore(root / "store").with_lock("unknown-key", hold, ttl_seconds=1)
