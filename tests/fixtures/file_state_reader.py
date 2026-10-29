"""Hold a synthetic state target open until the parent closes stdin."""

import sys
from pathlib import Path

from prism_harness.stores._file_lock_read import shared_owner_reader

path = Path(sys.argv[1])
if sys.platform == "win32" and sys.argv[2] == "share-delete":
    with shared_owner_reader(path):
        print("ready", flush=True)
        sys.stdin.readline()
else:
    with path.open("rb") as reader:
        reader.read()
        print("ready", flush=True)
        sys.stdin.readline()
