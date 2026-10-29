"""Close state readers before parsing; bound Windows replacement contention."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

from prism_harness.stores._file_lock_read import _windows_api


def read_state_text(path: Path) -> str:
    if sys.platform != "win32":
        return path.read_text(encoding="utf-8")
    api = _windows_api()
    chunks: list[bytes] = []
    buffer, length = ctypes.create_string_buffer(65536), wintypes.DWORD()
    # The state path retains normal file/symlink read semantics; owner-record
    # reads separately use OPEN_REPARSE_POINT to validate lock identity.
    handle = api.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    # Share deletion, but keep the handle only for the read: even a share-delete
    # target can block Windows replacement. Decode and parse after closing it.
    try:
        while True:
            if not api.ReadFile(handle, buffer, len(buffer), ctypes.byref(length), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if length.value == 0:
                break
            chunks.append(buffer.raw[: length.value])
    finally:
        if not api.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    return b"".join(chunks).decode("utf-8")


def replace_state_file(temporary: Path, target: Path) -> None:
    try:
        os.replace(temporary, target)
        return
    except PermissionError:
        if sys.platform != "win32":
            raise
        # Retry only Windows permission errors. A permanent reader/ACL failure
        # remains the original native error, never a dropped or partial write.
        deadline = time.monotonic() + 0.75
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.01, remaining))
            if time.monotonic() >= deadline:
                raise
            try:
                os.replace(temporary, target)
                return
            except PermissionError:
                continue
