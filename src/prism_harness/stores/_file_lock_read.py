"""Bounded read-only owner reads; an open Windows target can still delay replacement."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from functools import cache
from pathlib import Path
from typing import Any


@cache
def _windows_api() -> Any:
    if sys.platform != "win32":
        raise OSError("Windows owner reader is unavailable on this platform")
    api = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    api.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    api.CreateFileW.restype = wintypes.HANDLE
    api.ReadFile.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    api.ReadFile.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    return api


@contextmanager
def shared_owner_reader(path: Path) -> Iterator[int]:
    """GENERIC_READ, FILE_SHARE_READ|WRITE|DELETE, OPEN_EXISTING; never writes."""
    if sys.platform != "win32":
        raise OSError("Windows owner reader is unavailable on this platform")
    api = _windows_api()
    handle = api.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x00200080, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield int(handle)
    finally:
        if not api.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def read_owner_bytes(path: Path) -> bytes:
    if sys.platform != "win32":
        with path.open("rb") as stream:
            return stream.read(4097)
    api = _windows_api()
    buffer, length = ctypes.create_string_buffer(4097), wintypes.DWORD()
    with shared_owner_reader(path) as handle:
        if not api.ReadFile(handle, buffer, 4097, ctypes.byref(length), None):
            raise ctypes.WinError(ctypes.get_last_error())
    return buffer.raw[: length.value]
