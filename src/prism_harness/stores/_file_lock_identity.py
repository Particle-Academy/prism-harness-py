"""Native immutable boot/PID realm and read-only liveness, decision 0022."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Literal, TypedDict

from prism_harness.stores._file_lock_windows_identity import windows_identity


class LockIdentity(TypedDict):
    platform: Literal["linux", "darwin", "win32"]
    boot_id: str
    pid_namespace: str | None


_established: LockIdentity | None = None
_established_pid: int | None = None
_identity_gate = threading.Lock()
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def native_identity() -> LockIdentity | None:
    """Sample once per process, retaining unknown proof until process exit."""
    global _established, _established_pid
    with _identity_gate:
        if _established_pid != os.getpid():
            _established = _read_native_identity()
            _established_pid = os.getpid()
        return None if _established is None else _established.copy()


def _reset_identity_after_fork() -> None:
    global _established, _established_pid, _identity_gate
    _established = None
    _established_pid = None
    _identity_gate = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_identity_after_fork)


def _read_native_identity() -> LockIdentity | None:
    identity: LockIdentity | None = None
    try:
        if sys.platform == "win32":
            result = windows_identity()
            if result is not None:
                identity = {"platform": "win32", "boot_id": result[0], "pid_namespace": result[1]}
        elif sys.platform == "linux":
            boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
            if _UUID.fullmatch(boot) is None:
                return None
            realm = None
            try:
                candidate = os.readlink("/proc/self/ns/pid")
                if re.fullmatch(r"pid:\[[0-9]+\]", candidate):
                    realm = candidate
            except OSError:
                pass
            identity = {"platform": "linux", "boot_id": boot.lower(), "pid_namespace": realm}
        elif sys.platform == "darwin":
            child = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            boot = child.stdout.strip()
            if _UUID.fullmatch(boot) is None:
                return None
            identity = {
                "platform": "darwin",
                "boot_id": boot.lower(),
                "pid_namespace": "darwin:kernel",
            }
    except (OSError, subprocess.SubprocessError):
        return None
    return identity


def same_identity(owner: LockIdentity, local: LockIdentity) -> bool:
    return (
        owner["pid_namespace"] is not None and local["pid_namespace"] is not None and owner == local
    )


def pid_is_dead(pid: int) -> bool:
    if sys.platform == "win32":
        return _windows_pid_is_dead(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, OverflowError, ValueError):
        return False
    return False


def _windows_pid_is_dead(pid: int) -> bool:
    # Python os.kill(pid, 0) terminates Windows processes. Query handles only.
    if sys.platform != "win32":
        return False
    try:
        api = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.GetExitCodeProcess.restype = wintypes.BOOL
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 87  # exact valid flags; no such PID
        try:
            status = wintypes.DWORD()
            if not api.GetExitCodeProcess(handle, ctypes.byref(status)):
                return False
            return status.value != 259  # STILL_ACTIVE can also be an ambiguous exit value
        finally:
            api.CloseHandle(handle)
    except (OSError, AttributeError):
        return False
