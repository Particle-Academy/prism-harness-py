"""Bind production identity to fixed, bounded first-party metadata only."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from prism_harness.stores import _file_lock_identity as dispatcher
from prism_harness.stores import _file_lock_windows_identity as boundary


def guard(source: str) -> None:
    tree = ast.parse(source)
    imports = {
        item.module if isinstance(item, ast.ImportFrom) else item.names[0].name
        for item in ast.walk(tree)
        if isinstance(item, (ast.Import, ast.ImportFrom))
    }
    assert imports == {"__future__", "ntpath", "os", "re", "subprocess", "sys"}
    for forbidden in (
        "Add-Type",
        "DllImport",
        "GetValue(",
        "IsProcessInJob",
        "QueryInformationJobObject",
        "CreateJobObject",
        "OpenJobObject",
        "NtQueryInformationProcess",
    ):
        assert forbidden not in source
    assert "$PSModuleAutoloadingPreference = 'None'" in source
    assert r"$PSHOME + '\Modules\CimCmdlets\CimCmdlets.psd1'" in source
    assert r"CimCmdlets\Get-CimInstance -ClassName Win32_OperatingSystem" in source
    assert "ToUniversalTime().Ticks.ToString" in source
    assert "RegistryView]::Registry64" in source
    assert "GetValueNames() -contains 'ContainerType'" in source


def test_fixed_metadata_boundary_rejects_compilation_value_reads_and_job_calls() -> None:
    assert boundary.__file__ is not None
    source = Path(boundary.__file__).read_text(encoding="utf-8")
    guard(source)
    for addition in (
        "\nimport ctypes",
        "\n_COMMAND += 'Add-Type'",
        "\n_COMMAND += 'GetValue('",
        "\n_COMMAND += 'IsProcessInJob'",
    ):
        with pytest.raises(AssertionError):
            guard(source + addition)


def test_dispatcher_has_only_the_declared_read_only_pid_symbols() -> None:
    assert dispatcher.__file__ is not None
    source = Path(dispatcher.__file__).read_text(encoding="utf-8")

    def check(value: str) -> None:
        tree = ast.parse(value)
        symbols = {
            item.attr
            for item in ast.walk(tree)
            if isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Name)
            and item.value.id == "api"
        }
        assert symbols == {"OpenProcess", "GetExitCodeProcess", "CloseHandle"}
        imports = {
            item.module if isinstance(item, ast.ImportFrom) else item.names[0].name
            for item in ast.walk(tree)
            if isinstance(item, (ast.Import, ast.ImportFrom))
        }
        assert imports == {
            "__future__",
            "ctypes",
            "os",
            "re",
            "subprocess",
            "sys",
            "threading",
            "time",
            "pathlib",
            "typing",
            "prism_harness.stores._file_lock_windows_identity",
        }
        assert "IsProcessInJob" not in value and "Add-Type" not in value

    check(source)
    for addition in ("\napi.IsProcessInJob(0,0,0)", "\nimport native_addon"):
        with pytest.raises(AssertionError):
            check(source + addition)


@pytest.fixture
def metadata_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("TEMP", r"C:\owned-temp")
    monkeypatch.setenv("TMP", r"C:\owned-temp")
    monkeypatch.setenv("PSModulePath", r"C:\untrusted-modules")


def test_absolute_binary_fixed_script_minimal_environment_and_timeout(
    metadata_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv, 0, "boot=638000000000000000\ncontainer=absent\n", ""
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert boundary.windows_identity() == ("cim-utc-ticks:638000000000000000", "win32:kernel")
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv == [
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-NoLogo",
        "-Command",
        boundary._COMMAND,
    ]
    assert r"C:\Windows" not in boundary._COMMAND
    assert options["env"] == {
        "SystemRoot": r"C:\Windows",
        "windir": r"C:\Windows",
        "TEMP": r"C:\owned-temp",
        "TMP": r"C:\owned-temp",
    }
    assert options["timeout"] == 15 and options["shell"] is False
    assert options["creationflags"] == 0x08000000 and options["check"] is True


@pytest.mark.parametrize(
    "stdout",
    [
        "boot=0\ncontainer=absent\n",
        "boot=3155378976000000000\ncontainer=absent\n",
        "boot=638000000000000000\ncontainer=unknown\n",
        "boot=638000000000000000\ncontainer=absent",
        "boot=638000000000000000\ncontainer=absent\nextra\n",
        "boot=638000000000000000\ncontainer=absent\n\n",
        "\nboot=638000000000000000\ncontainer=absent\n",
    ],
)
def test_strict_two_line_metadata_refuses_other_output(
    stdout: str, metadata_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout, "")
    )
    assert boundary.windows_identity() is None


def test_container_present_remains_unknown(
    metadata_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, "boot=638000000000000000\ncontainer=present\n", ""
        ),
    )
    assert boundary.windows_identity() == ("cim-utc-ticks:638000000000000000", None)


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, []),
        subprocess.TimeoutExpired([], 5),
        OSError("owned unavailable"),
        UnicodeError("owned invalid output encoding"),
    ],
)
def test_failed_metadata_query_returns_unknown(
    error: Exception, metadata_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    assert boundary.windows_identity() is None


def test_stderr_is_unknown(metadata_environment: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, "boot=638000000000000000\ncontainer=absent\n", "owned warning"
        ),
    )
    assert boundary.windows_identity() is None


def test_missing_or_relative_systemroot_never_searches_path(
    metadata_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("PATH search forbidden")

    monkeypatch.setattr(subprocess, "run", forbidden)
    for root in ("", "Windows", r"\Windows", "C:Windows", r"\\host\Windows"):
        monkeypatch.setenv("SystemRoot", root)
        assert boundary.windows_identity() is None
