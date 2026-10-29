"""One fixed first-party metadata query per process, master2797 / decision0022."""

from __future__ import annotations

import ntpath
import os
import re
import subprocess
import sys

# Kept byte-identical to TS; native_identity caches every result, including unknown.
_COMMAND = r"""
$ErrorActionPreference = 'Stop'
$PSModuleAutoloadingPreference = 'None'
try {
  Import-Module -Name ($PSHOME + '\Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
  Import-Module -Name ($PSHOME + '\Modules\CimCmdlets\CimCmdlets.psd1') -ErrorAction Stop
  $boot = (CimCmdlets\Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime
  if ($null -eq $boot) { throw 'Missing boot identity' }
  $ticks = $boot.ToUniversalTime().Ticks.ToString([Globalization.CultureInfo]::InvariantCulture)
  $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
  try {
    $key = $base.OpenSubKey('SYSTEM\CurrentControlSet\Control')
    if ($null -eq $key) { throw 'Missing registry key' }
    try { $present = $key.GetValueNames() -contains 'ContainerType' }
    finally { $key.Dispose() }
  } finally { $base.Dispose() }
  [Console]::Out.WriteLine('boot=' + $ticks)
  if ($present) { [Console]::Out.WriteLine('container=present') }
  else { [Console]::Out.WriteLine('container=absent') }
} catch { exit 1 }
"""


def windows_identity() -> tuple[str, str | None] | None:
    if sys.platform != "win32":
        return None
    root = os.environ.get("SystemRoot")
    if not root or re.match(r"[a-zA-Z]:[\\/]", root) is None or not ntpath.isabs(root):
        return None
    env = {"SystemRoot": root, "windir": root}
    for key in ("TEMP", "TMP"):
        if value := os.environ.get(key):
            env[key] = value
    binary = ntpath.join(
        env["SystemRoot"], "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
    )
    try:
        child = subprocess.run(
            [binary, "-NoProfile", "-NonInteractive", "-NoLogo", "-Command", _COMMAND],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            creationflags=0x08000000,
            shell=False,
        )
        if child.stderr != "":
            return None
        match = re.fullmatch(
            r"boot=([1-9][0-9]{0,18})\r?\ncontainer=(present|absent)\r?\n", child.stdout
        )
        if match is None or int(match[1]) > 3155378975999999999:
            return None
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    return "cim-utc-ticks:" + match[1], "win32:kernel" if match[2] == "absent" else None
