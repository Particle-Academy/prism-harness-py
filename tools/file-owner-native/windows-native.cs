using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

// Review tooling only. Never creates a container, changes an existing Job,
// writes the registry, or obtains a handle to an unrelated process.
public static class PrismLockNativeReview {
  [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attributes, string name);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job, int cls, ref ExtendedLimits limits, uint size);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool CreateProcess(string exe, StringBuilder commandLine, IntPtr processAttributes, IntPtr threadAttributes, bool inheritHandles, uint flags, IntPtr environment, string cwd, ref StartupInfo startup, out ProcessInfo process);
  [DllImport("kernel32.dll", SetLastError=true)] static extern uint ResumeThread(IntPtr thread);
  [DllImport("kernel32.dll", SetLastError=true)] static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr process, out uint code);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateProcess(IntPtr process, uint code);
  [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);

  [StructLayout(LayoutKind.Sequential)] struct BasicLimits {
    public long ProcessTime, JobTime; public uint Flags;
    public UIntPtr MinWorkingSet, MaxWorkingSet; public uint ActiveProcessLimit;
    public UIntPtr Affinity; public uint PriorityClass, SchedulingClass;
  }
  [StructLayout(LayoutKind.Sequential)] struct IoCounters { public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes; }
  [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits { public BasicLimits Basic; public IoCounters Io; public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] struct StartupInfo {
    public uint Size; public string Reserved, Desktop, Title;
    public uint X, Y, XSize, YSize, XCountChars, YCountChars, FillAttribute, Flags;
    public ushort ShowWindow, ReservedSize; public IntPtr ReservedBytes, Input, Output, Error;
  }
  [StructLayout(LayoutKind.Sequential)] struct ProcessInfo { public IntPtr Process, Thread; public uint Pid, Tid; }

  static string Quote(string value) {
    var result = new StringBuilder("\""); int slashes = 0;
    foreach (char character in value) {
      if (character == '\\') { slashes++; continue; }
      result.Append('\\', character == '"' ? slashes * 2 + 1 : slashes);
      result.Append(character); slashes = 0;
    }
    return result.Append('\\', slashes * 2).Append('"').ToString();
  }
  static Exception NativeError(string operation) { return new Win32Exception(Marshal.GetLastWin32Error(), operation); }

  public static string Launch(string exe, string[] args, string cwd, int depth, bool requireNoJob, bool breakaway) {
    if (depth < 0 || depth > 2) throw new ArgumentOutOfRangeException("depth");
    if (requireNoJob && !breakaway) throw new ArgumentException("No-Job observation requires a breakaway request");
    var command = new StringBuilder(Quote(exe));
    foreach (string arg in args) command.Append(' ').Append(Quote(arg));
    var startup = new StartupInfo(); startup.Size = (uint)Marshal.SizeOf(startup);
    ProcessInfo process;
    // Suspended + breakaway + no visible window. Never modifies a parent Job.
    uint flags = 0x08000004 | (breakaway ? 0x01000000u : 0u);
    if (!CreateProcess(exe, command, IntPtr.Zero, IntPtr.Zero, false, flags, IntPtr.Zero, cwd, ref startup, out process)) throw NativeError("Create owned review child (breakaway=" + breakaway + ")");
    var jobs = new IntPtr[depth]; bool exited = false;
    try {
      bool inJob;
      if (!IsProcessInJob(process.Process, IntPtr.Zero, out inJob)) throw NativeError("Query initial child Job membership");
      bool initialNoJob = !inJob;
      if (requireNoJob && inJob) return "{\"pid\":" + process.Pid + ",\"status\":\"unmeasured\",\"reason\":\"ambient supervisor Job cannot be left on this host (CREATE_BREAKAWAY_FROM_JOB accepted, membership retained)\",\"membership_checked\":true,\"breakaway_requested\":true,\"initial_no_job\":false,\"child_ran\":false}";
      for (int index = 0; index < depth; index++) {
        jobs[index] = CreateJobObject(IntPtr.Zero, null);
        if (jobs[index] == IntPtr.Zero) throw NativeError("Create owned ordinary Job");
        var limits = new ExtendedLimits(); limits.Basic.Flags = 0x00002000; // KILL_ON_JOB_CLOSE, owned children only.
        if (!SetInformationJobObject(jobs[index], 9, ref limits, (uint)Marshal.SizeOf(limits))) throw NativeError("Bound owned Job lifetime");
        if (!AssignProcessToJobObject(jobs[index], process.Process)) throw NativeError("Assign owned child to nested Job");
      }
      for (int index = 0; index < depth; index++) {
        if (!IsProcessInJob(process.Process, jobs[index], out inJob) || !inJob) throw new Exception("Required owned Job membership not established");
      }
      if (ResumeThread(process.Thread) == 0xffffffff) throw NativeError("Resume owned child");
      uint waited = WaitForSingleObject(process.Process, 60000);
      if (waited != 0) throw new Exception("Owned child deadline/wait failure: " + waited);
      exited = true;
      uint code;
      if (!GetExitCodeProcess(process.Process, out code)) throw NativeError("Read owned child exit");
      if (code != 0) throw new Exception("Owned child failed: " + code);
      return "{\"pid\":" + process.Pid + ",\"status\":\"passed\",\"membership_checked\":true,\"breakaway_requested\":" + (breakaway ? "true" : "false") + ",\"initial_no_job\":" + (initialNoJob ? "true" : "false") + ",\"verified_owned_job_depth\":" + depth + ",\"exit\":0}";
    } finally {
      bool cleanupFailed = false;
      if (!exited) { TerminateProcess(process.Process, 2); cleanupFailed = WaitForSingleObject(process.Process, 5000) != 0; }
      CloseHandle(process.Thread); CloseHandle(process.Process);
      for (int index = depth - 1; index >= 0; index--) if (jobs[index] != IntPtr.Zero) CloseHandle(jobs[index]);
      if (cleanupFailed) throw new Exception("Owned review child cleanup did not complete");
    }
  }
}
