$batPath = "C:\Users\Extre\Desktop\Work\UnseenMedia\Aiva_2.0\start_aiva.bat"
# Matches the console title set by start_aiva.bat ("title AivaConsole").
# Deliberately NOT "Aiva": on Win11 PowerShell runs inside Windows Terminal,
# whose window takes the SHORTCUT'S name as its title — a shortcut named
# "Aiva" would make this script find and focus its own host window instead
# of launching. Never name the shortcut "AivaConsole".
$windowTitleMatch = "AivaConsole"
$log = "$env:TEMP\aiva_launcher.log"
"[$(Get-Date -Format 'HH:mm:ss')] launcher started (PID $PID)" | Out-File $log -Encoding utf8

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;
using System.Collections.Generic;

public class WinFinder {
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);
    [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();

    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    public static List<IntPtr> FindWindowsByTitle(string titlePart, int excludePid) {
        // Skip our own console window and anything from our own process:
        // when this script runs from a shortcut named "Aiva", its console
        // window is ALSO titled "Aiva" and it would find (and focus) itself.
        IntPtr ownConsole = GetConsoleWindow();
        List<IntPtr> results = new List<IntPtr>();
        EnumWindows(delegate(IntPtr hWnd, IntPtr lParam) {
            if (IsWindowVisible(hWnd) && hWnd != ownConsole) {
                uint pid;
                GetWindowThreadProcessId(hWnd, out pid);
                if (pid == (uint)excludePid) { return true; }
                StringBuilder sb = new StringBuilder(256);
                GetWindowText(hWnd, sb, 256);
                string title = sb.ToString();
                if (string.Equals(title, titlePart, StringComparison.OrdinalIgnoreCase)) {
                    results.Add(hWnd);
                }
            }
            return true;
        }, IntPtr.Zero);
        return results;
    }

    public static void ForceFocus(IntPtr hWnd) {
        if (IsIconic(hWnd)) { ShowWindow(hWnd, 9); }
        IntPtr fgWindow = GetForegroundWindow();
        uint dummy;
        uint fgThread = GetWindowThreadProcessId(fgWindow, out dummy);
        uint curThread = GetCurrentThreadId();
        AttachThreadInput(curThread, fgThread, true);
        ShowWindow(hWnd, 9);
        SetForegroundWindow(hWnd);
        AttachThreadInput(curThread, fgThread, false);
    }
}
"@

try {
    $found = [WinFinder]::FindWindowsByTitle($windowTitleMatch, $PID)
    "[$(Get-Date -Format 'HH:mm:ss')] search done, found: $($found.Count)" | Out-File $log -Append -Encoding utf8

    if ($found.Count -eq 0) {
        # Launch through the shell (explorer) so the console is ALWAYS visible:
        # a directly spawned console child of a hidden PowerShell inherits the
        # hidden state and Aiva boots invisibly.
        Start-Process explorer.exe -ArgumentList "`"$batPath`""
        "[$(Get-Date -Format 'HH:mm:ss')] launched bat via explorer" | Out-File $log -Append -Encoding utf8
    } else {
        [WinFinder]::ForceFocus($found[0])
        "[$(Get-Date -Format 'HH:mm:ss')] focused existing window $($found[0])" | Out-File $log -Append -Encoding utf8
    }
} catch {
    "[$(Get-Date -Format 'HH:mm:ss')] FAILED: $($_.Exception.Message)`n$($_.ScriptStackTrace)" | Out-File $log -Append -Encoding utf8
}