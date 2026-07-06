$batPath = "C:\Users\Extre\Desktop\Work\UnseenMedia\Aiva_2.0\start_aiva.bat"
$windowTitleMatch = "Aiva"

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

$found = [WinFinder]::FindWindowsByTitle($windowTitleMatch, $PID)

if ($found.Count -eq 0) {
    # Launch through the shell (explorer) so the console is ALWAYS visible:
    # when this script runs from a "-WindowStyle Hidden" shortcut, directly
    # spawned console children inherit the hidden state and Aiva boots
    # invisibly.
    Start-Process explorer.exe -ArgumentList "`"$batPath`""
} else {
    [WinFinder]::ForceFocus($found[0])
}