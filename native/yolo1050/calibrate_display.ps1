param([string]$Output = (Join-Path $PSScriptRoot 'mouse_calibration.json'))
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Threading;
public static class YOLO1050Calibration {
    [StructLayout(LayoutKind.Sequential)] public struct Point { public int X; public int Y; }
    [DllImport("user32.dll")] private static extern IntPtr SetThreadDpiAwarenessContext(IntPtr value);
    [DllImport("user32.dll")] private static extern int GetSystemMetrics(int index);
    [DllImport("user32.dll")] private static extern short GetAsyncKeyState(int key);
    [DllImport("user32.dll")] private static extern bool GetCursorPos(out Point point);
    private static bool Down() { return (GetAsyncKeyState(0xBB) & 0x8000) != 0; }
    public static int[] Capture() {
        IntPtr previous = SetThreadDpiAwarenessContext(new IntPtr(-4));
        try {
            int width = GetSystemMetrics(0), height = GetSystemMetrics(1);
            if (width != 1920 || height != 1080)
                throw new InvalidOperationException("This package expects the primary display to be 1920x1080.");
            bool wasDown = Down();
            for (;;) {
                bool down = Down();
                if (down && !wasDown) {
                    Point point;
                    if (!GetCursorPos(out point) || point.X < 0 || point.X >= width || point.Y < 0 || point.Y >= height)
                        throw new InvalidOperationException("The locked cursor is outside the primary display.");
                    return new int[] { point.X, point.Y, width, height };
                }
                wasDown = down;
                Thread.Sleep(10);
            }
        } finally {
            if (previous != IntPtr.Zero) SetThreadDpiAwarenessContext(previous);
        }
    }
}
'@
Write-Host 'Join Rivals, enter the practice area, lock the mouse to the center, keep it still, then press =.'
Write-Host 'Close this window to cancel. The detector must be closed during calibration.'
$taskPoint = [YOLO1050Calibration]::Capture()
$taskCalibration = [ordered]@{
    schema_version = 1
    locked_x = $taskPoint[0]
    locked_y = $taskPoint[1]
    screen_width = $taskPoint[2]
    screen_height = $taskPoint[3]
    calibrated_at_utc = [DateTime]::UtcNow.ToString('o')
}
$taskPath = [IO.Path]::GetFullPath($Output)
$taskTemporary = "$taskPath.tmp"
[IO.File]::WriteAllText($taskTemporary, ($taskCalibration | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $taskTemporary -Destination $taskPath -Force
Write-Host "Saved display calibration: $taskPath"
