"""Click-through Win32 overlay for live detection boxes on the primary display."""

import ctypes
import colorsys
import threading
import time

import win32api
import win32con
import win32gui


OVERLAY_FPS = 60
WDA_EXCLUDEFROMCAPTURE = 0x11

Detection = tuple[float, float, float, float, float, int, str]
DrawDetection = tuple[int, int, int, int, int, int, str]
MAX_DIRTY_BOXES = 48
WM_FRAME_AVAILABLE = win32con.WM_APP + 1


def class_color(class_id: int) -> int:
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return win32api.RGB(round(red * 255), round(green * 255), round(blue * 255))

_text_out = ctypes.WinDLL("gdi32").TextOutW
_text_out.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                      ctypes.c_wchar_p, ctypes.c_int]
_text_out.restype = ctypes.c_int

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateWaitableTimerExW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint]
_kernel32.CreateWaitableTimerExW.restype = ctypes.c_void_p
_kernel32.SetWaitableTimer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong),
                                     ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_kernel32.SetWaitableTimer.restype = ctypes.c_int
_kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
_kernel32.CloseHandle.restype = ctypes.c_int
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.MsgWaitForMultipleObjectsEx.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
_user32.MsgWaitForMultipleObjectsEx.restype = ctypes.c_uint


class DetectionOverlay:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.detections: tuple[DrawDetection, ...] = ()
        self.rendered_detections: tuple[DrawDetection, ...] = ()
        self.pens: dict[int, int] = {}
        self.colors: dict[int, int] = {}
        self.back_buffer_dc = None
        self.back_buffer_bitmap = None
        self.back_buffer_old_bitmap = None
        self.hwnd: int | None = None
        self.error: BaseException | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="detection-overlay", daemon=True)
        self.dirty = False

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise RuntimeError("Detection overlay did not start")
        if self.error is not None:
            raise RuntimeError(f"Detection overlay could not start: {self.error}") from self.error

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def update(self, detections: tuple[Detection, ...]) -> None:
        visible = tuple((round(left), round(top), round(right), round(bottom),
                         round(confidence * 100), class_id, class_name)
                        for left, top, right, bottom, confidence, class_id, class_name in detections)
        with self.lock:
            if self.detections == visible:
                return
            notify = not self.dirty
            self.detections = visible
            self.dirty = True
            hwnd = self.hwnd
        if notify and hwnd is not None:
            try:
                win32gui.PostMessage(hwnd, WM_FRAME_AVAILABLE, 0, 0)
            except win32gui.error:
                if win32gui.IsWindow(hwnd):
                    raise

    def update_rows(self, rows, names: dict[int, str], *, source_size=None,
                    confidence: float = 0.0) -> None:
        """Draw XYXY prediction rows, scaling source-image coordinates to the display."""
        source_width, source_height = source_size or (self.width, self.height)
        scale_x, scale_y = self.width / source_width, self.height / source_height
        self.update(tuple(
            (float(x1) * scale_x, float(y1) * scale_y,
             float(x2) * scale_x, float(y2) * scale_y, float(score),
             int(class_id), names.get(int(class_id), f"class {int(class_id)}"))
            for x1, y1, x2, y2, score, class_id in rows if score >= confidence
        ))

    def _bounds(self, detection: DrawDetection) -> tuple[int, int, int, int]:
        x1, y1, x2, y2, confidence, _, class_name = detection
        label_top = max(0, y1 - 18)
        label_width = 12 * len(f"{class_name} {confidence}%")
        return (max(0, min(x1, x2) - 3),
                max(0, min(y1, y2, label_top) - 3),
                min(self.width, max(x1, x2, x1 + 2 + label_width) + 3),
                min(self.height, max(y1, y2, label_top + 18) + 3))

    def _invalidate_changes(self, hwnd: int, current: tuple[DrawDetection, ...]) -> None:
        changed = set(self.rendered_detections) ^ set(current)
        self.rendered_detections = current
        if not changed:
            return
        if len(changed) > MAX_DIRTY_BOXES:
            win32gui.InvalidateRect(hwnd, None, False)
            return
        bounds = [self._bounds(detection) for detection in changed]
        if sum((right - left) * (bottom - top) for left, top, right, bottom in bounds) > self.width * self.height // 2:
            win32gui.InvalidateRect(hwnd, None, False)
            return
        for left, top, right, bottom in bounds:
            if right > left and bottom > top:
                win32gui.InvalidateRect(hwnd, (left, top, right, bottom), False)

    def close(self) -> None:
        with self.lock:
            hwnd = self.hwnd
        if hwnd is not None:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        if self.thread.ident is not None:
            self.thread.join(timeout=2)

    def _run(self) -> None:
        hwnd = instance = None
        registered = False
        class_name = f"YoloDetectionOverlay_{id(self)}"
        try:
            instance = win32api.GetModuleHandle(None)
            window_class = win32gui.WNDCLASS()
            window_class.hInstance = instance
            window_class.lpszClassName = class_name
            window_class.lpfnWndProc = self._window_proc
            window_class.hbrBackground = 0
            win32gui.RegisterClass(window_class)
            registered = True
            styles = (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT |
                      win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW |
                      win32con.WS_EX_NOACTIVATE)
            hwnd = win32gui.CreateWindowEx(
                styles, class_name, "YOLO detections", win32con.WS_POPUP,
                0, 0, self.width, self.height, 0, 0, instance, None,
            )
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_COLORKEY)
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            user32.SetWindowDisplayAffinity.restype = ctypes.c_int
            if not user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
                print("Warning: Windows could not exclude the overlay from screen capture; "
                      "drawn boxes may appear in inference frames.")
            with self.lock:
                self.hwnd = hwnd
            win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
            self._pump_frames(hwnd)
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            if hwnd is not None and win32gui.IsWindow(hwnd):
                win32gui.DestroyWindow(hwnd)
            if self.back_buffer_dc is not None:
                if self.back_buffer_old_bitmap is not None:
                    win32gui.SelectObject(self.back_buffer_dc, self.back_buffer_old_bitmap)
                if self.back_buffer_bitmap is not None:
                    win32gui.DeleteObject(self.back_buffer_bitmap)
                win32gui.DeleteDC(self.back_buffer_dc)
                self.back_buffer_dc = self.back_buffer_bitmap = self.back_buffer_old_bitmap = None
            for pen in self.pens.values():
                win32gui.DeleteObject(pen)
            self.pens.clear()
            if registered:
                win32gui.UnregisterClass(class_name, instance)
            with self.lock:
                self.hwnd = None

    def _pump_frames(self, hwnd: int) -> None:
        # WM_TIMER is low priority and its 17 ms interval can become ~31 ms.
        # Wake on new data, then rate-limit with high-resolution 60 Hz deadlines.
        # A late producer must not wait an entire extra frame for a periodic tick.
        timer = _kernel32.CreateWaitableTimerExW(None, None, 0x2, 0x1F0003)
        if not timer:
            raise ctypes.WinError(ctypes.get_last_error())
        handles = (ctypes.c_void_p * 1)(timer)
        period = round(1_000_000_000 / OVERLAY_FPS)
        deadline = time.perf_counter_ns()
        try:
            self.ready.set()
            while not win32gui.PumpWaitingMessages():
                now = time.perf_counter_ns()
                if now >= deadline:
                    with self.lock:
                        current = self.detections if self.dirty else None
                        self.dirty = False
                    if current is not None:
                        self._invalidate_changes(hwnd, current)
                        # Paint now instead of queuing another low-priority message.
                        win32gui.UpdateWindow(hwnd)
                        deadline = deadline + period if now < deadline + period else now + period
                with self.lock:
                    pending = self.dirty
                if pending:
                    due = ctypes.c_longlong(-max(1, (deadline - time.perf_counter_ns() + 99) // 100))
                    if not _kernel32.SetWaitableTimer(timer, ctypes.byref(due), 0, None, None, False):
                        raise ctypes.WinError(ctypes.get_last_error())
                result = _user32.MsgWaitForMultipleObjectsEx(
                    1 if pending else 0, handles if pending else None, 0xFFFFFFFF, 0x04FF, 0x0004
                )
                if result == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _kernel32.CloseHandle(timer)

    def _window_proc(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
        if message == WM_FRAME_AVAILABLE:
            return 0
        if message == win32con.WM_ERASEBKGND:
            return 1  # The back buffer owns clearing; never erase the visible surface.
        if message == win32con.WM_PAINT:
            target_dc, paint = win32gui.BeginPaint(hwnd)
            try:
                if self.back_buffer_dc is None:
                    self.back_buffer_dc = win32gui.CreateCompatibleDC(target_dc)
                    self.back_buffer_bitmap = win32gui.CreateCompatibleBitmap(target_dc, self.width, self.height)
                    self.back_buffer_old_bitmap = win32gui.SelectObject(self.back_buffer_dc, self.back_buffer_bitmap)
                hdc = self.back_buffer_dc
                win32gui.FillRect(hdc, paint[2],
                                  win32gui.GetStockObject(win32con.BLACK_BRUSH))
                old_pen = win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_PEN))
                old_brush = win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                try:
                    win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
                    for x1, y1, x2, y2, confidence, class_id, class_name in self.rendered_detections:
                        pen = self.pens.get(class_id)
                        if pen is None:
                            color = class_color(class_id)
                            pen = win32gui.CreatePen(win32con.PS_SOLID, 2, color)
                            self.pens[class_id] = pen
                            self.colors[class_id] = color
                        win32gui.SelectObject(hdc, pen)
                        win32gui.SetTextColor(hdc, self.colors[class_id])
                        win32gui.Rectangle(hdc, x1, y1, x2, y2)
                        label = f"{class_name} {confidence}%"
                        _text_out(hdc, x1 + 2, max(0, y1 - 18), label, len(label))
                finally:
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.SelectObject(hdc, old_brush)
                left, top, right, bottom = paint[2]
                # Publish the cleared background and complete boxes together.
                if right > left and bottom > top:
                    win32gui.BitBlt(target_dc, left, top, right - left, bottom - top,
                                   hdc, left, top, win32con.SRCCOPY)
            finally:
                win32gui.EndPaint(hwnd, paint)
            return 0
        if message == win32con.WM_CLOSE:
            win32gui.DestroyWindow(hwnd)
            return 0
        if message == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, message, wparam, lparam)
