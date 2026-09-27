"""Click-through Win32 overlay for live detection boxes on the primary display."""

import ctypes
import colorsys
import threading

import win32api
import win32con
import win32gui


OVERLAY_FPS = 60
WDA_EXCLUDEFROMCAPTURE = 0x11

Detection = tuple[float, float, float, float, float, int, str]
DrawDetection = tuple[int, int, int, int, int, int, str]
MAX_DIRTY_BOXES = 48


def class_color(class_id: int) -> int:
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return win32api.RGB(round(red * 255), round(green * 255), round(blue * 255))

_text_out = ctypes.WinDLL("gdi32").TextOutW
_text_out.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                      ctypes.c_wchar_p, ctypes.c_int]
_text_out.restype = ctypes.c_int


class DetectionOverlay:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.detections: tuple[DrawDetection, ...] = ()
        self.rendered_detections: tuple[DrawDetection, ...] = ()
        self.pens: dict[int, int] = {}
        self.colors: dict[int, int] = {}
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
            self.detections = visible
            self.dirty = True

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
        try:
            class_name = f"YoloDetectionOverlay_{id(self)}"
            instance = win32api.GetModuleHandle(None)
            window_class = win32gui.WNDCLASS()
            window_class.hInstance = instance
            window_class.lpszClassName = class_name
            window_class.lpfnWndProc = self._window_proc
            window_class.hbrBackground = win32gui.GetStockObject(win32con.BLACK_BRUSH)
            win32gui.RegisterClass(window_class)
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
            user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
            user32.SetTimer.restype = ctypes.c_size_t
            if not user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
                print("Warning: Windows could not exclude the overlay from screen capture; "
                      "drawn boxes may appear in inference frames.")
            with self.lock:
                self.hwnd = hwnd
            win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
            if not user32.SetTimer(hwnd, 1, round(1000 / OVERLAY_FPS), None):
                raise OSError(ctypes.get_last_error(), "Could not start overlay redraw timer")
            self.ready.set()
            win32gui.PumpMessages()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            for pen in self.pens.values():
                win32gui.DeleteObject(pen)
            with self.lock:
                self.hwnd = None

    def _window_proc(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
        if message == win32con.WM_TIMER:
            with self.lock:
                if not self.dirty:
                    return 0
                current = self.detections
                self.dirty = False
            self._invalidate_changes(hwnd, current)
            return 0
        if message == win32con.WM_PAINT:
            hdc, paint = win32gui.BeginPaint(hwnd)
            try:
                win32gui.FillRect(hdc, (0, 0, self.width, self.height),
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
