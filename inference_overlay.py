"""Click-through Win32 overlay for live detection boxes on the primary display."""

import ctypes
import colorsys
import threading

import win32api
import win32con
import win32gui


UPDATE_MESSAGE = win32con.WM_APP + 1
WDA_EXCLUDEFROMCAPTURE = 0x11

Detection = tuple[float, float, float, float, float, int, str]


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
        self.detections: tuple[Detection, ...] = ()
        self.hwnd: int | None = None
        self.error: BaseException | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="detection-overlay", daemon=True)
        self.redraw_pending = False

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise RuntimeError("Detection overlay did not start")
        if self.error is not None:
            raise RuntimeError(f"Detection overlay could not start: {self.error}") from self.error

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def update(self, detections: tuple[Detection, ...]) -> None:
        with self.lock:
            if self.detections == detections:
                return
            self.detections = detections
            if self.hwnd is None or self.redraw_pending:
                return
            self.redraw_pending = True
            hwnd = self.hwnd
        win32gui.PostMessage(hwnd, UPDATE_MESSAGE, 0, 0)

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
            if not user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
                print("Warning: Windows could not exclude the overlay from screen capture; "
                      "drawn boxes may appear in inference frames.")
            with self.lock:
                self.hwnd = hwnd
            win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
            self.ready.set()
            win32gui.PumpMessages()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            with self.lock:
                self.hwnd = None

    def _window_proc(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
        if message == UPDATE_MESSAGE:
            with self.lock:
                self.redraw_pending = False
            win32gui.InvalidateRect(hwnd, None, False)
            return 0
        if message == win32con.WM_PAINT:
            hdc, paint = win32gui.BeginPaint(hwnd)
            try:
                win32gui.FillRect(hdc, (0, 0, self.width, self.height),
                                  win32gui.GetStockObject(win32con.BLACK_BRUSH))
                with self.lock:
                    detections = self.detections
                pens = {}
                old_pen = win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_PEN))
                old_brush = win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                try:
                    win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
                    for left, top, right, bottom, confidence, class_id, class_name in detections:
                        color = class_color(class_id)
                        if class_id not in pens:
                            pens[class_id] = win32gui.CreatePen(win32con.PS_SOLID, 2, color)
                        win32gui.SelectObject(hdc, pens[class_id])
                        win32gui.SetTextColor(hdc, color)
                        x1, y1, x2, y2 = map(round, (left, top, right, bottom))
                        win32gui.Rectangle(hdc, x1, y1, x2, y2)
                        label = f"{class_name} {confidence:.0%}"
                        _text_out(hdc, x1 + 2, max(0, y1 - 18), label, len(label))
                finally:
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.SelectObject(hdc, old_brush)
                    for pen in pens.values():
                        win32gui.DeleteObject(pen)
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
