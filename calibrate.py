"""Save the locked Rivals cursor position for inference_bot.py.

Join Rivals, enter the test area, lock the mouse to the center, keep it still,
then press =. Press Ctrl+C to cancel.
"""

import ctypes
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import win32api
import win32con


CALIBRATION_PATH = Path(__file__).resolve().parent / "mouse_calibration.json"
CALIBRATE_KEY = 0xBB  # = / +


def calibration_instructions() -> str:
    return ("Run python calibrate.py. Join Rivals, go to the test area, "
            "lock the mouse to the center, keep it still, then press =.")


def make_dpi_aware() -> None:
    """Keep Win32 cursor coordinates aligned with DXcam's screen pixels."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware.
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def main() -> None:
    make_dpi_aware()
    print(calibration_instructions())
    print("Waiting for =. Press Ctrl+C to cancel.")
    was_down = bool(win32api.GetAsyncKeyState(CALIBRATE_KEY) & 0x8000)
    try:
        while True:
            is_down = bool(win32api.GetAsyncKeyState(CALIBRATE_KEY) & 0x8000)
            if is_down and not was_down:
                x, y = win32api.GetCursorPos()
                width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
                height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError("The locked mouse is outside the primary display. "
                                     + calibration_instructions())
                data = {
                    "schema_version": 1,
                    "locked_x": x,
                    "locked_y": y,
                    "screen_width": width,
                    "screen_height": height,
                    "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                descriptor, temporary = tempfile.mkstemp(
                    prefix=".mouse-calibration-", suffix=".tmp", dir=CALIBRATION_PATH.parent
                )
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                        json.dump(data, output, indent=2)
                        output.write("\n")
                    os.replace(temporary, CALIBRATION_PATH)
                finally:
                    Path(temporary).unlink(missing_ok=True)
                print(f"Saved locked cursor position ({x}, {y}) to {CALIBRATION_PATH}")
                return
            was_down = is_down
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Calibration cancelled.")


if __name__ == "__main__":
    main()
