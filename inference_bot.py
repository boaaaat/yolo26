"""BF16, compiled YOLO inference with smooth enemy aiming on the primary display.

Press = to arm, - to pause, and Ctrl+C to exit. Settings are below.
"""

import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import dxcam
import numpy as np
import torch
import win32api
import win32con
from ultralytics import YOLO

from calibrate import CALIBRATION_PATH, calibration_instructions, make_dpi_aware


CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26m" / "weights" / "best.pt"
GPU_INDEX = 0
DXCAM_DEVICE_INDEX = 0
DXCAM_OUTPUT_INDEX = 0  # Primary display; coordinates below are primary-display coordinates.
CAPTURE_SIZE = 960  # Centered square crop in screen pixels.
IMAGE_SIZE = 640  # Fixed model input; smaller is faster but may miss small targets.
INFERENCE_TARGET_FPS = 30
CONFIDENCE = 0.50
ENEMY_CLASS_NAME = "enemy"
COMPILE_MODE = "reduce-overhead"
WARMUP_PASSES = 3
COMPILE_CACHE_DIR = Path(__file__).resolve().parent / ".inference_compile_cache"

AUTO_SHOOT = True
SHOOT_INTERVAL_SECONDS = 0.10
SHOOT_HOLD_SECONDS = 0.025
AIM_HEIGHT_FROM_BOTTOM = 0.90  # 90% up the box, or 10% down from its top.
AIM_TIME_CONSTANT_SECONDS = 0.030
MOUSE_UPDATE_HZ = 180
MAX_MOUSE_STEP_PIXELS = 70

START_KEY = 0xBB  # = / +
STOP_KEY = 0xBD  # - / _
HOTKEY_POLL_SECONDS = 0.003
SPIN_GUARD_NS = 600_000  # Sleep most of the wait, then use the clock for the last 0.6 ms.


def key_down(virtual_key: int) -> bool:
    return bool(win32api.GetAsyncKeyState(virtual_key) & 0x8000)


@contextmanager
def high_resolution_timer():
    """Request a 1 ms Windows timer resolution while the bot is running."""
    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod.argtypes = [ctypes.c_uint]
    winmm.timeBeginPeriod.restype = ctypes.c_uint
    winmm.timeEndPeriod.argtypes = [ctypes.c_uint]
    winmm.timeEndPeriod.restype = ctypes.c_uint
    requested = winmm.timeBeginPeriod(1) == 0
    if not requested:
        print("Windows did not grant a 1 ms timer resolution; pacing will use the performance clock.")
    try:
        yield
    finally:
        if requested:
            winmm.timeEndPeriod(1)


def wait_until(deadline_ns: int, running: threading.Event) -> bool:
    """Use sleep for the coarse wait and perf_counter_ns for the short tail."""
    while running.is_set():
        remaining = deadline_ns - time.perf_counter_ns()
        if remaining <= 0:
            return True
        if remaining > SPIN_GUARD_NS:
            time.sleep((remaining - SPIN_GUARD_NS) / 1_000_000_000)
    return False


class AimState:
    def __init__(self, locked_center: tuple[int, int]) -> None:
        self.running = threading.Event()
        self.shutdown = threading.Event()
        self.lock = threading.Lock()
        self.locked_center = locked_center
        self.aim_delta: tuple[float, float] | None = None
        self.generation = 0
        self.boxes: tuple[tuple[float, float, float, float], ...] = ()

    def set_boxes(self, boxes: tuple[tuple[float, float, float, float], ...]) -> None:
        if boxes:
            center_x, center_y = self.locked_center
            chosen = min(boxes, key=lambda box: (
                (box[0] + box[2] - 2 * center_x) ** 2
                + (box[1] + box[3] - 2 * center_y) ** 2
            ))
            aim_delta = ((chosen[0] + chosen[2]) / 2 - center_x,
                         chosen[3] - AIM_HEIGHT_FROM_BOTTOM * (chosen[3] - chosen[1]) - center_y)
        else:
            aim_delta = None
        with self.lock:
            if not self.running.is_set():
                return
            self.boxes = boxes
            self.aim_delta = aim_delta
            self.generation += 1

    def pause(self) -> None:
        with self.lock:
            self.running.clear()
            self.boxes = ()
            self.aim_delta = None
            self.generation += 1

    def clear(self) -> None:
        with self.lock:
            self.boxes = ()
            self.aim_delta = None
            self.generation += 1

    def snapshot(self):
        with self.lock:
            return self.aim_delta, self.boxes, self.generation


def watch_hotkeys(state: AimState) -> None:
    start_was_down = key_down(START_KEY)
    stop_was_down = key_down(STOP_KEY)
    while not state.shutdown.is_set():
        start_is_down = key_down(START_KEY)
        stop_is_down = key_down(STOP_KEY)
        if stop_is_down and not stop_was_down:
            state.pause()
            print("Paused. Press = to arm again.")
        elif start_is_down and not start_was_down:
            state.running.set()
            print("Armed. Press - to pause.")
        start_was_down, stop_was_down = start_is_down, stop_is_down
        state.shutdown.wait(HOTKEY_POLL_SECONDS)


def move_mouse(dx: float, dy: float) -> tuple[int, int]:
    distance = math.hypot(dx, dy)
    if distance < 0.5:
        return 0, 0
    scale = min(1.0, MAX_MOUSE_STEP_PIXELS / distance)
    step_x = round(dx * scale)
    step_y = round(dy * scale)
    if not step_x and abs(dx) >= 0.5:
        step_x = 1 if dx > 0 else -1
    if not step_y and abs(dy) >= 0.5:
        step_y = 1 if dy > 0 else -1
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, step_x, step_y, 0, 0)
    return step_x, step_y


def aim_loop(state: AimState) -> None:
    period_ns = round(1_000_000_000 / MOUSE_UPDATE_HZ)
    next_tick = time.perf_counter_ns()
    previous_tick = next_tick
    next_shot = 0
    release_at = 0
    mouse_down = False
    last_generation = -1
    remaining_x = remaining_y = 0.0
    try:
        while not state.shutdown.is_set():
            if not state.running.is_set():
                if mouse_down:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    mouse_down = False
                state.shutdown.wait(0.01)
                next_tick = previous_tick = time.perf_counter_ns()
                continue
            if not wait_until(next_tick, state.running):
                continue
            now = time.perf_counter_ns()
            dt = min((now - previous_tick) / 1_000_000_000, 0.05)
            previous_tick = now
            next_tick = max(next_tick + period_ns, now)
            if mouse_down and now >= release_at:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                mouse_down = False
            aim_delta, boxes, generation = state.snapshot()
            if generation != last_generation:
                last_generation = generation
                remaining_x, remaining_y = aim_delta if aim_delta is not None else (0.0, 0.0)
            if aim_delta is None or not state.running.is_set():
                continue
            gain = 1 - math.exp(-dt / AIM_TIME_CONSTANT_SECONDS)
            desired_x, desired_y = remaining_x * gain, remaining_y * gain
            if abs(remaining_x) >= 0.5 and abs(desired_x) < 0.5:
                desired_x = math.copysign(0.5, remaining_x)
            if abs(remaining_y) >= 0.5 and abs(desired_y) < 0.5:
                desired_y = math.copysign(0.5, remaining_y)
            moved_x, moved_y = move_mouse(desired_x, desired_y)
            remaining_x -= moved_x
            remaining_y -= moved_y
            if AUTO_SHOOT and not mouse_down and state.running.is_set() and now >= next_shot:
                actual_x, actual_y = win32api.GetCursorPos()
                if any(left <= actual_x <= right and top <= actual_y <= bottom
                       for left, top, right, bottom in boxes):
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    mouse_down = True
                    release_at = now + round(SHOOT_HOLD_SECONDS * 1_000_000_000)
                    next_shot = now + round(SHOOT_INTERVAL_SECONDS * 1_000_000_000)
    finally:
        if mouse_down:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def native_bf16_supported() -> bool:
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8


def compiled_cache_path(checkpoint: Path) -> Path:
    with checkpoint.open("rb") as source:
        checkpoint_hash = hashlib.file_digest(source, "sha256").hexdigest()
    try:
        triton_version = importlib.metadata.version("triton-windows")
    except importlib.metadata.PackageNotFoundError:
        triton_version = importlib.metadata.version("triton")
    identity = {
        "checkpoint_sha256": checkpoint_hash,
        "image_size": IMAGE_SIZE,
        "compile_mode": COMPILE_MODE,
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "triton": triton_version,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(GPU_INDEX),
        "gpu_capability": torch.cuda.get_device_capability(GPU_INDEX),
        "python": sys.version_info[:3],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return COMPILE_CACHE_DIR / f"model-{key}.ptcache"


def load_compiled_cache(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if torch.compiler.load_cache_artifacts(path.read_bytes()) is None:
            raise ValueError("cache contains no compiler artifacts")
    except Exception as exc:
        print(f"Could not load compiled cache ({exc}); rebuilding it.")
        return False
    print(f"Loaded compiled artifacts from {path}")
    return True


def save_compiled_cache(path: Path) -> None:
    temporary = None
    try:
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            print("PyTorch did not return compiler artifacts to save.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".compile-", suffix=".tmp",
                                         dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(artifacts[0])
        os.replace(temporary, path)
        print(f"Saved compiled artifacts to {path}")
    except Exception as exc:
        print(f"Could not save compiled cache ({exc}); inference can still run.")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def predict(model: YOLO, frame: np.ndarray, enemy_class_id: int):
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model.predict(
            source=frame,
            device=GPU_INDEX,
            imgsz=IMAGE_SIZE,
            rect=False,
            conf=CONFIDENCE,
            classes=[enemy_class_id],
            max_det=20,
            compile=COMPILE_MODE,
            verbose=False,
        )[0]


def enemy_boxes(result, left: int, top: int) -> tuple[tuple[float, float, float, float], ...]:
    if result.boxes is None or len(result.boxes) == 0:
        return ()
    coordinates = result.boxes.xyxy.detach().cpu().tolist()
    return tuple((left + x1, top + y1, left + x2, top + y2)
                 for x1, y1, x2, y2 in coordinates)


def load_locked_center() -> tuple[int, int]:
    if not CALIBRATION_PATH.is_file():
        raise FileNotFoundError(f"Mouse calibration is missing: {CALIBRATION_PATH}. "
                                + calibration_instructions())
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Mouse calibration cannot be read: {CALIBRATION_PATH}. "
                         + calibration_instructions()) from exc
    width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Mouse calibration format is invalid. " + calibration_instructions())
    x, y = data.get("locked_x"), data.get("locked_y")
    if (type(x) is not int or type(y) is not int or
            not 0 <= x < width or not 0 <= y < height or
            data.get("screen_width") != width or data.get("screen_height") != height):
        raise ValueError("Mouse calibration does not match the primary display. "
                         + calibration_instructions())
    return x, y


def main() -> None:
    if not (INFERENCE_TARGET_FPS > 0 and MOUSE_UPDATE_HZ > 0 and
            0 < AIM_TIME_CONSTANT_SECONDS and MAX_MOUSE_STEP_PIXELS > 0 and
            0 <= AIM_HEIGHT_FROM_BOTTOM <= 1 and 0 < CONFIDENCE < 1 and
            SHOOT_INTERVAL_SECONDS > SHOOT_HOLD_SECONDS > 0 and
            CAPTURE_SIZE > 0 and IMAGE_SIZE > 0 and WARMUP_PASSES > 0):
        raise ValueError("FPS, aim, confidence, or shooting settings are invalid")
    make_dpi_aware()
    locked_center = load_locked_center()
    checkpoint = CHECKPOINT_PATH.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for compiled BF16 inference")
    torch.cuda.set_device(GPU_INDEX)
    if not native_bf16_supported():
        raise RuntimeError("This GPU or PyTorch build does not support native CUDA BF16")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError(
            'Native Windows torch.compile needs Triton. In the yolo environment, run '
            'python -m pip install "triton-windows>=3.8,<3.9" for PyTorch 2.14.'
        )
    torch.backends.cudnn.benchmark = True  # Fixed image shape lets cuDNN choose fast kernels.
    torch.set_float32_matmul_precision("high")

    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError(f"Expected a detection checkpoint, got {model.task!r}")
    names = dict(model.names.items()) if isinstance(model.names, dict) else dict(enumerate(model.names))
    enemy_ids = [class_id for class_id, name in names.items() if name.casefold() == ENEMY_CLASS_NAME.casefold()]
    if len(enemy_ids) != 1:
        raise ValueError(f"Expected one {ENEMY_CLASS_NAME!r} class in checkpoint: {names}")
    cache_path = compiled_cache_path(checkpoint)
    cache_loaded = load_compiled_cache(cache_path)

    camera = dxcam.create(device_idx=DXCAM_DEVICE_INDEX, output_idx=DXCAM_OUTPUT_INDEX,
                          output_color="BGR")
    try:
        full_frame = camera.grab(new_frame_only=False)
        if full_frame is None:
            raise RuntimeError("DXcam did not provide a screen frame")
        screen_height, screen_width = full_frame.shape[:2]
        primary_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        primary_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        if (screen_width, screen_height) != (primary_width, primary_height):
            raise ValueError("Selected DXcam output is not the primary display; adjust the output index")
        side = min(CAPTURE_SIZE, screen_width, screen_height)
        left = (screen_width - side) // 2
        top = (screen_height - side) // 2
        region = (left, top, left + side, top + side)
        black_frame = np.zeros((side, side, 3), dtype=np.uint8)
        action = "Using cached compiler artifacts and warming" if cache_loaded else "Compiling and warming"
        print(f"{action} {checkpoint.name} on {torch.cuda.get_device_name(GPU_INDEX)}...")
        for _ in range(WARMUP_PASSES):
            predict(model, black_frame, enemy_ids[0])
        torch.cuda.synchronize(GPU_INDEX)
        if getattr(model.predictor.model, "_orig_mod", None) is None:
            raise RuntimeError("PyTorch compilation was unavailable; Ultralytics fell back to eager inference")
        if not cache_loaded:
            save_compiled_cache(cache_path)
        print(f"Ready: BF16 + compiled model, {IMAGE_SIZE}px input, {INFERENCE_TARGET_FPS} FPS target.")
        print("Press = to arm, - to pause, Ctrl+C to exit. Auto shoot:", AUTO_SHOOT)

        print(f"Calibrated locked cursor: {locked_center}")
        state = AimState(locked_center)
        hotkeys = threading.Thread(target=watch_hotkeys, args=(state,), daemon=True)
        mouse = threading.Thread(target=aim_loop, args=(state,), daemon=True)
        with high_resolution_timer():
            hotkeys.start()
            mouse.start()
            period_ns = round(1_000_000_000 / INFERENCE_TARGET_FPS)
            next_frame = time.perf_counter_ns()
            report_start = next_frame
            frames = 0
            try:
                while True:
                    if not hotkeys.is_alive() or not mouse.is_alive():
                        raise RuntimeError("A hotkey or mouse-control thread stopped unexpectedly")
                    if not state.running.is_set():
                        state.shutdown.wait(0.01)
                        next_frame = report_start = time.perf_counter_ns()
                        frames = 0
                        continue
                    if not wait_until(next_frame, state.running):
                        continue
                    frame = camera.grab(region=region, new_frame_only=False)
                    if frame is not None:
                        result = predict(model, frame, enemy_ids[0])
                        state.set_boxes(enemy_boxes(result, left, top))
                        torch.cuda.synchronize(GPU_INDEX)
                        frames += 1
                    now = time.perf_counter_ns()
                    if now - report_start >= 5_000_000_000:
                        actual_fps = frames * 1_000_000_000 / (now - report_start)
                        print(f"Actual inference: {actual_fps:.1f} FPS (target {INFERENCE_TARGET_FPS})")
                        report_start, frames = now, 0
                    next_frame = max(next_frame + period_ns, now)
            except KeyboardInterrupt:
                print("Bot stopped.")
            finally:
                state.running.clear()
                state.clear()
                state.shutdown.set()
                hotkeys.join(timeout=1)
                mouse.join(timeout=1)
    finally:
        camera.release()


if __name__ == "__main__":
    main()
