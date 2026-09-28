"""Shared Windows inference controls. No model, TensorRT, or capture imports."""
import ctypes
import json
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
import win32api
import win32con
from calibrate import CALIBRATION_PATH, calibration_instructions

BoxCoords = tuple[float, float, float, float]


@dataclass(frozen=True)
class ControlOptions:
    auto_shoot: bool = True
    instant_mouse: bool = False
    shoot_interval_seconds: float = 0.10
    shoot_hold_seconds: float = 0.09
    aim_height_from_bottom: float = 0.90
    aim_time_constant_seconds: float = 0.030
    mouse_update_hz: int = 240
    max_mouse_step_pixels: int = 70
    target_lost_frames: int = 4
    target_match_min_iou: float = 0.10
    target_match_max_center_distance: float = 0.65
    target_match_max_area_ratio: float = 4.0
    start_key: int = 0xBB
    stop_key: int = 0xBD
    hotkey_poll_seconds: float = 0.003
    spin_guard_ns: int = 600_000

    @classmethod
    def from_settings(cls, settings, **overrides):
        values = {}
        for field in fields(cls):
            if field.name in settings:
                values[field.name] = settings[field.name]
            elif field.name.upper() in settings:
                values[field.name] = settings[field.name.upper()]
        values.update(overrides)
        return cls(**values)


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


def wait_until(deadline_ns: int, running: threading.Event, *, spin_guard_ns: int = 0) -> bool:
    while running.is_set():
        remaining = deadline_ns - time.perf_counter_ns()
        if remaining <= 0:
            return True
        if spin_guard_ns == 0:
            time.sleep(min(remaining / 1_000_000_000, 0.002))
        elif remaining > spin_guard_ns:
            time.sleep((remaining - spin_guard_ns) / 1_000_000_000)
    return False


def box_center(box: BoxCoords) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def box_iou(first: BoxCoords, second: BoxCoords) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_target(previous: BoxCoords, boxes: tuple[BoxCoords, ...], options: ControlOptions) -> BoxCoords | None:
    previous_center = box_center(previous)
    previous_width = previous[2] - previous[0]
    previous_height = previous[3] - previous[1]
    previous_area = previous_width * previous_height
    previous_diagonal = math.hypot(previous_width, previous_height)
    overlapping: list[tuple[float, float, BoxCoords]] = []
    nearby: list[tuple[float, BoxCoords]] = []
    for box in boxes:
        width, height = box[2] - box[0], box[3] - box[1]
        area_ratio = width * height / previous_area if previous_area > 0 else 0
        if not 1 / options.target_match_max_area_ratio <= area_ratio <= options.target_match_max_area_ratio:
            continue
        center = box_center(box)
        distance = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
        overlap = box_iou(previous, box)
        if overlap >= options.target_match_min_iou:
            overlapping.append((overlap, -distance, box))
        elif distance <= options.target_match_max_center_distance * max(previous_diagonal, math.hypot(width, height)):
            nearby.append((distance, box))
    if overlapping:
        return max(overlapping, key=lambda item: (item[0], item[1]))[2]
    if nearby:
        return min(nearby, key=lambda item: item[0])[1]
    return None


class AimState:
    def __init__(self, locked_center: tuple[int, int], options: ControlOptions) -> None:
        self.options = options
        self.running = threading.Event()
        self.shutdown = threading.Event()
        self.lock = threading.Lock()
        self.locked_center = locked_center
        self.aim_delta: tuple[float, float] | None = None
        self.generation = 0
        self.target_box: BoxCoords | None = None
        self.visible_target_box: BoxCoords | None = None
        self.missed_target_frames = 0

    def set_boxes(self, boxes: tuple[BoxCoords, ...]) -> None:
        with self.lock:
            if not self.running.is_set():
                return
            chosen = match_target(self.target_box, boxes, self.options) if self.target_box is not None else None
            if self.target_box is not None and chosen is None:
                self.missed_target_frames += 1
                if self.missed_target_frames >= self.options.target_lost_frames:
                    self.target_box = None
                    self.missed_target_frames = 0
            elif chosen is not None:
                self.missed_target_frames = 0
            if self.target_box is None and boxes:
                center_x, center_y = self.locked_center
                chosen = min(boxes, key=lambda box: (
                    (box_center(box)[0] - center_x) ** 2
                    + (box_center(box)[1] - center_y) ** 2
                ))
            self.visible_target_box = chosen
            if chosen is not None:
                self.target_box = chosen
                center_x, center_y = self.locked_center
                self.aim_delta = (box_center(chosen)[0] - center_x,
                                  chosen[3] - self.options.aim_height_from_bottom * (chosen[3] - chosen[1]) - center_y)
            else:
                self.aim_delta = None
            self.generation += 1

    def pause(self) -> None:
        with self.lock:
            self.running.clear()
            self.target_box = None
            self.visible_target_box = None
            self.missed_target_frames = 0
            self.aim_delta = None
            self.generation += 1

    def clear(self) -> None:
        with self.lock:
            self.target_box = None
            self.visible_target_box = None
            self.missed_target_frames = 0
            self.aim_delta = None
            self.generation += 1

    def snapshot(self):
        with self.lock:
            return self.aim_delta, self.visible_target_box, self.generation


def watch_hotkeys(state: AimState) -> None:
    start_was_down = key_down(state.options.start_key)
    stop_was_down = key_down(state.options.stop_key)
    while not state.shutdown.is_set():
        start_is_down = key_down(state.options.start_key)
        stop_is_down = key_down(state.options.stop_key)
        if stop_is_down and not stop_was_down:
            state.pause()
            print("Paused. Press = to arm again.")
        elif start_is_down and not start_was_down:
            state.running.set()
            print("Armed. Press - to pause.")
        start_was_down, stop_was_down = start_is_down, stop_is_down
        state.shutdown.wait(state.options.hotkey_poll_seconds)


def move_mouse(dx: float, dy: float, *, options: ControlOptions, instant: bool = False) -> tuple[int, int]:
    distance = math.hypot(dx, dy)
    if distance < 0.5:
        return 0, 0
    scale = 1.0 if instant else min(1.0, options.max_mouse_step_pixels / distance)
    step_x = round(dx * scale)
    step_y = round(dy * scale)
    if not step_x and abs(dx) >= 0.5:
        step_x = 1 if dx > 0 else -1
    if not step_y and abs(dy) >= 0.5:
        step_y = 1 if dy > 0 else -1
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, step_x, step_y, 0, 0)
    return step_x, step_y


def aim_loop(state: AimState) -> None:
    period_ns = round(1_000_000_000 / state.options.mouse_update_hz)
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
            if not wait_until(next_tick, state.running, spin_guard_ns=state.options.spin_guard_ns):
                continue
            now = time.perf_counter_ns()
            dt = min((now - previous_tick) / 1_000_000_000, 0.05)
            previous_tick = now
            next_tick = max(next_tick + period_ns, now)
            if mouse_down and now >= release_at:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                mouse_down = False
            aim_delta, target_box, generation = state.snapshot()
            if generation != last_generation:
                last_generation = generation
                remaining_x, remaining_y = aim_delta if aim_delta is not None else (0.0, 0.0)
            if aim_delta is None or target_box is None or not state.running.is_set():
                if mouse_down:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    mouse_down = False
                continue
            if state.options.instant_mouse:
                desired_x, desired_y = remaining_x, remaining_y
            else:
                gain = 1 - math.exp(-dt / state.options.aim_time_constant_seconds)
                desired_x, desired_y = remaining_x * gain, remaining_y * gain
            if abs(remaining_x) >= 0.5 and abs(desired_x) < 0.5:
                desired_x = math.copysign(0.5, remaining_x)
            if abs(remaining_y) >= 0.5 and abs(desired_y) < 0.5:
                desired_y = math.copysign(0.5, remaining_y)
            moved_x, moved_y = move_mouse(desired_x, desired_y, options=state.options, instant=state.options.instant_mouse)
            if state.options.instant_mouse:
                remaining_x = remaining_y = 0.0
            else:
                remaining_x -= moved_x
                remaining_y -= moved_y
            if state.options.auto_shoot and not mouse_down and state.running.is_set() and now >= next_shot:
                actual_x, actual_y = win32api.GetCursorPos()
                left, top, right, bottom = target_box
                if left <= actual_x <= right and top <= actual_y <= bottom:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    mouse_down = True
                    release_at = now + round(state.options.shoot_hold_seconds * 1_000_000_000)
                    next_shot = now + round(state.options.shoot_interval_seconds * 1_000_000_000)
    finally:
        if mouse_down:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


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
