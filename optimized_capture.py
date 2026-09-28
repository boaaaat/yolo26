"""Full-screen capture and resize into an owned, bounded pinned-memory pool."""

from dataclasses import dataclass
import math
import threading
import time

import cv2
import dxcam
import numpy as np
import pythoncom
import torch


@dataclass(frozen=True)
class FrameGeometry:
    screen_width: int
    screen_height: int
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int
    scale: float

    @classmethod
    def from_screen(cls, width: int, height: int, image_size: int):
        scale = image_size / max(width, height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        tensor_width = math.ceil(resized_width / 32) * 32
        tensor_height = math.ceil(resized_height / 32) * 32
        return cls(width, height, tensor_width, tensor_height, resized_width, resized_height,
                   (tensor_width - resized_width) // 2, (tensor_height - resized_height) // 2, scale)

    def screen_detections(self, output: np.ndarray, confidence: float) -> np.ndarray:
        rows = output[np.isfinite(output).all(axis=1) & (output[:, 4] >= confidence)].astype(np.float32)
        rows[:, [0, 2]] = np.clip((rows[:, [0, 2]] - self.left) / self.scale, 0, self.screen_width)
        rows[:, [1, 3]] = np.clip((rows[:, [1, 3]] - self.top) / self.scale, 0, self.screen_height)
        return rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]


@dataclass
class FrameSlot:
    tensor: torch.Tensor
    view: np.ndarray
    resized: np.ndarray
    sequence: int = 0
    generation: int = 0
    capture_started_ns: int = 0
    capture_ms: float = 0.0
    resize_ms: float = 0.0
    original: np.ndarray | None = None
    status: str = "free"


class LatestCapture:
    """One producer owns DXcam. Slots cannot be reused until the GPU consumer releases them."""

    def __init__(self, geometry: FrameGeometry, state, fps: int, device_index: int,
                 output_index: int | None, collector=None):
        self.geometry = geometry
        self.state = state
        self.fps = fps
        self.device_index = device_index
        self.output_index = output_index
        self.collector = collector
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.latest = None
        self.captured = 0
        self.dropped = 0
        self.no_new_frame = 0
        self.slots = []
        for _ in range(3):
            tensor = torch.empty((geometry.height, geometry.width, 3), dtype=torch.uint8, pin_memory=True)
            view = tensor.numpy()
            view.fill(114)
            resized = np.empty((geometry.resized_height, geometry.resized_width, 3), dtype=np.uint8)
            self.slots.append(FrameSlot(tensor, view, resized))
        self.thread = threading.Thread(target=self._run, name="optimized-capture", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(10):
            raise RuntimeError("Capture initialization timed out")
        self.raise_if_failed()

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError(f"Capture stopped: {self.error}") from self.error
        if self.ready.is_set() and not self.thread.is_alive() and not self.stopping.is_set():
            raise RuntimeError("Capture thread stopped unexpectedly")

    def take(self, timeout: float = 0.02):
        with self.condition:
            if self.latest is None:
                self.condition.wait(timeout)
            self.raise_if_failed()
            slot = self.latest
            if slot is not None:
                self.latest = None
                slot.status = "processing"
            return slot

    def release(self, slot: FrameSlot):
        with self.condition:
            slot.original = None
            slot.status = "free"
            self.condition.notify_all()

    def discard_pending(self):
        with self.condition:
            if self.latest is not None:
                self.latest.original = None
                self.latest.status = "free"
                self.latest = None
                self.dropped += 1

    def counters(self):
        with self.condition:
            return self.captured, self.dropped, self.no_new_frame

    def close(self):
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Capture thread did not stop")

    def _run(self):
        camera = None
        com_initialized = False
        try:
            pythoncom.CoInitialize()
            com_initialized = True
            camera = dxcam.create(device_idx=self.device_index, output_idx=self.output_index,
                                  output_color="BGR", max_buffer_len=2)
            g = self.geometry
            if (camera.width, camera.height) != (g.screen_width, g.screen_height):
                raise ValueError("DXcam output does not match the calibrated primary display")
            self.ready.set()
            period_ns = round(1_000_000_000 / self.fps)
            next_capture = 0
            while not self.stopping.is_set():
                if not self.state.running.is_set():
                    self.discard_pending()
                    self.stopping.wait(0.005)
                    next_capture = 0
                    continue
                delay = (next_capture - time.perf_counter_ns()) / 1_000_000_000
                if delay > 0 and self.stopping.wait(delay):
                    break
                with self.state.lock:
                    generation = self.state.arm_generation
                started = time.perf_counter_ns()
                # Only this thread calls grab, so its internal view remains valid until
                # the next grab. Resize/copy it completely before touching DXcam again.
                frame = camera.grab(copy=False, new_frame_only=True)
                captured_at = time.perf_counter_ns()
                next_capture = started + period_ns
                if frame is None:
                    with self.condition:
                        self.no_new_frame += 1
                    continue
                if frame.shape != (g.screen_height, g.screen_width, 3):
                    raise RuntimeError("Display geometry changed; restart and recalibrate")
                with self.condition:
                    slot = next((item for item in self.slots if item.status == "free"), None)
                    if slot is None:
                        # Never overwrite the slot currently used by an asynchronous H2D copy.
                        self.dropped += 1
                        continue
                    slot.status = "writing"
                if (g.resized_width, g.resized_height) == (g.width, g.height):
                    cv2.resize(frame, (g.width, g.height), dst=slot.view, interpolation=cv2.INTER_LINEAR)
                else:
                    cv2.resize(frame, (g.resized_width, g.resized_height), dst=slot.resized,
                               interpolation=cv2.INTER_LINEAR)
                    slot.view[g.top:g.top + g.resized_height, g.left:g.left + g.resized_width] = slot.resized
                resized_at = time.perf_counter_ns()
                slot.original = frame.copy() if self.collector is not None and self.collector.due(resized_at) else None
                slot.generation = generation
                slot.capture_started_ns = started
                slot.capture_ms = (captured_at - started) / 1_000_000
                slot.resize_ms = (resized_at - captured_at) / 1_000_000
                with self.condition:
                    self.captured += 1
                    slot.sequence = self.captured
                    if self.latest is not None:
                        self.latest.original = None
                        self.latest.status = "free"
                        self.dropped += 1
                    slot.status = "ready"
                    self.latest = slot
                    self.condition.notify_all()
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready.set()
            with self.condition:
                self.condition.notify_all()
            try:
                if camera is not None:
                    camera.release()
            finally:
                if com_initialized:
                    pythoncom.CoUninitialize()
