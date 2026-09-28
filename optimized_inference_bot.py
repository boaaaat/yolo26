"""Fixed rectangular TensorRT inference for RTX Blackwell. Press = to arm, - to pause.

Build the engine without starting capture or controls: python optimized_inference_bot.py --build-only
"""

import argparse
from collections import deque
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch
import win32api
import win32con

import inference_bot as controls
from calibrate import make_dpi_aware
from optimized_capture import FrameGeometry, LatestCapture
from optimized_engine import ensure_engine
from optimized_runtime import TensorRTRunner


CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26m" / "weights" / "best.pt"
GPU_INDEX = 0
DXCAM_DEVICE_INDEX = 0
DXCAM_OUTPUT_INDEX = None  # Let DXcam select the primary display on this adapter.
IMAGE_SIZE = 1024  # Preserve the original long-edge scale; remove only unnecessary padding.
INFERENCE_TARGET_FPS = 120
CAPTURE_TARGET_FPS = 240
ENGINE_WORKSPACE_GIB = 2.0
USE_CUDA_GRAPH = True
FUSED_PREPROCESSING = True  # Uses triton-windows when available; otherwise CUDA tensor operations.
OPENCV_THREADS = 2
CONFIDENCE = 0.50
ENEMY_CLASS_NAME = "enemy"
draw_boxes_overlay = False
collect_data = False
MAX_TARGET_DETECTIONS = 20
MAX_DISPLAY_DETECTIONS = 100
MAX_RESULT_AGE_SECONDS = 0.100
REPORT_INTERVAL_SECONDS = 5.0

# Existing behavior is reused from inference_bot; these override its settings only
# inside this separate process. The original scripts and their settings are unchanged.
AUTO_SHOOT = True
instant_mouse = False
MOUSE_UPDATE_HZ = 240
AIM_HEIGHT_FROM_BOTTOM = 0.90
AIM_TIME_CONSTANT_SECONDS = 0.030
MAX_MOUSE_STEP_PIXELS = 70
TARGET_LOST_FRAMES = 4


def sleep_until(deadline_ns: int, running: threading.Event) -> bool:
    """Release the GIL while waiting; avoid the original Python busy-spin tail."""
    while running.is_set():
        remaining = (deadline_ns - time.perf_counter_ns()) / 1_000_000_000
        if remaining <= 0:
            return True
        time.sleep(min(remaining, 0.002))
    return False


class TimedAimState(controls.AimState):
    def __init__(self, center):
        super().__init__(center)
        self.lock = threading.RLock()
        self.arm_generation = 0
        self.result_started_ns = 0

    def pause(self):
        with self.lock:
            self.arm_generation += 1
            self.result_started_ns = 0
            super().pause()

    def publish(self, boxes, captured_ns, generation):
        with self.lock:
            if not self.running.is_set() or generation != self.arm_generation:
                return False
            if time.perf_counter_ns() - captured_ns > MAX_RESULT_AGE_SECONDS * 1_000_000_000:
                self.clear()
                return False
            self.result_started_ns = captured_ns
            self.set_boxes(boxes)
            return True

    def snapshot(self):
        with self.lock:
            if time.perf_counter_ns() - self.result_started_ns > MAX_RESULT_AGE_SECONDS * 1_000_000_000:
                if self.target_box is not None:
                    self.clear()
                return None, None, self.generation
            return super().snapshot()


def configure_controls():
    for name in ("AUTO_SHOOT", "instant_mouse", "MOUSE_UPDATE_HZ", "AIM_HEIGHT_FROM_BOTTOM",
                 "AIM_TIME_CONSTANT_SECONDS", "MAX_MOUSE_STEP_PIXELS", "TARGET_LOST_FRAMES"):
        setattr(controls, name, globals()[name])
    controls.wait_until = sleep_until


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true", help="Build/cache the engine without starting the bot")
    args = parser.parse_args()
    if (IMAGE_SIZE < 32 or IMAGE_SIZE % 32 or INFERENCE_TARGET_FPS <= 0
            or CAPTURE_TARGET_FPS < INFERENCE_TARGET_FPS or ENGINE_WORKSPACE_GIB <= 0
            or not 0 < CONFIDENCE < 1 or MAX_RESULT_AGE_SECONDS <= 0
            or REPORT_INTERVAL_SECONDS <= 0 or OPENCV_THREADS < 1
            or MOUSE_UPDATE_HZ <= 0 or AIM_TIME_CONSTANT_SECONDS <= 0
            or MAX_MOUSE_STEP_PIXELS <= 0 or TARGET_LOST_FRAMES < 1
            or not 0 <= AIM_HEIGHT_FROM_BOTTOM <= 1):
        raise ValueError("Invalid optimized inference settings")
    make_dpi_aware()
    checkpoint = CHECKPOINT_PATH.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation is required")
    torch.cuda.set_device(GPU_INDEX)
    if torch.cuda.get_device_capability(GPU_INDEX)[0] != 12:
        raise RuntimeError("This entry point targets RTX Blackwell (compute capability 12.x)")
    width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    geometry = FrameGeometry.from_screen(width, height, IMAGE_SIZE)
    center = None if args.build_only else controls.load_locked_center()
    print(f"Full display {width}x{height} -> {geometry.width}x{geometry.height} input; "
          f"long-edge detail {IMAGE_SIZE}px, no crop.")
    path = ensure_engine(checkpoint, geometry.height, geometry.width, GPU_INDEX, ENGINE_WORKSPACE_GIB)
    print(f"Engine: {path}")
    if args.build_only:
        return

    configure_controls()
    cv2.setNumThreads(OPENCV_THREADS)
    runner = TensorRTRunner(path, geometry.height, geometry.width, GPU_INDEX,
                           USE_CUDA_GRAPH, FUSED_PREPROCESSING)
    raw_names = runner.metadata["names"]
    names = ({int(key): value for key, value in raw_names.items()} if isinstance(raw_names, dict)
             else dict(enumerate(raw_names)))
    enemy_ids = [index for index, name in names.items() if name.casefold() == ENEMY_CLASS_NAME.casefold()]
    if len(enemy_ids) != 1:
        raise ValueError(f"Expected one {ENEMY_CLASS_NAME!r} class, found {names}")
    enemy_id = enemy_ids[0]
    state = TimedAimState(center)
    overlay = collector = capture = None
    threads = []
    try:
        if draw_boxes_overlay:
            from inference_overlay import DetectionOverlay
            overlay = DetectionOverlay(width, height)
            overlay.start()
        if collect_data:
            from inference_collection import BackgroundCollector
            try:
                collector = BackgroundCollector(checkpoint, names)
                collector.start()
            except Exception as exc:
                print(f"Collection unavailable: {exc}")
                collector = None
        capture = LatestCapture(geometry, state, CAPTURE_TARGET_FPS, DXCAM_DEVICE_INDEX,
                                DXCAM_OUTPUT_INDEX, collector)
        with controls.high_resolution_timer():
            capture.start()
            for target, name in ((controls.watch_hotkeys, "hotkeys"), (controls.aim_loop, "mouse")):
                thread = threading.Thread(target=target, args=(state,), name=name, daemon=True)
                thread.start()
                threads.append(thread)
            print(f"Ready on {torch.cuda.get_device_name(GPU_INDEX)}. Target: {INFERENCE_TARGET_FPS} fresh FPS. "
                  f"Overlay: {overlay is not None}; collection: {collector is not None}.")
            print("Press = to arm, - to pause, Ctrl+C to exit.")
            run_loop(runner, capture, state, geometry, names, enemy_id, overlay, collector, threads)
    except KeyboardInterrupt:
        print("Bot stopped.")
    finally:
        state.pause()
        state.shutdown.set()
        for thread in threads:
            thread.join(timeout=2)
        try:
            if capture is not None:
                capture.close()
        finally:
            try:
                runner.close()
            finally:
                try:
                    if collector is not None:
                        collector.close()
                finally:
                    if overlay is not None:
                        overlay.close()


def run_loop(runner, capture, state, geometry, names, enemy_id, overlay, collector, threads):
    period_ns = round(1_000_000_000 / INFERENCE_TARGET_FPS)
    next_frame = next_overlay = 0
    report_start = time.perf_counter_ns()
    counts_start = capture.counters()
    completed = published = stale = 0
    timings = deque(maxlen=4096)
    last_capture_ns = 0
    while not state.shutdown.is_set():
        capture.raise_if_failed()
        if any(not thread.is_alive() for thread in threads):
            raise RuntimeError("A control thread stopped unexpectedly")
        if overlay is not None and not overlay.is_alive():
            raise RuntimeError(f"Overlay stopped: {overlay.error}")
        if not state.running.is_set():
            capture.discard_pending()
            if overlay is not None:
                overlay.update(())
            state.shutdown.wait(0.01)
            next_frame = next_overlay = 0
            report_start = time.perf_counter_ns()
            counts_start = capture.counters()
            completed = published = stale = 0
            timings.clear()
            continue
        if not sleep_until(next_frame, state.running):
            continue
        slot = capture.take()
        submitted_ns = time.perf_counter_ns()
        if slot is not None and (slot.generation != state.arm_generation or not state.running.is_set()
                                 or submitted_ns - slot.capture_started_ns > MAX_RESULT_AGE_SECONDS * 1_000_000_000):
            stale += 1
            capture.release(slot)
            slot = None
        if slot is not None:
            next_frame = (max(next_frame + period_ns, submitted_ns) if next_frame
                          else submitted_ns + period_ns)
            try:
                runner.submit(slot.tensor)
                output = runner.finish()
                completed_ns = time.perf_counter_ns()
                completed += 1
                sample_due = (collector is not None and slot.original is not None
                              and collector.due(completed_ns))
                threshold = min(CONFIDENCE, collector.prediction_confidence) if sample_due else CONFIDENCE
                rows = geometry.screen_detections(output, threshold)
                targets = rows[(rows[:, 5] == enemy_id) & (rows[:, 4] >= CONFIDENCE)][:MAX_TARGET_DETECTIONS]
                boxes = tuple(tuple(float(v) for v in row[:4]) for row in targets)
                accepted = state.publish(boxes, slot.capture_started_ns, slot.generation)
                publication_ns = time.perf_counter_ns()
                if accepted:
                    published += 1
                    last_capture_ns = slot.capture_started_ns
                    if overlay is not None and completed_ns >= next_overlay:
                        overlay_rows = rows[rows[:, 4] >= CONFIDENCE][:MAX_DISPLAY_DETECTIONS]
                        overlay.update(tuple((float(x1), float(y1), float(x2), float(y2), float(score),
                                              int(cls), names.get(int(cls), f"class {int(cls)}"))
                                             for x1, y1, x2, y2, score, cls in overlay_rows))
                        next_overlay = completed_ns + round(1_000_000_000 / 60)
                    if sample_due:
                        collector.submit(slot.original, tuple(tuple(float(v) for v in row)
                                                              for row in rows[:MAX_DISPLAY_DETECTIONS]), completed_ns)
                else:
                    stale += 1
                timings.append((slot.capture_ms, slot.resize_ms,
                                (completed_ns - submitted_ns) / 1_000_000,
                                (publication_ns - slot.capture_started_ns) / 1_000_000))
            finally:
                # Never let capture overwrite pinned memory until all asynchronous GPU reads finish.
                if runner.pending:
                    runner.close()
                capture.release(slot)
        now = time.perf_counter_ns()
        if overlay is not None and now - last_capture_ns > MAX_RESULT_AGE_SECONDS * 1_000_000_000:
            overlay.update(())
        if now - report_start >= REPORT_INTERVAL_SECONDS * 1_000_000_000:
            seconds = (now - report_start) / 1_000_000_000
            counts = capture.counters()
            print(f"Fresh inference {completed / seconds:.1f} FPS; published {published / seconds:.1f} FPS; "
                  f"capture {(counts[0] - counts_start[0]) / seconds:.1f} FPS; "
                  f"superseded {counts[1] - counts_start[1]}; no-new-frame grabs {counts[2] - counts_start[2]}; "
                  f"stale/disarmed {stale}.")
            if timings:
                samples = np.asarray(timings)
                average = samples.mean(axis=0)
                p95 = np.percentile(samples[:, 3], 95)
                print(f"Mean ms: capture {average[0]:.2f}, resize {average[1]:.2f}, "
                      f"transfer+GPU+result {average[2]:.2f}; capture-start to publication "
                      f"{average[3]:.2f} (p95 {p95:.2f}).")
            report_start, counts_start = now, counts
            completed = published = stale = 0
            timings.clear()


if __name__ == "__main__":
    main()
