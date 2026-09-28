"""BF16, compiled YOLO inference with configurable enemy aiming on the primary display.

Press = to arm, - to pause, and Ctrl+C to exit. Settings are below.
"""

from types import SimpleNamespace
from inference_controls import (ControlOptions, AimState, BoxCoords, high_resolution_timer,
                                wait_until, watch_hotkeys, aim_loop, load_locked_center)
import threading
import time
from pathlib import Path

import dxcam
import win32api
import win32con

from calibrate import make_dpi_aware
from inference_runtime import InferenceConfig, InferenceRuntime



CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26n" / "weights" / "best.pt"
GPU_INDEX = 0
DXCAM_DEVICE_INDEX = 0
DXCAM_OUTPUT_INDEX = 0  # Primary display; coordinates below are primary-display coordinates.
IMAGE_SIZE = 1024  # Fixed model input; smaller is faster but may miss small targets.
INFERENCE_TARGET_FPS = 120
CONFIDENCE = 0.50
ENEMY_CLASS_NAME = "enemy"
PREDICT_NMS = False  # None uses the checkpoint's default head; False selects YOLO26's NMS-free head.
REPORT_STAGE_TIMES = False
PRECISION = "bf16"  # The FP32 entry point overrides this for Pascal GPUs.
COMPILE_MODE = "reduce-overhead"
WARMUP_PASSES = 3
COMPILE_CACHE_DIR = Path(__file__).resolve().parent / ".inference_compile_cache"

AUTO_SHOOT = True
collect_data = True  # Reuse live detections; save useful frames on a background thread.
draw_boxes_overlay = True  # Draw live boxes for every class over the primary display.
COLLECT_MAX_DETECTIONS = 100
instant_mouse = False  # Move to the aim point in one mouse event when enabled.
SHOOT_INTERVAL_SECONDS = 0.10
SHOOT_HOLD_SECONDS = 0.09
AIM_HEIGHT_FROM_BOTTOM = 0.90  # 90% up the box, or 10% down from its top.
AIM_TIME_CONSTANT_SECONDS = 0.030
MOUSE_UPDATE_HZ = 240
MAX_MOUSE_STEP_PIXELS = 70
TARGET_LOST_FRAMES = 4  # Hold the lock through short detection gaps.
TARGET_MATCH_MIN_IOU = 0.10
TARGET_MATCH_MAX_CENTER_DISTANCE = 0.65  # In units of the larger box diagonal.
TARGET_MATCH_MAX_AREA_RATIO = 4.0

START_KEY = 0xBB  # = / +
STOP_KEY = 0xBD  # - / _
HOTKEY_POLL_SECONDS = 0.003
SPIN_GUARD_NS = 600_000  # Sleep most of the wait, then use the clock for the last 0.6 ms.


def inference_options(**overrides):
    """Snapshot this script's editable settings, with explicit overrides for the FP32 entry point."""
    names = (
        'CHECKPOINT_PATH',
        'GPU_INDEX',
        'DXCAM_DEVICE_INDEX',
        'DXCAM_OUTPUT_INDEX',
        'IMAGE_SIZE',
        'INFERENCE_TARGET_FPS',
        'CONFIDENCE',
        'ENEMY_CLASS_NAME',
        'PREDICT_NMS',
        'REPORT_STAGE_TIMES',
        'PRECISION',
        'COMPILE_MODE',
        'WARMUP_PASSES',
        'COMPILE_CACHE_DIR',
        'AUTO_SHOOT',
        'collect_data',
        'draw_boxes_overlay',
        'COLLECT_MAX_DETECTIONS',
        'instant_mouse',
        'SHOOT_INTERVAL_SECONDS',
        'SHOOT_HOLD_SECONDS',
        'AIM_HEIGHT_FROM_BOTTOM',
        'AIM_TIME_CONSTANT_SECONDS',
        'MOUSE_UPDATE_HZ',
        'MAX_MOUSE_STEP_PIXELS',
        'TARGET_LOST_FRAMES',
        'TARGET_MATCH_MIN_IOU',
        'TARGET_MATCH_MAX_CENTER_DISTANCE',
        'TARGET_MATCH_MAX_AREA_RATIO',
        'START_KEY',
        'STOP_KEY',
        'HOTKEY_POLL_SECONDS',
        'SPIN_GUARD_NS',
    )
    unknown = overrides.keys() - set(names)
    if unknown:
        raise ValueError(f"Unknown inference settings: {sorted(unknown)}")
    values = {name: globals()[name] for name in names}
    values.update(overrides)
    return SimpleNamespace(**values)


def predict(runtime: InferenceRuntime, frame, enemy_class_id: int,
            collection_confidence: float | None = None, *, options):
    return runtime.predict(
        frame,
        confidence=min(options.CONFIDENCE, collection_confidence) if collection_confidence is not None else options.CONFIDENCE,
        classes=None if collection_confidence is not None or options.draw_boxes_overlay else [enemy_class_id],
        max_detections=(options.COLLECT_MAX_DETECTIONS if collection_confidence is not None
                        else 100 if options.draw_boxes_overlay else 20),
    )


def enemy_boxes(result) -> tuple[BoxCoords, ...]:
    if result.boxes is None or len(result.boxes) == 0:
        return ()
    coordinates = result.boxes.xyxy.detach().cpu().tolist()
    return tuple((x1, y1, x2, y2) for x1, y1, x2, y2 in coordinates)


def sampled_enemy_boxes(rows: tuple[tuple[float, ...], ...], enemy_class_id: int, *, options) -> tuple[BoxCoords, ...]:
    return tuple((x1, y1, x2, y2) for x1, y1, x2, y2, confidence, class_id in rows
                 if int(class_id) == enemy_class_id and confidence >= options.CONFIDENCE)


def main(options=None) -> None:
    options = inference_options() if options is None else options
    if not (options.PRECISION in {"bf16", "fp32"} and options.INFERENCE_TARGET_FPS > 0 and options.MOUSE_UPDATE_HZ > 0 and
            0 < options.AIM_TIME_CONSTANT_SECONDS and options.MAX_MOUSE_STEP_PIXELS > 0 and
            0 <= options.AIM_HEIGHT_FROM_BOTTOM <= 1 and 0 < options.CONFIDENCE < 1 and
            options.SHOOT_INTERVAL_SECONDS > options.SHOOT_HOLD_SECONDS > 0 and
            options.TARGET_LOST_FRAMES >= 1 and 0 <= options.TARGET_MATCH_MIN_IOU <= 1 and
            options.TARGET_MATCH_MAX_CENTER_DISTANCE > 0 and options.TARGET_MATCH_MAX_AREA_RATIO >= 1 and
            options.IMAGE_SIZE > 0 and options.WARMUP_PASSES > 0 and options.COLLECT_MAX_DETECTIONS > 0):
        raise ValueError("FPS, aim, confidence, or shooting settings are invalid")
    make_dpi_aware()
    locked_center = load_locked_center()
    runtime = InferenceRuntime(InferenceConfig(
        checkpoint=options.CHECKPOINT_PATH,
        gpu_index=options.GPU_INDEX,
        image_size=options.IMAGE_SIZE,
        precision=options.PRECISION,
        compile_mode=options.COMPILE_MODE,
        warmup_passes=options.WARMUP_PASSES,
        cache_dir=options.COMPILE_CACHE_DIR,
        nms=options.PREDICT_NMS,
    ))
    checkpoint = runtime.checkpoint
    names = runtime.names
    enemy_ids = [class_id for class_id, name in names.items() if name.casefold() == options.ENEMY_CLASS_NAME.casefold()]
    if len(enemy_ids) != 1:
        raise ValueError(f"Expected one {options.ENEMY_CLASS_NAME!r} class in checkpoint: {names}")
    camera = dxcam.create(device_idx=options.DXCAM_DEVICE_INDEX, output_idx=options.DXCAM_OUTPUT_INDEX,
                          output_color="BGR")
    overlay = None
    try:
        full_frame = camera.grab(new_frame_only=False)
        if full_frame is None:
            raise RuntimeError("DXcam did not provide a screen frame")
        screen_height, screen_width = full_frame.shape[:2]
        primary_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        primary_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        if (screen_width, screen_height) != (primary_width, primary_height):
            raise ValueError("Selected DXcam output is not the primary display; adjust the output index")
        runtime.warmup(
            frame_shape=full_frame.shape[:2], confidence=options.CONFIDENCE,
            classes=None if options.draw_boxes_overlay else [enemy_ids[0]],
            max_detections=100 if options.draw_boxes_overlay else 20,
        )
        print(f"Inference target: {options.INFERENCE_TARGET_FPS} FPS.")
        print(f"Press = to arm, - to pause, Ctrl+C to exit. Auto shoot: {options.AUTO_SHOOT}; instant mouse: {options.instant_mouse}")

        if options.draw_boxes_overlay:
            from inference_overlay import DetectionOverlay, OVERLAY_FPS

            overlay = DetectionOverlay(screen_width, screen_height)
            overlay.start()
            print(f"Detection overlay enabled (up to {OVERLAY_FPS} redraws/second).")

        collector = None
        if options.collect_data:
            try:
                from inference_collection import BackgroundCollector
                from active_collector import collection_options

                collector = BackgroundCollector(checkpoint, names, collection_options())
                collector.start()
            except Exception as exc:
                print(f"Data collection unavailable: {exc}")
                collector = None
        print(f"Calibrated locked cursor: {locked_center}")
        state = AimState(locked_center, ControlOptions.from_settings(vars(options)))
        hotkeys = threading.Thread(target=watch_hotkeys, args=(state,), daemon=True)
        mouse = threading.Thread(target=aim_loop, args=(state,), daemon=True)
        with high_resolution_timer():
            hotkeys.start()
            mouse.start()
            period_ns = round(1_000_000_000 / options.INFERENCE_TARGET_FPS)
            next_frame = time.perf_counter_ns()
            report_start = next_frame
            frames = 0
            stage_ms = {"frame": 0.0, "capture": 0.0, "preprocess": 0.0,
                        "inference": 0.0, "postprocess": 0.0}
            try:
                while True:
                    if not hotkeys.is_alive() or not mouse.is_alive():
                        raise RuntimeError("A hotkey or mouse-control thread stopped unexpectedly")
                    if overlay is not None and not overlay.is_alive():
                        raise RuntimeError(f"Detection overlay stopped unexpectedly: {overlay.error}")
                    if not state.running.is_set():
                        if overlay is not None:
                            overlay.update(())
                        state.shutdown.wait(0.01)
                        next_frame = report_start = time.perf_counter_ns()
                        frames = 0
                        stage_ms = {key: 0.0 for key in stage_ms}
                        continue
                    if not wait_until(next_frame, state.running, spin_guard_ns=options.SPIN_GUARD_NS):
                        continue
                    capture_start = time.perf_counter_ns()
                    frame = camera.grab(new_frame_only=False)
                    if frame is not None:
                        if options.REPORT_STAGE_TIMES:
                            stage_ms["capture"] += (time.perf_counter_ns() - capture_start) / 1_000_000
                        sample_due = collector is not None and collector.due(time.perf_counter_ns())
                        result = predict(runtime, frame, enemy_ids[0],
                                         collector.prediction_confidence if sample_due else None, options=options)
                        if sample_due or overlay is not None:
                            rows = (tuple(tuple(row) for row in result.boxes.data.detach().cpu().tolist())
                                    if result.boxes is not None and len(result.boxes) else ())
                            state.set_boxes(sampled_enemy_boxes(rows, enemy_ids[0], options=options))
                            if sample_due and state.running.is_set():
                                collector.submit(frame, rows, time.perf_counter_ns())
                            if overlay is not None:
                                overlay.update_rows(rows if state.running.is_set() else (), names,
                                                    confidence=options.CONFIDENCE)
                        else:
                            state.set_boxes(enemy_boxes(result))
                        if options.REPORT_STAGE_TIMES:
                            for key in ("preprocess", "inference", "postprocess"):
                                stage_ms[key] += result.speed[key]
                            stage_ms["frame"] += (time.perf_counter_ns() - capture_start) / 1_000_000
                        frames += 1
                    now = time.perf_counter_ns()
                    if now - report_start >= 5_000_000_000:
                        actual_fps = frames * 1_000_000_000 / (now - report_start)
                        print(f"Actual inference: {actual_fps:.1f} FPS (target {options.INFERENCE_TARGET_FPS})")
                        if options.REPORT_STAGE_TIMES and frames:
                            print("Average stage time: " + ", ".join(
                                f"{key} {stage_ms[key] / frames:.1f} ms" for key in stage_ms
                            ))
                        report_start, frames = now, 0
                        stage_ms = {key: 0.0 for key in stage_ms}
                    next_frame = max(next_frame + period_ns, now)
            except KeyboardInterrupt:
                print("Bot stopped.")
            finally:
                state.running.clear()
                state.clear()
                state.shutdown.set()
                hotkeys.join(timeout=1)
                mouse.join(timeout=1)
                if collector is not None:
                    collector.close()
    finally:
        if overlay is not None:
            overlay.close()
        camera.release()


if __name__ == "__main__":
    main()
