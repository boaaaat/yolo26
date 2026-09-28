"""Collect useful YOLO gameplay frames for later human review. Edit settings below."""

from inference_collection import CollectionOptions, SampleWriter
from recorder import record_video as write_video
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import win32api
import win32con

from calibrate import make_dpi_aware
from dataset_project import load_project
from inference_overlay import DetectionOverlay, OVERLAY_FPS

from dataset_utils import (class_name_map, prediction_rows)


# Settings
DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "rivals"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26n" / "weights" / "best.pt"
DEVICE_INDEX = 0
OUTPUT_INDEX = 0  # Primary monitor on the selected graphics device.
DEVICE = 0
IMAGE_SIZE = 1024
VIDEO_FPS = 20
JPEG_QUALITY = 95
INFERENCE_FPS = 1  # Review sampling rate; live overlay predictions target OVERLAY_FPS.
draw_boxes_overlay = True
OVERLAY_CONFIDENCE = 0.50
OVERLAY_MAX_AGE_SECONDS = 0.25
OPENCV_THREADS = 2
TORCH_CPU_THREADS = 1
PREDICTION_CONFIDENCE = 0.15
REVIEW_CONFIDENCE_LOW = 0.50
REVIEW_CONFIDENCE_HIGH = 0.80
MIN_SECONDS_BETWEEN_SAVES = 2
MAX_SAVES_PER_SESSION = 150
WEAK_SAMPLE_SECONDS = 20
HIGH_CONFIDENCE_SAMPLE_SECONDS = 120
RECENT_HASH_COUNT = 50
DUPLICATE_HASH_DISTANCE = 5  # 64-bit difference hash; lower values reject fewer frames.
KEY_POLL_SECONDS = 0.05
FFMPEG_EXECUTABLE = "ffmpeg"
NVENC_PRESET = "p5"
NVENC_QUALITY = 20

START_KEY = 0xBB  # = / +
STOP_KEY = 0xBD  # - / _
SAVE_KEY = 0x56  # V: save the current video frame for review.


def collection_options() -> CollectionOptions:
    return CollectionOptions(
        dataset_dir=DATASET_DIR,
        image_size=IMAGE_SIZE,
        jpeg_quality=JPEG_QUALITY,
        inference_fps=INFERENCE_FPS,
        prediction_confidence=PREDICTION_CONFIDENCE,
        review_confidence_low=REVIEW_CONFIDENCE_LOW,
        review_confidence_high=REVIEW_CONFIDENCE_HIGH,
        min_seconds_between_saves=MIN_SECONDS_BETWEEN_SAVES,
        max_saves_per_session=MAX_SAVES_PER_SESSION,
        weak_sample_seconds=WEAK_SAMPLE_SECONDS,
        high_confidence_sample_seconds=HIGH_CONFIDENCE_SAMPLE_SECONDS,
        recent_hash_count=RECENT_HASH_COUNT,
        duplicate_hash_distance=DUPLICATE_HASH_DISTANCE,
    )


def record_video(camera, ffmpeg_path: str, video_path: Path, stop_event: threading.Event,
                 state: dict, state_lock: threading.Condition) -> None:
    """Publish review frames while the shared recorder owns video encoding."""
    def resize_frame(frame):
        interpolation = cv2.INTER_AREA if max(frame.shape[:2]) > IMAGE_SIZE else cv2.INTER_LINEAR
        return cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation)

    def publish_preview(frame):
        preview_time = time.perf_counter()
        preview_frame = resize_frame(frame)
        with state_lock:
            state["preview_frame"] = preview_frame
            state["preview_time"] = preview_time
            state_lock.notify_all()

    def publish(frame, frame_index):
        # Encoding runs separately: use this exact encoded frame, never the
        # newer preview frame, so review images retain correct video indices.
        image_frame = resize_frame(frame)
        with state_lock:
            state["frame"] = image_frame
            state["frame_index"] = frame_index
            state["frame_time"] = time.perf_counter()
            state_lock.notify_all()
    try:
        write_video(camera, ffmpeg_path, video_path, stop_event, fps=VIDEO_FPS,
                    preset=NVENC_PRESET, quality=NVENC_QUALITY, on_frame=publish, paced=True,
                    preview_fps=OVERLAY_FPS if draw_boxes_overlay else None,
                    on_preview=publish_preview if draw_boxes_overlay else None)
        state["video_saved"] = True
    except Exception as exc:
        state["error"] = str(exc)
        state["video_saved"] = video_path.is_file()


def main() -> None:
    import dxcam
    import torch
    from ultralytics import YOLO
    cv2.setNumThreads(OPENCV_THREADS)
    torch.set_num_threads(TORCH_CPU_THREADS)
    if INFERENCE_FPS <= 0 or VIDEO_FPS <= 0 or MAX_SAVES_PER_SESSION <= 0 or MIN_SECONDS_BETWEEN_SAVES < 0:
        raise ValueError("FPS, save limit, and save interval settings are invalid")
    if not 0 <= PREDICTION_CONFIDENCE < REVIEW_CONFIDENCE_LOW < REVIEW_CONFIDENCE_HIGH <= 1:
        raise ValueError("Confidence settings must increase from prediction to review high")
    if not 0 <= OVERLAY_CONFIDENCE <= 1 or OVERLAY_MAX_AGE_SECONDS <= 0:
        raise ValueError("Overlay confidence and maximum age settings are invalid")
    dataset_dir = Path(DATASET_DIR).expanduser().resolve()
    checkpoint = Path(CHECKPOINT_PATH).expanduser().resolve()
    ffmpeg_path = shutil.which(FFMPEG_EXECUTABLE)
    if ffmpeg_path is None:
        raise FileNotFoundError(f"FFmpeg not found: {FFMPEG_EXECUTABLE}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    project = load_project(dataset_dir)
    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError("The checkpoint must be an object detection model")
    model_names = model.names
    model_names = class_name_map(model_names)
    if not set(model_names.values()).issubset(project.names):
        raise ValueError(f"Checkpoint classes {model_names} do not match dataset classes {project.names}")

    writer = SampleWriter(checkpoint, model_names, collection_options())
    output_dir = writer.output_dir
    started_at = time.perf_counter()
    collecting = False
    manual_save_pending = False
    next_due = started_at
    next_preview_due = started_at
    last_preview_time = None
    recording_thread = None
    recording_stop = None
    recording_state = None
    recording_lock = threading.Condition()
    video_path = None

    make_dpi_aware()
    camera = dxcam.create(device_idx=DEVICE_INDEX, output_idx=OUTPUT_INDEX, output_color="BGR")
    overlay = None
    start_was_down = bool(win32api.GetAsyncKeyState(START_KEY) & 0x8000)
    stop_was_down = bool(win32api.GetAsyncKeyState(STOP_KEY) & 0x8000)
    save_was_down = bool(win32api.GetAsyncKeyState(SAVE_KEY) & 0x8000)
    print(f"Ready: {project.root}. Press = to collect and record, V to save a frame, - to stop, Ctrl+C to exit.")
    print(f"First available image number: {writer.next_number}. Limit: {MAX_SAVES_PER_SESSION} per run.")
    try:
        if draw_boxes_overlay:
            width, height = camera.width, camera.height
            if (width, height) != (win32api.GetSystemMetrics(win32con.SM_CXSCREEN),
                                   win32api.GetSystemMetrics(win32con.SM_CYSCREEN)):
                raise ValueError("The overlay requires the primary display; adjust OUTPUT_INDEX")
            overlay = DetectionOverlay(width, height)
            overlay.start()
            print(f"Live overlay: {OVERLAY_FPS} FPS target, confidence >= {OVERLAY_CONFIDENCE:g}.")
        while True:
            if overlay is not None and not overlay.is_alive():
                raise RuntimeError(f"Detection overlay stopped unexpectedly: {overlay.error}")
            start_is_down = bool(win32api.GetAsyncKeyState(START_KEY) & 0x8000)
            stop_is_down = bool(win32api.GetAsyncKeyState(STOP_KEY) & 0x8000)
            save_key_state = win32api.GetAsyncKeyState(SAVE_KEY)
            save_is_down = bool(save_key_state & 0x8000)
            save_pressed = bool(save_key_state & 0x0001) or (save_is_down and not save_was_down)
            if stop_is_down and not stop_was_down and collecting:
                collecting = False
                manual_save_pending = False
                if overlay is not None:
                    overlay.update(())
                recording_stop.set()
                recording_thread.join()
                error = recording_state.get("error")
                if error:
                    print(f"Video recording error: {error}")
                elif recording_state.get("video_saved"):
                    print(f"Stopped. Saved {writer.saved_count} candidate images and {video_path.name}.")
                else:
                    print(f"Stopped. Saved {writer.saved_count} candidate images; no video was finalized.")
            elif start_is_down and not start_was_down and not collecting:
                if writer.saved_count < MAX_SAVES_PER_SESSION:
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    session_id = f"{timestamp}-{os.getpid()}"
                    video_dir = dataset_dir / "videos"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = video_dir / f"{session_id}.mp4"
                    recording_stop = threading.Event()
                    recording_state = {"frame": None, "frame_index": -1,
                                       "video_saved": False, "error": None}
                    recording_thread = threading.Thread(
                        target=record_video,
                        args=(camera, ffmpeg_path, video_path, recording_stop,
                              recording_state, recording_lock),
                        name="active-collector-video", daemon=True,
                    )
                    recording_thread.start()
                    collecting = True
                    next_due = time.perf_counter()
                    next_preview_due = next_due
                    last_preview_time = None
                    print(f"Collecting images at {INFERENCE_FPS:g} FPS and recording {VIDEO_FPS} FPS. Press V to save or - to stop.")
                else:
                    print("Session save limit reached. Restart the script for another session.")
            start_was_down, stop_was_down = start_is_down, stop_is_down
            save_was_down = save_is_down
            if save_pressed and collecting:
                manual_save_pending = True

            if collecting and recording_thread is not None and not recording_thread.is_alive():
                collecting = False
                manual_save_pending = False
                if overlay is not None:
                    overlay.update(())
                print(f"Video recording stopped unexpectedly: {recording_state.get('error') or 'unknown error'}")
                continue

            now = time.perf_counter()
            if not collecting:
                time.sleep(KEY_POLL_SECONDS)
                continue
            sample_due = now >= next_due or manual_save_pending
            preview_due = overlay is not None and now >= next_preview_due
            if not sample_due and not preview_due:
                wake_at = min(next_due, next_preview_due) if overlay is not None else next_due
                time.sleep(min(KEY_POLL_SECONDS, max(0, wake_at - now)))
                continue
            with recording_lock:
                frame = recording_state.get("frame" if sample_due else "preview_frame")
                frame_time = recording_state.get("frame_time" if sample_due else "preview_time")
                frame_index = recording_state.get("frame_index", -1)
                if (frame is None or (sample_due and frame_index < 0)
                        or (not sample_due and frame_time == last_preview_time)):
                    if overlay is not None and frame_time is not None and now - frame_time > OVERLAY_MAX_AGE_SECONDS:
                        overlay.update(())
                    recording_lock.wait(timeout=min(KEY_POLL_SECONDS, 1 / OVERLAY_FPS))
                    continue
                frame = frame.copy()
            manual_save = manual_save_pending if sample_due else False
            if sample_due:
                manual_save_pending = False
                next_due = now + 1 / INFERENCE_FPS
            next_preview_due += 1 / OVERLAY_FPS
            if next_preview_due < now:
                next_preview_due = now + 1 / OVERLAY_FPS
            last_preview_time = frame_time
            confidence = PREDICTION_CONFIDENCE if sample_due else OVERLAY_CONFIDENCE
            if overlay is not None:
                confidence = min(confidence, OVERLAY_CONFIDENCE)
            result = model.predict(source=frame, conf=confidence,
                                   imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)[0]
            rows = prediction_rows(result)
            if overlay is not None:
                fresh = time.perf_counter() - frame_time <= OVERLAY_MAX_AGE_SECONDS
                overlay.update_rows(rows if fresh else (), model_names,
                                    source_size=(frame.shape[1], frame.shape[0]),
                                    confidence=OVERLAY_CONFIDENCE)
            if sample_due:
                writer.session_id = session_id
                writer.save(frame, rows, time.monotonic(), manual=manual_save, extra_metadata={
                    "video": {"path": video_path.relative_to(dataset_dir).as_posix(),
                              "frame_index": frame_index, "fps": VIDEO_FPS}})
            if writer.saved_count >= MAX_SAVES_PER_SESSION:
                collecting = False
                if overlay is not None:
                    overlay.update(())
                recording_stop.set()
                recording_thread.join()
                error = recording_state.get("error")
                if error:
                    print(f"Video recording error: {error}")
                else:
                    print("Session save limit reached and video saved. Press Ctrl+C to exit.")
    except KeyboardInterrupt:
        print(f"Stopped. Saved {writer.saved_count} candidate images in {output_dir}.")
    finally:
        try:
            if overlay is not None:
                overlay.close()
        finally:
            try:
                if recording_thread is not None and recording_thread.is_alive():
                    recording_stop.set()
                    recording_thread.join()
            finally:
                camera.release()


if __name__ == "__main__":
    main()
