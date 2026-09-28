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

from dataset_project import load_project

from dataset_utils import (class_name_map, prediction_rows)


# Settings
DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "rivals"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26m" / "weights" / "best.pt"
DEVICE_INDEX = 0
OUTPUT_INDEX = 0  # Primary monitor on the selected graphics device.
DEVICE = 0
IMAGE_SIZE = 1024
VIDEO_FPS = 20
JPEG_QUALITY = 95
INFERENCE_FPS = 1
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
                 state: dict, state_lock: threading.Lock) -> None:
    """Publish review frames while the shared recorder owns video encoding."""
    def publish(frame, frame_index):
        interpolation = cv2.INTER_AREA if max(frame.shape[:2]) > IMAGE_SIZE else cv2.INTER_LINEAR
        image_frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation)
        with state_lock:
            state["frame"] = image_frame
            state["frame_index"] = frame_index
    try:
        write_video(camera, ffmpeg_path, video_path, stop_event, fps=VIDEO_FPS,
                    preset=NVENC_PRESET, quality=NVENC_QUALITY, on_frame=publish, paced=True)
        state["video_saved"] = True
    except Exception as exc:
        state["error"] = str(exc)
        state["video_saved"] = video_path.is_file()


def main() -> None:
    import dxcam
    from ultralytics import YOLO
    if INFERENCE_FPS <= 0 or VIDEO_FPS <= 0 or MAX_SAVES_PER_SESSION <= 0 or MIN_SECONDS_BETWEEN_SAVES < 0:
        raise ValueError("FPS, save limit, and save interval settings are invalid")
    if not 0 <= PREDICTION_CONFIDENCE < REVIEW_CONFIDENCE_LOW < REVIEW_CONFIDENCE_HIGH <= 1:
        raise ValueError("Confidence settings must increase from prediction to review high")
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
    started_at = time.monotonic()
    collecting = False
    manual_save_pending = False
    next_due = started_at
    recording_thread = None
    recording_stop = None
    recording_state = None
    recording_lock = threading.Lock()
    video_path = None

    camera = dxcam.create(device_idx=DEVICE_INDEX, output_idx=OUTPUT_INDEX, output_color="BGR")
    start_was_down = bool(win32api.GetAsyncKeyState(START_KEY) & 0x8000)
    stop_was_down = bool(win32api.GetAsyncKeyState(STOP_KEY) & 0x8000)
    save_was_down = bool(win32api.GetAsyncKeyState(SAVE_KEY) & 0x8000)
    print(f"Ready: {project.root}. Press = to collect and record, V to save a frame, - to stop, Ctrl+C to exit.")
    print(f"First available image number: {writer.next_number}. Limit: {MAX_SAVES_PER_SESSION} per run.")
    try:
        while True:
            start_is_down = bool(win32api.GetAsyncKeyState(START_KEY) & 0x8000)
            stop_is_down = bool(win32api.GetAsyncKeyState(STOP_KEY) & 0x8000)
            save_key_state = win32api.GetAsyncKeyState(SAVE_KEY)
            save_is_down = bool(save_key_state & 0x8000)
            save_pressed = bool(save_key_state & 0x0001) or (save_is_down and not save_was_down)
            if stop_is_down and not stop_was_down and collecting:
                collecting = False
                manual_save_pending = False
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
                    next_due = time.monotonic()
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
                print(f"Video recording stopped unexpectedly: {recording_state.get('error') or 'unknown error'}")
                continue

            now = time.monotonic()
            if not collecting or (now < next_due and not manual_save_pending):
                time.sleep(KEY_POLL_SECONDS)
                continue
            with recording_lock:
                frame = recording_state.get("frame")
                frame_index = recording_state.get("frame_index", -1)
                if frame is not None:
                    frame = frame.copy()
            if frame is None or frame_index < 0:
                time.sleep(KEY_POLL_SECONDS)
                continue
            manual_save = manual_save_pending
            manual_save_pending = False
            next_due = now + 1 / INFERENCE_FPS
            result = model.predict(source=frame, conf=PREDICTION_CONFIDENCE,
                                   imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)[0]
            writer.session_id = session_id
            writer.save(frame, prediction_rows(result), now, manual=manual_save, extra_metadata={
                "video": {"path": video_path.relative_to(dataset_dir).as_posix(),
                          "frame_index": frame_index, "fps": VIDEO_FPS}})
            if writer.saved_count >= MAX_SAVES_PER_SESSION:
                collecting = False
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
        if recording_thread is not None and recording_thread.is_alive():
            recording_stop.set()
            recording_thread.join()
        camera.release()


if __name__ == "__main__":
    main()
