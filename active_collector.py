"""Collect useful YOLO gameplay frames for later human review. Edit settings below."""

import os
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import dxcam
import win32api
from PIL import Image
from ultralytics import YOLO

from dataset_project import load_project
from review_metadata import (
    difference_hash, hash_distance, save_review_metadata,
)


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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_files(folder: Path):
    if folder.is_dir():
        yield from (path for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def starting_number(dataset_dir: Path) -> tuple[int, set[int]]:
    folders = [dataset_dir / split / "images" for split in ("train", "valid", "test", "labeled")]
    folders.append(dataset_dir / "unlabeled")
    paths = [path for folder in folders for path in image_files(folder)]
    used = {int(path.stem) for path in paths if path.stem.isdecimal()}
    number = len(paths) + 1
    while number in used:
        number += 1
    return number, used


def choose_reason(confidences: list[float], now: float, last_reason_saved: dict[str, float]) -> str | None:
    if not confidences:
        return None
    if any(REVIEW_CONFIDENCE_LOW <= value <= REVIEW_CONFIDENCE_HIGH for value in confidences):
        return "uncertain"
    if any(value < REVIEW_CONFIDENCE_LOW for value in confidences):
        reason, interval = "weak", WEAK_SAMPLE_SECONDS
    else:
        reason, interval = "high_confidence_audit", HIGH_CONFIDENCE_SAMPLE_SECONDS
    return reason if now - last_reason_saved[reason] >= interval else None


def save_candidate(output_dir: Path, frame, metadata: dict, number: int,
                   used_numbers: set[int]) -> tuple[Path, int]:
    encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not encoded_ok:
        raise RuntimeError("Could not encode the captured frame")
    folders = [output_dir.parent / split / "images" for split in ("train", "valid", "test", "labeled")]
    folders.append(output_dir)
    while True:
        while number in used_numbers or any(
            path.suffix.lower() in IMAGE_EXTENSIONS
            for folder in folders if folder.is_dir() for path in folder.glob(f"{number}.*")
        ):
            used_numbers.add(number)
            number += 1
        image_path = output_dir / f"{number}.jpg"
        try:
            with image_path.open("xb") as image_file:
                image_file.write(encoded.tobytes())
            break
        except FileExistsError:
            used_numbers.add(number)
            number += 1
    try:
        save_review_metadata(image_path, metadata)
    except Exception:
        image_path.unlink(missing_ok=True)
        raise
    used_numbers.add(number)
    return image_path, number + 1


def record_video(camera, ffmpeg_path: str, video_path: Path, stop_event: threading.Event,
                 state: dict, state_lock: threading.Lock) -> None:
    """Capture 20 FPS at DXcam's source resolution and publish 1024px review frames."""
    log_path = video_path.with_suffix(".ffmpeg.log")
    partial_path = video_path.with_name(f".{video_path.stem}.partial.mp4")
    process = None
    log_file = None
    exit_code = None
    camera_started = False
    try:
        camera.start(target_fps=VIDEO_FPS, video_mode=True)
        camera_started = True
        frame = camera.get_latest_frame(copy=True)
        if frame is None:
            raise RuntimeError("DXcam did not provide a frame")
        source_height, source_width = frame.shape[:2]
        command = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{source_width}x{source_height}", "-framerate", str(VIDEO_FPS),
            "-i", "pipe:0", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "h264_nvenc", "-preset", NVENC_PRESET,
            "-rc", "vbr", "-cq", str(NVENC_QUALITY), "-b:v", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial_path),
        ]
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_file,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        frame_index = 0
        next_frame_at = time.monotonic()
        while not stop_event.is_set():
            if process.poll() is not None:
                raise RuntimeError(f"FFmpeg stopped unexpectedly. See {log_path}")
            delay = next_frame_at - time.monotonic()
            if delay > 0 and stop_event.wait(delay):
                break
            current = camera.get_latest_frame(copy=True)
            if current is None:
                continue
            if current.shape[:2] != (source_height, source_width):
                raise RuntimeError("Screen size changed during recording")
            try:
                process.stdin.write(current.tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"FFmpeg could not accept frames. See {log_path}") from exc
            interpolation = cv2.INTER_AREA if max(current.shape[:2]) > IMAGE_SIZE else cv2.INTER_LINEAR
            image_frame = cv2.resize(current, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation)
            with state_lock:
                state["frame"] = image_frame
                state["frame_index"] = frame_index
            frame_index += 1
            next_frame_at += 1 / VIDEO_FPS
            # If encoding or capture falls behind, skip elapsed ticks instead of speeding up later.
            if next_frame_at < time.monotonic() - 1 / VIDEO_FPS:
                next_frame_at = time.monotonic()
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        if camera_started and camera.is_capturing:
            camera.stop()
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                exit_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                exit_code = process.returncode
        if log_file is not None:
            log_file.close()
        if process is not None and exit_code == 0:
            try:
                partial_path.replace(video_path)
                state["video_saved"] = True
                print(f"Saved {video_path} ({frame_index} frames at {VIDEO_FPS} FPS).")
            except OSError as exc:
                state["error"] = f"Could not finalize video: {exc}"
        elif process is not None and exit_code != 0:
            state["error"] = f"FFmpeg exited with code {exit_code}. See {log_path}"


def main() -> None:
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
    model_names = dict(model_names.items()) if isinstance(model_names, dict) else dict(enumerate(model_names))
    if not set(model_names.values()).issubset(project.names):
        raise ValueError(f"Checkpoint classes {model_names} do not match dataset classes {project.names}")

    output_dir = dataset_dir / "unlabeled"
    output_dir.mkdir(parents=True, exist_ok=True)
    next_number, used_numbers = starting_number(dataset_dir)
    checkpoint_stat = checkpoint.stat()
    checkpoint_identity = {"path": str(checkpoint), "size": checkpoint_stat.st_size,
                           "modified_ns": checkpoint_stat.st_mtime_ns}
    session_id = ""
    recent_hashes: deque[int] = deque(maxlen=RECENT_HASH_COUNT)
    last_saved_at = float("-inf")
    started_at = time.monotonic()
    last_reason_saved = {"weak": started_at, "high_confidence_audit": started_at,
                         "uncertain": float("-inf")}
    saved_count = 0
    collecting = False
    next_due = started_at
    recording_thread = None
    recording_stop = None
    recording_state = None
    recording_lock = threading.Lock()
    video_path = None

    camera = dxcam.create(device_idx=DEVICE_INDEX, output_idx=OUTPUT_INDEX, output_color="BGR")
    start_was_down = bool(win32api.GetAsyncKeyState(START_KEY) & 0x8000)
    stop_was_down = bool(win32api.GetAsyncKeyState(STOP_KEY) & 0x8000)
    print(f"Ready: {project.root}. Press = to collect and record, - to stop, Ctrl+C to exit.")
    print(f"First available image number: {next_number}. Limit: {MAX_SAVES_PER_SESSION} per run.")
    try:
        while True:
            start_is_down = bool(win32api.GetAsyncKeyState(START_KEY) & 0x8000)
            stop_is_down = bool(win32api.GetAsyncKeyState(STOP_KEY) & 0x8000)
            if stop_is_down and not stop_was_down and collecting:
                collecting = False
                recording_stop.set()
                recording_thread.join()
                error = recording_state.get("error")
                if error:
                    print(f"Video recording error: {error}")
                elif recording_state.get("video_saved"):
                    print(f"Stopped. Saved {saved_count} candidate images and {video_path.name}.")
                else:
                    print(f"Stopped. Saved {saved_count} candidate images; no video was finalized.")
            elif start_is_down and not start_was_down and not collecting:
                if saved_count < MAX_SAVES_PER_SESSION:
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
                    print(f"Collecting images at {INFERENCE_FPS:g} FPS and recording {VIDEO_FPS} FPS. Press - to stop.")
                else:
                    print("Session save limit reached. Restart the script for another session.")
            start_was_down, stop_was_down = start_is_down, stop_is_down

            if collecting and recording_thread is not None and not recording_thread.is_alive():
                collecting = False
                print(f"Video recording stopped unexpectedly: {recording_state.get('error') or 'unknown error'}")
                continue

            now = time.monotonic()
            if not collecting or now < next_due:
                time.sleep(KEY_POLL_SECONDS)
                continue
            next_due = now + 1 / INFERENCE_FPS
            with recording_lock:
                frame = recording_state.get("frame")
                frame_index = recording_state.get("frame_index", -1)
                if frame is not None:
                    frame = frame.copy()
            if frame is None or frame_index < 0:
                continue
            result = model.predict(source=frame, conf=PREDICTION_CONFIDENCE,
                                   imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)[0]
            draft_boxes = []
            if result.boxes is not None:
                for predicted in result.boxes:
                    class_name = model_names[int(predicted.cls.item())]
                    draft_boxes.append({
                        "class_name": class_name,
                        "confidence": float(predicted.conf.item()),
                        "xywhn": [float(value) for value in predicted.xywhn[0].tolist()],
                    })
            reason = choose_reason([box["confidence"] for box in draft_boxes], now, last_reason_saved)
            if reason is None or now - last_saved_at < MIN_SECONDS_BETWEEN_SAVES:
                continue
            frame_hash = difference_hash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if any(hash_distance(frame_hash, previous) <= DUPLICATE_HASH_DISTANCE
                   for previous in recent_hashes):
                continue
            metadata = {
                "schema_version": 1,
                "session_id": session_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "selection_reason": reason,
                "checkpoint": checkpoint_identity,
                "dhash": f"{frame_hash:016x}",
                "boxes": draft_boxes,
                "video": {
                    "path": video_path.relative_to(dataset_dir).as_posix(),
                    "frame_index": frame_index,
                    "fps": VIDEO_FPS,
                },
            }
            image_path, next_number = save_candidate(output_dir, frame, metadata, next_number, used_numbers)
            recent_hashes.append(frame_hash)
            last_saved_at = now
            last_reason_saved[reason] = now
            saved_count += 1
            print(f"Saved {image_path.name} ({reason}, {len(draft_boxes)} draft boxes; {saved_count}/{MAX_SAVES_PER_SESSION})")
            if saved_count >= MAX_SAVES_PER_SESSION:
                collecting = False
                recording_stop.set()
                recording_thread.join()
                error = recording_state.get("error")
                if error:
                    print(f"Video recording error: {error}")
                else:
                    print("Session save limit reached and video saved. Press Ctrl+C to exit.")
    except KeyboardInterrupt:
        print(f"Stopped. Saved {saved_count} candidate images in {output_dir}.")
    finally:
        if recording_thread is not None and recording_thread.is_alive():
            recording_stop.set()
            recording_thread.join()
        camera.release()


if __name__ == "__main__":
    main()
