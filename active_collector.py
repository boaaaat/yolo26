"""Collect useful YOLO gameplay frames for later human review. Edit settings below."""

import os
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
DATASET_DIR = Path(__file__).resolve().parent / "datasets"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26m" / "weights" / "best.pt"
DEVICE_INDEX = 0
OUTPUT_INDEX = 0  # Primary monitor on the selected graphics device.
DEVICE = 0
IMAGE_SIZE = 1024
JPEG_QUALITY = 95
INFERENCE_FPS = 1
PREDICTION_CONFIDENCE = 0.15
REVIEW_CONFIDENCE_LOW = 0.50
REVIEW_CONFIDENCE_HIGH = 0.80
MIN_SECONDS_BETWEEN_SAVES = 3
MAX_SAVES_PER_SESSION = 150
WEAK_SAMPLE_SECONDS = 20
EMPTY_SAMPLE_SECONDS = 60
HIGH_CONFIDENCE_SAMPLE_SECONDS = 120
RECENT_HASH_COUNT = 50
DUPLICATE_HASH_DISTANCE = 5  # 64-bit difference hash; lower values reject fewer frames.
KEY_POLL_SECONDS = 0.05

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
    if any(REVIEW_CONFIDENCE_LOW <= value <= REVIEW_CONFIDENCE_HIGH for value in confidences):
        return "uncertain"
    if not confidences:
        reason, interval = "empty", EMPTY_SAMPLE_SECONDS
    elif any(value < REVIEW_CONFIDENCE_LOW for value in confidences):
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


def main() -> None:
    if INFERENCE_FPS <= 0 or MAX_SAVES_PER_SESSION <= 0 or MIN_SECONDS_BETWEEN_SAVES < 0:
        raise ValueError("FPS, save limit, and save interval settings are invalid")
    if not 0 <= PREDICTION_CONFIDENCE < REVIEW_CONFIDENCE_LOW < REVIEW_CONFIDENCE_HIGH <= 1:
        raise ValueError("Confidence settings must increase from prediction to review high")
    dataset_dir = Path(DATASET_DIR).expanduser().resolve()
    checkpoint = Path(CHECKPOINT_PATH).expanduser().resolve()
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
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    recent_hashes: deque[int] = deque(maxlen=RECENT_HASH_COUNT)
    last_saved_at = float("-inf")
    started_at = time.monotonic()
    last_reason_saved = {"empty": started_at, "weak": started_at,
                         "high_confidence_audit": started_at, "uncertain": float("-inf")}
    saved_count = 0
    collecting = False
    next_due = started_at

    camera = dxcam.create(device_idx=DEVICE_INDEX, output_idx=OUTPUT_INDEX, output_color="BGR")
    start_was_down = bool(win32api.GetAsyncKeyState(START_KEY) & 0x8000)
    stop_was_down = bool(win32api.GetAsyncKeyState(STOP_KEY) & 0x8000)
    print(f"Ready: {project.root}. Press = to collect, - to pause, Ctrl+C to exit.")
    print(f"First available image number: {next_number}. Limit: {MAX_SAVES_PER_SESSION} per run.")
    try:
        while True:
            start_is_down = bool(win32api.GetAsyncKeyState(START_KEY) & 0x8000)
            stop_is_down = bool(win32api.GetAsyncKeyState(STOP_KEY) & 0x8000)
            if stop_is_down and not stop_was_down and collecting:
                collecting = False
                print(f"Paused. Saved {saved_count} candidate images.")
            elif start_is_down and not start_was_down and not collecting:
                if saved_count < MAX_SAVES_PER_SESSION:
                    collecting = True
                    next_due = time.monotonic()
                    print(f"Collecting at {INFERENCE_FPS:g} FPS. Press - to pause.")
                else:
                    print("Session save limit reached. Restart the script for another session.")
            start_was_down, stop_was_down = start_is_down, stop_is_down

            now = time.monotonic()
            if not collecting or now < next_due:
                time.sleep(KEY_POLL_SECONDS)
                continue
            next_due = now + 1 / INFERENCE_FPS
            frame = camera.grab(new_frame_only=False)
            if frame is None:
                continue
            interpolation = cv2.INTER_AREA if max(frame.shape[:2]) > IMAGE_SIZE else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation)
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
            }
            image_path, next_number = save_candidate(output_dir, frame, metadata, next_number, used_numbers)
            recent_hashes.append(frame_hash)
            last_saved_at = now
            last_reason_saved[reason] = now
            saved_count += 1
            print(f"Saved {image_path.name} ({reason}, {len(draft_boxes)} draft boxes; {saved_count}/{MAX_SAVES_PER_SESSION})")
            if saved_count >= MAX_SAVES_PER_SESSION:
                collecting = False
                print("Session save limit reached. Press Ctrl+C to exit.")
    except KeyboardInterrupt:
        print(f"Stopped. Saved {saved_count} candidate images in {output_dir}.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
