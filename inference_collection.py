"""Shared candidate selection/saving, with an optional bounded background writer."""
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
import cv2
import numpy as np
from PIL import Image
from dataset_project import load_project
from dataset_utils import IMAGE_SUFFIXES, image_files
from review_metadata import difference_hash, hash_distance, save_review_metadata


@dataclass(frozen=True)
class CollectionOptions:
    dataset_dir: Path
    image_size: int
    jpeg_quality: int
    inference_fps: float
    prediction_confidence: float
    review_confidence_low: float
    review_confidence_high: float
    min_seconds_between_saves: float
    max_saves_per_session: int
    weak_sample_seconds: float
    high_confidence_sample_seconds: float
    recent_hash_count: int
    duplicate_hash_distance: int


def next_image_number(paths, *, start_at=None):
    paths = list(paths)
    used = {int(path.stem) for path in paths if path.stem.isdecimal()}
    number = len(paths) + 1 if start_at is None else start_at
    while number in used:
        number += 1
    return number, used


def starting_number(dataset_dir: Path):
    folders = [dataset_dir / split / "images" for split in ("train", "valid", "test", "labeled")]
    folders.append(dataset_dir / "unlabeled")
    return next_image_number(path for folder in folders for path in image_files(folder))


def choose_reason(confidences: list[float], now: float, last_reason_saved: dict[str, float], options: CollectionOptions) -> str | None:
    if not confidences:
        return None
    if any(options.review_confidence_low <= value <= options.review_confidence_high for value in confidences):
        return "uncertain"
    if any(value < options.review_confidence_low for value in confidences):
        reason, interval = "weak", options.weak_sample_seconds
    else:
        reason, interval = "high_confidence_audit", options.high_confidence_sample_seconds
    return reason if now - last_reason_saved[reason] >= interval else None


def save_candidate(output_dir: Path, frame, metadata: dict | None, number: int,
                   used_numbers: set[int], *, jpeg_quality: int, check_folders=True) -> tuple[Path, int]:
    encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not encoded_ok:
        raise RuntimeError("Could not encode the captured frame")
    folders = [output_dir.parent / split / "images" for split in ("train", "valid", "test", "labeled")]
    folders.append(output_dir)
    while True:
        while number in used_numbers or (check_folders and any(
            path.suffix.lower() in IMAGE_SUFFIXES
            for folder in folders if folder.is_dir() for path in folder.glob(f"{number}.*")
        )):
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
        if metadata is not None:
            save_review_metadata(image_path, metadata)
    except Exception:
        image_path.unlink(missing_ok=True)
        raise
    used_numbers.add(number)
    return image_path, number + 1


class SampleWriter:
    """Synchronous selection and persistence shared by active and live collection."""
    def __init__(self, checkpoint: Path, model_names: dict[int, str], options: CollectionOptions) -> None:
        self.options = options
        if (self.options.inference_fps <= 0 or self.options.max_saves_per_session <= 0
                or self.options.min_seconds_between_saves < 0 or self.options.image_size <= 0):
            raise ValueError("Active collector FPS, save limit, interval, and image size must be positive")
        if not (0 <= self.options.prediction_confidence < self.options.review_confidence_low
                < self.options.review_confidence_high <= 1):
            raise ValueError("Active collector confidence settings must increase from prediction to review high")
        dataset_dir = Path(self.options.dataset_dir).expanduser().resolve()
        if not (dataset_dir / "labeler.yaml").is_file() and not (dataset_dir / "data.yaml").is_file():
            raise FileNotFoundError(f"No dataset metadata in {dataset_dir}; open the dataset in labeler.py first")
        self.project = load_project(dataset_dir)
        self.names = model_names
        self.prediction_confidence = self.options.prediction_confidence
        if not set(model_names.values()).issubset(self.project.names):
            raise ValueError(
                f"Checkpoint classes {model_names} do not match dataset classes {self.project.names} "
                f"in {self.project.root}"
            )
        self.output_dir = self.project.root / "unlabeled"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.next_number, self.used_numbers = starting_number(self.project.root)
        checkpoint_stat = checkpoint.stat()
        self.checkpoint_identity = {
            "path": str(checkpoint),
            "size": checkpoint_stat.st_size,
            "modified_ns": checkpoint_stat.st_mtime_ns,
        }
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        started_at = time.monotonic()
        self.last_reason_saved = {
            "weak": started_at,
            "high_confidence_audit": started_at,
            "uncertain": float("-inf"),
        }
        self.last_saved_at = float("-inf")
        self.recent_hashes: deque[int] = deque(maxlen=self.options.recent_hash_count)
        self.saved_count = 0

    def save(self, frame: np.ndarray, rows: tuple[tuple[float, ...], ...], captured_at: float, *, manual=False, extra_metadata=None) -> None:
        if self.saved_count >= self.options.max_saves_per_session:
            return
        height, width = frame.shape[:2]
        draft_boxes = []
        for row in rows:
            x1, y1, x2, y2, confidence, class_id = map(float, row)
            if confidence < self.options.prediction_confidence:
                continue
            class_name = self.names.get(int(class_id))
            if class_name is None:
                continue
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            draft_boxes.append({
                "class_name": class_name,
                "confidence": float(confidence),
                "xywhn": [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                           (x2 - x1) / width, (y2 - y1) / height],
            })
        reason = "manual" if manual else choose_reason(
            [box["confidence"] for box in draft_boxes], captured_at, self.last_reason_saved, self.options
        )
        if not manual and (reason is None or captured_at - self.last_saved_at < self.options.min_seconds_between_saves):
            return
        interpolation = cv2.INTER_AREA if max(frame.shape[:2]) > self.options.image_size else cv2.INTER_LINEAR
        if frame.shape[:2] != (self.options.image_size, self.options.image_size):
            frame = cv2.resize(frame, (self.options.image_size, self.options.image_size),
                               interpolation=interpolation)
        frame_hash = difference_hash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not manual and any(hash_distance(frame_hash, previous) <= self.options.duplicate_hash_distance
               for previous in self.recent_hashes):
            return
        metadata = {
            "schema_version": 1,
            "session_id": self.session_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "selection_reason": reason,
            "checkpoint": self.checkpoint_identity,
            "dhash": f"{frame_hash:016x}",
            "boxes": draft_boxes,
        }
        metadata.update(extra_metadata or {})
        image_path, self.next_number = save_candidate(
            self.output_dir, frame, metadata, self.next_number, self.used_numbers, jpeg_quality=self.options.jpeg_quality
        )
        self.recent_hashes.append(frame_hash)
        self.last_saved_at = captured_at
        if not manual:
            self.last_reason_saved[reason] = captured_at
        self.saved_count += 1
        print(f"Saved {image_path.name} ({reason}, {len(draft_boxes)} draft boxes; "
              f"{self.saved_count}/{self.options.max_saves_per_session})")


class BackgroundCollector(SampleWriter):
    """Keep at most one copied frame waiting for the disk writer."""
    def __init__(self, checkpoint, model_names, options):
        super().__init__(checkpoint, model_names, options)
        self.next_due_ns = 0
        self.period_ns = round(1_000_000_000 / options.inference_fps)
        self.pending = Queue(maxsize=1)
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._run, name="inference-data-writer", daemon=True)

    def start(self) -> None:
        self.worker.start()
        print(f"Data collection enabled: {self.output_dir} ({self.options.inference_fps:g} sample/s).")

    def due(self, now_ns: int) -> bool:
        return (not self.stopping.is_set() and self.saved_count < self.options.max_saves_per_session
                and now_ns >= self.next_due_ns and not self.pending.full())

    def submit(self, frame: np.ndarray, rows: tuple[tuple[float, ...], ...], now_ns: int) -> None:
        """Copy only sampled frames; drop a sample if disk processing is still busy."""
        self.next_due_ns = now_ns + self.period_ns
        if self.stopping.is_set() or self.pending.full():
            return
        try:
            self.pending.put_nowait((frame.copy(), rows, time.monotonic()))
        except Full:
            pass

    def _run(self) -> None:
        while not self.stopping.is_set() or not self.pending.empty():
            try:
                frame, rows, captured_at = self.pending.get(timeout=0.1)
            except Empty:
                continue
            try:
                self.save(frame, rows, captured_at)
            except Exception as exc:
                print(f"Data collection stopped: {exc}")
                self.stopping.set()
                break
            finally:
                self.pending.task_done()

    def close(self) -> None:
        self.stopping.set()
        self.worker.join()
        print(f"Data collection saved {self.saved_count} image(s) in {self.output_dir}.")
