"""Save active-learning samples from an inference bot without another model pass."""

import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
from PIL import Image

import active_collector
from dataset_project import load_project
from review_metadata import difference_hash, hash_distance


class BackgroundCollector:
    """Consume at most one copied inference frame at a time on a writer thread."""

    def __init__(self, checkpoint: Path, model_names: dict[int, str]) -> None:
        if (active_collector.INFERENCE_FPS <= 0 or active_collector.MAX_SAVES_PER_SESSION <= 0
                or active_collector.MIN_SECONDS_BETWEEN_SAVES < 0 or active_collector.IMAGE_SIZE <= 0):
            raise ValueError("Active collector FPS, save limit, interval, and image size must be positive")
        if not (0 <= active_collector.PREDICTION_CONFIDENCE < active_collector.REVIEW_CONFIDENCE_LOW
                < active_collector.REVIEW_CONFIDENCE_HIGH <= 1):
            raise ValueError("Active collector confidence settings must increase from prediction to review high")
        dataset_dir = Path(active_collector.DATASET_DIR).expanduser().resolve()
        if not (dataset_dir / "labeler.yaml").is_file() and not (dataset_dir / "data.yaml").is_file():
            raise FileNotFoundError(f"No dataset metadata in {dataset_dir}; open the dataset in labeler.py first")
        self.project = load_project(dataset_dir)
        self.names = model_names
        self.prediction_confidence = active_collector.PREDICTION_CONFIDENCE
        if not set(model_names.values()).issubset(self.project.names):
            raise ValueError(
                f"Checkpoint classes {model_names} do not match dataset classes {self.project.names} "
                f"in {self.project.root}"
            )
        self.output_dir = self.project.root / "unlabeled"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.next_number, self.used_numbers = active_collector.starting_number(self.project.root)
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
        self.recent_hashes: deque[int] = deque(maxlen=active_collector.RECENT_HASH_COUNT)
        self.saved_count = 0
        self.next_due_ns = 0
        self.period_ns = round(1_000_000_000 / active_collector.INFERENCE_FPS)
        self.pending: Queue[tuple[np.ndarray, tuple[tuple[float, ...], ...], float]] = Queue(maxsize=1)
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._run, name="inference-data-writer", daemon=True)

    def start(self) -> None:
        self.worker.start()
        print(f"Data collection enabled: {self.output_dir} ({active_collector.INFERENCE_FPS:g} sample/s).")

    def due(self, now_ns: int) -> bool:
        return (not self.stopping.is_set() and self.saved_count < active_collector.MAX_SAVES_PER_SESSION
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

    def _save_sample(self, frame: np.ndarray, rows: tuple[tuple[float, ...], ...], captured_at: float) -> None:
        if self.saved_count >= active_collector.MAX_SAVES_PER_SESSION:
            return
        height, width = frame.shape[:2]
        draft_boxes = []
        for x1, y1, x2, y2, confidence, class_id in rows:
            if confidence < active_collector.PREDICTION_CONFIDENCE:
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
        reason = active_collector.choose_reason(
            [box["confidence"] for box in draft_boxes], captured_at, self.last_reason_saved
        )
        if reason is None or captured_at - self.last_saved_at < active_collector.MIN_SECONDS_BETWEEN_SAVES:
            return
        interpolation = cv2.INTER_AREA if max(frame.shape[:2]) > active_collector.IMAGE_SIZE else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (active_collector.IMAGE_SIZE, active_collector.IMAGE_SIZE),
                           interpolation=interpolation)
        frame_hash = difference_hash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if any(hash_distance(frame_hash, previous) <= active_collector.DUPLICATE_HASH_DISTANCE
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
        image_path, self.next_number = active_collector.save_candidate(
            self.output_dir, frame, metadata, self.next_number, self.used_numbers
        )
        self.recent_hashes.append(frame_hash)
        self.last_saved_at = captured_at
        self.last_reason_saved[reason] = captured_at
        self.saved_count += 1
        print(f"Saved {image_path.name} ({reason}, {len(draft_boxes)} draft boxes; "
              f"{self.saved_count}/{active_collector.MAX_SAVES_PER_SESSION})")

    def _run(self) -> None:
        while not self.stopping.is_set() or not self.pending.empty():
            try:
                frame, rows, captured_at = self.pending.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._save_sample(frame, rows, captured_at)
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
