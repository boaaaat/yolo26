"""Shared dataset primitives; no model, capture, or UI dependencies."""

import math
import os
import tempfile
from pathlib import Path


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def image_files(folder: Path, *, missing_ok=True):
    if not folder.is_dir():
        if missing_ok:
            return
        raise FileNotFoundError(f"Image folder not found: {folder}")
    yield from (path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def atomic_write(path: Path, content: str | bytes) -> None:
    """Replace one file using a temporary file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        mode = "wb" if isinstance(content, bytes) else "w"
        kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
        with os.fdopen(descriptor, mode, **kwargs) as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def normalize_names(names) -> list[str]:
    """Read YAML/model class names in numeric class-ID order."""
    if isinstance(names, dict):
        names = [value for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("Expected a nonempty ordered list of class names")
    return list(names)


def class_name_map(names) -> dict[int, str]:
    """Normalize model metadata without renumbering an existing class-ID mapping."""
    if isinstance(names, dict):
        return {int(class_id): name for class_id, name in names.items()}
    return dict(enumerate(normalize_names(names)))


def label_class_ids(path: Path, *, decimal_only=False) -> frozenset[int]:
    """Read queue/filter IDs; full box validation happens when the image is opened."""
    if not path.is_file():
        return frozenset()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    result = set()
    for line in lines:
        parts = line.split()
        if not parts or (decimal_only and not parts[0].isdecimal()):
            continue
        try:
            result.add(int(parts[0]))
        except ValueError:
            continue
    return frozenset(result)


def parse_yolo_labels(text: str, class_count: int | None, *, source="labels", tolerance=0.0):
    """Return normalized class/cx/cy/w/h rows with an explicit edge tolerance."""
    tx, ty = tolerance if isinstance(tolerance, tuple) else (tolerance, tolerance)
    boxes = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("expected five values")
            class_id = int(parts[0])
            cx, cy, width, height = map(float, parts[1:])
            if class_id < 0 or (class_count is not None and class_id >= class_count):
                raise ValueError("class ID out of range")
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (cx, cy, width, height)):
                raise ValueError("coordinates must be finite and within 0–1")
            if (width <= 0 or height <= 0 or cx - width / 2 < -tx or cy - height / 2 < -ty
                    or cx + width / 2 > 1 + tx or cy + height / 2 > 1 + ty):
                raise ValueError("box extends outside image or has zero area")
            boxes.append((class_id, cx, cy, width, height))
        except ValueError as exc:
            raise ValueError(f"{source}, line {number}: {exc}") from exc
    return boxes


def read_yolo_labels(path: Path, class_count: int | None, *, missing_ok=False, tolerance=0.0):
    if missing_ok and not path.is_file():
        return []
    return parse_yolo_labels(path.read_text(encoding="utf-8"), class_count,
                             source=path, tolerance=tolerance)


def format_yolo_labels(boxes, *, precision=10):
    return "".join(f"{class_id} {cx:.{precision}f} {cy:.{precision}f} "
                   f"{width:.{precision}f} {height:.{precision}f}\n"
                   for class_id, cx, cy, width, height in boxes)


def write_yolo_labels(path: Path, boxes, *, precision=10):
    # Callers retain their own validation policy, including rounding/boundary tolerance.
    atomic_write(path, format_yolo_labels(boxes, precision=precision))


def prediction_rows(result):
    """Copy all detection boxes from the device once, returning native Python values."""
    return result.boxes.data.detach().cpu().tolist() if result.boxes is not None else []


def normalized_predictions(result):
    """Yield class/cx/cy/w/h/confidence after one device-to-host transfer."""
    height, width = result.orig_shape
    for x1, y1, x2, y2, confidence, class_id in prediction_rows(result):
        yield (int(class_id), (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
               (x2 - x1) / width, (y2 - y1) / height, confidence)
