"""Draft prediction metadata shared by the live collector, labeler, and generator."""

import json
from pathlib import Path

from PIL import Image

from dataset_utils import (atomic_write)


SCHEMA_VERSION = 1
SIMILARITY_MAX_DISTANCE = 5


def metadata_path(image_path: Path) -> Path:
    return image_path.parent / ".review" / f"{image_path.stem}.json"


def difference_hash(image: Image.Image) -> int:
    pixels = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def image_difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        return difference_hash(image)


def hash_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def load_review_metadata(image_path: Path) -> dict | None:
    path = metadata_path(image_path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Invalid review metadata: {path}")
    if not isinstance(data.get("boxes"), list):
        raise ValueError(f"Missing draft boxes in {path}")
    return data


def save_review_metadata(image_path: Path, data: dict) -> None:
    atomic_write(metadata_path(image_path), json.dumps(data, indent=2) + "\n")
