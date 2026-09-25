"""Create versioned YOLO datasets from reviewed images and labels."""

import hashlib
import math
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from review_metadata import (
    SIMILARITY_MAX_DISTANCE, hash_distance, image_difference_hash, load_review_metadata,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")
SPLIT_MANIFEST = "split_assignments.yaml"


@dataclass(frozen=True)
class GenerateConfig:
    included_class_ids: tuple[int, ...]
    train_percent: int = 80
    valid_percent: int = 15
    augment_copies: int = 1
    horizontal_flip: bool = True
    rotation_degrees: int = 0
    brightness_percent: int = 15
    contrast_percent: int = 15
    blur_radius: float = 0.0


@dataclass(frozen=True)
class CollectedImage:
    source_name: str
    image_path: Path
    boxes: list[tuple[int, float, float, float, float]]
    content_hash: str
    dhash: int
    session_id: str | None
    include_in_version: bool

    @property
    def captured(self) -> bool:
        return self.session_id is not None


def _load_split_manifest(dataset_dir: Path) -> dict | None:
    manifest_path = dataset_dir / SPLIT_MANIFEST
    versions_dir = dataset_dir / "versions"
    versions = sorted(
        (path for path in versions_dir.glob("v*/" + SPLIT_MANIFEST)
         if path.parent.name[1:].isdigit()),
        key=lambda path: int(path.parent.name[1:]), reverse=True,
    ) if versions_dir.is_dir() else []
    if versions and (not manifest_path.is_file() or versions[0].stat().st_mtime_ns > manifest_path.stat().st_mtime_ns):
        manifest_path = versions[0]
    if not manifest_path.is_file():
        return None
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("assignments"), dict):
        raise ValueError(f"Invalid split assignments: {manifest_path}")
    try:
        train_percent, valid_percent = int(data["train_percent"]), int(data["valid_percent"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid split percentages in {manifest_path}") from exc
    if not 1 <= train_percent <= 98 or not 1 <= valid_percent <= 98 or train_percent + valid_percent >= 100:
        raise ValueError(f"Invalid split percentages in {manifest_path}")
    for entry in data["assignments"].values():
        if not isinstance(entry, dict) or entry.get("split") not in SPLITS or not isinstance(entry.get("dhash"), str):
            raise ValueError(f"Invalid split assignment in {manifest_path}")
        try:
            int(entry["dhash"], 16)
        except ValueError as exc:
            raise ValueError(f"Invalid image hash in {manifest_path}") from exc
    return data


def get_split_percentages(dataset_dir: Path) -> tuple[int, int] | None:
    manifest = _load_split_manifest(dataset_dir)
    return (int(manifest["train_percent"]), int(manifest["valid_percent"])) if manifest else None


def _initial_split_assignments(images: list[CollectedImage], config: GenerateConfig) -> dict[str, dict]:
    def visually_distinct_groups(candidates: list[CollectedImage]) -> list[list[CollectedImage]]:
        parents = list(range(len(candidates)))
        session_representatives: dict[str, int] = {}

        def root(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        for left, image in enumerate(candidates):
            if image.session_id:
                previous = session_representatives.setdefault(image.session_id, left)
                parents[root(left)] = root(previous)
            for right in range(left):
                if hash_distance(image.dhash, candidates[right].dhash) <= SIMILARITY_MAX_DISTANCE:
                    parents[root(left)] = root(right)
        groups: dict[int, list[CollectedImage]] = {}
        for index, image in enumerate(candidates):
            groups.setdefault(root(index), []).append(image)
        return list(groups.values())

    legacy = [image for image in images if not image.captured]
    candidates = legacy if len(legacy) >= 3 else images
    group_list = visually_distinct_groups(candidates)
    if len(group_list) < 3 and candidates is not images:
        candidates = images
        group_list = visually_distinct_groups(candidates)
    if len(group_list) < 3:
        raise ValueError("At least three visually distinct reviewed image groups are needed for train, validation, and test")

    rng = random.Random(42)
    rng.shuffle(group_list)
    group_list.sort(key=len)
    test_target = max(1, round(len(candidates) * (100 - config.train_percent - config.valid_percent) / 100))
    valid_target = max(1, round(len(candidates) * config.valid_percent / 100))
    counts = {split: 0 for split in SPLITS}
    assignments = {}
    for index, group in enumerate(group_list):
        remaining_groups = len(group_list) - index
        if counts["test"] < test_target and remaining_groups > 2:
            split = "test"
        elif counts["valid"] < valid_target and remaining_groups > 1:
            split = "valid"
        else:
            split = "train"
        for image in group:
            assignments[image.content_hash] = {"split": split, "dhash": f"{image.dhash:016x}",
                                               "session_id": image.session_id}
        counts[split] += len(group)
    if not all(counts.values()):
        raise ValueError("Could not make three nonempty, visually separate splits")
    return assignments


def _assign_splits(dataset_dir: Path, images: list[CollectedImage], config: GenerateConfig):
    manifest = _load_split_manifest(dataset_dir)
    if manifest is None:
        assignments = _initial_split_assignments(images, config)
        manifest = {"schema_version": 1, "train_percent": config.train_percent,
                    "valid_percent": config.valid_percent, "assignments": assignments}
    else:
        if (int(manifest["train_percent"]), int(manifest["valid_percent"])) != (
            config.train_percent, config.valid_percent
        ):
            raise ValueError("Split percentages are frozen. Use the saved percentages shown in the generator dialog")
        assignments = manifest["assignments"].copy()
        manifest = {**manifest, "assignments": assignments}

    references = [(int(entry["dhash"], 16), entry["split"]) for entry in assignments.values()]
    split_images = {split: [] for split in SPLITS}
    conflicts = 0
    for image in images:
        entry = assignments.get(image.content_hash)
        if entry is None:
            nearby = {split for dhash, split in references
                      if hash_distance(image.dhash, dhash) <= SIMILARITY_MAX_DISTANCE}
            if len(nearby) > 1:
                conflicts += 1
                continue
            split = nearby.pop() if nearby else "train"
            entry = {"split": split, "dhash": f"{image.dhash:016x}", "session_id": image.session_id}
            assignments[image.content_hash] = entry
            references.append((image.dhash, split))
        split_images[entry["split"]].append(image)
    if any(not split_images[split] for split in SPLITS):
        raise ValueError("The fixed split has no images in train, validation, or test")
    return split_images, manifest, conflicts


def _read_boxes(path: Path, class_count: int) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("expected five values")
            class_id = int(parts[0])
            cx, cy, width, height = map(float, parts[1:])
            if not 0 <= class_id < class_count or not all(
                math.isfinite(value) and 0 <= value <= 1 for value in (cx, cy, width, height)
            ) or width <= 0 or height <= 0 or not (
                0 <= cx - width / 2 < cx + width / 2 <= 1
                and 0 <= cy - height / 2 < cy + height / 2 <= 1
            ):
                raise ValueError("class or box coordinates out of range")
            boxes.append((class_id, cx, cy, width, height))
        except ValueError as exc:
            raise ValueError(f"{path}, line {line_number}: {exc}") from exc
    return boxes


def _sources(dataset_dir: Path):
    image_dir = dataset_dir / "labeled" / "images"
    label_dir = dataset_dir / "labeled" / "labels"
    if not image_dir.is_dir():
        return
    for image_path in sorted(image_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                yield "labeled", image_path, label_path


def _image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_boxes(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
             for class_id, cx, cy, width, height in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _rotate_boxes(boxes, angle: float, image_width: int, image_height: int):
    theta = math.radians(angle)
    cosine, sine = math.cos(theta), math.sin(theta)
    rotated = []
    for class_id, cx, cy, width, height in boxes:
        x1, x2 = (cx - width / 2) * image_width, (cx + width / 2) * image_width
        y1, y2 = (cy - height / 2) * image_height, (cy + height / 2) * image_height
        points = []
        for x in (x1, x2):
            for y in (y1, y2):
                dx, dy = x - image_width / 2, y - image_height / 2
                points.append((image_width / 2 + cosine * dx + sine * dy,
                               image_height / 2 - sine * dx + cosine * dy))
        left, right = min(x for x, _ in points), max(x for x, _ in points)
        top, bottom = min(y for _, y in points), max(y for _, y in points)
        clipped_left, clipped_right = max(0, left), min(image_width, right)
        clipped_top, clipped_bottom = max(0, top), min(image_height, bottom)
        original_area = (right - left) * (bottom - top)
        clipped_area = max(0, clipped_right - clipped_left) * max(0, clipped_bottom - clipped_top)
        if original_area <= 0 or clipped_area / original_area < 0.5:
            continue
        rotated.append((class_id,
                        (clipped_left + clipped_right) / (2 * image_width),
                        (clipped_top + clipped_bottom) / (2 * image_height),
                        (clipped_right - clipped_left) / image_width,
                        (clipped_bottom - clipped_top) / image_height))
    return rotated


def _augment(image: Image.Image, boxes, config: GenerateConfig, rng: random.Random):
    enabled = []
    if config.horizontal_flip:
        enabled.append("flip")
    if config.rotation_degrees:
        enabled.append("rotate")
    if config.brightness_percent:
        enabled.append("brightness")
    if config.contrast_percent:
        enabled.append("contrast")
    if config.blur_radius:
        enabled.append("blur")
    selected = {name for name in enabled if rng.random() < 0.5}
    if not selected:
        selected.add(rng.choice(enabled))
    boxes = list(boxes)
    if "flip" in selected:
        image = ImageOps.mirror(image)
        boxes = [(class_id, 1 - cx, cy, width, height) for class_id, cx, cy, width, height in boxes]
    if "rotate" in selected:
        angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
        boxes = _rotate_boxes(boxes, angle, *image.size)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC,
                             fillcolor=(17, 24, 39))
    if "brightness" in selected:
        strength = config.brightness_percent / 100
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(1 - strength, 1 + strength))
    if "contrast" in selected:
        strength = config.contrast_percent / 100
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(1 - strength, 1 + strength))
    if "blur" in selected:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, config.blur_radius)))
    return image, boxes


def generate_dataset(
    dataset_dir: Path, class_names: list[str], config: GenerateConfig,
    progress: Callable[[str], None] | None = None,
    class_colors: list[str] | None = None,
) -> tuple[Path, dict]:
    """Build a new immutable version; existing source images and versions are untouched."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if not config.included_class_ids or len(set(config.included_class_ids)) != len(config.included_class_ids):
        raise ValueError("Select at least one unique class")
    if any(not 0 <= class_id < len(class_names) for class_id in config.included_class_ids):
        raise ValueError("Selected class is out of range")
    if not 1 <= config.train_percent <= 98 or not 1 <= config.valid_percent <= 98 or config.train_percent + config.valid_percent >= 100:
        raise ValueError("Train and validation percentages must leave room for test")
    if config.augment_copies < 0 or config.augment_copies > 10:
        raise ValueError("Augmented copies must be between 0 and 10")
    if config.augment_copies and not any((config.horizontal_flip, config.rotation_degrees,
                                          config.brightness_percent, config.contrast_percent,
                                          config.blur_radius)):
        raise ValueError("Choose an augmentation or set augmented copies to zero")
    for value in (config.rotation_degrees, config.brightness_percent, config.contrast_percent, config.blur_radius):
        if value < 0:
            raise ValueError("Augmentation strengths cannot be negative")

    remap = {old_id: new_id for new_id, old_id in enumerate(config.included_class_ids)}
    collected = []
    seen_hashes = set()
    excluded_only = 0
    duplicates = 0
    scanned = 0
    for source_name, image_path, label_path in _sources(dataset_dir):
        scanned += 1
        if progress and scanned % 50 == 0:
            progress(f"Reading source images and labels: {scanned} scanned...")
        original_boxes = _read_boxes(label_path, len(class_names))
        boxes = [(remap[class_id], cx, cy, width, height)
                 for class_id, cx, cy, width, height in original_boxes if class_id in remap]
        include_in_version = not (original_boxes and not boxes)
        if not include_in_version:
            excluded_only += 1
        image_hash = _image_hash(image_path)
        if image_hash in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(image_hash)
        review_data = load_review_metadata(image_path)
        session_id = review_data.get("session_id") if review_data else None
        if session_id is not None and not isinstance(session_id, str):
            raise ValueError(f"Invalid capture session for {image_path}")
        collected.append(CollectedImage(
            source_name, image_path, boxes, image_hash, image_difference_hash(image_path),
            session_id or None, include_in_version,
        ))
    if len(collected) < 3:
        raise ValueError("At least three paired labeled images are needed for train, validation, and test")

    versions_dir = dataset_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    version_number = 1
    while (versions_dir / f"v{version_number}").exists():
        version_number += 1
    version_path = versions_dir / f"v{version_number}"
    split_images, split_manifest, split_conflicts = _assign_splits(dataset_dir, collected, config)
    for split in SPLITS:
        split_images[split] = [image for image in split_images[split] if image.include_in_version]
    if any(not split_images[split] for split in SPLITS):
        raise ValueError("Selected classes leave train, validation, or test without any labeled images")
    rng = random.Random(42)
    for items in split_images.values():
        rng.shuffle(items)
    split_counts = {name: len(items) for name, items in split_images.items()}
    total_split_images = sum(split_counts.values())
    generated_count = 0
    if progress:
        progress(f"Building {version_path.name} from {total_split_images} source images...")
    with tempfile.TemporaryDirectory(prefix=".building-", dir=versions_dir) as temporary:
        build_dir = Path(temporary)
        for split in SPLITS:
            (build_dir / split / "images").mkdir(parents=True)
            (build_dir / split / "labels").mkdir()
        processed = 0
        for split, items in split_images.items():
            for item in items:
                source_name, image_path, boxes = item.source_name, item.image_path, item.boxes
                identifier = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:8]
                stem = f"{source_name}__{image_path.stem}__{identifier}"
                image_target = build_dir / split / "images" / f"{stem}{image_path.suffix.lower()}"
                label_target = build_dir / split / "labels" / f"{stem}.txt"
                shutil.copy2(image_path, image_target)
                _write_boxes(label_target, boxes)
                if split == "train" and config.augment_copies:
                    with Image.open(image_path) as original:
                        original = original.convert("RGB")
                        for copy_index in range(1, config.augment_copies + 1):
                            for _ in range(5):
                                augmented, augmented_boxes = _augment(original.copy(), boxes, config, rng)
                                if not boxes or augmented_boxes:
                                    break
                            if boxes and not augmented_boxes:
                                continue
                            augmented_stem = f"{stem}__aug{copy_index}"
                            augmented.save(build_dir / split / "images" / f"{augmented_stem}.jpg",
                                           format="JPEG", quality=95)
                            _write_boxes(build_dir / split / "labels" / f"{augmented_stem}.txt",
                                         augmented_boxes)
                            generated_count += 1
                processed += 1
                if progress and (processed % 20 == 0 or processed == total_split_images):
                    progress(f"Processed {processed}/{total_split_images} images...")
        selected_names = [class_names[class_id] for class_id in config.included_class_ids]
        data = {"path": str(version_path), "train": "train/images", "val": "valid/images",
                "test": "test/images", "nc": len(selected_names), "names": selected_names}
        (build_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        if class_colors is not None:
            selected_classes = [{"name": class_names[class_id], "color": class_colors[class_id]}
                                for class_id in config.included_class_ids]
            labeler_data = {"schema_version": 1, "classes": selected_classes,
                            "last_folder": "train/images", "last_image": None, "active_class": 0,
                            "generator_settings": None}
            (build_dir / "labeler.yaml").write_text(
                yaml.safe_dump(labeler_data, sort_keys=False), encoding="utf-8")
        metadata = {"source_images": total_split_images, "split_images": split_counts,
                    "augmented_train_images": generated_count, "excluded_only_images": excluded_only,
                    "duplicate_images": duplicates, "split_conflicts_skipped": split_conflicts,
                    "seed": 42, "settings": asdict(config)}
        (build_dir / "generation.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        (build_dir / SPLIT_MANIFEST).write_text(
            yaml.safe_dump(split_manifest, sort_keys=False), encoding="utf-8")
        build_dir.rename(version_path)
    manifest_path = dataset_dir / SPLIT_MANIFEST
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dataset_dir,
                                         prefix=".split-", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            yaml.safe_dump(split_manifest, temporary, sort_keys=False)
        temporary_path.replace(manifest_path)
    except OSError as exc:
        # The version contains a copy, so the next generation can recover from it.
        metadata["split_manifest_warning"] = str(exc)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if progress:
        progress(f"Created {version_path}")
    return version_path, metadata
