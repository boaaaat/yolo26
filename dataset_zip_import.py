"""Import paired images, YOLO labels, and collector drafts from a ZIP archive."""

import io
import json
import math
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IGNORED_SOURCE_FOLDERS = {"versions", ".trash", ".invalid-labels", ".imports", "__macosx"}
MAX_IMAGE_BYTES = 150 * 1024 * 1024
MAX_SIDECAR_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ImportResult:
    labeled: int
    unlabeled: int
    collector_drafts: int
    issues: int
    skipped_images: int
    assumed_class_ids: int
    manifest_path: Path


def _key(path: PurePosixPath) -> str:
    return "/".join(part.casefold() for part in path.parts)


def _entries(archive: zipfile.ZipFile) -> dict[str, tuple[PurePosixPath, zipfile.ZipInfo]]:
    entries = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if (name.startswith("/") or not path.parts or
                any(part == ".." or ":" in part for part in path.parts)):
            raise ValueError(f"Unsafe path in ZIP: {info.filename}")
        key = _key(path)
        if key in entries:
            raise ValueError(f"Duplicate path in ZIP: {info.filename}")
        entries[key] = path, info
    return entries


def _read_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    if info.file_size > limit:
        raise ValueError(f"File is too large in ZIP: {info.filename}")
    with archive.open(info) as source:
        content = source.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"File is too large in ZIP: {info.filename}")
    return content


def _sidecar(entries: dict, image_path: PurePosixPath, suffix: str):
    candidates = [image_path.with_suffix(suffix)]
    parts = list(image_path.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index].casefold() == "images":
            replacement = parts.copy()
            replacement[index] = "labels"
            candidates.append(PurePosixPath(*replacement).with_suffix(suffix))
            break
    candidates.append(image_path.parent / "labels" / f"{image_path.stem}{suffix}")
    return next((entries[_key(path)] for path in candidates if _key(path) in entries), None)


def _review_entry(entries: dict, image_path: PurePosixPath):
    review_path = image_path.parent / ".review" / f"{image_path.stem}.json"
    return entries.get(_key(review_path))


def _source_class_names(archive: zipfile.ZipFile, entries: dict,
                        image_path: PurePosixPath, cache: dict[str, list[str]]) -> list[str] | None:
    for parent in (image_path.parent, *image_path.parents):
        for filename in ("labeler.yaml", "data.yaml"):
            metadata_key = _key(parent / filename)
            entry = entries.get(metadata_key)
            if entry is None:
                continue
            if metadata_key in cache:
                return cache[metadata_key]
            data = yaml.safe_load(_read_entry(archive, entry[1], MAX_SIDECAR_BYTES))
            if not isinstance(data, dict):
                raise ValueError(f"Invalid class metadata in {entry[1].filename}")
            if filename == "labeler.yaml":
                classes = data.get("classes")
                names = ([item.get("name") for item in classes]
                         if isinstance(classes, list) and all(isinstance(item, dict) for item in classes)
                         else None)
            else:
                names = data.get("names")
                if isinstance(names, dict):
                    try:
                        names = [value for _, value in sorted(names.items(), key=lambda pair: int(pair[0]))]
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid class IDs in {entry[1].filename}") from exc
            if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
                raise ValueError(f"Invalid class names in {entry[1].filename}")
            cache[metadata_key] = names
            return names
    return None


def _convert_label(content: bytes, source_names: list[str] | None,
                   target_names: list[str]) -> bytes:
    target_ids = {name.casefold(): index for index, name in enumerate(target_names)}
    lines = []
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Label line {line_number}: expected a class and four coordinates")
        try:
            class_id = int(parts[0])
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"Label line {line_number}: invalid number") from exc
        if (class_id < 0 or not all(math.isfinite(value) for value in (cx, cy, width, height)) or
                not (0 <= cx <= 1 and 0 <= cy <= 1 and width > 0 and height > 0 and
                     0 <= cx - width / 2 < cx + width / 2 <= 1 and
                     0 <= cy - height / 2 < cy + height / 2 <= 1)):
            raise ValueError(f"Label line {line_number}: class or box coordinates out of range")
        if source_names is None:
            if class_id >= len(target_names):
                raise ValueError(f"Label line {line_number}: class {class_id} is not in this dataset")
            mapped_id = class_id
        else:
            if class_id >= len(source_names):
                raise ValueError(f"Label line {line_number}: class {class_id} is missing from source metadata")
            mapped_id = target_ids.get(source_names[class_id].casefold())
            if mapped_id is None:
                raise ValueError(f"Label line {line_number}: class {source_names[class_id]!r} is not in this dataset")
        lines.append(f"{mapped_id} {' '.join(parts[1:])}")
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _convert_review(content: bytes, target_names: list[str]) -> tuple[bytes, list[str]]:
    data = json.loads(content.decode("utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("boxes"), list):
        raise ValueError("Invalid collector review metadata")
    if data.get("session_id") is not None and not isinstance(data["session_id"], str):
        raise ValueError("Invalid collector session ID")
    target_by_name = {name.casefold(): name for name in target_names}
    unknown_names = set()
    for draft in data["boxes"]:
        if not isinstance(draft, dict) or not isinstance(draft.get("class_name"), str):
            raise ValueError("Invalid collector draft class")
        coords = draft.get("xywhn")
        confidence = draft.get("confidence")
        if (not isinstance(coords, list) or len(coords) != 4 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in coords) or
                (confidence is not None and (type(confidence) not in (int, float) or
                                             not math.isfinite(confidence) or not 0 <= confidence <= 1))):
            raise ValueError("Invalid collector draft coordinates or confidence")
        cx, cy, width, height = coords
        if not (width > 0 and height > 0 and
                -0.0001 <= cx - width / 2 < cx + width / 2 <= 1.0001 and
                -0.0001 <= cy - height / 2 < cy + height / 2 <= 1.0001):
            raise ValueError("Collector draft box is out of range")
        canonical = target_by_name.get(draft["class_name"].casefold())
        if canonical is None:
            unknown_names.add(draft["class_name"])
        else:
            draft["class_name"] = canonical
    return (json.dumps(data, indent=2) + "\n").encode("utf-8"), sorted(unknown_names)


def import_dataset_zip(archive_path: Path, dataset_dir: Path, class_names: list[str],
                       progress: Callable[[int, int], None] | None = None) -> ImportResult:
    """Stage a ZIP import, then commit all renamed image groups together."""
    token = uuid.uuid4().hex[:12]
    labeled = unlabeled = collector_drafts = issues = skipped = assumed = 0
    records = []
    source_names_cache: dict[str, list[str]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        entries = _entries(archive)
        images = sorted((entry for entry in entries.values()
                         if entry[0].suffix.casefold() in IMAGE_SUFFIXES and
                         not any(part.casefold() in IGNORED_SOURCE_FOLDERS for part in entry[0].parts[:-1])),
                        key=lambda entry: _key(entry[0]))
        if not images:
            raise ValueError("The ZIP contains no supported images")
        if progress is not None:
            progress(0, len(images))
        with tempfile.TemporaryDirectory(prefix=".zip-import-", dir=dataset_dir) as temporary:
            stage = Path(temporary)
            staged_paths: list[Path] = []

            def stage_file(relative: Path, content: bytes) -> None:
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                staged_paths.append(relative)

            for index, (image_path, image_info) in enumerate(images, 1):
                try:
                    image_bytes = _read_entry(archive, image_info, MAX_IMAGE_BYTES)
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    skipped += 1
                    records.append({"source": image_info.filename, "skipped": str(exc)})
                    if progress is not None:
                        progress(index, len(images))
                    continue

                label_entry = _sidecar(entries, image_path, ".txt")
                review_entry = _review_entry(entries, image_path)
                parts = {part.casefold() for part in image_path.parts[:-1]}
                explicit_unlabeled = "unlabeled" in parts
                explicit_labeled = bool(parts & {"labeled", "train", "valid", "val", "test"})
                destination = ("unlabeled" if explicit_unlabeled else
                               "labeled" if explicit_labeled else
                               "unlabeled" if review_entry is not None else
                               "labeled" if label_entry is not None else "unlabeled")
                stem = f"import_{token}_{index:06d}"
                suffix = image_path.suffix.lower()
                issue_messages = []
                label_content = None
                original_label = None
                if label_entry is not None:
                    try:
                        original_label = _read_entry(archive, label_entry[1], MAX_SIDECAR_BYTES)
                        source_names = _source_class_names(archive, entries, image_path, source_names_cache)
                        label_content = _convert_label(original_label, source_names, class_names)
                        if source_names is None and label_content:
                            assumed += 1
                    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                        issue_messages.append(f"Imported label {label_entry[1].filename}: {exc}")
                        destination = "unlabeled"
                elif destination == "labeled":
                    issue_messages.append("Image came from a labeled folder without a matching .txt label")
                    destination = "unlabeled"
                if label_entry is None and explicit_unlabeled:
                    invalid_dir = image_path.parent / ".invalid-labels"
                    backup_entry = entries.get(_key(invalid_dir / f"{image_path.stem}.txt"))
                    source_issue = entries.get(_key(invalid_dir / f"{image_path.stem}.issue.txt"))
                    if backup_entry is not None or source_issue is not None:
                        destination = "unlabeled"
                        if backup_entry is not None:
                            try:
                                original_label = _read_entry(archive, backup_entry[1], MAX_SIDECAR_BYTES)
                            except (OSError, ValueError) as exc:
                                issue_messages.append(f"Could not preserve source invalid label: {exc}")
                        if source_issue is not None:
                            try:
                                source_note = _read_entry(archive, source_issue[1], MAX_SIDECAR_BYTES)
                                issue_messages.append("Source issue: " + source_note.decode("utf-8-sig").strip())
                            except (OSError, UnicodeError, ValueError) as exc:
                                issue_messages.append(f"Could not read source issue note: {exc}")
                        elif backup_entry is not None:
                            issue_messages.append("Source had an invalid label saved for review")

                review_content = None
                original_review = None
                if review_entry is not None:
                    try:
                        original_review = _read_entry(archive, review_entry[1], MAX_SIDECAR_BYTES)
                        review_content, unknown_names = _convert_review(original_review, class_names)
                        collector_drafts += 1
                        if unknown_names and destination == "unlabeled":
                            issue_messages.append("Collector draft classes not in this dataset: " +
                                                  ", ".join(unknown_names))
                    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                        issue_messages.append(f"Imported review {review_entry[1].filename}: {exc}")
                        destination = "unlabeled"

                if destination == "labeled":
                    image_relative = Path("labeled", "images", stem + suffix)
                    label_relative = Path("labeled", "labels", stem + ".txt")
                    labeled += 1
                else:
                    image_relative = Path("unlabeled", stem + suffix)
                    label_relative = Path("unlabeled", stem + ".txt")
                    unlabeled += 1
                stage_file(image_relative, image_bytes)
                if label_content is not None:
                    stage_file(label_relative, label_content)
                if review_content is not None:
                    review_relative = (Path("unlabeled", ".review", stem + ".json") if destination == "unlabeled"
                                       else Path("labeled", "images", ".review", stem + ".json"))
                    stage_file(review_relative, review_content)
                if issue_messages:
                    issues += 1
                    invalid_dir = Path("unlabeled", ".invalid-labels")
                    stage_file(invalid_dir / f"{stem}.issue.txt", ("\n".join(issue_messages) + "\n").encode("utf-8"))
                    if original_label is not None and label_content is None:
                        stage_file(invalid_dir / f"{stem}.txt", original_label)
                    if original_review is not None and review_content is None:
                        stage_file(invalid_dir / f"{stem}.review.json", original_review)
                records.append({"source": image_info.filename,
                                "destination": image_relative.as_posix(),
                                "label_source": label_entry[1].filename if label_entry else None,
                                "review_source": review_entry[1].filename if review_entry else None,
                                "issues": issue_messages})
                if progress is not None:
                    progress(index, len(images))

            manifest_relative = Path(".imports", f"{token}.json")
            stage_file(manifest_relative, (json.dumps({"archive": str(archive_path),
                                                      "images": records}, indent=2) + "\n").encode("utf-8"))
            committed = []
            try:
                for relative in staged_paths:
                    destination_path = dataset_dir / relative
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    if destination_path.exists():
                        raise FileExistsError(f"Import destination already exists: {destination_path}")
                    (stage / relative).rename(destination_path)
                    committed.append(destination_path)
            except OSError:
                for path in reversed(committed):
                    path.unlink(missing_ok=True)
                raise
    return ImportResult(labeled, unlabeled, collector_drafts, issues, skipped, assumed,
                        dataset_dir / manifest_relative)
