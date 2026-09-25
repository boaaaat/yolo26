"""Dataset-local class definitions and labeler session state."""

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_COLORS = ("#ff6b6b", "#ffc857", "#56d6a5", "#71b7ff", "#c995ff")
COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}\Z")
RECENT_DATASET_FILE = Path(__file__).resolve().parent / "labeler_recent.yaml"


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        temp_path = Path(temporary)
        if isinstance(content, bytes):
            temp_path.write_bytes(content)
        else:
            temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _names_from_data(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML: {path}")
    names = data.get("names")
    if isinstance(names, dict):
        names = [value for _, value in sorted(names.items(), key=lambda pair: int(pair[0]))]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Invalid class names in {path}")
    if data.get("nc", len(names)) != len(names):
        raise ValueError(f"Class count does not match names in {path}")
    return names


def _validate_classes(classes: list[dict]) -> list[dict]:
    if not isinstance(classes, list) or not classes:
        raise ValueError("A dataset needs at least one class")
    cleaned = []
    seen = set()
    for entry in classes:
        if not isinstance(entry, dict):
            raise ValueError("Invalid class entry in labeler.yaml")
        name = entry.get("name")
        color = entry.get("color")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Class names cannot be empty")
        name = name.strip()
        if name.casefold() in seen:
            raise ValueError(f"Duplicate class name: {name}")
        if not isinstance(color, str) or not COLOR_PATTERN.fullmatch(color):
            raise ValueError(f"Invalid color for class {name}: {color}")
        seen.add(name.casefold())
        cleaned.append({"name": name, "color": color.lower()})
    return cleaned


def _inferred_class_count(root: Path) -> int:
    directories = [root, root / "labels", root / "unlabeled"]
    directories.extend(root / split / "labels" for split in ("labeled", "train", "valid", "test"))
    highest_id = -1
    for directory in directories:
        if not directory.is_dir():
            continue
        for label_path in directory.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts and parts[0].isdecimal():
                    highest_id = max(highest_id, int(parts[0]))
    return highest_id + 1


@dataclass
class DatasetProject:
    root: Path
    classes: list[dict]
    last_folder: str | None = None
    last_image: str | None = None
    active_class: int = 0
    generator_settings: dict | None = None

    @property
    def names(self) -> list[str]:
        return [entry["name"] for entry in self.classes]

    @property
    def colors(self) -> list[str]:
        return [entry["color"] for entry in self.classes]

    @property
    def state_path(self) -> Path:
        return self.root / "labeler.yaml"

    def _payload(self, classes: list[dict] | None = None) -> str:
        return yaml.safe_dump({
            "schema_version": 1,
            "classes": self.classes if classes is None else classes,
            "last_folder": self.last_folder,
            "last_image": self.last_image,
            "active_class": self.active_class,
            "generator_settings": self.generator_settings,
        }, sort_keys=False)

    def save_state(self) -> None:
        _atomic_write(self.state_path, self._payload())

    def save_classes(self, classes: list[dict]) -> None:
        classes = _validate_classes(classes)
        data_path = self.root / "data.yaml"
        previous = data_path.read_bytes() if data_path.exists() else None
        data = yaml.safe_load(previous.decode("utf-8")) if previous is not None else {
            "train": "train/images", "val": "valid/images", "test": "test/images"
        }
        if not isinstance(data, dict):
            raise ValueError(f"Invalid dataset YAML: {data_path}")
        data["nc"] = len(classes)
        data["names"] = [entry["name"] for entry in classes]
        _atomic_write(data_path, yaml.safe_dump(data, sort_keys=False))
        try:
            _atomic_write(self.state_path, self._payload(classes))
        except Exception:
            if previous is None:
                data_path.unlink(missing_ok=True)
            else:
                _atomic_write(data_path, previous)
            raise
        self.classes = classes


def load_project(root: Path) -> DatasetProject:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "labeler.yaml"
    if state_path.is_file():
        data = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError(f"Invalid labeler state: {state_path}")
        classes = _validate_classes(data.get("classes"))
        project = DatasetProject(
            root, classes, data.get("last_folder"), data.get("last_image"),
            int(data.get("active_class", 0)), data.get("generator_settings"),
        )
        names_in_yaml = _names_from_data(root / "data.yaml")
        if names_in_yaml != project.names:
            project.save_classes(project.classes)
        return project

    names = _names_from_data(root / "data.yaml")
    if names is None:
        inferred_count = _inferred_class_count(root)
        names = [f"class_{index}" for index in range(inferred_count)] if inferred_count else ["object"]
    classes = [{"name": name, "color": DEFAULT_COLORS[index % len(DEFAULT_COLORS)]}
               for index, name in enumerate(names)]
    project = DatasetProject(root, _validate_classes(classes))
    if not (root / "data.yaml").is_file():
        project.save_classes(project.classes)
    else:
        project.save_state()
    return project


def find_dataset_root(folder: Path, current_root: Path) -> Path:
    folder = folder.expanduser().resolve()
    for candidate in (folder, *folder.parents):
        if (candidate / "labeler.yaml").is_file() or (candidate / "data.yaml").is_file():
            return candidate
    if folder == current_root or current_root in folder.parents:
        return current_root
    return folder.parent if folder.name.lower() == "images" else folder


def recent_dataset(default_root: Path) -> Path:
    if RECENT_DATASET_FILE.is_file():
        try:
            data = yaml.safe_load(RECENT_DATASET_FILE.read_text(encoding="utf-8"))
            saved = data.get("last_dataset") if isinstance(data, dict) else None
            if isinstance(saved, str) and Path(saved).is_dir():
                return Path(saved).expanduser().resolve()
        except (OSError, yaml.YAMLError):
            pass
    return default_root.expanduser().resolve()


def remember_dataset(root: Path) -> None:
    _atomic_write(RECENT_DATASET_FILE, yaml.safe_dump({"last_dataset": str(root)}, sort_keys=False))
