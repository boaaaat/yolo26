"""Read-only browser for images and YOLO labels in generated dataset splits."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from dataset_project import DEFAULT_COLORS, DatasetProject


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class DatasetImage:
    path: Path
    label_path: Path
    split: str
    class_ids: frozenset[int]


def dataset_versions(root: Path) -> list[Path]:
    versions_dir = root / "versions"
    versions = []
    if versions_dir.is_dir():
        versions = sorted((path for path in versions_dir.iterdir()
                           if path.is_dir() and (path / "data.yaml").is_file()),
                          key=lambda path: (int(path.name[1:]) if path.name.startswith("v")
                                            and path.name[1:].isdigit() else -1, path.name),
                          reverse=True)
    if (root / "data.yaml").is_file() and any((root / split / "images").is_dir()
                                              for split in (*SPLITS, "val")):
        versions.append(root)
    return versions


def label_class_ids(path: Path) -> frozenset[int]:
    if not path.is_file():
        return frozenset()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()  # A detailed error appears when the image is selected.
    class_ids = set()
    for line in lines:
        parts = line.split()
        if parts:
            try:
                class_ids.add(int(parts[0]))
            except ValueError:
                continue
    return frozenset(class_ids)


def read_boxes(path: Path, width: int, height: int, class_count: int) -> list[tuple[int, float, float, float, float]]:
    if not path.is_file():
        return []
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path.name}, line {line_number}: expected five values")
        try:
            class_id = int(parts[0])
            cx, cy, box_width, box_height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path.name}, line {line_number}: invalid number") from exc
        if (not 0 <= class_id < class_count or
                not all(math.isfinite(value) and 0 <= value <= 1
                        for value in (cx, cy, box_width, box_height)) or
                box_width <= 0 or box_height <= 0 or
                cx - box_width / 2 < -1e-5 or cy - box_height / 2 < -1e-5 or
                cx + box_width / 2 > 1 + 1e-5 or cy + box_height / 2 > 1 + 1e-5):
            raise ValueError(f"{path.name}, line {line_number}: class or box out of range")
        boxes.append((class_id, (cx - box_width / 2) * width, (cy - box_height / 2) * height,
                      box_width * width, box_height * height))
    return boxes


class DatasetPreview(QGraphicsView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#111827")))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.has_image = False

    def clear_image(self) -> None:
        self.scene().clear()
        self.has_image = False

    def show_image(self, pixmap: QPixmap, boxes: list[tuple], names: list[str], colors: list[str]) -> None:
        self.scene().clear()
        self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(0, 0, pixmap.width(), pixmap.height())
        for class_id, x, y, width, height in boxes:
            color = QColor(colors[class_id])
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            self.scene().addRect(x, y, width, height, pen,
                                 QBrush(QColor(color.red(), color.green(), color.blue(), 28)))
            text = self.scene().addSimpleText(names[class_id], QFont("Segoe UI", 10, QFont.Weight.DemiBold))
            text.setBrush(QBrush(color))
            text.setPos(x, max(0, y - 23))
        self.has_image = True
        self.fit_image()

    def fit_image(self) -> None:
        if self.has_image:
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if self.has_image:
            self.scale(1.2 if event.angleDelta().y() > 0 else 1 / 1.2,
                       1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            event.accept()
        else:
            super().wheelEvent(event)


class DatasetBrowserDialog(QDialog):
    def __init__(self, project: DatasetProject, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.entries: list[DatasetImage] = []
        self.by_path: dict[str, DatasetImage] = {}
        self.names: list[str] = []
        self.colors: list[str] = []
        self.setWindowTitle(f"Browse dataset · {project.root.name}")
        self.resize(1250, 780)

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.version_filter = QComboBox()
        for version in dataset_versions(project.root):
            self.version_filter.addItem(version.name, str(version))
        controls.addWidget(QLabel("Version"))
        controls.addWidget(self.version_filter)
        self.split_filter = QComboBox()
        self.split_filter.addItem("All splits", None)
        for split, name in (("train", "Train"), ("valid", "Validation"), ("test", "Test")):
            self.split_filter.addItem(name, split)
        controls.addWidget(QLabel("Split"))
        controls.addWidget(self.split_filter)
        self.class_filter = QComboBox()
        self.class_filter.addItem("All classes", None)
        controls.addWidget(QLabel("Class"))
        controls.addWidget(self.class_filter)
        self.sort_filter = QComboBox()
        self.sort_filter.addItem("Sort: filename", "filename")
        self.sort_filter.addItem("Sort: class", "class")
        self.sort_filter.addItem("Sort: split", "split")
        controls.addWidget(self.sort_filter)
        layout.addLayout(controls)

        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image_list = QListWidget()
        splitter.addWidget(self.image_list)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        self.image_title = QLabel("Select an image")
        preview_layout.addWidget(self.image_title)
        self.label_info = QLabel("Boxes and class names are drawn on the image.")
        self.label_info.setWordWrap(True)
        preview_layout.addWidget(self.label_info)
        self.preview = DatasetPreview()
        preview_layout.addWidget(self.preview, 1)
        fit_button = QPushButton("Fit image")
        fit_button.clicked.connect(self.preview.fit_image)
        preview_layout.addWidget(fit_button)
        splitter.addWidget(preview_panel)
        splitter.setSizes((350, 900))
        layout.addWidget(splitter, 1)

        self.version_filter.currentIndexChanged.connect(self.load_version)
        self.split_filter.currentIndexChanged.connect(self.apply_filters)
        self.class_filter.currentIndexChanged.connect(self.apply_filters)
        self.sort_filter.currentIndexChanged.connect(self.apply_filters)
        self.image_list.currentItemChanged.connect(self.show_image)
        self.load_version()

    def load_version(self, *_args) -> None:
        self.entries.clear()
        self.by_path.clear()
        self.preview.clear_image()
        self.image_title.setText("Select an image")
        root_value = self.version_filter.currentData()
        if root_value is None:
            self.image_list.clear()
            self.count_label.setText("No generated dataset found. Generate a dataset first.")
            return
        root = Path(root_value)
        try:
            data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("data.yaml must contain a mapping")
            names = data.get("names")
            if isinstance(names, dict):
                names = [name for _, name in sorted(names.items(), key=lambda pair: int(pair[0]))]
            if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
                raise ValueError("data.yaml has invalid class names")
            self.names = names
            palette = {name.casefold(): color for name, color in zip(self.project.names, self.project.colors)}
            self.colors = [palette.get(name.casefold(), DEFAULT_COLORS[index % len(DEFAULT_COLORS)])
                           for index, name in enumerate(names)]
            for split, key, fallback in (("train", "train", "train/images"),
                                         ("valid", "val", "valid/images"),
                                         ("test", "test", "test/images")):
                configured = data.get(key, fallback)
                if not isinstance(configured, str):
                    continue
                image_dir = Path(configured)
                if not image_dir.is_absolute():
                    image_dir = root / image_dir
                if not image_dir.is_dir():
                    continue
                label_dir = image_dir.parent / "labels"
                for image_path in image_dir.iterdir():
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                        label_path = label_dir / f"{image_path.stem}.txt"
                        entry = DatasetImage(image_path, label_path, split, label_class_ids(label_path))
                        self.entries.append(entry)
                        self.by_path[str(image_path)] = entry
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self.image_list.clear()
            self.count_label.setText(f"Could not read {root}: {exc}")
            return
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItem("All classes", None)
        for class_id, name in enumerate(self.names):
            self.class_filter.addItem(name, class_id)
        self.class_filter.blockSignals(False)
        self.apply_filters()

    def apply_filters(self, *_args) -> None:
        current = self.image_list.currentItem()
        selected_path = current.data(Qt.ItemDataRole.UserRole) if current else None
        split = self.split_filter.currentData()
        class_id = self.class_filter.currentData()
        entries = [entry for entry in self.entries
                   if (split is None or entry.split == split)
                   and (class_id is None or class_id in entry.class_ids)]
        split_order = {name: index for index, name in enumerate(SPLITS)}
        order = self.sort_filter.currentData()
        if order == "class":
            entries.sort(key=lambda entry: (min(entry.class_ids, default=len(self.names)),
                                            split_order[entry.split], entry.path.name.casefold()))
        elif order == "split":
            entries.sort(key=lambda entry: (split_order[entry.split], entry.path.name.casefold()))
        else:
            entries.sort(key=lambda entry: (entry.path.name.casefold(), split_order[entry.split]))
        self.image_list.blockSignals(True)
        self.image_list.clear()
        selected_row = -1
        for index, entry in enumerate(entries):
            classes = ", ".join(self.names[class_id] for class_id in sorted(entry.class_ids)
                                if 0 <= class_id < len(self.names)) or "no labels"
            item = QListWidgetItem(f"{entry.split} · {entry.path.name} · {classes}")
            item.setData(Qt.ItemDataRole.UserRole, str(entry.path))
            self.image_list.addItem(item)
            if str(entry.path) == selected_path:
                selected_row = index
        self.image_list.blockSignals(False)
        self.count_label.setText(f"{len(entries)} of {len(self.entries)} images")
        if entries:
            self.image_list.setCurrentRow(max(0, selected_row))
        else:
            self.preview.clear_image()
            self.image_title.setText("No images match these filters")
            self.label_info.clear()

    def show_image(self, current, _previous) -> None:
        if current is None:
            return
        entry = self.by_path.get(current.data(Qt.ItemDataRole.UserRole))
        if entry is None:
            return
        self.image_title.setText(f"{entry.split} / {entry.path.name}")
        pixmap = QPixmap(str(entry.path))
        if pixmap.isNull():
            self.preview.clear_image()
            self.label_info.setText(f"Could not open image: {entry.path}")
            return
        try:
            boxes = read_boxes(entry.label_path, pixmap.width(), pixmap.height(), len(self.names))
        except (OSError, ValueError) as exc:
            boxes = []
            self.label_info.setText(f"Label issue: {exc}")
        else:
            classes = ", ".join(self.names[box[0]] for box in boxes)
            self.label_info.setText(f"{len(boxes)} boxes · {classes or 'No label boxes'}"
                                    + ("" if entry.label_path.is_file() else " · Label file missing"))
        self.preview.show_image(pixmap, boxes, self.names, self.colors)
