"""Desktop box-labeling app for the local YOLO26 detection dataset."""

import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from dataset_generator import GenerateConfig, generate_dataset, get_split_percentages
from dataset_project import (
    DEFAULT_COLORS, DatasetProject, find_dataset_root, load_project,
    recent_dataset, remember_dataset,
)
from review_metadata import load_review_metadata, metadata_path, save_review_metadata
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QBrush, QCursor, QFont, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO, YOLOE


# Settings: edit these paths and confidence values for your dataset.
DATASET_DIR = Path(__file__).resolve().parent / "datasets"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26m" / "weights" / "best.pt"
YOLOE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yoloe-26s-seg.pt"
YOLOE_PROMPT_PROFILE = Path(__file__).resolve().parent / "models" / "roblox-yoloe-26s-visual.npz"
SUGGESTION_CONFIDENCE_BY_CLASS = {
    "dead": 0.50,
    "enemy": 0.50,
    "katana": 0.50,
    "teammate": 0.50,
}
YOLOE_CONFIDENCE_BY_CLASS = {
    "dead": 0.25,
    "enemy": 0.25,
    "katana": 0.25,
    "teammate": 0.25,
}
MODEL_IMAGE_SIZE = 1024
DEVICE = 0  # First NVIDIA GPU; use "cpu" if needed.

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIN_BOX_SIZE = 3.0


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    suggested: bool = False
    confidence: float | None = None

    def rect(self) -> QRectF:
        return QRectF(self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)


def read_labels(label_path: Path, width: int, height: int, class_count: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path.name}, line {line_number}: expected five values")
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{label_path.name}, line {line_number}: invalid number") from exc
        if not 0 <= class_id < class_count or not all(0 <= value <= 1 for value in (cx, cy, w, h)):
            raise ValueError(f"{label_path.name}, line {line_number}: class or coordinates out of range")
        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
        if w <= 0 or h <= 0 or x1 < -0.01 or y1 < -0.01 or x2 > width + 0.01 or y2 > height + 0.01:
            raise ValueError(f"{label_path.name}, line {line_number}: invalid box")
        boxes.append(Box(class_id, max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    return boxes


def accepted_signature(boxes: list[Box]) -> tuple:
    return tuple(
        (box.class_id, box.x1, box.y1, box.x2, box.y2)
        for box in boxes if not box.suggested
    )


def folder_layout(folder: Path) -> tuple[Path, Path]:
    """Find images and their YOLO labels in a selected dataset folder."""
    image_dir = folder / "images" if (folder / "images").is_dir() else folder
    if image_dir.name.lower() == "images":
        label_dir = image_dir.parent / "labels"
    elif (folder / "labels").is_dir():
        label_dir = folder / "labels"
    else:
        label_dir = image_dir
    return image_dir, label_dir


class LabelCanvas(QGraphicsView):
    changed = Signal(object)  # Snapshot before an edit, for undo.
    selection_changed = Signal(int)

    def __init__(self, class_names: list[str], class_colors: list[str]) -> None:
        super().__init__()
        from PySide6.QtWidgets import QGraphicsScene

        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#111827")))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.class_names = class_names
        self.class_colors = class_colors
        self.boxes: list[Box] = []
        self.selected = -1
        self.active_class = 0
        self.mode = "draw"
        self.image_width = 0
        self.image_height = 0
        self.annotation_items = []
        self.preview_item = None
        self.horizontal_guide = None
        self.vertical_guide = None
        self.drag = None
        self.pan_position = None

    def load_image(self, image_path: Path, boxes: list[Box]) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            raise ValueError(f"Could not open image: {image_path}")
        self.scene().clear()
        self.annotation_items.clear()
        self.preview_item = None
        self.horizontal_guide = None
        self.vertical_guide = None
        self.scene().addPixmap(pixmap)
        self.image_width, self.image_height = pixmap.width(), pixmap.height()
        self.scene().setSceneRect(0, 0, self.image_width, self.image_height)
        self._create_guides()
        self.boxes = [replace(box) for box in boxes]
        self.select(-1)
        self.redraw()
        QTimer.singleShot(0, self.fit_image)

    def clear_image(self) -> None:
        self.scene().clear()
        self.annotation_items.clear()
        self.horizontal_guide = None
        self.vertical_guide = None
        self.boxes.clear()
        self.image_width = self.image_height = 0
        self.select(-1)

    def snapshot(self) -> list[Box]:
        return [replace(box) for box in self.boxes]

    def set_boxes(self, boxes: list[Box]) -> None:
        self.boxes = [replace(box) for box in boxes]
        self.select(-1)
        self.redraw()

    def select(self, index: int) -> None:
        self.selected = index if 0 <= index < len(self.boxes) else -1
        self.selection_changed.emit(self.selected)
        self.redraw()

    def fit_image(self) -> None:
        if self.image_width:
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.redraw()
            self._update_guides(self.viewport().mapFromGlobal(QCursor.pos()))

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        position = self.viewport().mapFromGlobal(QCursor.pos())
        self._update_guides(position)

    def _create_guides(self) -> None:
        pen = QPen(QColor(255, 255, 255, 170), 1, Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        self.horizontal_guide = self.scene().addLine(0, 0, self.image_width, 0, pen)
        self.vertical_guide = self.scene().addLine(0, 0, 0, self.image_height, pen)
        for guide in (self.horizontal_guide, self.vertical_guide):
            guide.setZValue(1000)
            guide.hide()

    def _update_guides(self, view_position) -> None:
        if self.horizontal_guide is None or self.vertical_guide is None:
            return
        if self.mode != "draw" or not self.viewport().rect().contains(view_position):
            self.horizontal_guide.hide()
            self.vertical_guide.hide()
            return
        point = self.mapToScene(view_position)
        if not (0 <= point.x() <= self.image_width and 0 <= point.y() <= self.image_height):
            self.horizontal_guide.hide()
            self.vertical_guide.hide()
            return
        self.horizontal_guide.setLine(0, point.y(), self.image_width, point.y())
        self.vertical_guide.setLine(point.x(), 0, point.x(), self.image_height)
        self.horizontal_guide.show()
        self.vertical_guide.show()

    def redraw(self) -> None:
        for item in self.annotation_items:
            self.scene().removeItem(item)
        self.annotation_items.clear()
        if not self.image_width:
            return
        for index, box in enumerate(self.boxes):
            color = QColor(self.class_colors[box.class_id])
            pen = QPen(color, 2.5 if index == self.selected else 1.8)
            pen.setCosmetic(True)
            if box.suggested:
                pen.setStyle(Qt.PenStyle.DashLine)
            rectangle = self.scene().addRect(box.rect(), pen, QBrush(QColor(color.red(), color.green(), color.blue(), 28)))
            self.annotation_items.append(rectangle)
            suggestion_text = (f" · {box.confidence:.0%} suggestion" if box.confidence is not None
                               else " · suggestion") if box.suggested else ""
            text = self.scene().addSimpleText(
                self.class_names[box.class_id] + suggestion_text,
                QFont("Segoe UI", 9, QFont.Weight.DemiBold),
            )
            text.setBrush(QBrush(color))
            text.setPos(box.x1, max(0, box.y1 - 21))
            self.annotation_items.append(text)
        if self.selected >= 0:
            box = self.boxes[self.selected]
            size = 9 / max(self.transform().m11(), 0.01)
            handle_pen = QPen(QColor("#111827"), 1)
            handle_pen.setCosmetic(True)
            for point in self._handle_points(box).values():
                handle = self.scene().addRect(
                    point.x() - size / 2, point.y() - size / 2, size, size,
                    handle_pen, QBrush(QColor("#ffffff")),
                )
                self.annotation_items.append(handle)

    @staticmethod
    def _handle_points(box: Box) -> dict[str, QPointF]:
        mid_x, mid_y = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        return {
            "nw": QPointF(box.x1, box.y1), "n": QPointF(mid_x, box.y1),
            "ne": QPointF(box.x2, box.y1), "e": QPointF(box.x2, mid_y),
            "se": QPointF(box.x2, box.y2), "s": QPointF(mid_x, box.y2),
            "sw": QPointF(box.x1, box.y2), "w": QPointF(box.x1, mid_y),
        }

    def _point(self, view_position) -> QPointF:
        point = self.mapToScene(view_position)
        return QPointF(
            max(0, min(self.image_width, point.x())),
            max(0, min(self.image_height, point.y())),
        )

    def _hit(self, point: QPointF) -> tuple[int, str | None]:
        if self.selected >= 0:
            tolerance = 10 / max(self.transform().m11(), 0.01)
            for handle, center in self._handle_points(self.boxes[self.selected]).items():
                if abs(point.x() - center.x()) <= tolerance and abs(point.y() - center.y()) <= tolerance:
                    return self.selected, handle
        for index in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[index].rect().contains(point):
                return index, None
        return -1, None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_position = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or not self.image_width:
            return super().mousePressEvent(event)
        point = self._point(event.pos())
        before = self.snapshot()
        if self.mode == "draw":
            self.drag = {"kind": "draw", "start": point, "before": before}
            self.preview_item = self.scene().addRect(QRectF(point, point), QPen(QColor("#ffffff"), 2))
        else:
            index, handle = self._hit(point)
            self.select(index)
            if index >= 0:
                self.drag = {
                    "kind": "resize" if handle else "move",
                    "index": index, "handle": handle, "start": point,
                    "original": replace(self.boxes[index]), "before": before,
                }
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self.pan_position is not None:
            delta = event.pos() - self.pan_position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan_position = event.pos()
            self._update_guides(event.pos())
            event.accept()
            return
        self._update_guides(event.pos())
        if self.drag is None:
            return super().mouseMoveEvent(event)
        point = self._point(event.pos())
        drag = self.drag
        if drag["kind"] == "draw":
            self.preview_item.setRect(QRectF(drag["start"], point).normalized())
        else:
            box = replace(drag["original"])
            if drag["kind"] == "move":
                dx, dy = point.x() - drag["start"].x(), point.y() - drag["start"].y()
                dx = max(-box.x1, min(self.image_width - box.x2, dx))
                dy = max(-box.y1, min(self.image_height - box.y2, dy))
                box.x1 += dx; box.x2 += dx
                box.y1 += dy; box.y2 += dy
            else:
                handle = drag["handle"]
                if "w" in handle:
                    box.x1 = min(point.x(), box.x2 - MIN_BOX_SIZE)
                if "e" in handle:
                    box.x2 = max(point.x(), box.x1 + MIN_BOX_SIZE)
                if "n" in handle:
                    box.y1 = min(point.y(), box.y2 - MIN_BOX_SIZE)
                if "s" in handle:
                    box.y2 = max(point.y(), box.y1 + MIN_BOX_SIZE)
            self.boxes[drag["index"]] = box
            self.redraw()
        event.accept()

    def leaveEvent(self, event) -> None:
        if self.horizontal_guide is not None:
            self.horizontal_guide.hide()
            self.vertical_guide.hide()
        super().leaveEvent(event)

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.Type.Leave and getattr(self, "horizontal_guide", None) is not None:
            self.horizontal_guide.hide()
            self.vertical_guide.hide()
        return super().viewportEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_position = None
            self.unsetCursor()
            self._update_guides(event.pos())
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or self.drag is None:
            return super().mouseReleaseEvent(event)
        drag = self.drag
        if drag["kind"] == "draw":
            rect = self.preview_item.rect()
            self.scene().removeItem(self.preview_item)
            self.preview_item = None
            if rect.width() >= MIN_BOX_SIZE and rect.height() >= MIN_BOX_SIZE:
                self.boxes.append(Box(self.active_class, rect.left(), rect.top(), rect.right(), rect.bottom()))
                self.select(len(self.boxes) - 1)
        self.drag = None
        if self.snapshot() != drag["before"]:
            self.changed.emit(drag["before"])
        self.redraw()
        event.accept()

    def wheelEvent(self, event) -> None:
        if not self.image_width:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        new_scale = self.transform().m11() * factor
        if 0.05 <= new_scale <= 20:
            self.scale(factor, factor)
            self.redraw()
            self._update_guides(self.viewport().mapFromGlobal(QCursor.pos()))
        event.accept()


class SuggestionWorker(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, class_names: list[str]) -> None:
        super().__init__()
        self.class_names = class_names
        self.models = {}

    @Slot(str, str)
    def suggest(self, image_path: str, source: str) -> None:
        try:
            if source not in self.models:
                if source == "trained":
                    model = YOLO(str(Path(CHECKPOINT_PATH).expanduser().resolve()))
                    if model.task != "detect":
                        raise ValueError("Checkpoint must be an object detection model")
                elif source == "yoloe":
                    model = YOLOE(str(Path(YOLOE_MODEL_PATH).expanduser().resolve()))
                    model.load_prompt_embeddings(Path(YOLOE_PROMPT_PROFILE).expanduser().resolve())
                else:
                    raise ValueError(f"Unknown suggestion source: {source}")
                self.models[source] = model
            model = self.models[source]
            model_names = model.names
            model_names = dict(model_names.items()) if isinstance(model_names, dict) else dict(enumerate(model_names))
            if not set(model_names.values()).issubset(self.class_names):
                raise ValueError(f"Model classes {model_names} are not in dataset classes {self.class_names}")
            thresholds = SUGGESTION_CONFIDENCE_BY_CLASS if source == "trained" else YOLOE_CONFIDENCE_BY_CLASS
            default_threshold = 0.50 if source == "trained" else 0.25
            if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in thresholds.values()):
                raise ValueError("Suggestion confidence values must be between 0 and 1")

            result = model.predict(
                source=image_path,
                conf=min(thresholds.get(name, default_threshold) for name in model_names.values()),
                imgsz=MODEL_IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
            )[0]
            suggestions = []
            if result.boxes is not None:
                for predicted in result.boxes:
                    model_class_id = int(predicted.cls.item())
                    class_name = model_names[model_class_id]
                    if float(predicted.conf.item()) < thresholds.get(class_name, default_threshold):
                        continue
                    dataset_class_id = self.class_names.index(class_name)
                    suggestions.append((dataset_class_id, *predicted.xywhn[0].tolist()))
            self.finished.emit(image_path, suggestions)
        except Exception as exc:
            self.failed.emit(image_path, str(exc))


class ClassManagerDialog(QDialog):
    def __init__(self, classes: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage dataset classes")
        self.setMinimumWidth(420)
        self.classes = [entry.copy() for entry in classes]
        layout = QVBoxLayout(self)
        hint = QLabel("Class IDs keep their order so existing box labels stay valid. Exclude classes when generating a dataset.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.class_list = QListWidget()
        self.class_list.itemDoubleClicked.connect(self.rename_class)
        layout.addWidget(self.class_list)
        self._refresh()

        edits = QHBoxLayout()
        add_button = QPushButton("Add class")
        add_button.clicked.connect(self.add_class)
        edits.addWidget(add_button)
        rename_button = QPushButton("Rename")
        rename_button.clicked.connect(self.rename_class)
        edits.addWidget(rename_button)
        color_button = QPushButton("Change color")
        color_button.clicked.connect(self.change_color)
        edits.addWidget(color_button)
        layout.addLayout(edits)

        actions = QHBoxLayout()
        actions.addStretch()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        actions.addWidget(cancel_button)
        save_button = QPushButton("Save classes")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self.accept)
        actions.addWidget(save_button)
        layout.addLayout(actions)

    def _refresh(self, selected: int = 0) -> None:
        self.class_list.clear()
        for index, entry in enumerate(self.classes):
            item = QListWidgetItem(f"{index}   {entry['name']}")
            item.setForeground(QBrush(QColor(entry["color"])))
            self.class_list.addItem(item)
        self.class_list.setCurrentRow(min(selected, len(self.classes) - 1))

    def _name_is_valid(self, name: str, except_index: int = -1) -> bool:
        if not name.strip():
            QMessageBox.warning(self, "Invalid name", "Enter a class name.")
            return False
        if any(index != except_index and entry["name"].casefold() == name.strip().casefold()
               for index, entry in enumerate(self.classes)):
            QMessageBox.warning(self, "Duplicate name", "Each class needs a unique name.")
            return False
        return True

    def add_class(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add class", "Class name")
        if not accepted or not self._name_is_valid(name):
            return
        self.classes.append({"name": name.strip(),
                             "color": DEFAULT_COLORS[len(self.classes) % len(DEFAULT_COLORS)]})
        self._refresh(len(self.classes) - 1)

    def rename_class(self, *_args) -> None:
        index = self.class_list.currentRow()
        if index < 0:
            return
        name, accepted = QInputDialog.getText(
            self, "Rename class", "Class name", text=self.classes[index]["name"]
        )
        if not accepted or not self._name_is_valid(name, index):
            return
        self.classes[index]["name"] = name.strip()
        self._refresh(index)

    def change_color(self) -> None:
        index = self.class_list.currentRow()
        if index < 0:
            return
        color = QColorDialog.getColor(QColor(self.classes[index]["color"]), self, "Class color")
        if not color.isValid():
            return
        self.classes[index]["color"] = color.name()
        self._refresh(index)


class GenerationWorker(QObject):
    progress = Signal(str)
    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(self, dataset_dir: Path, class_names: list[str], class_colors: list[str],
                 config: GenerateConfig) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.class_names = class_names
        self.class_colors = class_colors
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            path, metadata = generate_dataset(
                self.dataset_dir, self.class_names, self.config, self.progress.emit,
                self.class_colors,
            )
            self.finished.emit(str(path), metadata)
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetGeneratorDialog(QDialog):
    def __init__(self, project: DatasetProject, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.dataset_dir = project.root
        self.class_names = project.names
        self.class_colors = project.colors
        self.generation_thread: QThread | None = None
        self.generation_worker: GenerationWorker | None = None
        self.setWindowTitle("Generate YOLO dataset")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        intro = QLabel("Build a new dataset version from labeled/images and labeled/labels. Source images stay in place.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(QLabel("INCLUDE CLASSES"))
        self.class_checks = []
        classes = QHBoxLayout()
        for name in self.class_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            self.class_checks.append(checkbox)
            classes.addWidget(checkbox)
        classes.addStretch()
        layout.addLayout(classes)

        form = QFormLayout()
        self.train_percent = QSpinBox()
        self.train_percent.setRange(1, 98)
        self.train_percent.setValue(80)
        self.train_percent.setSuffix("%")
        form.addRow("Target training split", self.train_percent)
        self.valid_percent = QSpinBox()
        self.valid_percent.setRange(1, 98)
        self.valid_percent.setValue(15)
        self.valid_percent.setSuffix("%")
        form.addRow("Target validation split", self.valid_percent)
        self.test_percent = QLabel()
        form.addRow("Target test split", self.test_percent)
        self.train_percent.valueChanged.connect(self._update_test_percent)
        self.valid_percent.valueChanged.connect(self._update_test_percent)
        self._update_test_percent()

        self.augment_copies = QSpinBox()
        self.augment_copies.setRange(0, 10)
        self.augment_copies.setValue(1)
        form.addRow("Extra training copies / image", self.augment_copies)

        self.flip = QCheckBox("Horizontal flip")
        self.flip.setChecked(True)
        form.addRow(self.flip)
        self.rotation = QCheckBox("Rotation")
        self.rotation_degrees = QSpinBox()
        self.rotation_degrees.setRange(1, 30)
        self.rotation_degrees.setValue(5)
        self.rotation_degrees.setSuffix("° max")
        form.addRow(self.rotation, self.rotation_degrees)
        self.brightness = QCheckBox("Brightness")
        self.brightness.setChecked(True)
        self.brightness_percent = QSpinBox()
        self.brightness_percent.setRange(1, 50)
        self.brightness_percent.setValue(15)
        self.brightness_percent.setSuffix("% max")
        form.addRow(self.brightness, self.brightness_percent)
        self.contrast = QCheckBox("Contrast")
        self.contrast.setChecked(True)
        self.contrast_percent = QSpinBox()
        self.contrast_percent.setRange(1, 50)
        self.contrast_percent.setValue(15)
        self.contrast_percent.setSuffix("% max")
        form.addRow(self.contrast, self.contrast_percent)
        self.blur = QCheckBox("Gaussian blur")
        self.blur_radius = QDoubleSpinBox()
        self.blur_radius.setRange(0.3, 3.0)
        self.blur_radius.setSingleStep(0.1)
        self.blur_radius.setValue(1.0)
        self.blur_radius.setSuffix(" px max")
        form.addRow(self.blur, self.blur_radius)
        layout.addLayout(form)

        hint = QLabel(
            "Each enabled augmentation has a 50% chance per training copy. Validation and test use "
            "originals. Images with only excluded classes are skipped."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.progress_label = QLabel("Ready")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        actions = QHBoxLayout()
        actions.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        self.generate_button = QPushButton("Generate version")
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(self.start_generation)
        actions.addWidget(self.generate_button)
        layout.addLayout(actions)
        self._restore_settings()
        fixed_split = get_split_percentages(self.dataset_dir)
        if fixed_split is not None:
            self.train_percent.setValue(fixed_split[0])
            self.valid_percent.setValue(fixed_split[1])
            self.train_percent.setEnabled(False)
            self.valid_percent.setEnabled(False)
            hint.setText(hint.text() + " Validation and test assignments are fixed across versions.")

    def _restore_settings(self) -> None:
        settings = self.project.generator_settings
        if not isinstance(settings, dict):
            return
        included = set(settings.get("included_class_ids", range(len(self.class_checks))))
        for index, checkbox in enumerate(self.class_checks):
            checkbox.setChecked(index in included)
        self.train_percent.setValue(int(settings.get("train_percent", 80)))
        self.valid_percent.setValue(int(settings.get("valid_percent", 15)))
        self.augment_copies.setValue(int(settings.get("augment_copies", 1)))
        self.flip.setChecked(bool(settings.get("horizontal_flip", True)))
        self.rotation.setChecked(bool(settings.get("rotation_degrees", 0)))
        self.rotation_degrees.setValue(max(1, int(settings.get("rotation_degrees", 5))))
        self.brightness.setChecked(bool(settings.get("brightness_percent", 0)))
        self.brightness_percent.setValue(max(1, int(settings.get("brightness_percent", 15))))
        self.contrast.setChecked(bool(settings.get("contrast_percent", 0)))
        self.contrast_percent.setValue(max(1, int(settings.get("contrast_percent", 15))))
        self.blur.setChecked(bool(settings.get("blur_radius", 0)))
        self.blur_radius.setValue(max(0.3, float(settings.get("blur_radius", 1.0))))

    def _update_test_percent(self, *_args) -> None:
        remaining = 100 - self.train_percent.value() - self.valid_percent.value()
        self.test_percent.setText(f"{remaining}%" if remaining > 0 else "Set train + validation below 100%")

    def start_generation(self) -> None:
        included = tuple(index for index, checkbox in enumerate(self.class_checks) if checkbox.isChecked())
        if not included:
            QMessageBox.warning(self, "No classes selected", "Select at least one class.")
            return
        if self.train_percent.value() + self.valid_percent.value() >= 100:
            QMessageBox.warning(self, "Invalid splits", "Leave at least 1% for the test split.")
            return
        config = GenerateConfig(
            included_class_ids=included,
            train_percent=self.train_percent.value(),
            valid_percent=self.valid_percent.value(),
            augment_copies=self.augment_copies.value(),
            horizontal_flip=self.flip.isChecked(),
            rotation_degrees=self.rotation_degrees.value() if self.rotation.isChecked() else 0,
            brightness_percent=self.brightness_percent.value() if self.brightness.isChecked() else 0,
            contrast_percent=self.contrast_percent.value() if self.contrast.isChecked() else 0,
            blur_radius=self.blur_radius.value() if self.blur.isChecked() else 0.0,
        )
        self.project.generator_settings = asdict(config)
        try:
            self.project.save_state()
        except OSError as exc:
            QMessageBox.warning(self, "Could not save settings", str(exc))
            return
        self.generation_thread = QThread(self)
        self.generation_worker = GenerationWorker(
            self.dataset_dir, self.class_names, self.class_colors, config
        )
        self.generation_worker.moveToThread(self.generation_thread)
        self.generation_thread.started.connect(self.generation_worker.run)
        self.generation_worker.progress.connect(self.progress_label.setText)
        self.generation_worker.finished.connect(self.on_generation_finished)
        self.generation_worker.failed.connect(self.on_generation_failed)
        self.generation_worker.finished.connect(self.generation_thread.quit)
        self.generation_worker.failed.connect(self.generation_thread.quit)
        self.generation_thread.finished.connect(self.generation_worker.deleteLater)
        self.generation_thread.finished.connect(self.on_thread_finished)
        self.generate_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.progress_label.setText("Preparing dataset...")
        self.generation_thread.start()

    def on_generation_finished(self, path: str, metadata: dict) -> None:
        counts = metadata["split_images"]
        self.progress_label.setText(
            f"Created {path}\n{counts['train']} train, {counts['valid']} validation, "
            f"{counts['test']} test originals; {metadata['augmented_train_images']} augmented training images."
            + (f" {metadata['split_conflicts_skipped']} images skipped to keep splits separate."
               if metadata.get("split_conflicts_skipped") else "")
            + (" Split assignments were saved in this version, but could not be copied to the dataset root."
               if metadata.get("split_manifest_warning") else "")
        )

    def on_generation_failed(self, message: str) -> None:
        self.progress_label.setText(f"Generation failed: {message}")
        QMessageBox.warning(self, "Dataset generation failed", message)

    def on_thread_finished(self) -> None:
        self.generation_thread.deleteLater()
        self.generation_thread = None
        self.generation_worker = None
        self.generate_button.setEnabled(True)
        self.close_button.setEnabled(True)

    def closeEvent(self, event) -> None:
        if self.generation_thread is not None:
            event.ignore()
        else:
            event.accept()


class LabelerWindow(QMainWindow):
    request_suggestions = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.project = load_project(recent_dataset(Path(DATASET_DIR)))
        self.dataset_dir = self.project.root
        self.class_names = self.project.names
        self.class_colors = self.project.colors
        restored_folder = self._restored_folder(self.project)
        self.source_dir, self.labels_dir = folder_layout(restored_folder)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.current_path: Path | None = None
        self.review_data: dict | None = None
        self.saved_signature: tuple = ()
        self.image_class_ids: dict[Path, frozenset[int]] = {}
        self.undo_history: list[list[Box]] = []
        self.redo_history: list[list[Box]] = []
        self.suggestion_thread: QThread | None = None
        self.suggestion_worker: SuggestionWorker | None = None
        restored_image = self.project.last_image
        self._build_ui()
        self._build_shortcuts()
        self.class_list.setCurrentRow(min(self.project.active_class, len(self.class_names) - 1))
        self.refresh_queue(select_row=0, select_name=restored_image)

    @staticmethod
    def _restored_folder(project: DatasetProject) -> Path:
        if project.last_folder:
            saved = Path(project.last_folder)
            candidate = saved if saved.is_absolute() else project.root / saved
            if candidate.is_dir():
                return candidate
        return project.root / "unlabeled"

    def _button(self, text: str, callback, *, prominent: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        if prominent:
            button.setObjectName("primaryButton")
        return button

    def _build_ui(self) -> None:
        self.setWindowTitle("Labeler · YOLO26")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0b1220; color: #e5eaf3; font-family: 'Segoe UI'; font-size: 13px; }
            QLabel#heading { font-size: 18px; font-weight: 700; color: #ffffff; }
            QLabel#muted { color: #9ba9bf; }
            QPushButton { background: #1c2a40; border: 1px solid #304159; border-radius: 7px;
                          padding: 8px 12px; min-height: 18px; }
            QPushButton:hover { background: #2b3c56; }
            QPushButton:checked, QPushButton#primaryButton { background: #2563eb; border-color: #3b82f6; color: white; }
            QPushButton:disabled { color: #6b7890; background: #152033; }
            QLineEdit, QComboBox, QListWidget, QSpinBox, QDoubleSpinBox {
                background: #111c2d; border: 1px solid #2a3a53; border-radius: 7px; padding: 5px;
            }
            QListWidget::item { padding: 7px; border-radius: 4px; }
            QListWidget::item:selected { background: #1f477d; color: white; }
            QScrollBar:vertical { background: #17263b; width: 14px; margin: 0; border: none; }
            QScrollBar:horizontal { background: #17263b; height: 14px; margin: 0; border: none; }
            QScrollBar::handle:vertical { background: #54799f; min-height: 28px; margin: 2px;
                                          border-radius: 5px; }
            QScrollBar::handle:horizontal { background: #54799f; min-width: 28px; margin: 2px;
                                            border-radius: 5px; }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #7199c2; }
            QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; }
            QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
            QSplitter::handle { background: #26354b; }
            QStatusBar { color: #aebbd0; }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(10)

        toolbar = QHBoxLayout()
        title = QLabel("Labeler")
        title.setObjectName("heading")
        toolbar.addWidget(title)
        toolbar.addSpacing(16)
        self.draw_button = self._button("Draw boxes  B", lambda: self.set_mode("draw"))
        self.select_button = self._button("Select / edit  V", lambda: self.set_mode("select"))
        for button in (self.draw_button, self.select_button):
            button.setCheckable(True)
            toolbar.addWidget(button)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.draw_button)
        mode_group.addButton(self.select_button)
        self.draw_button.setChecked(True)
        toolbar.addWidget(self._button("Fit image", self.canvas_fit_later))
        toolbar.addStretch()
        toolbar.addWidget(self._button("Undo", self.undo))
        toolbar.addWidget(self._button("Redo", self.redo))
        toolbar.addWidget(self._button("Save", self.save_current))
        self.finish_button = self._button("Finish → Labeled", self.finish_current, prominent=True)
        toolbar.addWidget(self.finish_button)
        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("IMAGE QUEUE"))
        self.folder_label = QLabel()
        self.folder_label.setObjectName("muted")
        self.folder_label.setWordWrap(True)
        left_layout.addWidget(self.folder_label)
        left_layout.addWidget(self._button("Open dataset…", self.open_dataset))
        left_layout.addWidget(self._button("Open unlabeled", self.open_unlabeled))
        left_layout.addWidget(self._button("Open labeled", self.open_labeled))
        left_layout.addWidget(self._button("Generate dataset…", self.open_generator))
        self.queue_count = QLabel()
        self.queue_count.setObjectName("muted")
        left_layout.addWidget(self.queue_count)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search filenames")
        self.search.textChanged.connect(self.apply_filter)
        left_layout.addWidget(self.search)
        self.queue_filter = QComboBox()
        self._fill_queue_filter()
        self._last_filter_data = "all"
        self.queue_filter.currentIndexChanged.connect(self.on_queue_filter_changed)
        left_layout.addWidget(self.queue_filter)
        left_layout.addWidget(self._button("Refresh queue", self.refresh_queue_from_button))
        left_layout.addWidget(self._button("Delete image…  Shift+Del", self.delete_current_image))
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)
        left_layout.addWidget(self.image_list, 1)
        nav = QHBoxLayout()
        nav.addWidget(self._button("← Previous", self.previous_image))
        nav.addWidget(self._button("Next →", self.next_image))
        left_layout.addLayout(nav)
        splitter.addWidget(left)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(2, 4, 2, 4)
        self.image_title = QLabel("No images in unlabeled")
        self.image_title.setObjectName("heading")
        center_layout.addWidget(self.image_title)
        self.review_info = QLabel()
        self.review_info.setObjectName("muted")
        center_layout.addWidget(self.review_info)
        self.canvas = LabelCanvas(self.class_names, self.class_colors)
        self.canvas.changed.connect(self.record_change)
        self.canvas.selection_changed.connect(self.on_canvas_selection)
        center_layout.addWidget(self.canvas, 1)
        hint = QLabel("Draw: drag a box  ·  Select: drag to move, use white handles to resize  ·  Wheel: zoom  ·  Middle drag: pan")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        center_layout.addWidget(hint)
        splitter.addWidget(center)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(QLabel("CLASSES"))
        self.class_list = QListWidget()
        self._fill_class_list()
        self.class_list.setCurrentRow(0)
        self.class_list.currentRowChanged.connect(self.on_class_selected)
        right_layout.addWidget(self.class_list)
        right_layout.addWidget(self._button("Manage classes…", self.manage_classes))
        class_hint = QLabel("Click a class to change the selected box, or choose the class for new boxes.")
        class_hint.setObjectName("muted")
        class_hint.setWordWrap(True)
        right_layout.addWidget(class_hint)
        right_layout.addWidget(QLabel("BOXES"))
        self.box_list = QListWidget()
        self.box_list.currentRowChanged.connect(self.on_box_selected)
        right_layout.addWidget(self.box_list, 1)
        self.box_count = QLabel("0 accepted · 0 suggestions")
        self.box_count.setObjectName("muted")
        right_layout.addWidget(self.box_count)
        right_layout.addWidget(self._button("Delete selected  Del", self.delete_selected))
        right_layout.addWidget(self._button("Accept selected  Enter", self.accept_selected))
        right_layout.addWidget(self._button("Accept all suggestions", self.accept_all))
        right_layout.addSpacing(12)
        right_layout.addWidget(QLabel("SUGGESTION MODEL"))
        self.suggestion_source = QComboBox()
        self.suggestion_source.addItem("Trained YOLO26", "trained")
        self.suggestion_source.addItem("YOLOE-26s · visual prompts (experimental)", "yoloe")
        right_layout.addWidget(self.suggestion_source)
        self.suggest_button = self._button("Suggest boxes", self.suggest_boxes)
        right_layout.addWidget(self.suggest_button)
        checkpoint_hint = QLabel("Suggestions are dashed. Accept or delete them before finishing. Review every suggested box and class.")
        checkpoint_hint.setObjectName("muted")
        checkpoint_hint.setWordWrap(True)
        right_layout.addWidget(checkpoint_hint)
        splitter.addWidget(right)
        splitter.setSizes((270, 890, 300))

        self.statusBar().showMessage("Ready · Ctrl+S save · Ctrl+Z/Y undo/redo · 1–9 set box class")

    def _build_shortcuts(self) -> None:
        bindings = {
            "B": lambda: self.set_mode("draw"),
            "V": lambda: self.set_mode("select"),
            "Ctrl+S": self.save_current,
            "Ctrl+Z": self.undo,
            "Ctrl+Y": self.redo,
            "Delete": self.delete_selected,
            "Shift+Delete": self.delete_current_image,
            "Return": self.accept_selected,
            "Left": self.previous_image,
            "Right": self.next_image,
        }
        for sequence, callback in bindings.items():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
        for index in range(9):
            shortcut = QShortcut(QKeySequence(str(index + 1)), self)
            shortcut.activated.connect(lambda index=index: self.class_list.setCurrentRow(index))

    def _fill_class_list(self) -> None:
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for index, name in enumerate(self.class_names):
            item = QListWidgetItem(f"{index + 1}   {name}")
            item.setForeground(QBrush(QColor(self.class_colors[index])))
            self.class_list.addItem(item)
        self.class_list.blockSignals(False)

    def _fill_queue_filter(self) -> None:
        selected = self.queue_filter.currentData()
        self.queue_filter.blockSignals(True)
        self.queue_filter.clear()
        self.queue_filter.addItem("All images", "all")
        self.queue_filter.addItem("Needs labels", "needs_labels")
        self.queue_filter.addItem("Has labels", "has_labels")
        self.queue_filter.insertSeparator(self.queue_filter.count())
        for class_id, name in enumerate(self.class_names):
            self.queue_filter.addItem(f"Class: {name}", f"class:{class_id}")
        selected_index = self.queue_filter.findData(selected)
        self.queue_filter.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        self.queue_filter.blockSignals(False)

    def canvas_fit_later(self) -> None:
        self.canvas.fit_image()

    def set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.draw_button.setChecked(mode == "draw")
        self.select_button.setChecked(mode == "select")
        self.statusBar().showMessage("Draw a new box" if mode == "draw" else "Select, move, or resize a box")

    def label_path(self, image_path: Path) -> Path:
        preferred = self.labels_dir / f"{image_path.stem}.txt"
        sidecar = image_path.with_suffix(".txt")
        if preferred.exists() or not sidecar.exists():
            return preferred
        return sidecar

    def label_class_ids(self, image_path: Path) -> frozenset[int]:
        label_path = self.label_path(image_path)
        if not label_path.is_file():
            return frozenset()
        try:
            return frozenset(
                int(parts[0]) for line in label_path.read_text(encoding="utf-8").splitlines()
                if (parts := line.split()) and parts[0].isdecimal()
            )
        except OSError:
            return frozenset()

    def open_unlabeled(self) -> None:
        self.switch_folder(self.dataset_dir / "unlabeled", dataset_root=self.dataset_dir)

    def open_dataset(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Open dataset root", str(self.dataset_dir.parent))
        if selected:
            folder = Path(selected).expanduser().resolve()
            dataset_root = find_dataset_root(folder, self.dataset_dir)
            self.switch_folder(dataset_root / "unlabeled", dataset_root=dataset_root)

    def open_labeled(self) -> None:
        dataset_root = find_dataset_root(self.dataset_dir, self.dataset_dir)
        self.switch_folder(dataset_root / "labeled", dataset_root=dataset_root)

    def switch_folder(self, folder: Path, *, restore_state: bool = False,
                      dataset_root: Path | None = None, queue_filter_data: str = "all") -> None:
        folder = folder.expanduser().resolve()
        dataset_root = find_dataset_root(dataset_root, self.dataset_dir) if dataset_root else find_dataset_root(folder, self.dataset_dir)
        if not self.suggest_button.isEnabled():
            QMessageBox.information(self, "Suggestion in progress", "Wait for the current suggestion to finish.")
            return
        if dataset_root == self.dataset_dir and folder == self.source_dir and not restore_state:
            return
        if not self.maybe_leave_current():
            return
        self._save_session()
        if dataset_root != self.dataset_dir:
            try:
                project = load_project(dataset_root)
            except Exception as exc:
                QMessageBox.warning(self, "Cannot open dataset", str(exc))
                return
            self._stop_suggestion_worker()
            self.project = project
            self.dataset_dir = project.root
            try:
                remember_dataset(project.root)
            except OSError as exc:
                self.statusBar().showMessage(f"Could not remember dataset: {exc}")
            self.class_names[:] = project.names
            self.class_colors[:] = project.colors
            self._fill_class_list()
            self._fill_queue_filter()
            active = min(max(0, project.active_class), len(self.class_names) - 1)
            self.class_list.blockSignals(True)
            self.class_list.setCurrentRow(active)
            self.class_list.blockSignals(False)
            self.canvas.active_class = active
        if restore_state:
            folder = self._restored_folder(self.project)
        elif folder == dataset_root and not (folder / "images").is_dir() and not any(
            child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES for child in folder.iterdir()
        ):
            folder = dataset_root / "unlabeled"
        if folder == self.dataset_dir / "labeled":
            (folder / "images").mkdir(parents=True, exist_ok=True)
            (folder / "labels").mkdir(parents=True, exist_ok=True)
        image_dir, label_dir = folder_layout(folder)
        if image_dir == self.dataset_dir / "unlabeled":
            image_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = image_dir
        self.labels_dir = label_dir
        self.current_path = None
        self.review_data = None
        self.review_info.clear()
        self.canvas.clear_image()
        self.refresh_box_list()
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.queue_filter.blockSignals(True)
        filter_index = self.queue_filter.findData(queue_filter_data)
        self.queue_filter.setCurrentIndex(filter_index if filter_index >= 0 else 0)
        self.queue_filter.blockSignals(False)
        self._last_filter_data = self.queue_filter.currentData()
        restored_image = self.project.last_image if restore_state else None
        self.refresh_queue(select_row=0, select_name=restored_image)
        self._save_session()
        self.statusBar().showMessage(f"Opened {self.source_dir}")

    def open_generator(self) -> None:
        if not self.maybe_leave_current():
            return
        dialog = DatasetGeneratorDialog(self.project, self)
        dialog.exec()

    def manage_classes(self) -> None:
        if not self.suggest_button.isEnabled():
            QMessageBox.information(self, "Suggestion in progress", "Wait for the current suggestion to finish.")
            return
        dialog = ClassManagerDialog(self.project.classes, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.classes == self.project.classes:
            return
        previous_count = len(self.project.classes)
        try:
            self.project.save_classes(dialog.classes)
        except Exception as exc:
            QMessageBox.warning(self, "Could not save classes", str(exc))
            return
        if isinstance(self.project.generator_settings, dict) and len(dialog.classes) > previous_count:
            included = list(self.project.generator_settings.get("included_class_ids", range(previous_count)))
            included.extend(range(previous_count, len(dialog.classes)))
            self.project.generator_settings["included_class_ids"] = included
        self._stop_suggestion_worker()
        selected = min(max(0, self.class_list.currentRow()), len(dialog.classes) - 1)
        self.class_names[:] = self.project.names
        self.class_colors[:] = self.project.colors
        self._fill_class_list()
        self._fill_queue_filter()
        self._last_filter_data = self.queue_filter.currentData()
        self.class_list.blockSignals(True)
        self.class_list.setCurrentRow(selected)
        self.class_list.blockSignals(False)
        self.canvas.active_class = selected
        self.canvas.redraw()
        self.refresh_box_list()
        self.apply_filter()
        self._save_session()

    def _stop_suggestion_worker(self) -> None:
        if self.suggestion_thread is not None:
            self.suggestion_thread.quit()
            self.suggestion_thread.wait()
            self.suggestion_thread.deleteLater()
            self.suggestion_thread = None
            self.suggestion_worker = None

    def _save_session(self) -> None:
        try:
            try:
                folder = self.source_dir.relative_to(self.dataset_dir)
            except ValueError:
                folder = self.source_dir
            self.project.last_folder = str(folder)
            self.project.last_image = self.current_path.name if self.current_path else None
            self.project.active_class = max(0, self.class_list.currentRow())
            self.project.save_state()
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save labeler state: {exc}")

    def refresh_queue(self, select_row: int | None = None, select_name: str | None = None) -> None:
        images = sorted(
            path for path in self.source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.image_class_ids = {path: self.label_class_ids(path) for path in images}
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for path in images:
            saved = self.label_path(path).exists()
            item = QListWidgetItem(("●  " if saved else "○  ") + path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            if saved:
                item.setForeground(QBrush(QColor("#56d6a5")))
            self.image_list.addItem(item)
        self.image_list.blockSignals(False)
        self.folder_label.setText(str(self.source_dir))
        self.folder_label.setToolTip(f"Images: {self.source_dir}\nLabels: {self.labels_dir}")
        self.finish_button.setEnabled(self.source_dir == self.dataset_dir / "unlabeled")
        self.queue_count.setText(f"{len(images)} images · {sum(self.label_path(path).exists() for path in images)} labeled")
        self.apply_filter()
        if select_row is not None and images:
            visible_rows = [index for index in range(self.image_list.count())
                            if not self.image_list.item(index).isHidden()]
            if visible_rows:
                restored_row = next((index for index in visible_rows
                                     if Path(self.image_list.item(index).data(Qt.ItemDataRole.UserRole)).name == select_name),
                                    None)
                self.image_list.setCurrentRow(
                    restored_row if restored_row is not None else
                    min(visible_rows, key=lambda index: abs(index - select_row))
                )
            else:
                self.current_path = None
                self.review_data = None
                self.review_info.clear()
                self.image_title.setText("No images match this filter")
                self.canvas.clear_image()
                self.refresh_box_list()
        elif not images:
            self.current_path = None
            self.review_data = None
            self.review_info.clear()
            self.image_title.setText(f"No images in {self.source_dir.name}")
            self.canvas.clear_image()
            self.refresh_box_list()

    def refresh_queue_from_button(self) -> None:
        if not self.maybe_leave_current():
            return
        row = max(0, self.image_list.currentRow())
        self.current_path = None
        self.refresh_queue(select_row=row)

    def delete_current_image(self) -> None:
        image_path = self.current_path
        if image_path is None:
            return
        if not self.suggest_button.isEnabled():
            QMessageBox.information(self, "Suggestion in progress", "Wait for the current suggestion to finish.")
            return
        related = [image_path]
        label_path = self.label_path(image_path)
        shares_label = any(
            path != image_path and path.is_file() and path.stem == image_path.stem
            and path.suffix.lower() in IMAGE_SUFFIXES
            for path in self.source_dir.iterdir()
        )
        if label_path.is_file() and not shares_label:
            related.append(label_path)
        review_path = metadata_path(image_path)
        if review_path.is_file():
            related.append(review_path)
        try:
            relative_paths = [path.relative_to(self.dataset_dir) for path in related]
        except ValueError:
            QMessageBox.warning(self, "Cannot delete image", "The selected image is outside this dataset root.")
            return
        trash_root = self.dataset_dir / ".trash"
        answer = QMessageBox.question(
            self, "Delete image",
            f"Move {image_path.name} and its associated files to the dataset trash?\n\n{trash_root}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        moved = []
        trash_dir = None
        try:
            trash_root.mkdir(parents=True, exist_ok=True)
            trash_dir = Path(tempfile.mkdtemp(prefix=f"{image_path.stem}-", dir=trash_root))
            for source, relative in zip(related, relative_paths):
                destination = trash_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
                moved.append((source, destination))
        except OSError as exc:
            rollback_errors = []
            for source, destination in reversed(moved):
                try:
                    destination.rename(source)
                except OSError as rollback_exc:
                    rollback_errors.append(str(rollback_exc))
            detail = f"\nSome files remain in {trash_dir}." if rollback_errors else ""
            QMessageBox.critical(self, "Delete failed", f"{exc}{detail}")
            return
        row = self.image_list.currentRow()
        self.current_path = None
        self.review_data = None
        self.review_info.clear()
        self.refresh_queue(select_row=row)
        self._save_session()
        self.statusBar().showMessage(f"Moved {image_path.name} to {trash_dir}")

    def on_queue_filter_changed(self, *_args) -> None:
        requested = self.queue_filter.currentData()
        previous = self._last_filter_data
        if isinstance(requested, str) and requested.startswith("class:") and (
            self.source_dir != self.dataset_dir / "labeled" / "images"
        ):
            self.switch_folder(self.dataset_dir / "labeled", dataset_root=self.dataset_dir,
                               queue_filter_data=requested)
            if self.source_dir == self.dataset_dir / "labeled" / "images":
                return
        elif self.maybe_leave_current():
            self._last_filter_data = requested
            self.apply_filter()
            self._select_visible_queue_item()
            return
        self.queue_filter.blockSignals(True)
        previous_index = self.queue_filter.findData(previous)
        self.queue_filter.setCurrentIndex(previous_index if previous_index >= 0 else 0)
        self.queue_filter.blockSignals(False)
        self.apply_filter()

    def _select_visible_queue_item(self) -> None:
        current = self.image_list.currentItem()
        if current is not None and not current.isHidden():
            return
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if not item.isHidden():
                self.image_list.setCurrentItem(item)
                return
        self.image_list.blockSignals(True)
        self.image_list.setCurrentRow(-1)
        self.image_list.blockSignals(False)
        self.current_path = None
        self.review_data = None
        self.review_info.clear()
        self.image_title.setText("No images match this filter")
        self.canvas.clear_image()
        self.refresh_box_list()
        self._save_session()

    def apply_filter(self, *_args) -> None:
        query = self.search.text().casefold()
        filter_mode = self.queue_filter.currentData()
        class_id = (int(filter_mode.split(":", 1)[1])
                    if isinstance(filter_mode, str) and filter_mode.startswith("class:") else None)
        matching = 0
        saved_count = 0
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            path = Path(item.data(Qt.ItemDataRole.UserRole))
            saved = self.label_path(path).exists()
            visible = query in path.name.casefold()
            if filter_mode == "needs_labels":
                visible = visible and not saved
            elif filter_mode == "has_labels":
                visible = visible and saved
            elif class_id is not None:
                visible = visible and saved and class_id in self.image_class_ids.get(path, frozenset())
            item.setHidden(not visible)
            matching += visible
            saved_count += saved
        if filter_mode == "all" and not query:
            self.queue_count.setText(f"{self.image_list.count()} images · {saved_count} labeled")
        else:
            self.queue_count.setText(f"{matching} matching of {self.image_list.count()} images")

    def _update_queue_status(self) -> None:
        saved_count = 0
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            path = Path(item.data(Qt.ItemDataRole.UserRole))
            saved = self.label_path(path).exists()
            self.image_class_ids[path] = self.label_class_ids(path)
            saved_count += saved
            item.setText(("●  " if saved else "○  ") + path.name)
            item.setForeground(QBrush(QColor("#56d6a5" if saved else "#e5eaf3")))
        self.queue_count.setText(f"{self.image_list.count()} images · {saved_count} labeled")
        self.apply_filter()
        if isinstance(self.queue_filter.currentData(), str) and self.queue_filter.currentData().startswith("class:"):
            QTimer.singleShot(0, self._select_visible_queue_item)

    def on_image_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        if path == self.current_path:
            return
        if not self.maybe_leave_current():
            self.image_list.blockSignals(True)
            if previous is None:
                self.image_list.setCurrentRow(-1)
            else:
                self.image_list.setCurrentItem(previous)
            self.image_list.blockSignals(False)
            return
        try:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                raise ValueError(f"Could not read {path}")
            boxes = read_labels(self.label_path(path), pixmap.width(), pixmap.height(), len(self.class_names))
            review_data = load_review_metadata(path) if path.parent == self.dataset_dir / "unlabeled" else None
            unknown_drafts = 0
            if review_data is not None:
                for draft in review_data["boxes"]:
                    if not isinstance(draft, dict) or not isinstance(draft.get("class_name"), str):
                        raise ValueError(f"Invalid draft class in {metadata_path(path)}")
                    if draft["class_name"] not in self.class_names:
                        unknown_drafts += 1
                        continue
                    coords = draft.get("xywhn")
                    confidence = draft.get("confidence")
                    if not isinstance(coords, list) or len(coords) != 4 or not all(
                        isinstance(value, (int, float)) and math.isfinite(value) for value in coords
                    ) or (confidence is not None and (
                        not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
                    )):
                        raise ValueError(f"Invalid draft box in {metadata_path(path)}")
                    cx, cy, bw, bh = coords
                    if not ((confidence is None or 0 <= confidence <= 1) and bw > 0 and bh > 0 and
                            -0.0001 <= cx - bw / 2 < cx + bw / 2 <= 1.0001 and
                            -0.0001 <= cy - bh / 2 < cy + bh / 2 <= 1.0001):
                        raise ValueError(f"Draft box is out of range in {metadata_path(path)}")
                    candidate = Box(self.class_names.index(draft["class_name"]),
                                    max(0, (cx - bw / 2) * pixmap.width()),
                                    max(0, (cy - bh / 2) * pixmap.height()),
                                    min(pixmap.width(), (cx + bw / 2) * pixmap.width()),
                                    min(pixmap.height(), (cy + bh / 2) * pixmap.height()),
                                    suggested=True, confidence=float(confidence) if confidence is not None else None)
                    if not any(self._box_iou(candidate, existing) > 0.8 and candidate.class_id == existing.class_id
                               for existing in boxes):
                        boxes.append(candidate)
            self.canvas.load_image(path, boxes)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open image", str(exc))
            self.image_list.blockSignals(True)
            if previous is None:
                self.image_list.setCurrentRow(-1)
            else:
                self.image_list.setCurrentItem(previous)
            self.image_list.blockSignals(False)
            return
        self.current_path = path
        self.review_data = review_data
        self.saved_signature = accepted_signature(boxes)
        self.undo_history.clear()
        self.redo_history.clear()
        self.image_title.setText(path.name)
        reason = review_data.get("selection_reason", "review") if review_data else ""
        review_text = f"Captured for review: {str(reason).replace('_', ' ')}" if review_data else ""
        if unknown_drafts:
            review_text += f" · {unknown_drafts} suggestions with unknown classes skipped"
        self.review_info.setText(review_text)
        self.refresh_box_list()
        self._save_session()
        self.statusBar().showMessage(f"Editing {path.name}")

    def maybe_leave_current(self) -> bool:
        if self.current_path is None:
            return True
        if self.review_data is not None:
            return self._sync_review_state()
        if any(box.suggested for box in self.canvas.boxes):
            answer = QMessageBox.question(
                self, "Unreviewed suggestions",
                "Discard unaccepted suggestions and leave this image? Accepted boxes will be saved.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        return self.save_current(force=False)

    def _sync_review_state(self) -> bool:
        if self.current_path is None or self.review_data is None:
            return True
        if not self.save_current(force=False):
            return False
        width, height = self.canvas.image_width, self.canvas.image_height
        pending = []
        for box in self.canvas.boxes:
            if not box.suggested:
                continue
            pending.append({
                "class_name": self.class_names[box.class_id],
                "confidence": box.confidence,
                "xywhn": [(box.x1 + box.x2) / (2 * width), (box.y1 + box.y2) / (2 * height),
                           (box.x2 - box.x1) / width, (box.y2 - box.y1) / height],
            })
        updated = {**self.review_data, "boxes": pending}
        try:
            save_review_metadata(self.current_path, updated)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save review draft", str(exc))
            return False
        self.review_data = updated
        return True

    def save_current(self, _checked=False, *, force: bool = True) -> bool:
        if self.current_path is None:
            return True
        signature = accepted_signature(self.canvas.boxes)
        if not force and signature == self.saved_signature:
            return True
        width, height = self.canvas.image_width, self.canvas.image_height
        lines = []
        for box in self.canvas.boxes:
            if box.suggested:
                continue
            if not (0 <= box.class_id < len(self.class_names) and
                    0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
                QMessageBox.warning(self, "Invalid box", "Fix boxes that extend outside the image before saving.")
                return False
            cx = (box.x1 + box.x2) / (2 * width)
            cy = (box.y1 + box.y2) / (2 * height)
            bw = (box.x2 - box.x1) / width
            bh = (box.y2 - box.y1) / height
            lines.append(f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path = self.label_path(self.current_path)
        temp_path = label_path.with_suffix(".txt.tmp")
        try:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            os.replace(temp_path, label_path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.saved_signature = signature
        self._update_queue_status()
        self.statusBar().showMessage(f"Saved {label_path.name}")
        return True

    def finish_current(self) -> None:
        if self.current_path is None:
            return
        if self.source_dir != self.dataset_dir / "unlabeled":
            return
        if any(box.suggested for box in self.canvas.boxes):
            QMessageBox.information(self, "Review suggestions", "Accept or delete all dashed suggestions first.")
            return
        if not self.save_current(force=True):
            return
        if not self._sync_review_state():
            return
        image_path = self.current_path
        label_path = self.label_path(image_path)
        destination_images = self.dataset_dir / "labeled" / "images"
        destination_labels = self.dataset_dir / "labeled" / "labels"
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)
        image_target = destination_images / image_path.name
        label_target = destination_labels / label_path.name
        review_source = metadata_path(image_path)
        review_target = metadata_path(image_target)
        if image_target.exists() or label_target.exists() or (review_source.exists() and review_target.exists()):
            QMessageBox.warning(self, "Name collision", f"A destination file already exists for {image_path.name}.")
            return
        moves = [(image_path, image_target), (label_path, label_target)]
        if review_source.exists():
            review_target.parent.mkdir(parents=True, exist_ok=True)
            moves.append((review_source, review_target))
        completed = []
        try:
            for source, target in moves:
                source.rename(target)
                completed.append((source, target))
        except OSError as exc:
            for source, target in reversed(completed):
                target.rename(source)
            QMessageBox.critical(self, "Move failed", str(exc))
            return
        row = self.image_list.currentRow()
        self.current_path = None
        self.review_data = None
        self.review_info.clear()
        self.refresh_queue(select_row=row)
        self._save_session()
        self.statusBar().showMessage(f"Moved {image_path.name} and {label_path.name} to labeled")

    def previous_image(self) -> None:
        self._step_image(-1)

    def next_image(self) -> None:
        self._step_image(1)

    def _step_image(self, direction: int) -> None:
        row = self.image_list.currentRow() + direction
        while 0 <= row < self.image_list.count():
            if not self.image_list.item(row).isHidden():
                self.image_list.setCurrentRow(row)
                return
            row += direction

    def refresh_box_list(self) -> None:
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for index, box in enumerate(self.canvas.boxes):
            text = f"{index + 1:02d}  {self.class_names[box.class_id]}"
            if box.suggested:
                text += f"   · {box.confidence:.0%} suggestion" if box.confidence is not None else "   · suggestion"
            item = QListWidgetItem(text)
            item.setForeground(QBrush(QColor(self.class_colors[box.class_id])))
            self.box_list.addItem(item)
        self.box_list.setCurrentRow(self.canvas.selected)
        self.box_list.blockSignals(False)
        accepted = sum(not box.suggested for box in self.canvas.boxes)
        suggested = len(self.canvas.boxes) - accepted
        self.box_count.setText(f"{accepted} accepted · {suggested} suggestions")

    def on_canvas_selection(self, index: int) -> None:
        self.box_list.blockSignals(True)
        self.box_list.setCurrentRow(index)
        self.box_list.blockSignals(False)
        if index >= 0:
            self.class_list.blockSignals(True)
            self.class_list.setCurrentRow(self.canvas.boxes[index].class_id)
            self.class_list.blockSignals(False)
            self.canvas.active_class = self.canvas.boxes[index].class_id
            self._save_session()
        self.refresh_box_list()

    def on_box_selected(self, index: int) -> None:
        self.canvas.select(index)

    def on_class_selected(self, index: int) -> None:
        if index < 0:
            return
        self.canvas.active_class = index
        selected = self.canvas.selected
        if self.current_path is not None and selected >= 0 and self.canvas.boxes[selected].class_id != index:
            before = self.canvas.snapshot()
            self.canvas.boxes[selected].class_id = index
            self.canvas.redraw()
            self.record_change(before)
        self._save_session()

    def record_change(self, before: list[Box]) -> None:
        if before != self.canvas.boxes:
            self.undo_history.append([replace(box) for box in before])
            self.redo_history.clear()
            if self.review_data is not None:
                self._sync_review_state()
        self.refresh_box_list()

    def undo(self) -> None:
        if not self.undo_history:
            return
        self.redo_history.append(self.canvas.snapshot())
        self.canvas.set_boxes(self.undo_history.pop())
        if self.review_data is not None:
            self._sync_review_state()
        self.refresh_box_list()

    def redo(self) -> None:
        if not self.redo_history:
            return
        self.undo_history.append(self.canvas.snapshot())
        self.canvas.set_boxes(self.redo_history.pop())
        if self.review_data is not None:
            self._sync_review_state()
        self.refresh_box_list()

    def delete_selected(self) -> None:
        index = self.canvas.selected
        if index < 0:
            return
        before = self.canvas.snapshot()
        self.canvas.boxes.pop(index)
        self.canvas.select(-1)
        self.record_change(before)

    def accept_selected(self) -> None:
        index = self.canvas.selected
        if index < 0 or not self.canvas.boxes[index].suggested:
            return
        before = self.canvas.snapshot()
        self.canvas.boxes[index].suggested = False
        self.canvas.redraw()
        self.record_change(before)

    def accept_all(self) -> None:
        if not any(box.suggested for box in self.canvas.boxes):
            return
        before = self.canvas.snapshot()
        for box in self.canvas.boxes:
            box.suggested = False
        self.canvas.redraw()
        self.record_change(before)

    def suggest_boxes(self) -> None:
        if self.current_path is None:
            return
        source = self.suggestion_source.currentData()
        required = [Path(CHECKPOINT_PATH)] if source == "trained" else [Path(YOLOE_MODEL_PATH), Path(YOLOE_PROMPT_PROFILE)]
        for path in required:
            if not path.expanduser().resolve().is_file():
                QMessageBox.warning(self, "Model file missing", f"Set the model paths at the top of labeler.py.\n{path}")
                return
        if self.suggestion_thread is None:
            self.suggestion_thread = QThread(self)
            self.suggestion_worker = SuggestionWorker(self.class_names)
            self.suggestion_worker.moveToThread(self.suggestion_thread)
            self.request_suggestions.connect(self.suggestion_worker.suggest)
            self.suggestion_worker.finished.connect(self.on_suggestions)
            self.suggestion_worker.failed.connect(self.on_suggestion_error)
            self.suggestion_thread.finished.connect(self.suggestion_worker.deleteLater)
            self.suggestion_thread.start()
        self.suggest_button.setEnabled(False)
        self.statusBar().showMessage(f"Running {self.suggestion_source.currentText()} predictions...")
        self.request_suggestions.emit(str(self.current_path), source)

    def on_suggestions(self, image_path: str, predictions: list[tuple]) -> None:
        self.suggest_button.setEnabled(True)
        if self.current_path is None or image_path != str(self.current_path):
            return
        width, height = self.canvas.image_width, self.canvas.image_height
        before = self.canvas.snapshot()
        added = 0
        for class_id, cx, cy, bw, bh in predictions:
            x1, y1 = max(0, (cx - bw / 2) * width), max(0, (cy - bh / 2) * height)
            x2, y2 = min(width, (cx + bw / 2) * width), min(height, (cy + bh / 2) * height)
            if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
                continue
            candidate = Box(class_id, x1, y1, x2, y2, suggested=True)
            if any(self._box_iou(candidate, existing) > 0.8 and candidate.class_id == existing.class_id
                   for existing in self.canvas.boxes):
                continue
            self.canvas.boxes.append(candidate)
            added += 1
        self.canvas.redraw()
        self.record_change(before)
        self.statusBar().showMessage(f"Added {added} editable suggestions")

    @staticmethod
    def _box_iou(a: Box, b: Box) -> float:
        intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0

    def on_suggestion_error(self, image_path: str, message: str) -> None:
        self.suggest_button.setEnabled(True)
        if self.current_path is not None and image_path == str(self.current_path):
            QMessageBox.warning(self, "Suggestion failed", message)
        self.statusBar().showMessage("Model suggestion failed")

    def closeEvent(self, event) -> None:
        if not self.maybe_leave_current():
            event.ignore()
            return
        self._save_session()
        self._stop_suggestion_worker()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    try:
        window = LabelerWindow()
    except Exception as exc:
        QMessageBox.critical(None, "Labeler could not start", str(exc))
        raise SystemExit(1) from exc
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
