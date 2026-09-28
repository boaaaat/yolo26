"""Distill a trained YOLO26 detection run. Edit the settings below, then run this file."""

import math
from multiprocessing import freeze_support
from pathlib import Path

import yaml
from ultralytics import YOLO

from training_dashboard import attach_dashboard
from train import resume_checkpoint_state

from dataset_utils import (normalize_names)


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
TEACHER_RUN = RUNS_DIR / "yolo26m"  # Run folder, or its weights/best.pt or weights/last.pt.
STUDENT_MODEL = "yolo26n.pt"  # Used for a new run. Use "yolo26s.pt" for small.
DATASET_PATH = None  # Use the saved dataset; set a folder or data.yaml path if it moved.
EPOCHS = 200  # Total epochs for a new run; resume keeps the checkpoint's original total.
IMAGE_SIZE = None  # Use the teacher run's size for new training or the student checkpoint's size for resume.
BATCH_SIZE = 4  # The teacher and student both need GPU memory.
DEVICE = 0
WORKERS = 4
DISTILL_WEIGHT = 6.0
RUN_NAME = None  # Default: <teacher run>-<student model>-distill.
TRAIN_MODE = "new"  # "new" or "resume" an interrupted distillation run.
CHECKPOINT_PATH = RUNS_DIR / "yolo26m-yolo26n-distill" / "weights" / "last.pt"  # Set to the interrupted student run.
OPEN_DASHBOARD = True


def run_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if not candidate.is_absolute() and (RUNS_DIR / candidate).exists():
        return (RUNS_DIR / candidate).resolve()
    raise FileNotFoundError(f"Training run or checkpoint not found: {value}")


def teacher_checkpoint(path: Path) -> tuple[Path, Path]:
    if path.is_file():
        if path.suffix.lower() != ".pt" or path.parent.name != "weights":
            raise ValueError("A checkpoint path must point to a .pt file inside a run's weights folder")
        return path, path.parent.parent
    if not path.is_dir():
        raise ValueError(f"Not a training run directory: {path}")
    for name in ("best.pt", "last.pt"):
        checkpoint = path / "weights" / name
        if checkpoint.is_file():
            return checkpoint, path
    raise FileNotFoundError(f"No best.pt or last.pt found in {path / 'weights'}")


def run_settings(run_dir: Path) -> dict:
    path = run_dir / "args.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Training settings not found: {path}. Use a run created by Ultralytics.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid training settings: {path}")
    return data


def dataset_yaml(value: str | Path | None, settings: dict, run_dir: Path) -> Path:
    raw = value or settings.get("data")
    if not isinstance(raw, (str, Path)) or not raw:
        raise ValueError(f"No dataset recorded in {run_dir / 'args.yaml'}; set DATASET_PATH above")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.is_dir():
        path /= "data.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}. Set DATASET_PATH above to its current path.")
    return path


def dataset_names(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML: {path}")
    return normalize_names(data.get("names"))


def main() -> None:
    if TRAIN_MODE not in {"new", "resume"}:
        raise ValueError("TRAIN_MODE must be 'new' or 'resume'")
    if BATCH_SIZE < 1 or WORKERS < 0:
        raise ValueError("BATCH_SIZE must be positive and WORKERS must be nonnegative")

    if TRAIN_MODE == "resume":
        checkpoint = Path(CHECKPOINT_PATH).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Student checkpoint not found: {checkpoint}")
        student = YOLO(str(checkpoint))
        if student.task != "detect":
            raise ValueError(f"Student checkpoint must be a detection model: {checkpoint}")
        saved_args, saved_epoch, saved_epochs = resume_checkpoint_state(student.ckpt,
            "This checkpoint cannot resume an interrupted distillation run. "
            "Use its unstripped last.pt before the original epoch total is reached.")
        teacher_path = saved_args.get("distill_model")
        if not isinstance(teacher_path, str) or not teacher_path:
            raise ValueError("This checkpoint has no saved distillation teacher; it is not a distillation run.")
        if not Path(teacher_path).expanduser().is_file():
            raise FileNotFoundError(f"Saved teacher checkpoint not found: {teacher_path}")
        data_path = dataset_yaml(DATASET_PATH, saved_args, checkpoint.parent.parent)
        image_size = IMAGE_SIZE if IMAGE_SIZE is not None else saved_args.get("imgsz", 1024)
    else:
        if EPOCHS < 1 or not math.isfinite(DISTILL_WEIGHT) or DISTILL_WEIGHT <= 0:
            raise ValueError("EPOCHS and DISTILL_WEIGHT must be positive")
        path = run_path(TEACHER_RUN)
        checkpoint, run_dir = teacher_checkpoint(path)
        settings = run_settings(run_dir)
        if settings.get("task") not in (None, "detect"):
            raise ValueError(f"Teacher run must be an object detection run, got {settings.get('task')!r}")
        data_path = dataset_yaml(DATASET_PATH, settings, run_dir)
        image_size = IMAGE_SIZE if IMAGE_SIZE is not None else settings.get("imgsz", 1024)
        student_path = Path(STUDENT_MODEL).expanduser()
        selected_student = str(student_path.resolve()) if student_path.is_file() else str(STUDENT_MODEL)
        if not selected_student.lower().endswith(".pt"):
            raise ValueError("STUDENT_MODEL must be a YOLO26 .pt checkpoint, such as yolo26n.pt or yolo26s.pt")
        teacher = YOLO(str(checkpoint))
        if teacher.task != "detect":
            raise ValueError(f"Teacher checkpoint must be a detection model: {checkpoint}")
        teacher_names = normalize_names(teacher.names)
        names = dataset_names(data_path)
        if teacher_names != names:
            raise ValueError(f"Teacher classes {teacher_names} differ from dataset classes {names}. "
                             "Set DATASET_PATH to the dataset used for this run.")
        del teacher
        student = YOLO(selected_student)
        if student.task != "detect":
            raise ValueError(f"Student model must be a detection model: {selected_student}")
        name = RUN_NAME or f"{run_dir.name}-{Path(selected_student).stem}-distill"

    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError("IMAGE_SIZE or the saved imgsz must be a positive integer")
    attach_dashboard(student, open_browser=OPEN_DASHBOARD)

    if TRAIN_MODE == "resume":
        print(f"Resuming student: {checkpoint} from epoch {saved_epoch + 2} of {saved_epochs}")
        print(f"Saved teacher: {teacher_path}")
        print(f"Dataset: {data_path}")
        # Ultralytics restores the saved optimizer, epoch, teacher, and original total epochs.
        # EPOCHS, STUDENT_MODEL, DISTILL_WEIGHT, and RUN_NAME do not change this run.
        student.train(
            resume=True,
            data=str(data_path),
            imgsz=image_size,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
        )
        print(f"Student run saved to: {student.trainer.save_dir}")
        return

    print(f"Teacher: {checkpoint}")
    print(f"Student: {selected_student}")
    print(f"Dataset: {data_path}")
    print(f"Requested student run: {RUNS_DIR / name}")
    student.train(
        data=str(data_path),
        distill_model=str(checkpoint),
        dis=DISTILL_WEIGHT,
        epochs=EPOCHS,
        imgsz=image_size,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name=name,
    )
    print(f"Student run saved to: {student.trainer.save_dir}")


if __name__ == "__main__":
    freeze_support()  # Required for Windows data-loader workers.
    main()
