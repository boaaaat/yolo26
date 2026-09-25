"""Train a YOLO26 detection model. Edit the settings below, then run this file."""

from multiprocessing import freeze_support
from pathlib import Path

from dataset_project import recent_dataset
from ultralytics import YOLO


# Train on generated v1. Set to None to use the latest generated version instead.
DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "versions" / "v1"
MODEL = "yolo26m.pt"  # Pretrained medium model used for a new run.
EPOCHS = 250  # Total for "new"; additional epochs for "continue".
IMAGE_SIZE = 1024
BATCH_SIZE = 4
DEVICE = 0  # First NVIDIA GPU; use "cpu" to train without CUDA.
WORKERS = 4
RUNS_DIR = Path(__file__).resolve().parent / "runs"
RUN_NAME = "yolo26m"
TRAIN_MODE = "new"  # "new", "resume" an interrupted run, or "continue" a completed run.
CHECKPOINT_PATH = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
CONTINUE_RUN_NAME = f"{RUN_NAME}_continue"


def main() -> None:
    if DATASET_PATH is None:
        dataset_root = recent_dataset(Path(__file__).resolve().parent / "datasets")
        versions_dir = dataset_root / "versions"
        versions = [path for path in versions_dir.iterdir()
                    if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit()
                    and (path / "data.yaml").is_file()] if versions_dir.is_dir() else []
        if not versions:
            raise FileNotFoundError(
                "No generated dataset found. Use Generate dataset in labeler.py first, "
                "or set DATASET_PATH to an existing data.yaml."
            )
        dataset_path = max(versions, key=lambda path: int(path.name[1:]))
    else:
        dataset_path = Path(DATASET_PATH).expanduser().resolve()
    data_yaml = dataset_path / "data.yaml" if dataset_path.is_dir() else dataset_path
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. Set DATASET_PATH above to a "
            "dataset directory containing data.yaml or to the YAML file itself."
        )

    if TRAIN_MODE not in {"new", "resume", "continue"}:
        raise ValueError("TRAIN_MODE must be 'new', 'resume', or 'continue'")

    if TRAIN_MODE == "new":
        model = YOLO(MODEL)
    else:
        checkpoint = Path(CHECKPOINT_PATH).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))

    if TRAIN_MODE == "resume":
        saved = model.ckpt or {}
        saved_epoch = saved.get("epoch", -1)
        saved_epochs = saved.get("train_args", {}).get("epochs", 0)
        if saved_epoch < 0 or saved.get("optimizer") is None or saved_epoch + 1 >= saved_epochs:
            raise ValueError(
                "This checkpoint cannot resume an interrupted run. "
                "Set TRAIN_MODE = 'continue' to train further from its weights."
            )
        # Ultralytics restores the saved epoch, optimizer, and original total epochs.
        # EPOCHS and RUN_NAME below do not change the interrupted run.
        model.train(
            resume=True,
            data=str(data_yaml),
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
        )
        return

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name=RUN_NAME if TRAIN_MODE == "new" else CONTINUE_RUN_NAME,
    )


if __name__ == "__main__":
    freeze_support()  # Required for Windows data-loader workers.
    main()
