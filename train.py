"""Train a YOLO26 detection model. Edit the settings below, then run this file."""

from multiprocessing import freeze_support
from pathlib import Path

from dataset_project import latest_generated_version, recent_dataset
from training_dashboard import attach_dashboard


# Use the latest generated version in the most recently opened dataset.
# Set a specific version path here to train that version instead.
DATASET_PATH = None
# MODEL = "yolo26m.pt"  # Pretrained medium model used for a new run.
MODEL = r"C:\Users\Abhil\Desktop\yolo26\runs\yolo26m\weights\best.pt"

# EPOCHS = 250  # Total for "new"; additional epochs for "continue".
EPOCHS = 50

IMAGE_SIZE = 1024
BATCH_SIZE = 4
DEVICE = 0  # First NVIDIA GPU; use "cpu" to train without CUDA.
WORKERS = 4
RUNS_DIR = Path(__file__).resolve().parent / "runs"
RUN_NAME = "yolo26m"
TRAIN_MODE = "new"  # "new", "resume" an interrupted run, or "continue" a completed run.

RESUME_CHECKPOINT_PATH = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
CONTINUE_CHECKPOINT_PATH = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
CONTINUE_RUN_NAME = f"{RUN_NAME}_continue"
OPEN_DASHBOARD = True  # Open a live browser dashboard; a PNG is also saved in the run folder.


def resume_checkpoint_state(checkpoint: dict | None, error_message: str):
    """Validate resumability; callers retain their normal/distillation-specific policy."""
    saved = checkpoint or {}
    arguments = saved.get("train_args") or {}
    epoch, epochs = saved.get("epoch", -1), arguments.get("epochs", 0)
    if epoch < 0 or saved.get("optimizer") is None or epoch + 1 >= epochs:
        raise ValueError(error_message)
    return arguments, epoch, epochs


def main() -> None:
    from ultralytics import YOLO
    if DATASET_PATH is None:
        dataset_root = recent_dataset(Path(__file__).resolve().parent / "datasets" / "rivals")
        dataset_path = latest_generated_version(dataset_root)
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
        checkpoint_path = RESUME_CHECKPOINT_PATH if TRAIN_MODE == "resume" else CONTINUE_CHECKPOINT_PATH
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))

    attach_dashboard(model, open_browser=OPEN_DASHBOARD)

    if TRAIN_MODE == "resume":
        resume_checkpoint_state(model.ckpt,
            "This checkpoint cannot resume an interrupted run. "
            "Set TRAIN_MODE = 'continue' to train further from its weights.")
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
