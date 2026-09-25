"""Run a YOLO26 checkpoint on every frame of a video and save the annotated video."""

from pathlib import Path

from ultralytics import YOLO


# Settings
CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26m" / "weights" / "best.pt"
VIDEO_PATH = Path(__file__).resolve().parent / "recordings" / "recording.mp4"
OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "video_test"
IMAGE_SIZE = 1024
CONFIDENCE = 0.25
DEVICE = 0  # First NVIDIA GPU; use "cpu" if needed.


def main() -> None:
    checkpoint = Path(CHECKPOINT_PATH).expanduser().resolve()
    video = Path(VIDEO_PATH).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    model = YOLO(str(checkpoint))
    frame_count = 0
    results = model.predict(
        source=str(video),
        stream=True,
        save=True,
        vid_stride=1,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        project=str(Path(OUTPUT_DIR).expanduser().resolve()),
        name=video.stem,
        verbose=False,
    )
    for _ in results:
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    print(f"Finished {frame_count} frames. Annotated video saved in {model.predictor.save_dir}")


if __name__ == "__main__":
    main()
