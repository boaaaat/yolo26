"""Press X to capture numbered, unlabeled training images. Press Ctrl+C to stop."""

from inference_collection import next_image_number, save_candidate
import time
from pathlib import Path

import cv2
import dxcam
import win32api

from dataset_utils import (image_files as iter_image_files)


# Settings
DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "rivals"
OUTPUT_DIR_NAME = "unlabeled"
IMAGE_SIZE = 1024
JPEG_QUALITY = 95
DEVICE_INDEX = 0
OUTPUT_INDEX = 0  # Primary monitor on the selected graphics device.
POLL_INTERVAL_SECONDS = 0.01

X_KEY = ord("X")
SPLITS = ("train", "valid", "test", "labeled")


def image_files(folder: Path):
    return iter_image_files(folder, missing_ok=False)


def main() -> None:
    dataset_dir = Path(DATASET_DIR).expanduser().resolve()
    (dataset_dir / "labeled" / "images").mkdir(parents=True, exist_ok=True)
    split_folders = [dataset_dir / split / "images" for split in SPLITS]
    source_images = [path for folder in split_folders for path in image_files(folder)]
    next_number = len(source_images) + 1

    output_dir = dataset_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    next_number, used_numbers = next_image_number(
        [*source_images, *image_files(output_dir)], start_at=next_number)

    print(f"Found {len(source_images)} images across train, valid, test, and labeled.")
    print(f"Press X to save {next_number}.jpg in {output_dir}. Press Ctrl+C to stop.")

    camera = dxcam.create(device_idx=DEVICE_INDEX, output_idx=OUTPUT_INDEX, output_color="BGR")
    was_down = bool(win32api.GetAsyncKeyState(X_KEY) & 0x8000)
    try:
        while True:
            is_down = bool(win32api.GetAsyncKeyState(X_KEY) & 0x8000)
            if is_down and not was_down:
                frame = camera.grab(new_frame_only=False)
                if frame is None:
                    print("No frame available; press X again.")
                else:
                    resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                    image_path, next_number = save_candidate(
                        output_dir, resized, None, next_number, used_numbers,
                        jpeg_quality=JPEG_QUALITY, check_folders=False)
                    print(f"Saved {image_path}")
            was_down = is_down
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Stopped collecting images.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
