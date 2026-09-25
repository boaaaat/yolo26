"""Press X to capture numbered, unlabeled training images. Press Ctrl+C to stop."""

import time
from pathlib import Path

import cv2
import dxcam
import win32api


# Settings
DATASET_DIR = Path(__file__).resolve().parent / "datasets"
OUTPUT_DIR_NAME = "unlabeled"
IMAGE_SIZE = 1024
JPEG_QUALITY = 95
DEVICE_INDEX = 0
OUTPUT_INDEX = 0  # Primary monitor on the selected graphics device.
POLL_INTERVAL_SECONDS = 0.01

X_KEY = ord("X")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test", "labeled")


def image_files(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    dataset_dir = Path(DATASET_DIR).expanduser().resolve()
    (dataset_dir / "labeled" / "images").mkdir(parents=True, exist_ok=True)
    split_folders = [dataset_dir / split / "images" for split in SPLITS]
    source_images = [path for folder in split_folders for path in image_files(folder)]
    next_number = len(source_images) + 1

    output_dir = dataset_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    used_numbers = {
        int(path.stem)
        for path in (*source_images, *image_files(output_dir))
        if path.stem.isdecimal()
    }
    while next_number in used_numbers:
        next_number += 1

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
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )
                    if not encoded_ok:
                        raise RuntimeError("Could not encode the screenshot as JPEG")

                    while True:
                        while next_number in used_numbers:
                            next_number += 1
                        image_path = output_dir / f"{next_number}.jpg"
                        try:
                            with image_path.open("xb") as image_file:
                                image_file.write(encoded.tobytes())
                            break
                        except FileExistsError:
                            used_numbers.add(next_number)
                            next_number += 1

                    print(f"Saved {image_path}")
                    used_numbers.add(next_number)
                    next_number += 1
            was_down = is_down
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Stopped collecting images.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
