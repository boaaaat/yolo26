"""Generate YOLO detection labels from a trained checkpoint or YOLOE. Edit settings below."""

from pathlib import Path

import yaml
from ultralytics import YOLO, YOLOE

from dataset_utils import (
    class_name_map, normalize_names, normalized_predictions,
    write_yolo_labels, image_files,
)


# Settings
MODEL_SOURCE = "trained"  # "trained" or "yoloe"; trained is much better on this Roblox dataset.
CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26m" / "weights" / "best.pt"
YOLOE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yoloe-26s-seg.pt"
YOLOE_PROMPT_PROFILE = Path(__file__).resolve().parent / "models" / "roblox-yoloe-26s-visual.npz"
DATA_YAML = Path(__file__).resolve().parent / "datasets" / "rivals" / "data.yaml"
IMAGE_DIR = Path(__file__).resolve().parent / "datasets" / "rivals" / "unlabeled"
MIN_CONFIDENCE_BY_CLASS = {
    "dead": 0.50,
    "enemy": 0.50,
    "katana": 0.50,
    "teammate": 0.50,
}
YOLOE_MIN_CONFIDENCE_BY_CLASS = {
    "dead": 0.25,
    "enemy": 0.25,
    "katana": 0.25,
    "teammate": 0.25,
}
IMAGE_SIZE = 1024
DEVICE = 0  # First NVIDIA GPU; use "cpu" if needed.
OVERWRITE_EXISTING_LABELS = False


def main() -> None:
    image_dir = Path(IMAGE_DIR).expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_dir}")

    images = sorted(image_files(image_dir))
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    if len({path.stem.casefold() for path in images}) != len(images):
        raise ValueError("Images with the same name and different extensions would share a label file")

    # Ultralytics expects a sibling labels/ folder when images are in images/.
    # For datasets/rivals/unlabeled, labels sit next to their matching images.
    label_dir = image_dir.parent / "labels" if image_dir.name.lower() == "images" else image_dir
    label_dir.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(Path(DATA_YAML).expanduser().resolve().read_text(encoding="utf-8"))
    dataset_names = normalize_names(data["names"])

    if MODEL_SOURCE == "trained":
        checkpoint = Path(CHECKPOINT_PATH).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))
        if model.task != "detect":
            raise ValueError(f"Expected a detection checkpoint, got task={model.task!r}")
        thresholds = MIN_CONFIDENCE_BY_CLASS
    elif MODEL_SOURCE == "yoloe":
        model_path = Path(YOLOE_MODEL_PATH).expanduser().resolve()
        profile_path = Path(YOLOE_PROMPT_PROFILE).expanduser().resolve()
        for path in (model_path, profile_path):
            if not path.is_file():
                raise FileNotFoundError(f"YOLOE file not found: {path}")
        model = YOLOE(str(model_path))
        model.load_prompt_embeddings(profile_path)
        thresholds = YOLOE_MIN_CONFIDENCE_BY_CLASS
        print("YOLOE visual prompts are experimental on this Roblox dataset; review every label.")
    else:
        raise ValueError("MODEL_SOURCE must be 'trained' or 'yoloe'")

    class_names = class_name_map(model.names)
    model_names = set(class_names.values())
    if not model_names.issubset(dataset_names):
        raise ValueError(
            "Every model class needs a data.yaml class. "
            f"Model classes: {class_names}; dataset classes: {dataset_names}"
        )
    for name, threshold in thresholds.items():
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"Confidence for {name!r} must be a number from 0 to 1")

    default_threshold = 0.50 if MODEL_SOURCE == "trained" else 0.25
    inference_confidence = min(thresholds.get(name, default_threshold) for name in model_names)
    written = skipped = 0
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not OVERWRITE_EXISTING_LABELS:
            skipped += 1
            continue

        result = model.predict(
            source=str(image_path),
            conf=inference_confidence,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )[0]
        labels = []
        dataset_ids = {name: index for index, name in enumerate(dataset_names)}
        for class_id, cx, cy, width, height, confidence in normalized_predictions(result):
            class_name = class_names[class_id]
            if confidence >= thresholds.get(class_name, default_threshold):
                labels.append((dataset_ids[class_name], cx, cy, width, height))
        # An empty file is the YOLO label for an image with no accepted objects.
        write_yolo_labels(label_path, labels, precision=6)
        written += 1
        print(f"{image_path.name} -> {label_path.name}: {len(labels)} objects")

    print(f"Done: wrote {written} labels, skipped {skipped} existing labels in {label_dir}")


if __name__ == "__main__":
    main()
