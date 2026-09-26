"""Generate YOLO detection labels from a trained checkpoint or YOLOE. Edit settings below."""

from pathlib import Path

import yaml
from ultralytics import YOLO, YOLOE


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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    image_dir = Path(IMAGE_DIR).expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_dir}")

    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    if len({path.stem.casefold() for path in images}) != len(images):
        raise ValueError("Images with the same name and different extensions would share a label file")

    # Ultralytics expects a sibling labels/ folder when images are in images/.
    # For datasets/rivals/unlabeled, labels sit next to their matching images.
    label_dir = image_dir.parent / "labels" if image_dir.name.lower() == "images" else image_dir
    label_dir.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(Path(DATA_YAML).expanduser().resolve().read_text(encoding="utf-8"))
    dataset_names = data["names"]
    if isinstance(dataset_names, dict):
        dataset_names = [name for _, name in sorted(dataset_names.items(), key=lambda pair: int(pair[0]))]
    if not isinstance(dataset_names, list) or not dataset_names:
        raise ValueError("DATA_YAML must contain an ordered list of class names")

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

    class_names = dict(model.names.items()) if isinstance(model.names, dict) else dict(enumerate(model.names))
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
        lines = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = class_names[class_id]
                if float(box.conf.item()) < thresholds.get(class_name, default_threshold):
                    continue
                dataset_class_id = dataset_names.index(class_name)
                x, y, width, height = box.xywhn[0].tolist()
                lines.append(
                    f"{dataset_class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"
                )

        # An empty file is the YOLO label for an image with no accepted objects.
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
        print(f"{image_path.name} -> {label_path.name}: {len(lines)} objects")

    print(f"Done: wrote {written} labels, skipped {skipped} existing labels in {label_dir}")


if __name__ == "__main__":
    main()
