"""Build a YOLOE visual prompt profile from this dataset's training boxes."""

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn.functional import normalize
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from dataset_project import latest_generated_version


# Change these values here; this script has no command line arguments.
ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "datasets" / "rivals"
MODEL_PATH = ROOT / "models" / "yoloe-26s-seg.pt"
PROFILE_PATH = ROOT / "models" / "roblox-yoloe-26s-visual.npz"
REFERENCES_PER_CLASS = 16
DEVICE = 0
IMAGE_SIZE = 1024


def main() -> None:
    version = latest_generated_version(DATASET_ROOT)
    train_images = version / "train" / "images"
    train_labels = version / "train" / "labels"
    data = yaml.safe_load((version / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        names = [value for _, value in sorted(names.items(), key=lambda pair: int(pair[0]))]
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    candidates: dict[int, list[tuple[float, Path, np.ndarray]]] = {i: [] for i in range(len(names))}
    for label_path in train_labels.glob("*.txt"):
        image_path = next((train_images / f"{label_path.stem}{ext}" for ext in (".jpg", ".jpeg", ".png")
                          if (train_images / f"{label_path.stem}{ext}").is_file()), None)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_text, cx, cy, width, height = line.split()
            class_id = int(class_text)
            cx, cy, width, height = map(float, (cx, cy, width, height))
            area = width * height
            if not (0.012 <= area <= 0.30 and height >= 0.15 and cy >= 0.17):
                continue
            xyxy = np.array([(cx - width / 2) * image_width, (cy - height / 2) * image_height,
                             (cx + width / 2) * image_width, (cy + height / 2) * image_height],
                            dtype=np.float32)
            candidates[class_id].append((area, image_path, xyxy))

    model = YOLOE(str(MODEL_PATH))
    embeddings = []
    for class_id, name in enumerate(names):
        examples = sorted(candidates[class_id], key=lambda item: item[0], reverse=True)
        if len(examples) < REFERENCES_PER_CLASS:
            raise ValueError(f"Only {len(examples)} reference boxes for {name}")
        vectors = []
        used_images = set()
        for _, image_path, xyxy in examples:
            if image_path in used_images:
                continue
            used_images.add(image_path)
            model.predict(
                str(image_path),
                refer_image=str(image_path),
                visual_prompts={"bboxes": xyxy[None, :], "cls": np.array([0])},
                predictor=YOLOEVPSegPredictor,
                imgsz=IMAGE_SIZE,
                conf=0.25,
                device=DEVICE,
                verbose=False,
            )
            vectors.append(model.model.pe.detach().cpu()[0, 0])
            if len(vectors) >= REFERENCES_PER_CLASS:
                break
        if len(vectors) < REFERENCES_PER_CLASS:
            raise ValueError(f"Only {len(vectors)} distinct reference images for {name}")
        embeddings.append(normalize(torch.stack(vectors).mean(0), dim=0))
        print(f"{name}: encoded {len(vectors)} Roblox examples", flush=True)

    model.set_classes(names, torch.stack(embeddings)[None, :, :])
    model.save_prompt_embeddings(PROFILE_PATH)
    print(f"Saved {PROFILE_PATH}")


if __name__ == "__main__":
    main()
