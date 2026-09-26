"""Evaluate NVIDIA LocateAnything on Roblox validation images (run inside WSL)."""

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from dataset_project import latest_generated_version


# Edit these values instead of passing command line arguments.
ROOT = Path(__file__).resolve().parent
WSL_MODEL_PATH = Path.home() / "models" / "LocateAnything-3B"
MODEL_PATH = WSL_MODEL_PATH if WSL_MODEL_PATH.is_dir() else ROOT / "models" / "LocateAnything-3B"
DATASET_ROOT = ROOT / "datasets" / "rivals"
OUTPUT_PATH = ROOT / "models" / "locateanything-roblox-validation.json"
IMAGE_NAMES = []  # Empty means all validation images; otherwise list image filenames.
DESCRIPTIONS = ["person"]
PRINT_EACH_RESULT = False
GENERATION_MODE = "hybrid"
MAX_NEW_TOKENS = 512


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def main() -> None:
    version = latest_generated_version(DATASET_ROOT)
    image_dir = version / "valid" / "images"
    label_dir = version / "valid" / "labels"
    images = [image_dir / name for name in IMAGE_NAMES] if IMAGE_NAMES else sorted(image_dir.glob("*.jpg"))
    if any(not path.is_file() for path in images):
        raise FileNotFoundError("One or more IMAGE_NAMES are missing from the validation folder")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to("cuda").eval()
    torch.manual_seed(0)

    records = []
    for image_path in images:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        labels = []
        for line in (label_dir / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
            class_id, cx, cy, box_width, box_height = map(float, line.split())
            labels.append({"class_id": int(class_id), "xyxy": [
                (cx - box_width / 2) * width, (cy - box_height / 2) * height,
                (cx + box_width / 2) * width, (cy + box_height / 2) * height,
            ]})
        for description in DESCRIPTIONS:
            query = f"Locate all the instances that match the following description: {description}."
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ]}]
            prompt = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pixels, videos = processor.process_vision_info(messages)
            inputs = processor(text=[prompt], images=pixels, videos=videos, return_tensors="pt").to("cuda")
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad():
                response = model.generate(
                    pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=tokenizer,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    generation_mode=GENERATION_MODE,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    verbose=False,
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            answer = response[0] if isinstance(response, tuple) else response
            predictions = [
                [int(value) / 1000 * scale for value, scale in zip(match, (width, height, width, height))]
                for match in re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)
            ]
            best_ious = [max((iou(label["xyxy"], box) for box in predictions), default=0) for label in labels]
            record = {
                "image": image_path.name, "query": query, "answer": answer,
                "predicted_boxes": predictions, "labels": labels, "best_iou_per_label": best_ious,
                "seconds": seconds, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            records.append(record)
            if PRINT_EACH_RESULT:
                print(json.dumps({"image": image_path.name, "description": description,
                                  "answer_preview": answer[:160], "box_count": len(predictions), "best_ious": best_ious,
                                  "seconds": round(seconds, 2),
                                  "peak_allocated_gb": round(record["peak_allocated_gb"], 2)}), flush=True)
        if len(records) % 25 == 0:
            print(f"Processed {len(records)}/{len(images) * len(DESCRIPTIONS)} queries", flush=True)
    OUTPUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
