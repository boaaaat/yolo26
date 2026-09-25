"""Build the cached text prompts used by YOLOE in this Roblox dataset."""

import os
from pathlib import Path

import yaml
from ultralytics import YOLOE


# Edit these prompts if your game's visual cues change, then rerun this script.
ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "datasets" / "data.yaml"
MODEL_PATH = ROOT / "models" / "yoloe-26s-seg.pt"
PROMPT_PROFILE_PATH = ROOT / "models" / "roblox-yoloe-26s.npz"
PROMPTS_BY_CLASS = {
    "dead": "dead Roblox player avatar",
    "enemy": "enemy Roblox player avatar",
    "teammate": "friendly Roblox teammate avatar",
}


def main() -> None:
    dataset = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    names = dataset["names"]
    if isinstance(names, dict):
        names = [value for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    if set(names) != set(PROMPTS_BY_CLASS):
        raise ValueError(f"Set one prompt for each dataset class: {names}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    # Ultralytics finds the downloaded MobileCLIP text encoder in the current directory.
    os.chdir(MODEL_PATH.parent)
    model = YOLOE(str(MODEL_PATH))
    prompts = [PROMPTS_BY_CLASS[name] for name in names]
    model.set_classes(prompts)
    model.save_prompt_embeddings(PROMPT_PROFILE_PATH)
    print(f"Saved {len(prompts)} prompts to {PROMPT_PROFILE_PATH}")


if __name__ == "__main__":
    main()
