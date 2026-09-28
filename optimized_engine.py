"""Build and cache fixed-shape YOLO26 TensorRT engines in a separate process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
from filelock import FileLock


CACHE_DIR = Path(__file__).resolve().parent / ".optimized_engine_cache"
ENGINE_SCHEMA = 1
MAX_CANDIDATES = 300  # Match the original predictor's head before class filtering.


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def engine_spec(checkpoint: Path, height: int, width: int, gpu: int, workspace: float):
    import tensorrt as trt

    props = torch.cuda.get_device_properties(gpu)
    identity = {
        "schema": ENGINE_SCHEMA,
        "checkpoint_sha256": sha256(checkpoint),
        "input_shape": [1, 3, height, width],
        "precision": "fp16", "nms": False, "max_det": MAX_CANDIDATES,
        "workspace_gib": workspace,
        "gpu": props.name, "capability": [props.major, props.minor],
        "gpu_memory": props.total_memory,
        "tensorrt": trt.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "onnx": importlib.metadata.version("onnx"),
        "onnxslim": importlib.metadata.version("onnxslim"),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return identity, CACHE_DIR / f"yolo26-{key}.engine"


def read_engine(path: Path):
    """Our exporter writes a length-prefixed JSON header before the TRT plan."""
    with path.open("rb") as source:
        length = int.from_bytes(source.read(4), "little", signed=True)
        if not 0 < length <= 1_000_000:
            raise ValueError(f"Invalid Ultralytics engine header: {path}")
        metadata = json.loads(source.read(length).decode("utf-8"))
        payload = source.read()
    if not payload or metadata.get("task") != "detect" or metadata.get("end2end") is not True:
        raise ValueError("Expected a YOLO26 detect engine with its NMS-free head enabled")
    return metadata, payload


def cached_engine_matches(path: Path, identity: dict) -> bool:
    try:
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return manifest["identity"] == identity and manifest["engine_sha256"] == sha256(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def build_engine(checkpoint: Path, height: int, width: int, gpu: int, workspace: float) -> Path:
    """Export/build only. No capture, controls, validation, or inference benchmark."""
    from ultralytics import YOLO

    torch.cuda.set_device(gpu)
    identity, destination = engine_spec(checkpoint, height, width, gpu, workspace)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if cached_engine_matches(destination, identity):
            return destination
        # Export beside a temporary copy, never into the training run directory.
        with tempfile.TemporaryDirectory(prefix="build-", dir=CACHE_DIR) as scratch:
            copied_checkpoint = Path(scratch) / "model.pt"
            shutil.copy2(checkpoint, copied_checkpoint)
            if sha256(copied_checkpoint) != identity["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint changed during preparation; retry after training saves it")
            model = YOLO(str(copied_checkpoint))
            if model.task != "detect":
                raise ValueError(f"This runtime requires a detection checkpoint, got task {model.task!r}")
            head = model.model.model[-1]
            if any(getattr(head, name, None) is None for name in ("one2one_cv2", "one2one_cv3")):
                raise ValueError("This runtime requires a checkpoint with trained one-to-one detection heads")
            # Saved checkpoints can select one-to-many inference even when both
            # trained heads are present. Select the NMS-free branch before export
            # and fusion; nms=False below preserves that selection in the exporter.
            model.model.end2end = True
            if not model.model.end2end:
                raise ValueError("Could not enable the checkpoint's NMS-free detection head")
            print(f"Building FP16 TensorRT engine: {width}x{height}, batch 1. This can take several minutes.", flush=True)
            exported = Path(model.export(
                format="engine", device=gpu, imgsz=(height, width), batch=1,
                dynamic=False, quantize=16, nms=False, max_det=MAX_CANDIDATES,
                simplify=True, workspace=workspace, verbose=False,
            ))
            metadata, _ = read_engine(exported)
            if list(metadata["imgsz"]) != [height, width]:
                raise RuntimeError("Exporter changed the requested fixed input dimensions")
            manifest = {"identity": identity, "engine_sha256": sha256(exported)}
            manifest_tmp = Path(scratch) / "manifest.json"
            manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(exported, destination)
            os.replace(manifest_tmp, destination.with_suffix(".json"))
    return destination


def ensure_engine(checkpoint: Path, height: int, width: int, gpu: int, workspace: float) -> Path:
    identity, path = engine_spec(checkpoint, height, width, gpu, workspace)
    if not cached_engine_matches(path, identity):
        # Releasing this process also releases exporter weights and builder workspace.
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--checkpoint", str(checkpoint), "--height", str(height), "--width", str(width),
            "--gpu", str(gpu), "--workspace", str(workspace),
        ], check=True, env=environment)
        if not cached_engine_matches(path, identity):
            raise RuntimeError("Engine cache was not created for the requested configuration")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workspace", type=float, default=2.0)
    args = parser.parse_args()
    if min(args.height, args.width) < 32 or args.height % 32 or args.width % 32 or args.workspace <= 0:
        parser.error("Dimensions must be positive multiples of 32; workspace must be positive")
    print(build_engine(args.checkpoint.resolve(), args.height, args.width, args.gpu, args.workspace))
