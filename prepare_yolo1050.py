"""Prepare immutable ONNX exports and training-only calibration images. No validation/benchmark."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
PREPROCESSING_ID = "y1050-bilinear11-u8-rgb-nchw-v1"
EXPECTED_NAMES = ["dead", "enemy", "teammate"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def unwrap_student(yolo):
    """Keep the trained student from a distillation checkpoint without fusing training layers."""
    while hasattr(yolo.model, "student_model"):
        wrapper = yolo.model
        wrapper._remove_feature_hooks()
        yolo.model = wrapper.student_model
        # load_checkpoint normalizes the wrapper's args; the serialized student can retain a training namespace.
        yolo.model.args = dict(wrapper.args) if isinstance(wrapper.args, dict) else vars(wrapper.args).copy()
        yolo.model.pt_path = wrapper.pt_path
        yolo.model.task = wrapper.task
    return yolo


def geometry(width: int, height: int) -> dict:
    if width <= 0 or height <= 0:
        raise ValueError("Screen dimensions must be positive")
    scale = 1024 / max(width, height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    tensor_w, tensor_h = math.ceil(resized_w / 32) * 32, math.ceil(resized_h / 32) * 32
    return {"screen_size": [width, height], "input_shape": [1, 3, tensor_h, tensor_w],
            "resized_size": [resized_w, resized_h], "padding": [(tensor_w - resized_w) // 2,
                                                              (tensor_h - resized_h) // 2]}


def axis(source: int, destination: int):
    import numpy as np
    position = (np.arange(destination, dtype=np.float64) + 0.5) * source / destination - 0.5
    first = np.floor(position).astype(np.int32)
    fraction = position - first
    fraction[(first < 0) | (first >= source - 1)] = 0
    first = np.clip(first, 0, source - 1)
    a1 = np.rint(fraction * 2048).astype(np.int32)
    return first, np.minimum(first + 1, source - 1), 2048 - a1, a1


def resize_u8(image, width: int, height: int):
    """The same 11-bit coefficients, border clamping, and rounding as both native capture paths."""
    import numpy as np
    x0, x1, ax0, ax1 = axis(image.shape[1], width)
    y0, y1, ay0, ay1 = axis(image.shape[0], height)
    result = np.empty((height, width, 3), dtype=np.uint8)
    for y in range(height):
        upper = image[y0[y], x0].astype(np.int32) * ax0[:, None] + image[y0[y], x1].astype(np.int32) * ax1[:, None]
        lower = image[y1[y], x0].astype(np.int32) * ax0[:, None] + image[y1[y], x1].astype(np.int32) * ax1[:, None]
        result[y] = ((upper * ay0[y] + lower * ay1[y] + (1 << 21)) >> 22).astype(np.uint8)
    return result


def letterbox(image, target: dict):
    import numpy as np
    height, width = target["input_shape"][2:]
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    rw, rh = round(source_w * scale), round(source_h * scale)
    left, top = (width - rw) // 2, (height - rh) // 2
    output = np.full((height, width, 3), 114, dtype=np.uint8)
    output[top:top + rh, left:left + rw] = resize_u8(image, rw, rh)
    return output


def image_files(value, dataset_root: Path) -> list[Path]:
    if isinstance(value, list):
        paths = [p for item in value for p in image_files(item, dataset_root)]
    else:
        path = Path(value)
        path = (path if path.is_absolute() else dataset_root / path).resolve()
        if path.is_dir():
            paths = [p.resolve() for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]
        elif path.suffix.lower() == ".txt":
            paths = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip())
                paths.append((image if image.is_absolute() else path.parent / image).resolve())
        else:
            raise ValueError(f"Unsupported image split: {path}")
    return sorted(set(paths))


def dataset_splits(path: Path):
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data["names"]
    names = [names[key] for key in sorted(names, key=int)] if isinstance(names, dict) else names
    if names != EXPECTED_NAMES:
        raise ValueError(f"Dataset classes must be {EXPECTED_NAMES}, got {names}")
    root = Path(data.get("path", path.parent))
    root = (root if root.is_absolute() else path.parent / root).resolve()
    splits = {key: image_files(data[key], root) if data.get(key) else [] for key in ("train", "val", "test")}
    if not splits["train"]:
        raise ValueError("Training image split is empty")
    held_out = set(splits["val"]) | set(splits["test"])
    if set(splits["train"]) & held_out:
        raise ValueError("Dataset has training/held-out path overlap")
    return splits


def bucket(path: Path) -> tuple:
    from PIL import Image
    classes, small_enemy = set(), False
    label = path.parent.parent / "labels" / (path.stem + ".txt")
    if label.is_file():
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Invalid detection label: {label}")
            values = [float(v) for v in fields]
            if not all(math.isfinite(v) for v in values) or values[0] not in (0, 1, 2):
                raise ValueError(f"Invalid class/coordinate in {label}")
            cls = int(values[0]); classes.add(cls)
            small_enemy |= cls == 1 and values[3] * values[4] < 0.005
    with Image.open(path) as image:
        tiny = image.convert("L").resize((24, 24))
        brightness = sum(tiny.getdata()) / (24 * 24)
    scene = re.sub(r"[\d_-]+$", "", path.stem).lower()[:24]
    return (tuple(sorted(classes)), small_enemy, min(3, int(brightness / 64)), scene)


def calibration_selection(splits: dict, count: int, seed: int, explicit: Path | None) -> list[Path]:
    train = splits["train"]
    if explicit:
        selected = [(explicit.parent / line.strip()).resolve() for line in explicit.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        if len(selected) != count or len(set(selected)) != count or not set(selected) <= set(train):
            raise ValueError("Calibration list must contain the requested number of distinct training images")
    else:
        if len(train) < count:
            raise ValueError(f"Requested {count} calibration images; training split contains {len(train)}")
        groups = defaultdict(list)
        for image in train:
            groups[bucket(image)].append(image)
        rng = random.Random(seed)
        for group in groups.values():
            rng.shuffle(group)
        keys = list(groups); rng.shuffle(keys); selected = []
        while len(selected) < count:
            for key in keys:
                if groups[key]:
                    selected.append(groups[key].pop())
                    if len(selected) == count:
                        break
    # Reject duplicated held-out content even when it has a different filename.
    held_out_hashes = {sha256(p) for p in splits["val"] + splits["test"]}
    hashes = [sha256(p) for p in selected]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Selected calibration images contain duplicate file content; use a curated list")
    if held_out_hashes.intersection(hashes):
        raise ValueError("Calibration selection duplicates held-out image content")
    return selected


def prepare_calibration(destination: Path, target: dict, dataset: Path, count: int, seed: int, explicit: Path | None):
    import cv2
    import numpy as np
    splits = dataset_splits(dataset)
    selected = calibration_selection(splits, count, seed, explicit)
    directory = destination / "calibration-images"; directory.mkdir()
    images = []
    for index, source in enumerate(selected):
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {source}")
        output = directory / f"{index:04d}.png"
        ok, png = cv2.imencode(".png", letterbox(image, target), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError(f"Cannot encode calibration image {source}")
        output.write_bytes(png.tobytes())
        images.append({"path": output.relative_to(destination).as_posix(), "sha256": sha256(output),
                       "source_sha256": sha256(source), "source": str(source), "stratum": list(bucket(source))})
    write_json(destination / "calibration.json", {
        "schema": 1, "split": "train", "dataset_sha256": sha256(dataset), "preprocessing_id": PREPROCESSING_ID,
        "input_shape": target["input_shape"], "seed": seed, "count": len(images),
        "selection": "class presence, small enemies, brightness quartiles, filename scene prefix; round-robin",
        "images": images})


def inspect_export(path: Path, shape: list[int], head_index: int, qat: bool):
    import numpy as np
    import onnx
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one image input and one NMS-free output")
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    if input_shape != shape or output_shape != [1, 300, 6]:
        raise ValueError(f"Expected static {shape} -> [1,300,6], got {input_shape} -> {output_shape}")
    if any(t.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for t in (*model.graph.input, *model.graph.output)):
        raise ValueError("Native runtime requires FP32 image and detection I/O")
    constants = {v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = onnx.numpy_helper.to_array(attribute.t)
    sensitive, quantizers = [], 0
    for node in model.graph.node:
        if node.op_type == "Conv" and f"/model.{head_index}/" in node.name and len(node.input) >= 2:
            weight = constants.get(node.input[1])
            if weight is not None and weight.shape[0] in (3, 4):
                sensitive.append(node.name)
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        quantizers += node.op_type == "QuantizeLinear"
        scale = constants.get(node.input[1])
        zero = constants.get(node.input[2]) if len(node.input) > 2 else None
        if scale is None or scale.dtype != np.float32 or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("TensorRT 8.6 requires constant positive FP32 quantization scales")
        if zero is None or zero.dtype != np.int8 or np.any(zero != 0):
            raise ValueError("TensorRT 8.6 requires explicit symmetric signed INT8 zero points")
        if node.op_type == "QuantizeLinear" and scale.size > 1:
            weight = constants.get(node.input[0])
            axis_value = next((a.i for a in node.attribute if a.name == "axis"), 1)
            if weight is None or axis_value != 0 or scale.ndim != 1 or scale.size != weight.shape[0]:
                raise ValueError("Use per-tensor activations and output-channel-axis per-channel weights")
    if qat != bool(quantizers):
        raise ValueError("Checkpoint quantization and ONNX Q/DQ nodes disagree")
    return sensitive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/yolo26n/weights/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/yolo1050/nano-fp32")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/rivals/versions/v1/data.yaml")
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--calibration-list", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    checkpoint, destination = args.checkpoint.resolve(), args.output.resolve()
    if destination.exists():
        raise ValueError(f"Retain the existing candidate: choose a new output directory instead of {destination}")
    if not checkpoint.is_file() or args.calibration_count < 1:
        raise ValueError("A checkpoint and positive calibration count are required")
    target = geometry(args.screen_width, args.screen_height)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(checkpoint)
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import is_qat
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=destination.parent) as temporary:
        scratch = Path(temporary); copied = scratch / "source.pt"
        shutil.copy2(checkpoint, copied)
        if sha256(copied) != source_hash or sha256(checkpoint) != source_hash:
            raise RuntimeError("Checkpoint changed during preparation; retry after the save finishes")
        model = unwrap_student(YOLO(str(copied)))
        names = [model.names[i] for i in sorted(model.names)]
        if model.task != "detect" or names != EXPECTED_NAMES:
            raise ValueError(f"Expected detect checkpoint with {EXPECTED_NAMES}, got {model.task}: {names}")
        head = model.model.model[-1]
        if any(getattr(head, key, None) is None for key in ("one2one_cv2", "one2one_cv3")):
            raise ValueError("Checkpoint needs trained one-to-one detection heads")
        qat = is_qat(model.model)
        model.model.end2end = True
        model.model.float()
        exported = Path(model.export(format="onnx", imgsz=tuple(target["input_shape"][2:]), device="cpu",
                                     batch=1, dynamic=False, nms=False, max_det=300, opset=17,
                                     quantize=8 if qat else 32, simplify=not qat, verbose=False))
        sensitive = inspect_export(exported, target["input_shape"], len(model.model.model) - 1, qat)
        final_onnx = scratch / "model.onnx"
        if exported != final_onnx:
            exported.replace(final_onnx)
        write_json(scratch / "export.json", {
            "schema": 1, "checkpoint_sha256": source_hash, "onnx": "model.onnx", "onnx_sha256": sha256(final_onnx),
            "names": names, "nms_free": True, "output_shape": [1, 300, 6], "opset": 17,
            "quantization": "qat-int8" if qat else "fp32", "sensitive_fp32_layer_patterns": sensitive,
            "preprocessing": {"id": PREPROCESSING_ID, "padding_value": 114, "channel_order": "RGB",
                              "layout": "NCHW", "normalization": "uint8 / 255", "coefficient_bits": 11},
            "software": {package: importlib.metadata.version(package) for package in ("ultralytics", "torch", "onnx")},
            "qualified": False, **target})
        if not args.skip_calibration and not qat:
            prepare_calibration(scratch, target, args.dataset.resolve(), args.calibration_count, args.seed,
                                args.calibration_list.resolve() if args.calibration_list else None)
        # Keep the immutable source copy with the candidate; no changes to training checkpoints.
        shutil.copytree(scratch, destination)
    print(f"Prepared {destination}: {target['input_shape']}, {'QAT INT8' if qat else 'FP32 + optional PTQ calibration'}")
    print("No accuracy evaluation, FPS benchmark, TensorRT engine build, or live controls were run.")


if __name__ == "__main__":
    main()
