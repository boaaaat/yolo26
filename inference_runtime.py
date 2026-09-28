"""Shared fixed-shape PyTorch YOLO inference, precision, warmup, and compiler cache."""

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from dataset_utils import atomic_write, class_name_map


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: Path
    gpu_index: int = 0
    image_size: int = 1024
    precision: str = "bf16"
    compile_mode: str | bool = "reduce-overhead"
    warmup_passes: int = 3
    cache_dir: Path = Path(__file__).resolve().parent / ".inference_compile_cache"
    nms: bool | None = False


def native_bf16_supported() -> bool:
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8


def compiled_cache_path(checkpoint: Path, config: InferenceConfig) -> Path:
    with checkpoint.open("rb") as source:
        checkpoint_hash = hashlib.file_digest(source, "sha256").hexdigest()
    try:
        triton_version = importlib.metadata.version("triton-windows")
    except importlib.metadata.PackageNotFoundError:
        triton_version = importlib.metadata.version("triton")
    identity = {
        "checkpoint_sha256": checkpoint_hash,
        "image_size": config.image_size,
        "compile_mode": config.compile_mode,
        "dtype": config.precision,
        "nms": config.nms,
        "torch": torch.__version__,
        "triton": triton_version,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(config.gpu_index),
        "gpu_capability": torch.cuda.get_device_capability(config.gpu_index),
        "python": sys.version_info[:3],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return Path(config.cache_dir) / f"model-{key}.ptcache"


def load_compiled_cache(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if torch.compiler.load_cache_artifacts(path.read_bytes()) is None:
            raise ValueError("cache contains no compiler artifacts")
    except Exception as exc:
        print(f"Could not load compiled cache ({exc}); rebuilding it.")
        return False
    print(f"Loaded compiled artifacts from {path}")
    return True


def save_compiled_cache(path: Path) -> None:
    try:
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            print("PyTorch did not return compiler artifacts to save.")
            return
        atomic_write(path, artifacts[0])
        print(f"Saved compiled artifacts to {path}")
    except Exception as exc:
        print(f"Could not save compiled cache ({exc}); inference can still run.")


class InferenceRuntime:
    def __init__(self, config: InferenceConfig):
        if config.precision not in {"bf16", "fp32"} or config.image_size <= 0 or config.warmup_passes <= 0:
            raise ValueError("Inference precision, image size, or warmup settings are invalid")
        self.config = config
        self.checkpoint = Path(config.checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-enabled PyTorch installation and NVIDIA GPU are required")
        torch.cuda.set_device(config.gpu_index)
        capability = torch.cuda.get_device_capability(config.gpu_index)
        supported_arches = torch.cuda.get_arch_list()
        if capability[0] == 6 and supported_arches and not any(
            arch.startswith("sm_") and arch[3:].isdigit()
            and int(arch[3:]) // 10 == capability[0]
            and int(arch[3:]) % 10 <= capability[1]
            for arch in supported_arches
        ):
            raise RuntimeError(
                f"This PyTorch CUDA build does not include Pascal support (GPU sm_{capability[0]}{capability[1]}). "
                "Install a PyTorch CUDA 12.6 build in this environment; CUDA 13 builds omit Pascal."
            )
        if config.precision == "bf16" and not native_bf16_supported():
            raise RuntimeError("This GPU or PyTorch build does not support native CUDA BF16")
        if config.compile_mode and importlib.util.find_spec("triton") is None:
            raise RuntimeError(
                'Native Windows torch.compile needs Triton. In the yolo environment, run '
                'python -m pip install "triton-windows>=3.8,<3.9" for PyTorch 2.14.'
            )
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high" if config.precision == "bf16" else "highest")
        self.model = YOLO(str(self.checkpoint))
        if self.model.task != "detect":
            raise ValueError(f"Expected a detection checkpoint, got {self.model.task!r}")
        self.names = class_name_map(self.model.names)
        self.cache_path = compiled_cache_path(self.checkpoint, config) if config.compile_mode else None
        self.cache_loaded = load_compiled_cache(self.cache_path) if self.cache_path is not None else False

    def predict(self, frame: np.ndarray, *, confidence: float, classes=None, max_detections: int = 100):
        config = self.config
        precision_context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                             if config.precision == "bf16" else nullcontext())
        with torch.inference_mode(), precision_context:
            return self.model.predict(
                source=frame, device=config.gpu_index, imgsz=config.image_size,
                rect=False, conf=confidence, classes=classes, max_det=max_detections,
                nms=config.nms, compile=config.compile_mode, quantize=32, verbose=False,
            )[0]

    def warmup(self, *, frame_shape=None, confidence: float = 0.5, classes=None,
               max_detections: int = 100) -> None:
        config = self.config
        height, width = frame_shape or (config.image_size, config.image_size)
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        action = ("Using cached compiler artifacts and warming" if self.cache_loaded else
                  "Compiling and warming" if config.compile_mode else "Warming")
        print(f"{action} {self.checkpoint.name} on {torch.cuda.get_device_name(config.gpu_index)}...")
        for _ in range(config.warmup_passes):
            self.predict(black_frame, confidence=confidence, classes=classes, max_detections=max_detections)
        torch.cuda.synchronize(config.gpu_index)
        if config.compile_mode and getattr(self.model.predictor.model, "_orig_mod", None) is None:
            raise RuntimeError("PyTorch compilation was unavailable; Ultralytics fell back to eager inference")
        if self.cache_path is not None and not self.cache_loaded:
            save_compiled_cache(self.cache_path)
        description = "compiled" if config.compile_mode else "eager"
        print(f"Ready: {config.precision.upper()} + {description} model, {config.image_size}px input.")
