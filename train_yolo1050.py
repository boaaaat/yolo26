"""Prepare narrower YOLO26 architectures or explicitly train/distill/QAT them. No evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from multiprocessing import freeze_support
from pathlib import Path

from prepare_yolo1050 import EXPECTED_NAMES, ROOT, dataset_splits, sha256, unwrap_student, write_json


def narrower_yaml(model, ratio: float) -> dict:
    architecture = deepcopy(model.model.yaml)
    architecture["nc"] = len(EXPECTED_NAMES)
    scales = architecture.get("scales")
    if not scales or "n" not in scales:
        raise ValueError("A YOLO26 nano checkpoint with the original n scale is required")
    selected = list(scales["n"])
    selected[1] *= ratio
    # Ultralytics constructs channels through make_divisible(..., 8), including connected residual branches.
    architecture["scales"] = {"n": selected}
    architecture["scale"] = "n"
    return architecture


def transfer_weights(source, destination) -> dict:
    """Initialize a physically narrower architecture; retraining, not a zero-mask model, recovers accuracy."""
    import torch
    source_weights = source.model.state_dict()
    exact = sliced = missing = 0
    with torch.no_grad():
        for name, target in destination.model.state_dict().items():
            original = source_weights.get(name)
            if original is None or target.ndim != original.ndim:
                missing += 1
            elif target.shape == original.shape:
                target.copy_(original); exact += 1
            elif all(new <= old for new, old in zip(target.shape, original.shape)):
                indices = tuple(slice(0, size) for size in target.shape)
                target.copy_(original[indices]); sliced += 1
            else:
                missing += 1
    destination.model.names = dict(source.model.names)
    return {"exact_tensors": exact, "prefix_initialized_tensors": sliced, "new_tensors": missing,
            "note": "Narrow architectures require distillation/fine-tuning; prefix initialization is not accuracy equivalence."}


def no_evaluation_trainer(initial_model):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class NoEvaluationTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # YAML-created YOLO objects otherwise discard the transferred initialization at train setup.
            return super().get_model(cfg=cfg, weights=weights if weights is not None else initial_model,
                                     verbose=verbose)

        def validate(self):
            # Ultralytics can otherwise validate on the final epoch even when val=False.
            return {}, 0.0

        def final_eval(self):
            # No automatic best.pt/last.pt validation at the end of training.
            return None

    return NoEvaluationTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "distill", "qat"), default="prepare")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/yolo26n/weights/best.pt")
    parser.add_argument("--teacher", type=Path, default=ROOT / "runs/yolo26m/weights/best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/rivals/versions/v1/data.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/yolo1050/training")
    parser.add_argument("--width-ratio", type=float, choices=(1.0, .75, .5), default=.75)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--distill-weight", type=float, default=6.0)
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.distill_weight <= 0 or (args.epochs is not None and args.epochs < 1):
        raise ValueError("Invalid training settings")
    if args.device not in ("cpu", "mps") and not args.device.isdigit():
        raise ValueError("Select one training device; the custom no-evaluation trainer does not use distributed workers")
    from ultralytics import YOLO
    import yaml
    source = unwrap_student(YOLO(str(args.checkpoint.resolve())))
    names = [source.names[i] for i in sorted(source.names)]
    if source.task != "detect" or names != EXPECTED_NAMES:
        raise ValueError(f"Expected nano detect checkpoint with {EXPECTED_NAMES}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        for ratio in (.75, .5):
            name = f"yolo26n-width{round(ratio * 100)}"
            path = output / f"{name}.yaml"
            if path.exists():
                raise ValueError(f"Retain existing candidate configuration: {path}")
            path.write_text(yaml.safe_dump(narrower_yaml(source, ratio), sort_keys=False), encoding="utf-8")
            write_json(output / f"{name}.recipe.json", {
                "schema": 1, "source_checkpoint_sha256": sha256(args.checkpoint.resolve()),
                "teacher": str(args.teacher.resolve()), "width_ratio": ratio, "round_channels_to": 8,
                "keep_detection_scales": True, "classes": EXPECTED_NAMES, "image_size": 1024,
                "distill_weight": args.distill_weight, "qualified": False,
                "accuracy_evaluation": "disabled; best.pt is not a quality-selected checkpoint"})
            print(f"Prepared {path}; no training or evaluation started.")
        return
    dataset_splits(args.dataset.resolve())
    training_name = f"distill-width{round(args.width_ratio * 100)}" if args.mode == "distill" else "qat"
    run = output / training_name
    if run.exists():
        raise ValueError(f"Retain existing run: choose a new --output directory instead of {run}")
    initialization = None
    if args.mode == "distill":
        teacher = YOLO(str(args.teacher.resolve()))
        if teacher.task != "detect" or teacher.names != source.names:
            raise ValueError("Teacher must have identical detection classes")
        del teacher
        student = source
        if args.width_ratio != 1:
            yaml_path = output / f"yolo26n-width{round(args.width_ratio * 100)}.yaml"
            architecture = narrower_yaml(source, args.width_ratio)
            if yaml_path.exists():
                if yaml.safe_load(yaml_path.read_text(encoding="utf-8")) != architecture:
                    raise ValueError("Prepared architecture differs from this checkpoint; choose a new output")
            else:
                yaml_path.write_text(yaml.safe_dump(architecture, sort_keys=False), encoding="utf-8")
            student = YOLO(str(yaml_path), task="detect")
            initialization = transfer_weights(source, student)
        extra = {"distill_model": str(args.teacher.resolve()), "dis": args.distill_weight, "quantize": 32}
        epochs, learning_rate = args.epochs or 100, .001
    else:
        from ultralytics.utils import torch_utils
        if not hasattr(torch_utils, "prepare_qat"):
            raise RuntimeError("This Ultralytics installation lacks QAT support; use the preparation requirements")
        # Use --checkpoint to select the trained nano/narrow model. QAT does not change its architecture.
        student = source
        extra = {"quantize": 8, "distill_model": None, "optimizer": "AdamW", "lrf": .1, "warmup_epochs": .5,
                 "cos_lr": True, "mosaic": 0.0}
        epochs, learning_rate = args.epochs or 5, .00001
    student.train(
        trainer=no_evaluation_trainer(student.model), data=str(args.dataset.resolve()), imgsz=1024,
        epochs=epochs, batch=args.batch, device=args.device, workers=args.workers,
        project=str(output), name=training_name, exist_ok=False, lr0=learning_rate,
        amp=False, val=False, plots=False, patience=0, resume=False, compile=False, **extra)
    run = Path(student.trainer.save_dir)
    write_json(run / "deployment-candidate.json", {
        "schema": 1, "source_checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "mode": args.mode, "width_ratio": args.width_ratio if args.mode == "distill" else None,
        "initialization": initialization, "classes": EXPECTED_NAMES, "qualified": False,
        "accuracy_evaluation": "not performed", "deploy_checkpoint": str(run / "weights/last.pt")})
    print(f"Training saved to {run}. Use weights/last.pt for preparation; accuracy qualification is pending.")


if __name__ == "__main__":
    freeze_support()
    main()
