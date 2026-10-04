"""Select an engine from user-supplied qualification measurements. Does not measure or run models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from prepare_yolo1050 import sha256, write_json


def number(record: dict, key: str, fraction: bool = False) -> float:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite measured number")
    if value < 0 or (fraction and value > 1):
        raise ValueError(f"{key} is outside its range; accuracy metrics use fractions, not percentages")
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    measurements = args.measurements.resolve()
    document = json.loads(measurements.read_text(encoding="utf-8"))
    if document["schema"] != 1:
        raise ValueError("Unsupported qualification measurement schema")
    baseline = document["baseline"]
    baseline_map = number(baseline, "map50_95", True)
    baseline_recall = number(baseline, "enemy_recall", True)
    baseline_distant = number(baseline, "distant_enemy_recall", True)
    baseline_fps = number(baseline, "roblox_fps")
    baseline_p95 = number(baseline, "roblox_p95_frame_ms")
    if baseline_fps <= 0 or baseline_p95 <= 0:
        raise ValueError("Positive Roblox baseline FPS and frame times are required")
    qualified, observed = [], []
    for candidate in document["candidates"]:
        label = candidate["name"]
        if candidate.get("same_gameplay_conditions") is not True or candidate.get("thermal_stable") is not True:
            print(f"{label}: rejected; matched gameplay and thermal stabilization are required")
            continue
        fps = number(candidate, "fresh_predictions_per_second")
        p95 = number(candidate, "capture_to_publication_p95_ms")
        observed.append((fps, label))
        print(f"{label}: {fps:.2f} fresh predictions/s, capture-to-publication p95 {p95:.2f} ms")
        failures = []
        for bad, reason in (
            (fps < 60, "fresh detector rate below 60 FPS"),
            (p95 > 1000 / 60, "capture-to-publication p95 above 16.67 ms"),
            (number(candidate, "map50_95", True) < baseline_map - .01, "mAP loss above 1 percentage point"),
            (number(candidate, "enemy_recall", True) < baseline_recall - .02, "enemy recall loss above 2 points"),
            (number(candidate, "distant_enemy_recall", True) < baseline_distant - .02, "distant enemy recall loss above 2 points"),
            (number(candidate, "roblox_fps") < baseline_fps * .95, "Roblox FPS loss above 5%"),
            (number(candidate, "roblox_p95_frame_ms") > baseline_p95 * 1.05, "Roblox p95 frame-time regression above 5%"),
            (number(candidate, "runtime_memory_increment_mib") > 512, "runtime VRAM increment above 512 MiB"),
        ):
            if bad:
                failures.append(reason)
        engine = (measurements.parent / candidate["engine"]).resolve()
        metadata = json.loads(Path(str(engine) + ".json").read_text(encoding="utf-8"))
        if sha256(engine) != candidate["engine_sha256"] or metadata["engine_sha256"] != candidate["engine_sha256"]:
            raise ValueError(f"{label}: engine does not match the measured artifact")
        if failures:
            print(f"{label}: rejected; {'; '.join(failures)}")
        else:
            qualified.append((fps, -p95, candidate, engine, metadata))
    if not qualified:
        if observed:
            rate, label = max(observed)
            print(f"Highest supplied measured rate: {rate:.2f} fresh predictions/s ({label}); no fully qualified candidate.")
        raise RuntimeError("No candidate meets all selected constraints. No configuration changed; detector shortfalls remain visible.")
    _, _, candidate, engine, metadata = max(qualified, key=lambda item: item[:2])
    source_config = (measurements.parent / candidate["config"]).resolve()
    config = json.loads(source_config.read_text(encoding="utf-8"))
    # Freeze the selected engine; resolve paths before moving the configuration to another directory.
    defaults = {"calibration_manifest": "calibration.json", "cache_dir": "cache",
                "mouse_calibration": "../../mouse_calibration.json"}
    for key in ("export_manifest", *defaults):
        config[key] = str((source_config.parent / config.get(key, defaults.get(key))).resolve())
    export = json.loads(Path(config["export_manifest"]).read_text(encoding="utf-8"))
    if export != metadata["export"]:
        raise ValueError("Measured configuration and selected engine refer to different model exports")
    config["engine"] = str(engine)
    identity = metadata["identity"]
    config["precision"] = identity["precision"]
    config["workspace_mib"] = identity["workspace_mib"]
    config["fp32_layer_patterns"] = identity["fp32_layers"]
    config["inference_hz"] = 60
    config["qualification"] = {"measurements_sha256": sha256(measurements), "candidate": candidate["name"],
                               "engine_sha256": candidate["engine_sha256"]}
    output = args.output_config.resolve()
    if output.exists():
        raise ValueError("Preserve the existing configuration: choose a new --output-config")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, config)
    print(f"Selected {candidate['name']}: {output}. Measurements were supplied by the user; no evaluation or benchmark was run.")


if __name__ == "__main__":
    main()
