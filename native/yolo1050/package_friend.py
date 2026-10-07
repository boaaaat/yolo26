"""Create a self-contained 1080p GTX 1050 handoff package. No runtime, test, or benchmark execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(__file__).resolve().parent
VIEW_LAUNCHERS = ("4-view-detections-fp32.cmd", "4-view-detections-int8.cmd")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy(source: Path, target: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_archive(output: Path, archive_path: Path, include_folder: bool = True):
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                relative = path.relative_to(output)
                archive.write(path, Path(output.name) / relative if include_folder else relative)
    print(f"Ready: {archive_path} ({archive_path.stat().st_size / 1048576:.1f} MiB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/yolo1050/nano-fp32")
    parser.add_argument("--toolchain", type=Path, default=ROOT / "artifacts/yolo1050/toolchain")
    parser.add_argument("--binary", type=Path, default=NATIVE / "build/release-11.8/yolo1050.exe")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/yolo1050/friend-1080p-ready")
    parser.add_argument("--runtime-update-only", action="store_true",
                        help="Package only the rebuilt executable and capture instructions; use a new --output")
    args = parser.parse_args()
    model, sdk, output = args.model.resolve(), args.toolchain.resolve(), args.output.resolve()
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists():
        raise ValueError("Preserve an existing package; choose a new --output")
    if args.runtime_update_only:
        copy(args.binary.resolve(), output / "runtime/yolo1050.exe")
        copy(NATIVE / "CAPTURE-FIX.txt", output / "CAPTURE-FIX.txt")
        copy(NATIVE / "apply_runtime_update.ps1", output / "apply_runtime_update.ps1")
        copy(NATIVE / "APPLY-CAPTURE-UPDATE.cmd", output / "APPLY-CAPTURE-UPDATE.cmd")
        for name in VIEW_LAUNCHERS:
            copy(NATIVE / name, output / name)
        base_package = json.loads((ROOT / "artifacts/yolo1050/friend-1080p-ready/package-manifest.json").read_text(encoding="utf-8"))
        runtime_files = {Path(name).name: info for name, info in base_package["files"].items()
                         if name.startswith("runtime/") and name.endswith(".dll")}
        dependencies = sorted(runtime_files)
        record = {"schema": 2, "kind": "runtime-update", "qualified": False,
                  "change": "Detection counts/confidence, separate capture-miss counters, SSE2 CPU resize, optional detection views",
                  "engine_cache": "Reuse existing engines; engine identity, model, and preprocessing unchanged",
                  "required_runtime_dlls": dependencies,
                  "required_runtime_files": runtime_files,
                  "install_root_files": list(VIEW_LAUNCHERS),
                  "files": {p.relative_to(output).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                            for p in sorted(output.rglob("*")) if p.is_file()}}
        (output / "runtime-update.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Packaging runtime update only; no executable launched.", flush=True)
        # Updates merge directly into an existing installation's runtime folder.
        write_archive(output, archive_path, include_folder=False)
        return
    manifest = json.loads((model / "export.json").read_text(encoding="utf-8"))
    calibration = json.loads((model / "calibration.json").read_text(encoding="utf-8"))
    if manifest["input_shape"] != [1, 3, 576, 1024] or calibration["count"] != 512 or calibration["split"] != "train":
        raise ValueError("Expected the prepared 1080p model with 512 training-only calibration images")
    if digest(model / manifest["onnx"]) != manifest["onnx_sha256"]:
        raise ValueError("Prepared model hash mismatch")
    runtime = output / "runtime"
    runtime.mkdir(parents=True)
    copy(args.binary.resolve(), runtime / "yolo1050.exe")
    trt = sdk / "sdk/TensorRT-8.6.1.6"
    cudnn = sdk / "sdk/cudnn-windows-x86_64-8.9.0.131_cuda11-archive"
    cuda = sdk / "cuda-11.8"
    opencv = sdk / "opencv-sdk/opencv"
    for name in ("nvinfer.dll", "nvinfer_plugin.dll", "nvonnxparser.dll", "nvinfer_builder_resource.dll"):
        copy(trt / "lib" / name, runtime / name)
    for path in (cudnn / "bin").glob("*.dll"):
        copy(path, runtime / path.name)
    for name in ("cudart64_110.dll", "cublas64_11.dll", "cublasLt64_11.dll", "nvrtc64_112_0.dll", "nvrtc-builtins64_118.dll"):
        copy(cuda / "bin" / name, runtime / name)
    copy(opencv / "build/x64/vc16/bin/opencv_world4100.dll", runtime / "opencv_world4100.dll")
    copy(sdk / "zlib-runtime/zlibwapi.dll", runtime / "zlibwapi.dll")
    redist_root = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC")
    versions = sorted((p for p in redist_root.iterdir() if p.is_dir() and p.name[0].isdigit()),
                      key=lambda p: tuple(map(int, p.name.split("."))))
    redist = versions[-1] / "x64/Microsoft.VC143.CRT"
    for path in redist.glob("*.dll"):
        copy(path, runtime / path.name)
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        if not (runtime / name).is_file():
            raise FileNotFoundError(f"Microsoft runtime dependency: {name}")
    target_model = output / "model"
    copy(model / "model.onnx", target_model / "model.onnx")
    copy(model / "export.json", target_model / "export.json")
    copy(model / "calibration.json", target_model / "calibration.json")
    shutil.copytree(model / "calibration-images", target_model / "calibration-images")
    for image in calibration["images"]:
        if digest(target_model / image["path"]) != image["sha256"]:
            raise ValueError("Calibration file hash mismatch")
    config = json.loads((NATIVE / "settings.example.json").read_text(encoding="utf-8"))
    config.update(export_manifest="model/export.json", calibration_manifest="model/calibration.json",
                  cache_dir="engine-cache", mouse_calibration="mouse_calibration.json", engine=None)
    (output / "settings.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    copy(NATIVE / "calibrate_display.ps1", output / "calibrate_display.ps1")
    copy(NATIVE / "CAPTURE-FIX.txt", output / "CAPTURE-FIX.txt")
    for name in VIEW_LAUNCHERS:
        copy(NATIVE / name, output / name)
    copy(ROOT / "calibrate.py", output / "calibrate.py")
    copy(ROOT / "dataset_utils.py", output / "dataset_utils.py")
    prefix = '@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\nset "PATH=%~dp0runtime;%PATH%"\r\n'
    build = prefix + r'''powershell.exe -NoProfile -Command "if (Get-Process -Name RobloxPlayerBeta,yolo1050 -ErrorAction SilentlyContinue) { Write-Host 'Close Roblox and the detector before building engines.'; exit 1 }; exit 0"
if errorlevel 1 goto fail
"runtime\yolo1050.exe" --config "settings.json" --precision fp32 --build-only
if errorlevel 1 goto fail
"runtime\yolo1050.exe" --config "settings.json" --precision int8 --build-only
if errorlevel 1 goto fail
echo Both engines are ready. Next run 2-calibrate-display.cmd.
pause
exit /b 0
:fail
echo Engine build stopped. Keep this window open to read the error.
pause
exit /b 1
'''
    calibrate = prefix + r'''powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0calibrate_display.ps1" -Output "%~dp0mouse_calibration.json"
if errorlevel 1 goto fail
echo Calibration ready. Next run 3-start-int8.cmd.
pause
exit /b 0
:fail
echo Calibration was not completed. Keep this window open to read the error.
pause
exit /b 1
'''
    start = prefix + '"runtime\\yolo1050.exe" --config "settings.json"\r\n' + 'pause\r\n'
    start_fp32 = prefix + '"runtime\\yolo1050.exe" --config "settings.json" --precision fp32\r\n' + 'pause\r\n'
    for name, body in {"1-build-engines.cmd": build, "2-calibrate-display.cmd": calibrate,
                       "3-start-int8.cmd": start, "start-fp32.cmd": start_fp32}.items():
        (output / name).write_bytes(body.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))
    (output / "START-HERE.txt").write_text(
        "GTX 1050 laptop / primary 1920x1080 60 Hz display\n\n"
        "Extract the entire ZIP to a local folder before running these launchers.\n"
        "1. With Roblox closed, double-click 1-build-engines.cmd. Keep the window open until both builds finish.\n"
        "   This builds FP32 and calibrated INT8 engines on this GTX 1050; it is not an FPS benchmark.\n"
        "2. Open Roblox/Rivals, enter practice, and lock the cursor. Double-click 2-calibrate-display.cmd.\n"
        "   Keep the cursor still, then press =. This saves your own display calibration.\n"
        "3. Double-click 3-start-int8.cmd. Press = to arm, - to pause, Ctrl+C to exit.\n\n"
        "No Python, C++ compiler, CUDA toolkit installation, or model training is needed on this laptop.\n"
        "A working NVIDIA driver compatible with the GTX 1050 and CUDA 11.8 is required.\n"
        "The runtime detects whether Intel or NVIDIA drives the screen and logs the selected capture path.\n"
        "Overlay and collection are off. The model sees the full screen at 1024x576, with no crop.\n"
        "Do not reuse another GPU's engine cache or another display's mouse calibration.\n"
        "Re-run calibration after changing the primary display resolution.\n\n"
        "If step 3 reports Desktop Duplication / 0x887A0004, read CAPTURE-FIX.txt.\n\n"
        "The model was exported and the Release executable was compiled. No tests, accuracy evaluations,\n"
        "FPS benchmarks, Roblox-performance measurements, target-machine engine builds, or live inference\n"
        "were run during package preparation. 60 FPS, accuracy, and Roblox impact remain unverified.\n",
        encoding="utf-8")
    license_paths = []
    for component, base in (("tensorrt", trt), ("cudnn", cudnn), ("cuda", cuda), ("opencv", opencv),
                            ("zlib", sdk / "zlib-runtime")):
        for path in base.rglob("*"):
            if path.is_file() and (path.name.lower().startswith(("license", "eula", "copyright", "notice"))):
                relative = Path(component) / path.relative_to(base)
                copy(path, output / "licenses" / relative)
                license_paths.append(relative.as_posix())
    copy(NATIVE / "zlibwapi.def", output / "licenses/zlib/zlibwapi.def")
    toolchain_record = json.loads((sdk / "provenance.json").read_text(encoding="utf-8"))
    (output / "dependency-provenance.json").write_text(json.dumps(toolchain_record, indent=2), encoding="utf-8")
    record = {"schema": 1, "display": [1920, 1080], "input_shape": manifest["input_shape"],
              "classes": manifest["names"], "checkpoint_sha256": manifest["checkpoint_sha256"],
              "onnx_sha256": manifest["onnx_sha256"], "calibration_count": 512,
              "precision_default": "int8", "inference_target_hz": 60, "qualified": False,
              "validation": "not performed; source export and native Release compilation only",
              "files": {p.relative_to(output).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                        for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "package-manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Packaging", len(record["files"]), "files; no executable or calibration UI launched.", flush=True)
    write_archive(output, archive_path)


if __name__ == "__main__":
    main()
