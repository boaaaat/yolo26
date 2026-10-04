# Native GTX 1050 inference

This is an additive C++17 deployment for a Pascal SM 6.1 GPU. Python is used only for preparing ONNX/calibration artifacts and optional training. Live operation uses TensorRT 8.6.1, CUDA 11.8, D3D11, Win32 controls, and OpenCV's image decoder for engine calibration.

The default target is **60 fresh desktop-capture predictions/s**, full-screen coverage, and a **1024-pixel long edge**. A 16:9 display uses a 1024×576 input. Neither 60 FPS alongside Roblox, the accuracy limits, nor the 512 MiB additional VRAM target has been measured. Roblox's 5% performance allowance is a qualification criterion; this program cannot reserve a fixed GPU percentage or guarantee game performance as workload changes.

No tests, smoke runs, model exports, training, engine builds, live inference, or benchmarks were run while implementing these files. The native binary has not been compiled: the development machine currently exposes CUDA 13, rather than the required CUDA 11.8/TensorRT 8.6.1 SDK.

## 1. Prepare a candidate on the training machine

Use the existing training environment, or a separate Python 3.11+ environment with the project's CUDA PyTorch installation and:

```powershell
python -m pip install -r requirements-yolo1050-preparation.txt
python prepare_yolo1050.py --screen-width 1920 --screen-height 1080
```

The screen arguments describe the intended aspect ratio. All 16:9 display resolutions produce the same engine input shape; runtime coordinates and mouse calibration use the friend's actual display dimensions. Other aspect ratios require a separately prepared/exported shape. The long edge is always 1024.

Preparation retains a hashed source checkpoint, FP32 static ONNX (opset 17, batch 1, trained one-to-one head), export metadata, and 512 training-only calibration PNGs under `artifacts/yolo1050/nano-fp32`. The source checkpoint and original inference scripts are not changed. Existing candidate directories are never overwritten.

Calibration selection cycles through class presence, small enemies, brightness bins, and filename prefixes. Filename prefixes are only a proxy for capture sessions/maps; use `--calibration-list path.txt` for a curated list of 512 distinct training image paths when explicit scene coverage is needed. Held-out paths and duplicated held-out file content are rejected. `--skip-calibration` prepares an FP32-only baseline or a QAT candidate without PTQ data.

Both native capture paths and the calibration helper use a defined 11-bit fixed-point bilinear resize, uint8 rounding, 114 padding, RGB NCHW, and division by 255. This makes their preprocessing consistent. It can differ slightly from the original OpenCV/Ultralytics interpolation, so the total accuracy allowance includes this change and rectangular padding.

## 2. Compile the native application

Install these **side by side with the existing Blackwell environment**, without replacing it:

- TensorRT **8.6.1 Windows x64 CUDA 11.8 ZIP SDK**, including its `include` and `lib` folders.
- CUDA Toolkit **11.8**, with Visual Studio integration.
- The matching **cuDNN 8.9.0 CUDA 11.x** runtime dependencies.
- Visual Studio 2019 C++ build tools, Windows SDK, and CMake 3.24+.
- OpenCV 4 for Windows, using a compatible MSVC build.

From the repository root, adjusting SDK paths:

```powershell
cmake -S native/yolo1050 -B native/yolo1050/build -G "Visual Studio 16 2019" -A x64 -T "cuda=11.8" -DTENSORRT_ROOT=C:/SDKs/TensorRT-8.6.1.6 -DOpenCV_DIR=C:/SDKs/opencv/build
cmake --build native/yolo1050/build --config Release
```

The build targets `sm_61` explicitly and rejects non-8.6.1 TensorRT headers and non-11.8 CUDA compilers. It obtains nlohmann/json 3.11.3 during configuration unless that version is already installed. It does not configure or create a test suite.

The output is `native/yolo1050/build/Release/yolo1050.exe`. The SDK runtime DLL directories must be on the executable's DLL search path: TensorRT `lib`, CUDA 11.8 `bin`, cuDNN `bin`, and the OpenCV runtime directory (commonly `build/x64/vc16/bin`). Keep the Microsoft Visual C++ runtime installed. The friend's live deployment does not need Python, PyTorch, Triton, ModelOpt, or the training dataset. Engine construction additionally uses TensorRT's ONNX parser and the prepared calibration bundle.

## 3. Build engines on the GTX 1050

Copy the native executable, matching runtime dependencies, and `artifacts/yolo1050/nano-fp32` bundle to the friend's machine. Retain the repository-style relative layout or adjust configuration paths. Build with Roblox closed.

```powershell
Copy-Item native/yolo1050/settings.example.json native/yolo1050/settings.local.json
native/yolo1050/build/Release/yolo1050.exe --config native/yolo1050/settings.local.json --precision fp32 --build-only
native/yolo1050/build/Release/yolo1050.exe --config native/yolo1050/settings.local.json --precision int8 --build-only
```

`--build-only` does not start capture, control threads, an overlay, model validation, or an FPS benchmark. TensorRT necessarily executes calibration/model kernels and times tactics during construction. Each precision gets a separate cache key and retains its own engine, calibration cache, and timing cache.

Cache identity includes ONNX/source hashes, input geometry, preprocessing, calibration-manifest hash, workspace, precision, FP32 layer overrides, GPU identity, driver, and CUDA/TensorRT/cuDNN versions. The separate `artifacts/yolo1050/engine-cache` never reads the Blackwell cache. Live startup requires a matching engine and never builds one automatically.

INT8 uses TensorRT 8.6's entropy calibration rather than the modern Ultralytics TensorRT exporter. Final box/classification convolutions identified in the ONNX export remain FP32. Additional sensitive layers can be selected by substrings of their ONNX layer names through `fp32_layer_patterns`. Missing patterns fail explicitly. QAT exports derive precision from Q/DQ nodes instead, and reject conflicting layer overrides.

## 4. Configure and run

Run the existing `calibrate.py` on the friend's machine once its display resolution is set. The calibration file must match the actual primary display.

```powershell
python calibrate.py
native/yolo1050/build/Release/yolo1050.exe --config native/yolo1050/settings.local.json
```

Press **=** to arm, **-** to pause, and **Ctrl+C** to stop. The existing aim smoothing, target matching, miss count, enemy filtering, and mouse/shooting defaults are ported to C++. `controls.auto_shoot` and `controls.instant_mouse` remain editable. The application starts paused.

Paths resolve relative to the settings file. Automatic capture selection finds the calibrated primary display's adapter and output; this can differ from `gpu_index`. `capture_backend` accepts `auto`, `cpu`, or `cuda`. Requesting an unavailable CUDA path fails rather than hiding the issue. Automatic mode logs its pinned CPU fallback.

Capture uses three owned slots, one inference in flight, and one replaceable pending frame. GPU-read slots cannot be overwritten. Pointer-only capture notifications and repeated desktop presents do not count as fresh captures. Both generation checks and a 100 ms deadline prevent old results from surviving pause/rearm. Display loss pauses controls; geometry changes require restart and recalibration. Rotated displays are rejected.

`overlay` defaults to false. When enabled, it is click-through, draws bounded all-class boxes/labels, and clears on pause/staleness. It disables itself if Windows capture exclusion is unavailable. Collection and recording remain in the existing Python tools; requesting them in this native configuration fails explicitly.

One nonblocking CUDA stream, persistent device/pinned buffers, precomputed resize coefficients, a blocking completion event, and an optional TensorRT-only CUDA graph reduce overhead. D3D resource mapping is outside the graph. If graph capture fails, the program logs its asynchronous `enqueueV3` fallback. Capture on the NVIDIA display uses persistent registered D3D textures and fused CUDA preprocessing. Intel-driven displays use staging readback and resize directly into pinned BGR, avoiding a second full-frame CPU color-conversion buffer.

Every five seconds while armed, reporting shows fresh desktop inference FPS, accepted publication FPS, capture FPS, superseded frames, unchanged notifications, stale results, stage wall times, and accepted capture-start-to-publication mean/p95. `stage_timing: true` additionally enables CUDA event timing for upload/preprocessing, model execution, and result download. GPU memory used/free describes **all processes**, not the application's incremental memory alone. Optional NVML temperature/SM-clock reporting uses the installed driver DLL; it is omitted if unavailable.

Fresh desktop presents are not a proof of unique game-rendered frames. An unchanged screen can correctly report few predictions. FPS, overlay refresh, capture FPS, and aim updates are separate rates. The runtime does not reduce input resolution, substitute tracking, switch inference backends, or raise process priority to meet the target.

## 5. Recover accuracy or reduce model computation

The helper defaults to preparation only:

```powershell
python train_yolo1050.py --mode prepare
```

This writes 75% and 50% width architecture configurations. Ultralytics rounds physical channel widths to multiples of eight and constructs compatible connected/residual dimensions. All detection scales and classes are retained. These are narrower architectures, not zero-masked models. Compatible tensors and leading-channel slices initialize a new candidate; this initialization does not claim feature/accuracy equivalence and requires training.

When training is explicitly wanted:

```powershell
python train_yolo1050.py --mode distill --width-ratio 0.75 --output artifacts/yolo1050/distill75
python train_yolo1050.py --mode distill --width-ratio 0.5 --output artifacts/yolo1050/distill50
```

The existing medium checkpoint is the teacher, default training is 100 epochs at image size 1024, and `--width-ratio 1.0` can fine-tune the unchanged nano. Distillation checkpoints are unwrapped to retain the trained student; narrower candidates retain their transferred initialization when the trainer creates the model. Each run retains its own artifacts. The script uses one training device, disables early stopping, and suppresses per-epoch/final evaluation and AMP's automatic compatibility comparison, so training does not run an accuracy evaluation or smoke comparison. Use `weights/last.pt`: `best.pt` is not an accuracy-selected checkpoint with validation disabled.

For QAT, install ModelOpt in the training environment only, then fine-tune an existing trained checkpoint:

```powershell
python -m pip install "nvidia-modelopt[torch]"
python train_yolo1050.py --mode qat --checkpoint runs/yolo26n/weights/best.pt --output artifacts/yolo1050/qat-nano
python prepare_yolo1050.py --checkpoint artifacts/yolo1050/qat-nano/qat/weights/last.pt --output artifacts/yolo1050/nano-qat --skip-calibration
```

QAT keeps the selected checkpoint's architecture; its default fine-tuning is five epochs at learning rate 1e-5. ONNX preparation checks for positive constant FP32 scales, signed symmetric INT8 zero points, per-tensor activations, and output-channel-axis per-channel weights compatible with TensorRT 8.6. These structural checks do not measure accuracy. Distillation/QAT recover accuracy; faster inference comes from the exported narrower/quantized graph.

## 6. Qualification remains explicit

No performance/accuracy qualification is automatic. When measurement is requested, compare the current square FP32 nano baseline with each full native candidate under matched gameplay after thermal stabilization. Include distant enemies, teammate confusion, both capture backends, pause/rearm, capture interruption, empty scenes, and memory pressure. The existing 186 validation and 22 test images may need more independent held-out scenes.

`qualification.example.json` is an empty measurement template, not generated benchmark data. Accuracy values use fractions: a one-percentage-point allowance is 0.01. Supply measured engine hashes and paths, then:

```powershell
python select_yolo1050.py --measurements native/yolo1050/qualification.local.json --output-config native/yolo1050/settings.selected.local.json
```

Selection only reads supplied measurements; it never runs a model. It requires 60 fresh predictions/s, capture-to-publication p95 at or below 16.67 ms, mAP loss at most 0.01, enemy and distant-enemy recall loss at most 0.02, Roblox FPS/p95 frame-time regression at most 5%, and additional VRAM at most 512 MiB. Use accepted publication FPS for the fresh prediction measurement. Engine hashes and the measured configuration's export are checked against cache metadata. It chooses highest fresh throughput, then lowest latency among qualifying engines; no passing candidate means it reports the supplied rates and writes no deployment configuration.

Qualification records refer to one measured engine, gameplay workload, and runtime configuration. Changed overlays, capture backends, drivers, power limits, or game workloads need new measurements. The native telemetry provides pipeline evidence but does not itself measure Roblox FPS or mAP. CPU/iGPU inference, alternate compilers, lower resolution, center crops, fewer predictions, or lower Roblox graphics are outside this implementation and remain separate decisions.

## References

- [TensorRT 8.6.1 support matrix](https://archive.docs.nvidia.com/tensorrt/tensorrt-861/support-matrix/index.html)
- [TensorRT 8.6 ONNX operator support](https://github.com/onnx/onnx-tensorrt/blob/8.6-GA/docs/operators.md)
- [CUDA 11.8 D3D11 interoperability](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/group__CUDART__D3D11.html)
- [Ultralytics NMS-free export](https://docs.ultralytics.com/modes/export)
