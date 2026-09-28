# YOLO26 labeling and training tools

Python tools for labeling object detection images, generating versioned YOLO datasets, and training a YOLO26 model. The labeler is a PySide6 desktop app. It starts with `datasets/rivals/unlabeled` and switches between the unlabeled and labeled folders of the selected dataset root.

## Shared helpers

Run the existing scripts directly and edit their settings at the top as before. All scripts remain in the repository root.

- `dataset_utils.py` shares YOLO label parsing/writing, class-name handling, image discovery, atomic file writes, and bulk prediction transfers. Callers retain their own label boundary tolerances and missing-file policies.
- `inference_controls.py` shares target matching, aim state, mouse controls, calibration loading, and timing. Each bot passes its own options; the FP32 entry point passes overrides to the original PyTorch loop without changing its module globals. TensorRT and PyTorch keep their existing inference pipelines.
- `inference_runtime.py` shares the PyTorch model setup, BF16 autocast or FP32 prediction, fixed input shape, compilation, warmup, and compiler caching between `inference_bot.py` and `active_collector.py`. The FP32 entry point uses the same runtime with compilation disabled.
- `inference_collection.py` shares candidate selection, duplicate rejection, numbering, and image/review saving. Active collection and live inference use its bounded background writer so hashing, JPEG encoding, and disk writes do not block prediction. Active collection includes the exact recorded frame index and capture session in each queued sample. Collection settings remain in `active_collector.py`.
- `inference_overlay.py` shares the click-through, capture-excluded 60 FPS overlay, class colors, confidence labels, and prediction-to-screen coordinate conversion between both inference bots and active collection. High-resolution frame deadlines replace the coarse Windows message timer. Boxes are drawn into a reusable back buffer and presented together to prevent clear-then-draw flicker.
- `recorder.py` exposes the encoder used by both standalone recording and active collection, with an optional review-frame callback and faster preview callback. Preview recording uses a bounded encoder queue so video writes cannot block capture; overload drops are reported. Review frame indices refer to encoded video frames. Completed recordings are finalized from a partial MP4.
- `training_dashboard.py` registers common dashboard callbacks, and `train.py` exposes the checkpoint-resume check reused by distillation. Training modes and dashboard behavior remain unchanged.

The labeler, auto-labeler, and active collector copy detection boxes from GPU to CPU once per result.

## Optimized inference on RTX 5070 / 5080

`optimized_inference_bot.py` uses the medium checkpoint with a fixed rectangular FP16 TensorRT engine. It preserves the original 1024-pixel long-edge scale and full-screen coverage: a 3840x2160 or 2560x1440 display uses a 1024x576 input instead of adding padding to 1024x1024. Other aspect ratios receive only the padding needed for a stride of 32. No crosshair crop or automatic resolution reduction is used. Removing padding changes boundary context, and FP16 can change predictions; accuracy and the 120 FPS target have not been measured.

Activate the existing CUDA 13 `yolo` environment and install the additional dependencies:

```powershell
conda activate yolo
python -m pip install -r requirements-inference-blackwell.txt
python optimized_inference_bot.py
```

The pinned TensorRT 10.16.1 CUDA 13 package includes the native builder/runtime libraries and Python bindings. A separate SDK ZIP or Python interpreter is unnecessary for this implementation. It does not install `trtexec`. The existing CUDA PyTorch and Ultralytics installation remains required; this implementation uses the `quantize` export API in Ultralytics 8.4.162. DXcam must support `grab(copy=False, new_frame_only=True)` (the installed DXcam does). Optional fused preprocessing uses the existing `triton-windows` installation, with a PyTorch CUDA fallback when unavailable.

Settings are at the top of `optimized_inference_bot.py`. Your current `draw_boxes_overlay` setting is `True`, and `collect_data` is `False`; set both to `False` for the 120 FPS target. Existing calibration is required for live operation (`python calibrate.py`). Press `=` to arm, `-` to pause, and Ctrl+C to exit. Target matching and mouse/shooting behavior use `inference_controls.py`; control settings are exposed in the optimized entry point.

The first launch exports/builds an engine in a separate process; this can take several minutes and use substantial GPU resources. Later launches reuse `.optimized_engine_cache/` when the checkpoint contents, shape, GPU, precision, and software versions match. Export intermediates stay in a temporary cache directory, leaving training checkpoints untouched. Each GPU builds its own engine. To build the engine without starting capture or controls:

```powershell
python optimized_inference_bot.py --build-only
```

Engine building necessarily traces the model and lets TensorRT time kernel choices; it does not run a detection validation suite or FPS benchmark. The live runtime warms its buffers and captures a CUDA graph before arming. Graph capture and fused preprocessing each have an explicit, logged fallback; TensorRT failures do not silently fall back to the old predictor. Remove the affected `.engine` and matching `.json` from `.optimized_engine_cache/` to force a rebuild.

The capture worker owns DXcam and resizes into a three-slot pinned buffer pool. Only the newest pending frame is consumed; in-flight buffers cannot be overwritten. The direct runtime uses persistent device buffers, asynchronous copies, the NMS-free head with 300 candidates, and one completion wait per result. Capture/resize overlap GPU processing. Stale results expire after 100 ms, and results from before a pause cannot be applied after rearming. The original Python busy-spin wait is replaced in this entry point with sleeping waits.

Every five seconds while armed, the bot reports fresh desktop-capture inference FPS, published-result FPS, capture rate, superseded captures, no-new-frame grabs, and stale results. Timings show capture, CPU resize, transfer plus GPU plus result completion, and capture-start-to-publication mean/p95. These are wall-clock pipeline measurements, not isolated GPU-kernel timings or display-presentation latency. Fresh desktop captures can include desktop/overlay changes; they are not a measurement of unique game-rendered frames. An unchanged screen may report very low FPS because cached frames are deliberately not inferred again.

CPU capture/resize remains the initial capture backend. Native Direct3D/CUDA capture, TensorRT-RTX, INT8, and FP8 are later options if measurements justify them; they are not enabled here. No tests, smoke tests, engine builds, or FPS benchmarks were run as part of implementing this entry point.

## Setup

Install Python 3.10 or newer and the packages needed for labeling and training:

```powershell
python -m pip install PySide6 PyYAML Pillow ultralytics opencv-python dxcam pywin32
```

Run the labeler from this directory:

```powershell
python labeler.py
```

The default dataset is `datasets/rivals`, with its own metadata, `labeled`, `unlabeled`, and `versions` folders. Use **Open dataset** to work with another dataset, or **Import ZIP…** to merge labeled images, unlabeled images, and active collector captures into the open dataset with unique filenames. **Browse dataset splits…** opens a read-only view of generated versions with label boxes, class and split filters, and filename/class/split sorting. The suggestion model list finds `best.pt` and `last.pt` in every folder under `runs`; reopen the list or use **Refresh training runs** after a new run finishes. See [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md) for labeling, class management, and dataset generation.

After generating a dataset version, edit the settings near the top of `train.py` and run:

```powershell
python train.py
```

Training opens a dashboard in your browser and saves `training_dashboard.html` and `training_dashboard.png` in the run folder. It shows mAP, precision, recall, losses, and learning rate, refreshing after each epoch. With `TRAIN_MODE = "resume"`, the graph loads earlier epochs from that run's `results.csv` before training continues. Set `OPEN_DASHBOARD = False` in `train.py` if you only want the saved files.

To train a smaller model from an existing run with knowledge distillation, edit the settings at the top of `train_distill.py` and run:

```powershell
python train_distill.py
```

Set `TEACHER_RUN` to the source run or a specific checkpoint. The script uses its `weights/best.pt` (or `last.pt` if there is no best checkpoint), dataset from `args.yaml`, and image size. `STUDENT_MODEL` defaults to pretrained `yolo26n.pt`; set it to `yolo26s.pt` for small. The other editable settings include `DATASET_PATH`, `EPOCHS`, `BATCH_SIZE`, `DISTILL_WEIGHT`, `RUN_NAME`, and `OPEN_DASHBOARD`. Results go into a new run under `runs/`. To resume an interrupted student run, set `TRAIN_MODE = "resume"` and `CHECKPOINT_PATH` to that run's `weights/last.pt`. Resume restores the saved epoch, optimizer, teacher, and original epoch total; the dashboard reads the existing metrics.

To collect gameplay frames for review, set `CHECKPOINT_PATH` and other variables at the top of `active_collector.py`, then run `python active_collector.py`. Press `=` to start collecting and recording, `V` to save a review frame, `-` to stop, and Ctrl+C to exit. With `draw_boxes_overlay = True` (the default), live predictions and the shared overlay target 60 FPS; achieved detection FPS depends on inference and capture speed. The overlay shows all classes above `OVERLAY_CONFIDENCE` (0.50), scales boxes to the primary display, and clears on stop or stale results. Set `draw_boxes_overlay = False` to disable the preview work. Review sampling remains controlled by `INFERENCE_FPS` (1 FPS) and video encoding by `VIDEO_FPS` (20 FPS). Selected numbered JPGs are saved in `datasets/rivals/unlabeled`, with matching encoded video frame indices. Predictions live in `.review` JSON files until you accept or correct them in the labeler. Refresh the labeler's queue if it was already open.

Active collection defaults to `PRECISION = "bf16"`, `COMPILE_MODE = "reduce-overhead"`, and the YOLO26 NMS-free head, matching the PyTorch inference bot. Compilation and warmup happen before recording starts. Both scripts share `.inference_compile_cache` when the checkpoint, GPU, input size, precision, head mode, and software versions match. The first launch can take longer to compile; later launches load saved compiler artifacts. These settings remain editable at the top of each script.

Preview inference follows fresh capture frames directly, without a second FPS timer. Review samples reuse predictions when the exact recorded frame is in the recent prediction cache; older review frames never replace the live overlay. Every `REPORT_INTERVAL_SECONDS` (5 by default), the collector reports capture, fresh preview, model-call, and paint rates, plus model/result-transfer mean and p95 latency. Unchanged boxes do not repaint, so a low paint rate on a static or empty scene is expected.

While armed with `collect_data = True`, inference bots reuse the live model for occasional all-class, low-confidence samples (1 per second by default); a bounded writer thread applies the active collector's selection rules and saves chosen images and `.review` metadata to `datasets/rivals/unlabeled`. Resize, duplicate checking, JPEG encoding, and disk writes run on that thread. Edit the settings in `active_collector.py` to change the collection rate, confidence ranges, or save limits. Set `collect_data = False` in the inference script to disable collection. If dataset metadata is missing or the model classes do not match, inference continues and prints why collection was disabled.

Before inference, run `python calibrate.py`. Join Rivals, go to the test area, lock the mouse to the center, keep it still, and press `=`. This saves `mouse_calibration.json` with the locked cursor position. If the file is missing or the primary display resolution changes, inference tells you to calibrate again.

To run compiled BF16 inference with enemy aiming, edit `inference_bot.py` and run `python inference_bot.py` on Windows. It uses the primary display and `runs/yolo26m/weights/best.pt` by default. Press `=` to arm, `-` to pause, and Ctrl+C to exit. When several enemies are detected, it picks the one closest to the calibrated locked cursor and follows that target across frames. A brief detection gap pauses aiming; after four missed frames it may acquire a new target. Auto shoot only clicks inside the locked target's box. `AUTO_SHOOT` defaults to `True`; set it to `False` to aim without clicking. `instant_mouse` defaults to `False` for smooth movement; set it to `True` to move the full distance to each detected aim point in one mouse event. `INFERENCE_TARGET_FPS` is a pacing target, and the script reports its measured inference FPS. It requires a CUDA GPU with native BF16 support and a PyTorch build that can compile the model. On native Windows with PyTorch 2.14, install the matching compiler package in the same environment with `python -m pip install "triton-windows>=3.8,<3.9"`. The first run saves PyTorch compiler artifacts in `.inference_compile_cache`; later runs reuse them when the checkpoint, model settings, GPU, and compiler versions match. Each process still warms up CUDA graphs before arming.

For a Pascal GPU such as a GeForce GTX 1050, edit `inference_bot_fp32.py` and run `python inference_bot_fp32.py` instead. It uses FP32 CUDA inference without BF16, Triton, or `torch.compile`, while sharing the same calibration and controls. Its default model input is 1024×1024 and its 30 FPS setting is a target, not a guaranteed speed. `NMS_FREE = True` selects YOLO26's one-to-one detection head to skip NMS; set it to `False` to use the original head if detections change. Every five seconds it reports average frame, capture, preprocess, inference, and postprocess times alongside actual FPS. A smaller checkpoint can reduce model time without lowering the input resolution. Install a PyTorch build with CUDA 12.6 support for Pascal in your friend's environment; CUDA 13 PyTorch builds do not include Pascal kernels.

Set `draw_boxes_overlay = True` in either inference script to draw boxes for every detected class over the primary display. Each class has its own color, and each box shows its class name and confidence. Aiming and auto shoot still use only the enemy class. The overlay is click-through and clears when the bot is paused. It redraws independently at up to 60 FPS, using the latest inference result. Windows capture exclusion is requested for the overlay; if that request fails, the script warns that boxes may appear in captured frames. Windowed or borderless game mode is recommended because exclusive fullscreen can cover desktop overlays.

Use **Generate dataset…** after finishing a batch of images, then run `train.py` separately. Its default is a new training run from pretrained `yolo26m.pt` on the latest generated dataset version. Set the collector's `CHECKPOINT_PATH` to a new `best.pt` when you decide to use that model.

The default training image size is 1024 and batch size is 4. Training also needs a compatible PyTorch installation and model weights. Ultralytics can obtain its standard model weights by name; the other model preparation and inference scripts may require files under `models/`, `weights/`, or `runs/` and additional packages such as OpenCV, DXcam, pywin32, or Transformers.

Datasets, annotations, model weights, run output, recordings, and local labeler state are intentionally excluded from Git. Add your own images under `datasets/rivals/unlabeled` or use **Open dataset** in the labeler.
