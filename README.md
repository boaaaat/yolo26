# YOLO26 labeling and training tools

Python tools for labeling object detection images, generating versioned YOLO datasets, and training a YOLO26 model. The labeler is a PySide6 desktop app. It starts with `datasets/rivals/unlabeled` and switches between the unlabeled and labeled folders of the selected dataset root.

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

To collect gameplay frames for review, set `CHECKPOINT_PATH` and other variables at the top of `active_collector.py`, then run `python active_collector.py`. Press `=` to start collecting, `-` to pause, and Ctrl+C to exit. It runs detection at 1 FPS and saves selected numbered JPGs in `datasets/rivals/unlabeled`. Predictions live in `.review` JSON files until you accept or correct them in the labeler. Refresh the labeler's queue if it was already open.

Before inference, run `python calibrate.py`. Join Rivals, go to the test area, lock the mouse to the center, keep it still, and press `=`. This saves `mouse_calibration.json` with the locked cursor position. If the file is missing or the primary display resolution changes, inference tells you to calibrate again.

To run compiled BF16 inference with enemy aiming, edit `inference_bot.py` and run `python inference_bot.py` on Windows. It uses the primary display and `runs/yolo26m/weights/best.pt` by default. Press `=` to arm, `-` to pause, and Ctrl+C to exit. When several enemies are detected, it picks the one closest to the calibrated locked cursor and follows that target across frames. A brief detection gap pauses aiming; after four missed frames it may acquire a new target. Auto shoot only clicks inside the locked target's box. `AUTO_SHOOT` defaults to `True`; set it to `False` to aim without clicking. `instant_mouse` defaults to `False` for smooth movement; set it to `True` to move the full distance to each detected aim point in one mouse event. `INFERENCE_TARGET_FPS` is a pacing target, and the script reports its measured inference FPS. It requires a CUDA GPU with native BF16 support and a PyTorch build that can compile the model. On native Windows with PyTorch 2.14, install the matching compiler package in the same environment with `python -m pip install "triton-windows>=3.8,<3.9"`. The first run saves PyTorch compiler artifacts in `.inference_compile_cache`; later runs reuse them when the checkpoint, model settings, GPU, and compiler versions match. Each process still warms up CUDA graphs before arming.

Use **Generate dataset…** after finishing a batch of images, then run `train.py` separately. Its default is a new training run from pretrained `yolo26m.pt` on the latest generated dataset version. Set the collector's `CHECKPOINT_PATH` to a new `best.pt` when you decide to use that model.

The default training image size is 1024 and batch size is 4. Training also needs a compatible PyTorch installation and model weights. Ultralytics can obtain its standard model weights by name; the other model preparation and inference scripts may require files under `models/`, `weights/`, or `runs/` and additional packages such as OpenCV, DXcam, pywin32, or Transformers.

Datasets, annotations, model weights, run output, recordings, and local labeler state are intentionally excluded from Git. Add your own images under `datasets/rivals/unlabeled` or use **Open dataset** in the labeler.
