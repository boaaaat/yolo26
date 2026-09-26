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

The default dataset is `datasets/rivals`, with its own metadata, `labeled`, `unlabeled`, and `versions` folders. Use **Open dataset** to work with another dataset. See [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md) for labeling, class management, and dataset generation.

After generating a dataset version, edit the settings near the top of `train.py` and run:

```powershell
python train.py
```

To collect gameplay frames for review, set `CHECKPOINT_PATH` and other variables at the top of `active_collector.py`, then run `python active_collector.py`. Press `=` to start collecting, `-` to pause, and Ctrl+C to exit. It runs detection at 1 FPS and saves selected numbered JPGs in `datasets/rivals/unlabeled`. Predictions live in `.review` JSON files until you accept or correct them in the labeler. Refresh the labeler's queue if it was already open.

Use **Generate dataset…** after finishing a batch of images, then run `train.py` separately. Its default is a new training run from pretrained `yolo26m.pt` on the latest generated dataset version. Set the collector's `CHECKPOINT_PATH` to a new `best.pt` when you decide to use that model.

The default training image size is 1024 and batch size is 4. Training also needs a compatible PyTorch installation and model weights. Ultralytics can obtain its standard model weights by name; the other model preparation and inference scripts may require files under `models/`, `weights/`, or `runs/` and additional packages such as OpenCV, DXcam, pywin32, or Transformers.

Datasets, annotations, model weights, run output, recordings, and local labeler state are intentionally excluded from Git. Add your own images under `datasets/rivals/unlabeled` or use **Open dataset** in the labeler.
