# YOLO26 labeling and training tools

Python tools for labeling object detection images, generating versioned YOLO datasets, and training a YOLO26 model. The labeler is a PySide6 desktop app. It starts with `datasets/unlabeled`, can open other image folders, and finds the corresponding YOLO labels when editing existing images.

## Setup

Install Python 3.10 or newer and the packages needed for labeling and training:

```powershell
python -m pip install PySide6 PyYAML Pillow ultralytics
```

Run the labeler from this directory:

```powershell
python labeler.py
```

The labeler creates and stores dataset metadata in your local `datasets` folder. Use **Open dataset** to work with another dataset. See [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md) for labeling, class management, and dataset generation.

After generating a dataset version, edit the settings near the top of `train.py` and run:

```powershell
python train.py
```

The default training image size is 1024 and batch size is 2. Training also needs a compatible PyTorch installation and model weights. Ultralytics can obtain its standard model weights by name; the other model preparation and inference scripts may require files under `models/`, `weights/`, or `runs/` and additional packages such as OpenCV, DXcam, pywin32, or Transformers.

Datasets, annotations, model weights, run output, recordings, and local labeler state are intentionally excluded from Git. Add your own images under `datasets/unlabeled` or use **Open dataset** in the labeler.
