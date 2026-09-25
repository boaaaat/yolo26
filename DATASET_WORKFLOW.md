# Label and generate datasets

1. Open `labeler.py`. It resumes the last folder and image for that dataset, or starts in `datasets/unlabeled` the first time. Use **Open dataset…** to switch datasets and **Open folder…** to choose an image folder.
2. Draw or edit boxes. **Save** keeps a draft beside the image in `unlabeled`.
3. **Finish → Labeled** moves the reviewed image and its label to `datasets/labeled/images` and `datasets/labeled/labels`. Use **Open labeled** to review or edit it again.
4. Click **Generate dataset…**. Choose the classes, split percentages, and how many augmented training copies to make. The dialog can include the existing `train`, `valid`, and `test` images without moving them.

Use **Manage classes…** to add or rename classes and choose their box colors. Class IDs keep their order so existing YOLO labels remain valid. The dataset's `labeler.yaml` stores these classes, colors, generation settings, and the last folder, image, and active class. `labeler_recent.yaml` beside the script remembers which dataset was last open. When classes change, the dataset's `data.yaml` gets updated with their names and count. A dataset without metadata starts with names from its `data.yaml`; if it has only label files, placeholder names are inferred from the IDs for you to rename.

Each generation creates a new `datasets/versions/vN` folder with YOLO `train`, `valid`, and `test` image/label folders, `data.yaml`, `labeler.yaml`, and `generation.yaml`. The version's `data.yaml` contains only the selected classes with contiguous IDs; its `labeler.yaml` keeps their colors. Existing versions and labeled sources remain untouched. Existing split images are combined and split again. Exact duplicate images are included once. Images containing only excluded classes are skipped; mixed images keep only included boxes, and included class IDs are renumbered.

Augmentations affect only additional training copies. Original images go into all three splits, and images are split before copies are made. The available options are horizontal flip, rotation, brightness, contrast, and Gaussian blur. The generator keeps the source image dimensions; `train.py` resizes them to its configured `IMAGE_SIZE` during training.

`train.py` automatically uses the newest generated version in the most recently opened dataset and requires a version before training. To train a particular version or the old dataset, set `DATASET_PATH` in `train.py` to its `data.yaml`.
