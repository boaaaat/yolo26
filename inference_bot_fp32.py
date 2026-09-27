"""FP32 inference for Pascal GPUs such as the GeForce GTX 1050.

Uses the same calibration, hotkeys, target lock, aiming, and shooting behavior as
inference_bot.py. Pascal cannot use its BF16 or Triton compilation path.
Edit the settings below, then run: python inference_bot_fp32.py
"""

from pathlib import Path

import inference_bot as bot


CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs" / "yolo26m" / "weights" / "best.pt"
GPU_INDEX = 0
DXCAM_DEVICE_INDEX = 0
DXCAM_OUTPUT_INDEX = 0
IMAGE_SIZE = 1024  # Keep the full 1024x1024 model input.
INFERENCE_TARGET_FPS = 30  # Pacing target; actual FPS depends on the GPU and model.
CONFIDENCE = 0.50
ENEMY_CLASS_NAME = "enemy"
NMS_FREE = True  # Use YOLO26's one-to-one head and skip non-maximum suppression.
REPORT_STAGE_TIMES = True  # Show capture, preprocessing, model, and postprocessing costs.
WARMUP_PASSES = 2

AUTO_SHOOT = True
instant_mouse = False


def main() -> None:
    bot.CHECKPOINT_PATH = CHECKPOINT_PATH
    bot.GPU_INDEX = GPU_INDEX
    bot.DXCAM_DEVICE_INDEX = DXCAM_DEVICE_INDEX
    bot.DXCAM_OUTPUT_INDEX = DXCAM_OUTPUT_INDEX
    bot.IMAGE_SIZE = IMAGE_SIZE
    bot.INFERENCE_TARGET_FPS = INFERENCE_TARGET_FPS
    bot.CONFIDENCE = CONFIDENCE
    bot.ENEMY_CLASS_NAME = ENEMY_CLASS_NAME
    bot.PREDICT_NMS = False if NMS_FREE else None
    bot.REPORT_STAGE_TIMES = REPORT_STAGE_TIMES
    bot.WARMUP_PASSES = WARMUP_PASSES
    bot.AUTO_SHOOT = AUTO_SHOOT
    bot.instant_mouse = instant_mouse
    bot.PRECISION = "fp32"
    bot.COMPILE_MODE = False
    bot.main()


if __name__ == "__main__":
    main()
