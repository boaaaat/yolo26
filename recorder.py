"""Press = to start screen recording and - to stop. Press Ctrl+C to exit."""

import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import dxcam
import win32api


# Settings
FPS = 60
OUTPUT_DIR = Path(__file__).resolve().parent / "recordings"
FFMPEG_EXECUTABLE = "ffmpeg"
DEVICE_INDEX = 0
OUTPUT_INDEX = 0  # Primary monitor on the selected graphics device.
NVENC_PRESET = "p5"
NVENC_QUALITY = 20  # Smaller values give higher quality and larger files.
KEY_POLL_SECONDS = 0.01

START_KEY = 0xBB  # VK_OEM_PLUS: the = / + key.
STOP_KEY = 0xBD  # VK_OEM_MINUS: the - / _ key.


def key_down(virtual_key: int) -> bool:
    return bool(win32api.GetAsyncKeyState(virtual_key) & 0x8000)


def watch_hotkeys(
    start_event: threading.Event,
    stop_event: threading.Event,
    shutdown_event: threading.Event,
) -> None:
    start_was_down = key_down(START_KEY)
    stop_was_down = key_down(STOP_KEY)
    while not shutdown_event.is_set():
        start_is_down = key_down(START_KEY)
        stop_is_down = key_down(STOP_KEY)
        if start_is_down and not start_was_down:
            start_event.set()
        if stop_is_down and not stop_was_down:
            stop_event.set()
        start_was_down = start_is_down
        stop_was_down = stop_is_down
        shutdown_event.wait(KEY_POLL_SECONDS)


def record_once(camera, ffmpeg_path: str, stop_event: threading.Event) -> Path:
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"recording_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
    log_path = video_path.with_suffix(".ffmpeg.log")
    process = None
    log_file = None
    frame_count = 0

    try:
        camera.start(target_fps=FPS, video_mode=True)
        frame = camera.get_latest_frame(copy=True)
        if frame is None:
            raise RuntimeError("DXcam did not provide a frame")
        height, width = frame.shape[:2]

        command = [
            ffmpeg_path,
            "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", str(FPS),
            "-i", "pipe:0",
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "h264_nvenc", "-preset", NVENC_PRESET,
            "-rc", "vbr", "-cq", str(NVENC_QUALITY), "-b:v", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(video_path),
        ]
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        print(f"Recording to {video_path} at {FPS} FPS. Press - to stop.")

        while not stop_event.is_set():
            if process.poll() is not None:
                raise RuntimeError(f"FFmpeg stopped unexpectedly. See {log_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Screen size changed during recording")
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError(f"FFmpeg could not accept frames. See {log_path}") from exc
            frame_count += 1
            next_frame = camera.get_latest_frame(copy=True)
            if next_frame is not None:
                frame = next_frame
    finally:
        if camera.is_capturing:
            camera.stop()
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                exit_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                exit_code = process.returncode
        else:
            exit_code = None
        if log_file is not None:
            log_file.close()

    if exit_code != 0:
        raise RuntimeError(f"FFmpeg exited with code {exit_code}. See {log_path}")
    print(f"Saved {video_path} ({frame_count} frames). Press = to record again.")
    return video_path


def main() -> None:
    if not isinstance(FPS, int) or FPS <= 0:
        raise ValueError("FPS must be a positive integer")
    ffmpeg_path = shutil.which(FFMPEG_EXECUTABLE)
    if ffmpeg_path is None:
        raise FileNotFoundError(f"FFmpeg not found: {FFMPEG_EXECUTABLE}")

    camera = dxcam.create(device_idx=DEVICE_INDEX, output_idx=OUTPUT_INDEX, output_color="BGR")
    start_event = threading.Event()
    stop_event = threading.Event()
    shutdown_event = threading.Event()
    hotkey_thread = threading.Thread(
        target=watch_hotkeys,
        args=(start_event, stop_event, shutdown_event),
        daemon=True,
    )
    hotkey_thread.start()
    print("Press = to start recording, - to stop, and Ctrl+C to exit.")
    try:
        while True:
            if not start_event.wait(timeout=0.1):
                continue
            start_event.clear()
            stop_event.clear()
            record_once(camera, ffmpeg_path, stop_event)
            start_event.clear()
            stop_event.clear()
    except KeyboardInterrupt:
        print("Recorder stopped.")
    finally:
        shutdown_event.set()
        hotkey_thread.join(timeout=1)
        camera.release()


if __name__ == "__main__":
    main()
