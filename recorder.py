"""Press = to start screen recording and - to stop. Press Ctrl+C to exit."""

import shutil
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

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


def record_video(camera, ffmpeg_path: str, video_path: Path, stop_event: threading.Event,
                 *, fps: int, preset: str, quality: int, on_frame=None, paced=False,
                 preview_fps: int | None = None, on_preview=None) -> int:
    """Encode frames, with optional faster previews and an indexed encoded-frame callback."""
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("FPS must be a positive integer")
    if preview_fps is not None and (not isinstance(preview_fps, int) or preview_fps <= 0):
        raise ValueError("Preview FPS must be a positive integer")
    capture_fps = max(fps, preview_fps or fps)
    paced = paced or preview_fps is not None
    if video_path.exists():
        raise FileExistsError(video_path)
    log_path = video_path.with_suffix(".ffmpeg.log")
    partial_path = video_path.with_name(f".{video_path.stem}.partial.mp4")
    process = log_file = None
    exit_code = None
    camera_started = False
    frame_count = 0
    dropped_video_frames = 0
    encoder_thread = None
    encoder_stop = threading.Event()
    encoder_queue = queue.Queue(maxsize=2)
    encoder_errors = []

    def encode_frame(frame):
        nonlocal frame_count
        try:
            payload = memoryview(frame).cast("B") if frame.flags.c_contiguous else frame.tobytes()
            process.stdin.write(payload)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"FFmpeg could not accept frames. See {log_path}") from exc
        if on_frame is not None:
            on_frame(frame, frame_count)
        frame_count += 1

    def encode_pending():
        try:
            while not encoder_stop.is_set() or not encoder_queue.empty():
                try:
                    pending = encoder_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                encode_frame(pending)
        except Exception as exc:
            encoder_errors.append(exc)

    try:
        camera.start(target_fps=capture_fps, video_mode=True)
        camera_started = True
        frame = camera.get_latest_frame(copy=True)
        if frame is None:
            raise RuntimeError("DXcam did not provide a frame")
        height, width = frame.shape[:2]
        command = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "h264_nvenc", "-preset", preset,
            "-rc", "vbr", "-cq", str(quality), "-b:v", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial_path),
        ]
        log_file = log_path.open("wb")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=log_file, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if preview_fps is not None:
            encoder_thread = threading.Thread(target=encode_pending, name="video-encoder", daemon=True)
            encoder_thread.start()
        next_frame_at = time.perf_counter()
        next_video_at = next_frame_at
        while not stop_event.is_set():
            if encoder_errors:
                raise RuntimeError("Video encoder failed") from encoder_errors[0]
            if process.poll() is not None:
                raise RuntimeError(f"FFmpeg stopped unexpectedly. See {log_path}")
            if paced:
                delay = next_frame_at - time.perf_counter()
                # DXcam already paces fresh preview frames. A second sleep can
                # miss a capture and turn a 60 Hz preview into a 30 Hz preview.
                if preview_fps is None and delay > 0 and stop_event.wait(delay):
                    break
                frame = camera.get_latest_frame(copy=True)
                if frame is None:
                    continue
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Screen size changed during recording")
            if on_preview is not None:
                on_preview(frame)
            if preview_fps is None or time.perf_counter() >= next_video_at:
                if encoder_thread is None:
                    encode_frame(frame)
                else:
                    try:
                        encoder_queue.put_nowait(frame)
                    except queue.Full:
                        # Never let a stalled encoder block fresh preview frames.
                        dropped_video_frames += 1
                next_video_at += 1 / fps
                if next_video_at < time.perf_counter() - 1 / fps:
                    next_video_at = time.perf_counter()
            if paced:
                next_frame_at += 1 / capture_fps
                if next_frame_at < time.perf_counter() - 1 / capture_fps:
                    next_frame_at = time.perf_counter()
            else:
                next_frame = camera.get_latest_frame(copy=True)
                if next_frame is not None:
                    frame = next_frame
    finally:
        try:
            if camera_started and camera.is_capturing:
                camera.stop()
        finally:
            if encoder_thread is not None:
                encoder_stop.set()
                encoder_thread.join(timeout=10)
                if encoder_thread.is_alive():
                    process.kill()
                    encoder_thread.join(timeout=2)
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
            if log_file is not None:
                log_file.close()
            if process is not None and exit_code == 0:
                # Same-directory rename publishes only a finalized video.
                partial_path.replace(video_path)
                print(f"Saved {video_path} ({frame_count} frames at {fps} FPS).")
    if exit_code != 0:
        raise RuntimeError(f"FFmpeg exited with code {exit_code}. See {log_path}")
    if encoder_errors:
        raise RuntimeError("Video encoder failed") from encoder_errors[0]
    if dropped_video_frames:
        print(f"Warning: video encoding fell behind; dropped {dropped_video_frames} video frames.")
    return frame_count


def record_once(camera, ffmpeg_path: str, stop_event: threading.Event) -> Path:
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"recording_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
    print(f"Recording to {video_path} at {FPS} FPS. Press - to stop.")
    record_video(camera, ffmpeg_path, video_path, stop_event,
                 fps=FPS, preset=NVENC_PRESET, quality=NVENC_QUALITY)
    print("Press = to record again.")
    return video_path


def main() -> None:
    import dxcam
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
