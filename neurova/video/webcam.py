# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Webcam capture for Neurova using system tools (ffmpeg)."""

from __future__ import annotations
from collections import deque
import numpy as np
import subprocess
import sys
import platform
import threading
import time
import select
from typing import Optional, Iterator
from neurova.core.errors import VideoError

try:
    from neurova.video import camera_native as _camera_native

    _HAS_NATIVE_CAMERA = hasattr(_camera_native, "CameraCapture")
except Exception:  # pragma: no cover - optional native module
    _camera_native = None
    _HAS_NATIVE_CAMERA = False


class WebcamCapture:
    """Webcam capture using ffmpeg system tool.
    
    captures frames from webcam without requiring neurova or other python libraries.
    uses ffmpeg which must be installed on the system.
    
    examples:
        # basic usage
        cam = WebcamCapture()
        for frame in cam.frames(max_frames=100):
            # process frame (numpy array h x w x 3 rgb)
            pass
        cam.release()
        
        # with context manager
        with WebcamCapture() as cam:
            frame = cam.read()
    """
    
    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        pix_fmt: str = "rgb24",
    ):
        """initialize webcam capture.
        
        args:
            device: camera device index (0 for default camera)
            width: frame width in pixels
            height: frame height in pixels
            fps: frames per second
        """
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt = pix_fmt
        self._process: Optional[subprocess.Popen] = None
        self._is_open = False

        # Optional background reader (low latency)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._cond = threading.Condition()
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_id = 0

        # Keep a small tail of ffmpeg stderr for debugging.
        self._stderr_tail: deque[str] = deque(maxlen=80)
        
        # detect platform and set device path
        self._system = platform.system()
        self._device_path = self._get_device_path()
        self._native = None
    
    def _get_device_path(self) -> str:
        """get the device path for the current platform."""
        if self._system == "Darwin":  # macos
            # avfoundation expects "<video_index>:<audio_index>". Use video-only by default.
            return f"{self.device}:none"
        elif self._system == "Linux":
            return f"/dev/video{self.device}"
        elif self._system == "Windows":
            return f"video={self.device}"
        else:
            return str(self.device)
    
    def _get_ffmpeg_input_args(self) -> list:
        """get ffmpeg input arguments for the current platform."""
        if self._system == "Darwin":  # macos
            return [
                "-f", "avfoundation",
                "-nostdin",
                "-loglevel", "error",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-framerate", str(self.fps),
                "-video_size", f"{self.width}x{self.height}",
                "-i", self._device_path,
            ]
        elif self._system == "Linux":
            return [
                "-f", "v4l2",
                "-framerate", str(self.fps),
                "-video_size", f"{self.width}x{self.height}",
                "-i", self._device_path,
            ]
        elif self._system == "Windows":
            return [
                "-f", "dshow",
                "-framerate", str(self.fps),
                "-video_size", f"{self.width}x{self.height}",
                "-i", self._device_path,
            ]
        else:
            raise VideoError(f"unsupported platform: {self._system}")
    
    def _check_ffmpeg(self) -> bool:
        """check if ffmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def open(self) -> bool:
        """open the webcam.
        
        returns:
            true if successfully opened, false otherwise
        """
        if self._is_open:
            return True
        
        if _HAS_NATIVE_CAMERA:
            try:
                self._native = _camera_native.CameraCapture(
                    int(self.device), int(self.width), int(self.height)
                )
                if hasattr(self._native, "open"):
                    if not self._native.open():
                        self._native = None
                    else:
                        self._is_open = True
                        return True
            except Exception:
                self._native = None

        if not self._check_ffmpeg():
            raise VideoError(
                "ffmpeg not found. install ffmpeg:\n"
                "  macos: brew install ffmpeg\n"
                "  linux: sudo apt install ffmpeg\n"
                "  windows: download from ffmpeg.org"
            )
        
        try:
            # Build ffmpeg command (try a few input pixel formats on macOS)
            input_pixel_formats = [None]
            if self._system == "Darwin":
                # avfoundation requires choosing one of the device-supported input pixel formats.
                input_pixel_formats = ["nv12", "uyvy422", "yuyv422", "0rgb", "bgr0"]

            last_err = None
            for inp_pix in input_pixel_formats:
                cmd = ["ffmpeg", "-y"]
                args = self._get_ffmpeg_input_args()

                # Insert avfoundation input pixel format as an input option.
                if inp_pix is not None and self._system == "Darwin":
                    try:
                        # args starts with: -f avfoundation ... -i <device>
                        f_i = args.index("-f")
                        if args[f_i + 1] == "avfoundation":
                            insert_at = f_i + 2
                        else:
                            insert_at = 0
                    except Exception:
                        insert_at = 0
                    args = args[:insert_at] + ["-pixel_format", inp_pix] + args[insert_at:]

                cmd.extend(args)
                cmd.extend([
                    "-f", "rawvideo",
                    "-pix_fmt", str(self.pix_fmt),
                    "-",
                ])

                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                time.sleep(0.15)

                # If ffmpeg already exited, collect stderr and try next format.
                if self._process.poll() is not None:
                    try:
                        err = self._process.stderr.read().decode("utf-8", errors="ignore")
                    except Exception:
                        err = ""
                    last_err = (err.strip() or last_err)
                    self._process = None
                    continue

                # Validate that ffmpeg is actually producing frame bytes quickly.
                if self._process.stdout is not None and self._system != "Windows":
                    try:
                        r, _, _ = select.select([self._process.stdout], [], [], 2.0)
                    except Exception:
                        r = [self._process.stdout]
                    if not r:
                        # No stdout data within 2s => likely permissions/device mismatch.
                        err_snip = ""
                        if self._process.stderr is not None:
                            try:
                                er, _, _ = select.select([self._process.stderr], [], [], 0.25)
                                if er:
                                    err_snip = self._process.stderr.read(8192).decode(
                                        "utf-8", errors="ignore"
                                    )
                            except Exception:
                                err_snip = ""
                        try:
                            self._process.terminate()
                            self._process.wait(timeout=1.0)
                        except Exception:
                            try:
                                self._process.kill()
                            except Exception:
                                pass
                        last_err = (err_snip.strip() or last_err)
                        self._process = None
                        continue

                # Drain stderr so ffmpeg doesn't block (now that we know it runs).
                self._stderr_thread = threading.Thread(
                    target=self._drain_stream, args=(self._process.stderr,), daemon=True
                )
                self._stderr_thread.start()

                self._is_open = True
                return True

                try:
                    err = self._process.stderr.read().decode("utf-8", errors="ignore")
                except Exception:
                    err = ""
                last_err = err.strip() or last_err
                self._process = None

            hint = ""
            if self._system == "Darwin":
                hint = (
                    "\n\nmacOS hints:\n"
                    "- Make sure your terminal/Python has Camera permission in System Settings → Privacy & Security → Camera.\n"
                    "- Verify the device index with: ffmpeg -f avfoundation -list_devices true -i \"\""
                )
            raise VideoError(f"failed to open webcam via ffmpeg. {last_err or ''}{hint}")
            
        except Exception as e:
            raise VideoError(f"failed to open webcam: {e}")

    def _drain_stream(self, stream):
        """Continuously drain stderr, keeping a tail for debugging."""
        buf = b""
        try:
            while stream is not None:
                chunk = stream.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode("utf-8", errors="ignore").strip()
                    if s:
                        self._stderr_tail.append(s)
        except Exception:
            pass
        if buf:
            s = buf.decode("utf-8", errors="ignore").strip()
            if s:
                self._stderr_tail.append(s)

    def _read_exactly(self, nbytes: int) -> Optional[bytes]:
        if self._process is None or self._process.stdout is None:
            return None
        chunks = []
        remaining = int(nbytes)
        while remaining > 0:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    
    def read(self) -> Optional[np.ndarray]:
        """read a single frame from webcam.
        
        returns:
            frame as numpy array (height x width x 3 rgb), or none if failed
        """
        if not self._is_open:
            self.open()
        
        if self._native is not None:
            frame = self._native.read()
            if frame is None:
                return None
            arr = np.array(frame, copy=True)
            return arr

        if self._process is None or self._process.stdout is None:
            return None
        
        channels = 3  # rgb24 / bgr24
        frame_size = int(self.width) * int(self.height) * channels
        raw = self._read_exactly(frame_size)
        if raw is None:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((int(self.height), int(self.width), channels))
        return frame

    def start(self) -> None:
        """Start background reading; use read_latest() for lowest latency."""
        if not self._is_open:
            self.open()
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while self._running:
            frame = self.read()
            if frame is None:
                time.sleep(0.002)
                continue
            with self._cond:
                self._last_frame = frame
                self._last_frame_id += 1
                self._cond.notify_all()

    def read_latest(self, timeout: Optional[float] = None, copy: bool = False) -> Optional[np.ndarray]:
        """Return the newest frame (drops older frames)."""
        if not self._running:
            self.start()
        with self._cond:
            if self._last_frame is None and timeout is not None:
                self._cond.wait(timeout=float(timeout))
            frame = self._last_frame
        if frame is None:
            return None
        if copy:
            return np.array(frame, copy=True)
        return frame
    
    def frames(self, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        """iterate over frames from webcam.
        
        args:
            max_frames: maximum number of frames to capture (none for unlimited)
            
        yields:
            frames as numpy arrays
        """
        if not self._is_open:
            self.open()
        
        frame_count = 0
        while True:
            frame = self.read()
            if frame is None:
                break
            
            yield frame
            frame_count += 1
            
            if max_frames is not None and frame_count >= max_frames:
                break
    
    def release(self):
        """release webcam resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
        self._thread = None

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        if self._stderr_thread and self._stderr_thread.is_alive():
            try:
                self._stderr_thread.join(timeout=0.2)
            except Exception:
                pass
        self._stderr_thread = None

        if self._native is not None:
            try:
                self._native.release()
            except Exception:
                pass
            self._native = None

        self._is_open = False
    
    def is_opened(self) -> bool:
        """check if webcam is opened."""
        return self._is_open
    
    def __enter__(self):
        """context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """context manager exit."""
        self.release()
    
    def __del__(self):
        """destructor to ensure cleanup."""
        self.release()


__all__ = ["WebcamCapture"]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.