# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Neurova Video Capture Module
Native API for camera/video capture.
"""

import numpy as np
import subprocess
import platform
import threading
import time


class VideoCapture:
    """
    Video capture from camera or video file.
    
    Neurova native video capture implementation.
    
    Usage:
        cap = nv.VideoCapture(0)  # Open camera 0
        ret, frame = cap.read()    # Read frame
        cap.release()              # Release camera
    """
    
    def __init__(self, index=0, width=640, height=480, fps=30):
        """
        Initialize video capture.
        
        Args:
            index: Camera index (0 for default camera) or video file path
            width: Capture width (default: 640)
            height: Capture height (default: 480)
            fps: Frames per second (default: 30)
        """
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._process = None
        self._system = platform.system()
        self._running = False
        self._thread = None
        self._stderr_thread = None
        self._lock = threading.Lock()
        self._last_frame = None
        self._opened = False
        
    def isOpened(self):
        """Check if camera is opened successfully."""
        return self._opened and self._process is not None and self._process.poll() is None
    
    def open(self, index=None):
        """
        Open camera or video file.
        
        Args:
            index: Camera index or file path (uses constructor value if None)
            
        Returns:
            bool: True if opened successfully
        """
        if index is not None:
            self.index = index
            
        if self._opened:
            return True
            
        try:
            if self._system == "Darwin":  # macOS
                return self._open_avfoundation()
            elif self._system == "Linux":
                return self._open_v4l2()
            else:
                raise RuntimeError(f"Unsupported platform: {self._system}")
        except Exception as e:
            print(f"Failed to open camera: {e}")
            return False
    
    def _open_avfoundation(self):
        """Open camera using AVFoundation (macOS)."""
        pixel_formats = ["nv12", "uyvy422", "yuyv422", "0rgb", "bgr0"]
        
        for pix in pixel_formats:
            cmd = [
                "ffmpeg", "-f", "avfoundation",
                "-nostdin",
                "-loglevel", "error",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-pixel_format", pix,
                "-framerate", str(self.fps),
                "-video_size", f"{self.width}x{self.height}",
                "-i", str(self.index),
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-"
            ]
            
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            time.sleep(0.2)
            
            if self._process.poll() is None:
                # Start stderr drain thread
                self._stderr_thread = threading.Thread(
                    target=self._drain_stream,
                    args=(self._process.stderr,),
                    daemon=True
                )
                self._stderr_thread.start()
                
                # Start capture thread
                self._running = True
                self._thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._thread.start()
                
                # Wait for first frame
                time.sleep(0.5)
                
                self._opened = True
                return True
                
            self._process = None
        
        return False
    
    def _open_v4l2(self):
        """Open camera using V4L2 (Linux)."""
        cmd = [
            "ffmpeg", "-f", "v4l2",
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", f"/dev/video{self.index}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-"
        ]
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        
        time.sleep(0.5)
        
        self._opened = True
        return True
    
    def _drain_stream(self, stream):
        """Drain stderr to prevent blocking."""
        try:
            while stream is not None:
                chunk = stream.read(4096)
                if not chunk:
                    break
        except:
            pass
    
    def _read_exactly(self, nbytes):
        """Read exact number of bytes from pipe."""
        if self._process is None or self._process.stdout is None:
            return None
        
        chunks = []
        remaining = nbytes
        
        while remaining > 0:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        
        return b"".join(chunks)
    
    def _reader_loop(self):
        """Background thread to continuously read frames."""
        frame_size = self.width * self.height * 3
        
        while self._running and self._process is not None:
            raw_data = self._read_exactly(frame_size)
            if raw_data is None:
                break
            
            frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()
            
            with self._lock:
                self._last_frame = frame
        
        self._running = False
    
    def read(self):
        """
        Read next frame from camera.
        
        Returns:
            tuple: (ret, frame) where ret is True if frame was read successfully,
                   frame is numpy array (height, width, 3) in RGB format
        """
        if not self._opened:
            if not self.open():
                return False, None
        
        with self._lock:
            if self._last_frame is None:
                return False, None
            frame = self._last_frame.copy()
        
        return True, frame
    
    def release(self):
        """Release camera and cleanup resources."""
        self._running = False
        self._opened = False
        
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=0.5)
            except:
                try:
                    self._process.kill()
                except:
                    pass
            self._process = None
        
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.5)
            except:
                pass
        
        self._thread = None
        
        if self._stderr_thread and self._stderr_thread.is_alive():
            try:
                self._stderr_thread.join(timeout=0.2)
            except:
                pass
        
        self._stderr_thread = None
        
        with self._lock:
            self._last_frame = None
    
    def get(self, propId):
        """
        Get camera property (Neurova compatibility).
        
        Args:
            propId: Property ID (e.g., CAP_PROP_FRAME_WIDTH)
        
        Returns:
            Property value
        """
        # Map common Neurova properties
        if propId == 3:  # CAP_PROP_FRAME_WIDTH
            return self.width
        elif propId == 4:  # CAP_PROP_FRAME_HEIGHT
            return self.height
        elif propId == 5:  # CAP_PROP_FPS
            return self.fps
        return 0
    
    def set(self, propId, value):
        """
        Set camera property (Neurova compatibility).
        
        Args:
            propId: Property ID
            value: New value
            
        Returns:
            bool: True if successful
        """
        # Properties can only be set before opening
        if self._opened:
            return False
        
        if propId == 3:  # CAP_PROP_FRAME_WIDTH
            self.width = int(value)
            return True
        elif propId == 4:  # CAP_PROP_FRAME_HEIGHT
            self.height = int(value)
            return True
        elif propId == 5:  # CAP_PROP_FPS
            self.fps = int(value)
            return True
        
        return False
    
    def __enter__(self):
        """Context manager support."""
        if not self._opened:
            self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.release()
        return False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()


# Neurova compatibility constants
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.