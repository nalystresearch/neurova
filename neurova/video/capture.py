# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Video capture and frame extraction for Neurova."""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Iterator, Union
from neurova.core.errors import VideoError, ValidationError
from neurova.core.image import Image
from neurova import io as nv_io

try:
    from neurova.video import ffmpeg_backend as _ffmpeg_backend

    _HAS_LIBAV_BACKEND = True
except Exception:  # pragma: no cover - optional backend
    _ffmpeg_backend = None
    _HAS_LIBAV_BACKEND = False


class VideoCapture:
    """Video capture from file or image sequence.
    
    This class provides a unified interface for reading video frames from:
    - Image sequences (frame_%04d.png, etc.)
    - Video files via FFmpeg subprocess (optional, requires ffmpeg installed)
    
    Examples:
        # read from image sequence
        cap = VideoCapture("frames/frame_%04d.png", fps=30)
        for frame in cap:
            # process frame
            pass
        
        # read from video file
        cap = VideoCapture("video.mp4")
        frame = cap.read()
    """
    
    def __init__(
        self,
        source: Union[str, Path, list[str]],
        fps: float = 30.0,
        use_ffmpeg: bool = False,
    ):
        """Initialize video capture.
        
        Args:
            source: Video source - can be:
                - Path pattern for image sequence (e.g., "frame_%04d.png")
                - List of image paths
                - Video file path (requires use_ffmpeg=True)
            fps: Frames per second (for image sequences)
            use_ffmpeg: Use FFmpeg for video decoding (requires ffmpeg installed)
        """
        self.source = source
        self.fps = fps
        self.use_ffmpeg = use_ffmpeg
        self._frame_idx = 0
        self._frames: Optional[list[str]] = None
        self._total_frames = 0
        self._libav_reader = None
        
        # initialize based on source type
        if isinstance(source, list):
            self._frames = source
            self._total_frames = len(source)
            self._mode = "list"
        elif isinstance(source, (str, Path)):
            source_path = Path(source)
            if use_ffmpeg:
                if _HAS_LIBAV_BACKEND:
                    self._mode = "libav"
                    self._init_libav(source_path)
                else:
                    self._mode = "ffmpeg_cli"
                    self._init_ffmpeg_cli(source_path)
            else:
                self._mode = "pattern"
                self._init_pattern(source_path)
        else:
            raise ValidationError(
                "source",
                type(source).__name__,
                "str, Path, or list of str"
            )
    
    def _init_pattern(self, pattern: Path):
        """Initialize from image sequence pattern."""
        # try to find matching files
        parent = pattern.parent
        pattern_str = pattern.name
        
        # simple glob-based matching
        if '%' in pattern_str:
            # convert printf-style to glob pattern
            import re
            glob_pattern = re.sub(r'%\d*d', '*', pattern_str)
            if parent.exists():
                self._frames = sorted([str(p) for p in parent.glob(glob_pattern)])
                self._total_frames = len(self._frames)
            else:
                raise VideoError(f"Pattern parent directory does not exist: {parent}")
        else:
            # single file
            if pattern.exists():
                self._frames = [str(pattern)]
                self._total_frames = 1
            else:
                raise VideoError(f"Video file does not exist: {pattern}")
    
    def _init_ffmpeg_cli(self, video_path: Path):
        """Initialize FFmpeg-based video reading."""
        if not video_path.exists():
            raise VideoError(f"Video file does not exist: {video_path}")
        
        # check if ffmpeg is available
        import subprocess
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise VideoError(
                "FFmpeg not found. Install ffmpeg or use image sequences instead."
            )
        
        # get video info using ffprobe
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-count_packets",
                    "-show_entries", "stream=nb_read_packets,r_frame_rate",
                    "-of", "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout.strip().split(',')
            if len(output) >= 2:
                # parse frame rate
                fps_str = output[0]
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    self.fps = float(num) / float(den)
                else:
                    self.fps = float(fps_str)
                
                # parse frame count
                self._total_frames = int(output[1])
        except Exception as e:
            raise VideoError(f"Failed to probe video: {e}")
        
        self._video_path = video_path
        self._ffmpeg_process = None

    def _init_libav(self, video_path: Path):
        if not _HAS_LIBAV_BACKEND:
            raise VideoError("Libav backend not available")
        try:
            self._libav_reader = _ffmpeg_backend.FFmpegReader(str(video_path))
            if hasattr(self._libav_reader, "frame_count"):
                self._total_frames = int(getattr(self._libav_reader, "frame_count"))
            if hasattr(self._libav_reader, "fps"):
                self.fps = float(getattr(self._libav_reader, "fps"))
        except Exception as exc:  # pragma: no cover - depends on libav
            raise VideoError(f"Failed to initialize libav backend: {exc}")
    
    def read(self) -> Optional[np.ndarray]:
        """Read next frame.
        
        Returns:
            Frame as numpy array (HxWx3 RGB), or None if no more frames
        """
        if self._mode == "libav":
            return self._read_libav()
        if self._mode == "ffmpeg_cli":
            return self._read_ffmpeg()
        else:
            return self._read_sequence()
    
    def _read_sequence(self) -> Optional[np.ndarray]:
        """Read frame from image sequence."""
        if self._frames is None or self._frame_idx >= len(self._frames):
            return None
        
        frame_path = self._frames[self._frame_idx]
        self._frame_idx += 1
        
        try:
            img = nv_io.read_image(frame_path)
            return img.as_array()
        except Exception as e:
            raise VideoError(f"Failed to read frame {frame_path}: {e}")
    
    def _read_ffmpeg(self) -> Optional[np.ndarray]:
        """Read frame using FFmpeg."""
        import subprocess
        
        if self._ffmpeg_process is None:
            # start FFmpeg process
            self._ffmpeg_process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-i", str(self._video_path),
                    "-f", "image2pipe",
                    "-pix_fmt", "rgb24",
                    "-vcodec", "rawvideo",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            # get frame dimensions (would need separate probe)
            # for now, assume we know dimensions or read first frame to detect
            self._frame_width = None
            self._frame_height = None
        
        # this is a simplified implementation
        # real implementation would need proper frame size detection
        raise NotImplementedError(
            "FFmpeg video reading not fully implemented. Use image sequences instead."
        )

    def _read_libav(self) -> Optional[np.ndarray]:
        if self._libav_reader is None:
            return None
        frame = self._libav_reader.read()
        if frame is None:
            return None
        arr = np.array(frame, copy=True)
        return arr
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame
    
    def __len__(self) -> int:
        """Total number of frames."""
        return self._total_frames
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def release(self):
        """Release resources."""
        if self._ffmpeg_process is not None:
            self._ffmpeg_process.terminate()
            self._ffmpeg_process.wait()
            self._ffmpeg_process = None
        if self._libav_reader is not None:
            try:
                self._libav_reader.close()
            except Exception:
                pass
            self._libav_reader = None
    
    def get(self, prop: str):
        """Get video property.
        
        Args:
            prop: Property name ("fps", "frame_count", "frame_width", "frame_height")
            
        Returns:
            Property value
        """
        if prop == "fps":
            return self.fps
        elif prop == "frame_count":
            return self._total_frames
        elif prop in ("frame_width", "frame_height"):
            # would need to read first frame to detect
            raise NotImplementedError(f"Property {prop} not implemented")
        else:
            raise ValidationError("prop", prop, "valid property name")


class VideoWriter:
    """Write video from frames.
    
    This class provides a simple interface for writing frames to:
    - Image sequences
    - Video files via FFmpeg (optional)
    """
    
    def __init__(
        self,
        output: Union[str, Path],
        fps: float = 30.0,
        fourcc: Optional[str] = None,
    ):
        """Initialize video writer.
        
        Args:
            output: Output path - can be:
                - Path pattern for image sequence (e.g., "out/frame_%04d.png")
                - Video file path (requires FFmpeg)
            fps: Frames per second
            fourcc: Video codec fourcc code (e.g., "mp4v", "h264")
        """
        self.output = Path(output)
        self.fps = fps
        self.fourcc = fourcc
        self._frame_idx = 0
        
        # determine mode
        if fourcc is not None or str(output).endswith(('.mp4', '.avi', '.mov')):
            self._mode = "ffmpeg"
            raise NotImplementedError(
                "FFmpeg video writing not implemented. Use image sequences instead."
            )
        else:
            self._mode = "sequence"
            # create output directory
            self.output.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, frame: np.ndarray):
        """Write a frame.
        
        Args:
            frame: Frame as numpy array (HxWx3 RGB or HxW grayscale)
        """
        if self._mode == "sequence":
            # write as image file
            if '%' in self.output.name:
                # use pattern
                frame_path = self.output.parent / (self.output.stem % self._frame_idx + self.output.suffix)
            else:
                # append frame number
                frame_path = self.output.parent / f"{self.output.stem}_{self._frame_idx:04d}{self.output.suffix}"
            
            nv_io.write_image(str(frame_path), frame)
            self._frame_idx += 1
    
    def release(self):
        """Release resources."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


__all__ = [
    "VideoCapture",
    "VideoWriter",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.