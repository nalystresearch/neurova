# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Basic image file writers for Neurova.

This module intentionally keeps dependencies minimal.

Supported formats (native):
- PNG (8-bit grayscale/RGB/RGBA)
- BMP (8-bit grayscale, 24-bit RGB)
- PPM/PGM (binary P6/P5)

Notes
-----
- JPEG encoding is not included in the native writer due to complexity.
- For additional formats, prefer adding optional plugins behind a stable
  Neurova interface so the core package remains lightweight.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional, Union

import numpy as np

from neurova.core.constants import ColorSpace
from neurova.core.errors import FileFormatError, IOError, ValidationError
from neurova.core.image import Image


ArrayLikeImage = Union[Image, np.ndarray]


def write_image(
    filepath: Union[str, Path],
    image: ArrayLikeImage,
    format: str = "auto",
    *,
    color_space: Optional[ColorSpace] = None,
) -> None:
    """Write an image to disk.

    Parameters
    ----------
    filepath:
        Output path.
    image:
        `neurova.core.image.Image` or a NumPy array shaped (H, W) or (H, W, C).
    format:
        'auto' (from extension) or explicit: 'png', 'bmp', 'ppm', 'pgm'.
    color_space:
        Optional override of color space when `image` is a NumPy array.

    Raises
    ------
    IOError
        If the file cannot be written.
    FileFormatError
        If the format is unsupported.
    ValidationError
        If the array shape is unsupported.
    """

    path = Path(filepath)

    if isinstance(image, Image):
        arr = image.data
        cs = image.color_space
    else:
        arr = np.asarray(image)
        if color_space is None:
            if arr.ndim == 2:
                cs = ColorSpace.GRAY
            elif arr.ndim == 3 and arr.shape[2] == 3:
                cs = ColorSpace.RGB
            elif arr.ndim == 3 and arr.shape[2] == 4:
                cs = ColorSpace.RGBA
            else:
                raise ValidationError("image", arr.shape, "(H,W), (H,W,3), or (H,W,4)")
        else:
            cs = color_space

    if format == "auto":
        ext = path.suffix.lower().lstrip(".")
        if ext == "":
            raise FileFormatError("Output format could not be inferred (missing file extension)")
        format = ext

    fmt = format.lower()
    try:
        if fmt == "png":
            _write_png(path, arr, cs)
        elif fmt == "bmp":
            _write_bmp(path, arr, cs)
        elif fmt in ("ppm", "pgm"):
            _write_ppm_pgm(path, arr, fmt)
        elif fmt in ("jpg", "jpeg"):
            _write_with_pillow(path, arr)
        else:
            # optional Pillow fallback for extended formats.
            _write_with_pillow(path, arr)
    except OSError as e:
        raise IOError(f"Failed to write file '{path}': {e}")
    except ImportError:
        raise FileFormatError(
            f"Unsupported output format: {fmt}. Install Pillow for extended format support: "
            "pip install neurova[io]"
        )


def imwrite(filepath: Union[str, Path], image: ArrayLikeImage) -> bool:
    """Neurova convenience wrapper.

    Returns True on success, False on failure.
    """

    try:
        write_image(filepath, image, format="auto")
        return True
    except Exception:
        return False


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    length = struct.pack(">I", len(chunk_data))
    crc = struct.pack(">I", _crc32(chunk_type + chunk_data))
    return length + chunk_type + chunk_data + crc


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """Convert common numeric dtypes to uint8 for file encoders."""

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        a = np.asarray(arr, dtype=np.float64)
        # heuristic: if in [0,1], scale. Otherwise clip to [0,255].
        if a.size > 0 and float(a.min()) >= 0.0 and float(a.max()) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0.0, 255.0)
        return a.round().astype(np.uint8)

    if np.issubdtype(arr.dtype, np.integer):
        a = np.asarray(arr, dtype=np.int64)
        a = np.clip(a, 0, 255)
        return a.astype(np.uint8)

    raise ValidationError("dtype", str(arr.dtype), "numeric dtype")


def _write_with_pillow(path: Path, arr: np.ndarray) -> None:
    """Write using Pillow if installed.

    This is used for formats not supported by native writers (e.g. JPEG).
    """

    from PIL import Image as PILImage

    a8 = _to_uint8_image(np.asarray(arr))
    img = PILImage.fromarray(a8)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _write_png(path: Path, arr: np.ndarray, color_space: ColorSpace) -> None:
    """Write a basic 8-bit PNG using filter type 0 for all scanlines."""

    if arr.ndim == 2:
        channels = 1
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        channels = int(arr.shape[2])
    else:
        raise ValidationError("image", arr.shape, "(H,W), (H,W,3), or (H,W,4)")

    height, width = int(arr.shape[0]), int(arr.shape[1])
    if height <= 0 or width <= 0:
        raise ValidationError("image", arr.shape, "non-empty image")

    a8 = _to_uint8_image(arr)

    if channels == 1:
        color_type = 0  # grayscale
        raw = a8.reshape(height, width)
        scanlines = b"".join(b"\x00" + raw[y].tobytes() for y in range(height))
    elif channels == 3:
        color_type = 2  # RGB
        raw = a8.reshape(height, width, 3)
        scanlines = b"".join(b"\x00" + raw[y].tobytes() for y in range(height))
    else:
        color_type = 6  # RGBA
        raw = a8.reshape(height, width, 4)
        scanlines = b"".join(b"\x00" + raw[y].tobytes() for y in range(height))

    compressed = zlib.compress(scanlines, level=6)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)

    png_bytes = (
        signature
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes)


def _write_bmp(path: Path, arr: np.ndarray, color_space: ColorSpace) -> None:
    """Write an uncompressed BMP (8-bit grayscale or 24-bit RGB)."""

    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValidationError("image", arr.shape, "(H,W) grayscale or (H,W,3) RGB")

    height, width = int(arr.shape[0]), int(arr.shape[1])
    if height <= 0 or width <= 0:
        raise ValidationError("image", arr.shape, "non-empty image")

    a8 = _to_uint8_image(arr)

    if mode == "RGB":
        # bMP is BGR, bottom-up rows with padding to 4-byte boundaries.
        row_stride = width * 3
        pad = (4 - (row_stride % 4)) % 4
        row_size = row_stride + pad
        pixel_bytes = bytearray()
        rgb = a8.reshape(height, width, 3)
        for y in range(height - 1, -1, -1):
            row = rgb[y][:, ::-1].tobytes()  # RGB -> BGR
            pixel_bytes.extend(row)
            if pad:
                pixel_bytes.extend(b"\x00" * pad)

        pixel_offset = 14 + 40
        file_size = pixel_offset + len(pixel_bytes)

        file_header = (
            b"BM"
            + struct.pack("<I", file_size)
            + struct.pack("<HH", 0, 0)
            + struct.pack("<I", pixel_offset)
        )

        dib_header = struct.pack(
            "<IIIHHIIIIII",
            40,  # header size
            width,
            height,
            1,  # planes
            24,  # bits
            0,  # compression
            len(pixel_bytes),
            2835,
            2835,
            0,
            0,
        )

        bmp_bytes = file_header + dib_header + bytes(pixel_bytes)

    else:
        # 8-bit grayscale with palette.
        row_stride = width
        pad = (4 - (row_stride % 4)) % 4
        row_size = row_stride + pad
        pixel_bytes = bytearray()
        gray = a8.reshape(height, width)
        for y in range(height - 1, -1, -1):
            pixel_bytes.extend(gray[y].tobytes())
            if pad:
                pixel_bytes.extend(b"\x00" * pad)

        palette = bytearray()
        for i in range(256):
            palette.extend(bytes((i, i, i, 0)))  # B, G, R, 0

        pixel_offset = 14 + 40 + len(palette)
        file_size = pixel_offset + len(pixel_bytes)

        file_header = (
            b"BM"
            + struct.pack("<I", file_size)
            + struct.pack("<HH", 0, 0)
            + struct.pack("<I", pixel_offset)
        )

        dib_header = struct.pack(
            "<IIIHHIIIIII",
            40,
            width,
            height,
            1,
            8,
            0,
            len(pixel_bytes),
            2835,
            2835,
            256,
            0,
        )

        bmp_bytes = file_header + dib_header + bytes(palette) + bytes(pixel_bytes)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bmp_bytes)


def _write_ppm_pgm(path: Path, arr: np.ndarray, fmt: str) -> None:
    """Write PPM (P6) or PGM (P5) binary formats."""

    a8 = _to_uint8_image(arr)

    if fmt == "pgm" or (a8.ndim == 2):
        if a8.ndim != 2:
            raise ValidationError("image", a8.shape, "(H,W) for PGM")
        height, width = int(a8.shape[0]), int(a8.shape[1])
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        payload = a8.tobytes()
    else:
        if a8.ndim != 3 or a8.shape[2] != 3:
            raise ValidationError("image", a8.shape, "(H,W,3) for PPM")
        height, width = int(a8.shape[0]), int(a8.shape[1])
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        payload = a8.reshape(height, width, 3).tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.