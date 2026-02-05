# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Edge detection operations for Neurova."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from neurova.core.constants import ColorSpace
from neurova.core.color import to_grayscale
from neurova.core.errors import ValidationError
from neurova.filters.blur import gaussian_blur
from neurova.filters.convolution import convolve2d
from neurova.filters.kernels import laplacian_kernel, scharr_kernels, sobel_kernels


BorderModeStr = Literal["constant", "reflect", "replicate", "wrap"]


def sobel(
    image: np.ndarray,
    *,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
    from_space: ColorSpace = ColorSpace.RGB,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Sobel gradients (Gx, Gy)."""

    img = np.asarray(image)
    if img.ndim == 3:
        img = to_grayscale(img, from_space=from_space)

    if img.ndim != 2:
        raise ValidationError("image", getattr(image, "shape", None), "2D or 3D array")

    gx_k, gy_k = sobel_kernels(dtype=np.float64)
    gx = convolve2d(img, gx_k, border_mode=border_mode, constant_value=constant_value).astype(np.float64)
    gy = convolve2d(img, gy_k, border_mode=border_mode, constant_value=constant_value).astype(np.float64)
    return gx, gy


def scharr(
    image: np.ndarray,
    *,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
    from_space: ColorSpace = ColorSpace.RGB,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Scharr gradients (Gx, Gy)."""

    img = np.asarray(image)
    if img.ndim == 3:
        img = to_grayscale(img, from_space=from_space)

    if img.ndim != 2:
        raise ValidationError("image", getattr(image, "shape", None), "2D or 3D array")

    gx_k, gy_k = scharr_kernels(dtype=np.float64)
    gx = convolve2d(img, gx_k, border_mode=border_mode, constant_value=constant_value).astype(np.float64)
    gy = convolve2d(img, gy_k, border_mode=border_mode, constant_value=constant_value).astype(np.float64)
    return gx, gy


def laplacian(
    image: np.ndarray,
    *,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
    from_space: ColorSpace = ColorSpace.RGB,
) -> np.ndarray:
    """Compute the Laplacian response."""

    img = np.asarray(image)
    if img.ndim == 3:
        img = to_grayscale(img, from_space=from_space)

    if img.ndim != 2:
        raise ValidationError("image", getattr(image, "shape", None), "2D or 3D array")

    k = laplacian_kernel(dtype=np.float64)
    return convolve2d(img, k, border_mode=border_mode, constant_value=constant_value).astype(np.float64)


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude from gradients."""

    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    return np.hypot(gx, gy)


def canny(
    image: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
    sigma: float = 1.0,
    blur_ksize: int = 5,
    from_space: ColorSpace = ColorSpace.RGB,
    weak_value: int = 75,
    strong_value: int = 255,
) -> np.ndarray:
    """Canny edge detector.

    Returns
    -------
    np.ndarray
        2D uint8 image with edges as 255 and background as 0.
    """

    if high_threshold < low_threshold:
        raise ValidationError("high_threshold", high_threshold, ">= low_threshold")

    img = np.asarray(image)
    if img.ndim == 3:
        img = to_grayscale(img, from_space=from_space)

    if img.ndim != 2:
        raise ValidationError("image", getattr(image, "shape", None), "2D or 3D array")

    img_f = img.astype(np.float64)
    blurred = gaussian_blur(img_f, blur_ksize, sigma=sigma)

    gx, gy = sobel(blurred)
    mag = gradient_magnitude(gx, gy)
    ang = np.arctan2(gy, gx)

    nms = _non_max_suppression(mag, ang)
    dt = _double_threshold(nms, low_threshold, high_threshold, weak_value=weak_value, strong_value=strong_value)
    edges = _hysteresis(dt, weak_value=weak_value, strong_value=strong_value)
    return edges.astype(np.uint8, copy=False)


def canny_edges(
    image: np.ndarray,
    *,
    low: float,
    high: float,
    sigma: float = 1.0,
    blur_ksize: int = 5,
    from_space: ColorSpace = ColorSpace.RGB,
) -> np.ndarray:
    """Compatibility wrapper for README-style usage."""

    return canny(
        image,
        low_threshold=low,
        high_threshold=high,
        sigma=sigma,
        blur_ksize=blur_ksize,
        from_space=from_space,
    )


def _non_max_suppression(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    h, w = mag.shape

    angle = (np.rad2deg(ang) + 180.0) % 180.0
    out = np.zeros((h, w), dtype=np.float64)

    # quantize directions to 0, 45, 90, 135.
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            m = mag[i, j]

            if (0.0 <= a < 22.5) or (157.5 <= a < 180.0):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif 22.5 <= a < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            else:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if m >= q and m >= r:
                out[i, j] = m

    return out


def _double_threshold(
    img: np.ndarray,
    low: float,
    high: float,
    *,
    weak_value: int,
    strong_value: int,
) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.uint8)

    strong = img >= high
    weak = (img >= low) & ~strong

    out[strong] = np.uint8(strong_value)
    out[weak] = np.uint8(weak_value)
    return out


def _hysteresis(img: np.ndarray, *, weak_value: int, strong_value: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape

    weak = np.uint8(weak_value)
    strong = np.uint8(strong_value)

    changed = True
    while changed:
        changed = False
        # iterate interior; edges are left as-is.
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if out[i, j] != weak:
                    continue

                patch = out[i - 1 : i + 2, j - 1 : j + 2]
                if np.any(patch == strong):
                    out[i, j] = strong
                    changed = True

    out[out != strong] = 0
    return out
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.