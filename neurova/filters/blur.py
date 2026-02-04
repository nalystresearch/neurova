# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Blur filters for Neurova."""

from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np

from neurova.core.array_ops import pad_array
from neurova.core.errors import ValidationError
from neurova.filters.convolution import convolve2d
from neurova.filters.kernels import box_kernel, gaussian_kernel


BorderModeStr = Literal["constant", "reflect", "replicate", "wrap"]
IntOrPair = Union[int, Tuple[int, int]]


def box_blur(
    image: np.ndarray,
    ksize: IntOrPair = 3,
    *,
    kernel_size: IntOrPair | None = None,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Apply a box blur (mean filter)."""

    if kernel_size is not None:
        ksize = kernel_size


    k = box_kernel(ksize, normalize=True)
    return convolve2d(image, k, border_mode=border_mode, constant_value=constant_value)


def gaussian_blur(
    image: np.ndarray,
    ksize: IntOrPair = 5,
    *,
    kernel_size: IntOrPair | None = None,
    sigma: float | None = None,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Apply a Gaussian blur."""

    if kernel_size is not None:
        ksize = kernel_size


    k = gaussian_kernel(ksize, sigma=sigma, normalize=True)
    return convolve2d(image, k, border_mode=border_mode, constant_value=constant_value)


def sharpen(
    image: np.ndarray,
    *,
    sigma: float = 1.0,
    ksize: IntOrPair = 5,
    amount: float = 1.0,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Sharpen an image using a simple unsharp mask.

    Parameters
    ----------
    sigma:
        Gaussian sigma used for the blur.
    ksize:
        Kernel size for the blur.
    amount:
        Sharpening strength.
    """

    img = np.asarray(image)
    blurred = gaussian_blur(
        img.astype(np.float64),
        ksize,
        sigma=sigma,
        border_mode=border_mode,
        constant_value=constant_value,
    )

    out = img.astype(np.float64) + float(amount) * (img.astype(np.float64) - blurred)

    # cast back like convolution does.
    if np.issubdtype(img.dtype, np.floating):
        return out.astype(img.dtype)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(out, info.min, info.max)
        return np.rint(out).astype(img.dtype)
    return out.astype(img.dtype)


def median_blur(
    image: np.ndarray,
    ksize: int = 3,
    *,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Apply a median filter.

    Notes
    -----
    - This implementation is NumPy-based and intended for small kernels.
    - For multi-channel images, each channel is filtered independently.
    """

    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValidationError("image", img.shape, "2D or 3D array")

    k = int(ksize)
    if k <= 0 or (k % 2) == 0:
        raise ValidationError("ksize", ksize, "positive odd int")


    pad = k // 2

    if img.ndim == 2:
        padded = pad_array(img, ((pad, pad), (pad, pad)), mode=border_mode, constant_value=constant_value)
        out = _median2d_single_channel(padded, k)
        return out.astype(img.dtype, copy=False)

    padded = pad_array(img, ((pad, pad), (pad, pad), (0, 0)), mode=border_mode, constant_value=constant_value)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _median2d_single_channel(padded[:, :, c], k).astype(img.dtype, copy=False)
    return out


def _median2d_single_channel(padded: np.ndarray, k: int) -> np.ndarray:
    ph, pw = padded.shape
    out_h = ph - k + 1
    out_w = pw - k + 1

    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(out_h, out_w, k, k),
        strides=(s0, s1, s0, s1),
        writeable=False,
    )

    # median over last two axes.
    return np.median(windows, axis=(2, 3))
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.