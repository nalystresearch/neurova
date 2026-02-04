# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Convolution operations for Neurova.

The functions here are NumPy-based and designed as building blocks for blur,
edge detection, and other filtering operations.

Public API
----------
- convolve2d(image, kernel, border_mode='reflect', constant_value=0.0)
- filter2d(image, kernel, border_mode='reflect', constant_value=0.0)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from neurova.core.array_ops import pad_array
from neurova.core.errors import ValidationError


BorderModeStr = Literal["constant", "reflect", "replicate", "wrap"]


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Convolve a 2D kernel over a 2D/3D image.

    Parameters
    ----------
    image:
        Input array shaped (H, W) or (H, W, C).
    kernel:
        2D kernel shaped (Kh, Kw). Must have odd spatial dimensions.
    border_mode:
        Padding mode: 'constant', 'reflect', 'replicate', or 'wrap'.
    constant_value:
        Constant used when border_mode='constant'.

    Returns
    -------
    np.ndarray
        Output array with same shape as input.
    """

    img = np.asarray(image)
    ker = np.asarray(kernel)

    if img.ndim not in (2, 3):
        raise ValidationError("image", img.shape, "2D or 3D array")

    if ker.ndim != 2:
        raise ValidationError("kernel", ker.shape, "2D array")

    kh, kw = int(ker.shape[0]), int(ker.shape[1])
    if kh <= 0 or kw <= 0:
        raise ValidationError("kernel", ker.shape, "non-empty kernel")

    if (kh % 2) == 0 or (kw % 2) == 0:
        raise ValidationError("kernel", ker.shape, "odd kernel dimensions")

    pad_h = kh // 2
    pad_w = kw // 2

    if img.ndim == 2:
        padded = pad_array(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=border_mode, constant_value=constant_value)
        out = _convolve2d_single_channel(padded, ker)
        return _cast_like(out, img.dtype)

    # 3D: convolve each channel.
    padded = pad_array(
        img,
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode=border_mode,
        constant_value=constant_value,
    )

    out = _convolve2d_multi_channel(padded, ker)
    return _cast_like(out, img.dtype)


def filter2d(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    border_mode: BorderModeStr = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Alias of convolve2d for familiarity."""

    return convolve2d(image, kernel, border_mode=border_mode, constant_value=constant_value)


def _convolve2d_single_channel(padded: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ph, pw = padded.shape
    kh, kw = kernel.shape
    out_h = ph - kh + 1
    out_w = pw - kw + 1

    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(out_h, out_w, kh, kw),
        strides=(s0, s1, s0, s1),
        writeable=False,
    )

    ker = kernel.astype(np.float64, copy=False)
    return np.einsum("ijmn,mn->ij", windows.astype(np.float64, copy=False), ker, optimize=True)


def _convolve2d_multi_channel(padded: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ph, pw, c = padded.shape
    kh, kw = kernel.shape
    out_h = ph - kh + 1
    out_w = pw - kw + 1

    s0, s1, s2 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(out_h, out_w, kh, kw, c),
        strides=(s0, s1, s0, s1, s2),
        writeable=False,
    )

    ker = kernel.astype(np.float64, copy=False)
    # multiply over kh,kw and sum.
    out = np.einsum("ijmnc,mn->ijc", windows.astype(np.float64, copy=False), ker, optimize=True)
    return out


def _cast_like(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype, copy=False)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        clipped = np.clip(arr, info.min, info.max)
        return np.rint(clipped).astype(dtype)

    return arr.astype(dtype)
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.