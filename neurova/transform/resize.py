# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Image resizing for Neurova.

This module implements common resampling methods with NumPy.

API
---
- resize(image, out_shape, interpolation=InterpolationMode.LINEAR)

Notes
-----
- `out_shape` is (out_height, out_width) to match NumPy array conventions.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from neurova.core.constants import InterpolationMode
from neurova.core.errors import ValidationError


def resize(
    image: np.ndarray,
    out_shape: Tuple[int, int],
    interpolation: InterpolationMode = InterpolationMode.LINEAR,
) -> np.ndarray:
    """Resize an image array.

    Parameters
    ----------
    image:
        Input array shaped (H, W) or (H, W, C).
    out_shape:
        (out_height, out_width).
    interpolation:
        Interpolation method.

    Returns
    -------
    np.ndarray
        Resized array.
    """

    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValidationError("image", img.shape, "2D or 3D array")

    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    if out_h <= 0 or out_w <= 0:
        raise ValidationError("out_shape", out_shape, "positive (height, width)")

    in_h, in_w = int(img.shape[0]), int(img.shape[1])
    if in_h == out_h and in_w == out_w:
        return img.copy()

    if interpolation == InterpolationMode.NEAREST:
        return _resize_nearest(img, out_h, out_w)

    if interpolation == InterpolationMode.LINEAR:
        return _resize_bilinear(img, out_h, out_w)

    raise ValidationError(
        "interpolation",
        int(interpolation),
        "InterpolationMode.NEAREST or InterpolationMode.LINEAR",
    )


def _resize_nearest(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = img.shape[0], img.shape[1]

    # map output coordinates to input coordinates.
    y = (np.arange(out_h) + 0.5) * (in_h / out_h) - 0.5
    x = (np.arange(out_w) + 0.5) * (in_w / out_w) - 0.5

    yi = np.clip(np.rint(y).astype(np.int64), 0, in_h - 1)
    xi = np.clip(np.rint(x).astype(np.int64), 0, in_w - 1)

    if img.ndim == 2:
        return img[yi[:, None], xi[None, :]].copy()
    return img[yi[:, None], xi[None, :], :].copy()


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = img.shape[0], img.shape[1]

    # floating point coordinate transform (pixel center alignment).
    y = (np.arange(out_h) + 0.5) * (in_h / out_h) - 0.5
    x = (np.arange(out_w) + 0.5) * (in_w / out_w) - 0.5

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    y0c = np.clip(y0, 0, in_h - 1)
    y1c = np.clip(y1, 0, in_h - 1)
    x0c = np.clip(x0, 0, in_w - 1)
    x1c = np.clip(x1, 0, in_w - 1)

    wy = (y - y0).astype(np.float64)
    wx = (x - x0).astype(np.float64)

    if img.ndim == 2:
        Ia = img[y0c[:, None], x0c[None, :]].astype(np.float64)
        Ib = img[y0c[:, None], x1c[None, :]].astype(np.float64)
        Ic = img[y1c[:, None], x0c[None, :]].astype(np.float64)
        Id = img[y1c[:, None], x1c[None, :]].astype(np.float64)

        wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
        wb = (1.0 - wy)[:, None] * wx[None, :]
        wc = wy[:, None] * (1.0 - wx)[None, :]
        wd = wy[:, None] * wx[None, :]

        out = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return _cast_like(out, img.dtype)

    # color image
    Ia = img[y0c[:, None], x0c[None, :], :].astype(np.float64)
    Ib = img[y0c[:, None], x1c[None, :], :].astype(np.float64)
    Ic = img[y1c[:, None], x0c[None, :], :].astype(np.float64)
    Id = img[y1c[:, None], x1c[None, :], :].astype(np.float64)

    wa = ((1.0 - wy)[:, None] * (1.0 - wx)[None, :])[:, :, None]
    wb = ((1.0 - wy)[:, None] * wx[None, :])[:, :, None]
    wc = (wy[:, None] * (1.0 - wx)[None, :])[:, :, None]
    wd = (wy[:, None] * wx[None, :])[:, :, None]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return _cast_like(out, img.dtype)


def _cast_like(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float array back to the input dtype with safe clipping."""

    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype, copy=False)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        clipped = np.clip(arr, info.min, info.max)
        return np.rint(clipped).astype(dtype)

    # fallback
    return arr.astype(dtype)
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.