# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Warping operations for Neurova.

This module provides affine warping for 2D images (grayscale or multi-channel).
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from neurova.core.constants import InterpolationMode
from neurova.core.errors import ValidationError


BorderModeStr = Literal["constant", "reflect", "replicate", "wrap"]


def warp_affine(
    image: np.ndarray,
    matrix: np.ndarray,
    dsize: Tuple[int, int],
    *,
    interpolation: InterpolationMode = InterpolationMode.LINEAR,
    border_mode: BorderModeStr = "constant",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Apply a 2x3 affine transform.

    Parameters
    ----------
    image:
        Input array shaped (H, W) or (H, W, C).
    matrix:
        2x3 affine matrix mapping input->output (same convention as Neurova).
    dsize:
        (out_w, out_h).
    interpolation:
        NEAREST or LINEAR.
    border_mode:
        constant, reflect, replicate, wrap.

    Returns
    -------
    np.ndarray
        Warped image shaped (out_h, out_w) or (out_h, out_w, C).
    """

    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValidationError("image", img.shape, "2D or 3D array")

    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3):
        raise ValidationError("matrix", m.shape, "shape (2,3)")

    out_w, out_h = int(dsize[0]), int(dsize[1])
    if out_w <= 0 or out_h <= 0:
        raise ValidationError("dsize", dsize, "positive (out_w, out_h)")

    # build inverse mapping: for each output pixel, compute source coordinate.
    m33 = np.vstack([m, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(m33)[0:2, :]

    yy, xx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    ones = np.ones_like(xx, dtype=np.float64)

    src_x = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2] * ones
    src_y = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2] * ones

    if interpolation == InterpolationMode.NEAREST:
        out = _sample_nearest(img, src_x, src_y, border_mode=border_mode, constant_value=constant_value)
    elif interpolation == InterpolationMode.LINEAR:
        out = _sample_bilinear(img, src_x, src_y, border_mode=border_mode, constant_value=constant_value)
    else:
        raise ValidationError("interpolation", interpolation, "InterpolationMode.NEAREST or InterpolationMode.LINEAR")

    return _cast_like(out, img.dtype)


def _reflect_indices(i: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(i)
    period = 2 * (size - 1)
    x = np.mod(i, period)
    return np.where(x <= (size - 1), x, period - x)


def _wrap_indices(i: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return np.zeros_like(i)
    return np.mod(i, size)


def _map_coords(
    x: np.ndarray,
    y: np.ndarray,
    h: int,
    w: int,
    *,
    border_mode: BorderModeStr,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map possibly out-of-bounds coords into valid coords.

    Returns (mx, my, in_bounds_mask).
    """

    if border_mode == "replicate":
        mx = np.clip(x, 0, w - 1)
        my = np.clip(y, 0, h - 1)
        mask = np.ones_like(mx, dtype=bool)
        return mx, my, mask

    if border_mode == "reflect":
        mx = _reflect_indices(x, w)
        my = _reflect_indices(y, h)
        mask = np.ones_like(mx, dtype=bool)
        return mx, my, mask

    if border_mode == "wrap":
        mx = _wrap_indices(x, w)
        my = _wrap_indices(y, h)
        mask = np.ones_like(mx, dtype=bool)
        return mx, my, mask

    # constant
    mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    mx = np.clip(x, 0, w - 1)
    my = np.clip(y, 0, h - 1)
    return mx, my, mask


def _sample_nearest(
    img: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    *,
    border_mode: BorderModeStr,
    constant_value: float,
) -> np.ndarray:
    h, w = int(img.shape[0]), int(img.shape[1])

    xi = np.rint(src_x).astype(np.int64)
    yi = np.rint(src_y).astype(np.int64)

    mx, my, mask = _map_coords(xi, yi, h, w, border_mode=border_mode)

    if img.ndim == 2:
        out = img[my, mx].astype(np.float64)
        if border_mode == "constant":
            out = np.where(mask, out, float(constant_value))
        return out

    out = img[my, mx, :].astype(np.float64)
    if border_mode == "constant":
        out[~mask] = float(constant_value)
    return out


def _sample_bilinear(
    img: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    *,
    border_mode: BorderModeStr,
    constant_value: float,
) -> np.ndarray:
    h, w = int(img.shape[0]), int(img.shape[1])

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = src_x - x0
    wy = src_y - y0

    x0m, y0m, m00 = _map_coords(x0, y0, h, w, border_mode=border_mode)
    x1m, y0m2, m10 = _map_coords(x1, y0, h, w, border_mode=border_mode)
    x0m2, y1m, m01 = _map_coords(x0, y1, h, w, border_mode=border_mode)
    x1m2, y1m2, m11 = _map_coords(x1, y1, h, w, border_mode=border_mode)

    # sanity: y maps should match for same y inputs.
    y0m = y0m  # keep naming consistent
    y1m = y1m

    if img.ndim == 2:
        i00 = img[y0m, x0m].astype(np.float64)
        i10 = img[y0m2, x1m].astype(np.float64)
        i01 = img[y1m, x0m2].astype(np.float64)
        i11 = img[y1m2, x1m2].astype(np.float64)

        if border_mode == "constant":
            cv = float(constant_value)
            i00 = np.where(m00, i00, cv)
            i10 = np.where(m10, i10, cv)
            i01 = np.where(m01, i01, cv)
            i11 = np.where(m11, i11, cv)

        a = (1.0 - wx) * (1.0 - wy)
        b = wx * (1.0 - wy)
        c = (1.0 - wx) * wy
        d = wx * wy
        return a * i00 + b * i10 + c * i01 + d * i11

    i00 = img[y0m, x0m, :].astype(np.float64)
    i10 = img[y0m2, x1m, :].astype(np.float64)
    i01 = img[y1m, x0m2, :].astype(np.float64)
    i11 = img[y1m2, x1m2, :].astype(np.float64)

    if border_mode == "constant":
        cv = float(constant_value)
        i00[~m00] = cv
        i10[~m10] = cv
        i01[~m01] = cv
        i11[~m11] = cv

    a = ((1.0 - wx) * (1.0 - wy))[:, :, None]
    b = (wx * (1.0 - wy))[:, :, None]
    c = ((1.0 - wx) * wy)[:, :, None]
    d = (wx * wy)[:, :, None]

    return a * i00 + b * i10 + c * i01 + d * i11


def _cast_like(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype, copy=False)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        clipped = np.clip(arr, info.min, info.max)
        return np.rint(clipped).astype(dtype)

    return arr.astype(dtype)
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.