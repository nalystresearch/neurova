# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Kernel generation utilities for Neurova.

This module provides small, dependency-light kernel constructors that are used
by higher-level blur and edge detection operations.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from neurova.core.errors import ValidationError


IntOrPair = Union[int, Tuple[int, int]]


def _normalize_ksize(ksize: IntOrPair, *, name: str = "ksize") -> Tuple[int, int]:
    if isinstance(ksize, int):
        kh = kw = int(ksize)
    else:
        if len(ksize) != 2:
            raise ValidationError(name, ksize, "int or (kh, kw)")
        kh, kw = int(ksize[0]), int(ksize[1])

    if kh <= 0 or kw <= 0:
        raise ValidationError(name, ksize, "positive kernel size")
    if (kh % 2) == 0 or (kw % 2) == 0:
        raise ValidationError(name, ksize, "odd kernel size")
    return kh, kw


def box_kernel(ksize: IntOrPair, *, normalize: bool = True, dtype=np.float64) -> np.ndarray:
    """Create a 2D box (mean) kernel."""

    kh, kw = _normalize_ksize(ksize)
    k = np.ones((kh, kw), dtype=dtype)
    if normalize:
        k /= float(kh * kw)
    return k


def gaussian_kernel(
    ksize: IntOrPair,
    *,
    sigma: float | None = None,
    normalize: bool = True,
    dtype=np.float64,
) -> np.ndarray:
    """Create a 2D Gaussian kernel.

    Parameters
    ----------
    ksize:
        Kernel size as an int or (kh, kw). Must be odd.
    sigma:
        Standard deviation. If None, use a common heuristic based on kernel size.
    normalize:
        If True, normalize so the sum is 1.

    Returns
    -------
    np.ndarray
        2D kernel of shape (kh, kw).
    """

    kh, kw = _normalize_ksize(ksize)

    if sigma is None:
        # heuristic used in many CV libraries.
        sigma = 0.3 * ((max(kh, kw) - 1) * 0.5 - 1) + 0.8

    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValidationError("sigma", sigma, "> 0")

    y = np.arange(-(kh // 2), kh // 2 + 1, dtype=dtype)
    x = np.arange(-(kw // 2), kw // 2 + 1, dtype=dtype)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    g = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    if normalize:
        s = float(g.sum())
        if s != 0.0:
            g /= s
    return g


def sobel_kernels(dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Return the standard 3x3 Sobel kernels (Gx, Gy)."""

    gx = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=dtype,
    )
    gy = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=dtype,
    )
    return gx, gy


def scharr_kernels(dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Return the standard 3x3 Scharr kernels (Gx, Gy)."""

    gx = np.array(
        [
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3],
        ],
        dtype=dtype,
    )
    gy = np.array(
        [
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3],
        ],
        dtype=dtype,
    )
    return gx, gy


def laplacian_kernel(dtype=np.float64) -> np.ndarray:
    """Return a simple 3x3 Laplacian kernel."""

    return np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=dtype,
    )
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.