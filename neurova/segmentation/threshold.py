# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Thresholding operations for Neurova."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from neurova.core.constants import ThresholdMethod
from neurova.core.errors import ValidationError


def threshold(
    image: np.ndarray,
    thresh: float,
    *,
    max_value: float = 255.0,
    method: ThresholdMethod = ThresholdMethod.BINARY,
) -> Tuple[float, np.ndarray]:
    """Apply a threshold.

    Parameters
    ----------
    image:
        2D grayscale array.
    thresh:
        Threshold value.
    max_value:
        Output value used for "on" pixels in binary modes.
    method:
        Threshold method.

    Returns
    -------
    (used_thresh, out)
        used_thresh is the threshold actually applied (useful for OTSU).
    """

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValidationError("image", img.shape, "2D array")

    t = float(thresh)
    mv = float(max_value)

    if method == ThresholdMethod.OTSU:
        t = otsu_threshold(img)
        method = ThresholdMethod.BINARY

    if method == ThresholdMethod.BINARY:
        out = np.where(img > t, mv, 0.0)
    elif method == ThresholdMethod.BINARY_INV:
        out = np.where(img > t, 0.0, mv)
    elif method == ThresholdMethod.TRUNCATE:
        out = np.minimum(img.astype(np.float64), t)
    elif method == ThresholdMethod.TO_ZERO:
        out = np.where(img > t, img.astype(np.float64), 0.0)
    elif method == ThresholdMethod.TO_ZERO_INV:
        out = np.where(img > t, 0.0, img.astype(np.float64))
    else:
        raise ValidationError("method", method, "supported ThresholdMethod")

    return t, _cast_back(out, img.dtype)


def otsu_threshold(image: np.ndarray) -> float:
    """Compute Otsu's threshold for an 8-bit grayscale image."""

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValidationError("image", img.shape, "2D array")

    if not np.issubdtype(img.dtype, np.integer):
        # scale floats into [0, 255] if needed.
        img = img.astype(np.float64)
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    else:
        if img.min() < 0 or img.max() > 255:
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8, copy=False)

    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0.0:
        return 0.0

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    # between-class variance.
    denom = omega * (1.0 - omega)
    denom[denom == 0.0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    t = int(np.nanargmax(sigma_b2))
    return float(t)


def _cast_back(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype, copy=False)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.clip(arr, info.min, info.max)
        return np.rint(arr).astype(dtype)

    return arr.astype(dtype)
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.