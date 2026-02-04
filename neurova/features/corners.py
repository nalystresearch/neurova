# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Corner detection for Neurova."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from neurova.core.constants import ColorSpace
from neurova.core.color import to_grayscale
from neurova.core.errors import ValidationError
from neurova.core.image import Image
from neurova.filters.blur import box_blur, gaussian_blur
from neurova.filters.edges import sobel


CornerMethod = Literal["harris", "shi_tomasi"]


def detect_corners(
    image: Image | np.ndarray,
    *,
    method: CornerMethod = "harris",
    max_corners: Optional[int] = 200,
    quality_level: float = 0.01,
    min_distance: int = 5,
    block_size: int = 3,
    k: float = 0.04,
    use_gaussian: bool = False,
    from_space: ColorSpace = ColorSpace.RGB,
) -> np.ndarray:
    """Detect corners in an image.

    Returns
    -------
    np.ndarray
        Array of corner coordinates shaped (N, 2) as (x, y).
    """

    if isinstance(image, Image):
        arr = image.data
        from_space = image.color_space
    else:
        arr = np.asarray(image)

    if arr.ndim == 3:
        arr = to_grayscale(arr, from_space=from_space)

    if arr.ndim != 2:
        raise ValidationError("image", arr.shape, "2D grayscale or 3D color")

    arr_f = arr.astype(np.float64)

    if method == "harris":
        resp = harris_response(arr_f, block_size=block_size, k=k, use_gaussian=use_gaussian)
    elif method == "shi_tomasi":
        resp = shi_tomasi_response(arr_f, block_size=block_size, use_gaussian=use_gaussian)
    else:
        raise ValidationError("method", method, "harris or shi_tomasi")

    coords = _pick_keypoints_from_response(
        resp,
        max_points=max_corners,
        quality_level=quality_level,
        min_distance=min_distance,
    )

    # coords are (y,x) -> return (x,y)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return coords[:, ::-1].astype(np.float64)


def harris_response(
    gray: np.ndarray,
    *,
    block_size: int = 3,
    k: float = 0.04,
    use_gaussian: bool = False,
) -> np.ndarray:
    """Compute Harris corner response image."""

    if gray.ndim != 2:
        raise ValidationError("gray", gray.shape, "2D array")

    if block_size <= 0 or (block_size % 2) == 0:
        raise ValidationError("block_size", block_size, "positive odd")

    gx, gy = sobel(gray)
    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    if use_gaussian:
        sxx = gaussian_blur(ixx, block_size, sigma=1.0)
        syy = gaussian_blur(iyy, block_size, sigma=1.0)
        sxy = gaussian_blur(ixy, block_size, sigma=1.0)
    else:
        sxx = box_blur(ixx, block_size)
        syy = box_blur(iyy, block_size)
        sxy = box_blur(ixy, block_size)

    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    r = det - float(k) * (trace * trace)
    return r


def shi_tomasi_response(
    gray: np.ndarray,
    *,
    block_size: int = 3,
    use_gaussian: bool = False,
) -> np.ndarray:
    """Compute Shi-Tomasi (min eigenvalue) response image."""

    if gray.ndim != 2:
        raise ValidationError("gray", gray.shape, "2D array")

    if block_size <= 0 or (block_size % 2) == 0:
        raise ValidationError("block_size", block_size, "positive odd")

    gx, gy = sobel(gray)
    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    if use_gaussian:
        sxx = gaussian_blur(ixx, block_size, sigma=1.0)
        syy = gaussian_blur(iyy, block_size, sigma=1.0)
        sxy = gaussian_blur(ixy, block_size, sigma=1.0)
    else:
        sxx = box_blur(ixx, block_size)
        syy = box_blur(iyy, block_size)
        sxy = box_blur(ixy, block_size)

    # eigenvalues of [[a,b],[b,c]]
    a, b, c = sxx, sxy, syy
    t = a + c
    d = np.maximum((a - c) * (a - c) + 4.0 * b * b, 0.0)
    sqrt_d = np.sqrt(d)
    lam1 = 0.5 * (t + sqrt_d)
    lam2 = 0.5 * (t - sqrt_d)
    return np.minimum(lam1, lam2)


def _pick_keypoints_from_response(
    resp: np.ndarray,
    *,
    max_points: Optional[int],
    quality_level: float,
    min_distance: int,
) -> np.ndarray:
    if resp.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    r = resp.copy()
    r[np.isnan(r)] = 0.0

    rmax = float(np.max(r))
    if rmax <= 0.0:
        return np.zeros((0, 2), dtype=np.int64)

    thresh = float(quality_level) * rmax

    # local maxima via max-filter.
    radius = int(max(0, min_distance))
    if radius == 0:
        is_max = r >= thresh
    else:
        is_max = _local_max_mask(r, radius)
        is_max &= r >= thresh

    ys, xs = np.nonzero(is_max)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    scores = r[ys, xs]
    order = np.argsort(scores)[::-1]
    ys = ys[order]
    xs = xs[order]

    coords = np.stack([ys, xs], axis=1)

    if max_points is not None:
        coords = coords[: int(max_points)]

    return coords


def _local_max_mask(r: np.ndarray, radius: int) -> np.ndarray:
    h, w = r.shape
    k = 2 * radius + 1

    pad = radius
    padded = np.pad(r, ((pad, pad), (pad, pad)), mode="reflect")

    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(h, w, k, k),
        strides=(s0, s1, s0, s1),
        writeable=False,
    )

    local_max = windows.max(axis=(2, 3))
    return r >= local_max
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.