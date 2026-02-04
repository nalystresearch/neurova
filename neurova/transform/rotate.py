# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Rotation operations for Neurova."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from neurova.core.constants import InterpolationMode
from neurova.core.errors import ValidationError
from neurova.transform.affine import get_rotation_matrix2d
from neurova.transform.warp import warp_affine


BorderModeStr = Literal["constant", "reflect", "replicate", "wrap"]


def rotate(
    image: np.ndarray,
    angle_degrees: float,
    *,
    center: Tuple[float, float] | None = None,
    scale: float = 1.0,
    interpolation: InterpolationMode = InterpolationMode.LINEAR,
    border_mode: BorderModeStr = "constant",
    constant_value: float = 0.0,
    keep_size: bool = True,
) -> np.ndarray:
    """Rotate an image by an angle in degrees.

    Parameters
    ----------
    keep_size:
        If True, output size equals input size.
        If False, expand the canvas to fit the rotated image.
    """

    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValidationError("image", img.shape, "2D or 3D array")

    h, w = int(img.shape[0]), int(img.shape[1])

    if center is None:
        center = (w * 0.5, h * 0.5)

    m = get_rotation_matrix2d(center, angle_degrees, scale=scale)

    if keep_size:
        out_w, out_h = w, h
        return warp_affine(
            img,
            m,
            (out_w, out_h),
            interpolation=interpolation,
            border_mode=border_mode,
            constant_value=constant_value,
        )

    # compute bounds by rotating corner points.
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [w - 1.0, 0.0, 1.0],
            [0.0, h - 1.0, 1.0],
            [w - 1.0, h - 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    m33 = np.vstack([m, [0.0, 0.0, 1.0]])
    rc = (m33 @ corners.T).T
    xs = rc[:, 0]
    ys = rc[:, 1]

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    out_w = int(np.ceil(max_x - min_x + 1.0))
    out_h = int(np.ceil(max_y - min_y + 1.0))

    if out_w <= 0 or out_h <= 0:
        raise ValidationError("output", (out_h, out_w), "positive output size")

    # translate so that min corner maps to (0,0).
    m2 = m.copy()
    m2[0, 2] -= min_x
    m2[1, 2] -= min_y

    return warp_affine(
        img,
        m2,
        (out_w, out_h),
        interpolation=interpolation,
        border_mode=border_mode,
        constant_value=constant_value,
    )
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.