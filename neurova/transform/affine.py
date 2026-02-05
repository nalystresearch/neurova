# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Affine transformation utilities for Neurova."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def get_rotation_matrix2d(
    center: Tuple[float, float],
    angle_degrees: float,
    scale: float = 1.0,
) -> np.ndarray:
    """Create a 2x3 rotation matrix around a center.

    Parameters
    ----------
    center:
        (cx, cy) in pixel coordinates.
    angle_degrees:
        Rotation angle in degrees. Positive values rotate counter-clockwise.
    scale:
        Uniform scale.

    Returns
    -------
    np.ndarray
        2x3 affine transform matrix.
    """

    cx, cy = float(center[0]), float(center[1])
    a = np.deg2rad(float(angle_degrees))
    s = float(scale)

    alpha = s * float(np.cos(a))
    beta = s * float(np.sin(a))

    # matches Neurova getRotationMatrix2D convention.
    m = np.array(
        [
            [alpha, beta, (1.0 - alpha) * cx - beta * cy],
            [-beta, alpha, beta * cx + (1.0 - alpha) * cy],
        ],
        dtype=np.float64,
    )

    return m
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.