# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Keypoint detection utilities for Neurova."""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np

from neurova.core.constants import CornerDetectionMethod
from neurova.core.image import Image
from neurova.features.corners import detect_corners
from neurova.features.types import Keypoint


KeypointMethod = Literal["harris", "shi_tomasi"]


def detect_keypoints(
    image: Image | np.ndarray,
    *,
    method: KeypointMethod = "harris",
    max_keypoints: Optional[int] = 500,
    quality_level: float = 0.01,
    min_distance: int = 5,
) -> List[Keypoint]:
    """Detect keypoints.

    This currently uses corner detectors as a stable, minimal-dependency basis.

    Returns
    -------
    List[Keypoint]
    """

    coords = detect_corners(
        image,
        method=method,
        max_corners=max_keypoints,
        quality_level=quality_level,
        min_distance=min_distance,
    )

    kps: List[Keypoint] = []
    for x, y in coords:
        kps.append(Keypoint(x=float(x), y=float(y), response=0.0, size=1.0))

    return kps
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.