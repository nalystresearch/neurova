# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Common types for feature detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Keypoint:
    """A minimal keypoint representation."""

    x: float
    y: float
    response: float = 0.0
    size: float = 1.0
    angle: float = 0.0
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.