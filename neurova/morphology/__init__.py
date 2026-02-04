# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Morphology operations for Neurova."""

from neurova.morphology.binary import (
    binary_dilate,
    binary_erode,
    binary_open,
    binary_close,
    binary_gradient,
    structuring_element,
)

# Neurova morphology functions
from neurova.morphology.morph_ops import (
    getStructuringElement,
    erode,
    dilate,
    morphologyEx,
    MORPH_RECT,
    MORPH_CROSS,
    MORPH_ELLIPSE,
    MORPH_ERODE,
    MORPH_DILATE,
    MORPH_OPEN,
    MORPH_CLOSE,
    MORPH_GRADIENT,
    MORPH_TOPHAT,
    MORPH_BLACKHAT,
    MORPH_HITMISS,
)

__all__ = [
    # Binary operations
    "binary_dilate",
    "binary_erode",
    "binary_open",
    "binary_close",
    "binary_gradient",
    "structuring_element",
    # Neurova functions
    "getStructuringElement",
    "erode",
    "dilate",
    "morphologyEx",
    # Shape constants
    "MORPH_RECT",
    "MORPH_CROSS",
    "MORPH_ELLIPSE",
    # Operation constants
    "MORPH_ERODE",
    "MORPH_DILATE",
    "MORPH_OPEN",
    "MORPH_CLOSE",
    "MORPH_GRADIENT",
    "MORPH_TOPHAT",
    "MORPH_BLACKHAT",
    "MORPH_HITMISS",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.