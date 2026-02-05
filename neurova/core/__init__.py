# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Core module for Neurova - Image processing fundamentals"""

from __future__ import annotations

from typing import Optional

import numpy as np

from neurova.core.constants import (
    ColorSpace,
    BorderMode,
    InterpolationMode,
    ThresholdMethod,
    MorphologyOp,
    KernelShape,
    EdgeDetectionMethod,
    CornerDetectionMethod,
    DescriptorType,
    MatcherType,
    DistanceMetric,
    ActivationFunction,
    LossFunction,
    OptimizerType,
    PaddingMode,
    PoolingType,
)

from neurova.core.errors import (
    NeurovaError,
    ImageError,
    InvalidImageError,
    DimensionError,
    ColorSpaceError,
    ValidationError,
    ShapeError,
    DataTypeError,
)

from neurova.core.dtypes import (
    DataType,
    DType,
    get_dtype,
    is_integer_dtype,
    is_floating_dtype,
    is_unsigned_dtype,
    get_dtype_range,
    convert_dtype,
    safe_cast,
    uint8,
    uint16,
    float32,
    float64,
)

from neurova.core.array_ops import (
    ensure_array,
    ensure_2d,
    ensure_3d,
    ensure_4d,
    validate_shape,
    validate_image_shape,
    get_spatial_shape,
    get_num_channels,
    normalize,
    standardize,
    pad_array,
    clip_array,
)

# Neurova core operations
from neurova.core.ops import (
    add, subtract, multiply, divide, addWeighted, absdiff, convertScaleAbs,
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
    flip, rotate, split, merge, minMaxLoc, countNonZero,
    mean, meanStdDev, LUT, copyMakeBorder, magnitude, phase, cartToPolar, polarToCart,
    FLIP_HORIZONTAL, FLIP_VERTICAL, FLIP_BOTH,
    ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE,
    BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
    BORDER_WRAP, BORDER_REFLECT_101, BORDER_DEFAULT,
    BORDER_TRANSPARENT, BORDER_ISOLATED,
)

from neurova.core.image import (
    Image,
    ImageInfo,
    create_blank_image,
    create_from_array,
)

from neurova.core.color import (
    convert_color_space as _convert_color_space,
    to_grayscale as _to_grayscale,
)


def to_grayscale(
    image: Image | np.ndarray,
    *,
    from_space: Optional[ColorSpace] = None,
    weights: Optional[tuple] = None,
) -> np.ndarray:
    """Convert an image to grayscale.

    This is a convenience wrapper around `neurova.core.color.to_grayscale` that
    also accepts `neurova.core.image.Image`.

    Parameters
    ----------
    image:
        Input `Image` or NumPy array.
    from_space:
        Source color space. If None and `image` is an `Image`, uses
        `image.color_space`. If None and `image` is an array, assumes RGB.
    weights:
        Optional (R, G, B) weights.
    """

    if isinstance(image, Image):
        fs = image.color_space if from_space is None else from_space
        return _to_grayscale(image.data, from_space=fs, weights=weights)

    fs = ColorSpace.RGB if from_space is None else from_space
    return _to_grayscale(np.asarray(image), from_space=fs, weights=weights)


def convert_color_space(
    image: Image | np.ndarray,
    to_space: ColorSpace,
    *,
    from_space: Optional[ColorSpace] = None,
) -> np.ndarray:
    """Convert an image between color spaces.

    This is a convenience wrapper around `neurova.core.color.convert_color_space`
    that also accepts `neurova.core.image.Image`.
    """

    if isinstance(image, Image):
        fs = image.color_space if from_space is None else from_space
        return _convert_color_space(image.data, fs, to_space)

    if from_space is None:
        raise ValueError("from_space is required when converting a NumPy array")

    return _convert_color_space(np.asarray(image), from_space, to_space)

__all__ = [
    # constants and Enums
    "ColorSpace",
    "BorderMode",
    "InterpolationMode",
    "ThresholdMethod",
    "MorphologyOp",
    "KernelShape",
    "EdgeDetectionMethod",
    "CornerDetectionMethod",
    "DescriptorType",
    "MatcherType",
    "DistanceMetric",
    "ActivationFunction",
    "LossFunction",
    "OptimizerType",
    "PaddingMode",
    "PoolingType",
    # errors
    "NeurovaError",
    "ImageError",
    "InvalidImageError",
    "DimensionError",
    "ColorSpaceError",
    "ValidationError",
    "ShapeError",
    "DataTypeError",
    # data Types
    "DataType",
    "DType",
    "get_dtype",
    "is_integer_dtype",
    "is_floating_dtype",
    "is_unsigned_dtype",
    "get_dtype_range",
    "convert_dtype",
    "safe_cast",
    "uint8",
    "uint16",
    "float32",
    "float64",
    # array Operations
    "ensure_array",
    "ensure_2d",
    "ensure_3d",
    "ensure_4d",
    "validate_shape",
    "validate_image_shape",
    "get_spatial_shape",
    "get_num_channels",
    "normalize",
    "standardize",
    "pad_array",
    "clip_array",
    # Neurova core operations
    "add",
    "subtract",
    "multiply",
    "divide",
    "addWeighted",
    "absdiff",
    "convertScaleAbs",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "flip",
    "rotate",
    "split",
    "merge",
    "minMaxLoc",
    "countNonZero",
    "mean",
    "meanStdDev",
    "LUT",
    "copyMakeBorder",
    "magnitude",
    "phase",
    "cartToPolar",
    "polarToCart",
    # Flip/rotate/border constants
    "FLIP_HORIZONTAL",
    "FLIP_VERTICAL",
    "FLIP_BOTH",
    "ROTATE_90_CLOCKWISE",
    "ROTATE_180",
    "ROTATE_90_COUNTERCLOCKWISE",
    "BORDER_CONSTANT",
    "BORDER_REPLICATE",
    "BORDER_REFLECT",
    "BORDER_WRAP",
    "BORDER_REFLECT_101",
    "BORDER_DEFAULT",
    "BORDER_TRANSPARENT",
    "BORDER_ISOLATED",
    # image Class
    "Image",
    "ImageInfo",
    "create_blank_image",
    "create_from_array",
    # color API convenience
    "to_grayscale",
    "convert_color_space",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.