# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova.core.ops - Core array operations

Provides Neurova arithmetic, bitwise, and utility operations.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np


# Flip Codes

FLIP_HORIZONTAL = 1
FLIP_VERTICAL = 0
FLIP_BOTH = -1

# Rotation flags
ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2

# Border types
BORDER_CONSTANT = 0
BORDER_REPLICATE = 1
BORDER_REFLECT = 2
BORDER_WRAP = 3
BORDER_REFLECT_101 = 4
BORDER_DEFAULT = BORDER_REFLECT_101
BORDER_TRANSPARENT = 5
BORDER_ISOLATED = 16


# Arithmetic Operations

def add(
    src1: np.ndarray,
    src2: Union[np.ndarray, float, Tuple],
    dst: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    dtype: int = -1
) -> np.ndarray:
    """Calculate the per-element sum of two arrays or array and scalar.
    
    Args:
        src1: First input array
        src2: Second input array or scalar
        dst: Optional output array
        mask: Optional mask (8-bit single channel)
        dtype: Output array depth (-1 for same as src1)
    
    Returns:
        Sum of inputs with saturation
    """
    if isinstance(src2, (int, float)):
        result = src1.astype(np.float64) + src2
    elif isinstance(src2, tuple):
        result = src1.astype(np.float64) + np.array(src2)
    else:
        result = src1.astype(np.float64) + src2.astype(np.float64)
    
    # Apply saturation based on dtype
    out_dtype = src1.dtype if dtype == -1 else np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        result = np.clip(result, info.min, info.max)
    
    result = result.astype(out_dtype)
    
    if mask is not None:
        if dst is None:
            dst = src1.copy()
        mask_bool = mask > 0
        if result.ndim > mask.ndim:
            mask_bool = mask_bool[:, :, np.newaxis]
        dst = np.where(mask_bool, result, dst)
        return dst
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def subtract(
    src1: np.ndarray,
    src2: Union[np.ndarray, float, Tuple],
    dst: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    dtype: int = -1
) -> np.ndarray:
    """Calculate the per-element difference between two arrays.
    
    Args:
        src1: First input array
        src2: Second input array or scalar
        dst: Optional output array
        mask: Optional mask
        dtype: Output array depth
    
    Returns:
        Difference with saturation
    """
    if isinstance(src2, (int, float)):
        result = src1.astype(np.float64) - src2
    elif isinstance(src2, tuple):
        result = src1.astype(np.float64) - np.array(src2)
    else:
        result = src1.astype(np.float64) - src2.astype(np.float64)
    
    out_dtype = src1.dtype if dtype == -1 else np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        result = np.clip(result, info.min, info.max)
    
    result = result.astype(out_dtype)
    
    if mask is not None:
        if dst is None:
            dst = src1.copy()
        mask_bool = mask > 0
        if result.ndim > mask.ndim:
            mask_bool = mask_bool[:, :, np.newaxis]
        dst = np.where(mask_bool, result, dst)
        return dst
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def multiply(
    src1: np.ndarray,
    src2: Union[np.ndarray, float],
    dst: Optional[np.ndarray] = None,
    scale: float = 1.0,
    dtype: int = -1
) -> np.ndarray:
    """Calculate the per-element scaled product of two arrays.
    
    Args:
        src1: First input array
        src2: Second input array or scalar
        dst: Optional output array
        scale: Scale factor
        dtype: Output depth
    
    Returns:
        Scaled product
    """
    if isinstance(src2, (int, float)):
        result = src1.astype(np.float64) * src2 * scale
    else:
        result = src1.astype(np.float64) * src2.astype(np.float64) * scale
    
    out_dtype = src1.dtype if dtype == -1 else np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        result = np.clip(result, info.min, info.max)
    
    result = result.astype(out_dtype)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def divide(
    src1: np.ndarray,
    src2: Union[np.ndarray, float],
    dst: Optional[np.ndarray] = None,
    scale: float = 1.0,
    dtype: int = -1
) -> np.ndarray:
    """Perform per-element division of two arrays.
    
    Args:
        src1: First input array (numerator)
        src2: Second input array or scalar (denominator)
        dst: Optional output array
        scale: Scale factor
        dtype: Output depth
    
    Returns:
        Division result
    """
    if isinstance(src2, (int, float)):
        if src2 == 0:
            result = np.zeros_like(src1, dtype=np.float64)
        else:
            result = src1.astype(np.float64) / src2 * scale
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(src2 != 0, 
                            src1.astype(np.float64) / src2.astype(np.float64) * scale, 
                            0)
    
    out_dtype = src1.dtype if dtype == -1 else np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        result = np.clip(result, info.min, info.max)
    
    result = result.astype(out_dtype)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def addWeighted(
    src1: np.ndarray,
    alpha: float,
    src2: np.ndarray,
    beta: float,
    gamma: float,
    dst: Optional[np.ndarray] = None,
    dtype: int = -1
) -> np.ndarray:
    """Calculate the weighted sum of two arrays.
    
    dst = src1*alpha + src2*beta + gamma
    
    Args:
        src1: First input array
        alpha: Weight for first array
        src2: Second input array
        beta: Weight for second array
        gamma: Scalar added to sum
        dst: Optional output array
        dtype: Output depth
    
    Returns:
        Weighted sum
    """
    result = src1.astype(np.float64) * alpha + src2.astype(np.float64) * beta + gamma
    
    out_dtype = src1.dtype if dtype == -1 else np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        result = np.clip(result, info.min, info.max)
    
    result = result.astype(out_dtype)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def absdiff(
    src1: np.ndarray,
    src2: Union[np.ndarray, float],
    dst: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate the absolute difference between two arrays.
    
    Args:
        src1: First input array
        src2: Second input array or scalar
        dst: Optional output array
    
    Returns:
        Absolute difference
    """
    if isinstance(src2, (int, float)):
        result = np.abs(src1.astype(np.float64) - src2)
    else:
        result = np.abs(src1.astype(np.float64) - src2.astype(np.float64))
    
    result = result.astype(src1.dtype)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def convertScaleAbs(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.0
) -> np.ndarray:
    """Scale, compute absolute, and convert to 8-bit.
    
    dst = saturate(|src*alpha + beta|)
    
    Args:
        src: Input array
        dst: Optional output array
        alpha: Scale factor
        beta: Added value
    
    Returns:
        Scaled absolute value as uint8
    """
    result = np.abs(src.astype(np.float64) * alpha + beta)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


# Bitwise Operations

def bitwise_and(
    src1: np.ndarray,
    src2: np.ndarray,
    dst: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate the per-element bitwise conjunction.
    
    Args:
        src1: First input array
        src2: Second input array
        dst: Optional output array
        mask: Optional mask
    
    Returns:
        Bitwise AND result
    """
    result = np.bitwise_and(src1, src2)
    
    if mask is not None:
        if dst is None:
            dst = np.zeros_like(src1)
        mask_bool = mask > 0
        if result.ndim > mask.ndim:
            mask_bool = mask_bool[:, :, np.newaxis]
        result = np.where(mask_bool, result, dst)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def bitwise_or(
    src1: np.ndarray,
    src2: np.ndarray,
    dst: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate the per-element bitwise disjunction.
    
    Args:
        src1: First input array
        src2: Second input array
        dst: Optional output array
        mask: Optional mask
    
    Returns:
        Bitwise OR result
    """
    result = np.bitwise_or(src1, src2)
    
    if mask is not None:
        if dst is None:
            dst = np.zeros_like(src1)
        mask_bool = mask > 0
        if result.ndim > mask.ndim:
            mask_bool = mask_bool[:, :, np.newaxis]
        result = np.where(mask_bool, result, dst)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def bitwise_xor(
    src1: np.ndarray,
    src2: np.ndarray,
    dst: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate the per-element bitwise exclusive or.
    
    Args:
        src1: First input array
        src2: Second input array
        dst: Optional output array
        mask: Optional mask
    
    Returns:
        Bitwise XOR result
    """
    result = np.bitwise_xor(src1, src2)
    
    if mask is not None:
        if dst is None:
            dst = np.zeros_like(src1)
        mask_bool = mask > 0
        if result.ndim > mask.ndim:
            mask_bool = mask_bool[:, :, np.newaxis]
        result = np.where(mask_bool, result, dst)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def bitwise_not(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Invert every bit of an array.
    
    Args:
        src: Input array
        dst: Optional output array
        mask: Optional mask
    
    Returns:
        Bitwise NOT result
    """
    result = np.bitwise_not(src)
    
    if mask is not None:
        if dst is None:
            dst = src.copy()
        mask_bool = mask > 0
        if result.ndim > mask.ndim:
            mask_bool = mask_bool[:, :, np.newaxis]
        result = np.where(mask_bool, result, dst)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


# Array Manipulation

def flip(src: np.ndarray, flipCode: int) -> np.ndarray:
    """Flip a 2D array around vertical, horizontal, or both axes.
    
    Args:
        src: Input array
        flipCode: 0=vertical, 1=horizontal, -1=both
    
    Returns:
        Flipped array
    """
    if flipCode == 0:
        return np.flipud(src)
    elif flipCode > 0:
        return np.fliplr(src)
    else:
        return np.flipud(np.fliplr(src))


def rotate(src: np.ndarray, rotateCode: int) -> np.ndarray:
    """Rotate array by 90, 180, or 270 degrees.
    
    Args:
        src: Input array
        rotateCode: ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
    
    Returns:
        Rotated array
    """
    if rotateCode == ROTATE_90_CLOCKWISE:
        return np.rot90(src, k=-1)
    elif rotateCode == ROTATE_180:
        return np.rot90(src, k=2)
    elif rotateCode == ROTATE_90_COUNTERCLOCKWISE:
        return np.rot90(src, k=1)
    else:
        return src.copy()


def split(m: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Divide a multi-channel array into separate single-channel arrays.
    
    Args:
        m: Multi-channel array
    
    Returns:
        Tuple of single-channel arrays
    """
    if m.ndim < 3:
        return (m.copy(),)
    
    return tuple(m[:, :, i].copy() for i in range(m.shape[2]))


def merge(mv: Union[Tuple[np.ndarray, ...], list]) -> np.ndarray:
    """Create one multi-channel array from several single-channel arrays.
    
    Args:
        mv: Tuple/list of single-channel arrays
    
    Returns:
        Multi-channel array
    """
    if len(mv) == 1:
        return mv[0].copy()
    
    return np.dstack(mv)


def minMaxLoc(
    src: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, float, Tuple[int, int], Tuple[int, int]]:
    """Find minimum and maximum element values and their positions.
    
    Args:
        src: Input single-channel array
        mask: Optional mask
    
    Returns:
        Tuple of (minVal, maxVal, minLoc, maxLoc)
    """
    if mask is not None:
        masked = np.ma.array(src, mask=~(mask > 0))
        min_val = float(masked.min())
        max_val = float(masked.max())
        
        # Find locations
        min_loc = np.unravel_index(masked.argmin(), src.shape)
        max_loc = np.unravel_index(masked.argmax(), src.shape)
    else:
        min_val = float(src.min())
        max_val = float(src.max())
        min_loc = np.unravel_index(src.argmin(), src.shape)
        max_loc = np.unravel_index(src.argmax(), src.shape)
    
    # Neurova returns (x, y) not (row, col)
    return (min_val, max_val, 
            (int(min_loc[1]), int(min_loc[0])), 
            (int(max_loc[1]), int(max_loc[0])))


def normalize(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    norm_type: int = 4,  # NORM_L2
    dtype: int = -1,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Normalize array norm or value range.
    
    Args:
        src: Input array
        dst: Optional output array
        alpha: Norm value or range minimum
        beta: Range maximum (for NORM_MINMAX)
        norm_type: Normalization type
        dtype: Output type
        mask: Optional mask
    
    Returns:
        Normalized array
    """
    NORM_INF = 1
    NORM_L1 = 2
    NORM_L2 = 4
    NORM_MINMAX = 32
    
    src_f = src.astype(np.float64)
    
    if mask is not None:
        mask_bool = mask > 0
    else:
        mask_bool = np.ones(src.shape[:2], dtype=bool)
    
    if norm_type == NORM_MINMAX:
        if mask is not None:
            min_val = src_f[mask_bool].min()
            max_val = src_f[mask_bool].max()
        else:
            min_val = src_f.min()
            max_val = src_f.max()
        
        if max_val - min_val > 0:
            result = (src_f - min_val) / (max_val - min_val) * (beta - alpha) + alpha
        else:
            result = np.full_like(src_f, alpha)
    elif norm_type == NORM_L2:
        if mask is not None:
            norm = np.sqrt(np.sum(src_f[mask_bool] ** 2))
        else:
            norm = np.linalg.norm(src_f)
        
        if norm > 0:
            result = src_f / norm * alpha
        else:
            result = src_f.copy()
    elif norm_type == NORM_L1:
        if mask is not None:
            norm = np.sum(np.abs(src_f[mask_bool]))
        else:
            norm = np.sum(np.abs(src_f))
        
        if norm > 0:
            result = src_f / norm * alpha
        else:
            result = src_f.copy()
    elif norm_type == NORM_INF:
        if mask is not None:
            norm = np.max(np.abs(src_f[mask_bool]))
        else:
            norm = np.max(np.abs(src_f))
        
        if norm > 0:
            result = src_f / norm * alpha
        else:
            result = src_f.copy()
    else:
        result = src_f.copy()
    
    out_dtype = src.dtype if dtype == -1 else np.dtype(dtype)
    result = result.astype(out_dtype)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def countNonZero(src: np.ndarray) -> int:
    """Count non-zero elements.
    
    Args:
        src: Input single-channel array
    
    Returns:
        Number of non-zero elements
    """
    return int(np.count_nonzero(src))


def mean(
    src: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, ...]:
    """Calculate mean of array elements.
    
    Args:
        src: Input array
        mask: Optional mask
    
    Returns:
        Tuple of mean values per channel
    """
    if mask is not None:
        mask_bool = mask > 0
        if src.ndim == 2:
            return (float(src[mask_bool].mean()),)
        else:
            means = []
            for c in range(src.shape[2]):
                means.append(float(src[:, :, c][mask_bool].mean()))
            return tuple(means)
    else:
        if src.ndim == 2:
            return (float(src.mean()),)
        else:
            return tuple(float(src[:, :, c].mean()) for c in range(src.shape[2]))


def meanStdDev(
    src: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and standard deviation.
    
    Args:
        src: Input array
        mask: Optional mask
    
    Returns:
        Tuple of (mean, stddev) arrays
    """
    if mask is not None:
        mask_bool = mask > 0
        if src.ndim == 2:
            mean_val = np.array([[src[mask_bool].mean()]])
            std_val = np.array([[src[mask_bool].std()]])
        else:
            means = []
            stds = []
            for c in range(src.shape[2]):
                means.append([src[:, :, c][mask_bool].mean()])
                stds.append([src[:, :, c][mask_bool].std()])
            mean_val = np.array(means)
            std_val = np.array(stds)
    else:
        if src.ndim == 2:
            mean_val = np.array([[src.mean()]])
            std_val = np.array([[src.std()]])
        else:
            means = []
            stds = []
            for c in range(src.shape[2]):
                means.append([src[:, :, c].mean()])
                stds.append([src[:, :, c].std()])
            mean_val = np.array(means)
            std_val = np.array(stds)
    
    return mean_val, std_val


def LUT(src: np.ndarray, lut: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Perform look-up table transform.
    
    Args:
        src: Input array of 8-bit elements
        lut: Look-up table of 256 elements
        dst: Optional output array
    
    Returns:
        Transformed array
    """
    result = lut.flatten()[src]
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def copyMakeBorder(
    src: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    borderType: int = BORDER_CONSTANT,
    value: Union[float, Tuple] = 0
) -> np.ndarray:
    """Add border around image.
    
    Args:
        src: Input image
        top, bottom, left, right: Border widths
        borderType: Border type (BORDER_*)
        value: Border value for BORDER_CONSTANT
    
    Returns:
        Image with border
    """
    mode_map = {
        BORDER_CONSTANT: 'constant',
        BORDER_REPLICATE: 'edge',
        BORDER_REFLECT: 'symmetric',
        BORDER_WRAP: 'wrap',
        BORDER_REFLECT_101: 'reflect',
    }
    
    mode = mode_map.get(borderType, 'constant')
    
    if src.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))
    
    if borderType == BORDER_CONSTANT:
        if isinstance(value, (int, float)):
            const_val = value
        else:
            const_val = 0  # np.pad constant_values doesn't support per-channel
        return np.pad(src, pad_width, mode='constant', constant_values=const_val)
    else:
        return np.pad(src, pad_width, mode=mode)


def magnitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate magnitude of 2D vectors.
    
    Args:
        x: X components
        y: Y components
    
    Returns:
        Magnitude sqrt(x^2 + y^2)
    """
    return np.sqrt(x.astype(np.float64)**2 + y.astype(np.float64)**2)


def phase(x: np.ndarray, y: np.ndarray, angleInDegrees: bool = False) -> np.ndarray:
    """Calculate angle of 2D vectors.
    
    Args:
        x: X components
        y: Y components
        angleInDegrees: If True, output in degrees
    
    Returns:
        Angle in radians or degrees
    """
    angles = np.arctan2(y.astype(np.float64), x.astype(np.float64))
    if angleInDegrees:
        angles = np.degrees(angles)
    return angles


def cartToPolar(
    x: np.ndarray,
    y: np.ndarray,
    angleInDegrees: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian to polar coordinates.
    
    Args:
        x: X coordinates
        y: Y coordinates
        angleInDegrees: If True, angles in degrees
    
    Returns:
        Tuple of (magnitude, angle)
    """
    mag = magnitude(x, y)
    ang = phase(x, y, angleInDegrees)
    return mag, ang


def polarToCart(
    magnitude: np.ndarray,
    angle: np.ndarray,
    angleInDegrees: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert polar to Cartesian coordinates.
    
    Args:
        magnitude: Magnitudes
        angle: Angles
        angleInDegrees: If True, angles in degrees
    
    Returns:
        Tuple of (x, y)
    """
    if angleInDegrees:
        angle = np.radians(angle)
    
    x = magnitude * np.cos(angle)
    y = magnitude * np.sin(angle)
    return x, y


# Array Concatenation and Manipulation

def hconcat(src: list) -> np.ndarray:
    """Concatenate arrays horizontally.
    
    Args:
        src: List of arrays to concatenate
    
    Returns:
        Horizontally concatenated array
    """
    return np.hstack(src)


def vconcat(src: list) -> np.ndarray:
    """Concatenate arrays vertically.
    
    Args:
        src: List of arrays to concatenate
    
    Returns:
        Vertically concatenated array
    """
    return np.vstack(src)


def repeat(src: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Repeat array in both directions.
    
    Args:
        src: Input array
        ny: Number of times to repeat along vertical axis
        nx: Number of times to repeat along horizontal axis
    
    Returns:
        Tiled array
    """
    if src.ndim == 2:
        return np.tile(src, (ny, nx))
    else:
        return np.tile(src, (ny, nx, 1))


def transpose(src: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Transpose a matrix.
    
    Args:
        src: Input array
        dst: Optional output array
    
    Returns:
        Transposed array
    """
    if src.ndim == 2:
        result = src.T
    else:
        result = np.transpose(src, (1, 0, 2))
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def reduce(
    src: np.ndarray,
    dim: int,
    rtype: int,
    dtype: int = -1
) -> np.ndarray:
    """Reduce a matrix along a dimension.
    
    Args:
        src: Input array
        dim: Dimension to reduce (0=rows, 1=cols)
        rtype: Reduction type (0=sum, 1=avg, 2=max, 3=min)
        dtype: Output dtype
    
    Returns:
        Reduced array
    """
    if rtype == 0:  # CV_REDUCE_SUM
        result = np.sum(src, axis=dim, keepdims=True)
    elif rtype == 1:  # CV_REDUCE_AVG
        result = np.mean(src, axis=dim, keepdims=True)
    elif rtype == 2:  # CV_REDUCE_MAX
        result = np.max(src, axis=dim, keepdims=True)
    elif rtype == 3:  # CV_REDUCE_MIN
        result = np.min(src, axis=dim, keepdims=True)
    else:
        result = np.sum(src, axis=dim, keepdims=True)
    
    if dtype != -1:
        result = result.astype(dtype)
    
    return result


# Reduce type constants
REDUCE_SUM = 0
REDUCE_AVG = 1
REDUCE_MAX = 2
REDUCE_MIN = 3


def inRange(
    src: np.ndarray,
    lowerb: Union[np.ndarray, Tuple, float],
    upperb: Union[np.ndarray, Tuple, float],
    dst: Optional[np.ndarray] = None
) -> np.ndarray:
    """Check if array elements lie between two boundaries.
    
    Args:
        src: Input array
        lowerb: Lower boundary
        upperb: Upper boundary
        dst: Optional output array
    
    Returns:
        Output array (255 for in-range, 0 otherwise)
    """
    lowerb = np.asarray(lowerb)
    upperb = np.asarray(upperb)
    
    mask = (src >= lowerb) & (src <= upperb)
    
    if src.ndim == 3:
        mask = np.all(mask, axis=2)
    
    result = (mask * 255).astype(np.uint8)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def compare(
    src1: np.ndarray,
    src2: Union[np.ndarray, float],
    cmpop: int
) -> np.ndarray:
    """Compare two arrays element-wise.
    
    Args:
        src1: First input array
        src2: Second input array or scalar
        cmpop: Comparison operation (CMP_EQ, CMP_GT, etc.)
    
    Returns:
        Comparison result (255 for true, 0 for false)
    """
    if cmpop == 0:  # CMP_EQ
        result = src1 == src2
    elif cmpop == 1:  # CMP_GT
        result = src1 > src2
    elif cmpop == 2:  # CMP_GE
        result = src1 >= src2
    elif cmpop == 3:  # CMP_LT
        result = src1 < src2
    elif cmpop == 4:  # CMP_LE
        result = src1 <= src2
    elif cmpop == 5:  # CMP_NE
        result = src1 != src2
    else:
        result = src1 == src2
    
    return (result * 255).astype(np.uint8)


# Compare operation constants
CMP_EQ = 0
CMP_GT = 1
CMP_GE = 2
CMP_LT = 3
CMP_LE = 4
CMP_NE = 5


def checkRange(
    a: np.ndarray,
    quiet: bool = True,
    minVal: float = float('-inf'),
    maxVal: float = float('inf')
) -> Union[bool, Tuple[bool, Tuple[int, int]]]:
    """Check if array elements are within a range.
    
    Args:
        a: Input array
        quiet: If True, return only bool; otherwise return position too
        minVal: Minimum value
        maxVal: Maximum value
    
    Returns:
        True if all elements in range, or (bool, position) if not quiet
    """
    valid = np.all((a >= minVal) & (a <= maxVal))
    
    if quiet:
        return bool(valid)
    
    if valid:
        return True, (-1, -1)
    
    # Find first out-of-range element
    mask = (a < minVal) | (a > maxVal)
    pos = np.argwhere(mask)
    if len(pos) > 0:
        return False, tuple(pos[0])
    return True, (-1, -1)


def sqrt(src: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate square root of array elements.
    
    Args:
        src: Input array
        dst: Optional output array
    
    Returns:
        Square root of elements
    """
    result = np.sqrt(src.astype(np.float64))
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def pow(src: np.ndarray, power: float, dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Raise array elements to a power.
    
    Args:
        src: Input array
        power: Exponent
        dst: Optional output array
    
    Returns:
        Array elements raised to power
    """
    result = np.power(src.astype(np.float64), power)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def exp(src: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate exponential of array elements.
    
    Args:
        src: Input array
        dst: Optional output array
    
    Returns:
        Exponential of elements
    """
    result = np.exp(src.astype(np.float64))
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def log(src: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate natural logarithm of array elements.
    
    Args:
        src: Input array
        dst: Optional output array
    
    Returns:
        Natural log of elements
    """
    result = np.log(src.astype(np.float64) + 1e-10)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def min(src1: np.ndarray, src2: Union[np.ndarray, float], dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate per-element minimum.
    
    Args:
        src1: First input array
        src2: Second input array or scalar
        dst: Optional output array
    
    Returns:
        Per-element minimum
    """
    result = np.minimum(src1, src2)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def max(src1: np.ndarray, src2: Union[np.ndarray, float], dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate per-element maximum.
    
    Args:
        src1: First input array
        src2: Second input array or scalar
        dst: Optional output array
    
    Returns:
        Per-element maximum
    """
    result = np.maximum(src1, src2)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def sum(src: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
    """Calculate sum of array elements.
    
    Args:
        src: Input array
        mask: Optional mask
    
    Returns:
        Scalar or tuple of channel sums
    """
    if mask is not None:
        src = src.copy()
        if src.ndim == 3 and mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        src = np.where(mask > 0, src, 0)
    
    if src.ndim == 2:
        return (float(np.sum(src)),)
    else:
        return tuple(float(np.sum(src[:, :, c])) for c in range(src.shape[2]))


def trace(mtx: np.ndarray) -> Tuple:
    """Calculate trace of a matrix.
    
    Args:
        mtx: Input matrix
    
    Returns:
        Sum of diagonal elements as tuple
    """
    if mtx.ndim == 2:
        return (float(np.trace(mtx)),)
    else:
        # Multi-channel
        return tuple(float(np.trace(mtx[:, :, c])) for c in range(mtx.shape[2]))


def determinant(mtx: np.ndarray) -> float:
    """Calculate determinant of a matrix.
    
    Args:
        mtx: Input square matrix
    
    Returns:
        Determinant value
    """
    return float(np.linalg.det(mtx))


def invert(src: np.ndarray, dst: Optional[np.ndarray] = None, flags: int = 0) -> Tuple[float, np.ndarray]:
    """Invert a matrix.
    
    Args:
        src: Input matrix
        dst: Optional output matrix
        flags: Inversion method (0=LU, 1=SVD, 2=EIGEN, 3=CHOLESKY)
    
    Returns:
        Tuple of (condition number, inverse matrix)
    """
    try:
        result = np.linalg.inv(src)
        cond = np.linalg.cond(src)
    except np.linalg.LinAlgError:
        result = np.zeros_like(src)
        cond = 0.0
    
    if dst is not None:
        np.copyto(dst, result)
        return cond, dst
    return cond, result


# Decomposition flags
DECOMP_LU = 0
DECOMP_SVD = 1
DECOMP_EIG = 2
DECOMP_CHOLESKY = 3


def solve(src1: np.ndarray, src2: np.ndarray, dst: Optional[np.ndarray] = None, flags: int = 0) -> Tuple[bool, np.ndarray]:
    """Solve a system of linear equations.
    
    Args:
        src1: Coefficient matrix
        src2: Right-hand side
        dst: Optional output
        flags: Solution method
    
    Returns:
        Tuple of (success, solution)
    """
    try:
        result = np.linalg.solve(src1, src2)
        success = True
    except np.linalg.LinAlgError:
        result = np.zeros_like(src2)
        success = False
    
    if dst is not None:
        np.copyto(dst, result)
        return success, dst
    return success, result


def eigen(src: np.ndarray, eigenvalues: Optional[np.ndarray] = None, eigenvectors: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Calculate eigenvalues and eigenvectors.
    
    Args:
        src: Input symmetric matrix
        eigenvalues: Optional output eigenvalues
        eigenvectors: Optional output eigenvectors
    
    Returns:
        Tuple of (success, eigenvalues, eigenvectors)
    """
    try:
        vals, vecs = np.linalg.eig(src)
        # Sort by decreasing eigenvalue
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        success = True
    except np.linalg.LinAlgError:
        vals = np.zeros(src.shape[0])
        vecs = np.zeros_like(src)
        success = False
    
    if eigenvalues is not None:
        np.copyto(eigenvalues, vals)
    if eigenvectors is not None:
        np.copyto(eigenvectors, vecs)
    
    return success, vals, vecs


def SVDecomp(src: np.ndarray, w: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None, vt: Optional[np.ndarray] = None, flags: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform SVD decomposition.
    
    Args:
        src: Input matrix
        w: Optional singular values output
        u: Optional left singular vectors
        vt: Optional right singular vectors (transposed)
        flags: Operation flags
    
    Returns:
        Tuple of (w, u, vt)
    """
    u_out, w_out, vt_out = np.linalg.svd(src, full_matrices=True)
    
    if w is not None:
        np.copyto(w, w_out)
    if u is not None:
        np.copyto(u, u_out)
    if vt is not None:
        np.copyto(vt, vt_out)
    
    return w_out, u_out, vt_out


def gemm(src1: np.ndarray, src2: np.ndarray, alpha: float, src3: np.ndarray, beta: float, dst: Optional[np.ndarray] = None, flags: int = 0) -> np.ndarray:
    """Generalized matrix multiplication.
    
    Computes: dst = alpha * src1 * src2 + beta * src3
    
    Args:
        src1: First input matrix
        src2: Second input matrix
        alpha: Scale factor for product
        src3: Third input matrix
        beta: Scale factor for src3
        dst: Optional output
        flags: Operation flags (1=transpose src1, 2=transpose src2)
    
    Returns:
        Result matrix
    """
    a = src1.T if flags & 1 else src1
    b = src2.T if flags & 2 else src2
    
    result = alpha * (a @ b) + beta * src3
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


# GEMM flags
GEMM_1_T = 1
GEMM_2_T = 2
GEMM_3_T = 4


def mulTransposed(src: np.ndarray, aTa: bool, dst: Optional[np.ndarray] = None, delta: Optional[np.ndarray] = None, scale: float = 1.0, dtype: int = -1) -> np.ndarray:
    """Calculate product of transposed matrix.
    
    Args:
        src: Input matrix
        aTa: If True, compute src.T @ src; else src @ src.T
        dst: Optional output
        delta: Optional array to subtract from src
        scale: Scale factor
        dtype: Output dtype
    
    Returns:
        Result matrix
    """
    if delta is not None:
        src = src - delta
    
    if aTa:
        result = src.T @ src
    else:
        result = src @ src.T
    
    result = scale * result
    
    if dtype != -1:
        result = result.astype(dtype)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def completeSymm(m: np.ndarray, lowerToUpper: bool = False) -> None:
    """Complete symmetric matrix from lower/upper triangle.
    
    Args:
        m: Input/output matrix
        lowerToUpper: If True, copy lower to upper; else upper to lower
    """
    if lowerToUpper:
        m[:] = np.tril(m) + np.tril(m, -1).T
    else:
        m[:] = np.triu(m) + np.triu(m, 1).T


def setIdentity(mtx: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Set matrix to identity.
    
    Args:
        mtx: Matrix to modify
        s: Diagonal value
    
    Returns:
        Modified matrix
    """
    mtx[:] = 0
    np.fill_diagonal(mtx, s)
    return mtx


# Exports

__all__ = [
    # Arithmetic
    "add", "subtract", "multiply", "divide",
    "addWeighted", "absdiff", "convertScaleAbs",
    
    # Bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    
    # Array manipulation
    "flip", "rotate", "split", "merge",
    "minMaxLoc", "normalize", "countNonZero",
    "mean", "meanStdDev", "LUT", "copyMakeBorder",
    "magnitude", "phase", "cartToPolar", "polarToCart",
    
    # Array concatenation (NEW)
    "hconcat", "vconcat", "repeat", "transpose", "reduce",
    "REDUCE_SUM", "REDUCE_AVG", "REDUCE_MAX", "REDUCE_MIN",
    
    # Comparison (NEW)
    "inRange", "compare", "checkRange",
    "CMP_EQ", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_NE",
    
    # Math functions (NEW)
    "sqrt", "pow", "exp", "log", "min", "max", "sum", "trace",
    
    # Linear algebra (NEW)
    "determinant", "invert", "solve", "eigen", "SVDecomp",
    "gemm", "mulTransposed", "completeSymm", "setIdentity",
    "DECOMP_LU", "DECOMP_SVD", "DECOMP_EIG", "DECOMP_CHOLESKY",
    "GEMM_1_T", "GEMM_2_T", "GEMM_3_T",
    
    # Constants
    "FLIP_HORIZONTAL", "FLIP_VERTICAL", "FLIP_BOTH",
    "ROTATE_90_CLOCKWISE", "ROTATE_180", "ROTATE_90_COUNTERCLOCKWISE",
    "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
    "BORDER_WRAP", "BORDER_REFLECT_101", "BORDER_DEFAULT",
    "BORDER_TRANSPARENT", "BORDER_ISOLATED",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.