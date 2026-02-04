# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova.imgproc.threshold - Thresholding functions

Provides Neurova thresholding operations.
"""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np


# Threshold Types

THRESH_BINARY = 0
THRESH_BINARY_INV = 1
THRESH_TRUNC = 2
THRESH_TOZERO = 3
THRESH_TOZERO_INV = 4
THRESH_MASK = 7
THRESH_OTSU = 8
THRESH_TRIANGLE = 16

# Adaptive threshold methods
ADAPTIVE_THRESH_MEAN_C = 0
ADAPTIVE_THRESH_GAUSSIAN_C = 1


def threshold(
    src: np.ndarray,
    thresh: float,
    maxval: float,
    type: int
) -> Tuple[float, np.ndarray]:
    """Apply a fixed-level threshold to an image.
    
    Args:
        src: Input image (single-channel, 8-bit or 32-bit float)
        thresh: Threshold value
        maxval: Maximum value for THRESH_BINARY and THRESH_BINARY_INV
        type: Thresholding type (THRESH_*)
    
    Returns:
        Tuple of (computed_threshold, thresholded_image)
    """
    if src.size == 0:
        return thresh, src.copy()
    
    # Handle OTSU
    if type & THRESH_OTSU:
        thresh = _otsu_threshold(src)
        type = type & ~THRESH_OTSU
    
    # Handle TRIANGLE
    if type & THRESH_TRIANGLE:
        thresh = _triangle_threshold(src)
        type = type & ~THRESH_TRIANGLE
    
    # Apply threshold
    src_f = src.astype(np.float32)
    
    if type == THRESH_BINARY:
        dst = np.where(src_f > thresh, maxval, 0)
    elif type == THRESH_BINARY_INV:
        dst = np.where(src_f > thresh, 0, maxval)
    elif type == THRESH_TRUNC:
        dst = np.where(src_f > thresh, thresh, src_f)
    elif type == THRESH_TOZERO:
        dst = np.where(src_f > thresh, src_f, 0)
    elif type == THRESH_TOZERO_INV:
        dst = np.where(src_f > thresh, 0, src_f)
    else:
        dst = np.where(src_f > thresh, maxval, 0)
    
    return float(thresh), dst.astype(src.dtype)


def _otsu_threshold(src: np.ndarray) -> float:
    """Compute Otsu's threshold value."""
    # Flatten and ensure uint8
    pixels = src.flatten()
    if pixels.dtype != np.uint8:
        pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min() + 1e-10) * 255).astype(np.uint8)
    
    # Compute histogram
    hist = np.bincount(pixels, minlength=256).astype(np.float64)
    hist /= hist.sum() + 1e-10
    
    # Compute cumulative sums
    bins = np.arange(256)
    weight0 = np.cumsum(hist)
    weight1 = np.cumsum(hist[::-1])[::-1]
    
    mean0 = np.cumsum(hist * bins) / (weight0 + 1e-10)
    mean1 = (np.cumsum((hist * bins)[::-1]) / (weight1[::-1] + 1e-10))[::-1]
    
    # Compute between-class variance
    variance = weight0[:-1] * weight1[1:] * (mean0[:-1] - mean1[1:]) ** 2
    
    # Find optimal threshold
    idx = np.argmax(variance)
    
    return float(idx)


def _triangle_threshold(src: np.ndarray) -> float:
    """Compute triangle threshold value."""
    # Flatten and ensure uint8
    pixels = src.flatten()
    if pixels.dtype != np.uint8:
        pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min() + 1e-10) * 255).astype(np.uint8)
    
    # Compute histogram
    hist = np.bincount(pixels, minlength=256)
    
    # Find histogram max
    peak_idx = np.argmax(hist)
    
    # Find line endpoints
    left_idx = 0
    right_idx = 255
    
    for i in range(256):
        if hist[i] > 0:
            left_idx = i
            break
    
    for i in range(255, -1, -1):
        if hist[i] > 0:
            right_idx = i
            break
    
    # Use the longer side
    if peak_idx - left_idx > right_idx - peak_idx:
        # Use left side
        x1, y1 = left_idx, hist[left_idx]
        x2, y2 = peak_idx, hist[peak_idx]
    else:
        # Use right side
        x1, y1 = peak_idx, hist[peak_idx]
        x2, y2 = right_idx, hist[right_idx]
    
    # Find maximum distance from line to histogram
    max_dist = 0
    thresh = peak_idx
    
    line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if line_len < 1:
        return float(peak_idx)
    
    for x in range(x1, x2 + 1):
        # Distance from point to line
        dist = abs((y2 - y1) * x - (x2 - x1) * hist[x] + x2 * y1 - y2 * x1) / line_len
        if dist > max_dist:
            max_dist = dist
            thresh = x
    
    return float(thresh)


def adaptiveThreshold(
    src: np.ndarray,
    maxValue: float,
    adaptiveMethod: int,
    thresholdType: int,
    blockSize: int,
    C: float
) -> np.ndarray:
    """Apply adaptive thresholding.
    
    Args:
        src: Source 8-bit single-channel image
        maxValue: Value for pixels exceeding threshold
        adaptiveMethod: ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C
        thresholdType: THRESH_BINARY or THRESH_BINARY_INV
        blockSize: Size of neighborhood (must be odd, >= 3)
        C: Constant subtracted from mean/weighted mean
    
    Returns:
        Thresholded image
    """
    if src.size == 0:
        return src.copy()
    
    # Ensure blockSize is odd
    blockSize = max(3, blockSize | 1)
    
    # Compute local threshold based on method
    if adaptiveMethod == ADAPTIVE_THRESH_GAUSSIAN_C:
        # Gaussian weighted mean
        thresh_img = _gaussian_adaptive(src, blockSize)
    else:
        # Mean
        thresh_img = _mean_adaptive(src, blockSize)
    
    # Subtract constant
    thresh_img = thresh_img - C
    
    # Apply threshold
    src_f = src.astype(np.float32)
    
    if thresholdType == THRESH_BINARY:
        dst = np.where(src_f > thresh_img, maxValue, 0)
    elif thresholdType == THRESH_BINARY_INV:
        dst = np.where(src_f > thresh_img, 0, maxValue)
    else:
        dst = np.where(src_f > thresh_img, maxValue, 0)
    
    return dst.astype(np.uint8)


def _mean_adaptive(src: np.ndarray, block_size: int) -> np.ndarray:
    """Compute local mean for adaptive thresholding."""
    # Use cumulative sum for efficient computation
    h, w = src.shape
    pad = block_size // 2
    
    # Pad image
    padded = np.pad(src.astype(np.float64), pad, mode='reflect')
    
    # Compute cumulative sum
    cum_sum = np.zeros((h + block_size, w + block_size), dtype=np.float64)
    cum_sum[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    
    # Compute local mean using integral image
    local_mean = (
        cum_sum[block_size:, block_size:] -
        cum_sum[:-block_size, block_size:] -
        cum_sum[block_size:, :-block_size] +
        cum_sum[:-block_size, :-block_size]
    ) / (block_size * block_size)
    
    return local_mean[:h, :w]


def _gaussian_adaptive(src: np.ndarray, block_size: int) -> np.ndarray:
    """Compute Gaussian-weighted local mean for adaptive thresholding."""
    h, w = src.shape
    sigma = block_size / 6.0
    
    # Create 1D Gaussian kernel
    x = np.arange(block_size) - block_size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    
    # Apply separable convolution
    pad = block_size // 2
    padded = np.pad(src.astype(np.float64), pad, mode='reflect')
    
    # Horizontal pass
    temp = np.zeros_like(padded)
    for i in range(block_size):
        temp[:, pad:-pad] += padded[:, i:i+w] * kernel_1d[i]
    
    # Vertical pass
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(block_size):
        result += temp[i:i+h, pad:-pad] * kernel_1d[i]
    
    return result


def inRange(
    src: np.ndarray,
    lowerb: Union[np.ndarray, Tuple, float],
    upperb: Union[np.ndarray, Tuple, float]
) -> np.ndarray:
    """Check if array elements lie between two scalars.
    
    Args:
        src: Input array
        lowerb: Inclusive lower boundary
        upperb: Inclusive upper boundary
    
    Returns:
        Binary mask (255 where in range, 0 otherwise)
    """
    if src.size == 0:
        return np.zeros(src.shape[:2], dtype=np.uint8)
    
    # Convert boundaries to arrays
    if isinstance(lowerb, (int, float)):
        lower = np.full(src.shape[-1] if src.ndim == 3 else 1, lowerb)
    else:
        lower = np.asarray(lowerb)
    
    if isinstance(upperb, (int, float)):
        upper = np.full(src.shape[-1] if src.ndim == 3 else 1, upperb)
    else:
        upper = np.asarray(upperb)
    
    # Create mask
    if src.ndim == 2:
        mask = (src >= lower[0]) & (src <= upper[0])
    else:
        mask = np.ones(src.shape[:2], dtype=bool)
        for c in range(min(src.shape[2], len(lower))):
            mask &= (src[:, :, c] >= lower[c]) & (src[:, :, c] <= upper[c])
    
    return (mask * 255).astype(np.uint8)


# Exports

__all__ = [
    "threshold",
    "adaptiveThreshold",
    "inRange",
    # Threshold types
    "THRESH_BINARY",
    "THRESH_BINARY_INV",
    "THRESH_TRUNC",
    "THRESH_TOZERO",
    "THRESH_TOZERO_INV",
    "THRESH_MASK",
    "THRESH_OTSU",
    "THRESH_TRIANGLE",
    # Adaptive methods
    "ADAPTIVE_THRESH_MEAN_C",
    "ADAPTIVE_THRESH_GAUSSIAN_C",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.