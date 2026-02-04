# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Image pyramid operations for Neurova.

Provides pyrUp, pyrDown and buildPyramid functions for
multi-scale image processing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Border type constants (reuse from geometric if available)
BORDER_DEFAULT = 4


def pyrDown(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    dstsize: Optional[Tuple[int, int]] = None,
    borderType: int = BORDER_DEFAULT
) -> np.ndarray:
    """Blur and downsample an image.
    
    Applies Gaussian blur and then downsamples by factor of 2.
    
    Args:
        src: Input image
        dst: Output image (ignored, for compatibility)
        dstsize: Size of output image. Default is ((cols+1)/2, (rows+1)/2)
        borderType: Border mode (ignored for now)
    
    Returns:
        Downsampled image (approximately half the size)
    """
    img = np.asarray(src)
    
    # Calculate output size
    if dstsize is not None:
        new_width, new_height = dstsize
    else:
        new_height = (img.shape[0] + 1) // 2
        new_width = (img.shape[1] + 1) // 2
    
    # Handle color images
    if img.ndim == 3:
        channels = [pyrDown(img[:, :, c]) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    
    # Apply Gaussian blur first
    blurred = _gaussian_blur_5x5(img)
    
    # Downsample by taking every other pixel
    result = blurred[::2, ::2]
    
    # Resize to exact target size if needed
    if result.shape[0] != new_height or result.shape[1] != new_width:
        result = _simple_resize(result, (new_height, new_width))
    
    return result.astype(img.dtype)


def pyrUp(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    dstsize: Optional[Tuple[int, int]] = None,
    borderType: int = BORDER_DEFAULT
) -> np.ndarray:
    """Upsample and blur an image.
    
    Upsamples by factor of 2 and then applies Gaussian blur.
    
    Args:
        src: Input image
        dst: Output image (ignored, for compatibility)
        dstsize: Size of output image. Default is (cols*2, rows*2)
        borderType: Border mode (ignored for now)
    
    Returns:
        Upsampled image (approximately double the size)
    """
    img = np.asarray(src)
    
    # Calculate output size
    if dstsize is not None:
        new_width, new_height = dstsize
    else:
        new_height = img.shape[0] * 2
        new_width = img.shape[1] * 2
    
    # Handle color images
    if img.ndim == 3:
        channels = [pyrUp(img[:, :, c]) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    
    # Create upsampled image (zeros in between)
    upsampled = np.zeros((new_height, new_width), dtype=np.float64)
    upsampled[::2, ::2] = img[:min(img.shape[0], new_height//2+1), 
                              :min(img.shape[1], new_width//2+1)]
    
    # Apply Gaussian blur (with 4x multiplier to compensate for zeros)
    result = _gaussian_blur_5x5(upsampled) * 4
    
    return np.clip(result, 0, 255).astype(img.dtype) if img.dtype == np.uint8 else result.astype(img.dtype)


def buildPyramid(
    src: np.ndarray,
    maxlevel: int,
    borderType: int = BORDER_DEFAULT
) -> list:
    """Build a Gaussian pyramid.
    
    Args:
        src: Source image
        maxlevel: Maximum pyramid level (0-indexed)
        borderType: Border mode
    
    Returns:
        List of images forming the pyramid (src at index 0)
    """
    pyramid = [src.copy()]
    current = src
    
    for _ in range(maxlevel):
        current = pyrDown(current, borderType=borderType)
        pyramid.append(current)
    
    return pyramid


def _gaussian_blur_5x5(image: np.ndarray) -> np.ndarray:
    """Apply a 5x5 Gaussian blur for pyramid operations."""
    # Standard 5x5 Gaussian kernel (normalized)
    kernel = np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ], dtype=np.float64) / 256.0
    
    if HAS_SCIPY:
        return ndimage.convolve(image.astype(np.float64), kernel, mode='reflect')
    else:
        # Pure numpy convolution fallback
        return _convolve2d(image.astype(np.float64), kernel)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution for fallback."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return result


def _simple_resize(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Simple nearest-neighbor resize."""
    h, w = image.shape
    new_h, new_w = target_shape
    
    row_indices = (np.arange(new_h) * h / new_h).astype(int)
    col_indices = (np.arange(new_w) * w / new_w).astype(int)
    
    return image[row_indices[:, np.newaxis], col_indices]


__all__ = [
    "pyrDown",
    "pyrUp",
    "buildPyramid",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.