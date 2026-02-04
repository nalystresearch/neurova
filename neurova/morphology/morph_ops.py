# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Neurova morphological operations for Neurova.

Provides erode, dilate, morphologyEx, and getStructuringElement with
Neurova signatures that work on grayscale (not just binary) images.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Literal

import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Morphology shape constants (matching Neurova)
MORPH_RECT = 0
MORPH_CROSS = 1
MORPH_ELLIPSE = 2

# Morphology operation constants
MORPH_ERODE = 0
MORPH_DILATE = 1
MORPH_OPEN = 2
MORPH_CLOSE = 3
MORPH_GRADIENT = 4
MORPH_TOPHAT = 5
MORPH_BLACKHAT = 6
MORPH_HITMISS = 7


def getStructuringElement(
    shape: int,
    ksize: Tuple[int, int],
    anchor: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Create a structuring element for morphological operations.
    
    Args:
        shape: Element shape (MORPH_RECT, MORPH_CROSS, or MORPH_ELLIPSE)
        ksize: Size of the structuring element (width, height)
        anchor: Anchor position (ignored, for compatibility)
    
    Returns:
        Structuring element as numpy array
    """
    kw, kh = ksize
    
    if shape == MORPH_RECT:
        return np.ones((kh, kw), dtype=np.uint8)
    
    elif shape == MORPH_CROSS:
        kernel = np.zeros((kh, kw), dtype=np.uint8)
        kernel[kh // 2, :] = 1
        kernel[:, kw // 2] = 1
        return kernel
    
    elif shape == MORPH_ELLIPSE:
        yy, xx = np.meshgrid(np.arange(kh), np.arange(kw), indexing="ij")
        cy = (kh - 1) / 2.0
        cx = (kw - 1) / 2.0
        ry = kh / 2.0
        rx = kw / 2.0
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        return mask.astype(np.uint8)
    
    else:
        # Default to rectangular
        return np.ones((kh, kw), dtype=np.uint8)


def erode(
    src: np.ndarray,
    kernel: np.ndarray,
    dst: Optional[np.ndarray] = None,
    anchor: Optional[Tuple[int, int]] = None,
    iterations: int = 1,
    borderType: int = 0,
    borderValue: Union[int, float, Tuple] = 0
) -> np.ndarray:
    """Erode an image using a structuring element.
    
    Args:
        src: Input image (grayscale or color)
        kernel: Structuring element used for erosion
        dst: Output image (ignored, for compatibility)
        anchor: Anchor position within the kernel (ignored)
        iterations: Number of times erosion is applied
        borderType: Pixel extrapolation method (ignored)
        borderValue: Border value (ignored)
    
    Returns:
        Eroded image
    """
    img = np.asarray(src)
    k = np.asarray(kernel)
    
    # Handle color images
    if img.ndim == 3:
        channels = [erode(img[:, :, c], kernel, iterations=iterations) 
                    for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    
    result = img.copy()
    
    if HAS_SCIPY:
        for _ in range(iterations):
            result = ndimage.grey_erosion(result, footprint=k)
    else:
        # Pure numpy fallback using min filter approach
        for _ in range(iterations):
            result = _min_filter(result, k)
    
    return result.astype(img.dtype)


def dilate(
    src: np.ndarray,
    kernel: np.ndarray,
    dst: Optional[np.ndarray] = None,
    anchor: Optional[Tuple[int, int]] = None,
    iterations: int = 1,
    borderType: int = 0,
    borderValue: Union[int, float, Tuple] = 0
) -> np.ndarray:
    """Dilate an image using a structuring element.
    
    Args:
        src: Input image (grayscale or color)
        kernel: Structuring element used for dilation
        dst: Output image (ignored, for compatibility)
        anchor: Anchor position within the kernel (ignored)
        iterations: Number of times dilation is applied
        borderType: Pixel extrapolation method (ignored)
        borderValue: Border value (ignored)
    
    Returns:
        Dilated image
    """
    img = np.asarray(src)
    k = np.asarray(kernel)
    
    # Handle color images
    if img.ndim == 3:
        channels = [dilate(img[:, :, c], kernel, iterations=iterations) 
                    for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    
    result = img.copy()
    
    if HAS_SCIPY:
        for _ in range(iterations):
            result = ndimage.grey_dilation(result, footprint=k)
    else:
        # Pure numpy fallback using max filter approach
        for _ in range(iterations):
            result = _max_filter(result, k)
    
    return result.astype(img.dtype)


def morphologyEx(
    src: np.ndarray,
    op: int,
    kernel: np.ndarray,
    dst: Optional[np.ndarray] = None,
    anchor: Optional[Tuple[int, int]] = None,
    iterations: int = 1,
    borderType: int = 0,
    borderValue: Union[int, float, Tuple] = 0
) -> np.ndarray:
    """Perform advanced morphological operations.
    
    Args:
        src: Input image
        op: Type of morphological operation (MORPH_*)
        kernel: Structuring element
        dst: Output image (ignored, for compatibility)
        anchor: Anchor position (ignored)
        iterations: Number of iterations
        borderType: Border type (ignored)
        borderValue: Border value (ignored)
    
    Returns:
        Result of morphological operation
    """
    if op == MORPH_ERODE:
        return erode(src, kernel, iterations=iterations)
    
    elif op == MORPH_DILATE:
        return dilate(src, kernel, iterations=iterations)
    
    elif op == MORPH_OPEN:
        # Opening = erosion followed by dilation
        temp = erode(src, kernel, iterations=iterations)
        return dilate(temp, kernel, iterations=iterations)
    
    elif op == MORPH_CLOSE:
        # Closing = dilation followed by erosion
        temp = dilate(src, kernel, iterations=iterations)
        return erode(temp, kernel, iterations=iterations)
    
    elif op == MORPH_GRADIENT:
        # Gradient = dilation - erosion
        dil = dilate(src, kernel, iterations=iterations)
        ero = erode(src, kernel, iterations=iterations)
        return np.clip(dil.astype(np.int16) - ero.astype(np.int16), 0, 255).astype(src.dtype)
    
    elif op == MORPH_TOPHAT:
        # Top hat = src - opening
        opened = morphologyEx(src, MORPH_OPEN, kernel, iterations=iterations)
        return np.clip(src.astype(np.int16) - opened.astype(np.int16), 0, 255).astype(src.dtype)
    
    elif op == MORPH_BLACKHAT:
        # Black hat = closing - src
        closed = morphologyEx(src, MORPH_CLOSE, kernel, iterations=iterations)
        return np.clip(closed.astype(np.int16) - src.astype(np.int16), 0, 255).astype(src.dtype)
    
    elif op == MORPH_HITMISS:
        # Hit-or-miss transform (simplified)
        return erode(src, kernel, iterations=iterations)
    
    else:
        raise ValueError(f"Unknown morphological operation: {op}")


def _min_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply minimum filter (erosion) using pure numpy."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', 
                    constant_values=np.max(image))
    
    result = np.zeros_like(image)
    
    # Get kernel positions
    kernel_mask = kernel > 0
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.min(region[kernel_mask])
    
    return result


def _max_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply maximum filter (dilation) using pure numpy."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant',
                    constant_values=0)
    
    result = np.zeros_like(image)
    
    # Get kernel positions
    kernel_mask = kernel > 0
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.max(region[kernel_mask])
    
    return result


__all__ = [
    # Functions
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