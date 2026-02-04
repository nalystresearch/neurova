# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Template matching for Neurova.

Provides matchTemplate function for finding regions matching a template.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Template matching methods (matching Neurova)
TM_SQDIFF = 0
TM_SQDIFF_NORMED = 1
TM_CCORR = 2
TM_CCORR_NORMED = 3
TM_CCOEFF = 4
TM_CCOEFF_NORMED = 5


def matchTemplate(
    image: np.ndarray,
    templ: np.ndarray,
    method: int,
    result: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compare a template against overlapped image regions.
    
    The function slides through image, compares the overlapped patches
    against template using the specified method, and stores the comparison
    results in result.
    
    Args:
        image: Image where the search is running (8-bit or 32-bit float)
        templ: Searched template. Must be not greater than the source image
        method: Comparison method (TM_SQDIFF, TM_CCORR, TM_CCOEFF, or normalized versions)
        result: Output result map (ignored, computed automatically)
        mask: Optional mask for template (not fully supported)
    
    Returns:
        Result array of size (W-w+1) x (H-h+1) where (W,H) is image size and (w,h) is template size
        
    Note:
        For TM_SQDIFF and TM_SQDIFF_NORMED, best match is minimum value.
        For other methods, best match is maximum value.
    """
    img = np.asarray(image, dtype=np.float64)
    tmpl = np.asarray(templ, dtype=np.float64)
    
    # Handle color images by converting to grayscale
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    if tmpl.ndim == 3:
        tmpl = np.mean(tmpl, axis=2)
    
    h, w = img.shape
    th, tw = tmpl.shape
    
    if th > h or tw > w:
        raise ValueError("Template must be smaller than or equal to image")
    
    # Result size
    rh = h - th + 1
    rw = w - tw + 1
    
    if method == TM_SQDIFF:
        return _tm_sqdiff(img, tmpl, rh, rw)
    
    elif method == TM_SQDIFF_NORMED:
        return _tm_sqdiff_normed(img, tmpl, rh, rw)
    
    elif method == TM_CCORR:
        return _tm_ccorr(img, tmpl, rh, rw)
    
    elif method == TM_CCORR_NORMED:
        return _tm_ccorr_normed(img, tmpl, rh, rw)
    
    elif method == TM_CCOEFF:
        return _tm_ccoeff(img, tmpl, rh, rw)
    
    elif method == TM_CCOEFF_NORMED:
        return _tm_ccoeff_normed(img, tmpl, rh, rw)
    
    else:
        raise ValueError(f"Unknown template matching method: {method}")


def _tm_sqdiff(img: np.ndarray, tmpl: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Sum of squared differences."""
    th, tw = tmpl.shape
    result = np.zeros((rh, rw), dtype=np.float64)
    
    for i in range(rh):
        for j in range(rw):
            patch = img[i:i+th, j:j+tw]
            diff = patch - tmpl
            result[i, j] = np.sum(diff * diff)
    
    return result


def _tm_sqdiff_normed(img: np.ndarray, tmpl: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Normalized sum of squared differences."""
    th, tw = tmpl.shape
    tmpl_norm = np.sqrt(np.sum(tmpl * tmpl))
    result = np.zeros((rh, rw), dtype=np.float64)
    
    for i in range(rh):
        for j in range(rw):
            patch = img[i:i+th, j:j+tw]
            diff = patch - tmpl
            patch_norm = np.sqrt(np.sum(patch * patch))
            denom = patch_norm * tmpl_norm
            if denom > 1e-10:
                result[i, j] = np.sum(diff * diff) / (denom * denom)
            else:
                result[i, j] = 1.0
    
    return result


def _tm_ccorr(img: np.ndarray, tmpl: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Cross-correlation."""
    th, tw = tmpl.shape
    
    if HAS_SCIPY:
        # Use scipy for faster computation
        result = signal.correlate2d(img, tmpl, mode='valid')
        return result
    
    result = np.zeros((rh, rw), dtype=np.float64)
    for i in range(rh):
        for j in range(rw):
            patch = img[i:i+th, j:j+tw]
            result[i, j] = np.sum(patch * tmpl)
    
    return result


def _tm_ccorr_normed(img: np.ndarray, tmpl: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Normalized cross-correlation."""
    th, tw = tmpl.shape
    tmpl_norm = np.sqrt(np.sum(tmpl * tmpl))
    result = np.zeros((rh, rw), dtype=np.float64)
    
    for i in range(rh):
        for j in range(rw):
            patch = img[i:i+th, j:j+tw]
            patch_norm = np.sqrt(np.sum(patch * patch))
            denom = patch_norm * tmpl_norm
            if denom > 1e-10:
                result[i, j] = np.sum(patch * tmpl) / denom
            else:
                result[i, j] = 0.0
    
    return result


def _tm_ccoeff(img: np.ndarray, tmpl: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Correlation coefficient."""
    th, tw = tmpl.shape
    tmpl_mean = np.mean(tmpl)
    tmpl_centered = tmpl - tmpl_mean
    result = np.zeros((rh, rw), dtype=np.float64)
    
    for i in range(rh):
        for j in range(rw):
            patch = img[i:i+th, j:j+tw]
            patch_mean = np.mean(patch)
            patch_centered = patch - patch_mean
            result[i, j] = np.sum(patch_centered * tmpl_centered)
    
    return result


def _tm_ccoeff_normed(img: np.ndarray, tmpl: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Normalized correlation coefficient (most robust)."""
    th, tw = tmpl.shape
    tmpl_mean = np.mean(tmpl)
    tmpl_centered = tmpl - tmpl_mean
    tmpl_norm = np.sqrt(np.sum(tmpl_centered * tmpl_centered))
    result = np.zeros((rh, rw), dtype=np.float64)
    
    for i in range(rh):
        for j in range(rw):
            patch = img[i:i+th, j:j+tw]
            patch_mean = np.mean(patch)
            patch_centered = patch - patch_mean
            patch_norm = np.sqrt(np.sum(patch_centered * patch_centered))
            denom = patch_norm * tmpl_norm
            if denom > 1e-10:
                result[i, j] = np.sum(patch_centered * tmpl_centered) / denom
            else:
                result[i, j] = 0.0
    
    return result


def minMaxLoc(src: np.ndarray, mask: Optional[np.ndarray] = None):
    """Find global minimum and maximum in an array.
    
    Args:
        src: Input array
        mask: Optional mask (ignored for now)
    
    Returns:
        Tuple of (minVal, maxVal, minLoc, maxLoc)
        where Loc is (x, y) coordinate
    """
    arr = np.asarray(src)
    
    # Flatten for finding locations
    if arr.ndim > 2:
        arr = arr.reshape(-1)
    
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    
    # Find locations
    min_idx = np.unravel_index(np.argmin(arr), arr.shape)
    max_idx = np.unravel_index(np.argmax(arr), arr.shape)
    
    # Neurova returns (x, y) = (col, row)
    if len(min_idx) == 2:
        min_loc = (int(min_idx[1]), int(min_idx[0]))
        max_loc = (int(max_idx[1]), int(max_idx[0]))
    else:
        min_loc = (int(min_idx[0]), 0)
        max_loc = (int(max_idx[0]), 0)
    
    return min_val, max_val, min_loc, max_loc


__all__ = [
    "matchTemplate",
    "minMaxLoc",
    "TM_SQDIFF",
    "TM_SQDIFF_NORMED",
    "TM_CCORR",
    "TM_CCORR_NORMED",
    "TM_CCOEFF",
    "TM_CCOEFF_NORMED",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.