# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova.imgproc.histogram - Histogram operations

Provides Neurova histogram functions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np


# Histogram comparison methods
HISTCMP_CORREL = 0
HISTCMP_CHISQR = 1
HISTCMP_INTERSECT = 2
HISTCMP_BHATTACHARYYA = 3
HISTCMP_HELLINGER = HISTCMP_BHATTACHARYYA
HISTCMP_CHISQR_ALT = 4
HISTCMP_KL_DIV = 5


def calcHist(
    images: List[np.ndarray],
    channels: List[int],
    mask: Optional[np.ndarray],
    histSize: List[int],
    ranges: List[float],
    hist: Optional[np.ndarray] = None,
    accumulate: bool = False
) -> np.ndarray:
    """Calculate histogram of image(s).
    
    Args:
        images: Source images (must have same depth and size)
        channels: List of channel indices to compute histogram for
        mask: Optional mask (8-bit)
        histSize: Number of bins for each channel
        ranges: Ranges for each channel [min1, max1, min2, max2, ...]
        hist: Optional output histogram
        accumulate: If True, accumulate into hist
    
    Returns:
        Histogram array
    """
    # Handle single image case
    if not isinstance(images, (list, tuple)):
        images = [images]
    
    num_channels = len(channels)
    
    # Parse ranges
    channel_ranges = []
    for i in range(num_channels):
        channel_ranges.append((ranges[i*2], ranges[i*2 + 1]))
    
    # Extract channels
    channel_data = []
    for ch in channels:
        # Find which image and which channel within that image
        img_idx = 0
        ch_offset = ch
        
        for img in images:
            if img.ndim == 2:
                num_ch = 1
            else:
                num_ch = img.shape[2]
            
            if ch_offset < num_ch:
                if img.ndim == 2:
                    data = img.flatten()
                else:
                    data = img[:, :, ch_offset].flatten()
                break
            else:
                ch_offset -= num_ch
                img_idx += 1
        
        if mask is not None:
            data = data[mask.flatten() > 0]
        
        channel_data.append(data)
    
    if num_channels == 1:
        # 1D histogram
        result, _ = np.histogram(
            channel_data[0],
            bins=histSize[0],
            range=channel_ranges[0]
        )
        result = result.astype(np.float32).reshape(-1, 1)
    else:
        # Multi-dimensional histogram
        result, _ = np.histogramdd(
            np.column_stack(channel_data),
            bins=histSize,
            range=channel_ranges
        )
        result = result.astype(np.float32)
    
    if accumulate and hist is not None:
        result = hist + result
    
    if hist is not None:
        np.copyto(hist, result)
        return hist
    
    return result


def compareHist(
    H1: np.ndarray,
    H2: np.ndarray,
    method: int
) -> float:
    """Compare two histograms.
    
    Args:
        H1: First histogram
        H2: Second histogram
        method: Comparison method (HISTCMP_*)
    
    Returns:
        Comparison result
    """
    H1 = H1.flatten().astype(np.float64)
    H2 = H2.flatten().astype(np.float64)
    
    if method == HISTCMP_CORREL:
        # Correlation
        mean1 = np.mean(H1)
        mean2 = np.mean(H2)
        
        H1_centered = H1 - mean1
        H2_centered = H2 - mean2
        
        num = np.sum(H1_centered * H2_centered)
        denom = np.sqrt(np.sum(H1_centered**2) * np.sum(H2_centered**2))
        
        if denom == 0:
            return 0.0
        return float(num / denom)
    
    elif method == HISTCMP_CHISQR:
        # Chi-Square
        denom = H1 + 1e-10
        return float(np.sum((H1 - H2)**2 / denom))
    
    elif method == HISTCMP_INTERSECT:
        # Intersection
        return float(np.sum(np.minimum(H1, H2)))
    
    elif method == HISTCMP_BHATTACHARYYA:
        # Bhattacharyya distance
        H1_norm = H1 / (np.sum(H1) + 1e-10)
        H2_norm = H2 / (np.sum(H2) + 1e-10)
        
        bc = np.sum(np.sqrt(H1_norm * H2_norm))
        return float(np.sqrt(1 - bc))
    
    elif method == HISTCMP_CHISQR_ALT:
        # Alternative Chi-Square
        denom = H1 + H2 + 1e-10
        return float(2 * np.sum((H1 - H2)**2 / denom))
    
    elif method == HISTCMP_KL_DIV:
        # Kullback-Leibler divergence
        H1_norm = H1 / (np.sum(H1) + 1e-10)
        H2_norm = H2 / (np.sum(H2) + 1e-10)
        
        # Avoid log(0)
        mask = H1_norm > 0
        result = np.sum(H1_norm[mask] * np.log(H1_norm[mask] / (H2_norm[mask] + 1e-10)))
        return float(result)
    
    else:
        raise ValueError(f"Unknown comparison method: {method}")


def equalizeHist(src: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Equalize the histogram of a grayscale image.
    
    Args:
        src: Source 8-bit single channel image
        dst: Optional destination array
    
    Returns:
        Equalized image
    """
    if src.ndim != 2:
        raise ValueError("Input must be single-channel")
    
    # Compute histogram
    hist, _ = np.histogram(src.flatten(), bins=256, range=(0, 256))
    
    # Compute CDF
    cdf = hist.cumsum()
    
    # Normalize CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-10)
    cdf_normalized = cdf_normalized.astype(np.uint8)
    
    # Apply transform
    result = cdf_normalized[src]
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def calcBackProject(
    images: List[np.ndarray],
    channels: List[int],
    hist: np.ndarray,
    ranges: List[float],
    scale: float = 1.0
) -> np.ndarray:
    """Calculate back projection of a histogram.
    
    Args:
        images: Source images
        channels: Channel indices
        hist: Histogram to back-project
        ranges: Ranges for each channel
        scale: Scale factor
    
    Returns:
        Back projection image
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
    
    # Get image dimensions
    h, w = images[0].shape[:2]
    
    num_channels = len(channels)
    hist_shape = hist.shape
    
    # Extract channel data
    channel_images = []
    for ch in channels:
        img = images[0]
        if img.ndim == 2:
            channel_images.append(img)
        else:
            channel_images.append(img[:, :, ch])
    
    result = np.zeros((h, w), dtype=np.float32)
    
    if num_channels == 1:
        # 1D back projection
        bins = hist_shape[0]
        range_min, range_max = ranges[0], ranges[1]
        
        # Map pixel values to histogram bins
        img_scaled = (channel_images[0].astype(np.float32) - range_min) / (range_max - range_min) * bins
        img_scaled = np.clip(img_scaled, 0, bins - 1).astype(np.int32)
        
        result = hist.flatten()[img_scaled] * scale
    else:
        # Multi-dimensional back projection
        bins = list(hist_shape)
        
        # Map each channel to bins
        bin_indices = []
        for i, ch_img in enumerate(channel_images):
            range_min, range_max = ranges[i*2], ranges[i*2 + 1]
            scaled = (ch_img.astype(np.float32) - range_min) / (range_max - range_min) * bins[i]
            scaled = np.clip(scaled, 0, bins[i] - 1).astype(np.int32)
            bin_indices.append(scaled)
        
        # Look up in histogram
        if num_channels == 2:
            result = hist[bin_indices[0], bin_indices[1]] * scale
        elif num_channels == 3:
            result = hist[bin_indices[0], bin_indices[1], bin_indices[2]] * scale
    
    return result.astype(np.uint8)


def CLAHE_create(clipLimit: float = 40.0, tileGridSize: Tuple[int, int] = (8, 8)):
    """Create a CLAHE (Contrast Limited Adaptive Histogram Equalization) object.
    
    Args:
        clipLimit: Threshold for contrast limiting
        tileGridSize: Size of grid for histogram equalization
    
    Returns:
        CLAHE object
    """
    return CLAHE(clipLimit, tileGridSize)


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization."""
    
    def __init__(
        self,
        clipLimit: float = 40.0,
        tileGridSize: Tuple[int, int] = (8, 8)
    ):
        self._clipLimit = clipLimit
        self._tileGridSize = tileGridSize
    
    def apply(self, src: np.ndarray, dst: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply CLAHE to an image.
        
        Args:
            src: Source 8-bit single channel image
            dst: Optional destination
        
        Returns:
            Enhanced image
        """
        if src.ndim != 2:
            raise ValueError("Input must be single-channel")
        
        h, w = src.shape
        tile_h, tile_w = h // self._tileGridSize[1], w // self._tileGridSize[0]
        
        result = np.zeros_like(src)
        
        # Process each tile
        for ty in range(self._tileGridSize[1]):
            for tx in range(self._tileGridSize[0]):
                y1 = ty * tile_h
                y2 = (ty + 1) * tile_h if ty < self._tileGridSize[1] - 1 else h
                x1 = tx * tile_w
                x2 = (tx + 1) * tile_w if tx < self._tileGridSize[0] - 1 else w
                
                tile = src[y1:y2, x1:x2]
                
                # Compute histogram
                hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
                
                # Clip histogram
                clip_limit = int(self._clipLimit * tile.size / 256)
                excess = np.sum(np.maximum(hist - clip_limit, 0))
                hist = np.minimum(hist, clip_limit)
                hist = hist + excess // 256
                
                # Compute CDF
                cdf = hist.cumsum()
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-10)
                cdf_normalized = cdf_normalized.astype(np.uint8)
                
                result[y1:y2, x1:x2] = cdf_normalized[tile]
        
        if dst is not None:
            np.copyto(dst, result)
            return dst
        
        return result
    
    def getClipLimit(self) -> float:
        return self._clipLimit
    
    def setClipLimit(self, clipLimit: float):
        self._clipLimit = clipLimit
    
    def getTilesGridSize(self) -> Tuple[int, int]:
        return self._tileGridSize
    
    def setTilesGridSize(self, tileGridSize: Tuple[int, int]):
        self._tileGridSize = tileGridSize


def createCLAHE(
    clipLimit: float = 40.0,
    tileGridSize: Tuple[int, int] = (8, 8)
) -> CLAHE:
    """Create CLAHE object.
    
    Args:
        clipLimit: Threshold for contrast limiting
        tileGridSize: Grid size
    
    Returns:
        CLAHE object
    """
    return CLAHE(clipLimit, tileGridSize)


__all__ = [
    "calcHist",
    "compareHist",
    "equalizeHist",
    "calcBackProject",
    "CLAHE",
    "createCLAHE",
    
    # Constants
    "HISTCMP_CORREL",
    "HISTCMP_CHISQR",
    "HISTCMP_INTERSECT",
    "HISTCMP_BHATTACHARYYA",
    "HISTCMP_HELLINGER",
    "HISTCMP_CHISQR_ALT",
    "HISTCMP_KL_DIV",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.