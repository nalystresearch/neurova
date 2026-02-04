# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Segmentation functions for Neurova.

Provides floodFill, watershed, grabCut, and distanceTransform.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# FloodFill flags
FLOODFILL_FIXED_RANGE = 1 << 16
FLOODFILL_MASK_ONLY = 1 << 17

# Distance types
DIST_USER = -1
DIST_L1 = 1
DIST_L2 = 2
DIST_C = 3
DIST_L12 = 4
DIST_FAIR = 5
DIST_WELSCH = 6
DIST_HUBER = 7

# Distance mask sizes
DIST_MASK_3 = 3
DIST_MASK_5 = 5
DIST_MASK_PRECISE = 0

# GrabCut modes
GC_INIT_WITH_RECT = 0
GC_INIT_WITH_MASK = 1
GC_EVAL = 2
GC_EVAL_FREEZE_MODEL = 3

# GrabCut mask values
GC_BGD = 0
GC_FGD = 1
GC_PR_BGD = 2
GC_PR_FGD = 3


def floodFill(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    seedPoint: Tuple[int, int],
    newVal: Union[int, Tuple[int, ...]],
    loDiff: Union[int, Tuple[int, ...]] = 0,
    upDiff: Union[int, Tuple[int, ...]] = 0,
    flags: int = 4
) -> Tuple[int, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Fill a connected component starting from the seed point.
    
    Args:
        image: Input/output 1- or 3-channel, 8-bit, or floating-point image
        mask: Optional mask (h+2 x w+2). If None, created automatically
        seedPoint: Starting point (x, y)
        newVal: New value of the repainted domain pixels
        loDiff: Maximum lower brightness/color difference
        upDiff: Maximum upper brightness/color difference
        flags: Operation flags (connectivity, etc.)
    
    Returns:
        Tuple of (num_pixels_filled, modified_image, mask, bounding_rect)
    """
    img = np.asarray(image).copy()
    h, w = img.shape[:2]
    
    # Create mask if not provided
    if mask is None:
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # Extract connectivity from flags
    connectivity = flags & 0xFF
    if connectivity not in (4, 8):
        connectivity = 4
    
    # Convert scalar differences to tuples
    if isinstance(loDiff, (int, float)):
        loDiff = (loDiff,) * (img.shape[2] if img.ndim == 3 else 1)
    if isinstance(upDiff, (int, float)):
        upDiff = (upDiff,) * (img.shape[2] if img.ndim == 3 else 1)
    if isinstance(newVal, (int, float)):
        newVal = (int(newVal),) * (img.shape[2] if img.ndim == 3 else 1)
    
    # Seed point
    x, y = seedPoint
    if not (0 <= x < w and 0 <= y < h):
        return 0, img, mask, (0, 0, 0, 0)
    
    # Get seed value
    if img.ndim == 3:
        seed_val = img[y, x, :].astype(np.float32)
    else:
        seed_val = float(img[y, x])
    
    # BFS flood fill
    visited = np.zeros((h, w), dtype=bool)
    queue = [(x, y)]
    visited[y, x] = True
    
    filled_pixels = []
    min_x, min_y, max_x, max_y = x, y, x, y
    
    while queue:
        cx, cy = queue.pop(0)
        
        # Check if pixel is within tolerance
        if img.ndim == 3:
            pixel_val = img[cy, cx, :].astype(np.float32)
            in_range = True
            for c in range(img.shape[2]):
                if not (seed_val[c] - loDiff[c] <= pixel_val[c] <= seed_val[c] + upDiff[c]):
                    in_range = False
                    break
        else:
            pixel_val = float(img[cy, cx])
            lo = loDiff[0] if isinstance(loDiff, tuple) else loDiff
            up = upDiff[0] if isinstance(upDiff, tuple) else upDiff
            in_range = (seed_val - lo <= pixel_val <= seed_val + up)
        
        if not in_range and (cx, cy) != seedPoint:
            continue
        
        filled_pixels.append((cx, cy))
        min_x, min_y = min(min_x, cx), min(min_y, cy)
        max_x, max_y = max(max_x, cx), max(max_y, cy)
        mask[cy + 1, cx + 1] = 1
        
        # Get neighbors based on connectivity
        if connectivity == 4:
            neighbors = [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
        else:  # 8-connectivity
            neighbors = [
                (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1),
                (cx-1, cy-1), (cx+1, cy-1), (cx-1, cy+1), (cx+1, cy+1)
            ]
        
        for nx, ny in neighbors:
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))
    
    # Fill pixels with new value
    if not (flags & FLOODFILL_MASK_ONLY):
        for fx, fy in filled_pixels:
            if img.ndim == 3:
                for c in range(img.shape[2]):
                    img[fy, fx, c] = newVal[c] if c < len(newVal) else newVal[-1]
            else:
                img[fy, fx] = newVal[0] if isinstance(newVal, tuple) else newVal
    
    rect = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    
    return len(filled_pixels), img, mask, rect


def distanceTransform(
    src: np.ndarray,
    distanceType: int,
    maskSize: int,
    dst: Optional[np.ndarray] = None,
    dstType: int = 5  # CV_32F
) -> np.ndarray:
    """Calculate distance to closest zero pixel for each pixel.
    
    Args:
        src: 8-bit single-channel binary source image
        distanceType: Type of distance (DIST_L1, DIST_L2, DIST_C)
        maskSize: Size of distance transform mask (3, 5, or DIST_MASK_PRECISE)
        dst: Output image (ignored)
        dstType: Output type (ignored, always float32)
    
    Returns:
        Distance transform of the source image
    """
    img = np.asarray(src)
    
    if img.ndim != 2:
        raise ValueError("distanceTransform requires a 2D binary image")
    
    # Binarize
    binary = (img > 0).astype(np.uint8)
    
    if HAS_SCIPY:
        if distanceType == DIST_L1:
            # Manhattan distance
            result = ndimage.distance_transform_cdt(binary, metric='taxicab')
        elif distanceType == DIST_C:
            # Chessboard distance
            result = ndimage.distance_transform_cdt(binary, metric='chessboard')
        else:
            # Euclidean distance (default)
            result = ndimage.distance_transform_edt(binary)
    else:
        # Simple fallback using erosion-based approach
        result = _distance_transform_simple(binary, distanceType)
    
    return result.astype(np.float32)


def distanceTransformWithLabels(
    src: np.ndarray,
    distanceType: int,
    maskSize: int,
    labelType: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate distance transform with Voronoi labels.
    
    Args:
        src: 8-bit single-channel binary source image
        distanceType: Type of distance
        maskSize: Mask size
        labelType: Label type (ignored)
    
    Returns:
        Tuple of (distance_image, labels)
    """
    dist = distanceTransform(src, distanceType, maskSize)
    
    binary = (np.asarray(src) > 0).astype(np.uint8)
    
    if HAS_SCIPY:
        labels = ndimage.label(1 - binary)[0]
    else:
        labels = np.zeros_like(binary, dtype=np.int32)
    
    return dist, labels


def watershed(
    image: np.ndarray,
    markers: np.ndarray
) -> np.ndarray:
    """Perform marker-based watershed segmentation.
    
    Args:
        image: Input 8-bit 3-channel image
        markers: Input/output 32-bit single-channel marker image.
                 Markers are transformed in-place.
    
    Returns:
        Modified markers array with watershed boundaries marked as -1
    """
    img = np.asarray(image)
    marks = np.asarray(markers).copy()
    
    if img.ndim != 3:
        raise ValueError("watershed requires a 3-channel image")
    
    if HAS_SCIPY:
        from scipy import ndimage as ndi
        
        # Convert to grayscale for gradient
        gray = np.mean(img, axis=2)
        
        # Compute gradient magnitude
        sx = ndi.sobel(gray, axis=1)
        sy = ndi.sobel(gray, axis=0)
        gradient = np.sqrt(sx**2 + sy**2)
        
        # Use scipy's watershed
        result = ndi.watershed_ift(gradient.astype(np.uint8), marks.astype(np.int32))
        
        # Mark boundaries as -1
        boundary = np.zeros_like(result, dtype=bool)
        boundary[:-1, :] |= (result[:-1, :] != result[1:, :])
        boundary[:, :-1] |= (result[:, :-1] != result[:, 1:])
        
        result[boundary] = -1
        
        return result.astype(np.int32)
    else:
        # Simplified watershed using region growing
        return _watershed_simple(img, marks)


def grabCut(
    img: np.ndarray,
    mask: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]],
    bgdModel: np.ndarray,
    fgdModel: np.ndarray,
    iterCount: int,
    mode: int = GC_EVAL
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run GrabCut algorithm for foreground extraction.
    
    Args:
        img: Input 8-bit 3-channel image
        mask: Input/output mask (GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD)
        rect: ROI containing foreground (x, y, width, height) for GC_INIT_WITH_RECT
        bgdModel: Temporary background model array
        fgdModel: Temporary foreground model array
        iterCount: Number of iterations
        mode: GrabCut mode
    
    Returns:
        Tuple of (mask, bgdModel, fgdModel)
    """
    image = np.asarray(img)
    mask_out = np.asarray(mask).copy()
    
    if image.ndim != 3:
        raise ValueError("grabCut requires a 3-channel image")
    
    h, w = image.shape[:2]
    
    # Initialize mask from rectangle
    if mode == GC_INIT_WITH_RECT and rect is not None:
        x, y, rw, rh = rect
        mask_out[:] = GC_BGD
        mask_out[y:y+rh, x:x+rw] = GC_PR_FGD
    
    # Simplified GrabCut using color clustering
    # This is a simplified implementation - real GrabCut uses GMM
    for _ in range(iterCount):
        # Get foreground/background pixels
        fg_mask = (mask_out == GC_FGD) | (mask_out == GC_PR_FGD)
        bg_mask = (mask_out == GC_BGD) | (mask_out == GC_PR_BGD)
        
        if not np.any(fg_mask) or not np.any(bg_mask):
            break
        
        # Compute mean colors
        fg_pixels = image[fg_mask].astype(np.float64)
        bg_pixels = image[bg_mask].astype(np.float64)
        
        if len(fg_pixels) > 0:
            fg_mean = np.mean(fg_pixels, axis=0)
        else:
            fg_mean = np.array([128, 128, 128], dtype=np.float64)
        
        if len(bg_pixels) > 0:
            bg_mean = np.mean(bg_pixels, axis=0)
        else:
            bg_mean = np.array([0, 0, 0], dtype=np.float64)
        
        # Update probable regions based on color distance
        pr_mask = (mask_out == GC_PR_FGD) | (mask_out == GC_PR_BGD)
        
        for i in range(h):
            for j in range(w):
                if not pr_mask[i, j]:
                    continue
                
                pixel = image[i, j].astype(np.float64)
                fg_dist = np.sum((pixel - fg_mean) ** 2)
                bg_dist = np.sum((pixel - bg_mean) ** 2)
                
                if fg_dist < bg_dist:
                    mask_out[i, j] = GC_PR_FGD
                else:
                    mask_out[i, j] = GC_PR_BGD
    
    return mask_out, bgdModel, fgdModel


def pyrMeanShiftFiltering(
    src: np.ndarray,
    sp: float,
    sr: float,
    dst: Optional[np.ndarray] = None,
    maxLevel: int = 1,
    termcrit: Optional[Tuple[int, int, float]] = None
) -> np.ndarray:
    """Perform initial step of meanshift segmentation.
    
    Args:
        src: Input 8-bit 3-channel image
        sp: Spatial window radius
        sr: Color window radius
        dst: Output image (ignored)
        maxLevel: Maximum pyramid level
        termcrit: Termination criteria (ignored)
    
    Returns:
        Filtered image
    """
    img = np.asarray(src, dtype=np.float64)
    
    if img.ndim != 3:
        raise ValueError("pyrMeanShiftFiltering requires a 3-channel image")
    
    h, w, c = img.shape
    result = img.copy()
    
    # Simplified mean shift filtering
    for _ in range(5):  # iterations
        new_result = result.copy()
        
        for i in range(0, h, max(1, int(sp/2))):
            for j in range(0, w, max(1, int(sp/2))):
                # Define spatial window
                y0 = max(0, int(i - sp))
                y1 = min(h, int(i + sp + 1))
                x0 = max(0, int(j - sp))
                x1 = min(w, int(j + sp + 1))
                
                center_color = result[i, j]
                
                # Find mean of similar colors
                window = result[y0:y1, x0:x1]
                color_diff = np.sqrt(np.sum((window - center_color)**2, axis=2))
                similar = color_diff < sr
                
                if np.any(similar):
                    mean_color = np.mean(window[similar], axis=0)
                    new_result[i, j] = mean_color
        
        result = new_result
    
    return np.clip(result, 0, 255).astype(np.uint8)


def _distance_transform_simple(binary: np.ndarray, dist_type: int) -> np.ndarray:
    """Simple distance transform using iterative erosion."""
    h, w = binary.shape
    dist = np.zeros((h, w), dtype=np.float32)
    
    remaining = binary.copy()
    distance = 0
    
    while np.any(remaining):
        dist[remaining > 0] = distance
        
        # Erode
        eroded = np.zeros_like(remaining)
        for i in range(1, h-1):
            for j in range(1, w-1):
                if remaining[i, j]:
                    eroded[i, j] = (remaining[i-1, j] and remaining[i+1, j] and
                                    remaining[i, j-1] and remaining[i, j+1])
        
        remaining = eroded
        distance += 1
    
    return dist


def _watershed_simple(img: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Simple watershed implementation."""
    # Just return markers for now - full implementation would be complex
    return markers.astype(np.int32)


__all__ = [
    # Functions
    "floodFill",
    "distanceTransform",
    "distanceTransformWithLabels",
    "watershed",
    "grabCut",
    "pyrMeanShiftFiltering",
    # FloodFill flags
    "FLOODFILL_FIXED_RANGE",
    "FLOODFILL_MASK_ONLY",
    # Distance types
    "DIST_USER",
    "DIST_L1",
    "DIST_L2",
    "DIST_C",
    "DIST_L12",
    "DIST_FAIR",
    "DIST_WELSCH",
    "DIST_HUBER",
    # Distance masks
    "DIST_MASK_3",
    "DIST_MASK_5",
    "DIST_MASK_PRECISE",
    # GrabCut modes
    "GC_INIT_WITH_RECT",
    "GC_INIT_WITH_MASK",
    "GC_EVAL",
    "GC_EVAL_FREEZE_MODEL",
    # GrabCut mask values
    "GC_BGD",
    "GC_FGD",
    "GC_PR_BGD",
    "GC_PR_FGD",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.