# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova.video.optflow - Optical Flow algorithms

Provides Neurova optical flow implementations.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def calcOpticalFlowPyrLK(
    prevImg: np.ndarray,
    nextImg: np.ndarray,
    prevPts: np.ndarray,
    nextPts: Optional[np.ndarray] = None,
    winSize: Tuple[int, int] = (21, 21),
    maxLevel: int = 3,
    criteria: Optional[Tuple] = None,
    flags: int = 0,
    minEigThreshold: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Calculate optical flow using the Lucas-Kanade method with pyramids.
    
    Args:
        prevImg: First 8-bit grayscale image
        nextImg: Second 8-bit grayscale image
        prevPts: Points to track (N, 1, 2) or (N, 2) float32
        nextPts: Optional initial approximation of new points
        winSize: Search window size
        maxLevel: Maximum pyramid level
        criteria: Termination criteria (type, maxCount, epsilon)
        flags: Operation flags
        minEigThreshold: Minimum eigenvalue threshold
    
    Returns:
        Tuple of (nextPts, status, error)
    """
    if prevImg.ndim == 3:
        from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
        prevImg = cvtColor(prevImg, COLOR_BGR2GRAY)
    if nextImg.ndim == 3:
        from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
        nextImg = cvtColor(nextImg, COLOR_BGR2GRAY)
    
    prevPts = np.asarray(prevPts, dtype=np.float32)
    if prevPts.ndim == 3:
        prevPts = prevPts.reshape(-1, 2)
    
    num_pts = len(prevPts)
    
    # Default criteria
    if criteria is None:
        max_iter = 30
        epsilon = 0.01
    else:
        _, max_iter, epsilon = criteria
    
    # Build pyramids
    prev_pyr = _build_pyramid(prevImg, maxLevel)
    next_pyr = _build_pyramid(nextImg, maxLevel)
    
    # Initialize output
    if nextPts is not None:
        nextPts = np.asarray(nextPts, dtype=np.float32).reshape(-1, 2)
        result_pts = nextPts.copy()
    else:
        result_pts = prevPts.copy()
    
    status = np.ones(num_pts, dtype=np.uint8)
    err = np.zeros(num_pts, dtype=np.float32)
    
    win_h, win_w = winSize[1] // 2, winSize[0] // 2
    
    # Track from coarse to fine
    for level in range(maxLevel, -1, -1):
        scale = 2 ** level
        prev_level = prev_pyr[level]
        next_level = next_pyr[level]
        
        # Scale points for this level
        scaled_pts = prevPts / scale
        scaled_result = result_pts / scale
        
        for i in range(num_pts):
            if status[i] == 0:
                continue
            
            x, y = scaled_pts[i]
            dx, dy = scaled_result[i] - scaled_pts[i]
            
            # Extract window from previous image
            x_int, y_int = int(x), int(y)
            
            if (x_int - win_w < 0 or x_int + win_w >= prev_level.shape[1] or
                y_int - win_h < 0 or y_int + win_h >= prev_level.shape[0]):
                status[i] = 0
                continue
            
            template = prev_level[y_int - win_h:y_int + win_h + 1,
                                 x_int - win_w:x_int + win_w + 1].astype(np.float32)
            
            # Compute gradients
            Ix = np.zeros_like(template)
            Iy = np.zeros_like(template)
            Ix[:, 1:-1] = (template[:, 2:] - template[:, :-2]) / 2
            Iy[1:-1, :] = (template[2:, :] - template[:-2, :]) / 2
            
            # Compute structure tensor
            Ixx = np.sum(Ix * Ix)
            Ixy = np.sum(Ix * Iy)
            Iyy = np.sum(Iy * Iy)
            
            # Check minimum eigenvalue
            det = Ixx * Iyy - Ixy * Ixy
            trace = Ixx + Iyy
            
            if trace <= 0:
                status[i] = 0
                continue
            
            min_eig = (trace - np.sqrt(trace**2 - 4*det + 1e-10)) / 2
            if min_eig < minEigThreshold * winSize[0] * winSize[1]:
                status[i] = 0
                continue
            
            # Iterative Lucas-Kanade
            for _ in range(max_iter):
                nx, ny = x + dx, y + dy
                nx_int, ny_int = int(nx), int(ny)
                
                if (nx_int - win_w < 0 or nx_int + win_w >= next_level.shape[1] or
                    ny_int - win_h < 0 or ny_int + win_h >= next_level.shape[0]):
                    status[i] = 0
                    break
                
                search = next_level[ny_int - win_h:ny_int + win_h + 1,
                                   nx_int - win_w:nx_int + win_w + 1].astype(np.float32)
                
                if search.shape != template.shape:
                    status[i] = 0
                    break
                
                # Compute difference
                It = search - template
                
                # Compute flow update
                bx = np.sum(Ix * It)
                by = np.sum(Iy * It)
                
                # Solve 2x2 system
                det = Ixx * Iyy - Ixy * Ixy
                if abs(det) < 1e-10:
                    status[i] = 0
                    break
                
                ddx = -(Iyy * bx - Ixy * by) / det
                ddy = -(-Ixy * bx + Ixx * by) / det
                
                dx += ddx
                dy += ddy
                
                if ddx * ddx + ddy * ddy < epsilon * epsilon:
                    break
            
            # Update result
            if status[i]:
                result_pts[i] = np.array([x + dx, y + dy]) * scale
                err[i] = np.sqrt(np.mean(It**2)) if 'It' in dir() else 0
    
    return (result_pts.reshape(-1, 1, 2), 
            status.reshape(-1, 1), 
            err.reshape(-1, 1))


def calcOpticalFlowFarneback(
    prev: np.ndarray,
    next: np.ndarray,
    flow: Optional[np.ndarray],
    pyr_scale: float,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
    flags: int
) -> np.ndarray:
    """Compute dense optical flow using the Farneback algorithm.
    
    Args:
        prev: First 8-bit grayscale image
        next: Second 8-bit grayscale image
        flow: Computed flow or initial estimate
        pyr_scale: Pyramid scale (< 1)
        levels: Number of pyramid levels
        winsize: Averaging window size
        iterations: Number of iterations per level
        poly_n: Polynomial expansion neighborhood size
        poly_sigma: Gaussian sigma for polynomial expansion
        flags: Operation flags
    
    Returns:
        Flow image (H, W, 2) with x,y displacement per pixel
    """
    if prev.ndim == 3:
        from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
        prev = cvtColor(prev, COLOR_BGR2GRAY)
    if next.ndim == 3:
        from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
        next = cvtColor(next, COLOR_BGR2GRAY)
    
    h, w = prev.shape
    
    # Initialize flow
    if flow is not None:
        result = flow.copy()
    else:
        result = np.zeros((h, w, 2), dtype=np.float32)
    
    # Build pyramids
    prev_pyr = [prev.astype(np.float32)]
    next_pyr = [next.astype(np.float32)]
    
    for _ in range(1, levels):
        new_h = int(prev_pyr[-1].shape[0] * pyr_scale)
        new_w = int(prev_pyr[-1].shape[1] * pyr_scale)
        if new_h < 10 or new_w < 10:
            break
        
        from scipy.ndimage import zoom
        prev_pyr.append(zoom(prev_pyr[-1], pyr_scale, order=1))
        next_pyr.append(zoom(next_pyr[-1], pyr_scale, order=1))
    
    # Process from coarse to fine
    for level in range(len(prev_pyr) - 1, -1, -1):
        scale = pyr_scale ** level
        prev_level = prev_pyr[level]
        next_level = next_pyr[level]
        lh, lw = prev_level.shape
        
        # Scale flow for this level
        if level == len(prev_pyr) - 1:
            flow_level = np.zeros((lh, lw, 2), dtype=np.float32)
        else:
            flow_level = zoom(result, (scale, scale, 1), order=1)
            flow_level *= scale
        
        # Polynomial expansion (simplified)
        # Compute gradients
        Ix = np.zeros_like(prev_level)
        Iy = np.zeros_like(prev_level)
        Ix[:, 1:-1] = (prev_level[:, 2:] - prev_level[:, :-2]) / 2
        Iy[1:-1, :] = (prev_level[2:, :] - prev_level[:-2, :]) / 2
        
        # Iterate
        for _ in range(iterations):
            # Warp next image by current flow
            warped = _warp_image(next_level, flow_level)
            
            # Compute temporal derivative
            It = warped - prev_level
            
            # Compute flow update using local averaging
            half_win = winsize // 2
            
            for y in range(half_win, lh - half_win):
                for x in range(half_win, lw - half_win):
                    # Window
                    Ix_win = Ix[y-half_win:y+half_win+1, x-half_win:x+half_win+1]
                    Iy_win = Iy[y-half_win:y+half_win+1, x-half_win:x+half_win+1]
                    It_win = It[y-half_win:y+half_win+1, x-half_win:x+half_win+1]
                    
                    # Structure tensor
                    Ixx = np.sum(Ix_win * Ix_win)
                    Ixy = np.sum(Ix_win * Iy_win)
                    Iyy = np.sum(Iy_win * Iy_win)
                    
                    bx = -np.sum(Ix_win * It_win)
                    by = -np.sum(Iy_win * It_win)
                    
                    det = Ixx * Iyy - Ixy * Ixy
                    if abs(det) > 1e-6:
                        du = (Iyy * bx - Ixy * by) / det
                        dv = (-Ixy * bx + Ixx * by) / det
                        
                        flow_level[y, x, 0] += du * 0.5
                        flow_level[y, x, 1] += dv * 0.5
        
        result = flow_level
    
    # Scale back to original size
    if result.shape[0] != h or result.shape[1] != w:
        from scipy.ndimage import zoom
        result = zoom(result, (h / result.shape[0], w / result.shape[1], 1), order=1)
    
    return result


def _build_pyramid(img: np.ndarray, max_level: int) -> list:
    """Build image pyramid."""
    pyr = [img.astype(np.float32)]
    
    for _ in range(max_level):
        # Simple 2x downsampling
        prev = pyr[-1]
        h, w = prev.shape
        new_h, new_w = h // 2, w // 2
        if new_h < 5 or new_w < 5:
            break
        
        # Box filter and subsample
        downsampled = np.zeros((new_h, new_w), dtype=np.float32)
        for y in range(new_h):
            for x in range(new_w):
                downsampled[y, x] = np.mean(prev[y*2:y*2+2, x*2:x*2+2])
        
        pyr.append(downsampled)
    
    return pyr


def _warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp image by flow field."""
    h, w = img.shape[:2]
    
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    map_x = (x_coords + flow[:, :, 0]).astype(np.float32)
    map_y = (y_coords + flow[:, :, 1]).astype(np.float32)
    
    # Bilinear interpolation
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    
    xa = map_x - np.floor(map_x)
    ya = map_y - np.floor(map_y)
    
    result = (img[y0, x0] * (1 - xa) * (1 - ya) +
              img[y0, x1] * xa * (1 - ya) +
              img[y1, x0] * (1 - xa) * ya +
              img[y1, x1] * xa * ya)
    
    return result


# Optical flow flags
OPTFLOW_USE_INITIAL_FLOW = 4
OPTFLOW_LK_GET_MIN_EIGENVALS = 8
OPTFLOW_FARNEBACK_GAUSSIAN = 256


__all__ = [
    "calcOpticalFlowPyrLK",
    "calcOpticalFlowFarneback",
    "OPTFLOW_USE_INITIAL_FLOW",
    "OPTFLOW_LK_GET_MIN_EIGENVALS",
    "OPTFLOW_FARNEBACK_GAUSSIAN",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.