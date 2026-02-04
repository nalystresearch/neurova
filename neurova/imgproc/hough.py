# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova.imgproc.hough - Hough Transform functions

Provides Neurova Hough transform implementations for line and circle detection.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def HoughLines(
    image: np.ndarray,
    rho: float,
    theta: float,
    threshold: int,
    lines: Optional[np.ndarray] = None,
    srn: float = 0,
    stn: float = 0,
    min_theta: float = 0,
    max_theta: float = np.pi
) -> Optional[np.ndarray]:
    """Find lines in a binary image using the standard Hough transform.
    
    Args:
        image: 8-bit, single-channel binary source image
        rho: Distance resolution of the accumulator in pixels
        theta: Angle resolution of the accumulator in radians
        threshold: Accumulator threshold parameter
        lines: Output vector of lines (optional)
        srn: For multi-scale Hough transform (0 = classical)
        stn: For multi-scale Hough transform (0 = classical)
        min_theta: Minimum angle to check for lines
        max_theta: Maximum angle to check for lines
    
    Returns:
        Array of (rho, theta) pairs, shape (N, 1, 2)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be single-channel")
    
    height, width = image.shape
    
    # Maximum possible rho
    diag_len = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Accumulator dimensions
    num_rhos = int(np.ceil(2 * diag_len / rho))
    thetas = np.arange(min_theta, max_theta, theta)
    num_thetas = len(thetas)
    
    # Create accumulator
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)
    
    # Precompute cos and sin
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    # Find edge points
    y_coords, x_coords = np.nonzero(image)
    
    # Vote in accumulator
    for x, y in zip(x_coords, y_coords):
        for t_idx, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
            rho_val = x * cos_t + y * sin_t
            rho_idx = int(np.round((rho_val + diag_len) / rho))
            if 0 <= rho_idx < num_rhos:
                accumulator[rho_idx, t_idx] += 1
    
    # Extract lines above threshold
    line_indices = np.argwhere(accumulator >= threshold)
    
    if len(line_indices) == 0:
        return None
    
    # Convert indices back to (rho, theta)
    result_lines = []
    for rho_idx, theta_idx in line_indices:
        rho_val = rho_idx * rho - diag_len
        theta_val = thetas[theta_idx]
        result_lines.append([rho_val, theta_val])
    
    # Sort by accumulator value (strongest lines first)
    accum_vals = [accumulator[r, t] for r, t in line_indices]
    sorted_indices = np.argsort(accum_vals)[::-1]
    result_lines = [result_lines[i] for i in sorted_indices]
    
    return np.array(result_lines, dtype=np.float32).reshape(-1, 1, 2)


def HoughLinesP(
    image: np.ndarray,
    rho: float,
    theta: float,
    threshold: int,
    lines: Optional[np.ndarray] = None,
    minLineLength: float = 0,
    maxLineGap: float = 0
) -> Optional[np.ndarray]:
    """Find line segments using the probabilistic Hough transform.
    
    Args:
        image: 8-bit, single-channel binary source image
        rho: Distance resolution of the accumulator in pixels
        theta: Angle resolution of the accumulator in radians
        threshold: Accumulator threshold parameter
        lines: Output vector of lines
        minLineLength: Minimum line length
        maxLineGap: Maximum gap between points on the same line
    
    Returns:
        Array of (x1, y1, x2, y2), shape (N, 1, 4)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be single-channel")
    
    height, width = image.shape
    diag_len = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Accumulator setup
    num_rhos = int(np.ceil(2 * diag_len / rho))
    thetas = np.arange(0, np.pi, theta)
    num_thetas = len(thetas)
    
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    # Get edge points
    edge_points = list(zip(*np.nonzero(image)))
    np.random.shuffle(edge_points)
    
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)
    used_points = set()
    result_lines = []
    
    for y, x in edge_points:
        if (y, x) in used_points:
            continue
        
        # Vote
        for t_idx, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
            rho_val = x * cos_t + y * sin_t
            rho_idx = int(np.round((rho_val + diag_len) / rho))
            if 0 <= rho_idx < num_rhos:
                accumulator[rho_idx, t_idx] += 1
        
        # Check for lines
        max_idx = np.unravel_index(accumulator.argmax(), accumulator.shape)
        if accumulator[max_idx] >= threshold:
            rho_idx, theta_idx = max_idx
            rho_val = rho_idx * rho - diag_len
            theta_val = thetas[theta_idx]
            
            # Find line endpoints
            cos_t, sin_t = np.cos(theta_val), np.sin(theta_val)
            
            # Collect points on this line
            line_points = []
            for py, px in edge_points:
                if (py, px) in used_points:
                    continue
                
                dist = abs(px * cos_t + py * sin_t - rho_val)
                if dist < rho:
                    line_points.append((px, py))
            
            if len(line_points) >= 2:
                # Find endpoints
                if abs(cos_t) > abs(sin_t):
                    # More horizontal - sort by x
                    line_points.sort(key=lambda p: p[0])
                else:
                    # More vertical - sort by y
                    line_points.sort(key=lambda p: p[1])
                
                # Check for gaps and minimum length
                segments = []
                seg_start = 0
                
                for i in range(1, len(line_points)):
                    dx = line_points[i][0] - line_points[i-1][0]
                    dy = line_points[i][1] - line_points[i-1][1]
                    gap = np.sqrt(dx*dx + dy*dy)
                    
                    if gap > maxLineGap:
                        # End current segment
                        if i - seg_start >= 2:
                            segments.append((seg_start, i-1))
                        seg_start = i
                
                # Last segment
                if len(line_points) - seg_start >= 2:
                    segments.append((seg_start, len(line_points) - 1))
                
                # Add valid segments
                for start, end in segments:
                    x1, y1 = line_points[start]
                    x2, y2 = line_points[end]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if length >= minLineLength:
                        result_lines.append([x1, y1, x2, y2])
                        
                        # Mark points as used
                        for px, py in line_points[start:end+1]:
                            used_points.add((py, px))
            
            # Reset accumulator for this line
            accumulator[max_idx] = 0
    
    if len(result_lines) == 0:
        return None
    
    return np.array(result_lines, dtype=np.int32).reshape(-1, 1, 4)


def HoughCircles(
    image: np.ndarray,
    method: int,
    dp: float,
    minDist: float,
    param1: float = 100,
    param2: float = 100,
    minRadius: int = 0,
    maxRadius: int = 0
) -> Optional[np.ndarray]:
    """Find circles in a grayscale image using the Hough transform.
    
    Args:
        image: 8-bit, single-channel grayscale image
        method: Detection method (HOUGH_GRADIENT=3)
        dp: Inverse ratio of accumulator resolution (1 = same as image)
        minDist: Minimum distance between detected circle centers
        param1: First method-specific parameter (Canny high threshold)
        param2: Second method-specific parameter (accumulator threshold)
        minRadius: Minimum circle radius
        maxRadius: Maximum circle radius (0 = max possible)
    
    Returns:
        Array of (x, y, radius), shape (1, N, 3)
    """
    from ..filters import sobel, canny
    
    if image.ndim != 2:
        raise ValueError("Input image must be single-channel grayscale")
    
    height, width = image.shape
    
    if maxRadius <= 0:
        maxRadius = int(min(height, width) / 2)
    
    if minRadius < 0:
        minRadius = 0
    
    # Apply edge detection
    edges = canny(image, param1 / 2, param1)
    
    # Compute gradients
    grad_x = sobel(image, ddepth=-1, dx=1, dy=0, ksize=3)
    grad_y = sobel(image, ddepth=-1, dx=0, dy=1, ksize=3)
    
    # Accumulator
    acc_scale = 1.0 / dp
    acc_height = int(height * acc_scale)
    acc_width = int(width * acc_scale)
    
    circles = []
    
    # For each radius
    for r in range(minRadius, maxRadius + 1):
        accumulator = np.zeros((acc_height, acc_width), dtype=np.float32)
        
        # Find edge points
        edge_points = np.argwhere(edges > 0)
        
        for y, x in edge_points:
            gx = grad_x[y, x]
            gy = grad_y[y, x]
            
            # Gradient magnitude
            mag = np.sqrt(gx*gx + gy*gy)
            if mag < 1:
                continue
            
            # Unit gradient
            gx /= mag
            gy /= mag
            
            # Vote in both directions along gradient
            for direction in [-1, 1]:
                cx = int((x + direction * gx * r) * acc_scale)
                cy = int((y + direction * gy * r) * acc_scale)
                
                if 0 <= cx < acc_width and 0 <= cy < acc_height:
                    accumulator[cy, cx] += 1
        
        # Find local maxima above threshold
        threshold_val = param2
        candidates = np.argwhere(accumulator >= threshold_val)
        
        for cy, cx in candidates:
            # Convert back to image coordinates
            center_x = int(cx / acc_scale)
            center_y = int(cy / acc_scale)
            
            # Check if too close to existing circles
            too_close = False
            for existing in circles:
                dist = np.sqrt((center_x - existing[0])**2 + 
                              (center_y - existing[1])**2)
                if dist < minDist:
                    too_close = True
                    break
            
            if not too_close:
                circles.append([center_x, center_y, r])
    
    if len(circles) == 0:
        return None
    
    return np.array(circles, dtype=np.float32).reshape(1, -1, 3)


# Hough method constants
HOUGH_STANDARD = 0
HOUGH_PROBABILISTIC = 1
HOUGH_MULTI_SCALE = 2
HOUGH_GRADIENT = 3
HOUGH_GRADIENT_ALT = 4


__all__ = [
    "HoughLines",
    "HoughLinesP", 
    "HoughCircles",
    "HOUGH_STANDARD",
    "HOUGH_PROBABILISTIC",
    "HOUGH_MULTI_SCALE",
    "HOUGH_GRADIENT",
    "HOUGH_GRADIENT_ALT",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.