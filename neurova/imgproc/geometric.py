# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova.imgproc.geometric - Geometric transformations

Provides Neurova perspective and affine transformation functions.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np


# Interpolation flags
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3
INTER_LANCZOS4 = 4
INTER_LINEAR_EXACT = 5
INTER_NEAREST_EXACT = 6

# Warp flags
WARP_FILL_OUTLIERS = 8
WARP_INVERSE_MAP = 16


def getPerspectiveTransform(
    src: np.ndarray,
    dst: np.ndarray,
    solveMethod: int = 0
) -> np.ndarray:
    """Calculate a perspective transform from four pairs of corresponding points.
    
    Args:
        src: Source quadrangle vertices (4x2 array)
        dst: Destination quadrangle vertices (4x2 array)
        solveMethod: Method for solving the system
    
    Returns:
        3x3 perspective transformation matrix
    """
    src = np.asarray(src, dtype=np.float64).reshape(4, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(4, 2)
    
    # Build the system of equations
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        
        A[i*2] = [x, y, 1, 0, 0, 0, -u*x, -u*y]
        A[i*2 + 1] = [0, 0, 0, x, y, 1, -v*x, -v*y]
        b[i*2] = u
        b[i*2 + 1] = v
    
    # Solve the system
    h = np.linalg.solve(A, b)
    
    # Create 3x3 matrix
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)
    
    return H


def getAffineTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Calculate an affine transform from three pairs of corresponding points.
    
    Args:
        src: Source triangle vertices (3x2 array)
        dst: Destination triangle vertices (3x2 array)
    
    Returns:
        2x3 affine transformation matrix
    """
    src = np.asarray(src, dtype=np.float64).reshape(3, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(3, 2)
    
    # Build the system
    A = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros(6, dtype=np.float64)
    
    for i in range(3):
        x, y = src[i]
        u, v = dst[i]
        
        A[i*2] = [x, y, 1, 0, 0, 0]
        A[i*2 + 1] = [0, 0, 0, x, y, 1]
        b[i*2] = u
        b[i*2 + 1] = v
    
    # Solve
    m = np.linalg.solve(A, b)
    
    return np.array([
        [m[0], m[1], m[2]],
        [m[3], m[4], m[5]]
    ], dtype=np.float64)


def getRotationMatrix2D(
    center: Tuple[float, float],
    angle: float,
    scale: float
) -> np.ndarray:
    """Calculate an affine matrix of 2D rotation.
    
    Args:
        center: Center of rotation (x, y)
        angle: Rotation angle in degrees (positive = counter-clockwise)
        scale: Isotropic scale factor
    
    Returns:
        2x3 rotation matrix
    """
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    cx, cy = center
    
    # Rotation matrix with scaling
    alpha = scale * cos_a
    beta = scale * sin_a
    
    M = np.array([
        [alpha, beta, (1 - alpha) * cx - beta * cy],
        [-beta, alpha, beta * cx + (1 - alpha) * cy]
    ], dtype=np.float64)
    
    return M


def warpPerspective(
    src: np.ndarray,
    M: np.ndarray,
    dsize: Tuple[int, int],
    dst: Optional[np.ndarray] = None,
    flags: int = INTER_LINEAR,
    borderMode: int = 0,
    borderValue: Union[float, Tuple] = 0
) -> np.ndarray:
    """Apply a perspective transformation to an image.
    
    Args:
        src: Input image
        M: 3x3 transformation matrix
        dsize: Output size (width, height)
        dst: Optional output array
        flags: Interpolation method
        borderMode: Border extrapolation mode
        borderValue: Border value for BORDER_CONSTANT
    
    Returns:
        Transformed image
    """
    width, height = dsize
    
    if src.ndim == 3:
        result = np.zeros((height, width, src.shape[2]), dtype=src.dtype)
        if isinstance(borderValue, (int, float)):
            result[:] = borderValue
        else:
            result[:] = borderValue[:src.shape[2]]
    else:
        result = np.full((height, width), borderValue, dtype=src.dtype)
    
    # Check for inverse flag
    if flags & WARP_INVERSE_MAP:
        M_inv = M
    else:
        M_inv = np.linalg.inv(M)
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Homogeneous coordinates
    ones = np.ones_like(x_coords, dtype=np.float64)
    coords = np.stack([x_coords, y_coords, ones], axis=-1)
    
    # Apply inverse transform
    coords_flat = coords.reshape(-1, 3)
    src_coords = coords_flat @ M_inv.T
    
    # Normalize homogeneous coordinates
    w = src_coords[:, 2:3]
    w[w == 0] = 1e-10
    src_coords = src_coords[:, :2] / w
    
    src_x = src_coords[:, 0].reshape(height, width)
    src_y = src_coords[:, 1].reshape(height, width)
    
    # Bilinear interpolation
    src_h, src_w = src.shape[:2]
    
    # Find valid coordinates
    valid = (src_x >= 0) & (src_x < src_w - 1) & (src_y >= 0) & (src_y < src_h - 1)
    
    if flags & 0x7 == INTER_NEAREST:
        # Nearest neighbor
        src_xi = np.round(src_x).astype(np.int32)
        src_yi = np.round(src_y).astype(np.int32)
        
        valid_nn = (src_xi >= 0) & (src_xi < src_w) & (src_yi >= 0) & (src_yi < src_h)
        
        if src.ndim == 3:
            for c in range(src.shape[2]):
                result[:, :, c][valid_nn] = src[:, :, c][src_yi[valid_nn], src_xi[valid_nn]]
        else:
            result[valid_nn] = src[src_yi[valid_nn], src_xi[valid_nn]]
    else:
        # Bilinear interpolation
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        xa = src_x - x0
        ya = src_y - y0
        
        if src.ndim == 3:
            for c in range(src.shape[2]):
                v00 = src[:, :, c][np.clip(y0, 0, src_h-1), np.clip(x0, 0, src_w-1)]
                v01 = src[:, :, c][np.clip(y0, 0, src_h-1), np.clip(x1, 0, src_w-1)]
                v10 = src[:, :, c][np.clip(y1, 0, src_h-1), np.clip(x0, 0, src_w-1)]
                v11 = src[:, :, c][np.clip(y1, 0, src_h-1), np.clip(x1, 0, src_w-1)]
                
                interp = (v00 * (1 - xa) * (1 - ya) + 
                         v01 * xa * (1 - ya) +
                         v10 * (1 - xa) * ya +
                         v11 * xa * ya)
                
                result[:, :, c][valid] = interp[valid].astype(src.dtype)
        else:
            v00 = src[np.clip(y0, 0, src_h-1), np.clip(x0, 0, src_w-1)]
            v01 = src[np.clip(y0, 0, src_h-1), np.clip(x1, 0, src_w-1)]
            v10 = src[np.clip(y1, 0, src_h-1), np.clip(x0, 0, src_w-1)]
            v11 = src[np.clip(y1, 0, src_h-1), np.clip(x1, 0, src_w-1)]
            
            interp = (v00 * (1 - xa) * (1 - ya) + 
                     v01 * xa * (1 - ya) +
                     v10 * (1 - xa) * ya +
                     v11 * xa * ya)
            
            result[valid] = interp[valid].astype(src.dtype)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def warpAffine(
    src: np.ndarray,
    M: np.ndarray,
    dsize: Tuple[int, int],
    dst: Optional[np.ndarray] = None,
    flags: int = INTER_LINEAR,
    borderMode: int = 0,
    borderValue: Union[float, Tuple] = 0
) -> np.ndarray:
    """Apply an affine transformation to an image.
    
    Args:
        src: Input image
        M: 2x3 transformation matrix
        dsize: Output size (width, height)
        dst: Optional output array
        flags: Interpolation method
        borderMode: Border extrapolation mode
        borderValue: Border value for BORDER_CONSTANT
    
    Returns:
        Transformed image
    """
    # Convert 2x3 to 3x3 for perspective transform
    M_3x3 = np.vstack([M, [0, 0, 1]])
    
    return warpPerspective(src, M_3x3, dsize, dst, flags, borderMode, borderValue)


def remap(
    src: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    interpolation: int = INTER_LINEAR,
    borderMode: int = 0,
    borderValue: Union[float, Tuple] = 0
) -> np.ndarray:
    """Apply a generic geometric transformation to an image.
    
    Args:
        src: Input image
        map1: First map of x values or (x,y) points
        map2: Second map of y values (or empty if map1 contains (x,y))
        interpolation: Interpolation method
        borderMode: Border mode
        borderValue: Border value
    
    Returns:
        Remapped image
    """
    if map1.ndim == 3 and map1.shape[2] == 2:
        # map1 contains (x, y) coordinates
        map_x = map1[:, :, 0]
        map_y = map1[:, :, 1]
    else:
        map_x = map1
        map_y = map2
    
    height, width = map_x.shape
    src_h, src_w = src.shape[:2]
    
    if src.ndim == 3:
        result = np.zeros((height, width, src.shape[2]), dtype=src.dtype)
        if isinstance(borderValue, (int, float)):
            result[:] = borderValue
        else:
            result[:] = borderValue[:src.shape[2]]
    else:
        result = np.full((height, width), borderValue, dtype=src.dtype)
    
    valid = (map_x >= 0) & (map_x < src_w - 1) & (map_y >= 0) & (map_y < src_h - 1)
    
    if interpolation == INTER_NEAREST:
        xi = np.round(map_x).astype(np.int32)
        yi = np.round(map_y).astype(np.int32)
        
        valid_nn = (xi >= 0) & (xi < src_w) & (yi >= 0) & (yi < src_h)
        
        if src.ndim == 3:
            for c in range(src.shape[2]):
                result[:, :, c][valid_nn] = src[:, :, c][yi[valid_nn], xi[valid_nn]]
        else:
            result[valid_nn] = src[yi[valid_nn], xi[valid_nn]]
    else:
        # Bilinear
        x0 = np.floor(map_x).astype(np.int32)
        y0 = np.floor(map_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        xa = map_x - x0
        ya = map_y - y0
        
        if src.ndim == 3:
            for c in range(src.shape[2]):
                v00 = src[:, :, c][np.clip(y0, 0, src_h-1), np.clip(x0, 0, src_w-1)]
                v01 = src[:, :, c][np.clip(y0, 0, src_h-1), np.clip(x1, 0, src_w-1)]
                v10 = src[:, :, c][np.clip(y1, 0, src_h-1), np.clip(x0, 0, src_w-1)]
                v11 = src[:, :, c][np.clip(y1, 0, src_h-1), np.clip(x1, 0, src_w-1)]
                
                interp = (v00 * (1 - xa) * (1 - ya) + 
                         v01 * xa * (1 - ya) +
                         v10 * (1 - xa) * ya +
                         v11 * xa * ya)
                
                result[:, :, c][valid] = interp[valid].astype(src.dtype)
        else:
            v00 = src[np.clip(y0, 0, src_h-1), np.clip(x0, 0, src_w-1)]
            v01 = src[np.clip(y0, 0, src_h-1), np.clip(x1, 0, src_w-1)]
            v10 = src[np.clip(y1, 0, src_h-1), np.clip(x0, 0, src_w-1)]
            v11 = src[np.clip(y1, 0, src_h-1), np.clip(x1, 0, src_w-1)]
            
            interp = (v00 * (1 - xa) * (1 - ya) + 
                     v01 * xa * (1 - ya) +
                     v10 * (1 - xa) * ya +
                     v11 * xa * ya)
            
            result[valid] = interp[valid].astype(src.dtype)
    
    return result


def invertAffineTransform(M: np.ndarray) -> np.ndarray:
    """Invert an affine transformation.
    
    Args:
        M: 2x3 affine transformation matrix
    
    Returns:
        Inverted 2x3 matrix
    """
    # Extract rotation/scale matrix and translation
    A = M[:, :2]
    b = M[:, 2]
    
    # Invert
    A_inv = np.linalg.inv(A)
    b_inv = -A_inv @ b
    
    return np.hstack([A_inv, b_inv.reshape(2, 1)])


def perspectiveTransform(
    src: np.ndarray,
    m: np.ndarray
) -> np.ndarray:
    """Perform perspective transformation of points.
    
    Args:
        src: Input points (N, 1, 2) or (N, 2)
        m: 3x3 transformation matrix
    
    Returns:
        Transformed points
    """
    src = np.asarray(src, dtype=np.float64)
    original_shape = src.shape
    
    if src.ndim == 3:
        points = src.reshape(-1, 2)
    else:
        points = src.reshape(-1, 2)
    
    # Homogeneous coordinates
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    
    # Transform
    transformed = homogeneous @ m.T
    
    # Normalize
    w = transformed[:, 2:3]
    w[w == 0] = 1e-10
    result = transformed[:, :2] / w
    
    return result.reshape(original_shape)


def transform(src: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Perform affine transformation of points.
    
    Args:
        src: Input points
        m: 2x3 transformation matrix
    
    Returns:
        Transformed points
    """
    src = np.asarray(src, dtype=np.float64)
    original_shape = src.shape
    
    points = src.reshape(-1, 2)
    
    # Homogeneous coordinates
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    
    # Transform (2x3 matrix)
    result = homogeneous @ m.T
    
    return result.reshape(original_shape)


__all__ = [
    "getPerspectiveTransform",
    "getAffineTransform",
    "getRotationMatrix2D",
    "warpPerspective",
    "warpAffine",
    "remap",
    "invertAffineTransform",
    "perspectiveTransform",
    "transform",
    
    # Constants
    "INTER_NEAREST",
    "INTER_LINEAR",
    "INTER_CUBIC",
    "INTER_AREA",
    "INTER_LANCZOS4",
    "INTER_LINEAR_EXACT",
    "INTER_NEAREST_EXACT",
    "WARP_FILL_OUTLIERS",
    "WARP_INVERSE_MAP",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.