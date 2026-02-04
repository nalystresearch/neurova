# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Camera calibration for Neurova

Provides camera calibration functions for:
- Chessboard corner detection
- Camera intrinsic calibration
- Image undistortion
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np


# Calibration Pattern Flags

CALIB_CB_ADAPTIVE_THRESH = 1
CALIB_CB_NORMALIZE_IMAGE = 2
CALIB_CB_FILTER_QUADS = 4
CALIB_CB_FAST_CHECK = 8
CALIB_CB_EXHAUSTIVE = 16
CALIB_CB_ACCURACY = 32
CALIB_CB_LARGER = 64
CALIB_CB_MARKER = 128

# Calibration flags
CALIB_USE_INTRINSIC_GUESS = 1
CALIB_FIX_ASPECT_RATIO = 2
CALIB_FIX_PRINCIPAL_POINT = 4
CALIB_ZERO_TANGENT_DIST = 8
CALIB_FIX_FOCAL_LENGTH = 16
CALIB_FIX_K1 = 32
CALIB_FIX_K2 = 64
CALIB_FIX_K3 = 128
CALIB_RATIONAL_MODEL = 256
CALIB_THIN_PRISM_MODEL = 512
CALIB_TILTED_MODEL = 1024


def findChessboardCorners(
    image: np.ndarray,
    patternSize: Tuple[int, int],
    corners: Optional[np.ndarray] = None,
    flags: int = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE
) -> Tuple[bool, np.ndarray]:
    """Find the positions of internal corners of a chessboard.
    
    Args:
        image: Source chessboard image (grayscale or color)
        patternSize: Number of inner corners (columns, rows)
        corners: Optional output array for corners
        flags: Detection flags (CALIB_CB_*)
    
    Returns:
        Tuple of (success, corners array of shape [N, 1, 2])
    """
    if image.size == 0:
        return False, np.zeros((0, 1, 2), dtype=np.float32)
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        # Simple grayscale conversion
        gray = (0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 
                0.114 * image[:, :, 0]).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)
    
    # Normalize if flag is set
    if flags & CALIB_CB_NORMALIZE_IMAGE:
        gray = ((gray - gray.min()) * 255 / 
                (gray.max() - gray.min() + 1e-10)).astype(np.uint8)
    
    # Apply adaptive threshold if flag is set
    if flags & CALIB_CB_ADAPTIVE_THRESH:
        thresh = _adaptive_threshold(gray)
    else:
        thresh = (gray > 127).astype(np.uint8) * 255
    
    # Find corners using Harris or simplified detection
    found_corners = _detect_chessboard_corners(thresh, patternSize)
    
    if found_corners is None or len(found_corners) < patternSize[0] * patternSize[1]:
        return False, np.zeros((0, 1, 2), dtype=np.float32)
    
    # Reshape to Neurova format [N, 1, 2]
    result = np.array(found_corners, dtype=np.float32).reshape(-1, 1, 2)
    
    return True, result


def _adaptive_threshold(gray: np.ndarray, block_size: int = 11) -> np.ndarray:
    """Simple adaptive thresholding."""
    from scipy.ndimage import uniform_filter
    
    try:
        local_mean = uniform_filter(gray.astype(np.float32), block_size)
        return ((gray > local_mean - 5) * 255).astype(np.uint8)
    except ImportError:
        # Fallback to simple threshold
        return ((gray > gray.mean()) * 255).astype(np.uint8)


def _detect_chessboard_corners(
    thresh: np.ndarray, 
    pattern_size: Tuple[int, int]
) -> Optional[List[Tuple[float, float]]]:
    """Simplified chessboard corner detection.
    
    This is a placeholder - real implementation would use more robust detection.
    """
    rows, cols = pattern_size[1], pattern_size[0]
    h, w = thresh.shape
    
    # Simple grid-based detection (placeholder)
    step_x = w / (cols + 1)
    step_y = h / (rows + 1)
    
    corners = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 1) * step_x
            y = (r + 1) * step_y
            corners.append((x, y))
    
    return corners


def findChessboardCornersSB(
    image: np.ndarray,
    patternSize: Tuple[int, int],
    flags: int = 0
) -> Tuple[bool, np.ndarray]:
    """Find chessboard corners using sector-based approach.
    
    More robust than findChessboardCorners for some cases.
    """
    return findChessboardCorners(image, patternSize, None, flags)


def cornerSubPix(
    image: np.ndarray,
    corners: np.ndarray,
    winSize: Tuple[int, int],
    zeroZone: Tuple[int, int],
    criteria: Tuple[int, int, float]
) -> np.ndarray:
    """Refine corner locations to subpixel accuracy.
    
    Args:
        image: Source image (grayscale)
        corners: Initial corner positions [N, 1, 2]
        winSize: Half size of search window
        zeroZone: Half size of dead region in search zone
        criteria: Termination criteria (type, maxCount, epsilon)
    
    Returns:
        Refined corner positions
    """
    if corners.size == 0:
        return corners.copy()
    
    refined = corners.copy().astype(np.float32)
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 
                0.114 * image[:, :, 0]).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    h, w = gray.shape
    max_iter = criteria[1] if len(criteria) > 1 else 30
    epsilon = criteria[2] if len(criteria) > 2 else 0.001
    
    for idx in range(len(refined)):
        cx, cy = refined[idx, 0]
        
        for iteration in range(max_iter):
            # Define search window
            x0 = max(0, int(cx) - winSize[0])
            x1 = min(w - 1, int(cx) + winSize[0])
            y0 = max(0, int(cy) - winSize[1])
            y1 = min(h - 1, int(cy) + winSize[1])
            
            if x1 <= x0 or y1 <= y0:
                break
            
            # Extract window and compute gradient-based refinement
            window = gray[y0:y1+1, x0:x1+1]
            
            # Compute gradients
            gy, gx = np.gradient(window)
            
            # Weighted centroid based on gradient magnitude
            mag = np.sqrt(gx**2 + gy**2)
            total_mag = mag.sum() + 1e-10
            
            yy, xx = np.mgrid[0:window.shape[0], 0:window.shape[1]]
            new_x = x0 + (xx * mag).sum() / total_mag
            new_y = y0 + (yy * mag).sum() / total_mag
            
            # Check convergence
            dx = new_x - cx
            dy = new_y - cy
            if dx*dx + dy*dy < epsilon * epsilon:
                break
            
            cx, cy = new_x, new_y
        
        refined[idx, 0] = [cx, cy]
    
    return refined


def drawChessboardCorners(
    image: np.ndarray,
    patternSize: Tuple[int, int],
    corners: np.ndarray,
    patternWasFound: bool
) -> np.ndarray:
    """Draw detected chessboard corners on image.
    
    Args:
        image: Destination image (modified in place)
        patternSize: Number of inner corners (columns, rows)
        corners: Detected corner positions [N, 1, 2]
        patternWasFound: Whether full pattern was found
    
    Returns:
        Image with drawn corners
    """
    if corners.size == 0:
        return image
    
    pts = corners.reshape(-1, 2)
    
    # Choose colors based on whether pattern was found
    if patternWasFound:
        # Draw lines between corners
        rows, cols = patternSize[1], patternSize[0]
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= len(pts):
                    break
                
                pt = pts[idx].astype(int)
                
                # Draw corner
                _draw_circle(image, pt, 5, (0, 0, 255), -1)
                
                # Draw lines to neighbors
                if c < cols - 1 and idx + 1 < len(pts):
                    _draw_line(image, pt, pts[idx + 1].astype(int), (0, 255, 0), 1)
                if r < rows - 1 and idx + cols < len(pts):
                    _draw_line(image, pt, pts[idx + cols].astype(int), (0, 255, 0), 1)
    else:
        # Just draw red circles
        for pt in pts:
            _draw_circle(image, pt.astype(int), 5, (0, 0, 255), -1)
    
    return image


def _draw_circle(img: np.ndarray, center: Tuple[int, int], radius: int, 
                 color: Tuple[int, int, int], thickness: int) -> None:
    """Simple circle drawing helper."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    if thickness < 0:  # Filled
        mask = dist <= radius
    else:
        mask = (dist >= radius - thickness/2) & (dist <= radius + thickness/2)
    
    if img.ndim == 3:
        img[mask] = color
    else:
        img[mask] = color[0]


def _draw_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
               color: Tuple[int, int, int], thickness: int) -> None:
    """Simple line drawing helper."""
    x0, y0 = pt1
    x1, y1 = pt2
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx - dy
    h, w = img.shape[:2]
    
    while True:
        if 0 <= y0 < h and 0 <= x0 < w:
            if img.ndim == 3:
                img[y0, x0] = color
            else:
                img[y0, x0] = color[0]
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def calibrateCamera(
    objectPoints: List[np.ndarray],
    imagePoints: List[np.ndarray],
    imageSize: Tuple[int, int],
    cameraMatrix: Optional[np.ndarray] = None,
    distCoeffs: Optional[np.ndarray] = None,
    rvecs: Optional[List[np.ndarray]] = None,
    tvecs: Optional[List[np.ndarray]] = None,
    flags: int = 0,
    criteria: Tuple[int, int, float] = (3, 30, 1e-6)
) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Calibrate camera using multiple views of a calibration pattern.
    
    Args:
        objectPoints: List of 3D object points for each view
        imagePoints: List of 2D image points for each view
        imageSize: Image size (width, height)
        cameraMatrix: Optional initial camera matrix
        distCoeffs: Optional initial distortion coefficients
        rvecs: Optional output rotation vectors
        tvecs: Optional output translation vectors
        flags: Calibration flags (CALIB_*)
        criteria: Termination criteria
    
    Returns:
        Tuple of (reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs)
    """
    if not objectPoints or not imagePoints:
        return 0.0, np.eye(3), np.zeros(5), [], []
    
    w, h = imageSize
    
    # Initialize camera matrix with reasonable defaults
    if cameraMatrix is None or not (flags & CALIB_USE_INTRINSIC_GUESS):
        fx = fy = max(w, h)  # Initial focal length estimate
        cx, cy = w / 2, h / 2  # Principal point at center
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
    else:
        K = cameraMatrix.copy()
    
    # Initialize distortion coefficients
    if distCoeffs is None:
        D = np.zeros(5, dtype=np.float64)
    else:
        D = distCoeffs.flatten().copy()
        if len(D) < 5:
            D = np.concatenate([D, np.zeros(5 - len(D))])
    
    # Compute rotation and translation for each view
    out_rvecs = []
    out_tvecs = []
    
    for obj_pts, img_pts in zip(objectPoints, imagePoints):
        # Simple pose estimation using DLT-like approach
        rvec, tvec = _estimate_pose(obj_pts, img_pts, K)
        out_rvecs.append(rvec)
        out_tvecs.append(tvec)
    
    # Compute reprojection error
    total_error = 0.0
    total_points = 0
    
    for obj_pts, img_pts, rvec, tvec in zip(objectPoints, imagePoints, out_rvecs, out_tvecs):
        projected = _project_points(obj_pts, rvec, tvec, K, D)
        img_pts_flat = img_pts.reshape(-1, 2)
        error = np.sqrt(np.sum((projected - img_pts_flat)**2))
        total_error += error
        total_points += len(img_pts_flat)
    
    rms_error = total_error / (total_points + 1e-10)
    
    return rms_error, K, D, out_rvecs, out_tvecs


def _estimate_pose(
    obj_pts: np.ndarray, 
    img_pts: np.ndarray, 
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple pose estimation."""
    # Simplified pose estimation
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    
    # Estimate translation from centroid
    obj_center = obj_pts.reshape(-1, 3).mean(axis=0)
    img_center = img_pts.reshape(-1, 2).mean(axis=0)
    
    tvec[0] = (img_center[0] - K[0, 2]) / K[0, 0]
    tvec[1] = (img_center[1] - K[1, 2]) / K[1, 1]
    tvec[2] = K[0, 0] / (img_pts.reshape(-1, 2).max() - img_pts.reshape(-1, 2).min() + 1)
    
    return rvec.reshape(3, 1), tvec.reshape(3, 1)


def _project_points(
    obj_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray
) -> np.ndarray:
    """Project 3D points to 2D."""
    pts = obj_pts.reshape(-1, 3)
    
    # Apply rotation (simplified - using Rodrigues)
    R = _rodrigues(rvec.flatten())
    
    # Transform points
    transformed = (R @ pts.T).T + tvec.flatten()
    
    # Project
    x = transformed[:, 0] / (transformed[:, 2] + 1e-10)
    y = transformed[:, 1] / (transformed[:, 2] + 1e-10)
    
    # Apply camera matrix
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    
    return np.column_stack([u, v])


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to rotation matrix."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-10:
        return np.eye(3)
    
    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def getOptimalNewCameraMatrix(
    cameraMatrix: np.ndarray,
    distCoeffs: np.ndarray,
    imageSize: Tuple[int, int],
    alpha: float,
    newImgSize: Optional[Tuple[int, int]] = None,
    centerPrincipalPoint: bool = False
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Compute optimal new camera matrix for undistortion.
    
    Args:
        cameraMatrix: Input camera matrix
        distCoeffs: Distortion coefficients
        imageSize: Original image size (width, height)
        alpha: Free scaling parameter (0=no black, 1=all pixels)
        newImgSize: New image size (defaults to original)
        centerPrincipalPoint: Whether to center principal point
    
    Returns:
        Tuple of (new_camera_matrix, valid_roi)
    """
    if newImgSize is None or newImgSize == (0, 0):
        newImgSize = imageSize
    
    w, h = imageSize
    new_w, new_h = newImgSize
    
    # Scale camera matrix
    sx = new_w / w
    sy = new_h / h
    
    new_K = cameraMatrix.copy()
    new_K[0, 0] *= sx
    new_K[1, 1] *= sy
    new_K[0, 2] = new_K[0, 2] * sx if not centerPrincipalPoint else new_w / 2
    new_K[1, 2] = new_K[1, 2] * sy if not centerPrincipalPoint else new_h / 2
    
    # Blend based on alpha
    if alpha < 1.0:
        scale = 1.0 + alpha * 0.1
        new_K[0, 0] /= scale
        new_K[1, 1] /= scale
    
    # Compute valid ROI
    border = int(max(w, h) * (1 - alpha) * 0.1)
    valid_roi = (border, border, new_w - 2*border, new_h - 2*border)
    
    return new_K, valid_roi


def undistort(
    src: np.ndarray,
    cameraMatrix: np.ndarray,
    distCoeffs: np.ndarray,
    dst: Optional[np.ndarray] = None,
    newCameraMatrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """Undistort an image using camera calibration parameters.
    
    Args:
        src: Input distorted image
        cameraMatrix: Camera intrinsic matrix
        distCoeffs: Distortion coefficients
        dst: Optional output array
        newCameraMatrix: Optional new camera matrix
    
    Returns:
        Undistorted image
    """
    if newCameraMatrix is None:
        newCameraMatrix = cameraMatrix
    
    h, w = src.shape[:2]
    
    # Create output array
    if dst is None:
        if src.ndim == 3:
            result = np.zeros_like(src)
        else:
            result = np.zeros_like(src)
    else:
        result = dst
    
    # Get distortion coefficients
    D = distCoeffs.flatten()
    k1 = D[0] if len(D) > 0 else 0
    k2 = D[1] if len(D) > 1 else 0
    p1 = D[2] if len(D) > 2 else 0
    p2 = D[3] if len(D) > 3 else 0
    k3 = D[4] if len(D) > 4 else 0
    
    # Create undistortion map
    fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
    cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
    
    new_fx, new_fy = newCameraMatrix[0, 0], newCameraMatrix[1, 1]
    new_cx, new_cy = newCameraMatrix[0, 2], newCameraMatrix[1, 2]
    
    # For each output pixel, find corresponding input pixel
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Normalize to camera coordinates
    x_norm = (x_coords - new_cx) / new_fx
    y_norm = (y_coords - new_cy) / new_fy
    
    # Apply distortion
    r2 = x_norm**2 + y_norm**2
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    
    x_dist = x_norm * radial + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
    y_dist = y_norm * radial + p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm
    
    # Back to pixel coordinates
    map_x = x_dist * fx + cx
    map_y = y_dist * fy + cy
    
    # Remap using bilinear interpolation
    result = _remap_bilinear(src, map_x, map_y)
    
    return result


def _remap_bilinear(src: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Remap image using bilinear interpolation."""
    h, w = src.shape[:2]
    
    # Clip coordinates
    x0 = np.clip(map_x.astype(int), 0, w - 1)
    y0 = np.clip(map_y.astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    
    # Interpolation weights
    wx = np.clip(map_x - x0, 0, 1)
    wy = np.clip(map_y - y0, 0, 1)
    
    if src.ndim == 3:
        wx = wx[:, :, np.newaxis]
        wy = wy[:, :, np.newaxis]
    
    # Bilinear interpolation
    result = (
        src[y0, x0] * (1 - wx) * (1 - wy) +
        src[y0, x1] * wx * (1 - wy) +
        src[y1, x0] * (1 - wx) * wy +
        src[y1, x1] * wx * wy
    )
    
    return result.astype(src.dtype)


def undistortPoints(
    src: np.ndarray,
    cameraMatrix: np.ndarray,
    distCoeffs: np.ndarray,
    R: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None
) -> np.ndarray:
    """Undistort 2D points.
    
    Args:
        src: Input distorted points [N, 1, 2] or [N, 2]
        cameraMatrix: Camera matrix
        distCoeffs: Distortion coefficients
        R: Optional rectification transform
        P: Optional new camera matrix
    
    Returns:
        Undistorted points
    """
    if P is None:
        P = cameraMatrix
    
    pts = src.reshape(-1, 2).astype(np.float64)
    
    fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
    cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
    
    D = distCoeffs.flatten()
    k1 = D[0] if len(D) > 0 else 0
    k2 = D[1] if len(D) > 1 else 0
    p1 = D[2] if len(D) > 2 else 0
    p2 = D[3] if len(D) > 3 else 0
    k3 = D[4] if len(D) > 4 else 0
    
    # Normalize
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    
    # Iterative undistortion
    x0, y0 = x.copy(), y.copy()
    
    for _ in range(5):
        r2 = x**2 + y**2
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        
        dx = 2*p1*x*y + p2*(r2 + 2*x**2)
        dy = p1*(r2 + 2*y**2) + 2*p2*x*y
        
        x = (x0 - dx) / radial
        y = (y0 - dy) / radial
    
    # Apply new camera matrix
    new_fx, new_fy = P[0, 0], P[1, 1]
    new_cx, new_cy = P[0, 2], P[1, 2]
    
    result = np.column_stack([
        x * new_fx + new_cx,
        y * new_fy + new_cy
    ])
    
    return result.reshape(src.shape)


# Pose Estimation (NEW)

from neurova.calibration.pose import (
    solvePnP, solvePnPRansac, projectPoints,
    findHomography, findFundamentalMat, findEssentialMat,
    Rodrigues, decomposeHomographyMat, triangulatePoints,
    SOLVEPNP_ITERATIVE, SOLVEPNP_P3P, SOLVEPNP_AP3P, SOLVEPNP_EPNP,
    SOLVEPNP_DLS, SOLVEPNP_UPNP, SOLVEPNP_IPPE, SOLVEPNP_IPPE_SQUARE, SOLVEPNP_SQPNP,
    RANSAC, LMEDS, RHO,
    FM_7POINT, FM_8POINT, FM_RANSAC, FM_LMEDS,
)


# Exports

__all__ = [
    # Chessboard detection
    "findChessboardCorners",
    "findChessboardCornersSB",
    "cornerSubPix",
    "drawChessboardCorners",
    
    # Camera calibration
    "calibrateCamera",
    "getOptimalNewCameraMatrix",
    "undistort",
    "undistortPoints",
    
    # Flags
    "CALIB_CB_ADAPTIVE_THRESH",
    "CALIB_CB_NORMALIZE_IMAGE",
    "CALIB_CB_FILTER_QUADS",
    "CALIB_CB_FAST_CHECK",
    "CALIB_CB_EXHAUSTIVE",
    "CALIB_CB_ACCURACY",
    "CALIB_CB_LARGER",
    "CALIB_CB_MARKER",
    "CALIB_USE_INTRINSIC_GUESS",
    "CALIB_FIX_ASPECT_RATIO",
    "CALIB_FIX_PRINCIPAL_POINT",
    "CALIB_ZERO_TANGENT_DIST",
    "CALIB_FIX_FOCAL_LENGTH",
    "CALIB_FIX_K1",
    "CALIB_FIX_K2",
    "CALIB_FIX_K3",
    "CALIB_RATIONAL_MODEL",
    "CALIB_THIN_PRISM_MODEL",
    "CALIB_TILTED_MODEL",
    
    # Pose estimation (NEW)
    "solvePnP", "solvePnPRansac", "projectPoints",
    "findHomography", "findFundamentalMat", "findEssentialMat",
    "Rodrigues", "decomposeHomographyMat", "triangulatePoints",
    "SOLVEPNP_ITERATIVE", "SOLVEPNP_P3P", "SOLVEPNP_AP3P", "SOLVEPNP_EPNP",
    "SOLVEPNP_DLS", "SOLVEPNP_UPNP", "SOLVEPNP_IPPE", "SOLVEPNP_IPPE_SQUARE", "SOLVEPNP_SQPNP",
    "RANSAC", "LMEDS", "RHO",
    "FM_7POINT", "FM_8POINT", "FM_RANSAC", "FM_LMEDS",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.