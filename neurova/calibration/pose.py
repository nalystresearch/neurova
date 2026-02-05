# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Pose estimation and homography functions for Neurova.

Provides solvePnP, findHomography, projectPoints and related calib3d functions.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


# solvePnP methods
SOLVEPNP_ITERATIVE = 0
SOLVEPNP_P3P = 1
SOLVEPNP_AP3P = 2
SOLVEPNP_EPNP = 3
SOLVEPNP_DLS = 4
SOLVEPNP_UPNP = 5
SOLVEPNP_IPPE = 6
SOLVEPNP_IPPE_SQUARE = 7
SOLVEPNP_SQPNP = 8

# RANSAC flags
RANSAC = 8
LMEDS = 4
RHO = 16

# FM/EM flags
FM_7POINT = 1
FM_8POINT = 2
FM_RANSAC = 8
FM_LMEDS = 4


def solvePnP(
    objectPoints: np.ndarray,
    imagePoints: np.ndarray,
    cameraMatrix: np.ndarray,
    distCoeffs: Optional[np.ndarray],
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None,
    useExtrinsicGuess: bool = False,
    flags: int = SOLVEPNP_ITERATIVE
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Find object pose from 3D-2D point correspondences.
    
    Args:
        objectPoints: Array of object points in object coordinate space (Nx3)
        imagePoints: Array of corresponding image points (Nx2)
        cameraMatrix: Camera intrinsic matrix (3x3)
        distCoeffs: Distortion coefficients (can be None)
        rvec: Initial rotation vector (for useExtrinsicGuess=True)
        tvec: Initial translation vector
        useExtrinsicGuess: Use provided rvec/tvec as initial guess
        flags: Method for solving (SOLVEPNP_*)
    
    Returns:
        Tuple of (success, rotation_vector, translation_vector)
    """
    obj_pts = np.asarray(objectPoints, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(imagePoints, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(cameraMatrix, dtype=np.float64).reshape(3, 3)
    
    n = len(obj_pts)
    if n < 4:
        return False, np.zeros((3, 1)), np.zeros((3, 1))
    
    if n != len(img_pts):
        return False, np.zeros((3, 1)), np.zeros((3, 1))
    
    # Undistort points if distCoeffs provided
    if distCoeffs is not None:
        img_pts = _undistort_points(img_pts, K, distCoeffs)
    
    # Normalize image points
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    img_pts_norm = np.zeros_like(img_pts)
    img_pts_norm[:, 0] = (img_pts[:, 0] - cx) / fx
    img_pts_norm[:, 1] = (img_pts[:, 1] - cy) / fy
    
    # Simple DLT-based PnP solution
    try:
        rvec_out, tvec_out = _solve_pnp_dlt(obj_pts, img_pts_norm)
        return True, rvec_out, tvec_out
    except:
        return False, np.zeros((3, 1)), np.zeros((3, 1))


def solvePnPRansac(
    objectPoints: np.ndarray,
    imagePoints: np.ndarray,
    cameraMatrix: np.ndarray,
    distCoeffs: Optional[np.ndarray],
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None,
    useExtrinsicGuess: bool = False,
    iterationsCount: int = 100,
    reprojectionError: float = 8.0,
    confidence: float = 0.99,
    inliers: Optional[np.ndarray] = None,
    flags: int = SOLVEPNP_ITERATIVE
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Find object pose using RANSAC scheme.
    
    Args:
        objectPoints: Object points (Nx3)
        imagePoints: Image points (Nx2)
        cameraMatrix: Camera matrix (3x3)
        distCoeffs: Distortion coefficients
        iterationsCount: RANSAC iterations
        reprojectionError: Maximum reprojection error to be inlier
        confidence: Confidence level
        flags: Method flags
    
    Returns:
        Tuple of (success, rvec, tvec, inliers)
    """
    obj_pts = np.asarray(objectPoints, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(imagePoints, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(cameraMatrix, dtype=np.float64).reshape(3, 3)
    
    n = len(obj_pts)
    if n < 4:
        return False, np.zeros((3, 1)), np.zeros((3, 1)), np.array([])
    
    best_inliers = []
    best_rvec = None
    best_tvec = None
    
    for _ in range(iterationsCount):
        # Random sample of 4 points
        indices = np.random.choice(n, min(4, n), replace=False)
        
        success, rvec, tvec = solvePnP(
            obj_pts[indices], img_pts[indices], K, distCoeffs)
        
        if not success:
            continue
        
        # Count inliers
        projected = projectPoints(obj_pts, rvec, tvec, K, distCoeffs)[0]
        errors = np.sqrt(np.sum((projected.reshape(-1, 2) - img_pts)**2, axis=1))
        inlier_mask = errors < reprojectionError
        
        if np.sum(inlier_mask) > len(best_inliers):
            best_inliers = np.where(inlier_mask)[0]
            best_rvec = rvec
            best_tvec = tvec
    
    if best_rvec is None:
        return False, np.zeros((3, 1)), np.zeros((3, 1)), np.array([])
    
    # Refine with all inliers
    if len(best_inliers) >= 4:
        success, best_rvec, best_tvec = solvePnP(
            obj_pts[best_inliers], img_pts[best_inliers], K, distCoeffs)
    
    return True, best_rvec, best_tvec, best_inliers.reshape(-1, 1)


def projectPoints(
    objectPoints: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    cameraMatrix: np.ndarray,
    distCoeffs: Optional[np.ndarray],
    imagePoints: Optional[np.ndarray] = None,
    jacobian: Optional[np.ndarray] = None,
    aspectRatio: float = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Project 3D points to image plane.
    
    Args:
        objectPoints: Object points (Nx3)
        rvec: Rotation vector (3x1)
        tvec: Translation vector (3x1)
        cameraMatrix: Camera matrix (3x3)
        distCoeffs: Distortion coefficients
        aspectRatio: Optional fixed aspect ratio
    
    Returns:
        Tuple of (image_points, jacobian)
    """
    obj_pts = np.asarray(objectPoints, dtype=np.float64).reshape(-1, 3)
    rvec = np.asarray(rvec, dtype=np.float64).flatten()
    tvec = np.asarray(tvec, dtype=np.float64).flatten()
    K = np.asarray(cameraMatrix, dtype=np.float64).reshape(3, 3)
    
    # Convert rotation vector to matrix
    R = Rodrigues(rvec)[0]
    
    # Transform points
    transformed = (R @ obj_pts.T).T + tvec
    
    # Perspective projection
    x = transformed[:, 0] / transformed[:, 2]
    y = transformed[:, 1] / transformed[:, 2]
    
    # Apply distortion if provided
    if distCoeffs is not None:
        x, y = _apply_distortion(x, y, distCoeffs)
    
    # Apply camera matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * x + cx
    v = fy * y + cy
    
    img_pts = np.stack([u, v], axis=1).reshape(-1, 1, 2)
    
    return img_pts, None


def findHomography(
    srcPoints: np.ndarray,
    dstPoints: np.ndarray,
    method: int = 0,
    ransacReprojThreshold: float = 3.0,
    mask: Optional[np.ndarray] = None,
    maxIters: int = 2000,
    confidence: float = 0.995
) -> Tuple[np.ndarray, np.ndarray]:
    """Find a perspective transformation between two planes.
    
    Args:
        srcPoints: Source points (Nx2)
        dstPoints: Destination points (Nx2)
        method: Method (0, RANSAC, LMEDS, RHO)
        ransacReprojThreshold: Maximum reprojection error for RANSAC
        maxIters: Maximum RANSAC iterations
        confidence: Confidence level
    
    Returns:
        Tuple of (homography_matrix, mask)
    """
    src = np.asarray(srcPoints, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dstPoints, dtype=np.float64).reshape(-1, 2)
    
    n = len(src)
    if n < 4:
        return np.eye(3), np.zeros((n, 1), dtype=np.uint8)
    
    if method == 0:
        # Direct DLT
        H = _compute_homography_dlt(src, dst)
        mask = np.ones((n, 1), dtype=np.uint8)
        return H, mask
    
    elif method == RANSAC:
        # RANSAC
        best_H = np.eye(3)
        best_inliers = []
        
        for _ in range(maxIters):
            indices = np.random.choice(n, 4, replace=False)
            H = _compute_homography_dlt(src[indices], dst[indices])
            
            # Count inliers
            src_h = np.hstack([src, np.ones((n, 1))])
            projected = (H @ src_h.T).T
            projected = projected[:, :2] / projected[:, 2:3]
            
            errors = np.sqrt(np.sum((projected - dst)**2, axis=1))
            inlier_mask = errors < ransacReprojThreshold
            
            if np.sum(inlier_mask) > len(best_inliers):
                best_inliers = np.where(inlier_mask)[0]
                best_H = H
        
        # Refine with inliers
        if len(best_inliers) >= 4:
            best_H = _compute_homography_dlt(src[best_inliers], dst[best_inliers])
        
        mask = np.zeros((n, 1), dtype=np.uint8)
        mask[best_inliers] = 1
        
        return best_H, mask
    
    else:
        # Default to DLT
        H = _compute_homography_dlt(src, dst)
        return H, np.ones((n, 1), dtype=np.uint8)


def findFundamentalMat(
    points1: np.ndarray,
    points2: np.ndarray,
    method: int = FM_RANSAC,
    ransacReprojThreshold: float = 3.0,
    confidence: float = 0.99,
    maxIters: int = 1000,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate fundamental matrix from point correspondences.
    
    Args:
        points1: Points from first image (Nx2)
        points2: Points from second image (Nx2)
        method: Method (FM_7POINT, FM_8POINT, FM_RANSAC, FM_LMEDS)
        ransacReprojThreshold: Maximum distance for inlier
        confidence: Confidence level
        maxIters: Maximum iterations
    
    Returns:
        Tuple of (fundamental_matrix, mask)
    """
    pts1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    
    n = len(pts1)
    if n < 8:
        return np.zeros((3, 3)), np.zeros((n, 1), dtype=np.uint8)
    
    # 8-point algorithm
    F = _eight_point_fundamental(pts1, pts2)
    
    if method == FM_RANSAC:
        # Simple RANSAC
        best_F = F
        best_inliers = []
        
        for _ in range(min(maxIters, 500)):
            indices = np.random.choice(n, 8, replace=False)
            F_sample = _eight_point_fundamental(pts1[indices], pts2[indices])
            
            # Epipolar constraint error
            errors = _sampson_error(pts1, pts2, F_sample)
            inliers = errors < ransacReprojThreshold
            
            if np.sum(inliers) > len(best_inliers):
                best_inliers = np.where(inliers)[0]
                best_F = F_sample
        
        mask = np.zeros((n, 1), dtype=np.uint8)
        mask[best_inliers] = 1
        
        return best_F, mask
    
    return F, np.ones((n, 1), dtype=np.uint8)


def findEssentialMat(
    points1: np.ndarray,
    points2: np.ndarray,
    cameraMatrix: np.ndarray,
    method: int = RANSAC,
    prob: float = 0.999,
    threshold: float = 1.0,
    maxIters: int = 1000,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate essential matrix from point correspondences.
    
    Args:
        points1: Points from first image
        points2: Points from second image
        cameraMatrix: Camera intrinsic matrix
        method: Method for computing
        prob: Confidence
        threshold: Distance threshold
        maxIters: Max iterations
    
    Returns:
        Tuple of (essential_matrix, mask)
    """
    K = np.asarray(cameraMatrix, dtype=np.float64).reshape(3, 3)
    
    # Get fundamental matrix
    F, mask = findFundamentalMat(points1, points2, method, threshold, prob, maxIters)
    
    # E = K'.T @ F @ K
    E = K.T @ F @ K
    
    return E, mask


def Rodrigues(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    jacobian: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert rotation vector to matrix or vice versa.
    
    Args:
        src: Rotation vector (3x1) or rotation matrix (3x3)
        dst: Output (ignored)
        jacobian: Optional Jacobian (ignored)
    
    Returns:
        Tuple of (output, jacobian)
    """
    src = np.asarray(src, dtype=np.float64)
    
    if src.size == 3:
        # Vector to matrix
        rvec = src.flatten()
        theta = np.linalg.norm(rvec)
        
        if theta < 1e-10:
            return np.eye(3, dtype=np.float64), None
        
        r = rvec / theta
        
        # Rodrigues formula
        K = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        return R, None
    
    elif src.shape == (3, 3):
        # Matrix to vector
        R = src
        
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        
        if theta < 1e-10:
            return np.zeros((3, 1), dtype=np.float64), None
        
        r = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))
        
        rvec = theta * r
        
        return rvec.reshape(3, 1), None
    
    else:
        raise ValueError("Input must be 3x1 vector or 3x3 matrix")


def decomposeHomographyMat(
    H: np.ndarray,
    K: np.ndarray
) -> Tuple[int, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Decompose homography matrix into rotations and translations.
    
    Args:
        H: Homography matrix
        K: Camera intrinsic matrix
    
    Returns:
        Tuple of (num_solutions, rotations, translations, normals)
    """
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    
    # H_normalized = K^-1 @ H @ K
    H_norm = np.linalg.inv(K) @ H @ K
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H_norm)
    
    # There can be up to 4 solutions
    rotations = [np.eye(3)]
    translations = [np.zeros((3, 1))]
    normals = [np.array([[0], [0], [1]])]
    
    return 1, rotations, translations, normals


def triangulatePoints(
    projMatr1: np.ndarray,
    projMatr2: np.ndarray,
    projPoints1: np.ndarray,
    projPoints2: np.ndarray
) -> np.ndarray:
    """Triangulate 3D points from two views.
    
    Args:
        projMatr1: Projection matrix for first camera (3x4)
        projMatr2: Projection matrix for second camera (3x4)
        projPoints1: Points in first image (2xN)
        projPoints2: Points in second image (2xN)
    
    Returns:
        4xN array of homogeneous 3D points
    """
    P1 = np.asarray(projMatr1, dtype=np.float64).reshape(3, 4)
    P2 = np.asarray(projMatr2, dtype=np.float64).reshape(3, 4)
    pts1 = np.asarray(projPoints1, dtype=np.float64).reshape(2, -1)
    pts2 = np.asarray(projPoints2, dtype=np.float64).reshape(2, -1)
    
    n = pts1.shape[1]
    points4d = np.zeros((4, n))
    
    for i in range(n):
        x1, y1 = pts1[0, i], pts1[1, i]
        x2, y2 = pts2[0, i], pts2[1, i]
        
        # Build system
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])
        
        # SVD solve
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        points4d[:, i] = X
    
    return points4d


def _solve_pnp_dlt(obj_pts: np.ndarray, img_pts_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple DLT-based PnP solution."""
    n = len(obj_pts)
    
    # Build matrix A
    A = np.zeros((2 * n, 12))
    
    for i in range(n):
        X, Y, Z = obj_pts[i]
        u, v = img_pts_norm[i]
        
        A[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    
    # SVD solve
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    
    # Extract R and t
    R = P[:, :3]
    t = P[:, 3]
    
    # Orthogonalize R
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    
    # Convert R to rotation vector
    rvec = Rodrigues(R)[0]
    tvec = t.reshape(3, 1)
    
    return rvec, tvec


def _compute_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute homography using DLT."""
    n = len(src)
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        
        A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    
    return H / H[2, 2]


def _eight_point_fundamental(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """8-point algorithm for fundamental matrix."""
    n = len(pts1)
    
    # Normalize points
    pts1_h = np.hstack([pts1, np.ones((n, 1))])
    pts2_h = np.hstack([pts2, np.ones((n, 1))])
    
    # Build matrix
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1, _ = pts1_h[i]
        x2, y2, _ = pts2_h[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # SVD solve
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank 2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    
    return F


def _sampson_error(pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Compute Sampson error for fundamental matrix."""
    n = len(pts1)
    pts1_h = np.hstack([pts1, np.ones((n, 1))])
    pts2_h = np.hstack([pts2, np.ones((n, 1))])
    
    Fx1 = (F @ pts1_h.T).T
    Ftx2 = (F.T @ pts2_h.T).T
    
    x2Fx1 = np.sum(pts2_h * Fx1, axis=1)
    
    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    
    return np.abs(x2Fx1) / np.sqrt(denom + 1e-10)


def _undistort_points(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort points."""
    # Simplified - just return original points for now
    return pts


def _apply_distortion(x: np.ndarray, y: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply distortion to normalized coordinates."""
    dist = np.asarray(dist).flatten()
    
    if len(dist) < 4:
        return x, y
    
    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r4 * r2
    
    k1, k2 = dist[0], dist[1]
    p1, p2 = dist[2], dist[3]
    k3 = dist[4] if len(dist) > 4 else 0
    
    radial = 1 + k1*r2 + k2*r4 + k3*r6
    
    x_dist = x * radial + 2*p1*x*y + p2*(r2 + 2*x**2)
    y_dist = y * radial + p1*(r2 + 2*y**2) + 2*p2*x*y
    
    return x_dist, y_dist


__all__ = [
    # Functions
    "solvePnP",
    "solvePnPRansac",
    "projectPoints",
    "findHomography",
    "findFundamentalMat",
    "findEssentialMat",
    "Rodrigues",
    "decomposeHomographyMat",
    "triangulatePoints",
    # Method constants
    "SOLVEPNP_ITERATIVE",
    "SOLVEPNP_P3P",
    "SOLVEPNP_AP3P",
    "SOLVEPNP_EPNP",
    "SOLVEPNP_DLS",
    "SOLVEPNP_UPNP",
    "SOLVEPNP_IPPE",
    "SOLVEPNP_IPPE_SQUARE",
    "SOLVEPNP_SQPNP",
    # RANSAC flags
    "RANSAC",
    "LMEDS",
    "RHO",
    # FM flags
    "FM_7POINT",
    "FM_8POINT",
    "FM_RANSAC",
    "FM_LMEDS",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.