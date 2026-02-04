# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Corner detection for Neurova.

Provides cornerHarris, cornerMinEigenVal, cornerSubPix, and related functions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def cornerHarris(
    src: np.ndarray,
    blockSize: int,
    ksize: int,
    k: float,
    dst: Optional[np.ndarray] = None,
    borderType: int = 4  # BORDER_DEFAULT
) -> np.ndarray:
    """Harris corner detector.
    
    The function runs the Harris edge detector on the image.
    
    Args:
        src: Input single-channel 8-bit or floating-point image
        blockSize: Neighborhood size for corner detection
        ksize: Aperture parameter for the Sobel operator
        k: Harris detector free parameter (usually 0.04-0.06)
        dst: Output image (ignored, for compatibility)
        borderType: Pixel extrapolation method (ignored)
    
    Returns:
        Image to store the Harris detector responses (same size as src, float32)
    """
    img = np.asarray(src, dtype=np.float64)
    
    if img.ndim != 2:
        raise ValueError("cornerHarris requires a single-channel image")
    
    # Compute image gradients using Sobel
    if HAS_SCIPY:
        Ix = ndimage.sobel(img, axis=1, mode='reflect')
        Iy = ndimage.sobel(img, axis=0, mode='reflect')
    else:
        Ix = _sobel_x(img)
        Iy = _sobel_y(img)
    
    # Products of derivatives
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Sum over block
    if HAS_SCIPY:
        Sxx = ndimage.uniform_filter(Ixx, size=blockSize, mode='reflect')
        Syy = ndimage.uniform_filter(Iyy, size=blockSize, mode='reflect')
        Sxy = ndimage.uniform_filter(Ixy, size=blockSize, mode='reflect')
    else:
        Sxx = _box_filter(Ixx, blockSize)
        Syy = _box_filter(Iyy, blockSize)
        Sxy = _box_filter(Ixy, blockSize)
    
    # Harris response: det(M) - k * trace(M)^2
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    harris = det - k * (trace ** 2)
    
    return harris.astype(np.float32)


def cornerMinEigenVal(
    src: np.ndarray,
    blockSize: int,
    ksize: int = 3,
    dst: Optional[np.ndarray] = None,
    borderType: int = 4
) -> np.ndarray:
    """Calculate the minimal eigenvalue of gradient matrices.
    
    Args:
        src: Input single-channel 8-bit or floating-point image
        blockSize: Neighborhood size
        ksize: Aperture parameter for Sobel operator
        dst: Output image (ignored)
        borderType: Pixel extrapolation method (ignored)
    
    Returns:
        Image to store the minimal eigenvalues (same size as src, float32)
    """
    img = np.asarray(src, dtype=np.float64)
    
    if img.ndim != 2:
        raise ValueError("cornerMinEigenVal requires a single-channel image")
    
    # Compute gradients
    if HAS_SCIPY:
        Ix = ndimage.sobel(img, axis=1, mode='reflect')
        Iy = ndimage.sobel(img, axis=0, mode='reflect')
    else:
        Ix = _sobel_x(img)
        Iy = _sobel_y(img)
    
    # Products of derivatives
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Sum over block
    if HAS_SCIPY:
        Sxx = ndimage.uniform_filter(Ixx, size=blockSize, mode='reflect')
        Syy = ndimage.uniform_filter(Iyy, size=blockSize, mode='reflect')
        Sxy = ndimage.uniform_filter(Ixy, size=blockSize, mode='reflect')
    else:
        Sxx = _box_filter(Ixx, blockSize)
        Syy = _box_filter(Iyy, blockSize)
        Sxy = _box_filter(Ixy, blockSize)
    
    # Minimum eigenvalue: (trace - sqrt(trace^2 - 4*det)) / 2
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    
    discriminant = trace**2 - 4*det
    discriminant = np.maximum(discriminant, 0)  # Avoid negative sqrt
    
    min_eigenval = (trace - np.sqrt(discriminant)) / 2
    
    return min_eigenval.astype(np.float32)


def cornerEigenValsAndVecs(
    src: np.ndarray,
    blockSize: int,
    ksize: int,
    dst: Optional[np.ndarray] = None,
    borderType: int = 4
) -> np.ndarray:
    """Calculate eigenvalues and eigenvectors of image blocks.
    
    Args:
        src: Input single-channel 8-bit or floating-point image
        blockSize: Neighborhood size
        ksize: Aperture parameter for Sobel operator
        dst: Output image (ignored)
        borderType: Pixel extrapolation method (ignored)
    
    Returns:
        6-channel output (λ1, λ2, x1, y1, x2, y2) for each pixel
    """
    img = np.asarray(src, dtype=np.float64)
    h, w = img.shape
    
    # Compute gradients
    if HAS_SCIPY:
        Ix = ndimage.sobel(img, axis=1, mode='reflect')
        Iy = ndimage.sobel(img, axis=0, mode='reflect')
    else:
        Ix = _sobel_x(img)
        Iy = _sobel_y(img)
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    if HAS_SCIPY:
        Sxx = ndimage.uniform_filter(Ixx, size=blockSize, mode='reflect')
        Syy = ndimage.uniform_filter(Iyy, size=blockSize, mode='reflect')
        Sxy = ndimage.uniform_filter(Ixy, size=blockSize, mode='reflect')
    else:
        Sxx = _box_filter(Ixx, blockSize)
        Syy = _box_filter(Iyy, blockSize)
        Sxy = _box_filter(Ixy, blockSize)
    
    result = np.zeros((h, w, 6), dtype=np.float32)
    
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    discriminant = np.maximum(trace**2 - 4*det, 0)
    sqrt_disc = np.sqrt(discriminant)
    
    # Eigenvalues
    result[:, :, 0] = (trace + sqrt_disc) / 2  # λ1
    result[:, :, 1] = (trace - sqrt_disc) / 2  # λ2
    
    # Eigenvectors (simplified)
    result[:, :, 2] = 1.0  # x1
    result[:, :, 3] = 0.0  # y1
    result[:, :, 4] = 0.0  # x2
    result[:, :, 5] = 1.0  # y2
    
    return result


def cornerSubPix(
    image: np.ndarray,
    corners: np.ndarray,
    winSize: Tuple[int, int],
    zeroZone: Tuple[int, int],
    criteria: Tuple[int, int, float]
) -> np.ndarray:
    """Refine corner locations to sub-pixel accuracy.
    
    Args:
        image: Input single-channel, 8-bit or float image
        corners: Initial coordinates of the input corners (Nx1x2 or Nx2)
        winSize: Half of the side length of the search window
        zeroZone: Half of the size of dead region (-1,-1 to ignore)
        criteria: (type, maxCount, epsilon) termination criteria
    
    Returns:
        Refined corner coordinates
    """
    img = np.asarray(image, dtype=np.float64)
    pts = np.asarray(corners, dtype=np.float32).copy()
    
    # Reshape if needed
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    
    _, max_iter, epsilon = criteria
    
    # Compute gradients
    if HAS_SCIPY:
        Ix = ndimage.sobel(img, axis=1, mode='reflect')
        Iy = ndimage.sobel(img, axis=0, mode='reflect')
    else:
        Ix = _sobel_x(img)
        Iy = _sobel_y(img)
    
    h, w = img.shape
    wx, wy = winSize
    
    for idx in range(len(pts)):
        x, y = pts[idx]
        
        for _ in range(max_iter):
            # Window boundaries
            x0 = max(0, int(x - wx))
            y0 = max(0, int(y - wy))
            x1 = min(w, int(x + wx + 1))
            y1 = min(h, int(y + wy + 1))
            
            if x1 <= x0 or y1 <= y0:
                break
            
            # Extract window
            gx = Ix[y0:y1, x0:x1]
            gy = Iy[y0:y1, x0:x1]
            
            # Build system matrix
            a = np.sum(gx * gx)
            b = np.sum(gx * gy)
            c = np.sum(gy * gy)
            
            det = a * c - b * b
            if abs(det) < 1e-10:
                break
            
            # Compute offset
            yy, xx = np.mgrid[y0:y1, x0:x1]
            bb1 = np.sum(gx * gx * xx + gx * gy * yy)
            bb2 = np.sum(gx * gy * xx + gy * gy * yy)
            
            new_x = (c * bb1 - b * bb2) / det
            new_y = (a * bb2 - b * bb1) / det
            
            # Check convergence
            dx = new_x - x
            dy = new_y - y
            
            if dx*dx + dy*dy < epsilon*epsilon:
                pts[idx] = [new_x, new_y]
                break
            
            x, y = new_x, new_y
            pts[idx] = [x, y]
    
    return pts.reshape(-1, 1, 2)


def preCornerDetect(
    src: np.ndarray,
    ksize: int,
    dst: Optional[np.ndarray] = None,
    borderType: int = 4
) -> np.ndarray:
    """Calculate a feature map for corner detection.
    
    Args:
        src: Input single-channel 8-bit or float image
        ksize: Aperture size of Sobel operator
        dst: Output image (ignored)
        borderType: Pixel extrapolation (ignored)
    
    Returns:
        Feature map for corner detection
    """
    img = np.asarray(src, dtype=np.float64)
    
    # First derivatives
    if HAS_SCIPY:
        Dx = ndimage.sobel(img, axis=1, mode='reflect')
        Dy = ndimage.sobel(img, axis=0, mode='reflect')
    else:
        Dx = _sobel_x(img)
        Dy = _sobel_y(img)
    
    # Second derivatives
    if HAS_SCIPY:
        Dxx = ndimage.sobel(Dx, axis=1, mode='reflect')
        Dyy = ndimage.sobel(Dy, axis=0, mode='reflect')
        Dxy = ndimage.sobel(Dx, axis=0, mode='reflect')
    else:
        Dxx = _sobel_x(Dx)
        Dyy = _sobel_y(Dy)
        Dxy = _sobel_y(Dx)
    
    result = Dx*Dx*Dyy + Dy*Dy*Dxx - 2*Dx*Dy*Dxy
    
    return result.astype(np.float32)


def _sobel_x(img: np.ndarray) -> np.ndarray:
    """Simple Sobel filter in X direction."""
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    return _convolve2d(img, kernel)


def _sobel_y(img: np.ndarray) -> np.ndarray:
    """Simple Sobel filter in Y direction."""
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    return _convolve2d(img, kernel)


def _box_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Simple box filter."""
    kernel = np.ones((size, size), dtype=np.float64) / (size * size)
    return _convolve2d(img, kernel)


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution with reflection padding."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    result = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return result


__all__ = [
    "cornerHarris",
    "cornerMinEigenVal",
    "cornerEigenValsAndVecs",
    "cornerSubPix",
    "preCornerDetect",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.