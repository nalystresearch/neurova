# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Bilateral filtering and edge-preserving filters for Neurova.

Provides bilateralFilter, edgePreservingFilter, and related functions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def bilateralFilter(
    src: np.ndarray,
    d: int,
    sigmaColor: float,
    sigmaSpace: float,
    dst: Optional[np.ndarray] = None,
    borderType: int = 4  # BORDER_DEFAULT
) -> np.ndarray:
    """Apply bilateral filter to an image.
    
    Bilateral filtering smooths images while keeping edges sharp.
    It replaces the intensity of each pixel with a weighted average
    of nearby pixels, where weights depend on both spatial distance
    and intensity difference.
    
    Args:
        src: Source 8-bit or floating-point, 1-channel or 3-channel image
        d: Diameter of each pixel neighborhood. If negative, computed from sigmaSpace
        sigmaColor: Filter sigma in the color space
        sigmaSpace: Filter sigma in the coordinate space
        dst: Destination image (ignored, for compatibility)
        borderType: Border mode (ignored)
    
    Returns:
        Filtered image of same size and type as src
    """
    img = np.asarray(src, dtype=np.float64)
    
    # Determine kernel size
    if d <= 0:
        d = int(round(sigmaSpace * 6)) | 1  # Make odd
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
    radius = d // 2
    
    # Handle color images
    if img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            channels.append(_bilateral_single_channel(
                img[:, :, c], radius, sigmaColor, sigmaSpace))
        result = np.stack(channels, axis=2)
    else:
        result = _bilateral_single_channel(img, radius, sigmaColor, sigmaSpace)
    
    # Convert back to original dtype
    if src.dtype == np.uint8:
        return np.clip(result, 0, 255).astype(np.uint8)
    return result.astype(src.dtype)


def _bilateral_single_channel(
    img: np.ndarray,
    radius: int,
    sigma_color: float,
    sigma_space: float
) -> np.ndarray:
    """Apply bilateral filter to single channel image."""
    h, w = img.shape
    result = np.zeros_like(img)
    
    # Precompute spatial Gaussian weights
    y_range = np.arange(-radius, radius + 1)
    x_range = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(y_range, x_range, indexing='ij')
    spatial_weights = np.exp(-(xx**2 + yy**2) / (2 * sigma_space**2))
    
    # Pad image
    padded = np.pad(img, radius, mode='reflect')
    
    # Apply filter
    for i in range(h):
        for j in range(w):
            # Extract neighborhood
            neighborhood = padded[i:i+2*radius+1, j:j+2*radius+1]
            center_val = img[i, j]
            
            # Compute color weights
            color_diff = neighborhood - center_val
            color_weights = np.exp(-(color_diff**2) / (2 * sigma_color**2))
            
            # Combine weights
            weights = spatial_weights * color_weights
            weights_sum = np.sum(weights)
            
            if weights_sum > 1e-10:
                result[i, j] = np.sum(neighborhood * weights) / weights_sum
            else:
                result[i, j] = center_val
    
    return result


def boxFilter(
    src: np.ndarray,
    ddepth: int,
    ksize: tuple,
    dst: Optional[np.ndarray] = None,
    anchor: tuple = (-1, -1),
    normalize: bool = True,
    borderType: int = 4
) -> np.ndarray:
    """Blur an image using a box filter.
    
    Args:
        src: Input image
        ddepth: Output image depth (-1 for same as src)
        ksize: Blurring kernel size (width, height)
        dst: Output image (ignored)
        anchor: Anchor point (ignored)
        normalize: Whether to normalize the filter
        borderType: Border mode (ignored)
    
    Returns:
        Blurred image
    """
    img = np.asarray(src, dtype=np.float64)
    kw, kh = ksize
    
    kernel = np.ones((kh, kw), dtype=np.float64)
    if normalize:
        kernel /= (kh * kw)
    
    if HAS_SCIPY:
        if img.ndim == 3:
            result = np.zeros_like(img)
            for c in range(img.shape[2]):
                result[:, :, c] = ndimage.convolve(img[:, :, c], kernel, mode='reflect')
        else:
            result = ndimage.convolve(img, kernel, mode='reflect')
    else:
        result = _convolve2d(img, kernel)
    
    if src.dtype == np.uint8:
        return np.clip(result, 0, 255).astype(np.uint8)
    return result.astype(src.dtype)


def sqrBoxFilter(
    src: np.ndarray,
    ddepth: int,
    ksize: tuple,
    dst: Optional[np.ndarray] = None,
    anchor: tuple = (-1, -1),
    normalize: bool = True,
    borderType: int = 4
) -> np.ndarray:
    """Calculate the normalized sum of squares of the pixel values.
    
    Args:
        src: Input image
        ddepth: Output image depth
        ksize: Kernel size
        dst: Output image (ignored)
        anchor: Anchor point (ignored)
        normalize: Whether to normalize
        borderType: Border mode (ignored)
    
    Returns:
        Image with sum of squares in each neighborhood
    """
    img = np.asarray(src, dtype=np.float64)
    img_sq = img ** 2
    
    return boxFilter(img_sq, ddepth, ksize, normalize=normalize)


def getGaussianKernel(
    ksize: int,
    sigma: float,
    ktype: int = 6  # CV_64F
) -> np.ndarray:
    """Return Gaussian filter coefficients.
    
    Args:
        ksize: Aperture size (must be odd and positive)
        sigma: Gaussian standard deviation. If <= 0, computed from ksize
        ktype: Type of filter coefficients (ignored, always float64)
    
    Returns:
        1D Gaussian kernel (ksize x 1)
    """
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be positive and odd")
    
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    
    x = np.arange(ksize) - ksize // 2
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    
    return kernel.reshape(-1, 1)


def getGaborKernel(
    ksize: tuple,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float,
    psi: float = np.pi * 0.5,
    ktype: int = 6
) -> np.ndarray:
    """Return Gabor filter coefficients.
    
    Args:
        ksize: Size of the filter (width, height)
        sigma: Standard deviation of the gaussian envelope
        theta: Orientation of the normal to the parallel stripes
        lambd: Wavelength of the sinusoidal factor
        gamma: Spatial aspect ratio
        psi: Phase offset
        ktype: Type of filter coefficients (ignored)
    
    Returns:
        Gabor filter kernel
    """
    kw, kh = ksize
    
    # Generate coordinate grids
    y, x = np.mgrid[-(kh//2):kh//2+1, -(kw//2):kw//2+1]
    
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    # Gabor function
    gb = np.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
    gb *= np.cos(2 * np.pi * x_theta / lambd + psi)
    
    # Resize to exact ksize
    gb = gb[:kh, :kw]
    
    return gb.astype(np.float64)


def getDerivKernels(
    dx: int,
    dy: int,
    ksize: int,
    normalize: bool = False,
    ktype: int = 6
) -> tuple:
    """Return filter coefficients for computing spatial derivatives.
    
    Args:
        dx: Derivative order in x
        dy: Derivative order in y
        ksize: Aperture size
        normalize: Whether to normalize
        ktype: Type of coefficients (ignored)
    
    Returns:
        Tuple of (kx, ky) 1D kernels
    """
    if ksize == 1:
        kx = np.array([1.0])
        ky = np.array([1.0])
    elif ksize == 3:
        # Sobel kernels
        if dx == 0:
            kx = np.array([1.0, 2.0, 1.0])
        elif dx == 1:
            kx = np.array([-1.0, 0.0, 1.0])
        elif dx == 2:
            kx = np.array([1.0, -2.0, 1.0])
        else:
            kx = np.array([1.0])
        
        if dy == 0:
            ky = np.array([1.0, 2.0, 1.0])
        elif dy == 1:
            ky = np.array([-1.0, 0.0, 1.0])
        elif dy == 2:
            ky = np.array([1.0, -2.0, 1.0])
        else:
            ky = np.array([1.0])
    else:
        # Larger kernels - use binomial expansion
        kx = _deriv_kernel(ksize, dx)
        ky = _deriv_kernel(ksize, dy)
    
    if normalize:
        kx = kx / np.sum(np.abs(kx)) if np.sum(np.abs(kx)) > 0 else kx
        ky = ky / np.sum(np.abs(ky)) if np.sum(np.abs(ky)) > 0 else ky
    
    return kx.reshape(-1, 1), ky.reshape(-1, 1)


def _deriv_kernel(ksize: int, order: int) -> np.ndarray:
    """Generate derivative kernel of given size and order."""
    # Start with smoothing kernel (binomial)
    kernel = np.array([1.0])
    for _ in range(ksize - 1):
        kernel = np.convolve(kernel, [1, 1])
    
    # Apply derivatives
    for _ in range(order):
        kernel = np.convolve(kernel, [1, -1])[:-1]
    
    # Resize to ksize
    if len(kernel) > ksize:
        kernel = kernel[:ksize]
    elif len(kernel) < ksize:
        kernel = np.pad(kernel, (0, ksize - len(kernel)))
    
    return kernel


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution fallback."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    result = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return result


__all__ = [
    "bilateralFilter",
    "boxFilter",
    "sqrBoxFilter",
    "getGaussianKernel",
    "getGaborKernel",
    "getDerivKernels",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.