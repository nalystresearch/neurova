# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova.photo - Computational Photography

Provides photo processing functions including denoising,
inpainting, and seamless cloning.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np


# Inpainting methods
INPAINT_NS = 0  # Navier-Stokes
INPAINT_TELEA = 1  # Alexandru Telea's method


def fastNlMeansDenoising(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    h: float = 3.0,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21
) -> np.ndarray:
    """Perform image denoising using Non-local Means for grayscale images.
    
    Args:
        src: Input 8-bit grayscale image
        dst: Optional output image
        h: Filter strength (higher h removes more noise but loses detail)
        templateWindowSize: Size of template patch (odd number)
        searchWindowSize: Size of search window (odd number)
    
    Returns:
        Denoised image
    """
    if src.ndim != 2:
        raise ValueError("Input must be grayscale. Use fastNlMeansDenoisingColored for color images.")
    
    src = src.astype(np.float32)
    h, w = src.shape
    
    half_template = templateWindowSize // 2
    half_search = searchWindowSize // 2
    
    result = np.zeros_like(src)
    
    # Pad image
    padded = np.pad(src, half_search + half_template, mode='reflect')
    
    h_param_sq = h * h
    
    for y in range(h):
        for x in range(w):
            # Extract template
            ty = y + half_search + half_template
            tx = x + half_search + half_template
            
            template = padded[ty - half_template:ty + half_template + 1,
                             tx - half_template:tx + half_template + 1]
            
            weights_sum = 0.0
            weighted_sum = 0.0
            
            # Search in neighborhood
            for sy in range(y, y + searchWindowSize):
                for sx in range(x, x + searchWindowSize):
                    # Extract patch at search location
                    patch = padded[sy:sy + templateWindowSize,
                                  sx:sx + templateWindowSize]
                    
                    # Compute distance
                    diff = template - patch
                    dist = np.sum(diff * diff) / (templateWindowSize * templateWindowSize)
                    
                    # Compute weight
                    weight = np.exp(-dist / h_param_sq)
                    
                    # Accumulate
                    cy = sy + half_template
                    cx = sx + half_template
                    weighted_sum += weight * padded[cy, cx]
                    weights_sum += weight
            
            # Normalize
            if weights_sum > 0:
                result[y, x] = weighted_sum / weights_sum
            else:
                result[y, x] = src[y, x]
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def fastNlMeansDenoisingColored(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    h: float = 3.0,
    hForColorComponents: float = 3.0,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21
) -> np.ndarray:
    """Perform image denoising using Non-local Means for color images.
    
    Args:
        src: Input 8-bit color image (BGR)
        dst: Optional output image
        h: Filter strength for luminance
        hForColorComponents: Filter strength for color components
        templateWindowSize: Size of template patch
        searchWindowSize: Size of search window
    
    Returns:
        Denoised color image
    """
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("Input must be a 3-channel color image")
    
    # Convert to LAB color space for better denoising
    from ..imgproc.color import cvtColor, COLOR_BGR2LAB, COLOR_LAB2BGR
    
    lab = cvtColor(src, COLOR_BGR2LAB)
    
    # Denoise each channel
    l_channel = fastNlMeansDenoising(lab[:, :, 0], h=h, 
                                     templateWindowSize=templateWindowSize,
                                     searchWindowSize=searchWindowSize)
    a_channel = fastNlMeansDenoising(lab[:, :, 1], h=hForColorComponents,
                                     templateWindowSize=templateWindowSize,
                                     searchWindowSize=searchWindowSize)
    b_channel = fastNlMeansDenoising(lab[:, :, 2], h=hForColorComponents,
                                     templateWindowSize=templateWindowSize,
                                     searchWindowSize=searchWindowSize)
    
    # Merge and convert back
    denoised_lab = np.stack([l_channel, a_channel, b_channel], axis=2)
    result = cvtColor(denoised_lab, COLOR_LAB2BGR)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def inpaint(
    src: np.ndarray,
    inpaintMask: np.ndarray,
    inpaintRadius: float,
    flags: int = INPAINT_TELEA
) -> np.ndarray:
    """Restore selected regions using inpainting.
    
    Args:
        src: Input image (8-bit, 1 or 3 channel)
        inpaintMask: Mask where non-zero pixels indicate area to inpaint
        inpaintRadius: Radius of circular neighborhood for inpainting
        flags: INPAINT_NS or INPAINT_TELEA
    
    Returns:
        Inpainted image
    """
    if src.ndim == 2:
        is_gray = True
        src = src[:, :, np.newaxis]
    else:
        is_gray = False
    
    result = src.copy().astype(np.float32)
    mask = inpaintMask > 0
    
    h, w, c = src.shape
    radius = int(np.ceil(inpaintRadius))
    
    if flags == INPAINT_TELEA:
        # Fast Marching Method based inpainting
        result = _inpaint_telea(result, mask, radius)
    else:
        # Navier-Stokes based inpainting
        result = _inpaint_ns(result, mask, radius)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    if is_gray:
        result = result[:, :, 0]
    
    return result


def _inpaint_telea(
    img: np.ndarray,
    mask: np.ndarray,
    radius: int
) -> np.ndarray:
    """Telea's inpainting algorithm (simplified)."""
    result = img.copy()
    h, w, c = img.shape
    
    # Get boundary pixels
    from scipy.ndimage import binary_dilation, distance_transform_edt
    
    # Compute distance transform
    dist = distance_transform_edt(mask)
    
    # Process pixels in order of increasing distance
    max_dist = int(dist.max()) + 1
    
    for d in range(1, max_dist + 1):
        # Get pixels at this distance level
        level_mask = (dist > d - 1) & (dist <= d) & mask
        
        if not np.any(level_mask):
            continue
        
        coords = np.argwhere(level_mask)
        
        for y, x in coords:
            # Find valid neighbors within radius
            values = []
            weights = []
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    
                    if 0 <= ny < h and 0 <= nx < w:
                        # Only use non-masked pixels or already inpainted
                        if not mask[ny, nx] or dist[ny, nx] < dist[y, x]:
                            # Weight by distance
                            spatial_dist = np.sqrt(dy*dy + dx*dx)
                            if spatial_dist < radius:
                                weight = 1.0 / (spatial_dist + 0.1)
                                values.append(result[ny, nx])
                                weights.append(weight)
            
            if values:
                values = np.array(values)
                weights = np.array(weights)
                weights = weights / weights.sum()
                result[y, x] = np.sum(values * weights[:, np.newaxis], axis=0)
    
    return result


def _inpaint_ns(
    img: np.ndarray,
    mask: np.ndarray,
    radius: int
) -> np.ndarray:
    """Navier-Stokes based inpainting (simplified diffusion)."""
    result = img.copy()
    h, w, c = img.shape
    
    # Iterative diffusion
    iterations = max(50, radius * 10)
    
    for _ in range(iterations):
        new_result = result.copy()
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if mask[y, x]:
                    # Laplacian diffusion
                    neighbors = [
                        result[y-1, x],
                        result[y+1, x],
                        result[y, x-1],
                        result[y, x+1]
                    ]
                    new_result[y, x] = np.mean(neighbors, axis=0)
        
        result = new_result
    
    return result


def seamlessClone(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    p: Tuple[int, int],
    flags: int = 1  # NORMAL_CLONE
) -> np.ndarray:
    """Seamlessly clone a source image into a destination.
    
    Args:
        src: Source image to clone
        dst: Destination image
        mask: Mask determining region to clone
        p: Position (x, y) in destination for center of clone
        flags: Cloning mode (NORMAL_CLONE=1, MIXED_CLONE=2, MONOCHROME_TRANSFER=3)
    
    Returns:
        Blended result
    """
    NORMAL_CLONE = 1
    MIXED_CLONE = 2
    MONOCHROME_TRANSFER = 3
    
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    result = dst.copy()
    
    # Find mask bounding box
    mask_points = np.argwhere(mask > 0)
    if len(mask_points) == 0:
        return result.astype(np.uint8)
    
    y_min, x_min = mask_points.min(axis=0)
    y_max, x_max = mask_points.max(axis=0)
    
    mask_h = y_max - y_min + 1
    mask_w = x_max - x_min + 1
    
    # Destination position
    cx, cy = p
    dst_y1 = cy - mask_h // 2
    dst_x1 = cx - mask_w // 2
    dst_y2 = dst_y1 + mask_h
    dst_x2 = dst_x1 + mask_w
    
    # Ensure within bounds
    src_y1 = y_min
    src_x1 = x_min
    
    if dst_y1 < 0:
        src_y1 -= dst_y1
        dst_y1 = 0
    if dst_x1 < 0:
        src_x1 -= dst_x1
        dst_x1 = 0
    if dst_y2 > dst.shape[0]:
        dst_y2 = dst.shape[0]
    if dst_x2 > dst.shape[1]:
        dst_x2 = dst.shape[1]
    
    actual_h = min(dst_y2 - dst_y1, src.shape[0] - src_y1)
    actual_w = min(dst_x2 - dst_x1, src.shape[1] - src_x1)
    
    if actual_h <= 0 or actual_w <= 0:
        return result.astype(np.uint8)
    
    # Extract regions
    src_region = src[src_y1:src_y1 + actual_h, src_x1:src_x1 + actual_w]
    dst_region = dst[dst_y1:dst_y1 + actual_h, dst_x1:dst_x1 + actual_w]
    mask_region = mask[src_y1:src_y1 + actual_h, src_x1:src_x1 + actual_w] > 0
    
    # Simple Poisson blending approximation using gradient matching
    blended = _poisson_blend(src_region, dst_region, mask_region, flags)
    
    result[dst_y1:dst_y1 + actual_h, dst_x1:dst_x1 + actual_w] = blended
    
    return np.clip(result, 0, 255).astype(np.uint8)


def _poisson_blend(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    flags: int
) -> np.ndarray:
    """Simplified Poisson blending."""
    NORMAL_CLONE = 1
    MIXED_CLONE = 2
    
    result = dst.copy()
    h, w = mask.shape
    
    if src.ndim == 2:
        src = src[:, :, np.newaxis]
        dst = dst[:, :, np.newaxis]
        result = result[:, :, np.newaxis]
    
    c = src.shape[2]
    
    # Compute gradients
    src_grad_x = np.zeros_like(src)
    src_grad_y = np.zeros_like(src)
    dst_grad_x = np.zeros_like(dst)
    dst_grad_y = np.zeros_like(dst)
    
    src_grad_x[:, 1:] = src[:, 1:] - src[:, :-1]
    src_grad_y[1:, :] = src[1:, :] - src[:-1, :]
    dst_grad_x[:, 1:] = dst[:, 1:] - dst[:, :-1]
    dst_grad_y[1:, :] = dst[1:, :] - dst[:-1, :]
    
    # Choose gradient based on flags
    if flags == MIXED_CLONE:
        # Use gradient with larger magnitude
        src_mag = np.abs(src_grad_x) + np.abs(src_grad_y)
        dst_mag = np.abs(dst_grad_x) + np.abs(dst_grad_y)
        use_src = src_mag > dst_mag
        grad_x = np.where(use_src, src_grad_x, dst_grad_x)
        grad_y = np.where(use_src, src_grad_y, dst_grad_y)
    else:
        # Normal clone - use source gradient
        grad_x = src_grad_x
        grad_y = src_grad_y
    
    # Iterative solver (Gauss-Seidel)
    iterations = 100
    
    for _ in range(iterations):
        new_result = result.copy()
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if mask[y, x]:
                    # Solve Laplacian
                    laplacian = (grad_x[y, x] - grad_x[y, x-1] + 
                                grad_y[y, x] - grad_y[y-1, x])
                    
                    neighbors = (result[y-1, x] + result[y+1, x] + 
                               result[y, x-1] + result[y, x+1])
                    
                    new_result[y, x] = (neighbors - laplacian) / 4
        
        result = new_result
    
    if result.shape[2] == 1:
        result = result[:, :, 0]
    
    return result


def denoise_TVL1(
    observations: np.ndarray,
    result: Optional[np.ndarray] = None,
    lambda_val: float = 1.0,
    niters: int = 30
) -> np.ndarray:
    """Perform Total Variation L1 denoising.
    
    Args:
        observations: Input images
        result: Optional output
        lambda_val: Regularization parameter
        niters: Number of iterations
    
    Returns:
        Denoised image
    """
    if observations.ndim == 2:
        img = observations.astype(np.float32)
    else:
        img = observations[0].astype(np.float32)
    
    u = img.copy()
    
    for _ in range(niters):
        # Compute gradients
        grad_x = np.zeros_like(u)
        grad_y = np.zeros_like(u)
        grad_x[:, :-1] = u[:, 1:] - u[:, :-1]
        grad_y[:-1, :] = u[1:, :] - u[:-1, :]
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Normalize gradients
        grad_x /= grad_mag
        grad_y /= grad_mag
        
        # Compute divergence
        div = np.zeros_like(u)
        div[:, 1:] += grad_x[:, :-1] - grad_x[:, 1:]
        div[1:, :] += grad_y[:-1, :] - grad_y[1:, :]
        
        # Update
        u = img + lambda_val * div
    
    if result is not None:
        np.copyto(result, u)
        return result
    
    return u


def edgePreservingFilter(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    flags: int = 1,
    sigma_s: float = 60,
    sigma_r: float = 0.4
) -> np.ndarray:
    """Edge-preserving smoothing filter.
    
    Args:
        src: Input 8-bit 3-channel image
        dst: Optional output
        flags: Filter type (1=RECURS_FILTER, 2=NORMCONV_FILTER)
        sigma_s: Spatial sigma
        sigma_r: Range sigma
    
    Returns:
        Filtered image
    """
    from ..filters import bilateralFilter
    
    result = bilateralFilter(src, d=-1, sigmaColor=sigma_r*255, sigmaSpace=sigma_s)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def detailEnhance(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    sigma_s: float = 10,
    sigma_r: float = 0.15
) -> np.ndarray:
    """Enhance image details.
    
    Args:
        src: Input image
        dst: Optional output
        sigma_s: Spatial sigma
        sigma_r: Range sigma
    
    Returns:
        Detail-enhanced image
    """
    # Smooth the image
    smooth = edgePreservingFilter(src, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    
    # Extract detail
    detail = src.astype(np.float32) - smooth.astype(np.float32)
    
    # Enhance and add back
    enhanced = smooth.astype(np.float32) + 1.5 * detail
    result = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


def pencilSketch(
    src: np.ndarray,
    dst1: Optional[np.ndarray] = None,
    dst2: Optional[np.ndarray] = None,
    sigma_s: float = 60,
    sigma_r: float = 0.07,
    shade_factor: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """Create pencil sketch effect.
    
    Args:
        src: Input image
        dst1: Optional grayscale output
        dst2: Optional color output
        sigma_s: Spatial sigma
        sigma_r: Range sigma
        shade_factor: Shading factor
    
    Returns:
        Tuple of (grayscale_sketch, color_sketch)
    """
    from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
    
    gray = cvtColor(src, COLOR_BGR2GRAY)
    
    # Invert
    inverted = 255 - gray
    
    # Blur
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(inverted.astype(np.float32), sigma_s / 10)
    
    # Blend
    result_gray = gray.astype(np.float32) / (256 - blurred + 1e-6) * 256
    result_gray = np.clip(result_gray, 0, 255).astype(np.uint8)
    
    # Color version
    smooth = edgePreservingFilter(src, sigma_s=sigma_s, sigma_r=sigma_r)
    result_color = (smooth.astype(np.float32) * 
                   (result_gray[:, :, np.newaxis].astype(np.float32) / 255))
    result_color = np.clip(result_color, 0, 255).astype(np.uint8)
    
    return result_gray, result_color


def stylization(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    sigma_s: float = 60,
    sigma_r: float = 0.45
) -> np.ndarray:
    """Apply stylization filter (cartoon-like effect).
    
    Args:
        src: Input image
        dst: Optional output
        sigma_s: Spatial sigma
        sigma_r: Range sigma
    
    Returns:
        Stylized image
    """
    result = edgePreservingFilter(src, sigma_s=sigma_s, sigma_r=sigma_r)
    
    if dst is not None:
        np.copyto(dst, result)
        return dst
    
    return result


# Clone flags
NORMAL_CLONE = 1
MIXED_CLONE = 2
MONOCHROME_TRANSFER = 3

# Filter flags
RECURS_FILTER = 1
NORMCONV_FILTER = 2

# Import HDR and tonemap from photo.py module
from neurova.photo.photo import (
    Tonemap, TonemapDrago, TonemapReinhard, TonemapMantiuk,
    createTonemap, createTonemapDrago, createTonemapReinhard, createTonemapMantiuk,
    MergeDebevec, MergeMertens, CalibrateDebevec,
    createMergeDebevec, createMergeMertens, createCalibrateDebevec,
)


__all__ = [
    "fastNlMeansDenoising",
    "fastNlMeansDenoisingColored",
    "inpaint",
    "seamlessClone",
    "denoise_TVL1",
    "edgePreservingFilter",
    "detailEnhance",
    "pencilSketch",
    "stylization",
    
    # Constants
    "INPAINT_NS",
    "INPAINT_TELEA",
    "NORMAL_CLONE",
    "MIXED_CLONE",
    "MONOCHROME_TRANSFER",
    "RECURS_FILTER",
    "NORMCONV_FILTER",
    
    # HDR and Tonemap (NEW)
    "Tonemap", "TonemapDrago", "TonemapReinhard", "TonemapMantiuk",
    "createTonemap", "createTonemapDrago", "createTonemapReinhard", "createTonemapMantiuk",
    "MergeDebevec", "MergeMertens", "CalibrateDebevec",
    "createMergeDebevec", "createMergeMertens", "createCalibrateDebevec",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.