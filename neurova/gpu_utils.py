# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
GPU Utility Functions for Image/Video Processing.

This module provides GPU-accelerated versions of common image and video
operations using CuPy when available.
"""

import numpy as np
from typing import Tuple, Optional
from neurova.device import get_backend, get_device, to_device

# try to import cupy
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = None
    HAS_CUDA = False

try:
    from neurova.accel import cuda_native as _cuda_native

    HAS_CUDA_NATIVE = bool(_cuda_native.is_cuda_available())
except Exception:  # pragma: no cover - optional binary
    _cuda_native = None
    HAS_CUDA_NATIVE = False


def gaussian_blur_gpu(image, sigma: float = 1.0):
    """
    GPU-accelerated Gaussian blur.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Blurred image
    """
    backend = get_backend()
    
    # create Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    x = backend.arange(kernel_size) - kernel_size // 2
    kernel_1d = backend.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    
    # apply separable convolution (more efficient)
    from scipy import ndimage
    if get_device() == 'cuda' and HAS_CUDA:
        # use CuPy's faster implementation
        from cupyx.scipy import ndimage as ndimage_gpu
        return ndimage_gpu.gaussian_filter(image, sigma=sigma)
    else:
        return ndimage.gaussian_filter(image, sigma=sigma)


def sobel_filter_gpu(image):
    """
    GPU-accelerated Sobel edge detection.
    
    Args:
        image: Input grayscale image
    
    Returns:
        Edge magnitude
    """
    backend = get_backend()
    
    # sobel kernels
    sobel_x = backend.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = backend.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # convolve with kernels
    from scipy import ndimage
    if get_device() == 'cuda' and HAS_CUDA:
        from cupyx.scipy import ndimage as ndimage_gpu
        grad_x = ndimage_gpu.convolve(image, sobel_x)
        grad_y = ndimage_gpu.convolve(image, sobel_y)
    else:
        grad_x = ndimage.convolve(image, sobel_x)
        grad_y = ndimage.convolve(image, sobel_y)
    
    # compute magnitude
    magnitude = backend.sqrt(grad_x**2 + grad_y**2)
    return magnitude


def resize_gpu(image, output_shape: Tuple[int, int], order: int = 1):
    """
    GPU-accelerated image resize.
    
    Args:
        image: Input image
        output_shape: (height, width)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        Resized image
    """
    from scipy import ndimage
    
    # calculate zoom factors
    zoom_factors = [output_shape[0] / image.shape[0], 
                    output_shape[1] / image.shape[1]]
    
    if image.ndim == 3:  # Color image
        zoom_factors.append(1.0)  # Don't zoom channels
    
    if get_device() == 'cuda' and HAS_CUDA:
        from cupyx.scipy import ndimage as ndimage_gpu
        return ndimage_gpu.zoom(image, zoom_factors, order=order)
    else:
        return ndimage.zoom(image, zoom_factors, order=order)


def rotate_gpu(image, angle: float, reshape: bool = False):
    """
    GPU-accelerated image rotation.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        reshape: Whether to reshape output to fit rotated image
    
    Returns:
        Rotated image
    """
    from scipy import ndimage
    
    if get_device() == 'cuda' and HAS_CUDA:
        from cupyx.scipy import ndimage as ndimage_gpu
        return ndimage_gpu.rotate(image, angle, reshape=reshape)
    else:
        return ndimage.rotate(image, angle, reshape=reshape)


def affine_transform_gpu(image, matrix):
    """
    GPU-accelerated affine transformation.
    
    Args:
        image: Input image
        matrix: 2x3 affine transformation matrix
    
    Returns:
        Transformed image
    """
    from scipy import ndimage
    
    if get_device() == 'cuda' and HAS_CUDA:
        from cupyx.scipy import ndimage as ndimage_gpu
        return ndimage_gpu.affine_transform(image, matrix)
    else:
        return ndimage.affine_transform(image, matrix)


def threshold_gpu(image, threshold: float):
    """
    GPU-accelerated thresholding.
    
    Args:
        image: Input image
        threshold: Threshold value
    
    Returns:
        Binary image
    """
    backend = get_backend()
    return (image > threshold).astype(backend.uint8) * 255


def histogram_gpu(image, bins: int = 256):
    """
    GPU-accelerated histogram computation.
    
    Args:
        image: Input image
        bins: Number of bins
    
    Returns:
        Histogram array
    """
    backend = get_backend()
    
    if get_device() == 'cuda' and HAS_CUDA:
        # cuPy has optimized histogram
        hist, _ = cp.histogram(image.ravel(), bins=bins)
        return hist
    else:
        hist, _ = np.histogram(image.ravel(), bins=bins)
        return hist


def morphology_dilate_gpu(image, structure):
    """
    GPU-accelerated morphological dilation.
    
    Args:
        image: Binary input image
        structure: Structuring element
    
    Returns:
        Dilated image
    """
    from scipy import ndimage
    
    if get_device() == 'cuda' and HAS_CUDA:
        from cupyx.scipy import ndimage as ndimage_gpu
        return ndimage_gpu.binary_dilation(image, structure=structure)
    else:
        return ndimage.binary_dilation(image, structure=structure)


def morphology_erode_gpu(image, structure):
    """
    GPU-accelerated morphological erosion.
    
    Args:
        image: Binary input image
        structure: Structuring element
    
    Returns:
        Eroded image
    """
    from scipy import ndimage
    
    if get_device() == 'cuda' and HAS_CUDA:
        from cupyx.scipy import ndimage as ndimage_gpu
        return ndimage_gpu.binary_erosion(image, structure=structure)
    else:
        return ndimage.binary_erosion(image, structure=structure)


def template_matching_gpu(image, template):
    """
    GPU-accelerated template matching using normalized cross-correlation.
    
    Args:
        image: Input image
        template: Template to match
    
    Returns:
        Correlation map
    """
    backend = get_backend()
    
    # normalize image and template
    image_mean = backend.mean(image)
    image_std = backend.std(image)
    image_normalized = (image - image_mean) / (image_std + 1e-8)
    
    template_mean = backend.mean(template)
    template_std = backend.std(template)
    template_normalized = (template - template_mean) / (template_std + 1e-8)
    
    # cross-correlation using FFT (GPU-accelerated with CuPy)
    if get_device() == 'cuda' and HAS_CUDA:
        from cupyx.scipy import signal
        correlation = signal.correlate2d(image_normalized, template_normalized, mode='same')
    else:
        from scipy import signal
        correlation = signal.correlate2d(image_normalized, template_normalized, mode='same')
    
    return correlation


def optical_flow_gpu(frame1, frame2, window_size: int = 15):
    """
    GPU-accelerated Lucas-Kanade optical flow (simplified version).
    
    Args:
        frame1: First frame (grayscale)
        frame2: Second frame (grayscale)
        window_size: Window size for flow computation
    
    Returns:
        (flow_x, flow_y): Optical flow in x and y directions
    """
    backend = get_backend()
    
    # compute gradients
    from scipy import ndimage
    if get_device() == 'cuda' and HAS_CUDA:
        from cupyx.scipy import ndimage as ndimage_gpu
        
        # spatial gradients
        Ix = ndimage_gpu.sobel(frame1, axis=1)
        Iy = ndimage_gpu.sobel(frame1, axis=0)
        
        # temporal gradient
        It = frame2 - frame1
    else:
        # cPU version
        Ix = ndimage.sobel(frame1, axis=1)
        Iy = ndimage.sobel(frame1, axis=0)
        It = frame2 - frame1
    
    # simplified flow computation (Lucas-Kanade)
    # in practice, you'd want a more sophisticated implementation
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It
    
    # average over window (simplified - should use proper windowing)
    flow_x = -(IxIt) / (Ix2 + 1e-8)
    flow_y = -(IyIt) / (Iy2 + 1e-8)
    
    return flow_x, flow_y


def batch_process_frames_gpu(frames, process_func, batch_size: int = 32):
    """
    GPU-accelerated batch processing of video frames.
    
    Args:
        frames: List of frames or array of shape (N, H, W) or (N, H, W, C)
        process_func: Function to apply to each frame
        batch_size: Number of frames to process at once
    
    Returns:
        List of processed frames
    """
    backend = get_backend()
    
    # move frames to GPU if needed
    if get_device() == 'cuda':
        frames_gpu = to_device(frames)
    else:
        frames_gpu = frames
    
    results = []
    num_frames = len(frames_gpu)
    
    for i in range(0, num_frames, batch_size):
        batch = frames_gpu[i:i+batch_size]
        
        # process batch
        if hasattr(batch, '__len__'):
            batch_results = [process_func(frame) for frame in batch]
        else:
            batch_results = [process_func(batch)]
        
        results.extend(batch_results)
    
    return results


# gpu-accelerated color space conversions
def rgb_to_grayscale_gpu(image):
    """GPU-accelerated RGB to grayscale conversion."""
    backend = get_backend()
    
    # standard RGB to grayscale conversion
    # y = 0.299*R + 0.587*G + 0.114*B
    weights = backend.array([0.299, 0.587, 0.114])
    
    if HAS_CUDA_NATIVE and image.ndim == 3 and image.shape[2] >= 3:
        arr = np.ascontiguousarray(image.astype(np.uint8, copy=False))
        return _cuda_native.rgb_to_gray(arr)

    if image.ndim == 3:
        gray = backend.dot(image, weights)
    else:
        gray = image
    
    return gray.astype(backend.uint8)


def normalize_gpu(image, mean=None, std=None):
    """
    GPU-accelerated image normalization.
    
    Args:
        image: Input image
        mean: Mean value(s) to subtract (if None, computed from image)
        std: Std dev to divide by (if None, computed from image)
    
    Returns:
        Normalized image
    """
    backend = get_backend()
    
    if mean is None:
        mean = backend.mean(image)
    if std is None:
        std = backend.std(image)
    
    if HAS_CUDA_NATIVE and image.dtype == np.float32:
        arr = np.ascontiguousarray(image)
        return _cuda_native.normalize(arr, float(mean), float(std))

    normalized = (image - mean) / (std + 1e-8)
    return normalized
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.