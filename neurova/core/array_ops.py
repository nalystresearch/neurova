# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Core array operations and utilities"""

import numpy as np
from typing import Union, Tuple, Optional, List, Sequence
from neurova.core.errors import ShapeError, ValidationError
from neurova.core.constants import EPSILON


def ensure_array(data: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """
    Ensure input is a numpy array
    
    Args:
        data: Input data
        
    Returns:
        numpy array
    """
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is 2D (add channel dimension if needed)
    
    Args:
        arr: Input array
        
    Returns:
        2D array
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ShapeError((2,), arr.shape, "Array must be 1D or 2D")
    return arr


def ensure_3d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is 3D (H, W, C)
    
    Args:
        arr: Input array (H, W) or (H, W, C)
        
    Returns:
        3D array with shape (H, W, C)
    """
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        return arr
    else:
        raise ShapeError((2, 3), arr.shape, "Array must be 2D or 3D")


def ensure_4d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is 4D (N, H, W, C) - batch format
    
    Args:
        arr: Input array
        
    Returns:
        4D array with shape (N, H, W, C)
    """
    if arr.ndim == 2:
        return arr[np.newaxis, :, :, np.newaxis]
    elif arr.ndim == 3:
        return arr[np.newaxis, :, :, :]
    elif arr.ndim == 4:
        return arr
    else:
        raise ShapeError((2, 3, 4), arr.shape, "Array must be 2D, 3D, or 4D")


def validate_shape(arr: np.ndarray, expected_ndim: Union[int, Tuple[int, ...]], 
                    name: str = "array") -> None:
    """
    Validate array shape dimensions
    
    Args:
        arr: Input array
        expected_ndim: Expected number of dimensions (int or tuple of valid dims)
        name: Name for error messages
        
    Raises:
        ShapeError: If shape is invalid
    """
    if isinstance(expected_ndim, int):
        expected_ndim = (expected_ndim,)
    
    if arr.ndim not in expected_ndim:
        raise ShapeError(
            expected_ndim,
            arr.shape,
            f"{name} must have {expected_ndim} dimensions, got {arr.ndim}"
        )


def validate_image_shape(arr: np.ndarray, name: str = "image") -> None:
    """
    Validate that array is a valid image shape (H, W) or (H, W, C)
    
    Args:
        arr: Input array
        name: Name for error messages
        
    Raises:
        ShapeError: If shape is invalid
    """
    validate_shape(arr, (2, 3), name)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ShapeError(
            "non-zero dimensions",
            arr.shape,
            f"{name} must have non-zero height and width"
        )


def get_spatial_shape(arr: np.ndarray) -> Tuple[int, int]:
    """
    Get spatial dimensions (height, width) from array
    
    Args:
        arr: Input array with shape (..., H, W, ...)
        
    Returns:
        Tuple of (height, width)
    """
    if arr.ndim < 2:
        raise ShapeError((2,), arr.shape, "Array must have at least 2 dimensions")
    
    if arr.ndim == 2:
        return arr.shape
    else:
        # assume format is (H, W, C) or (N, H, W, C)
        return (arr.shape[-3], arr.shape[-2]) if arr.ndim >= 3 else arr.shape[:2]


def get_num_channels(arr: np.ndarray) -> int:
    """
    Get number of channels in array
    
    Args:
        arr: Input array
        
    Returns:
        Number of channels (1 for grayscale)
    """
    if arr.ndim == 2:
        return 1
    elif arr.ndim == 3:
        return arr.shape[2]
    elif arr.ndim == 4:
        return arr.shape[3]
    else:
        raise ShapeError((2, 3, 4), arr.shape, "Invalid image array shape")


def normalize(arr: np.ndarray, 
              min_val: Optional[float] = None, 
              max_val: Optional[float] = None,
              target_min: float = 0.0,
              target_max: float = 1.0) -> np.ndarray:
    """
    Normalize array to target range
    
    Args:
        arr: Input array
        min_val: Minimum value (if None, use arr.min())
        max_val: Maximum value (if None, use arr.max())
        target_min: Target minimum value
        target_max: Target maximum value
        
    Returns:
        Normalized array
    """
    arr = arr.astype(np.float64)
    
    if min_val is None:
        min_val = float(arr.min())
    if max_val is None:
        max_val = float(arr.max())
    
    if abs(max_val - min_val) < EPSILON:
        return np.full_like(arr, target_min)
    
    # normalize to [0, 1]
    normalized = (arr - min_val) / (max_val - min_val)
    
    # scale to target range
    return normalized * (target_max - target_min) + target_min


def standardize(arr: np.ndarray, 
                mean: Optional[float] = None,
                std: Optional[float] = None,
                epsilon: float = EPSILON) -> np.ndarray:
    """
    Standardize array to zero mean and unit variance
    
    Args:
        arr: Input array
        mean: Mean value (if None, use arr.mean())
        std: Standard deviation (if None, use arr.std())
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Standardized array
    """
    arr = arr.astype(np.float64)
    
    if mean is None:
        mean = float(arr.mean())
    if std is None:
        std = float(arr.std())
    
    return (arr - mean) / (std + epsilon)


def pad_array(arr: np.ndarray,
              pad_width: Union[
                  int,
                  Tuple[int, ...],
                  List[Tuple[int, int]],
                  Tuple[Tuple[int, int], ...],
                  Sequence[Tuple[int, int]],
              ],
              mode: str = 'constant',
              constant_value: float = 0) -> np.ndarray:
    """
    Pad array with specified mode
    
    Args:
        arr: Input array
        pad_width: Number of values to pad
        mode: Padding mode ('constant', 'reflect', 'replicate', 'wrap')
        constant_value: Value for constant padding
        
    Returns:
        Padded array
    """
    if mode == 'constant':
        return np.pad(arr, pad_width, mode='constant', constant_values=constant_value)
    elif mode == 'reflect':
        return np.pad(arr, pad_width, mode='reflect')
    elif mode == 'replicate':
        return np.pad(arr, pad_width, mode='edge')
    elif mode == 'wrap':
        return np.pad(arr, pad_width, mode='wrap')
    else:
        raise ValidationError('mode', mode, "one of: constant, reflect, replicate, wrap")


def clip_array(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Clip array values to specified range
    
    Args:
        arr: Input array
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped array
    """
    return np.clip(arr, min_val, max_val)


def resize_kernel(kernel: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize kernel by scaling factor
    
    Args:
        kernel: Input kernel
        scale: Scaling factor
        
    Returns:
        Resized kernel
    """
    if scale == 1.0:
        return kernel
    
    new_size = int(kernel.shape[0] * scale)
    if new_size % 2 == 0:
        new_size += 1  # Keep kernel size odd
    
    # simple nearest neighbor resize for kernels
    indices = np.linspace(0, kernel.shape[0] - 1, new_size, dtype=int)
    return kernel[indices][:, indices] if kernel.ndim == 2 else kernel[indices]


def meshgrid_2d(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 2D meshgrid for image coordinates
    
    Args:
        height: Image height
        width: Image width
        
    Returns:
        Tuple of (y_coords, x_coords) arrays
    """
    y = np.arange(height)
    x = np.arange(width)
    return np.meshgrid(y, x, indexing='ij')


def calculate_padding(input_size: int, kernel_size: int, stride: int = 1, 
                      dilation: int = 1, padding_mode: str = 'same') -> int:
    """
    Calculate padding needed for convolution
    
    Args:
        input_size: Input dimension size
        kernel_size: Kernel dimension size
        stride: Stride value
        dilation: Dilation value
        padding_mode: 'same' or 'valid'
        
    Returns:
        Padding size
    """
    if padding_mode == 'valid':
        return 0
    elif padding_mode == 'same':
        effective_kernel = dilation * (kernel_size - 1) + 1
        total_padding = max((input_size - 1) * stride + effective_kernel - input_size, 0)
        return total_padding // 2
    else:
        raise ValidationError('padding_mode', padding_mode, "one of: same, valid")


def calculate_output_size(input_size: int, kernel_size: int, stride: int = 1,
                          padding: int = 0, dilation: int = 1) -> int:
    """
    Calculate output size after convolution
    
    Args:
        input_size: Input dimension size
        kernel_size: Kernel dimension size
        stride: Stride value
        padding: Padding value
        dilation: Dilation value
        
    Returns:
        Output size
    """
    effective_kernel = dilation * (kernel_size - 1) + 1
    return (input_size + 2 * padding - effective_kernel) // stride + 1
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.