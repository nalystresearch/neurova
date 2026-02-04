# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Global Device Management for Neurova.

This module provides GPU/CPU selection for ALL Neurova operations:
- Machine Learning (classification, clustering, etc.)
- Deep Learning (neural networks)
- Image Processing (filters, transforms, etc.)
- Video Processing (frame processing, tracking, etc.)
- Computer Vision (feature detection, object detection, etc.)

Usage:
    import neurova as nv
    
    # set device globally
    nv.set_device('cuda')  # Use GPU for everything
    nv.set_device('cpu')   # Use CPU for everything
    
    # check current device
    device = nv.get_device()
    
    # check GPU availability
    if nv.cuda_is_available():
        nv.set_device('cuda')
    
    # get device info
    info = nv.get_device_info()
"""

import numpy as np
from typing import Optional, Union, Literal
import warnings

# try to import cupy for GPU support
try:
    import cupy as cp
    HAS_CUDA = True
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    HAS_CUDA = False
    CUDA_AVAILABLE = False

# global device configuration
_GLOBAL_DEVICE = 'cpu'  # Default to CPU
_DEVICE_CONTEXT_STACK = []  # For temporary device switching


class DeviceManager:
    """
    Global device manager for Neurova.
    
    Handles CPU/GPU selection and provides unified array operations
    that work on both CPU (NumPy) and GPU (CuPy).
    """
    
    def __init__(self):
        self._device = 'cpu'
        self._backend = np
        
        # detect GPU on initialization
        if HAS_CUDA:
            try:
                # test GPU access
                cp.cuda.Device(0).compute_capability
                print(f" GPU acceleration available: {self.get_gpu_name()}")
            except:
                warnings.warn("CuPy installed but no GPU detected. Using CPU.")
                self._device = 'cpu'
                self._backend = np
    
    def set_device(self, device: Literal['cpu', 'cuda', 'gpu']) -> None:
        """
        Set the global device for all Neurova operations.
        
        Args:
            device: 'cpu', 'cuda', or 'gpu'
        
        Example:
            >>> import neurova as nv
            >>> nv.set_device('cuda')  # Use GPU
            >>> nv.set_device('cpu')   # Use CPU
        """
        global _GLOBAL_DEVICE
        
        if device in ['cuda', 'gpu']:
            if not HAS_CUDA:
                raise RuntimeError(
                    "GPU device requested but CuPy not installed. "
                    "Install CuPy: pip install cupy-cuda12x"
                )
            self._device = 'cuda'
            self._backend = cp
            _GLOBAL_DEVICE = 'cuda'
            print(f" Device set to GPU: {self.get_gpu_name()}")
        
        elif device == 'cpu':
            self._device = 'cpu'
            self._backend = np
            _GLOBAL_DEVICE = 'cpu'
            print("  Device set to CPU")
        
        else:
            raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'cuda'")
    
    def get_device(self) -> str:
        """Get current device."""
        return self._device
    
    def get_backend(self):
        """Get current backend (numpy or cupy)."""
        return self._backend
    
    def cuda_is_available(self) -> bool:
        """Check if CUDA/GPU is available."""
        return HAS_CUDA
    
    def get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        if not HAS_CUDA:
            return 0
        try:
            return cp.cuda.runtime.getDeviceCount()
        except:
            return 0
    
    def get_gpu_name(self, device_id: int = 0) -> str:
        """Get GPU name."""
        if not HAS_CUDA:
            return "No GPU"
        try:
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            return props['name'].decode('utf-8')
        except:
            return "Unknown GPU"
    
    def get_device_info(self) -> dict:
        """
        Get detailed device information.
        
        Returns:
            Dictionary with device information
        """
        info = {
            'current_device': self._device,
            'cuda_available': HAS_CUDA,
            'gpu_count': self.get_gpu_count(),
        }
        
        if HAS_CUDA and self._device == 'cuda':
            info['gpu_name'] = self.get_gpu_name()
            info['compute_capability'] = self._get_compute_capability()
            info['memory_total'] = self._get_total_memory()
            info['memory_available'] = self._get_available_memory()
        
        return info
    
    def _get_compute_capability(self) -> str:
        """Get GPU compute capability."""
        if not HAS_CUDA:
            return "N/A"
        try:
            device = cp.cuda.Device(0)
            major, minor = device.compute_capability
            return f"{major}.{minor}"
        except:
            return "Unknown"
    
    def _get_total_memory(self) -> str:
        """Get total GPU memory."""
        if not HAS_CUDA:
            return "N/A"
        try:
            mem = cp.cuda.Device(0).mem_info[1]
            return f"{mem / 1e9:.2f} GB"
        except:
            return "Unknown"
    
    def _get_available_memory(self) -> str:
        """Get available GPU memory."""
        if not HAS_CUDA:
            return "N/A"
        try:
            mem = cp.cuda.Device(0).mem_info[0]
            return f"{mem / 1e9:.2f} GB"
        except:
            return "Unknown"
    
    def to_device(self, array, device: Optional[str] = None):
        """
        Move array to specified device.
        
        Args:
            array: NumPy or CuPy array
            device: Target device ('cpu' or 'cuda'). If None, uses current device.
        
        Returns:
            Array on target device
        """
        target_device = device or self._device
        
        if target_device == 'cpu':
            # move to CPU
            if isinstance(array, np.ndarray):
                return array
            else:
                # cuPy array -> NumPy
                return cp.asnumpy(array)
        
        elif target_device in ['cuda', 'gpu']:
            if not HAS_CUDA:
                warnings.warn("GPU requested but not available. Using CPU.")
                return np.asarray(array)
            
            # move to GPU
            if HAS_CUDA and hasattr(array, '__cuda_array_interface__'):
                return array  # Already on GPU
            else:
                # numPy array -> CuPy
                return cp.asarray(array)
        
        else:
            raise ValueError(f"Unknown device: {target_device}")
    
    def synchronize(self):
        """Synchronize GPU operations (wait for all GPU kernels to finish)."""
        if HAS_CUDA and self._device == 'cuda':
            cp.cuda.Stream.null.synchronize()
    
    def empty_cache(self):
        """Clear GPU memory cache."""
        if HAS_CUDA:
            cp.get_default_memory_pool().free_all_blocks()
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        if not HAS_CUDA or self._device == 'cpu':
            return {'used': 0, 'total': 0}
        
        mempool = cp.get_default_memory_pool()
        return {
            'used_bytes': mempool.used_bytes(),
            'total_bytes': mempool.total_bytes(),
            'used_gb': mempool.used_bytes() / 1e9,
            'total_gb': mempool.total_bytes() / 1e9,
        }


# global device manager instance
_device_manager = DeviceManager()


# public API functions
def set_device(device: Literal['cpu', 'cuda', 'gpu']) -> None:
    """
    Set global device for all Neurova operations.
    
    Args:
        device: 'cpu', 'cuda', or 'gpu'
    
    Example:
        >>> import neurova as nv
        >>> 
        >>> # Use GPU for all operations
        >>> nv.set_device('cuda')
        >>> 
        >>> # Process image on GPU
        >>> img = nv.io.imread('photo.jpg')
        >>> blurred = nv.filters.gaussian_blur(img, sigma=2.0)  # gpu-accelerated!
        >>> 
        >>> # Train model on GPU
        >>> model = nv.ml.RandomForest()
        >>> model.fit(X, y)  # gpu-accelerated!
    """
    _device_manager.set_device(device)


def get_device() -> str:
    """
    Get current device.
    
    Returns:
        'cpu' or 'cuda'
    """
    return _device_manager.get_device()


def get_backend():
    """
    Get current backend (numpy or cupy).
    
    Returns:
        numpy or cupy module
    """
    return _device_manager.get_backend()


def cuda_is_available() -> bool:
    """
    Check if CUDA/GPU is available.
    
    Returns:
        True if GPU is available, False otherwise
    
    Example:
        >>> import neurova as nv
        >>> if nv.cuda_is_available():
        ...     nv.set_device('cuda')
        ... else:
        ...     print("No GPU detected, using CPU")
    """
    return _device_manager.cuda_is_available()


def get_device_count() -> int:
    """Get number of available GPUs."""
    return _device_manager.get_gpu_count()


def get_device_name(device_id: int = 0) -> str:
    """Get GPU name."""
    return _device_manager.get_gpu_name(device_id)


def get_device_info() -> dict:
    """
    Get detailed device information.
    
    Returns:
        Dictionary with device info
    
    Example:
        >>> import neurova as nv
        >>> info = nv.get_device_info()
        >>> print(info)
        {
            'current_device': 'cuda',
            'cuda_available': True,
            'gpu_count': 1,
            'gpu_name': 'NVIDIA GeForce RTX 3080',
            'compute_capability': '8.6',
            'memory_total': '10.00 GB',
            'memory_available': '8.50 GB'
        }
    """
    return _device_manager.get_device_info()


def to_device(array, device: Optional[str] = None):
    """
    Move array to specified device.
    
    Args:
        array: NumPy or CuPy array
        device: 'cpu' or 'cuda'. If None, uses current device.
    
    Returns:
        Array on target device
    """
    return _device_manager.to_device(array, device)


def synchronize():
    """Wait for all GPU operations to complete."""
    _device_manager.synchronize()


def empty_cache():
    """Clear GPU memory cache."""
    _device_manager.empty_cache()


def get_memory_usage() -> dict:
    """Get current GPU memory usage."""
    return _device_manager.get_memory_usage()


class device_context:
    """
    Context manager for temporary device switching.
    
    Example:
        >>> import neurova as nv
        >>> 
        >>> nv.set_device('cpu')  # Default to CPU
        >>> 
        >>> # Temporarily use GPU for expensive operation
        >>> with nv.device_context('cuda'):
        ...     result = nv.filters.gaussian_blur(large_image, sigma=5.0)
        >>> 
        >>> # Back to CPU after context
        >>> assert nv.get_device() == 'cpu'
    """
    
    def __init__(self, device: str):
        self.device = device
        self.previous_device = None
    
    def __enter__(self):
        self.previous_device = get_device()
        set_device(self.device)
        return self
    
    def __exit__(self, *args):
        set_device(self.previous_device)


# helper functions for array operations
def array(data, device: Optional[str] = None):
    """
    Create array on specified device.
    
    Args:
        data: Array-like data
        device: 'cpu' or 'cuda'. If None, uses current device.
    
    Returns:
        NumPy or CuPy array
    """
    backend = get_backend() if device is None else (cp if device == 'cuda' else np)
    return backend.asarray(data)


def zeros(shape, dtype=None, device: Optional[str] = None):
    """Create array of zeros on specified device."""
    backend = get_backend() if device is None else (cp if device == 'cuda' else np)
    return backend.zeros(shape, dtype=dtype)


def ones(shape, dtype=None, device: Optional[str] = None):
    """Create array of ones on specified device."""
    backend = get_backend() if device is None else (cp if device == 'cuda' else np)
    return backend.ones(shape, dtype=dtype)


def empty(shape, dtype=None, device: Optional[str] = None):
    """Create empty array on specified device."""
    backend = get_backend() if device is None else (cp if device == 'cuda' else np)
    return backend.empty(shape, dtype=dtype)


# print initialization message
if HAS_CUDA:
    try:
        gpu_name = _device_manager.get_gpu_name()
        print(f" Neurova initialized with GPU support: {gpu_name}")
        print(f"   Use neurova.set_device('cuda') to enable GPU acceleration")
        print(f"   Current device: CPU (use neurova.set_device('cuda') to switch)")
    except:
        print("  CuPy installed but no GPU detected. Using CPU mode.")
else:
    print("  Neurova initialized in CPU mode")
    print("   Install CuPy for GPU acceleration: pip install cupy-cuda12x")
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.