# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
GPU-Accelerated Automatic Differentiation using CuPy.

This module provides GPU support for Neurova's autograd engine.
Falls back to CPU (NumPy) if CuPy is not installed.

Usage:
    # automatically uses GPU if available
    from neurova.nn.autograd_gpu import Tensor
    
    x = Tensor([1, 2, 3], device='cuda')  # GPU
    y = Tensor([4, 5, 6], device='cpu')   # CPU
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple, Callable
from contextlib import contextmanager

# try to import cupy for GPU support
try:
    import cupy as cp
    HAS_CUDA = True
    print(" GPU acceleration enabled (CuPy found)")
except ImportError:
    cp = np  # Fall back to NumPy
    HAS_CUDA = False
    print("  GPU acceleration disabled (CuPy not found, using CPU)")


def get_array_module(arr):
    """Get the appropriate module (cupy or numpy) for an array."""
    if HAS_CUDA:
        return cp.get_array_module(arr)
    return np


class Tensor:
    """
    GPU-accelerated Tensor with automatic differentiation.
    
    Supports both CPU (NumPy) and GPU (CuPy) backends with automatic
    device management and memory transfer.
    
    Parameters
    ----------
    data : array_like
        The tensor data
    requires_grad : bool, default=False
        If True, gradients will be computed
    device : str, default='cpu'
        Device to place tensor on: 'cpu' or 'cuda'
    
    Examples
    --------
    >>> # CPU tensor
    >>> x = Tensor([1, 2, 3], device='cpu')
    
    >>> # GPU tensor (if CuPy is installed)
    >>> x = Tensor([1, 2, 3], device='cuda')
    
    >>> # Automatic device detection
    >>> x = Tensor([1, 2, 3])  # Uses GPU if available
    """
    
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        device: Optional[str] = None,
        _children: Tuple[Tensor, ...] = (),
        _op: str = '',
    ):
        # determine device
        if device is None:
            device = 'cuda' if HAS_CUDA else 'cpu'
        
        self.device = device
        
        # choose appropriate module
        if device == 'cuda' and HAS_CUDA:
            xp = cp
        else:
            xp = np
        
        # convert data to appropriate array
        if not isinstance(data, (np.ndarray, cp.ndarray if HAS_CUDA else np.ndarray)):
            data = xp.array(data, dtype=xp.float32)
        elif isinstance(data, np.ndarray) and device == 'cuda' and HAS_CUDA:
            data = cp.asarray(data)
        elif HAS_CUDA and isinstance(data, cp.ndarray) and device == 'cpu':
            data = cp.asnumpy(data)
        
        self.data = data.astype(xp.float32) if data.dtype != xp.float32 else data
        self.requires_grad = requires_grad
        self.grad: Optional = None
        
        # computational graph
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor."""
        return self.data.shape
    
    @property
    def xp(self):
        """Get the array module (cupy or numpy) for this tensor."""
        return get_array_module(self.data)
    
    def cpu(self) -> Tensor:
        """Move tensor to CPU."""
        if self.device == 'cpu':
            return self
        data_cpu = cp.asnumpy(self.data) if HAS_CUDA else self.data
        return Tensor(data_cpu, requires_grad=self.requires_grad, device='cpu')
    
    def cuda(self) -> Tensor:
        """Move tensor to GPU."""
        if not HAS_CUDA:
            raise RuntimeError("CUDA not available - install CuPy for GPU support")
        if self.device == 'cuda':
            return self
        data_gpu = cp.asarray(self.data)
        return Tensor(data_gpu, requires_grad=self.requires_grad, device='cuda')
    
    def to(self, device: str) -> Tensor:
        """Move tensor to specified device."""
        if device == 'cuda':
            return self.cuda()
        elif device == 'cpu':
            return self.cpu()
        else:
            raise ValueError(f"Unknown device: {device}")
    
    def numpy(self):
        """Convert to NumPy array (always on CPU)."""
        if HAS_CUDA and isinstance(self.data, cp.ndarray):
            return cp.asnumpy(self.data)
        return self.data
    
    def __repr__(self) -> str:
        return f"Tensor({self.numpy()}, device='{self.device}', requires_grad={self.requires_grad})"
    
    # ==================== Arithmetic Operations ====================
    
    def __add__(self, other) -> Tensor:
        """Addition with broadcasting."""
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        xp = self.xp
        
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='+'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                # handle broadcasting
                ndims_added = grad.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = grad if self.grad is None else self.grad + grad
            
            if other.requires_grad:
                grad = out.grad
                ndims_added = grad.ndim - other.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other) -> Tensor:
        """Element-wise multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='*'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                ndims_added = grad.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = grad if self.grad is None else self.grad + grad
            
            if other.requires_grad:
                grad = out.grad * self.data
                ndims_added = grad.ndim - other.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication (GPU-accelerated)."""
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        xp = self.xp
        
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='@'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward
        return out
    
    def sum(self, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> Tensor:
        """Sum (GPU-accelerated)."""
        xp = self.xp
        
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='sum'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad = xp.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = xp.expand_dims(grad, axis=ax)
                grad = xp.broadcast_to(grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def relu(self) -> Tensor:
        """ReLU activation (GPU-accelerated)."""
        xp = self.xp
        
        out = Tensor(
            xp.maximum(0, self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='ReLU'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def backward(self, gradient: Optional = None) -> None:
        """Backpropagation (GPU-accelerated)."""
        if not self.requires_grad:
            return
        
        xp = self.xp
        
        # build topological order
        topo: List[Tensor] = []
        visited = set()
        
        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # initialize gradient
        if gradient is None:
            self.grad = xp.ones_like(self.data)
        else:
            self.grad = gradient
        
        # backpropagate
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self) -> None:
        """Zero out gradient."""
        self.grad = None
    
    # convenience methods
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1)
    def __pow__(self, other): 
        xp = self.xp
        return Tensor(self.data ** other, requires_grad=self.requires_grad, device=self.device)
    
    @property
    def T(self): 
        xp = self.xp
        return Tensor(self.data.T, requires_grad=self.requires_grad, device=self.device)


# helper functions
def get_default_device() -> str:
    """Get default device (cuda if available, else cpu)."""
    return 'cuda' if HAS_CUDA else 'cpu'


def set_default_device(device: str) -> None:
    """Set default device for new tensors."""
    global _default_device
    _default_device = device


# gPU info
def cuda_is_available() -> bool:
    """Check if CUDA is available."""
    return HAS_CUDA


def cuda_device_count() -> int:
    """Get number of CUDA devices."""
    if not HAS_CUDA:
        return 0
    return cp.cuda.runtime.getDeviceCount()


def cuda_get_device_name(device_id: int = 0) -> str:
    """Get CUDA device name."""
    if not HAS_CUDA:
        return "No CUDA device"
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return props['name'].decode('utf-8')


def cuda_memory_allocated(device: int = None) -> int:
    """Get allocated GPU memory in bytes."""
    if not HAS_CUDA:
        return 0
    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes()


def cuda_empty_cache() -> None:
    """Clear GPU memory cache."""
    if HAS_CUDA:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()


# print GPU status on import
if HAS_CUDA:
    try:
        device_name = cuda_get_device_name(0)
        device_count = cuda_device_count()
        print(f"   GPU: {device_name} (x{device_count})")
        print(f"   CuPy version: {cp.__version__}")
    except:
        print("   GPU detection failed")
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.