# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Automatic Differentiation Engine - autograd engine.

This module implements reverse-mode automatic differentiation (backpropagation)
with computation graphs, for automatic differentiation.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple, Callable, Any
from contextlib import contextmanager


# global flag for gradient computation
_grad_enabled = True


@contextmanager
def no_grad():
    """Context manager to disable gradient computation."""
    global _grad_enabled
    old_value = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old_value


class Tensor:
    """
    Tensor with automatic differentiation support.
    
    Neurova implementation with autograd capabilities.
    Tracks computational graph for automatic backpropagation.
    
    Parameters
    ----------
    data : array_like
        The tensor data
    requires_grad : bool, default=False
        If True, gradients will be computed for this tensor
    """
    
    def __init__(
        self,
        data: np.ndarray | float | int | list,
        requires_grad: bool = False,
        _children: Tuple[Tensor, ...] = (),
        _op: str = '',
    ):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        self.data = data.astype(np.float32) if data.dtype != np.float32 else data
        self.requires_grad = requires_grad and _grad_enabled
        self.grad: Optional[np.ndarray] = None
        
        # computational graph
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor."""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.data.size
    
    @property
    def dtype(self):
        """Data type of the tensor."""
        return self.data.dtype
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=self.requires_grad)
    
    # ==================== Arithmetic Operations ====================
    
    def __add__(self, other) -> Tensor:
        """Addition: self + other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
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
        """Multiplication: self * other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
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
    
    def __pow__(self, other: float | int) -> Tensor:
        """Power: self ** other"""
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(
            self.data ** other,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'**{other}'
        )
        
        def _backward():
            if self.requires_grad:
                grad = other * (self.data ** (other - 1)) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def __neg__(self) -> Tensor:
        """Negation: -self"""
        return self * -1
    
    def __radd__(self, other) -> Tensor:
        """Right addition: other + self"""
        return self + other
    
    def __sub__(self, other) -> Tensor:
        """Subtraction: self - other"""
        return self + (-other)
    
    def __rsub__(self, other) -> Tensor:
        """Right subtraction: other - self"""
        return other + (-self)
    
    def __rmul__(self, other) -> Tensor:
        """Right multiplication: other * self"""
        return self * other
    
    def __truediv__(self, other) -> Tensor:
        """Division: self / other"""
        return self * (other ** -1)
    
    def __rtruediv__(self, other) -> Tensor:
        """Right division: other / self"""
        return other * (self ** -1)
    
    # ==================== Matrix Operations ====================
    
    def matmul(self, other: Tensor) -> Tensor:
        """Matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
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
    
    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication: self @ other"""
        return self.matmul(other)
    
    # ==================== Reduction Operations ====================
    
    def sum(self, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> Tensor:
        """Sum of tensor elements."""
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> Tensor:
        """Mean of tensor elements."""
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='mean'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.shape)
                # divide by number of elements averaged
                if axis is None:
                    n = self.size
                elif isinstance(axis, int):
                    n = self.shape[axis]
                else:
                    n = np.prod([self.shape[ax] for ax in axis])
                grad = grad / n
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    # ==================== Shape Operations ====================
    
    def reshape(self, *shape) -> Tensor:
        """Reshape tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='reshape'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def view(self, *shape) -> Tensor:
        """Alias for reshape (compatibility alias)."""
        return self.reshape(*shape)
    
    def transpose(self, *axes) -> Tensor:
        """Transpose tensor dimensions."""
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            axes = axes[0]
        
        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='transpose'
        )
        
        def _backward():
            if self.requires_grad:
                # inverse transpose
                if axes is None:
                    grad = out.grad.T
                else:
                    # compute inverse permutation
                    inv_axes = np.argsort(axes)
                    grad = np.transpose(out.grad, inv_axes)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    @property
    def T(self) -> Tensor:
        """Transpose (2D only)."""
        return self.transpose()
    
    # ==================== Activation Functions ====================
    
    def relu(self) -> Tensor:
        """ReLU activation."""
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='ReLU'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self) -> Tensor:
        """Sigmoid activation."""
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(
            sig,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sigmoid'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * sig * (1 - sig)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def tanh(self) -> Tensor:
        """Tanh activation."""
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='tanh'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (1 - t ** 2)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    # ==================== Backpropagation ====================
    
    def backward(self, gradient: Optional[np.ndarray] = None) -> None:
        """
        Compute gradients using backpropagation.
        
        Parameters
        ----------
        gradient : np.ndarray, optional
            Gradient of the output (default: ones with same shape as self)
        """
        if not self.requires_grad:
            return
        
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
            self.grad = np.ones_like(self.data)
        else:
            self.grad = gradient
        
        # backpropagate
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self) -> None:
        """Zero out the gradient."""
        self.grad = None
    
    # ==================== Utilities ====================
    
    def item(self) -> float:
        """Get scalar value (for 1-element tensors)."""
        return float(self.data.item())
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return self.data
    
    def detach(self) -> Tensor:
        """Detach from computational graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def clone(self) -> Tensor:
        """Clone tensor with gradient tracking."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)


class Parameter(Tensor):
    """
    A trainable parameter (weight or bias).
    
    Neurova implementation - always requires gradient.
    """
    
    def __init__(self, data: np.ndarray | float | list):
        super().__init__(data, requires_grad=True)
    
    def __repr__(self) -> str:
        return f"Parameter({self.data})"


# helper functions for tensor creation
def tensor(data, requires_grad: bool = False) -> Tensor:
    """Create a tensor."""
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with zeros."""
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with ones."""
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False) -> Tensor:
    """Create a tensor with random normal values."""
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)


def rand(*shape, requires_grad: bool = False) -> Tensor:
    """Create a tensor with random uniform values."""
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad)
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.