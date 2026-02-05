# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Padding Layers.

Neurova implementation of padding layers for neural networks.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Union
from neurova.nn.layers import Module


class _ReflectionPadNd(Module):
    """Base class for reflection padding."""
    
    def __init__(self, padding: Union[int, Tuple[int, ...]]):
        super().__init__()
        self.padding = padding
    
    def extra_repr(self) -> str:
        return f'{self.padding}'


class ReflectionPad1d(_ReflectionPadNd):
    """
    Pads the input tensor using reflection of the input boundary.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on both sides.
        If tuple of 2 ints, uses (padding_left, padding_right).
    
    Shape
    -----
    - Input: (N, C, W_in)
    - Output: (N, C, W_out) where W_out = W_in + padding_left + padding_right
    
    Example
    -------
    >>> m = ReflectionPad1d(2)
    >>> input = np.random.randn(1, 3, 4)
    >>> output = m(input)
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        if isinstance(padding, int):
            padding = (padding, padding)
        super().__init__(padding)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        pad_left, pad_right = self.padding
        return np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)), mode='reflect')


class ReflectionPad2d(_ReflectionPadNd):
    """
    Pads the input tensor using reflection of the input boundary.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on all sides.
        If tuple of 4 ints, uses (left, right, top, bottom).
    
    Shape
    -----
    - Input: (N, C, H_in, W_in)
    - Output: (N, C, H_out, W_out)
    
    Example
    -------
    >>> m = ReflectionPad2d(2)
    >>> input = np.random.randn(1, 3, 4, 4)
    >>> output = m(input)
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        left, right, top, bottom = self.padding
        return np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='reflect')


class ReflectionPad3d(_ReflectionPadNd):
    """
    Pads the input tensor using reflection of the input boundary.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on all sides.
        If tuple of 6 ints, uses (left, right, top, bottom, front, back).
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        if isinstance(padding, int):
            padding = (padding,) * 6
        super().__init__(padding)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        left, right, top, bottom, front, back = self.padding
        return np.pad(x, ((0, 0), (0, 0), (front, back), (top, bottom), (left, right)), mode='reflect')


class _ReplicationPadNd(Module):
    """Base class for replication padding."""
    
    def __init__(self, padding: Union[int, Tuple[int, ...]]):
        super().__init__()
        self.padding = padding
    
    def extra_repr(self) -> str:
        return f'{self.padding}'


class ReplicationPad1d(_ReplicationPadNd):
    """
    Pads the input tensor using replication of the input boundary.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on both sides.
        If tuple of 2 ints, uses (padding_left, padding_right).
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        if isinstance(padding, int):
            padding = (padding, padding)
        super().__init__(padding)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        pad_left, pad_right = self.padding
        return np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)), mode='edge')


class ReplicationPad2d(_ReplicationPadNd):
    """
    Pads the input tensor using replication of the input boundary.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on all sides.
        If tuple of 4 ints, uses (left, right, top, bottom).
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        left, right, top, bottom = self.padding
        return np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='edge')


class ReplicationPad3d(_ReplicationPadNd):
    """
    Pads the input tensor using replication of the input boundary.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on all sides.
        If tuple of 6 ints, uses (left, right, top, bottom, front, back).
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        if isinstance(padding, int):
            padding = (padding,) * 6
        super().__init__(padding)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        left, right, top, bottom, front, back = self.padding
        return np.pad(x, ((0, 0), (0, 0), (front, back), (top, bottom), (left, right)), mode='edge')


class _ConstantPadNd(Module):
    """Base class for constant padding."""
    
    def __init__(self, padding: Union[int, Tuple[int, ...]], value: float = 0.0):
        super().__init__()
        self.padding = padding
        self.value = value
    
    def extra_repr(self) -> str:
        return f'padding={self.padding}, value={self.value}'


class ConstantPad1d(_ConstantPadNd):
    """
    Pads the input tensor with a constant value.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on both sides.
        If tuple of 2 ints, uses (padding_left, padding_right).
    value : float, default=0
        Fill value for padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int]], value: float = 0.0):
        if isinstance(padding, int):
            padding = (padding, padding)
        super().__init__(padding, value)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        pad_left, pad_right = self.padding
        return np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)), 
                      mode='constant', constant_values=self.value)


class ConstantPad2d(_ConstantPadNd):
    """
    Pads the input tensor with a constant value.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on all sides.
        If tuple of 4 ints, uses (left, right, top, bottom).
    value : float, default=0
        Fill value for padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]], value: float = 0.0):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding, value)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        left, right, top, bottom = self.padding
        return np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), 
                      mode='constant', constant_values=self.value)


class ConstantPad3d(_ConstantPadNd):
    """
    Pads the input tensor with a constant value.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding. If int, uses the same padding on all sides.
        If tuple of 6 ints, uses (left, right, top, bottom, front, back).
    value : float, default=0
        Fill value for padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]], value: float = 0.0):
        if isinstance(padding, int):
            padding = (padding,) * 6
        super().__init__(padding, value)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        left, right, top, bottom, front, back = self.padding
        return np.pad(x, ((0, 0), (0, 0), (front, back), (top, bottom), (left, right)), 
                      mode='constant', constant_values=self.value)


class ZeroPad1d(ConstantPad1d):
    """
    Pads the input tensor with zeros.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__(padding, 0.0)


class ZeroPad2d(ConstantPad2d):
    """
    Pads the input tensor with zeros.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__(padding, 0.0)


class ZeroPad3d(ConstantPad3d):
    """
    Pads the input tensor with zeros.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__(padding, 0.0)


class CircularPad1d(Module):
    """
    Pads the input tensor using circular/wrap padding.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        pad_left, pad_right = self.padding
        return np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)), mode='wrap')
    
    def extra_repr(self) -> str:
        return f'{self.padding}'


class CircularPad2d(Module):
    """
    Pads the input tensor using circular/wrap padding.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = padding
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        left, right, top, bottom = self.padding
        return np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='wrap')
    
    def extra_repr(self) -> str:
        return f'{self.padding}'


class CircularPad3d(Module):
    """
    Pads the input tensor using circular/wrap padding.
    
    Parameters
    ----------
    padding : int or tuple
        The size of the padding.
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding,) * 6
        self.padding = padding
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        left, right, top, bottom, front, back = self.padding
        return np.pad(x, ((0, 0), (0, 0), (front, back), (top, bottom), (left, right)), mode='wrap')
    
    def extra_repr(self) -> str:
        return f'{self.padding}'


__all__ = [
    'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d',
    'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d',
    'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
    'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d',
    'CircularPad1d', 'CircularPad2d', 'CircularPad3d',
]
