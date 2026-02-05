# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Convolutional layers for neural networks."""

from __future__ import annotations
import numpy as np
from neurova.neural.module import Module, Parameter
from neurova.neural.tensor import Tensor, tensor


def _im2col(x: np.ndarray, kernel_h: int, kernel_w: int, stride: int, padding: int) -> np.ndarray:
    """Convert image to column matrix for convolution.
    
    Args:
        x: Input (batch, height, width, channels)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride
        padding: Padding
        
    Returns:
        Column matrix (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
    """
    batch, h, w, c = x.shape
    
    # add padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        h += 2 * padding
        w += 2 * padding
    
    # output dimensions
    out_h = (h - kernel_h) // stride + 1
    out_w = (w - kernel_w) // stride + 1
    
    # create column matrix
    col = np.zeros((batch, out_h, out_w, kernel_h, kernel_w, c), dtype=x.dtype)
    
    for y in range(kernel_h):
        for x_pos in range(kernel_w):
            col[:, :, :, y, x_pos, :] = x[
                :,
                y:y + out_h * stride:stride,
                x_pos:x_pos + out_w * stride:stride,
                :
            ]
    
    col = col.reshape(batch * out_h * out_w, -1)
    return col


def _col2im(col: np.ndarray, x_shape: tuple, kernel_h: int, kernel_w: int, stride: int, padding: int) -> np.ndarray:
    """Convert column matrix back to image (for backprop).
    
    Args:
        col: Column matrix (batch * out_h * out_w, kernel_h * kernel_w * channels)
        x_shape: Original input shape (batch, h, w, c)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride
        padding: Padding
        
    Returns:
        Image (batch, height, width, channels)
    """
    batch, h, w, c = x_shape
    
    if padding > 0:
        h += 2 * padding
        w += 2 * padding
    
    out_h = (h - kernel_h) // stride + 1
    out_w = (w - kernel_w) // stride + 1
    
    col = col.reshape(batch, out_h, out_w, kernel_h, kernel_w, c)
    
    x = np.zeros((batch, h, w, c), dtype=col.dtype)
    
    for y in range(kernel_h):
        for x_pos in range(kernel_w):
            x[
                :,
                y:y + out_h * stride:stride,
                x_pos:x_pos + out_w * stride:stride,
                :
            ] += col[:, :, :, y, x_pos, :]
    
    if padding > 0:
        x = x[:, padding:-padding, padding:-padding, :]
    
    return x


class Conv2D(Module):
    """2D Convolutional layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (filters)
        kernel_size: Size of convolution kernel (int or tuple)
        stride: Stride of convolution
        padding: Zero-padding added to input
        bias: If True, adds learnable bias
        seed: Random seed for initialization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h, self.kernel_w = kernel_size
        self.stride = stride
        self.padding = padding
        
        # initialize weights (Kaiming/He initialization)
        rng = np.random.RandomState(seed)
        fan_in = in_channels * self.kernel_h * self.kernel_w
        std = np.sqrt(2.0 / fan_in)
        
        weight_data = rng.normal(0, std, (out_channels, self.kernel_h, self.kernel_w, in_channels))
        self.weight = Parameter(weight_data)
        
        if bias:
            bias_data = np.zeros(out_channels)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, height, width, channels) or (height, width, channels)
            
        Returns:
            Output tensor (batch, out_height, out_width, out_channels)
        """
        # handle both 3D and 4D input
        if x.data.ndim == 3:
            x_data = x.data[np.newaxis, ...]  # Add batch dimension
            remove_batch = True
        elif x.data.ndim == 4:
            x_data = x.data
            remove_batch = False
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.data.shape}")
        
        batch, h, w, c = x_data.shape
        
        # use im2col for efficient convolution
        col = _im2col(x_data, self.kernel_h, self.kernel_w, self.stride, self.padding)
        
        # reshape weight for matrix multiplication
        weight_col = self.weight.data.reshape(self.out_channels, -1).T
        
        # perform convolution as matrix multiplication
        out = col @ weight_col
        
        # add bias
        if self.bias is not None:
            out = out + self.bias.data
        
        # reshape output
        out_h = (h + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_w) // self.stride + 1
        out = out.reshape(batch, out_h, out_w, self.out_channels)
        
        if remove_batch:
            out = out[0]  # Remove batch dimension
        
        # wrap in Tensor (simplified, no autograd for now)
        return tensor(out, requires_grad=x.requires_grad)


class MaxPool2D(Module):
    """2D Max pooling layer.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (defaults to kernel_size)
    """
    
    def __init__(self, kernel_size: int | tuple[int, int] = 2, stride: int | None = None):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.kernel_h, self.kernel_w = kernel_size
        self.stride = stride if stride is not None else kernel_size[0]
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, height, width, channels) or (height, width, channels)
            
        Returns:
            Pooled tensor
        """
        # handle both 3D and 4D input
        if x.data.ndim == 3:
            x_data = x.data[np.newaxis, ...]
            remove_batch = True
        elif x.data.ndim == 4:
            x_data = x.data
            remove_batch = False
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.data.shape}")
        
        batch, h, w, c = x_data.shape
        
        out_h = (h - self.kernel_h) // self.stride + 1
        out_w = (w - self.kernel_w) // self.stride + 1
        
        out = np.zeros((batch, out_h, out_w, c), dtype=x_data.dtype)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_h
                w_end = w_start + self.kernel_w
                
                window = x_data[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = window.max(axis=(1, 2))
        
        if remove_batch:
            out = out[0]
        
        return tensor(out, requires_grad=x.requires_grad)


class AvgPool2D(Module):
    """2D Average pooling layer.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (defaults to kernel_size)
    """
    
    def __init__(self, kernel_size: int | tuple[int, int] = 2, stride: int | None = None):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.kernel_h, self.kernel_w = kernel_size
        self.stride = stride if stride is not None else kernel_size[0]
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, height, width, channels) or (height, width, channels)
            
        Returns:
            Pooled tensor
        """
        # handle both 3D and 4D input
        if x.data.ndim == 3:
            x_data = x.data[np.newaxis, ...]
            remove_batch = True
        elif x.data.ndim == 4:
            x_data = x.data
            remove_batch = False
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.data.shape}")
        
        batch, h, w, c = x_data.shape
        
        out_h = (h - self.kernel_h) // self.stride + 1
        out_w = (w - self.kernel_w) // self.stride + 1
        
        out = np.zeros((batch, out_h, out_w, c), dtype=x_data.dtype)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_h
                w_end = w_start + self.kernel_w
                
                window = x_data[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = window.mean(axis=(1, 2))
        
        if remove_batch:
            out = out[0]
        
        return tensor(out, requires_grad=x.requires_grad)


class Flatten(Module):
    """Flatten layer for transitioning from conv to fully connected layers."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Flatten all dimensions except batch dimension.
        
        Args:
            x: Input tensor (batch, ...) or (...)
            
        Returns:
            Flattened tensor
        """
        if x.data.ndim == 1:
            return x
        elif x.data.ndim == 2:
            return x  # Already flat (batch, features)
        else:
            # flatten all but first dimension
            batch_size = x.data.shape[0] if x.data.ndim > 1 else 1
            flattened = x.data.reshape(batch_size, -1)
            return tensor(flattened, requires_grad=x.requires_grad)


class BatchNorm2D(Module):
    """2D Batch Normalization layer.
    
    Normalizes the input by computing mean and variance per channel
    and applying a learnable affine transformation.
    
    Args:
        num_features: Number of channels (features) in the input
        eps: Small value for numerical stability
        momentum: Value for running mean/var computation
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Parameter(tensor(np.ones(num_features), requires_grad=True))
        self.beta = Parameter(tensor(np.zeros(num_features), requires_grad=True))
        
        # Running statistics (not learnable)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.training = True
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply batch normalization.
        
        Args:
            x: Input tensor (batch, height, width, channels) or (batch, channels, height, width)
            
        Returns:
            Normalized tensor
        """
        data = x.data
        
        if self.training:
            # Compute mean and variance along batch and spatial dimensions
            if data.ndim == 4:
                # Assume (batch, height, width, channels) format
                axes = (0, 1, 2)  # Average over batch, height, width
            else:
                axes = tuple(range(data.ndim - 1))
            
            mean = np.mean(data, axis=axes, keepdims=True)
            var = np.var(data, axis=axes, keepdims=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.flatten()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.flatten()
        else:
            mean = self.running_mean.reshape([1] * (data.ndim - 1) + [-1])
            var = self.running_var.reshape([1] * (data.ndim - 1) + [-1])
        
        # Normalize
        x_norm = (data - mean) / np.sqrt(var + self.eps)
        
        # Apply affine transformation
        gamma = self.gamma.data.data if hasattr(self.gamma.data, 'data') else self.gamma.data
        beta = self.beta.data.data if hasattr(self.beta.data, 'data') else self.beta.data
        
        out = gamma * x_norm + beta
        
        return tensor(out, requires_grad=x.requires_grad)


__all__ = [
    "Conv2D",
    "MaxPool2D",
    "AvgPool2D",
    "Flatten",
    "BatchNorm2D",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.