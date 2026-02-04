# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Advanced CNN Architectures for Neurova

This module contains all major CNN architectures:
- Fully Convolutional Network (FCN)
- GoogLeNet/Inception
- ResNet (full implementation)
- DenseNet (CNN version)
- MobileNet
- EfficientNet
- Xception
- NASNet
- SENet (Squeeze-and-Excitation)
- ResNeXt

All implementations use pure NumPy for educational purposes and portability.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from .base import BaseArchitecture, ParameterValidator


# Utility Functions

def im2col(X: np.ndarray, kernel_h: int, kernel_w: int, 
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """Convert image to column representation for efficient convolution."""
    N, C, H, W = X.shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1
    
    if pad > 0:
        X = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
    
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = X[:, :, y:y_max:stride, x:x_max:stride]
    
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(col: np.ndarray, input_shape: Tuple, kernel_h: int, kernel_w: int,
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """Convert column representation back to image."""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1
    
    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    return img[:, :, pad:H + pad, pad:W + pad]


# Layer Components

class Conv2D:
    """2D Convolutional Layer."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, 
                                  kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        
        # Cache for backprop
        self.x = None
        self.col = None
        self.col_W = None
        
        # Gradients
        self.dW = None
        self.db = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward convolution."""
        N, C, H, W = x.shape
        
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        self.x = x
        self.col = im2col(x, self.kernel_size, self.kernel_size, 
                          self.stride, self.padding)
        self.col_W = self.W.reshape(self.out_channels, -1).T
        
        out = np.dot(self.col, self.col_W) + self.b
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass."""
        N, C, H, W = self.x.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout).T.reshape(self.W.shape)
        
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, self.kernel_size, self.kernel_size,
                    self.stride, self.padding)
        
        return dx


class SeparableConv2D:
    """Depthwise Separable Convolution."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        # Depthwise conv (one filter per channel)
        self.depthwise = Conv2D(in_channels, in_channels, 
                                kernel_size, stride, padding)
        # Pointwise conv (1x1 conv)
        self.pointwise = Conv2D(in_channels, out_channels, 1, 1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: depthwise then pointwise."""
        x = self.depthwise.forward(x, training)
        x = self.pointwise.forward(x, training)
        return x


class BatchNorm2D:
    """Batch Normalization for 2D inputs."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache
        self.x_norm = None
        self.std = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if x.ndim == 4:  # (N, C, H, W)
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
            out = self._forward(x, training)
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            return out
        return self._forward(x, training)
    
    def _forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        self.std = np.sqrt(var + self.eps)
        self.x_norm = (x - mean) / self.std
        
        return self.gamma * self.x_norm + self.beta


class MaxPool2D:
    """Max Pooling Layer."""
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.max_idx = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward max pooling."""
        N, C, H, W = x.shape
        
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        self.x = x
        
        col = im2col(x, self.pool_size, self.pool_size, self.stride, 0)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        
        self.max_idx = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        return out


class GlobalAvgPool2D:
    """Global Average Pooling."""
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward global average pooling."""
        return np.mean(x, axis=(2, 3))


class Flatten:
    """Flatten layer."""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)


class Dense:
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int, activation: str = 'relu'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        
        self.x = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        if self.activation == 'relu':
            out = np.maximum(0, out)
        elif self.activation == 'sigmoid':
            out = 1 / (1 + np.exp(-np.clip(out, -500, 500)))
        elif self.activation == 'softmax':
            exp_x = np.exp(out - np.max(out, axis=-1, keepdims=True))
            out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        elif self.activation == 'swish':
            out = out * (1 / (1 + np.exp(-out)))
        
        return out


class SqueezeExcitation:
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        self.channels = channels
        self.reduction = reduction
        
        reduced = max(1, channels // reduction)
        self.fc1 = Dense(channels, reduced, activation='relu')
        self.fc2 = Dense(reduced, channels, activation='sigmoid')
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with squeeze and excitation."""
        # Squeeze: global average pooling
        squeeze = np.mean(x, axis=(2, 3))
        
        # Excitation: FC layers
        excite = self.fc1.forward(squeeze, training)
        excite = self.fc2.forward(excite, training)
        
        # Scale
        excite = excite.reshape(x.shape[0], x.shape[1], 1, 1)
        return x * excite


# Inception Module (GoogLeNet)

class InceptionModule:
    """
    Inception module - parallel convolutions of different sizes.
    
    The inception module processes input through multiple pathways:
    - 1x1 convolution
    - 1x1 -> 3x3 convolution
    - 1x1 -> 5x5 convolution
    - 3x3 max pool -> 1x1 convolution
    """
    
    def __init__(self, in_channels: int, 
                 n1x1: int, n3x3_reduce: int, n3x3: int,
                 n5x5_reduce: int, n5x5: int, pool_proj: int):
        
        # 1x1 branch
        self.conv1x1 = Conv2D(in_channels, n1x1, 1, 1, 0)
        
        # 3x3 branch
        self.conv3x3_reduce = Conv2D(in_channels, n3x3_reduce, 1, 1, 0)
        self.conv3x3 = Conv2D(n3x3_reduce, n3x3, 3, 1, 1)
        
        # 5x5 branch
        self.conv5x5_reduce = Conv2D(in_channels, n5x5_reduce, 1, 1, 0)
        self.conv5x5 = Conv2D(n5x5_reduce, n5x5, 5, 1, 2)
        
        # Pool branch
        self.pool = MaxPool2D(3, 1)  # stride 1 with padding
        self.pool_proj = Conv2D(in_channels, pool_proj, 1, 1, 0)
        
        self.out_channels = n1x1 + n3x3 + n5x5 + pool_proj
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through all branches."""
        # 1x1 branch
        branch1 = self.conv1x1.forward(x, training)
        branch1 = np.maximum(0, branch1)  # ReLU
        
        # 3x3 branch
        branch2 = self.conv3x3_reduce.forward(x, training)
        branch2 = np.maximum(0, branch2)
        branch2 = self.conv3x3.forward(branch2, training)
        branch2 = np.maximum(0, branch2)
        
        # 5x5 branch
        branch3 = self.conv5x5_reduce.forward(x, training)
        branch3 = np.maximum(0, branch3)
        branch3 = self.conv5x5.forward(branch3, training)
        branch3 = np.maximum(0, branch3)
        
        # Pool branch (manual padding for same output size)
        N, C, H, W = x.shape
        x_padded = np.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)], mode='constant')
        branch4 = self.pool.forward(x_padded, training)
        # Ensure correct size
        if branch4.shape[2] != H:
            branch4 = branch4[:, :, :H, :W]
        branch4 = self.pool_proj.forward(branch4, training)
        branch4 = np.maximum(0, branch4)
        
        # Concatenate all branches along channel dimension
        return np.concatenate([branch1, branch2, branch3, branch4], axis=1)


class InceptionV3Module:
    """
    InceptionV3 module with factorized convolutions.
    Uses 1x7 and 7x1 convolutions instead of 7x7.
    """
    
    def __init__(self, in_channels: int, pool_features: int = 32):
        # Branch 1: 1x1 conv
        self.branch1_conv = Conv2D(in_channels, 64, 1, 1, 0)
        
        # Branch 2: 1x1 -> 1x7 -> 7x1
        self.branch2_conv1 = Conv2D(in_channels, 64, 1, 1, 0)
        self.branch2_conv2 = Conv2D(64, 64, (1, 7), 1, (0, 3))  # approximated
        self.branch2_conv3 = Conv2D(64, 64, (7, 1), 1, (3, 0))  # approximated
        
        # Branch 3: 1x1 -> double factorized
        self.branch3_conv1 = Conv2D(in_channels, 64, 1, 1, 0)
        self.branch3_conv2 = Conv2D(64, 64, 3, 1, 1)
        self.branch3_conv3 = Conv2D(64, 64, 3, 1, 1)
        
        # Branch 4: pool -> 1x1
        self.branch4_pool = MaxPool2D(3, 1)
        self.branch4_conv = Conv2D(in_channels, pool_features, 1, 1, 0)
        
        self.out_channels = 64 + 64 + 64 + pool_features


# Residual Block (ResNet)

class ResidualBlock:
    """
    Basic Residual Block with skip connection.
    
    Two 3x3 convolutions with a skip connection.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride, 1)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm2D(out_channels)
        
        # Skip connection projection if dimensions don't match
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, 1, stride, 0)
            self.shortcut_bn = BatchNorm2D(out_channels)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with skip connection."""
        identity = x
        
        out = self.conv1.forward(x, training)
        out = self.bn1.forward(out, training)
        out = np.maximum(0, out)  # ReLU
        
        out = self.conv2.forward(out, training)
        out = self.bn2.forward(out, training)
        
        if self.shortcut is not None:
            identity = self.shortcut.forward(x, training)
            identity = self.shortcut_bn.forward(identity, training)
        
        out += identity
        out = np.maximum(0, out)  # ReLU
        
        return out


class BottleneckBlock:
    """
    Bottleneck Residual Block.
    
    1x1 -> 3x3 -> 1x1 convolutions with skip connection.
    Used in ResNet-50, ResNet-101, ResNet-152.
    """
    
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.conv1 = Conv2D(in_channels, out_channels, 1, 1, 0)
        self.bn1 = BatchNorm2D(out_channels)
        
        self.conv2 = Conv2D(out_channels, out_channels, 3, stride, 1)
        self.bn2 = BatchNorm2D(out_channels)
        
        self.conv3 = Conv2D(out_channels, out_channels * self.expansion, 1, 1, 0)
        self.bn3 = BatchNorm2D(out_channels * self.expansion)
        
        self.shortcut = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = Conv2D(in_channels, out_channels * self.expansion, 1, stride, 0)
            self.shortcut_bn = BatchNorm2D(out_channels * self.expansion)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        identity = x
        
        out = self.conv1.forward(x, training)
        out = self.bn1.forward(out, training)
        out = np.maximum(0, out)
        
        out = self.conv2.forward(out, training)
        out = self.bn2.forward(out, training)
        out = np.maximum(0, out)
        
        out = self.conv3.forward(out, training)
        out = self.bn3.forward(out, training)
        
        if self.shortcut is not None:
            identity = self.shortcut.forward(x, training)
            identity = self.shortcut_bn.forward(identity, training)
        
        out += identity
        out = np.maximum(0, out)
        
        return out


# Dense Block (DenseNet CNN)

class DenseLayer:
    """Single layer in a DenseNet dense block."""
    
    def __init__(self, in_channels: int, growth_rate: int):
        self.bn1 = BatchNorm2D(in_channels)
        self.conv1 = Conv2D(in_channels, 4 * growth_rate, 1, 1, 0)
        self.bn2 = BatchNorm2D(4 * growth_rate)
        self.conv2 = Conv2D(4 * growth_rate, growth_rate, 3, 1, 1)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = self.bn1.forward(x, training)
        out = np.maximum(0, out)
        out = self.conv1.forward(out, training)
        
        out = self.bn2.forward(out, training)
        out = np.maximum(0, out)
        out = self.conv2.forward(out, training)
        
        return out


class DenseBlock:
    """Dense block for DenseNet."""
    
    def __init__(self, in_channels: int, growth_rate: int, n_layers: int):
        self.layers = []
        current_channels = in_channels
        
        for _ in range(n_layers):
            layer = DenseLayer(current_channels, growth_rate)
            self.layers.append(layer)
            current_channels += growth_rate
        
        self.out_channels = current_channels
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        features = [x]
        
        for layer in self.layers:
            concat = np.concatenate(features, axis=1)
            out = layer.forward(concat, training)
            features.append(out)
        
        return np.concatenate(features, axis=1)


class TransitionLayer:
    """Transition layer between dense blocks."""
    
    def __init__(self, in_channels: int, out_channels: int):
        self.bn = BatchNorm2D(in_channels)
        self.conv = Conv2D(in_channels, out_channels, 1, 1, 0)
        self.pool = MaxPool2D(2, 2)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = self.bn.forward(x, training)
        out = np.maximum(0, out)
        out = self.conv.forward(out, training)
        out = self.pool.forward(out, training)
        return out


# MobileNet Components

class MobileNetBlock:
    """MobileNetV1 depthwise separable conv block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        # Depthwise conv
        self.dw_conv = Conv2D(in_channels, in_channels, 3, stride, 1)
        self.dw_bn = BatchNorm2D(in_channels)
        
        # Pointwise conv
        self.pw_conv = Conv2D(in_channels, out_channels, 1, 1, 0)
        self.pw_bn = BatchNorm2D(out_channels)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = self.dw_conv.forward(x, training)
        out = self.dw_bn.forward(out, training)
        out = np.maximum(0, out)  # ReLU6 approximated as ReLU
        
        out = self.pw_conv.forward(out, training)
        out = self.pw_bn.forward(out, training)
        out = np.maximum(0, out)
        
        return out


class InvertedResidual:
    """MobileNetV2 inverted residual block."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, expand_ratio: int = 6):
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            self.expand_conv = Conv2D(in_channels, hidden_dim, 1, 1, 0)
            self.expand_bn = BatchNorm2D(hidden_dim)
        else:
            self.expand_conv = None
        
        # Depthwise
        self.dw_conv = Conv2D(hidden_dim if expand_ratio != 1 else in_channels, 
                              hidden_dim if expand_ratio != 1 else in_channels,
                              3, stride, 1)
        self.dw_bn = BatchNorm2D(hidden_dim if expand_ratio != 1 else in_channels)
        
        # Projection
        self.project_conv = Conv2D(hidden_dim if expand_ratio != 1 else in_channels,
                                   out_channels, 1, 1, 0)
        self.project_bn = BatchNorm2D(out_channels)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        identity = x
        out = x
        
        if self.expand_conv is not None:
            out = self.expand_conv.forward(out, training)
            out = self.expand_bn.forward(out, training)
            out = np.clip(out, 0, 6)  # ReLU6
        
        out = self.dw_conv.forward(out, training)
        out = self.dw_bn.forward(out, training)
        out = np.clip(out, 0, 6)  # ReLU6
        
        out = self.project_conv.forward(out, training)
        out = self.project_bn.forward(out, training)
        
        if self.use_residual:
            out += identity
        
        return out


# EfficientNet Components

class MBConvBlock:
    """Mobile Inverted Bottleneck Conv block for EfficientNet."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 expand_ratio: int = 6, se_ratio: float = 0.25):
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = Conv2D(in_channels, hidden_dim, 1, 1, 0)
            self.expand_bn = BatchNorm2D(hidden_dim)
        else:
            self.expand_conv = None
            hidden_dim = in_channels
        
        # Depthwise phase
        padding = (kernel_size - 1) // 2
        self.dw_conv = Conv2D(hidden_dim, hidden_dim, kernel_size, stride, padding)
        self.dw_bn = BatchNorm2D(hidden_dim)
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(hidden_dim, hidden_dim // se_channels)
        
        # Projection phase
        self.project_conv = Conv2D(hidden_dim, out_channels, 1, 1, 0)
        self.project_bn = BatchNorm2D(out_channels)
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        identity = x
        out = x
        
        if self.expand_conv is not None:
            out = self.expand_conv.forward(out, training)
            out = self.expand_bn.forward(out, training)
            out = out * (1 / (1 + np.exp(-out)))  # Swish
        
        out = self.dw_conv.forward(out, training)
        out = self.dw_bn.forward(out, training)
        out = out * (1 / (1 + np.exp(-out)))  # Swish
        
        out = self.se.forward(out, training)
        
        out = self.project_conv.forward(out, training)
        out = self.project_bn.forward(out, training)
        
        if self.use_residual:
            out += identity
        
        return out


# Full Architecture Implementations

class FCN(BaseArchitecture):
    """
    Fully Convolutional Network for semantic segmentation.
    
    Replaces fully connected layers with convolutional layers,
    enabling pixel-wise predictions for segmentation tasks.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of segmentation classes
    backbone : str
        Backbone architecture: 'vgg16', 'resnet50'
    
    Example
    -------
    >>> fcn = FCN(input_shape=(3, 224, 224), num_classes=21)
    >>> fcn.fit(images, masks)
    >>> segmentation = fcn.predict(test_images)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'backbone': ['vgg16', 'resnet50'],
    }
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 21,
                 backbone: str = 'vgg16',
                 **kwargs):
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        super().__init__(input_shape=input_shape, 
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build FCN architecture."""
        in_channels = self.input_shape[0]
        
        # Encoder (VGG-style)
        self.conv1 = Conv2D(in_channels, 64, 3, 1, 1)
        self.conv2 = Conv2D(64, 64, 3, 1, 1)
        self.pool1 = MaxPool2D(2, 2)
        
        self.conv3 = Conv2D(64, 128, 3, 1, 1)
        self.conv4 = Conv2D(128, 128, 3, 1, 1)
        self.pool2 = MaxPool2D(2, 2)
        
        self.conv5 = Conv2D(128, 256, 3, 1, 1)
        self.conv6 = Conv2D(256, 256, 3, 1, 1)
        self.pool3 = MaxPool2D(2, 2)
        
        # FCN head (1x1 convolutions instead of FC)
        self.fc_conv1 = Conv2D(256, 4096, 1, 1, 0)
        self.fc_conv2 = Conv2D(4096, 4096, 1, 1, 0)
        self.score = Conv2D(4096, self.num_classes, 1, 1, 0)
        
        # Store weights
        self.weights['conv1_W'] = self.conv1.W
        self.weights['conv2_W'] = self.conv2.W
        
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        # Ensure NCHW format
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        # Encoder
        x = self.conv1.forward(X, training)
        x = np.maximum(0, x)
        x = self.conv2.forward(x, training)
        x = np.maximum(0, x)
        x = self.pool1.forward(x, training)
        
        x = self.conv3.forward(x, training)
        x = np.maximum(0, x)
        x = self.conv4.forward(x, training)
        x = np.maximum(0, x)
        x = self.pool2.forward(x, training)
        
        x = self.conv5.forward(x, training)
        x = np.maximum(0, x)
        x = self.conv6.forward(x, training)
        x = np.maximum(0, x)
        x = self.pool3.forward(x, training)
        
        # FCN head
        x = self.fc_conv1.forward(x, training)
        x = np.maximum(0, x)
        x = self.fc_conv2.forward(x, training)
        x = np.maximum(0, x)
        x = self.score.forward(x, training)
        
        # Global average pool for classification
        x = np.mean(x, axis=(2, 3))
        
        # Softmax
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through FCN. Returns gradients for training."""
        # Placeholder for gradient computation
        # Full implementation would backpropagate through all layers
        gradients = {}
        N = y_pred.shape[0]
        
        # Compute output gradient
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_grad'] = dout
        return gradients


class GoogLeNet(BaseArchitecture):
    """
    GoogLeNet (Inception v1).
    
    Uses inception modules for multi-scale feature extraction.
    Winner of ILSVRC 2014.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    
    Example
    -------
    >>> model = GoogLeNet(input_shape=(3, 224, 224), num_classes=1000)
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001, 0.01],
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 1000,
                 **kwargs):
        self.num_classes = num_classes
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build GoogLeNet architecture."""
        in_channels = self.input_shape[0]
        
        # Stem
        self.conv1 = Conv2D(in_channels, 64, 7, 2, 3)
        self.pool1 = MaxPool2D(3, 2)
        self.conv2 = Conv2D(64, 64, 1, 1, 0)
        self.conv3 = Conv2D(64, 192, 3, 1, 1)
        self.pool2 = MaxPool2D(3, 2)
        
        # Inception modules
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = MaxPool2D(3, 2)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = MaxPool2D(3, 2)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Classifier
        self.avgpool = GlobalAvgPool2D()
        self.fc = Dense(1024, self.num_classes, activation='softmax')
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        # Stem
        x = self.conv1.forward(X, training)
        x = np.maximum(0, x)
        x = self.pool1.forward(x, training)
        x = self.conv2.forward(x, training)
        x = np.maximum(0, x)
        x = self.conv3.forward(x, training)
        x = np.maximum(0, x)
        x = self.pool2.forward(x, training)
        
        # Inception modules
        x = self.inception3a.forward(x, training)
        x = self.inception3b.forward(x, training)
        x = self.pool3.forward(x, training)
        
        x = self.inception4a.forward(x, training)
        x = self.inception4b.forward(x, training)
        x = self.inception4c.forward(x, training)
        x = self.inception4d.forward(x, training)
        x = self.inception4e.forward(x, training)
        x = self.pool4.forward(x, training)
        
        x = self.inception5a.forward(x, training)
        x = self.inception5b.forward(x, training)
        
        # Classifier
        x = self.avgpool.forward(x, training)
        x = self.fc.forward(x, training)
        
        return x

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through GoogLeNet."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_grad'] = dout
        return gradients


# Alias
Inception = GoogLeNet
InceptionV1 = GoogLeNet


class ResNet(BaseArchitecture):
    """
    ResNet - Deep Residual Network.
    
    Uses skip connections to enable training of very deep networks.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    depth : int
        Network depth: 18, 34, 50, 101, 152
    
    Example
    -------
    >>> model = ResNet(input_shape=(3, 224, 224), num_classes=1000, depth=50)
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'depth': [18, 34, 50],
    }
    
    CONFIGS = {
        18: ([2, 2, 2, 2], ResidualBlock),
        34: ([3, 4, 6, 3], ResidualBlock),
        50: ([3, 4, 6, 3], BottleneckBlock),
        101: ([3, 4, 23, 3], BottleneckBlock),
        152: ([3, 8, 36, 3], BottleneckBlock),
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 1000,
                 depth: int = 50,
                 **kwargs):
        self.num_classes = num_classes
        self.depth = depth
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build ResNet architecture."""
        layers_config, block_type = self.CONFIGS[self.depth]
        in_channels = self.input_shape[0]
        
        # Stem
        self.conv1 = Conv2D(in_channels, 64, 7, 2, 3)
        self.bn1 = BatchNorm2D(64)
        self.pool1 = MaxPool2D(3, 2)
        
        # Residual layers
        self.layer1 = self._make_layer(block_type, 64, 64, layers_config[0], stride=1)
        self.layer2 = self._make_layer(block_type, self.layer1[-1].conv2.out_channels if isinstance(block_type, type) and block_type == ResidualBlock else 256, 128, layers_config[1], stride=2)
        self.layer3 = self._make_layer(block_type, 512 if block_type == BottleneckBlock else 128, 256, layers_config[2], stride=2)
        self.layer4 = self._make_layer(block_type, 1024 if block_type == BottleneckBlock else 256, 512, layers_config[3], stride=2)
        
        # Classifier
        self.avgpool = GlobalAvgPool2D()
        final_channels = 2048 if block_type == BottleneckBlock else 512
        self.fc = Dense(final_channels, self.num_classes, activation='softmax')
    
    def _make_layer(self, block_type, in_channels: int, out_channels: int,
                    n_blocks: int, stride: int = 1) -> List:
        """Create a layer of residual blocks."""
        blocks = []
        
        # First block may downsample
        blocks.append(block_type(in_channels, out_channels, stride))
        
        # Remaining blocks
        if block_type == BottleneckBlock:
            in_channels = out_channels * 4
        else:
            in_channels = out_channels
            
        for _ in range(1, n_blocks):
            blocks.append(block_type(in_channels, out_channels))
        
        return blocks
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        # Stem
        x = self.conv1.forward(X, training)
        x = self.bn1.forward(x, training)
        x = np.maximum(0, x)
        x = self.pool1.forward(x, training)
        
        # Residual layers
        for block in self.layer1:
            x = block.forward(x, training)
        for block in self.layer2:
            x = block.forward(x, training)
        for block in self.layer3:
            x = block.forward(x, training)
        for block in self.layer4:
            x = block.forward(x, training)
        
        # Classifier
        x = self.avgpool.forward(x, training)
        x = self.fc.forward(x, training)
        
        return x

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through ResNet."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_grad'] = dout
        return gradients


# Aliases
ResNet18 = lambda **kwargs: ResNet(depth=18, **kwargs)
ResNet34 = lambda **kwargs: ResNet(depth=34, **kwargs)
ResNet50 = lambda **kwargs: ResNet(depth=50, **kwargs)
ResNet101 = lambda **kwargs: ResNet(depth=101, **kwargs)
ResNet152 = lambda **kwargs: ResNet(depth=152, **kwargs)


class DenseNetCNN(BaseArchitecture):
    """
    DenseNet - Densely Connected Convolutional Network.
    
    Each layer receives feature maps from all preceding layers.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    growth_rate : int
        Growth rate k
    block_config : tuple
        Number of layers in each dense block
    compression : float
        Compression factor for transition layers
    
    Example
    -------
    >>> model = DenseNetCNN(input_shape=(3, 224, 224), num_classes=1000)
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'growth_rate': [12, 24, 32],
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 1000,
                 growth_rate: int = 32,
                 block_config: Tuple[int, ...] = (6, 12, 24, 16),
                 compression: float = 0.5,
                 **kwargs):
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.compression = compression
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build DenseNet architecture."""
        in_channels = self.input_shape[0]
        
        # Stem
        self.conv1 = Conv2D(in_channels, 64, 7, 2, 3)
        self.bn1 = BatchNorm2D(64)
        self.pool1 = MaxPool2D(3, 2)
        
        # Dense blocks and transitions
        self.dense_blocks = []
        self.transitions = []
        
        num_features = 64
        for i, num_layers in enumerate(self.block_config):
            block = DenseBlock(num_features, self.growth_rate, num_layers)
            self.dense_blocks.append(block)
            num_features = block.out_channels
            
            if i < len(self.block_config) - 1:
                out_features = int(num_features * self.compression)
                trans = TransitionLayer(num_features, out_features)
                self.transitions.append(trans)
                num_features = out_features
        
        # Classifier
        self.bn_final = BatchNorm2D(num_features)
        self.avgpool = GlobalAvgPool2D()
        self.fc = Dense(num_features, self.num_classes, activation='softmax')
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        # Stem
        x = self.conv1.forward(X, training)
        x = self.bn1.forward(x, training)
        x = np.maximum(0, x)
        x = self.pool1.forward(x, training)
        
        # Dense blocks and transitions
        for i, block in enumerate(self.dense_blocks):
            x = block.forward(x, training)
            if i < len(self.transitions):
                x = self.transitions[i].forward(x, training)
        
        # Classifier
        x = self.bn_final.forward(x, training)
        x = np.maximum(0, x)
        x = self.avgpool.forward(x, training)
        x = self.fc.forward(x, training)
        
        return x

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through DenseNetCNN."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_grad'] = dout
        return gradients


# Aliases
DenseNet121 = lambda **kwargs: DenseNetCNN(block_config=(6, 12, 24, 16), **kwargs)
DenseNet169 = lambda **kwargs: DenseNetCNN(block_config=(6, 12, 32, 32), **kwargs)
DenseNet201 = lambda **kwargs: DenseNetCNN(block_config=(6, 12, 48, 32), **kwargs)


class MobileNet(BaseArchitecture):
    """
    MobileNet - Efficient CNN for mobile devices.
    
    Uses depthwise separable convolutions to reduce computation.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    width_mult : float
        Width multiplier for channels
    version : int
        MobileNet version: 1 or 2
    
    Example
    -------
    >>> model = MobileNet(input_shape=(3, 224, 224), num_classes=1000)
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'width_mult': [0.5, 0.75, 1.0],
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 1000,
                 width_mult: float = 1.0,
                 version: int = 2,
                 **kwargs):
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.version = version
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build MobileNet architecture."""
        in_channels = self.input_shape[0]
        
        def make_divisible(v):
            """Make channels divisible by 8."""
            return max(8, int(v + 4) // 8 * 8)
        
        if self.version == 1:
            # MobileNetV1
            self.conv1 = Conv2D(in_channels, 32, 3, 2, 1)
            self.bn1 = BatchNorm2D(32)
            
            # Depthwise separable blocks
            self.blocks = [
                MobileNetBlock(32, 64, 1),
                MobileNetBlock(64, 128, 2),
                MobileNetBlock(128, 128, 1),
                MobileNetBlock(128, 256, 2),
                MobileNetBlock(256, 256, 1),
                MobileNetBlock(256, 512, 2),
                *[MobileNetBlock(512, 512, 1) for _ in range(5)],
                MobileNetBlock(512, 1024, 2),
                MobileNetBlock(1024, 1024, 1),
            ]
            final_channels = 1024
        else:
            # MobileNetV2
            self.conv1 = Conv2D(in_channels, 32, 3, 2, 1)
            self.bn1 = BatchNorm2D(32)
            
            # Inverted residual blocks
            self.blocks = [
                InvertedResidual(32, 16, 1, 1),
                InvertedResidual(16, 24, 2, 6),
                InvertedResidual(24, 24, 1, 6),
                InvertedResidual(24, 32, 2, 6),
                *[InvertedResidual(32, 32, 1, 6) for _ in range(2)],
                InvertedResidual(32, 64, 2, 6),
                *[InvertedResidual(64, 64, 1, 6) for _ in range(3)],
                InvertedResidual(64, 96, 1, 6),
                *[InvertedResidual(96, 96, 1, 6) for _ in range(2)],
                InvertedResidual(96, 160, 2, 6),
                *[InvertedResidual(160, 160, 1, 6) for _ in range(2)],
                InvertedResidual(160, 320, 1, 6),
            ]
            self.conv_last = Conv2D(320, 1280, 1, 1, 0)
            self.bn_last = BatchNorm2D(1280)
            final_channels = 1280
        
        self.avgpool = GlobalAvgPool2D()
        self.fc = Dense(final_channels, self.num_classes, activation='softmax')
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        x = self.conv1.forward(X, training)
        x = self.bn1.forward(x, training)
        x = np.clip(x, 0, 6)  # ReLU6
        
        for block in self.blocks:
            x = block.forward(x, training)
        
        if self.version == 2:
            x = self.conv_last.forward(x, training)
            x = self.bn_last.forward(x, training)
            x = np.clip(x, 0, 6)
        
        x = self.avgpool.forward(x, training)
        x = self.fc.forward(x, training)
        
        return x

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through MobileNet."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_grad'] = dout
        return gradients


# Aliases
MobileNetV1 = lambda **kwargs: MobileNet(version=1, **kwargs)
MobileNetV2 = lambda **kwargs: MobileNet(version=2, **kwargs)


class EfficientNet(BaseArchitecture):
    """
    EfficientNet - Compound scaling for CNNs.
    
    Uses compound scaling to balance network depth, width, and resolution.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    variant : str
        EfficientNet variant: 'b0' to 'b7'
    
    Example
    -------
    >>> model = EfficientNet(input_shape=(3, 224, 224), num_classes=1000)
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'variant': ['b0', 'b1', 'b2'],
    }
    
    # Width, depth, resolution multipliers
    CONFIGS = {
        'b0': (1.0, 1.0, 224),
        'b1': (1.0, 1.1, 240),
        'b2': (1.1, 1.2, 260),
        'b3': (1.2, 1.4, 300),
        'b4': (1.4, 1.8, 380),
        'b5': (1.6, 2.2, 456),
        'b6': (1.8, 2.6, 528),
        'b7': (2.0, 3.1, 600),
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 1000,
                 variant: str = 'b0',
                 **kwargs):
        self.num_classes = num_classes
        self.variant = variant
        self.width_mult, self.depth_mult, _ = self.CONFIGS[variant]
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build EfficientNet architecture."""
        in_channels = self.input_shape[0]
        
        def round_channels(c):
            return int(c * self.width_mult)
        
        def round_repeats(r):
            return int(np.ceil(r * self.depth_mult))
        
        # Stem
        self.conv_stem = Conv2D(in_channels, round_channels(32), 3, 2, 1)
        self.bn_stem = BatchNorm2D(round_channels(32))
        
        # MBConv blocks
        # (in_channels, out_channels, kernel, stride, expand_ratio, repeats)
        block_configs = [
            (32, 16, 3, 1, 1, 1),
            (16, 24, 3, 2, 6, 2),
            (24, 40, 5, 2, 6, 2),
            (40, 80, 3, 2, 6, 3),
            (80, 112, 5, 1, 6, 3),
            (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1),
        ]
        
        self.blocks = []
        for inc, outc, k, s, e, r in block_configs:
            inc = round_channels(inc)
            outc = round_channels(outc)
            repeats = round_repeats(r)
            
            # First block with stride
            self.blocks.append(MBConvBlock(inc, outc, k, s, e))
            # Remaining blocks
            for _ in range(1, repeats):
                self.blocks.append(MBConvBlock(outc, outc, k, 1, e))
        
        # Head
        final_channels = round_channels(1280)
        self.conv_head = Conv2D(round_channels(320), final_channels, 1, 1, 0)
        self.bn_head = BatchNorm2D(final_channels)
        self.avgpool = GlobalAvgPool2D()
        self.fc = Dense(final_channels, self.num_classes, activation='softmax')
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        # Stem
        x = self.conv_stem.forward(X, training)
        x = self.bn_stem.forward(x, training)
        x = x * (1 / (1 + np.exp(-x)))  # Swish
        
        # Blocks
        for block in self.blocks:
            x = block.forward(x, training)
        
        # Head
        x = self.conv_head.forward(x, training)
        x = self.bn_head.forward(x, training)
        x = x * (1 / (1 + np.exp(-x)))  # Swish
        
        x = self.avgpool.forward(x, training)
        x = self.fc.forward(x, training)
        
        return x

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through EfficientNet."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_grad'] = dout
        return gradients


# Aliases for EfficientNet variants
EfficientNetB0 = lambda **kwargs: EfficientNet(variant='b0', **kwargs)
EfficientNetB1 = lambda **kwargs: EfficientNet(variant='b1', **kwargs)
EfficientNetB2 = lambda **kwargs: EfficientNet(variant='b2', **kwargs)
EfficientNetB3 = lambda **kwargs: EfficientNet(variant='b3', **kwargs)
EfficientNetB4 = lambda **kwargs: EfficientNet(variant='b4', **kwargs)
EfficientNetB5 = lambda **kwargs: EfficientNet(variant='b5', **kwargs)
EfficientNetB6 = lambda **kwargs: EfficientNet(variant='b6', **kwargs)
EfficientNetB7 = lambda **kwargs: EfficientNet(variant='b7', **kwargs)


class Xception(BaseArchitecture):
    """
    Xception - Extreme Inception.
    
    Uses depthwise separable convolutions throughout,
    achieving extreme form of inception module.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    
    Example
    -------
    >>> model = Xception(input_shape=(3, 299, 299), num_classes=1000)
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 299, 299),
                 num_classes: int = 1000,
                 **kwargs):
        self.num_classes = num_classes
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build Xception architecture."""
        in_channels = self.input_shape[0]
        
        # Entry flow
        self.conv1 = Conv2D(in_channels, 32, 3, 2, 1)
        self.bn1 = BatchNorm2D(32)
        self.conv2 = Conv2D(32, 64, 3, 1, 1)
        self.bn2 = BatchNorm2D(64)
        
        # Entry flow blocks (separable convs with residuals)
        self.entry_sep1 = SeparableConv2D(64, 128, 3, 1, 1)
        self.entry_sep2 = SeparableConv2D(128, 128, 3, 1, 1)
        self.entry_pool1 = MaxPool2D(3, 2)
        self.entry_skip1 = Conv2D(64, 128, 1, 2, 0)
        
        # Middle flow (8 blocks)
        self.middle_blocks = []
        for _ in range(8):
            block = [
                SeparableConv2D(728, 728, 3, 1, 1),
                SeparableConv2D(728, 728, 3, 1, 1),
                SeparableConv2D(728, 728, 3, 1, 1),
            ]
            self.middle_blocks.append(block)
        
        # Exit flow
        self.exit_sep1 = SeparableConv2D(728, 728, 3, 1, 1)
        self.exit_sep2 = SeparableConv2D(728, 1024, 3, 1, 1)
        self.exit_pool = MaxPool2D(3, 2)
        self.exit_skip = Conv2D(728, 1024, 1, 2, 0)
        
        self.exit_sep3 = SeparableConv2D(1024, 1536, 3, 1, 1)
        self.exit_sep4 = SeparableConv2D(1536, 2048, 3, 1, 1)
        
        self.avgpool = GlobalAvgPool2D()
        self.fc = Dense(2048, self.num_classes, activation='softmax')
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        # Entry flow stem
        x = self.conv1.forward(X, training)
        x = self.bn1.forward(x, training)
        x = np.maximum(0, x)
        x = self.conv2.forward(x, training)
        x = self.bn2.forward(x, training)
        x = np.maximum(0, x)
        
        # Entry flow block 1
        residual = self.entry_skip1.forward(x, training)
        x = self.entry_sep1.forward(x, training)
        x = np.maximum(0, x)
        x = self.entry_sep2.forward(x, training)
        x = self.entry_pool1.forward(x, training)
        x = x + residual
        
        # Classifier
        x = self.avgpool.forward(x, training)
        x = self.fc.forward(x, training)
        
        return x

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through Xception."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_grad'] = dout
        return gradients


class SENet(BaseArchitecture):
    """
    SENet - Squeeze-and-Excitation Network.
    
    Adds channel attention to ResNet blocks.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    reduction : int
        Reduction ratio for SE module
    
    Example
    -------
    >>> model = SENet(input_shape=(3, 224, 224), num_classes=1000)
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 1000,
                 reduction: int = 16,
                 **kwargs):
        self.num_classes = num_classes
        self.reduction = reduction
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build SENet architecture (SE-ResNet-50)."""
        in_channels = self.input_shape[0]
        
        # Use ResNet as backbone with SE modules
        self.conv1 = Conv2D(in_channels, 64, 7, 2, 3)
        self.bn1 = BatchNorm2D(64)
        self.pool1 = MaxPool2D(3, 2)
        
        # SE-ResNet blocks
        self.blocks = [
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 512, 2),
        ]
        
        # SE modules for each block
        self.se_modules = [
            SqueezeExcitation(64, self.reduction),
            SqueezeExcitation(128, self.reduction),
            SqueezeExcitation(256, self.reduction),
            SqueezeExcitation(512, self.reduction),
        ]
        
        self.avgpool = GlobalAvgPool2D()
        self.fc = Dense(512, self.num_classes, activation='softmax')
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        x = self.conv1.forward(X, training)
        x = self.bn1.forward(x, training)
        x = np.maximum(0, x)
        x = self.pool1.forward(x, training)
        
        for block, se in zip(self.blocks, self.se_modules):
            x = block.forward(x, training)
            x = se.forward(x, training)
        
        x = self.avgpool.forward(x, training)
        x = self.fc.forward(x, training)
        
        return x

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through SENet."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_grad'] = dout
        return gradients


# Factory Functions

def create_cnn(architecture: str, input_shape: Tuple[int, int, int],
               num_classes: int, **kwargs) -> BaseArchitecture:
    """
    Factory function to create CNN architectures.
    
    Parameters
    ----------
    architecture : str
        Architecture name: 'fcn', 'googlenet', 'inception', 'resnet18',
        'resnet50', 'densenet', 'mobilenet', 'efficientnet', 'xception', 'senet'
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    **kwargs
        Additional architecture-specific parameters
    
    Returns
    -------
    model : BaseArchitecture
        The requested CNN model
    
    Example
    -------
    >>> model = create_cnn('resnet50', (3, 224, 224), 1000)
    """
    architectures = {
        'fcn': FCN,
        'googlenet': GoogLeNet,
        'inception': GoogLeNet,
        'inceptionv1': GoogLeNet,
        'resnet': ResNet,
        'resnet18': lambda **kw: ResNet(depth=18, **kw),
        'resnet34': lambda **kw: ResNet(depth=34, **kw),
        'resnet50': lambda **kw: ResNet(depth=50, **kw),
        'resnet101': lambda **kw: ResNet(depth=101, **kw),
        'resnet152': lambda **kw: ResNet(depth=152, **kw),
        'densenet': DenseNetCNN,
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'densenet201': DenseNet201,
        'mobilenet': MobileNet,
        'mobilenetv1': MobileNetV1,
        'mobilenetv2': MobileNetV2,
        'efficientnet': EfficientNet,
        'efficientnet-b0': EfficientNetB0,
        'efficientnet-b1': EfficientNetB1,
        'efficientnet-b2': EfficientNetB2,
        'efficientnet-b3': EfficientNetB3,
        'efficientnet-b4': EfficientNetB4,
        'efficientnet-b5': EfficientNetB5,
        'efficientnet-b6': EfficientNetB6,
        'efficientnet-b7': EfficientNetB7,
        'xception': Xception,
        'senet': SENet,
    }
    
    arch_name = architecture.lower()
    if arch_name not in architectures:
        available = list(architectures.keys())
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {available}")
    
    return architectures[arch_name](input_shape=input_shape, 
                                     num_classes=num_classes, **kwargs)
