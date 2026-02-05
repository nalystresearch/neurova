# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Linear (fully connected) layers.

Neurova implementation
"""

from __future__ import annotations
import numpy as np
from neurova.nn.layers import Module
from neurova.nn.autograd import Tensor, Parameter


class Linear(Module):
    """
    Linear (fully connected) layer: y = xW^T + b
    
    Neurova implementation
    
    Parameters
    ----------
    in_features : int
        Size of each input sample
    out_features : int
        Size of each output sample
    bias : bool, default=True
        If True, adds a learnable bias
    
    Attributes
    ----------
    weight : Parameter
        Learnable weights of shape (out_features, in_features)
    bias : Parameter or None
        Learnable bias of shape (out_features,)
    
    Examples
    --------
    >>> layer = Linear(784, 256)
    >>> x = Tensor(np.random.randn(32, 784), requires_grad=True)
    >>> y = layer(x)  # Shape: (32, 256)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # initialize weights with Kaiming/He initialization
        k = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(
            np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        )
        
        if bias:
            self.bias = Parameter(
                np.random.uniform(-k, k, (out_features,)).astype(np.float32)
            )
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (*, in_features)
        
        Returns
        -------
        Tensor
            Output tensor of shape (*, out_features)
        """
        # y = xW^T + b
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def __repr__(self) -> str:
        return (f'Linear(in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None})')


class Bilinear(Module):
    """
    Bilinear layer: y = x1 @ W @ x2 + b
    
    Neurova implementation
    
    Parameters
    ----------
    in1_features : int
        Size of first input
    in2_features : int
        Size of second input
    out_features : int
        Size of output
    bias : bool, default=True
        If True, adds learnable bias
    """
    
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        
        k = 1.0 / np.sqrt(in1_features)
        self.weight = Parameter(
            np.random.uniform(-k, k, (out_features, in1_features, in2_features))
        )
        
        if bias:
            self.bias = Parameter(np.random.uniform(-k, k, (out_features,)))
        else:
            self.bias = None
    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x1 : Tensor
            First input of shape (*, in1_features)
        x2 : Tensor
            Second input of shape (*, in2_features)
        
        Returns
        -------
        Tensor
            Output of shape (*, out_features)
        """
        # simplified bilinear: for each output dim, compute x1 @ W[i] @ x2
        batch_shape = x1.shape[:-1]
        x1_flat = x1.reshape(-1, self.in1_features)
        x2_flat = x2.reshape(-1, self.in2_features)
        
        batch_size = x1_flat.shape[0]
        output = np.zeros((batch_size, self.out_features), dtype=np.float32)
        
        for i in range(self.out_features):
            # output[:, i] = (x1 @ W[i]) * x2 summed
            temp = x1_flat.data @ self.weight.data[i]  # (batch, in2_features)
            output[:, i] = np.sum(temp * x2_flat.data, axis=1)
        
        out = Tensor(output, requires_grad=x1.requires_grad or x2.requires_grad)
        out = out.reshape(*batch_shape, self.out_features)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def __repr__(self) -> str:
        return (f'Bilinear(in1_features={self.in1_features}, '
                f'in2_features={self.in2_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None})')


class Identity(Module):
    """
    Identity layer - returns input unchanged.
    
    Neurova implementation
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - returns input unchanged."""
        return x
    
    def __repr__(self) -> str:
        return 'Identity()'
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.