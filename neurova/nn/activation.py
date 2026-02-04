# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Activation functions - Placeholder for full implementation."""
from neurova.nn.layers import Module
from neurova.nn.autograd import Tensor

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()  # Simplified

class PReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class ELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class SELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        return x  # Placeholder

class LogSoftmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        return x  # Placeholder

class Swish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()

class Mish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * x.tanh()


class SiLU(Module):
    """
    Sigmoid Linear Unit (SiLU/Swish) activation.
    
    f(x) = x * sigmoid(x)
    
    Also known as the Swish activation function.
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        sigmoid = 1 / (1 + np.exp(-np.clip(data, -500, 500)))
        result = data * sigmoid
        return Tensor(result) if hasattr(x, 'data') else result


class Softplus(Module):
    """
    Softplus activation: log(1 + exp(x)).
    
    Smooth approximation of ReLU.
    
    Parameters
    ----------
    beta : float, default=1.0
        Scaling factor. softplus(x) = (1/beta) * log(1 + exp(beta * x))
    threshold : float, default=20.0
        Values above this revert to linear function for stability.
    """
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        # For numerical stability
        scaled = self.beta * data
        result = np.where(
            scaled > self.threshold,
            data,  # Linear for large values
            (1 / self.beta) * np.log1p(np.exp(scaled))
        )
        return Tensor(result) if hasattr(x, 'data') else result


class Softsign(Module):
    """
    Softsign activation: x / (1 + |x|).
    
    Similar to tanh but converges more slowly to -1 and 1.
    """
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        result = data / (1 + np.abs(data))
        return Tensor(result) if hasattr(x, 'data') else result


class Hardtanh(Module):
    """
    Hard Tanh activation.
    
    Piecewise linear approximation of tanh.
    
    Parameters
    ----------
    min_val : float, default=-1.0
        Minimum value of the linear region
    max_val : float, default=1.0
        Maximum value of the linear region
    inplace : bool, default=False
        Whether to do the operation in-place
    """
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0,
                 inplace: bool = False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        result = np.clip(data, self.min_val, self.max_val)
        return Tensor(result) if hasattr(x, 'data') else result


class Hardswish(Module):
    """
    Hard Swish activation.
    
    Piecewise linear approximation of SiLU/Swish, used in MobileNetV3.
    
    f(x) = x * (ReLU6(x + 3) / 6)
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        result = data * np.clip(data + 3, 0, 6) / 6
        return Tensor(result) if hasattr(x, 'data') else result


class Hardsigmoid(Module):
    """
    Hard Sigmoid activation.
    
    Piecewise linear approximation of sigmoid.
    
    f(x) = ReLU6(x + 3) / 6
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        result = np.clip(data + 3, 0, 6) / 6
        return Tensor(result) if hasattr(x, 'data') else result


class ReLU6(Module):
    """
    ReLU6 activation: min(max(0, x), 6).
    
    Used in MobileNet architectures.
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        result = np.clip(data, 0, 6)
        return Tensor(result) if hasattr(x, 'data') else result


class CELU(Module):
    """
    Continuously Differentiable ELU.
    
    f(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    
    Parameters
    ----------
    alpha : float, default=1.0
        The alpha value for the CELU formulation
    """
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        result = np.maximum(0, data) + np.minimum(0, self.alpha * (np.exp(data / self.alpha) - 1))
        return Tensor(result) if hasattr(x, 'data') else result


class GLU(Module):
    """
    Gated Linear Unit.
    
    f(a, b) = a * sigmoid(b) where input is split in half along dim.
    
    Parameters
    ----------
    dim : int, default=-1
        Dimension along which to split the input
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        a, b = np.split(data, 2, axis=self.dim)
        sigmoid_b = 1 / (1 + np.exp(-np.clip(b, -500, 500)))
        result = a * sigmoid_b
        return Tensor(result) if hasattr(x, 'data') else result


class Threshold(Module):
    """
    Thresholds each element of the input.
    
    f(x) = x if x > threshold else value
    
    Parameters
    ----------
    threshold : float
        The threshold value
    value : float
        The replacement value
    """
    def __init__(self, threshold: float, value: float, inplace: bool = False):
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        result = np.where(data > self.threshold, data, self.value)
        return Tensor(result) if hasattr(x, 'data') else result


class Softmin(Module):
    """
    Softmin function: softmax(-x).
    
    Parameters
    ----------
    dim : int, default=-1
        Dimension along which to compute softmin
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        # Softmax of -x
        neg_data = -data
        exp_data = np.exp(neg_data - np.max(neg_data, axis=self.dim, keepdims=True))
        result = exp_data / np.sum(exp_data, axis=self.dim, keepdims=True)
        return Tensor(result) if hasattr(x, 'data') else result


class Softmax2d(Module):
    """
    Applies softmax over features to each spatial location.
    
    When given image of shape (N, C, H, W), applies softmax to (N, C, x, y).
    """
    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        data = x.data if hasattr(x, 'data') else x
        # Apply softmax over channel dimension (dim=1)
        exp_data = np.exp(data - np.max(data, axis=1, keepdims=True))
        result = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        return Tensor(result) if hasattr(x, 'data') else result


class MultiheadAttention(Module):
    """
    Multi-Head Attention mechanism.
    
    Parameters
    ----------
    embed_dim : int
        Total dimension of the model
    num_heads : int
        Number of parallel attention heads
    dropout : float, default=0.0
        Dropout probability on attention weights
    bias : bool, default=True
        Add bias to input/output projections
    add_bias_kv : bool, default=False
        Add bias to key and value sequences
    kdim : int, optional
        Total number of features for keys. Default: embed_dim
    vdim : int, optional
        Total number of features for values. Default: embed_dim
    batch_first : bool, default=False
        If True, input/output shape is (batch, seq, feature)
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, add_bias_kv: bool = False,
                 kdim: int = None, vdim: int = None, batch_first: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        # Initialize projection weights
        import numpy as np
        scale = np.sqrt(2.0 / (embed_dim + embed_dim))
        self.q_proj_weight = np.random.randn(embed_dim, embed_dim) * scale
        self.k_proj_weight = np.random.randn(embed_dim, self.kdim) * scale
        self.v_proj_weight = np.random.randn(embed_dim, self.vdim) * scale
        self.out_proj_weight = np.random.randn(embed_dim, embed_dim) * scale
        
        if bias:
            self.q_proj_bias = np.zeros(embed_dim)
            self.k_proj_bias = np.zeros(embed_dim)
            self.v_proj_bias = np.zeros(embed_dim)
            self.out_proj_bias = np.zeros(embed_dim)
        else:
            self.q_proj_bias = None
            self.k_proj_bias = None
            self.v_proj_bias = None
            self.out_proj_bias = None
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None,
                need_weights: bool = True):
        """
        Forward pass of multi-head attention.
        
        Parameters
        ----------
        query : Tensor of shape (L, N, E) or (N, L, E) if batch_first
        key : Tensor of shape (S, N, E) or (N, S, E) if batch_first
        value : Tensor of shape (S, N, E) or (N, S, E) if batch_first
        
        Returns
        -------
        attn_output : Tensor of shape (L, N, E) or (N, L, E)
        attn_output_weights : Tensor of shape (N, L, S) if need_weights
        """
        import numpy as np
        
        q = query.data if hasattr(query, 'data') else query
        k = key.data if hasattr(key, 'data') else key
        v = value.data if hasattr(value, 'data') else value
        
        if self.batch_first:
            q = np.transpose(q, (1, 0, 2))
            k = np.transpose(k, (1, 0, 2))
            v = np.transpose(v, (1, 0, 2))
        
        tgt_len, batch_size, embed_dim = q.shape
        src_len = k.shape[0]
        
        # Linear projections
        Q = np.dot(q.reshape(-1, embed_dim), self.q_proj_weight.T)
        K = np.dot(k.reshape(-1, self.kdim), self.k_proj_weight.T)
        V = np.dot(v.reshape(-1, self.vdim), self.v_proj_weight.T)
        
        if self.q_proj_bias is not None:
            Q = Q + self.q_proj_bias
            K = K + self.k_proj_bias
            V = V + self.v_proj_bias
        
        # Reshape for multi-head attention
        Q = Q.reshape(tgt_len, batch_size, self.num_heads, self.head_dim)
        K = K.reshape(src_len, batch_size, self.num_heads, self.head_dim)
        V = V.reshape(src_len, batch_size, self.num_heads, self.head_dim)
        
        # Transpose to (batch, heads, seq, dim)
        Q = np.transpose(Q, (1, 2, 0, 3))
        K = np.transpose(K, (1, 2, 0, 3))
        V = np.transpose(V, (1, 2, 0, 3))
        
        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        attn_weights = np.matmul(Q, np.transpose(K, (0, 1, 3, 2))) / scale
        
        if attn_mask is not None:
            mask = attn_mask.data if hasattr(attn_mask, 'data') else attn_mask
            attn_weights = attn_weights + mask
        
        if key_padding_mask is not None:
            mask = key_padding_mask.data if hasattr(key_padding_mask, 'data') else key_padding_mask
            attn_weights = np.where(mask[:, np.newaxis, np.newaxis, :], -np.inf, attn_weights)
        
        # Softmax
        attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-9)
        
        # Apply attention to values
        attn_output = np.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = np.transpose(attn_output, (2, 0, 1, 3))
        attn_output = attn_output.reshape(tgt_len, batch_size, embed_dim)
        
        # Output projection
        attn_output = np.dot(attn_output.reshape(-1, embed_dim), self.out_proj_weight.T)
        if self.out_proj_bias is not None:
            attn_output = attn_output + self.out_proj_bias
        attn_output = attn_output.reshape(tgt_len, batch_size, embed_dim)
        
        if self.batch_first:
            attn_output = np.transpose(attn_output, (1, 0, 2))
        
        attn_output = Tensor(attn_output) if hasattr(query, 'data') else attn_output
        
        if need_weights:
            avg_weights = np.mean(attn_weights, axis=1)
            return attn_output, avg_weights
        return attn_output, None


# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.