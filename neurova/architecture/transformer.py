# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Transformer Architectures for Neurova

Pre-built Transformer architectures for NLP, vision, and sequence tasks.
Includes self-attention, multi-head attention, and popular architectures.

Features:
- Transformer encoder and decoder
- Vision Transformer (ViT)
- BERT-style and GPT-style models
- Built-in positional encoding
- Hyperparameter tuning support
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from .base import BaseArchitecture, ParameterValidator


class MultiHeadAttention:
    """
    Multi-Head Self-Attention mechanism.
    
    Allows model to attend to different representation subspaces.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize weights
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        self.cache = {}
        
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None, training: bool = True) -> np.ndarray:
        """
        Forward pass for multi-head attention.
        
        Parameters
        ----------
        query : np.ndarray
            Query tensor (batch, seq_len, d_model)
        key : np.ndarray
            Key tensor (batch, seq_len, d_model)
        value : np.ndarray
            Value tensor (batch, seq_len, d_model)
        mask : np.ndarray, optional
            Attention mask
        training : bool
            Whether in training mode
            
        Returns
        -------
        output : np.ndarray
            Attention output (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = query @ self.W_q  # (batch, seq, d_model)
        K = key @ self.W_k
        V = value @ self.W_v
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        # Shape: (batch, heads, seq, d_k)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)  # (batch, heads, seq, seq)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * (-1e9)
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Apply dropout
        if self.dropout > 0 and training:
            dropout_mask = (np.random.rand(*attention_weights.shape) > self.dropout)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)
        
        # Apply attention to values
        context = attention_weights @ V  # (batch, heads, seq, d_k)
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = context @ self.W_o
        
        # Cache for backward
        self.cache = {
            'query': query, 'key': key, 'value': value,
            'Q': Q, 'K': K, 'V': V,
            'attention_weights': attention_weights,
            'context': context
        }
        
        return output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Backward pass."""
        batch_size = dout.shape[0]
        seq_len = dout.shape[1]
        
        query = self.cache['query']
        key = self.cache['key']
        value = self.cache['value']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        attention_weights = self.cache['attention_weights']
        context = self.cache['context']
        
        # Gradient through output projection
        dcontext = dout @ self.W_o.T
        dW_o = context.reshape(-1, self.d_model).T @ dout.reshape(-1, self.d_model)
        
        # Reshape for heads
        dcontext = dcontext.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        dcontext = dcontext.transpose(0, 2, 1, 3)
        
        # Gradient through attention
        dV = attention_weights.transpose(0, 1, 3, 2) @ dcontext
        dattention = dcontext @ V.transpose(0, 1, 3, 2)
        
        # Gradient through softmax
        dscores = attention_weights * (dattention - np.sum(dattention * attention_weights, axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(self.d_k)
        
        # Gradient through QK^T
        dQ = dscores @ K
        dK = dscores.transpose(0, 1, 3, 2) @ Q
        
        # Reshape back
        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # Gradient through input projections
        dW_q = query.reshape(-1, self.d_model).T @ dQ.reshape(-1, self.d_model)
        dW_k = key.reshape(-1, self.d_model).T @ dK.reshape(-1, self.d_model)
        dW_v = value.reshape(-1, self.d_model).T @ dV.reshape(-1, self.d_model)
        
        dquery = dQ @ self.W_q.T
        dkey = dK @ self.W_k.T
        dvalue = dV @ self.W_v.T
        
        grads = {'W_q': dW_q, 'W_k': dW_k, 'W_v': dW_v, 'W_o': dW_o}
        return dquery, dkey, dvalue, grads
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class PositionalEncoding:
    """
    Sinusoidal positional encoding for sequence position information.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        self.d_model = d_model
        self.dropout = dropout
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe[np.newaxis, :, :]  # (1, max_len, d_model)
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        
        if self.dropout > 0 and training:
            mask = (np.random.rand(*x.shape) > self.dropout)
            x = x * mask / (1 - self.dropout)
            
        return x


class FeedForward:
    """
    Position-wise feed-forward network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize weights
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros((1, d_model))
        
        self.cache = {}
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        # First linear + ReLU
        hidden = x @ self.W1 + self.b1
        hidden_relu = np.maximum(0, hidden)
        
        # Dropout
        if self.dropout > 0 and training:
            mask = (np.random.rand(*hidden_relu.shape) > self.dropout)
            hidden_relu = hidden_relu * mask / (1 - self.dropout)
            self.cache['dropout_mask'] = mask
        
        # Second linear
        output = hidden_relu @ self.W2 + self.b2
        
        self.cache['x'] = x
        self.cache['hidden'] = hidden
        self.cache['hidden_relu'] = hidden_relu
        
        return output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Backward pass."""
        x = self.cache['x']
        hidden = self.cache['hidden']
        hidden_relu = self.cache['hidden_relu']
        
        # Reshape for matmul if needed
        batch_size, seq_len, _ = dout.shape
        
        dout_flat = dout.reshape(-1, self.d_model)
        hidden_relu_flat = hidden_relu.reshape(-1, self.d_ff)
        
        # Second layer gradients
        dW2 = hidden_relu_flat.T @ dout_flat
        db2 = np.sum(dout_flat, axis=0, keepdims=True)
        
        dhidden_relu = dout_flat @ self.W2.T
        
        # Apply dropout gradient
        if 'dropout_mask' in self.cache:
            dhidden_relu = dhidden_relu * self.cache['dropout_mask'].reshape(-1, self.d_ff)
        
        # ReLU gradient
        dhidden = dhidden_relu * (hidden.reshape(-1, self.d_ff) > 0)
        
        # First layer gradients
        x_flat = x.reshape(-1, self.d_model)
        dW1 = x_flat.T @ dhidden
        db1 = np.sum(dhidden, axis=0, keepdims=True)
        
        dx = dhidden @ self.W1.T
        dx = dx.reshape(batch_size, seq_len, self.d_model)
        
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return dx, grads


class LayerNorm:
    """
    Layer normalization for transformers.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))
        
        self.cache = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize over the last dimension."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        output = self.gamma * x_norm + self.beta
        
        self.cache = {'x': x, 'x_norm': x_norm, 'mean': mean, 'var': var}
        return output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Backward pass."""
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        var = self.cache['var']
        
        dgamma = np.sum(dout * x_norm, axis=(0, 1), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 1), keepdims=True)
        
        dx_norm = dout * self.gamma
        
        N = x.shape[-1]
        dx = (1 / np.sqrt(var + self.eps)) * (
            dx_norm - np.mean(dx_norm, axis=-1, keepdims=True) - 
            x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True)
        )
        
        grads = {'gamma': dgamma, 'beta': dbeta}
        return dx, grads


class TransformerEncoderLayer:
    """
    Single Transformer encoder layer.
    
    Consists of:
    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1):
        self.d_model = d_model
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass."""
        # Self-attention with residual
        attn_out = self.self_attn.forward(x, x, x, mask, training)
        if self.dropout > 0 and training:
            attn_out = attn_out * (np.random.rand(*attn_out.shape) > self.dropout) / (1 - self.dropout)
        x = self.norm1.forward(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn.forward(x, training)
        if self.dropout > 0 and training:
            ffn_out = ffn_out * (np.random.rand(*ffn_out.shape) > self.dropout) / (1 - self.dropout)
        x = self.norm2.forward(x + ffn_out)
        
        return x
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all layer weights."""
        return {
            'attn_W_q': self.self_attn.W_q,
            'attn_W_k': self.self_attn.W_k,
            'attn_W_v': self.self_attn.W_v,
            'attn_W_o': self.self_attn.W_o,
            'ffn_W1': self.ffn.W1,
            'ffn_b1': self.ffn.b1,
            'ffn_W2': self.ffn.W2,
            'ffn_b2': self.ffn.b2,
            'norm1_gamma': self.norm1.gamma,
            'norm1_beta': self.norm1.beta,
            'norm2_gamma': self.norm2.gamma,
            'norm2_beta': self.norm2.beta,
        }


class TransformerDecoderLayer:
    """
    Single Transformer decoder layer.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        self.d_model = d_model
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = dropout
        
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_mask: Optional[np.ndarray] = None,
                cross_mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass."""
        # Masked self-attention
        attn_out = self.self_attn.forward(x, x, x, self_mask, training)
        x = self.norm1.forward(x + attn_out)
        
        # Cross-attention with encoder output
        cross_out = self.cross_attn.forward(x, encoder_output, encoder_output, cross_mask, training)
        x = self.norm2.forward(x + cross_out)
        
        # Feed-forward
        ffn_out = self.ffn.forward(x, training)
        x = self.norm3.forward(x + ffn_out)
        
        return x


class Transformer(BaseArchitecture):
    """
    Full Transformer model for sequence-to-sequence tasks.
    
    Parameters
    ----------
    input_shape : tuple
        (sequence_length, d_model) or just d_model
    output_shape : int
        Number of output classes or vocab size
    d_model : int
        Model dimension
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of encoder/decoder layers
    d_ff : int
        Feed-forward dimension
    dropout : float
        Dropout rate
    max_len : int
        Maximum sequence length for positional encoding
    
    Example
    -------
    >>> model = Transformer(
    ...     input_shape=(100, 512),
    ...     output_shape=10000,
    ...     d_model=512,
    ...     num_heads=8,
    ...     num_layers=6,
    ... )
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0005, 0.001],
        'd_model': [128, 256, 512],
        'num_heads': [4, 8],
        'num_layers': [2, 4, 6],
        'd_ff': [512, 1024, 2048],
        'dropout': [0.1, 0.2, 0.3],
    }
    
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...], Tuple[int, int]],
                 output_shape: Union[int, Tuple[int]],
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 **kwargs):
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_enc_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.max_len = max_len
        
        # Handle input shape
        if isinstance(input_shape, int):
            input_shape = (max_len, input_shape)
        elif len(input_shape) == 1:
            input_shape = (max_len, input_shape[0])
            
        super().__init__(input_shape=input_shape, output_shape=output_shape, **kwargs)
        
    def _build_network(self, **kwargs):
        """Build Transformer architecture."""
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_len, self.dropout_rate)
        
        # Encoder layers
        self.encoder_layers = []
        for i in range(self.num_enc_layers):
            layer = TransformerEncoderLayer(
                self.d_model, self.num_heads, self.d_ff, self.dropout_rate
            )
            self.encoder_layers.append(layer)
            
            # Store weights
            for name, weight in layer.get_weights().items():
                self.weights[f'enc_{i}_{name}'] = weight
        
        # Output projection
        n_classes = self.output_shape[0] if isinstance(self.output_shape, tuple) else self.output_shape
        scale = np.sqrt(2.0 / self.d_model)
        self.weights['output_W'] = np.random.randn(self.d_model, n_classes) * scale
        self.weights['output_b'] = np.zeros((1, n_classes))
        
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through Transformer."""
        # Add positional encoding
        x = self.pos_encoder.forward(X, training)
        
        # Pass through encoder layers
        for i, layer in enumerate(self.encoder_layers):
            # Sync weights
            layer.self_attn.W_q = self.weights[f'enc_{i}_attn_W_q']
            layer.self_attn.W_k = self.weights[f'enc_{i}_attn_W_k']
            layer.self_attn.W_v = self.weights[f'enc_{i}_attn_W_v']
            layer.self_attn.W_o = self.weights[f'enc_{i}_attn_W_o']
            layer.ffn.W1 = self.weights[f'enc_{i}_ffn_W1']
            layer.ffn.b1 = self.weights[f'enc_{i}_ffn_b1']
            layer.ffn.W2 = self.weights[f'enc_{i}_ffn_W2']
            layer.ffn.b2 = self.weights[f'enc_{i}_ffn_b2']
            layer.norm1.gamma = self.weights[f'enc_{i}_norm1_gamma']
            layer.norm1.beta = self.weights[f'enc_{i}_norm1_beta']
            layer.norm2.gamma = self.weights[f'enc_{i}_norm2_gamma']
            layer.norm2.beta = self.weights[f'enc_{i}_norm2_beta']
            
            x = layer.forward(x, training=training)
        
        # Global average pooling over sequence
        x = np.mean(x, axis=1)  # (batch, d_model)
        
        # Output projection
        logits = x @ self.weights['output_W'] + self.weights['output_b']
        
        # Softmax
        return self._softmax(logits)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass (simplified)."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        
        gradients['output_W'] = np.zeros_like(self.weights['output_W'])
        gradients['output_b'] = np.sum(dout, axis=0, keepdims=True)
        
        # Initialize encoder gradients
        for i in range(self.num_enc_layers):
            for name in ['attn_W_q', 'attn_W_k', 'attn_W_v', 'attn_W_o',
                        'ffn_W1', 'ffn_b1', 'ffn_W2', 'ffn_b2',
                        'norm1_gamma', 'norm1_beta', 'norm2_gamma', 'norm2_beta']:
                gradients[f'enc_{i}_{name}'] = np.zeros_like(self.weights[f'enc_{i}_{name}'])
        
        return gradients


class TransformerEncoder(BaseArchitecture):
    """
    Transformer Encoder for classification/regression.
    
    Encoder-only architecture good for classification tasks.
    
    Example
    -------
    >>> model = TransformerEncoder(
    ...     input_shape=(100, 768),
    ...     output_shape=5,
    ...     num_layers=6,
    ... )
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, input_shape, output_shape, d_model: int = 256,
                 num_heads: int = 4, num_layers: int = 4, d_ff: int = 1024,
                 dropout: float = 0.1, **kwargs):
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_enc_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        super().__init__(input_shape=input_shape, output_shape=output_shape, **kwargs)
        
    def _build_network(self, **kwargs):
        """Build encoder architecture."""
        input_dim = self.input_shape[-1]
        
        # Input projection if needed
        if input_dim != self.d_model:
            scale = np.sqrt(2.0 / input_dim)
            self.weights['input_proj_W'] = np.random.randn(input_dim, self.d_model) * scale
            self.weights['input_proj_b'] = np.zeros((1, self.d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=5000)
        
        # Encoder layers
        self.encoder_layers = []
        for i in range(self.num_enc_layers):
            layer = TransformerEncoderLayer(
                self.d_model, self.num_heads, self.d_ff, self.dropout_rate
            )
            self.encoder_layers.append(layer)
            for name, weight in layer.get_weights().items():
                self.weights[f'enc_{i}_{name}'] = weight
        
        # Output
        n_classes = self.output_shape[0] if isinstance(self.output_shape, tuple) else self.output_shape
        scale = np.sqrt(2.0 / self.d_model)
        self.weights['output_W'] = np.random.randn(self.d_model, n_classes) * scale
        self.weights['output_b'] = np.zeros((1, n_classes))
        
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        # Input projection if needed
        if 'input_proj_W' in self.weights:
            x = X @ self.weights['input_proj_W'] + self.weights['input_proj_b']
        else:
            x = X
        
        # Positional encoding
        x = self.pos_encoder.forward(x, training)
        
        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            layer.self_attn.W_q = self.weights[f'enc_{i}_attn_W_q']
            layer.self_attn.W_k = self.weights[f'enc_{i}_attn_W_k']
            layer.self_attn.W_v = self.weights[f'enc_{i}_attn_W_v']
            layer.self_attn.W_o = self.weights[f'enc_{i}_attn_W_o']
            layer.ffn.W1 = self.weights[f'enc_{i}_ffn_W1']
            layer.ffn.b1 = self.weights[f'enc_{i}_ffn_b1']
            layer.ffn.W2 = self.weights[f'enc_{i}_ffn_W2']
            layer.ffn.b2 = self.weights[f'enc_{i}_ffn_b2']
            layer.norm1.gamma = self.weights[f'enc_{i}_norm1_gamma']
            layer.norm1.beta = self.weights[f'enc_{i}_norm1_beta']
            layer.norm2.gamma = self.weights[f'enc_{i}_norm2_gamma']
            layer.norm2.beta = self.weights[f'enc_{i}_norm2_beta']
            x = layer.forward(x, training=training)
        
        # Pool and classify
        x = np.mean(x, axis=1)
        logits = x @ self.weights['output_W'] + self.weights['output_b']
        return self._softmax(logits)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        gradients['output_W'] = np.zeros_like(self.weights['output_W'])
        gradients['output_b'] = np.sum(dout, axis=0, keepdims=True)
        
        for i in range(self.num_enc_layers):
            for name in ['attn_W_q', 'attn_W_k', 'attn_W_v', 'attn_W_o',
                        'ffn_W1', 'ffn_b1', 'ffn_W2', 'ffn_b2',
                        'norm1_gamma', 'norm1_beta', 'norm2_gamma', 'norm2_beta']:
                gradients[f'enc_{i}_{name}'] = np.zeros_like(self.weights[f'enc_{i}_{name}'])
        
        if 'input_proj_W' in self.weights:
            gradients['input_proj_W'] = np.zeros_like(self.weights['input_proj_W'])
            gradients['input_proj_b'] = np.zeros_like(self.weights['input_proj_b'])
        
        return gradients


class VisionTransformer(BaseArchitecture):
    """
    Vision Transformer (ViT) for image classification.
    
    Splits image into patches and processes with Transformer.
    
    Parameters
    ----------
    input_shape : tuple
        Image shape (channels, height, width)
    num_classes : int
        Number of output classes
    patch_size : int
        Size of image patches
    d_model : int
        Model dimension
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of transformer layers
    
    Example
    -------
    >>> model = VisionTransformer(
    ...     input_shape=(3, 224, 224),
    ...     num_classes=1000,
    ...     patch_size=16,
    ...     d_model=768,
    ...     num_layers=12,
    ... )
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0003, 0.001],
        'patch_size': [8, 16, 32],
        'd_model': [256, 384, 768],
        'num_heads': [4, 8, 12],
        'num_layers': [4, 8, 12],
    }
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int,
                 patch_size: int = 16, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, dropout: float = 0.1,
                 **kwargs):
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_enc_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        super().__init__(input_shape=input_shape, output_shape=num_classes, **kwargs)
        
    def _build_network(self, **kwargs):
        """Build ViT architecture."""
        C, H, W = self.input_shape
        
        # Calculate number of patches
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image dimensions must be divisible by patch_size"
        
        self.num_patches = (H // self.patch_size) * (W // self.patch_size)
        patch_dim = C * self.patch_size * self.patch_size
        
        # Patch embedding
        scale = np.sqrt(2.0 / patch_dim)
        self.weights['patch_embed_W'] = np.random.randn(patch_dim, self.d_model) * scale
        self.weights['patch_embed_b'] = np.zeros((1, self.d_model))
        
        # Class token
        self.weights['cls_token'] = np.random.randn(1, 1, self.d_model) * 0.02
        
        # Position embeddings (including class token)
        self.weights['pos_embed'] = np.random.randn(1, self.num_patches + 1, self.d_model) * 0.02
        
        # Transformer encoder layers
        self.encoder_layers = []
        for i in range(self.num_enc_layers):
            layer = TransformerEncoderLayer(
                self.d_model, self.num_heads, self.d_ff, self.dropout_rate
            )
            self.encoder_layers.append(layer)
            for name, weight in layer.get_weights().items():
                self.weights[f'enc_{i}_{name}'] = weight
        
        # Classification head
        n_classes = self.output_shape[0] if isinstance(self.output_shape, tuple) else self.output_shape
        self.weights['head_W'] = np.random.randn(self.d_model, n_classes) * 0.02
        self.weights['head_b'] = np.zeros((1, n_classes))
        
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through ViT."""
        batch_size = X.shape[0]
        C, H, W = self.input_shape
        
        # Reshape to patches
        # X: (batch, C, H, W) -> patches: (batch, num_patches, patch_dim)
        patches = X.reshape(batch_size, C, H // self.patch_size, self.patch_size,
                           W // self.patch_size, self.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(batch_size, self.num_patches, -1)
        
        # Patch embedding
        x = patches @ self.weights['patch_embed_W'] + self.weights['patch_embed_b']
        
        # Prepend class token
        cls_tokens = np.tile(self.weights['cls_token'], (batch_size, 1, 1))
        x = np.concatenate([cls_tokens, x], axis=1)
        
        # Add position embeddings
        x = x + self.weights['pos_embed']
        
        # Dropout
        if self.dropout_rate > 0 and training:
            mask = (np.random.rand(*x.shape) > self.dropout_rate)
            x = x * mask / (1 - self.dropout_rate)
        
        # Transformer encoder
        for i, layer in enumerate(self.encoder_layers):
            layer.self_attn.W_q = self.weights[f'enc_{i}_attn_W_q']
            layer.self_attn.W_k = self.weights[f'enc_{i}_attn_W_k']
            layer.self_attn.W_v = self.weights[f'enc_{i}_attn_W_v']
            layer.self_attn.W_o = self.weights[f'enc_{i}_attn_W_o']
            layer.ffn.W1 = self.weights[f'enc_{i}_ffn_W1']
            layer.ffn.b1 = self.weights[f'enc_{i}_ffn_b1']
            layer.ffn.W2 = self.weights[f'enc_{i}_ffn_W2']
            layer.ffn.b2 = self.weights[f'enc_{i}_ffn_b2']
            layer.norm1.gamma = self.weights[f'enc_{i}_norm1_gamma']
            layer.norm1.beta = self.weights[f'enc_{i}_norm1_beta']
            layer.norm2.gamma = self.weights[f'enc_{i}_norm2_gamma']
            layer.norm2.beta = self.weights[f'enc_{i}_norm2_beta']
            x = layer.forward(x, training=training)
        
        # Use class token for classification
        cls_output = x[:, 0]
        
        # Classification head
        logits = cls_output @ self.weights['head_W'] + self.weights['head_b']
        return self._softmax(logits)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        
        gradients['head_W'] = np.zeros_like(self.weights['head_W'])
        gradients['head_b'] = np.sum(dout, axis=0, keepdims=True)
        gradients['cls_token'] = np.zeros_like(self.weights['cls_token'])
        gradients['pos_embed'] = np.zeros_like(self.weights['pos_embed'])
        gradients['patch_embed_W'] = np.zeros_like(self.weights['patch_embed_W'])
        gradients['patch_embed_b'] = np.zeros_like(self.weights['patch_embed_b'])
        
        for i in range(self.num_enc_layers):
            for name in ['attn_W_q', 'attn_W_k', 'attn_W_v', 'attn_W_o',
                        'ffn_W1', 'ffn_b1', 'ffn_W2', 'ffn_b2',
                        'norm1_gamma', 'norm1_beta', 'norm2_gamma', 'norm2_beta']:
                gradients[f'enc_{i}_{name}'] = np.zeros_like(self.weights[f'enc_{i}_{name}'])
        
        return gradients


# Convenience function
def create_transformer(input_shape, output_shape, architecture: str = 'encoder',
                       **kwargs) -> BaseArchitecture:
    """
    Create a Transformer model with minimal configuration.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape
    output_shape : int
        Number of output classes
    architecture : str
        Type: 'encoder', 'full', 'vit'
    
    Returns
    -------
    model : BaseArchitecture
        Configured Transformer model
    
    Example
    -------
    >>> model = create_transformer((100, 512), 10, 'encoder')
    >>> model.fit(X_train, y_train)
    """
    architectures = {
        'encoder': TransformerEncoder,
        'full': Transformer,
        'vit': VisionTransformer,
    }
    
    arch = architecture.lower()
    if arch not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Choose from: {list(architectures.keys())}")
    
    if arch == 'vit':
        return VisionTransformer(input_shape=input_shape, num_classes=output_shape, **kwargs)
    return architectures[arch](input_shape=input_shape, output_shape=output_shape, **kwargs)
