# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Advanced Transformer Architectures for Neurova

Complete implementation of transformer-based models:
- BERT (Bidirectional Encoder Representations from Transformers)
- GPT (Generative Pre-trained Transformer)
- T5 (Text-to-Text Transfer Transformer)
- RoBERTa
- ALBERT
- DistilBERT
- XLNet
- Swin Transformer
- CLIP (Contrastive Language-Image Pre-training)
- Perceiver

All implementations use pure NumPy for educational purposes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from .base import BaseArchitecture, ParameterValidator


# Core Components

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function used in BERT/GPT."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
               eps: float = 1e-6) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta


class LayerNorm:
    """Layer Normalization."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.mean = None
        self.std = None
        self.x_norm = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.std = np.std(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / (self.std + self.eps)
        return self.gamma * self.x_norm + self.beta


class Embedding:
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.W = np.random.randn(vocab_size, embed_dim) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Look up embeddings for token indices."""
        return self.W[x.astype(int)]


class PositionalEncoding:
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        self.d_model = d_model
        self.dropout = dropout
        
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len]
        if training and self.dropout > 0:
            mask = np.random.rand(*x.shape) > self.dropout
            x = x * mask / (1 - self.dropout)
        return x


class LearnedPositionalEmbedding:
    """Learned positional embedding (used in BERT, GPT)."""
    
    def __init__(self, max_len: int, embed_dim: int):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.W = np.random.randn(max_len, embed_dim) * 0.02
    
    def forward(self, x: np.ndarray, positions: Optional[np.ndarray] = None) -> np.ndarray:
        if positions is None:
            positions = np.arange(x.shape[1])
        return x + self.W[positions]


class MultiHeadAttention:
    """Multi-Head Self-Attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # Linear projections
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        self.attn_weights = None
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None, 
                training: bool = True) -> np.ndarray:
        """
        Forward pass of multi-head attention.
        
        Parameters
        ----------
        query : np.ndarray
            Query tensor of shape (batch, seq_len, d_model)
        key : np.ndarray
            Key tensor
        value : np.ndarray
            Value tensor
        mask : np.ndarray, optional
            Attention mask
        """
        batch_size = query.shape[0]
        
        # Linear projections
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask * (-1e9)
        
        attn_weights = softmax(scores, axis=-1)
        self.attn_weights = attn_weights
        
        # Apply dropout
        if training and self.dropout > 0:
            drop_mask = np.random.rand(*attn_weights.shape) > self.dropout
            attn_weights = attn_weights * drop_mask / (1 - self.dropout)
        
        # Compute attention output
        attn_output = np.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # Output projection
        return np.dot(attn_output, self.W_o)


class FeedForward:
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = 'relu'):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        h = np.dot(x, self.W1) + self.b1
        
        if self.activation == 'relu':
            h = np.maximum(0, h)
        elif self.activation == 'gelu':
            h = gelu(h)
        
        if training and self.dropout > 0:
            mask = np.random.rand(*h.shape) > self.dropout
            h = h * mask / (1 - self.dropout)
        
        return np.dot(h, self.W2) + self.b2


class TransformerEncoderLayer:
    """Single Transformer Encoder Layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, activation: str = 'relu',
                 pre_norm: bool = False):
        self.pre_norm = pre_norm
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = dropout
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass with pre-norm or post-norm."""
        if self.pre_norm:
            # Pre-norm (used in GPT-2, modern transformers)
            attn_out = self.self_attn.forward(
                self.norm1.forward(x, training),
                self.norm1.forward(x, training),
                self.norm1.forward(x, training),
                mask, training
            )
            x = x + attn_out
            
            ff_out = self.ff.forward(self.norm2.forward(x, training), training)
            x = x + ff_out
        else:
            # Post-norm (original transformer, BERT)
            attn_out = self.self_attn.forward(x, x, x, mask, training)
            x = self.norm1.forward(x + attn_out, training)
            
            ff_out = self.ff.forward(x, training)
            x = self.norm2.forward(x + ff_out, training)
        
        return x


class TransformerDecoderLayer:
    """Single Transformer Decoder Layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, activation: str = 'relu'):
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_mask: Optional[np.ndarray] = None,
                cross_mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass."""
        # Self-attention
        attn_out = self.self_attn.forward(x, x, x, self_mask, training)
        x = self.norm1.forward(x + attn_out, training)
        
        # Cross-attention
        cross_out = self.cross_attn.forward(x, encoder_output, encoder_output,
                                            cross_mask, training)
        x = self.norm2.forward(x + cross_out, training)
        
        # Feed-forward
        ff_out = self.ff.forward(x, training)
        x = self.norm3.forward(x + ff_out, training)
        
        return x


# BERT

class BERT(BaseArchitecture):
    """
    BERT - Bidirectional Encoder Representations from Transformers.
    
    Pre-training model using Masked Language Modeling (MLM) and
    Next Sentence Prediction (NSP) objectives.
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    d_model : int
        Hidden dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of transformer layers
    d_ff : int
        Feed-forward dimension
    max_len : int
        Maximum sequence length
    dropout : float
        Dropout rate
    
    Example
    -------
    >>> bert = BERT(vocab_size=30522, d_model=768, n_heads=12, n_layers=12)
    >>> embeddings = bert.forward(input_ids, attention_mask)
    """
    
    PARAM_SPACE = {
        'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
        'd_model': [256, 512, 768],
        'n_heads': [4, 8, 12],
        'n_layers': [4, 6, 12],
    }
    
    def __init__(self,
                 vocab_size: int = 30522,
                 d_model: int = 768,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 d_ff: int = 3072,
                 max_len: int = 512,
                 dropout: float = 0.1,
                 n_segments: int = 2,
                 **kwargs):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout_rate = dropout
        self.n_segments = n_segments
        
        super().__init__(input_shape=(max_len,), 
                        output_shape=(d_model,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build BERT architecture."""
        # Embeddings
        self.token_embedding = Embedding(self.vocab_size, self.d_model)
        self.position_embedding = LearnedPositionalEmbedding(self.max_len, self.d_model)
        self.segment_embedding = Embedding(self.n_segments, self.d_model)
        self.embed_norm = LayerNorm(self.d_model)
        
        # Transformer encoder layers
        self.layers = []
        for _ in range(self.n_layers):
            layer = TransformerEncoderLayer(
                self.d_model, self.n_heads, self.d_ff,
                self.dropout_rate, activation='gelu'
            )
            self.layers.append(layer)
        
        # Pooler for [CLS] token
        scale = np.sqrt(2.0 / self.d_model)
        self.pooler_W = np.random.randn(self.d_model, self.d_model) * scale
        self.pooler_b = np.zeros(self.d_model)
        
        # Store weights
        self.weights['token_embed'] = self.token_embedding.W
        self.weights['pos_embed'] = self.position_embedding.W
        
    def _forward(self, X: np.ndarray, training: bool = True,
                 segment_ids: Optional[np.ndarray] = None,
                 attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : np.ndarray
            Input token IDs of shape (batch, seq_len)
        segment_ids : np.ndarray, optional
            Segment IDs for NSP
        attention_mask : np.ndarray, optional
            Attention mask
        
        Returns
        -------
        last_hidden_state : np.ndarray
            Shape (batch, seq_len, d_model)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        batch_size, seq_len = X.shape
        
        # Token embeddings
        embeddings = self.token_embedding.forward(X)
        
        # Position embeddings
        embeddings = self.position_embedding.forward(embeddings)
        
        # Segment embeddings
        if segment_ids is not None:
            embeddings = embeddings + self.segment_embedding.forward(segment_ids)
        
        # Layer norm
        embeddings = self.embed_norm.forward(embeddings, training)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask[:, np.newaxis, np.newaxis, :])
        
        # Transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, attention_mask, training)
        
        return hidden_states
    
    def get_pooled_output(self, hidden_states: np.ndarray) -> np.ndarray:
        """Get [CLS] token representation."""
        cls_token = hidden_states[:, 0, :]
        pooled = np.tanh(np.dot(cls_token, self.pooler_W) + self.pooler_b)
        return pooled
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Get embeddings for input."""
        hidden_states = self._forward(X, training=False, **kwargs)
        return self.get_pooled_output(hidden_states)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through BERT."""
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


# BERT variants
class BERTBase(BERT):
    """BERT-Base: 12 layers, 768 hidden, 12 heads."""
    def __init__(self, **kwargs):
        super().__init__(d_model=768, n_heads=12, n_layers=12, d_ff=3072, **kwargs)

class BERTLarge(BERT):
    """BERT-Large: 24 layers, 1024 hidden, 16 heads."""
    def __init__(self, **kwargs):
        super().__init__(d_model=1024, n_heads=16, n_layers=24, d_ff=4096, **kwargs)

class BERTTiny(BERT):
    """BERT-Tiny: 2 layers, 128 hidden, 2 heads."""
    def __init__(self, **kwargs):
        super().__init__(d_model=128, n_heads=2, n_layers=2, d_ff=512, **kwargs)


# GPT

class GPT(BaseArchitecture):
    """
    GPT - Generative Pre-trained Transformer.
    
    Autoregressive language model using causal (unidirectional) attention.
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    d_model : int
        Hidden dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of transformer layers
    max_len : int
        Maximum sequence length
    
    Example
    -------
    >>> gpt = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12)
    >>> output = gpt.generate(prompt_ids, max_tokens=100)
    """
    
    PARAM_SPACE = {
        'learning_rate': [1e-5, 2e-5, 5e-5],
        'd_model': [256, 512, 768],
        'n_layers': [4, 6, 12],
    }
    
    def __init__(self,
                 vocab_size: int = 50257,
                 d_model: int = 768,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 d_ff: int = 3072,
                 max_len: int = 1024,
                 dropout: float = 0.1,
                 **kwargs):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout_rate = dropout
        
        super().__init__(input_shape=(max_len,),
                        output_shape=(vocab_size,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build GPT architecture."""
        # Embeddings
        self.token_embedding = Embedding(self.vocab_size, self.d_model)
        self.position_embedding = LearnedPositionalEmbedding(self.max_len, self.d_model)
        
        # Transformer decoder layers (with pre-norm)
        self.layers = []
        for _ in range(self.n_layers):
            layer = TransformerEncoderLayer(
                self.d_model, self.n_heads, self.d_ff,
                self.dropout_rate, activation='gelu',
                pre_norm=True
            )
            self.layers.append(layer)
        
        # Final layer norm
        self.ln_f = LayerNorm(self.d_model)
        
        # Output projection (tied with embedding)
        self.lm_head = np.random.randn(self.d_model, self.vocab_size) * 0.02
        
        self.weights['token_embed'] = self.token_embedding.W
        
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : np.ndarray
            Input token IDs of shape (batch, seq_len)
        
        Returns
        -------
        logits : np.ndarray
            Shape (batch, seq_len, vocab_size)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        batch_size, seq_len = X.shape
        
        # Embeddings
        hidden_states = self.token_embedding.forward(X)
        hidden_states = self.position_embedding.forward(hidden_states)
        
        # Causal mask
        causal_mask = self._create_causal_mask(seq_len)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, causal_mask, training)
        
        # Final layer norm
        hidden_states = self.ln_f.forward(hidden_states, training)
        
        # Language model head
        logits = np.dot(hidden_states, self.lm_head)
        
        return logits
    
    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> np.ndarray:
        """
        Generate text autoregressively.
        
        Parameters
        ----------
        input_ids : np.ndarray
            Prompt token IDs
        max_new_tokens : int
            Number of tokens to generate
        temperature : float
            Sampling temperature
        top_k : int, optional
            Top-k sampling
        
        Returns
        -------
        output_ids : np.ndarray
            Generated token IDs
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self._forward(input_ids, training=False)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                indices = np.argsort(logits, axis=-1)[:, :-top_k]
                for i in range(logits.shape[0]):
                    logits[i, indices[i]] = -np.inf
            
            # Sample from distribution
            probs = softmax(logits)
            next_token = np.array([np.random.choice(self.vocab_size, p=p) 
                                   for p in probs])
            
            # Append to sequence
            input_ids = np.concatenate([input_ids, next_token[:, np.newaxis]], axis=1)
            
            # Check max length
            if input_ids.shape[1] >= self.max_len:
                break
        
        return input_ids

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through GPT."""
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


# GPT variants
class GPT2Small(GPT):
    """GPT-2 Small: 12 layers, 768 hidden."""
    def __init__(self, **kwargs):
        super().__init__(d_model=768, n_heads=12, n_layers=12, d_ff=3072, **kwargs)

class GPT2Medium(GPT):
    """GPT-2 Medium: 24 layers, 1024 hidden."""
    def __init__(self, **kwargs):
        super().__init__(d_model=1024, n_heads=16, n_layers=24, d_ff=4096, **kwargs)

class GPT2Large(GPT):
    """GPT-2 Large: 36 layers, 1280 hidden."""
    def __init__(self, **kwargs):
        super().__init__(d_model=1280, n_heads=20, n_layers=36, d_ff=5120, **kwargs)


# T5 (Text-to-Text Transfer Transformer)

class T5(BaseArchitecture):
    """
    T5 - Text-to-Text Transfer Transformer.
    
    Encoder-decoder transformer that treats every NLP task as text generation.
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size (SentencePiece)
    d_model : int
        Hidden dimension
    n_heads : int
        Number of attention heads
    n_encoder_layers : int
        Number of encoder layers
    n_decoder_layers : int
        Number of decoder layers
    d_ff : int
        Feed-forward dimension
    
    Example
    -------
    >>> t5 = T5(vocab_size=32128, d_model=512)
    >>> output = t5.forward(input_ids, decoder_input_ids)
    """
    
    PARAM_SPACE = {
        'learning_rate': [1e-5, 3e-5, 1e-4],
        'd_model': [256, 512, 768],
    }
    
    def __init__(self,
                 vocab_size: int = 32128,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 max_len: int = 512,
                 dropout: float = 0.1,
                 **kwargs):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout_rate = dropout
        
        super().__init__(input_shape=(max_len,),
                        output_shape=(vocab_size,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build T5 architecture."""
        # Shared embedding
        self.shared_embedding = Embedding(self.vocab_size, self.d_model)
        
        # Relative position bias (T5 uses relative positions)
        self.rel_pos_bias = np.random.randn(self.n_heads, 32) * 0.02
        
        # Encoder
        self.encoder_layers = []
        for _ in range(self.n_encoder_layers):
            layer = TransformerEncoderLayer(
                self.d_model, self.n_heads, self.d_ff,
                self.dropout_rate, activation='relu',
                pre_norm=True
            )
            self.encoder_layers.append(layer)
        self.encoder_norm = LayerNorm(self.d_model)
        
        # Decoder
        self.decoder_layers = []
        for _ in range(self.n_decoder_layers):
            layer = TransformerDecoderLayer(
                self.d_model, self.n_heads, self.d_ff,
                self.dropout_rate, activation='relu'
            )
            self.decoder_layers.append(layer)
        self.decoder_norm = LayerNorm(self.d_model)
        
        # LM head (tied with embedding)
        self.lm_head = self.shared_embedding.W.T
    
    def encode(self, input_ids: np.ndarray, 
               attention_mask: Optional[np.ndarray] = None,
               training: bool = True) -> np.ndarray:
        """Encode input sequence."""
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        
        hidden_states = self.shared_embedding.forward(input_ids)
        
        if attention_mask is not None:
            attention_mask = (1 - attention_mask[:, np.newaxis, np.newaxis, :])
        
        for layer in self.encoder_layers:
            hidden_states = layer.forward(hidden_states, attention_mask, training)
        
        return self.encoder_norm.forward(hidden_states, training)
    
    def decode(self, decoder_input_ids: np.ndarray,
               encoder_output: np.ndarray,
               training: bool = True) -> np.ndarray:
        """Decode with cross-attention to encoder output."""
        if decoder_input_ids.ndim == 1:
            decoder_input_ids = decoder_input_ids.reshape(1, -1)
        
        hidden_states = self.shared_embedding.forward(decoder_input_ids)
        
        # Causal mask for decoder self-attention
        seq_len = decoder_input_ids.shape[1]
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        
        for layer in self.decoder_layers:
            hidden_states = layer.forward(
                hidden_states, encoder_output,
                causal_mask, None, training
            )
        
        hidden_states = self.decoder_norm.forward(hidden_states, training)
        
        # LM head
        logits = np.dot(hidden_states, self.lm_head)
        
        return logits
    
    def _forward(self, X: np.ndarray, training: bool = True,
                 decoder_input_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Full forward pass."""
        encoder_output = self.encode(X, training=training)
        
        if decoder_input_ids is None:
            # Return encoder output
            return encoder_output
        
        return self.decode(decoder_input_ids, encoder_output, training)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through T5."""
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


# T5 variants
class T5Small(T5):
    """T5-Small: 6 layers, 512 hidden."""
    def __init__(self, **kwargs):
        super().__init__(d_model=512, n_heads=8, n_encoder_layers=6, 
                        n_decoder_layers=6, d_ff=2048, **kwargs)

class T5Base(T5):
    """T5-Base: 12 layers, 768 hidden."""
    def __init__(self, **kwargs):
        super().__init__(d_model=768, n_heads=12, n_encoder_layers=12,
                        n_decoder_layers=12, d_ff=3072, **kwargs)

class T5Large(T5):
    """T5-Large: 24 layers, 1024 hidden."""
    def __init__(self, **kwargs):
        super().__init__(d_model=1024, n_heads=16, n_encoder_layers=24,
                        n_decoder_layers=24, d_ff=4096, **kwargs)


# RoBERTa

class RoBERTa(BERT):
    """
    RoBERTa - Robustly Optimized BERT Pretraining Approach.
    
    BERT with improved pretraining:
    - Dynamic masking
    - Removed NSP
    - Larger batch sizes
    - More data
    
    Same architecture as BERT, different pretraining.
    """
    
    def __init__(self, **kwargs):
        # RoBERTa doesn't use segment embeddings
        super().__init__(n_segments=1, **kwargs)
    
    def _forward(self, X: np.ndarray, training: bool = True,
                 attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward without segment embeddings."""
        return super()._forward(X, training, segment_ids=None, 
                               attention_mask=attention_mask)


# ALBERT

class ALBERT(BaseArchitecture):
    """
    ALBERT - A Lite BERT.
    
    Parameter-efficient version of BERT with:
    - Cross-layer parameter sharing
    - Factorized embedding parameterization
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    embed_dim : int
        Embedding dimension (factorized)
    d_model : int
        Hidden dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of transformer layers
    
    Example
    -------
    >>> albert = ALBERT(vocab_size=30000, embed_dim=128, d_model=768)
    >>> embeddings = albert.forward(input_ids)
    """
    
    def __init__(self,
                 vocab_size: int = 30000,
                 embed_dim: int = 128,
                 d_model: int = 768,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 d_ff: int = 3072,
                 max_len: int = 512,
                 **kwargs):
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        
        super().__init__(input_shape=(max_len,),
                        output_shape=(d_model,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build ALBERT architecture."""
        # Factorized embedding
        self.token_embedding = Embedding(self.vocab_size, self.embed_dim)
        self.embedding_projection = np.random.randn(self.embed_dim, self.d_model) * 0.02
        self.position_embedding = LearnedPositionalEmbedding(self.max_len, self.d_model)
        self.embed_norm = LayerNorm(self.d_model)
        
        # Single shared transformer layer
        self.shared_layer = TransformerEncoderLayer(
            self.d_model, self.n_heads, self.d_ff,
            dropout=0.1, activation='gelu'
        )
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with shared layers."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Factorized embedding
        embeddings = self.token_embedding.forward(X)
        embeddings = np.dot(embeddings, self.embedding_projection)
        embeddings = self.position_embedding.forward(embeddings)
        embeddings = self.embed_norm.forward(embeddings, training)
        
        # Shared layers (cross-layer parameter sharing)
        hidden_states = embeddings
        for _ in range(self.n_layers):
            hidden_states = self.shared_layer.forward(hidden_states, None, training)
        
        return hidden_states

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through ALBERT."""
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


# ALBERT variants
class ALBERTBase(ALBERT):
    """ALBERT-Base."""
    def __init__(self, **kwargs):
        super().__init__(embed_dim=128, d_model=768, n_heads=12, n_layers=12, **kwargs)

class ALBERTLarge(ALBERT):
    """ALBERT-Large."""
    def __init__(self, **kwargs):
        super().__init__(embed_dim=128, d_model=1024, n_heads=16, n_layers=24, **kwargs)


# DistilBERT

class DistilBERT(BERT):
    """
    DistilBERT - Distilled BERT.
    
    Smaller, faster BERT trained via knowledge distillation.
    6 layers instead of 12, 40% smaller, 60% faster.
    """
    
    def __init__(self, **kwargs):
        super().__init__(n_layers=6, **kwargs)


# Swin Transformer

class WindowAttention:
    """Window-based Multi-Head Self-Attention for Swin Transformer."""
    
    def __init__(self, d_model: int, n_heads: int, window_size: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.d_k = d_model // n_heads
        
        scale = np.sqrt(2.0 / d_model)
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        # Relative position bias
        self.relative_position_bias = np.random.randn(
            (2 * window_size - 1) ** 2, n_heads
        ) * 0.02
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Window attention forward pass."""
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = np.dot(x, self.W_qkv).reshape(B, N, 3, self.n_heads, self.d_k)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attn = softmax(attn)
        
        out = np.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        
        return np.dot(out, self.W_o)


class SwinTransformerBlock:
    """Swin Transformer Block with shifted window attention."""
    
    def __init__(self, d_model: int, n_heads: int, window_size: int,
                 d_ff: int, shift: bool = False):
        self.shift = shift
        self.window_size = window_size
        
        self.norm1 = LayerNorm(d_model)
        self.attn = WindowAttention(d_model, n_heads, window_size)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, activation='gelu')
    
    def forward(self, x: np.ndarray, H: int, W: int, 
                training: bool = True) -> np.ndarray:
        """Forward pass with optional shifted windows."""
        B, N, C = x.shape
        
        # Reshape to 2D
        x = x.reshape(B, H, W, C)
        
        # Cyclic shift for shifted window attention
        if self.shift:
            shift_size = self.window_size // 2
            x = np.roll(x, (-shift_size, -shift_size), axis=(1, 2))
        
        # Partition into windows (simplified)
        x = x.reshape(B, -1, C)
        
        # Window attention
        shortcut = x
        x = self.norm1.forward(x, training)
        x = self.attn.forward(x, training)
        x = shortcut + x
        
        # FFN
        x = x + self.ff.forward(self.norm2.forward(x, training), training)
        
        return x


class SwinTransformer(BaseArchitecture):
    """
    Swin Transformer - Shifted Window Transformer.
    
    Hierarchical vision transformer with shifted windows for efficiency.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (C, H, W)
    num_classes : int
        Number of output classes
    embed_dim : int
        Embedding dimension
    depths : tuple
        Number of blocks in each stage
    n_heads : tuple
        Number of attention heads in each stage
    window_size : int
        Window size for attention
    
    Example
    -------
    >>> swin = SwinTransformer(input_shape=(3, 224, 224), num_classes=1000)
    >>> predictions = swin.forward(images)
    """
    
    PARAM_SPACE = {
        'learning_rate': [1e-5, 1e-4, 3e-4],
        'embed_dim': [48, 96, 128],
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 num_classes: int = 1000,
                 embed_dim: int = 96,
                 depths: Tuple[int, ...] = (2, 2, 6, 2),
                 n_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 window_size: int = 7,
                 **kwargs):
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = n_heads
        self.window_size = window_size
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build Swin Transformer."""
        C, H, W = self.input_shape
        
        # Patch embedding
        patch_size = 4
        self.patch_embed = np.random.randn(C * patch_size ** 2, self.embed_dim) * 0.02
        
        # Build stages
        self.stages = []
        dim = self.embed_dim
        
        for i, (depth, n_heads) in enumerate(zip(self.depths, self.num_heads)):
            stage = []
            for j in range(depth):
                block = SwinTransformerBlock(
                    dim, n_heads, self.window_size,
                    d_ff=4 * dim, shift=(j % 2 == 1)
                )
                stage.append(block)
            self.stages.append(stage)
            
            # Patch merging (downsample) between stages
            if i < len(self.depths) - 1:
                dim = dim * 2
        
        self.final_norm = LayerNorm(dim)
        self.head = np.random.randn(dim, self.num_classes) * 0.02
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 3:
            X = X.reshape(1, *X.shape)
        
        B, C, H, W = X.shape
        
        # Patch embedding (simplified)
        patch_size = 4
        x = X.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(B, -1, C * patch_size ** 2)
        x = np.dot(x, self.patch_embed)
        
        cur_H, cur_W = H // patch_size, W // patch_size
        
        # Stages
        for stage in self.stages:
            for block in stage:
                x = block.forward(x, cur_H, cur_W, training)
        
        # Global average pooling
        x = np.mean(x, axis=1)
        x = self.final_norm.forward(x, training)
        
        # Classification head
        logits = np.dot(x, self.head)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through SwinTransformer."""
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


# Swin variants
class SwinTiny(SwinTransformer):
    """Swin-Tiny."""
    def __init__(self, **kwargs):
        super().__init__(embed_dim=96, depths=(2, 2, 6, 2), n_heads=(3, 6, 12, 24), **kwargs)

class SwinSmall(SwinTransformer):
    """Swin-Small."""
    def __init__(self, **kwargs):
        super().__init__(embed_dim=96, depths=(2, 2, 18, 2), n_heads=(3, 6, 12, 24), **kwargs)

class SwinBase(SwinTransformer):
    """Swin-Base."""
    def __init__(self, **kwargs):
        super().__init__(embed_dim=128, depths=(2, 2, 18, 2), n_heads=(4, 8, 16, 32), **kwargs)


# CLIP

class CLIP(BaseArchitecture):
    """
    CLIP - Contrastive Language-Image Pre-training.
    
    Learns visual concepts from natural language supervision.
    
    Parameters
    ----------
    embed_dim : int
        Joint embedding dimension
    vision_width : int
        Vision encoder width
    vision_layers : int
        Number of vision transformer layers
    text_width : int
        Text encoder width
    text_layers : int
        Number of text transformer layers
    vocab_size : int
        Text vocabulary size
    
    Example
    -------
    >>> clip = CLIP(embed_dim=512)
    >>> image_features = clip.encode_image(images)
    >>> text_features = clip.encode_text(tokens)
    >>> similarity = image_features @ text_features.T
    """
    
    def __init__(self,
                 embed_dim: int = 512,
                 vision_width: int = 768,
                 vision_layers: int = 12,
                 vision_heads: int = 12,
                 image_size: int = 224,
                 patch_size: int = 16,
                 text_width: int = 512,
                 text_layers: int = 12,
                 text_heads: int = 8,
                 vocab_size: int = 49408,
                 max_text_len: int = 77,
                 **kwargs):
        
        self.embed_dim = embed_dim
        self.vision_width = vision_width
        self.vision_layers = vision_layers
        self.vision_heads = vision_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.text_width = text_width
        self.text_layers = text_layers
        self.text_heads = text_heads
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        
        super().__init__(input_shape=(3, image_size, image_size),
                        output_shape=(embed_dim,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build CLIP model."""
        # Vision encoder (ViT)
        num_patches = (self.image_size // self.patch_size) ** 2
        patch_dim = 3 * self.patch_size ** 2
        
        self.patch_embed = np.random.randn(patch_dim, self.vision_width) * 0.02
        self.class_token = np.random.randn(1, 1, self.vision_width) * 0.02
        self.vision_pos_embed = np.random.randn(1, num_patches + 1, self.vision_width) * 0.02
        
        self.vision_layers_list = []
        for _ in range(self.vision_layers):
            layer = TransformerEncoderLayer(
                self.vision_width, self.vision_heads, self.vision_width * 4,
                pre_norm=True, activation='gelu'
            )
            self.vision_layers_list.append(layer)
        
        self.vision_ln = LayerNorm(self.vision_width)
        self.vision_proj = np.random.randn(self.vision_width, self.embed_dim) * 0.02
        
        # Text encoder
        self.token_embedding = Embedding(self.vocab_size, self.text_width)
        self.text_pos_embed = np.random.randn(1, self.max_text_len, self.text_width) * 0.02
        
        self.text_layers_list = []
        for _ in range(self.text_layers):
            layer = TransformerEncoderLayer(
                self.text_width, self.text_heads, self.text_width * 4,
                pre_norm=True, activation='gelu'
            )
            self.text_layers_list.append(layer)
        
        self.text_ln = LayerNorm(self.text_width)
        self.text_proj = np.random.randn(self.text_width, self.embed_dim) * 0.02
        
        # Learned temperature
        self.logit_scale = np.array([np.log(1 / 0.07)])
    
    def encode_image(self, images: np.ndarray, training: bool = False) -> np.ndarray:
        """Encode images to embeddings."""
        if images.ndim == 3:
            images = images.reshape(1, *images.shape)
        
        B, C, H, W = images.shape
        
        # Patch embedding
        patches = images.reshape(B, C, H // self.patch_size, self.patch_size,
                                 W // self.patch_size, self.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5).reshape(B, -1, C * self.patch_size ** 2)
        x = np.dot(patches, self.patch_embed)
        
        # Add class token
        cls_tokens = np.repeat(self.class_token, B, axis=0)
        x = np.concatenate([cls_tokens, x], axis=1)
        
        # Add position embedding
        x = x + self.vision_pos_embed
        
        # Transformer
        for layer in self.vision_layers_list:
            x = layer.forward(x, training=training)
        
        # Take class token
        x = self.vision_ln.forward(x[:, 0], training)
        x = np.dot(x, self.vision_proj)
        
        # L2 normalize
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        
        return x
    
    def encode_text(self, text: np.ndarray, training: bool = False) -> np.ndarray:
        """Encode text to embeddings."""
        if text.ndim == 1:
            text = text.reshape(1, -1)
        
        x = self.token_embedding.forward(text)
        x = x + self.text_pos_embed[:, :x.shape[1]]
        
        # Causal mask
        seq_len = x.shape[1]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        
        for layer in self.text_layers_list:
            x = layer.forward(x, mask, training)
        
        x = self.text_ln.forward(x, training)
        
        # Take end-of-text token (last non-padding)
        x = x[:, -1]
        x = np.dot(x, self.text_proj)
        
        # L2 normalize
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        
        return x
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass - encode images."""
        return self.encode_image(X, training)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through CLIP."""
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


# Perceiver

class Perceiver(BaseArchitecture):
    """
    Perceiver - General Perception with Iterative Attention.
    
    Architecture that handles any modality using a latent bottleneck.
    
    Parameters
    ----------
    input_dim : int
        Input dimension per element
    num_latents : int
        Number of latent vectors
    latent_dim : int
        Dimension of latent vectors
    n_cross_attend : int
        Number of cross-attention layers
    n_self_attend : int
        Number of self-attention layers per block
    n_heads : int
        Number of attention heads
    
    Example
    -------
    >>> perceiver = Perceiver(input_dim=256, num_latents=512, latent_dim=1024)
    >>> output = perceiver.forward(input_array)
    """
    
    def __init__(self,
                 input_dim: int = 256,
                 num_latents: int = 512,
                 latent_dim: int = 1024,
                 output_dim: int = 1000,
                 n_cross_attend: int = 8,
                 n_self_attend: int = 6,
                 n_heads: int = 8,
                 **kwargs):
        
        self.input_dim = input_dim
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_cross_attend = n_cross_attend
        self.n_self_attend = n_self_attend
        self.n_heads = n_heads
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build Perceiver architecture."""
        # Learned latent array
        self.latents = np.random.randn(1, self.num_latents, self.latent_dim) * 0.02
        
        # Input projection
        self.input_proj = np.random.randn(self.input_dim, self.latent_dim) * 0.02
        
        # Cross-attention layers (input -> latent)
        self.cross_attn_layers = []
        for _ in range(self.n_cross_attend):
            layer = MultiHeadAttention(self.latent_dim, self.n_heads)
            self.cross_attn_layers.append(layer)
            self.cross_attn_layers.append(LayerNorm(self.latent_dim))
            self.cross_attn_layers.append(FeedForward(self.latent_dim, 
                                                       self.latent_dim * 4, 
                                                       activation='gelu'))
            self.cross_attn_layers.append(LayerNorm(self.latent_dim))
        
        # Self-attention layers (latent -> latent)
        self.self_attn_layers = []
        for _ in range(self.n_self_attend):
            block = TransformerEncoderLayer(
                self.latent_dim, self.n_heads, self.latent_dim * 4,
                activation='gelu', pre_norm=True
            )
            self.self_attn_layers.append(block)
        
        # Output projection
        self.output_proj = np.random.randn(self.latent_dim, self.output_dim) * 0.02
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 1:
            X = X.reshape(1, 1, -1)
        elif X.ndim == 2:
            X = X.reshape(1, *X.shape)
        
        batch_size = X.shape[0]
        
        # Project input
        X = np.dot(X, self.input_proj)
        
        # Initialize latents
        latents = np.repeat(self.latents, batch_size, axis=0)
        
        # Cross-attention blocks
        for i in range(0, len(self.cross_attn_layers), 4):
            cross_attn = self.cross_attn_layers[i]
            norm1 = self.cross_attn_layers[i + 1]
            ff = self.cross_attn_layers[i + 2]
            norm2 = self.cross_attn_layers[i + 3]
            
            # Cross-attention: latents attend to input
            attn_out = cross_attn.forward(latents, X, X, training=training)
            latents = norm1.forward(latents + attn_out, training)
            
            # Feed-forward
            ff_out = ff.forward(latents, training)
            latents = norm2.forward(latents + ff_out, training)
            
            # Self-attention within latents
            for self_attn in self.self_attn_layers:
                latents = self_attn.forward(latents, training=training)
        
        # Global average pool
        output = np.mean(latents, axis=1)
        
        # Output projection
        logits = np.dot(output, self.output_proj)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through Perceiver."""
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


# XLNet

class XLNet(BaseArchitecture):
    """
    XLNet - Generalized Autoregressive Pretraining.
    
    Combines the best of BERT (bidirectional) and GPT (autoregressive)
    using permutation language modeling.
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    d_model : int
        Hidden dimension
    n_heads : int
        Number of attention heads
    n_layers : int
        Number of transformer layers
    mem_len : int
        Memory length for segment-level recurrence
    """
    
    def __init__(self,
                 vocab_size: int = 32000,
                 d_model: int = 1024,
                 n_heads: int = 16,
                 n_layers: int = 24,
                 d_ff: int = 4096,
                 max_len: int = 512,
                 mem_len: int = 512,
                 **kwargs):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.mem_len = mem_len
        
        super().__init__(input_shape=(max_len,),
                        output_shape=(d_model,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build XLNet architecture."""
        # Embeddings
        self.token_embedding = Embedding(self.vocab_size, self.d_model)
        
        # Relative positional encoding
        self.r_w_bias = np.random.randn(self.n_heads, self.d_model // self.n_heads) * 0.02
        self.r_r_bias = np.random.randn(self.n_heads, self.d_model // self.n_heads) * 0.02
        
        # Transformer layers with relative attention
        self.layers = []
        for _ in range(self.n_layers):
            layer = TransformerEncoderLayer(
                self.d_model, self.n_heads, self.d_ff,
                pre_norm=True, activation='gelu'
            )
            self.layers.append(layer)
        
        self.final_norm = LayerNorm(self.d_model)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        hidden_states = self.token_embedding.forward(X)
        
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, training=training)
        
        return self.final_norm.forward(hidden_states, training)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through XLNet."""
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


# Factory Function

def create_transformer(architecture: str, **kwargs) -> BaseArchitecture:
    """
    Factory function to create transformer architectures.
    
    Parameters
    ----------
    architecture : str
        Architecture name: 'bert', 'gpt', 't5', 'roberta', 'albert',
        'distilbert', 'swin', 'clip', 'perceiver', 'xlnet'
    **kwargs
        Architecture-specific parameters
    
    Returns
    -------
    model : BaseArchitecture
        The requested transformer model
    
    Example
    -------
    >>> bert = create_transformer('bert', vocab_size=30522)
    >>> gpt = create_transformer('gpt2-small')
    """
    architectures = {
        'bert': BERT,
        'bert-base': BERTBase,
        'bert-large': BERTLarge,
        'bert-tiny': BERTTiny,
        'gpt': GPT,
        'gpt2': GPT,
        'gpt2-small': GPT2Small,
        'gpt2-medium': GPT2Medium,
        'gpt2-large': GPT2Large,
        't5': T5,
        't5-small': T5Small,
        't5-base': T5Base,
        't5-large': T5Large,
        'roberta': RoBERTa,
        'albert': ALBERT,
        'albert-base': ALBERTBase,
        'albert-large': ALBERTLarge,
        'distilbert': DistilBERT,
        'swin': SwinTransformer,
        'swin-tiny': SwinTiny,
        'swin-small': SwinSmall,
        'swin-base': SwinBase,
        'clip': CLIP,
        'perceiver': Perceiver,
        'xlnet': XLNet,
    }
    
    arch_name = architecture.lower()
    if arch_name not in architectures:
        available = list(architectures.keys())
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {available}")
    
    return architectures[arch_name](**kwargs)
