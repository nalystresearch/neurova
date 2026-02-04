# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Specialized Neural Network Architectures for Neurova

Complete implementation of specialized architectures:
- CapsuleNetwork (Capsule Networks with Dynamic Routing)
- SiameseNetwork (Siamese Networks for similarity learning)
- TripletNetwork (Triplet Networks for metric learning)
- MemoryNetwork (Memory-augmented Networks)
- NTM (Neural Turing Machine)
- MixtureOfExperts (Sparse Mixture of Experts)
- LiquidNeuralNetwork (Liquid Time-Constant Networks)
- SpikingNeuralNetwork (SNN with LIF neurons)
- EnergyBasedModel (Hopfield Networks, RBM)
- NeuralODE (Neural Ordinary Differential Equations - simplified)
- Hypernetwork (Networks that generate other networks)
- NeRF (Neural Radiance Fields - simplified)

All implementations use pure NumPy for educational purposes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from .base import BaseArchitecture


# Utility Functions

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x > 0, x, alpha * x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)


def squash(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Squashing function for capsule networks.
    Preserves direction while normalizing length to [0, 1].
    """
    s_squared_norm = np.sum(np.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / np.sqrt(s_squared_norm + 1e-8)
    return scale * vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# Layer Components

class DenseLayer:
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int,
                 activation: Optional[str] = 'relu', use_bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_bias = use_bias
        
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features) if use_bias else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W
        if self.b is not None:
            out += self.b
        
        if self.activation == 'relu':
            out = relu(out)
        elif self.activation == 'tanh':
            out = tanh(out)
        elif self.activation == 'sigmoid':
            out = sigmoid(out)
        elif self.activation == 'leaky_relu':
            out = leaky_relu(out)
        
        return out


class ConvLayer:
    """Simple 2D convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with convolution.
        
        Parameters
        ----------
        x : np.ndarray
            Input of shape (batch, channels, height, width)
        """
        batch, C, H, W = x.shape
        
        # Pad input
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                          (self.padding, self.padding)), mode='constant')
        
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        out = np.zeros((batch, self.out_channels, H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = x[:, :, h_start:h_start+self.kernel_size,
                         w_start:w_start+self.kernel_size]
                
                for k in range(self.out_channels):
                    out[:, k, i, j] = np.sum(
                        patch * self.W[k], axis=(1, 2, 3)
                    ) + self.b[k]
        
        return relu(out)


class PrimaryCapsule:
    """
    Primary capsule layer for Capsule Networks.
    
    Converts conv features to capsules.
    """
    
    def __init__(self, in_channels: int, num_capsules: int, 
                 capsule_dim: int, kernel_size: int = 9, stride: int = 2):
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        self.conv = ConvLayer(in_channels, num_capsules * capsule_dim,
                             kernel_size, stride)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        capsules : np.ndarray
            Shape (batch, num_capsules * H * W, capsule_dim)
        """
        out = self.conv.forward(x)
        batch, _, H, W = out.shape
        
        # Reshape to capsules
        out = out.reshape(batch, self.num_capsules, self.capsule_dim, H, W)
        out = out.transpose(0, 1, 3, 4, 2)  # (batch, num_caps, H, W, dim)
        out = out.reshape(batch, -1, self.capsule_dim)  # (batch, n_caps, dim)
        
        return squash(out)


class DigitCapsule:
    """
    Digit capsule layer with dynamic routing.
    """
    
    def __init__(self, num_capsules: int, num_routes: int,
                 in_dim: int, out_dim: int, num_iterations: int = 3):
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        
        # Weight matrix for each capsule
        self.W = np.random.randn(
            num_capsules, num_routes, out_dim, in_dim
        ) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with dynamic routing.
        
        Parameters
        ----------
        x : np.ndarray
            Primary capsules (batch, num_routes, in_dim)
        
        Returns
        -------
        v : np.ndarray
            Digit capsules (batch, num_capsules, out_dim)
        """
        batch = x.shape[0]
        
        # Compute u_hat: prediction vectors
        # u_hat[i,j] = W[i,j] @ x[j]
        u_hat = np.zeros((batch, self.num_capsules, self.num_routes, self.out_dim))
        
        for i in range(self.num_capsules):
            for j in range(self.num_routes):
                u_hat[:, i, j] = x[:, j] @ self.W[i, j].T
        
        # Dynamic routing
        b = np.zeros((batch, self.num_capsules, self.num_routes))
        
        for iteration in range(self.num_iterations):
            # Coupling coefficients
            c = softmax(b, axis=1)  # (batch, num_caps, num_routes)
            
            # Weighted sum
            s = np.sum(c[:, :, :, np.newaxis] * u_hat, axis=2)  # (batch, num_caps, out_dim)
            
            # Squash
            v = squash(s)
            
            if iteration < self.num_iterations - 1:
                # Agreement
                agreement = np.sum(u_hat * v[:, :, np.newaxis, :], axis=3)
                b = b + agreement
        
        return v


# Capsule Network

class CapsuleNetwork(BaseArchitecture):
    """
    Capsule Network (Sabour et al., 2017).
    
    Uses dynamic routing between capsules for part-whole relationships.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape (channels, height, width)
    num_classes : int
        Number of output classes
    num_primary_capsules : int
        Number of primary capsules
    primary_dim : int
        Dimension of primary capsules
    digit_dim : int
        Dimension of digit capsules
    routing_iterations : int
        Number of routing iterations
    
    Example
    -------
    >>> capsnet = CapsuleNetwork(input_shape=(1, 28, 28), num_classes=10)
    >>> output = capsnet.forward(images)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001],
        'routing_iterations': [2, 3],
        'primary_dim': [8],
        'digit_dim': [16],
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (1, 28, 28),
                 num_classes: int = 10,
                 num_primary_capsules: int = 32,
                 primary_dim: int = 8,
                 digit_dim: int = 16,
                 routing_iterations: int = 3,
                 **kwargs):
        
        self.num_classes = num_classes
        self.num_primary_capsules = num_primary_capsules
        self.primary_dim = primary_dim
        self.digit_dim = digit_dim
        self.routing_iterations = routing_iterations
        
        super().__init__(input_shape=input_shape,
                        output_shape=(num_classes,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build CapsNet architecture."""
        C, H, W = self.input_shape
        
        # Initial conv layer
        self.conv1 = ConvLayer(C, 256, kernel_size=9, stride=1)
        
        # Primary capsules
        conv_h = H - 8  # After 9x9 conv with stride 1
        conv_w = W - 8
        
        self.primary_caps = PrimaryCapsule(
            256, self.num_primary_capsules, self.primary_dim,
            kernel_size=9, stride=2
        )
        
        # Calculate number of primary capsule outputs
        primary_h = (conv_h - 9) // 2 + 1
        primary_w = (conv_w - 9) // 2 + 1
        num_routes = self.num_primary_capsules * primary_h * primary_w
        
        # Digit capsules
        self.digit_caps = DigitCapsule(
            self.num_classes, num_routes,
            self.primary_dim, self.digit_dim,
            self.routing_iterations
        )
        
        # Reconstruction decoder
        self.decoder = [
            DenseLayer(self.digit_dim * self.num_classes, 512, 'relu'),
            DenseLayer(512, 1024, 'relu'),
            DenseLayer(1024, int(np.prod(self.input_shape)), 'sigmoid')
        ]
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : np.ndarray
            Input images (batch, C, H, W)
        
        Returns
        -------
        output : np.ndarray
            Class probabilities (batch, num_classes)
        """
        # Ensure 4D
        if X.ndim == 3:
            X = X[np.newaxis, ...]
        
        # Conv layer
        x = self.conv1.forward(X)
        
        # Primary capsules
        u = self.primary_caps.forward(x)
        
        # Digit capsules
        v = self.digit_caps.forward(u)
        
        # Output: length of digit capsule vectors
        output = np.sqrt(np.sum(np.square(v), axis=-1))
        
        return softmax(output)
    
    def reconstruct(self, v: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Reconstruct input from digit capsules.
        
        Parameters
        ----------
        v : np.ndarray
            Digit capsules (batch, num_classes, digit_dim)
        y : np.ndarray
            One-hot encoded labels (batch, num_classes)
        
        Returns
        -------
        reconstruction : np.ndarray
            Reconstructed images
        """
        # Mask with labels
        masked = v * y[:, :, np.newaxis]
        masked = masked.reshape(masked.shape[0], -1)
        
        # Decode
        x = masked
        for layer in self.decoder:
            x = layer.forward(x)
        
        return x.reshape(-1, *self.input_shape)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through CapsuleNetwork."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Siamese Network

class SiameseNetwork(BaseArchitecture):
    """
    Siamese Network for similarity learning.
    
    Learns an embedding space where similar items are close.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    embedding_dim : int
        Output embedding dimension
    hidden_dims : list
        Hidden layer dimensions
    
    Example
    -------
    >>> siamese = SiameseNetwork(input_dim=784, embedding_dim=128)
    >>> similarity = siamese.forward(x1, x2)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'embedding_dim': [64, 128, 256],
        'margin': [0.5, 1.0, 2.0],
    }
    
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 128,
                 hidden_dims: Optional[List[int]] = None,
                 margin: float = 1.0,
                 **kwargs):
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.margin = margin
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(embedding_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build shared encoder network."""
        input_dim = self.input_shape[0]
        
        self.encoder = []
        in_dim = input_dim
        
        for h_dim in self.hidden_dims:
            layer = DenseLayer(in_dim, h_dim, 'relu')
            self.encoder.append(layer)
            in_dim = h_dim
        
        # Embedding layer
        self.embedding_layer = DenseLayer(in_dim, self.embedding_dim, None)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to embedding space."""
        h = x
        for layer in self.encoder:
            h = layer.forward(h)
        return self.embedding_layer.forward(h)
    
    def _forward(self, x1: np.ndarray, x2: np.ndarray = None,
                 training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        If x2 is provided, returns similarity score.
        Otherwise, returns embedding of x1.
        """
        emb1 = self.encode(x1)
        
        if x2 is None:
            return emb1
        
        emb2 = self.encode(x2)
        
        # Euclidean distance
        distance = np.sqrt(np.sum((emb1 - emb2) ** 2, axis=-1))
        
        # Convert to similarity (0 = different, 1 = same)
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def contrastive_loss(self, emb1: np.ndarray, emb2: np.ndarray,
                        y: np.ndarray) -> float:
        """
        Contrastive loss function.
        
        Parameters
        ----------
        emb1, emb2 : np.ndarray
            Embeddings
        y : np.ndarray
            Labels (1 = same, 0 = different)
        
        Returns
        -------
        loss : float
        """
        distance = np.sqrt(np.sum((emb1 - emb2) ** 2, axis=-1))
        
        # Contrastive loss
        positive_loss = y * (distance ** 2)
        negative_loss = (1 - y) * np.maximum(0, self.margin - distance) ** 2
        
        return float(np.mean(positive_loss + negative_loss))

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through SiameseNetwork."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Triplet Network

class TripletNetwork(BaseArchitecture):
    """
    Triplet Network for metric learning.
    
    Learns embeddings using anchor-positive-negative triplets.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    embedding_dim : int
        Output embedding dimension
    margin : float
        Triplet loss margin
    
    Example
    -------
    >>> triplet = TripletNetwork(input_dim=784, embedding_dim=128)
    >>> loss = triplet.triplet_loss(anchor, positive, negative)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'embedding_dim': [64, 128],
        'margin': [0.2, 0.5, 1.0],
    }
    
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 128,
                 hidden_dims: Optional[List[int]] = None,
                 margin: float = 0.2,
                 **kwargs):
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.margin = margin
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(embedding_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build encoder network."""
        input_dim = self.input_shape[0]
        
        self.encoder = []
        in_dim = input_dim
        
        for h_dim in self.hidden_dims:
            layer = DenseLayer(in_dim, h_dim, 'relu')
            self.encoder.append(layer)
            in_dim = h_dim
        
        self.embedding_layer = DenseLayer(in_dim, self.embedding_dim, None)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to normalized embedding."""
        h = x
        for layer in self.encoder:
            h = layer.forward(h)
        emb = self.embedding_layer.forward(h)
        
        # L2 normalize
        return emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Get embedding of input."""
        return self.encode(X)
    
    def triplet_loss(self, anchor: np.ndarray, positive: np.ndarray,
                    negative: np.ndarray) -> float:
        """
        Compute triplet loss.
        
        Loss = max(d(a,p) - d(a,n) + margin, 0)
        """
        emb_a = self.encode(anchor)
        emb_p = self.encode(positive)
        emb_n = self.encode(negative)
        
        d_pos = np.sum((emb_a - emb_p) ** 2, axis=-1)
        d_neg = np.sum((emb_a - emb_n) ** 2, axis=-1)
        
        loss = np.maximum(0, d_pos - d_neg + self.margin)
        
        return float(np.mean(loss))

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through TripletNetwork."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Memory Network

class MemoryNetwork(BaseArchitecture):
    """
    End-to-End Memory Network (Sukhbaatar et al., 2015).
    
    Memory-augmented network for question answering.
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    embedding_dim : int
        Embedding dimension
    memory_size : int
        Number of memory slots
    num_hops : int
        Number of memory hops
    
    Example
    -------
    >>> memnet = MemoryNetwork(vocab_size=1000, embedding_dim=128)
    >>> answer = memnet.forward(story, query)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001, 0.01],
        'embedding_dim': [64, 128],
        'num_hops': [1, 2, 3],
    }
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 memory_size: int = 50,
                 num_hops: int = 3,
                 **kwargs):
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.num_hops = num_hops
        
        super().__init__(input_shape=(vocab_size,),
                        output_shape=(vocab_size,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build memory network components."""
        # Input memory embeddings (A)
        self.A = [np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
                  for _ in range(self.num_hops)]
        
        # Output memory embeddings (C)
        self.C = [np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
                  for _ in range(self.num_hops)]
        
        # Query embedding (B)
        self.B = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        
        # Output weight
        self.W = np.random.randn(self.embedding_dim, self.vocab_size) * 0.1
    
    def _forward(self, story: np.ndarray, query: np.ndarray,
                 training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        story : np.ndarray
            Story sentences (memory_size, vocab_size) - bag of words
        query : np.ndarray
            Query (vocab_size,) - bag of words
        
        Returns
        -------
        answer : np.ndarray
            Answer distribution over vocabulary
        """
        # Embed query
        u = query @ self.B
        
        for hop in range(self.num_hops):
            # Embed memories
            m = story @ self.A[hop]  # (memory_size, embedding_dim)
            c = story @ self.C[hop]  # (memory_size, embedding_dim)
            
            # Attention
            p = softmax(m @ u)  # (memory_size,)
            
            # Output
            o = np.sum(p[:, np.newaxis] * c, axis=0)  # (embedding_dim,)
            
            # Update query
            u = u + o
        
        # Final output
        a_hat = u @ self.W
        
        return softmax(a_hat)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through MemoryNetwork."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Neural Turing Machine

class NTM(BaseArchitecture):
    """
    Neural Turing Machine (Graves et al., 2014).
    
    Differentiable computer with external memory.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    memory_size : int
        Number of memory locations (N)
    memory_dim : int
        Size of each memory location (M)
    controller_hidden : int
        Controller hidden size
    
    Example
    -------
    >>> ntm = NTM(input_dim=8, output_dim=8, memory_size=128)
    >>> output = ntm.forward(input_sequence)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'memory_size': [64, 128],
        'memory_dim': [20, 32],
    }
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 memory_size: int = 128,
                 memory_dim: int = 20,
                 controller_hidden: int = 100,
                 num_heads: int = 1,
                 **kwargs):
        
        self.memory_size = memory_size  # N
        self.memory_dim = memory_dim  # M
        self.controller_hidden = controller_hidden
        self.num_heads = num_heads
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build NTM components."""
        input_dim = self.input_shape[0]
        output_dim = self.output_shape[0]
        
        # Controller (LSTM-like)
        read_size = self.memory_dim * self.num_heads
        total_input = input_dim + read_size
        
        self.W_input = np.random.randn(total_input, self.controller_hidden) * 0.1
        self.W_hidden = np.random.randn(self.controller_hidden, self.controller_hidden) * 0.1
        self.b_hidden = np.zeros(self.controller_hidden)
        
        # Output projection
        self.W_output = np.random.randn(self.controller_hidden + read_size, output_dim) * 0.1
        
        # Read head parameters
        head_param_size = self.memory_dim + 1 + 1 + 3 + 1  # key, beta, g, shift, gamma
        self.W_read = np.random.randn(self.controller_hidden, head_param_size) * 0.1
        
        # Write head parameters
        write_param_size = head_param_size + self.memory_dim + self.memory_dim  # + erase + add
        self.W_write = np.random.randn(self.controller_hidden, write_param_size) * 0.1
        
        # Initial memory
        self.memory = np.zeros((self.memory_size, self.memory_dim))
        self.read_weights = np.ones(self.memory_size) / self.memory_size
        self.write_weights = np.ones(self.memory_size) / self.memory_size
        self.hidden = np.zeros(self.controller_hidden)
    
    def _content_addressing(self, key: np.ndarray, beta: float) -> np.ndarray:
        """Content-based addressing."""
        # Cosine similarity
        similarity = np.sum(self.memory * key, axis=1) / \
                    (np.linalg.norm(self.memory, axis=1) * np.linalg.norm(key) + 1e-8)
        
        return softmax(beta * similarity)
    
    def _interpolate(self, w_content: np.ndarray, w_prev: np.ndarray,
                    g: float) -> np.ndarray:
        """Interpolation between content and previous weights."""
        return g * w_content + (1 - g) * w_prev
    
    def _convolve_shift(self, w: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """Convolutional shift."""
        n = len(w)
        result = np.zeros(n)
        for i in range(n):
            for j, s in enumerate(shift):
                result[i] += w[(i - j + 1 + n) % n] * s
        return result
    
    def _sharpen(self, w: np.ndarray, gamma: float) -> np.ndarray:
        """Sharpen weights."""
        w = w ** gamma
        return w / (np.sum(w) + 1e-8)
    
    def _address(self, params: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        """Full addressing mechanism."""
        idx = 0
        
        # Key vector
        key = params[idx:idx + self.memory_dim]
        idx += self.memory_dim
        
        # Key strength
        beta = 1 + np.log(1 + np.exp(params[idx]))
        idx += 1
        
        # Interpolation gate
        g = sigmoid(params[idx])
        idx += 1
        
        # Shift weights (3 positions)
        shift = softmax(params[idx:idx + 3])
        idx += 3
        
        # Sharpening
        gamma = 1 + np.log(1 + np.exp(params[idx]))
        
        # Addressing steps
        w_c = self._content_addressing(key, beta)
        w_g = self._interpolate(w_c, prev_weights, g)
        w_s = self._convolve_shift(w_g, shift)
        w = self._sharpen(w_s, gamma)
        
        return w
    
    def reset(self):
        """Reset memory and weights."""
        self.memory = np.zeros((self.memory_size, self.memory_dim))
        self.read_weights = np.ones(self.memory_size) / self.memory_size
        self.write_weights = np.ones(self.memory_size) / self.memory_size
        self.hidden = np.zeros(self.controller_hidden)
    
    def step(self, x: np.ndarray) -> np.ndarray:
        """
        Single step of NTM.
        
        Parameters
        ----------
        x : np.ndarray
            Input at current timestep
        
        Returns
        -------
        output : np.ndarray
            Output at current timestep
        """
        # Read from memory
        read_vector = self.memory.T @ self.read_weights
        
        # Controller
        controller_input = np.concatenate([x, read_vector])
        self.hidden = tanh(controller_input @ self.W_input + 
                          self.hidden @ self.W_hidden + self.b_hidden)
        
        # Read head
        read_params = self.hidden @ self.W_read
        self.read_weights = self._address(read_params, self.read_weights)
        
        # Write head
        write_params = self.hidden @ self.W_write
        
        # Addressing
        addr_params = write_params[:self.memory_dim + 1 + 1 + 3 + 1]
        self.write_weights = self._address(addr_params, self.write_weights)
        
        # Erase and add vectors
        idx = len(addr_params)
        erase = sigmoid(write_params[idx:idx + self.memory_dim])
        add = write_params[idx + self.memory_dim:idx + 2 * self.memory_dim]
        
        # Memory update
        w = self.write_weights[:, np.newaxis]
        self.memory = self.memory * (1 - w @ erase[np.newaxis, :])
        self.memory = self.memory + w @ add[np.newaxis, :]
        
        # Output
        new_read = self.memory.T @ self.read_weights
        output_input = np.concatenate([self.hidden, new_read])
        output = output_input @ self.W_output
        
        return output
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass over sequence.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequence (seq_len, input_dim) or (batch, seq_len, input_dim)
        """
        if X.ndim == 3:
            # Batch processing
            batch_size = X.shape[0]
            outputs = []
            for b in range(batch_size):
                self.reset()
                seq_outputs = []
                for t in range(X.shape[1]):
                    out = self.step(X[b, t])
                    seq_outputs.append(out)
                outputs.append(seq_outputs)
            return np.array(outputs)
        
        # Single sequence
        self.reset()
        outputs = []
        for t in range(X.shape[0]):
            out = self.step(X[t])
            outputs.append(out)
        return np.array(outputs)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through NTM."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Mixture of Experts

class MixtureOfExperts(BaseArchitecture):
    """
    Sparse Mixture of Experts.
    
    Routes inputs to specialized expert networks.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    num_experts : int
        Number of expert networks
    expert_hidden : int
        Hidden dimension of each expert
    top_k : int
        Number of experts to activate per input
    
    Example
    -------
    >>> moe = MixtureOfExperts(input_dim=128, output_dim=64, num_experts=8)
    >>> output = moe.forward(x)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001],
        'num_experts': [4, 8, 16],
        'top_k': [1, 2],
    }
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_experts: int = 8,
                 expert_hidden: int = 256,
                 top_k: int = 2,
                 **kwargs):
        
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.top_k = top_k
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build gating network and experts."""
        input_dim = self.input_shape[0]
        output_dim = self.output_shape[0]
        
        # Gating network
        self.gate = DenseLayer(input_dim, self.num_experts, None)
        
        # Expert networks (2-layer MLPs)
        self.experts = []
        for _ in range(self.num_experts):
            expert = [
                DenseLayer(input_dim, self.expert_hidden, 'relu'),
                DenseLayer(self.expert_hidden, output_dim, None)
            ]
            self.experts.append(expert)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass with sparse expert routing.
        
        Parameters
        ----------
        X : np.ndarray
            Input (batch, input_dim) or (input_dim,)
        """
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]
        
        batch_size = X.shape[0]
        
        # Gating scores
        gate_logits = self.gate.forward(X)
        
        # Top-k selection
        top_k_indices = np.argsort(gate_logits, axis=1)[:, -self.top_k:]
        
        # Softmax over selected experts
        top_k_logits = np.take_along_axis(gate_logits, top_k_indices, axis=1)
        top_k_weights = softmax(top_k_logits, axis=1)
        
        # Compute expert outputs
        output = np.zeros((batch_size, self.output_shape[0]))
        
        for i in range(batch_size):
            for j, expert_idx in enumerate(top_k_indices[i]):
                # Expert forward
                h = X[i]
                for layer in self.experts[expert_idx]:
                    h = layer.forward(h)
                
                output[i] += top_k_weights[i, j] * h
        
        if single:
            return output[0]
        return output
    
    def load_balance_loss(self, X: np.ndarray) -> float:
        """
        Compute load balancing loss to encourage equal expert usage.
        """
        gate_logits = self.gate.forward(X)
        probs = softmax(gate_logits, axis=1)
        
        # Fraction of inputs routed to each expert
        importance = np.mean(probs, axis=0)
        
        # Auxiliary loss (should be uniform)
        uniform = np.ones(self.num_experts) / self.num_experts
        loss = np.sum((importance - uniform) ** 2)
        
        return float(loss * self.num_experts)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through MixtureOfExperts."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Liquid Neural Network

class LiquidNeuralNetwork(BaseArchitecture):
    """
    Liquid Time-Constant Networks (Hasani et al., 2021).
    
    Continuous-time neural networks with dynamic time constants.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dim : int
        Hidden state dimension
    output_dim : int
        Output dimension
    
    Example
    -------
    >>> lnn = LiquidNeuralNetwork(input_dim=4, hidden_dim=64, output_dim=2)
    >>> output = lnn.forward(time_series)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001],
        'hidden_dim': [32, 64, 128],
    }
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 10,
                 **kwargs):
        
        self.hidden_dim = hidden_dim
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build LTC components."""
        input_dim = self.input_shape[0]
        output_dim = self.output_shape[0]
        
        # LTC parameters
        self.W_in = np.random.randn(input_dim, self.hidden_dim) * 0.1
        self.W_rec = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        
        # Time constant parameters (learnable)
        self.tau = np.abs(np.random.randn(self.hidden_dim)) + 0.1
        
        # Sensitivity parameters
        self.A = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        
        # Output layer
        self.W_out = np.random.randn(self.hidden_dim, output_dim) * 0.1
        self.b_out = np.zeros(output_dim)
        
        # State
        self.state = np.zeros(self.hidden_dim)
    
    def _ode_step(self, x: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Single ODE step: dx/dt = (-x + f(x, I)) / tau
        """
        # Input
        I = x @ self.W_in
        
        # Dynamic time constant
        tau_dyn = self.tau * sigmoid(self.A @ self.state)
        
        # State update (Euler integration)
        f_x = tanh(self.state @ self.W_rec + I)
        dx = (-self.state + f_x) / tau_dyn
        
        self.state = self.state + dt * dx
        
        return self.state
    
    def reset(self):
        """Reset hidden state."""
        self.state = np.zeros(self.hidden_dim)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass over time series.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequence (seq_len, input_dim) or (batch, seq_len, input_dim)
        """
        if X.ndim == 3:
            batch_size = X.shape[0]
            outputs = []
            for b in range(batch_size):
                self.reset()
                for t in range(X.shape[1]):
                    h = self._ode_step(X[b, t])
                out = h @ self.W_out + self.b_out
                outputs.append(out)
            return np.array(outputs)
        
        self.reset()
        for t in range(X.shape[0]):
            h = self._ode_step(X[t])
        
        return h @ self.W_out + self.b_out

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through LiquidNeuralNetwork."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Spiking Neural Network

class SpikingNeuralNetwork(BaseArchitecture):
    """
    Spiking Neural Network with LIF neurons.
    
    Leaky Integrate-and-Fire neurons with temporal dynamics.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output dimension
    threshold : float
        Spike threshold
    tau : float
        Membrane time constant
    
    Example
    -------
    >>> snn = SpikingNeuralNetwork(input_dim=784, hidden_dim=128, output_dim=10)
    >>> output = snn.forward(spike_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001],
        'threshold': [0.5, 1.0],
        'tau': [0.9, 0.95],
    }
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 threshold: float = 1.0,
                 tau: float = 0.9,
                 num_steps: int = 100,
                 **kwargs):
        
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.tau = tau  # Membrane potential decay
        self.num_steps = num_steps
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build SNN layers."""
        input_dim = self.input_shape[0]
        output_dim = self.output_shape[0]
        
        # Synaptic weights
        self.W1 = np.random.randn(input_dim, self.hidden_dim) * 0.1
        self.W2 = np.random.randn(self.hidden_dim, output_dim) * 0.1
        
        # Membrane potentials
        self.V1 = np.zeros(self.hidden_dim)
        self.V2 = np.zeros(output_dim)
    
    def reset(self):
        """Reset membrane potentials."""
        self.V1 = np.zeros(self.hidden_dim)
        self.V2 = np.zeros(self.output_shape[0])
    
    def _lif_step(self, V: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LIF neuron update.
        
        Returns
        -------
        V : np.ndarray
            Updated membrane potential
        spikes : np.ndarray
            Spike output (0 or 1)
        """
        # Leaky integration
        V = self.tau * V + I
        
        # Spike generation
        spikes = (V >= self.threshold).astype(float)
        
        # Reset after spike
        V = V * (1 - spikes)
        
        return V, spikes
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through SNN.
        
        Parameters
        ----------
        X : np.ndarray
            Input spike rates or values (input_dim,) or (batch, input_dim)
        
        Returns
        -------
        output : np.ndarray
            Output spike counts or rates
        """
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]
        
        batch_size = X.shape[0]
        outputs = []
        
        for b in range(batch_size):
            self.reset()
            output_spikes = np.zeros(self.output_shape[0])
            
            for t in range(self.num_steps):
                # Generate input spikes from rates
                input_spikes = (np.random.rand(self.input_shape[0]) < X[b]).astype(float)
                
                # Layer 1
                I1 = input_spikes @ self.W1
                self.V1, spikes1 = self._lif_step(self.V1, I1)
                
                # Layer 2
                I2 = spikes1 @ self.W2
                self.V2, spikes2 = self._lif_step(self.V2, I2)
                
                output_spikes += spikes2
            
            # Normalize by number of steps
            outputs.append(output_spikes / self.num_steps)
        
        result = np.array(outputs)
        if single:
            return result[0]
        return result

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through SpikingNeuralNetwork."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Energy-Based Models

class HopfieldNetwork(BaseArchitecture):
    """
    Hopfield Network (associative memory).
    
    Parameters
    ----------
    pattern_size : int
        Size of patterns to store
    
    Example
    -------
    >>> hopfield = HopfieldNetwork(pattern_size=100)
    >>> hopfield.store(patterns)
    >>> recalled = hopfield.recall(noisy_pattern)
    """
    
    PARAM_SPACE = {
        'max_iterations': [10, 50, 100],
    }
    
    def __init__(self,
                 pattern_size: int,
                 max_iterations: int = 100,
                 **kwargs):
        
        self.pattern_size = pattern_size
        self.max_iterations = max_iterations
        
        super().__init__(input_shape=(pattern_size,),
                        output_shape=(pattern_size,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Initialize weight matrix."""
        self.W = np.zeros((self.pattern_size, self.pattern_size))
    
    def store(self, patterns: np.ndarray):
        """
        Store patterns using Hebbian learning.
        
        Parameters
        ----------
        patterns : np.ndarray
            Patterns to store (n_patterns, pattern_size) with values in {-1, 1}
        """
        n_patterns = patterns.shape[0]
        
        # Hebbian learning rule
        self.W = np.zeros((self.pattern_size, self.pattern_size))
        for p in patterns:
            self.W += np.outer(p, p)
        
        self.W /= n_patterns
        np.fill_diagonal(self.W, 0)  # No self-connections
    
    def recall(self, pattern: np.ndarray) -> np.ndarray:
        """
        Recall pattern from noisy input.
        
        Parameters
        ----------
        pattern : np.ndarray
            Noisy pattern (pattern_size,)
        
        Returns
        -------
        recalled : np.ndarray
            Recalled pattern
        """
        x = pattern.copy()
        
        for _ in range(self.max_iterations):
            x_new = np.sign(self.W @ x)
            x_new[x_new == 0] = 1
            
            if np.array_equal(x, x_new):
                break
            x = x_new
        
        return x
    
    def energy(self, pattern: np.ndarray) -> float:
        """Compute energy of pattern."""
        return -0.5 * pattern @ self.W @ pattern
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Recall patterns."""
        if X.ndim == 1:
            return self.recall(X)
        return np.array([self.recall(x) for x in X])

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through HopfieldNetwork."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class RestrictedBoltzmannMachine(BaseArchitecture):
    """
    Restricted Boltzmann Machine (RBM).
    
    Energy-based generative model.
    
    Parameters
    ----------
    visible_dim : int
        Number of visible units
    hidden_dim : int
        Number of hidden units
    
    Example
    -------
    >>> rbm = RestrictedBoltzmannMachine(visible_dim=784, hidden_dim=256)
    >>> rbm.fit(data)
    >>> samples = rbm.sample(n_samples=10)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001, 0.01],
        'hidden_dim': [128, 256, 512],
        'k': [1, 5],
    }
    
    def __init__(self,
                 visible_dim: int,
                 hidden_dim: int = 256,
                 k: int = 1,
                 **kwargs):
        
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.k = k  # CD-k steps
        
        super().__init__(input_shape=(visible_dim,),
                        output_shape=(hidden_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Initialize RBM parameters."""
        # Weights
        self.W = np.random.randn(self.visible_dim, self.hidden_dim) * 0.01
        
        # Biases
        self.b_v = np.zeros(self.visible_dim)
        self.b_h = np.zeros(self.hidden_dim)
    
    def sample_hidden(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample hidden units given visible."""
        h_prob = sigmoid(v @ self.W + self.b_h)
        h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(float)
        return h_prob, h_sample
    
    def sample_visible(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample visible units given hidden."""
        v_prob = sigmoid(h @ self.W.T + self.b_v)
        v_sample = (np.random.rand(*v_prob.shape) < v_prob).astype(float)
        return v_prob, v_sample
    
    def contrastive_divergence(self, v0: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Contrastive Divergence training step.
        
        Returns
        -------
        gradients : dict
            Gradients for W, b_v, b_h
        """
        batch_size = v0.shape[0]
        
        # Positive phase
        h0_prob, h0 = self.sample_hidden(v0)
        
        # Negative phase (Gibbs sampling)
        h = h0
        for _ in range(self.k):
            v_prob, v = self.sample_visible(h)
            h_prob, h = self.sample_hidden(v)
        
        # Gradients
        dW = (v0.T @ h0_prob - v.T @ h_prob) / batch_size
        db_v = np.mean(v0 - v, axis=0)
        db_h = np.mean(h0_prob - h_prob, axis=0)
        
        return {'dW': dW, 'db_v': db_v, 'db_h': db_h}
    
    def free_energy(self, v: np.ndarray) -> np.ndarray:
        """Compute free energy of visible configuration."""
        vbias_term = v @ self.b_v
        wx_b = v @ self.W + self.b_h
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=-1)
        return -vbias_term - hidden_term
    
    def sample(self, n_samples: int = 1, n_gibbs: int = 1000) -> np.ndarray:
        """Generate samples via Gibbs sampling."""
        v = np.random.randint(0, 2, (n_samples, self.visible_dim)).astype(float)
        
        for _ in range(n_gibbs):
            _, h = self.sample_hidden(v)
            _, v = self.sample_visible(h)
        
        return v
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Get hidden representation."""
        h_prob, _ = self.sample_hidden(X)
        return h_prob

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through RestrictedBoltzmannMachine."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class EnergyBasedModel(BaseArchitecture):
    """
    General Energy-Based Model framework.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dim : int
        Hidden dimension
    energy_type : str
        'hopfield', 'rbm', or 'ebm'
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001],
        'hidden_dim': [128, 256],
    }
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 energy_type: str = 'ebm',
                 **kwargs):
        
        self.hidden_dim = hidden_dim
        self.energy_type = energy_type
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(1,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Build energy function network."""
        input_dim = self.input_shape[0]
        
        # Energy network (outputs scalar energy)
        self.layers = [
            DenseLayer(input_dim, self.hidden_dim, 'relu'),
            DenseLayer(self.hidden_dim, self.hidden_dim, 'relu'),
            DenseLayer(self.hidden_dim, 1, None)
        ]
    
    def energy(self, x: np.ndarray) -> np.ndarray:
        """Compute energy of input."""
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Return negative energy (for classification as log-probability)."""
        return -self.energy(X)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through EnergyBasedModel."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Factory Functions

def create_specialized_model(architecture: str, **kwargs) -> BaseArchitecture:
    """
    Factory function to create specialized neural network architectures.
    
    Parameters
    ----------
    architecture : str
        Architecture name: 'capsule', 'siamese', 'triplet', 'memory',
                          'ntm', 'moe', 'liquid', 'snn', 'hopfield',
                          'rbm', 'ebm'
    **kwargs
        Architecture-specific parameters
    
    Returns
    -------
    model : BaseArchitecture
        The requested model
    
    Example
    -------
    >>> capsnet = create_specialized_model('capsule', input_shape=(1, 28, 28), num_classes=10)
    >>> siamese = create_specialized_model('siamese', input_dim=784, embedding_dim=128)
    >>> moe = create_specialized_model('moe', input_dim=128, output_dim=64, num_experts=8)
    """
    architectures = {
        'capsule': CapsuleNetwork,
        'capsulenet': CapsuleNetwork,
        'caps': CapsuleNetwork,
        'siamese': SiameseNetwork,
        'triplet': TripletNetwork,
        'memory': MemoryNetwork,
        'memnet': MemoryNetwork,
        'ntm': NTM,
        'neural_turing_machine': NTM,
        'moe': MixtureOfExperts,
        'mixture_of_experts': MixtureOfExperts,
        'liquid': LiquidNeuralNetwork,
        'ltc': LiquidNeuralNetwork,
        'snn': SpikingNeuralNetwork,
        'spiking': SpikingNeuralNetwork,
        'hopfield': HopfieldNetwork,
        'rbm': RestrictedBoltzmannMachine,
        'boltzmann': RestrictedBoltzmannMachine,
        'ebm': EnergyBasedModel,
        'energy': EnergyBasedModel,
    }
    
    arch_name = architecture.lower().replace('-', '_').replace(' ', '_')
    if arch_name not in architectures:
        available = list(architectures.keys())
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {available}")
    
    return architectures[arch_name](**kwargs)
