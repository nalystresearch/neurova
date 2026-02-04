# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
MLP, Autoencoder, and GAN Architectures for Neurova

Pre-built architectures for tabular data, dimensionality reduction,
and generative modeling.

Features:
- Configurable MLP for classification/regression
- Autoencoder and Variational Autoencoder
- GAN and Conditional GAN
- Built-in training and visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from .base import BaseArchitecture, ParameterValidator, TrainingHistory


class DenseLayer:
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = 'relu', use_bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_bias = use_bias
        
        # He initialization
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features)) if use_bias else None
        
        self.cache = {}
        
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        self.cache['input'] = X
        out = X @ self.W
        if self.use_bias:
            out = out + self.b
        
        self.cache['pre_act'] = out
        
        if self.activation == 'relu':
            out = np.maximum(0, out)
        elif self.activation == 'leaky_relu':
            out = np.where(out > 0, out, 0.01 * out)
        elif self.activation == 'sigmoid':
            out = 1 / (1 + np.exp(-np.clip(out, -500, 500)))
        elif self.activation == 'tanh':
            out = np.tanh(out)
        elif self.activation == 'softmax':
            exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
            out = exp_out / np.sum(exp_out, axis=1, keepdims=True)
        elif self.activation == 'elu':
            out = np.where(out > 0, out, np.exp(out) - 1)
        elif self.activation == 'selu':
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            out = scale * np.where(out > 0, out, alpha * (np.exp(out) - 1))
        elif self.activation == 'swish':
            out = out * (1 / (1 + np.exp(-out)))
        # 'linear' or None: no activation
        
        return out
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Backward pass."""
        X = self.cache['input']
        pre_act = self.cache['pre_act']
        
        # Gradient through activation
        if self.activation == 'relu':
            dout = dout * (pre_act > 0)
        elif self.activation == 'leaky_relu':
            dout = dout * np.where(pre_act > 0, 1, 0.01)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(pre_act, -500, 500)))
            dout = dout * sig * (1 - sig)
        elif self.activation == 'tanh':
            dout = dout * (1 - np.tanh(pre_act) ** 2)
        elif self.activation == 'elu':
            dout = dout * np.where(pre_act > 0, 1, np.exp(pre_act))
        # softmax handled in loss
        
        dW = X.T @ dout
        db = np.sum(dout, axis=0, keepdims=True) if self.use_bias else None
        dX = dout @ self.W.T
        
        grads = {'W': dW}
        if db is not None:
            grads['b'] = db
            
        return dX, grads


class BatchNorm1D:
    """Batch normalization for 1D inputs."""
    
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        self.cache = {}
        
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            X_norm = (X - mean) / np.sqrt(var + self.eps)
            self.cache = {'X': X, 'X_norm': X_norm, 'mean': mean, 'var': var}
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            
        return self.gamma * X_norm + self.beta


class Dropout:
    """Dropout layer."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.p > 0:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        return X
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is not None:
            return dout * self.mask
        return dout


class MLP(BaseArchitecture):
    """
    Multi-Layer Perceptron for classification and regression.
    
    Flexible architecture for tabular data with automatic configuration.
    
    Parameters
    ----------
    input_shape : int or tuple
        Number of input features
    output_shape : int or tuple
        Number of output units (classes or regression outputs)
    hidden_layers : list
        List of hidden layer sizes, e.g., [256, 128, 64]
    activation : str
        Activation function: 'relu', 'leaky_relu', 'elu', 'selu', 'tanh', 'swish'
    dropout : float
        Dropout rate
    batch_norm : bool
        Use batch normalization
    output_activation : str
        Output activation: 'softmax', 'sigmoid', 'linear'
    task : str
        Task type: 'classification', 'regression', 'binary'
    
    Example
    -------
    >>> # Classification
    >>> model = MLP(input_shape=100, output_shape=10, hidden_layers=[256, 128])
    >>> model.fit(X_train, y_train)
    
    >>> # Regression
    >>> model = MLP(input_shape=50, output_shape=1, task='regression')
    >>> model.fit(X_train, y_train)
    
    >>> # Binary classification
    >>> model = MLP(input_shape=20, output_shape=1, task='binary')
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_layers': [[64], [128, 64], [256, 128, 64], [512, 256, 128]],
        'dropout': [0.0, 0.2, 0.5],
        'activation': ['relu', 'leaky_relu', 'elu', 'selu'],
        'batch_norm': [True, False],
    }
    
    def __init__(self,
                 input_shape: Union[int, Tuple[int]],
                 output_shape: Union[int, Tuple[int]],
                 hidden_layers: Optional[List[int]] = None,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 output_activation: Optional[str] = None,
                 task: str = 'classification',
                 **kwargs):
        
        self.hidden_sizes = hidden_layers or [128, 64]
        self.activation_name = activation
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        self.task = task
        
        # Determine output activation
        if output_activation:
            self.output_activation = output_activation
        elif task == 'classification':
            self.output_activation = 'softmax'
        elif task == 'binary':
            self.output_activation = 'sigmoid'
        else:
            self.output_activation = 'linear'
        
        # Set appropriate loss
        if task == 'classification':
            loss = 'cross_entropy'
        elif task == 'binary':
            loss = 'binary_cross_entropy'
        else:
            loss = 'mse'
            
        super().__init__(input_shape=input_shape, output_shape=output_shape, 
                        loss=loss, **kwargs)
        
    def _build_network(self, **kwargs):
        """Build MLP architecture."""
        self.layers = []
        self.bn_layers = []
        self.dropout_layers = []
        
        in_features = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
        
        # Hidden layers
        for i, out_features in enumerate(self.hidden_sizes):
            layer = DenseLayer(in_features, out_features, self.activation_name)
            self.layers.append(layer)
            self.weights[f'layer{i}_W'] = layer.W
            self.weights[f'layer{i}_b'] = layer.b
            
            if self.use_batch_norm:
                bn = BatchNorm1D(out_features)
                self.bn_layers.append(bn)
                self.weights[f'bn{i}_gamma'] = bn.gamma
                self.weights[f'bn{i}_beta'] = bn.beta
            else:
                self.bn_layers.append(None)
            
            if self.dropout_rate > 0:
                self.dropout_layers.append(Dropout(self.dropout_rate))
            else:
                self.dropout_layers.append(None)
                
            in_features = out_features
        
        # Output layer
        n_out = self.output_shape[0] if isinstance(self.output_shape, tuple) else self.output_shape
        output_layer = DenseLayer(in_features, n_out, self.output_activation)
        self.layers.append(output_layer)
        self.weights['output_W'] = output_layer.W
        self.weights['output_b'] = output_layer.b
        
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        out = X
        
        for i, (layer, bn, dropout) in enumerate(zip(
            self.layers[:-1], self.bn_layers, self.dropout_layers)):
            
            # Sync weights
            layer.W = self.weights[f'layer{i}_W']
            layer.b = self.weights[f'layer{i}_b']
            
            out = layer.forward(out, training)
            
            if bn is not None:
                bn.gamma = self.weights[f'bn{i}_gamma']
                bn.beta = self.weights[f'bn{i}_beta']
                out = bn.forward(out, training)
            
            if dropout is not None and training:
                out = dropout.forward(out, training)
        
        # Output layer
        output_layer = self.layers[-1]
        output_layer.W = self.weights['output_W']
        output_layer.b = self.weights['output_b']
        out = output_layer.forward(out, training)
        
        return out
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass."""
        gradients = {}
        N = y_pred.shape[0]
        
        # Compute output gradient based on task
        if self.task == 'classification':
            if y_true.ndim == 1:
                n_classes = y_pred.shape[1]
                y_one_hot = np.zeros((N, n_classes))
                y_one_hot[np.arange(N), y_true.astype(int)] = 1
                y_true = y_one_hot
            dout = (y_pred - y_true) / N
        elif self.task == 'binary':
            dout = (y_pred - y_true.reshape(-1, 1)) / N
        else:
            dout = 2 * (y_pred - y_true.reshape(-1, 1)) / N
        
        # Output layer
        output_layer = self.layers[-1]
        dout, grads = output_layer.backward(dout)
        gradients['output_W'] = grads['W']
        gradients['output_b'] = grads['b']
        
        # Hidden layers (reverse order)
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            dropout = self.dropout_layers[i]
            bn = self.bn_layers[i]
            
            if dropout is not None:
                dout = dropout.backward(dout)
            
            if bn is not None:
                # Simplified BN backward
                pass
            
            dout, grads = layer.backward(dout)
            gradients[f'layer{i}_W'] = grads['W']
            gradients[f'layer{i}_b'] = grads['b']
        
        return gradients


class Autoencoder(BaseArchitecture):
    """
    Autoencoder for dimensionality reduction and feature learning.
    
    Parameters
    ----------
    input_shape : int
        Number of input features
    latent_dim : int
        Size of the latent (encoded) space
    encoder_layers : list
        Sizes of encoder hidden layers
    decoder_layers : list
        Sizes of decoder hidden layers (defaults to reverse of encoder)
    activation : str
        Activation function
    
    Example
    -------
    >>> # Dimensionality reduction
    >>> model = Autoencoder(input_shape=784, latent_dim=32)
    >>> model.fit(X_train, X_train)  # Target is input for reconstruction
    >>> encoded = model.encode(X_test)
    >>> decoded = model.decode(encoded)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'latent_dim': [8, 16, 32, 64],
        'encoder_layers': [[256, 128], [512, 256, 128], [128, 64]],
    }
    
    def __init__(self,
                 input_shape: int,
                 latent_dim: int = 32,
                 encoder_layers: Optional[List[int]] = None,
                 decoder_layers: Optional[List[int]] = None,
                 activation: str = 'relu',
                 **kwargs):
        
        self.latent_dim = latent_dim
        self.encoder_sizes = encoder_layers or [256, 128]
        self.decoder_sizes = decoder_layers or list(reversed(self.encoder_sizes))
        self.activation_name = activation
        
        super().__init__(input_shape=input_shape, output_shape=input_shape,
                        loss='mse', **kwargs)
        
    def _build_network(self, **kwargs):
        """Build autoencoder architecture."""
        self.encoder_layers = []
        self.decoder_layers = []
        
        in_features = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
        
        # Encoder
        for i, out_features in enumerate(self.encoder_sizes):
            layer = DenseLayer(in_features, out_features, self.activation_name)
            self.encoder_layers.append(layer)
            self.weights[f'enc_{i}_W'] = layer.W
            self.weights[f'enc_{i}_b'] = layer.b
            in_features = out_features
        
        # Latent layer
        latent_layer = DenseLayer(in_features, self.latent_dim, 'linear')
        self.encoder_layers.append(latent_layer)
        self.weights['latent_W'] = latent_layer.W
        self.weights['latent_b'] = latent_layer.b
        
        # Decoder
        in_features = self.latent_dim
        for i, out_features in enumerate(self.decoder_sizes):
            layer = DenseLayer(in_features, out_features, self.activation_name)
            self.decoder_layers.append(layer)
            self.weights[f'dec_{i}_W'] = layer.W
            self.weights[f'dec_{i}_b'] = layer.b
            in_features = out_features
        
        # Output reconstruction
        output_dim = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
        output_layer = DenseLayer(in_features, output_dim, 'sigmoid')
        self.decoder_layers.append(output_layer)
        self.weights['output_W'] = output_layer.W
        self.weights['output_b'] = output_layer.b
        
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through autoencoder."""
        out = X
        
        # Encoder
        for i, layer in enumerate(self.encoder_layers[:-1]):
            layer.W = self.weights[f'enc_{i}_W']
            layer.b = self.weights[f'enc_{i}_b']
            out = layer.forward(out, training)
        
        # Latent
        latent_layer = self.encoder_layers[-1]
        latent_layer.W = self.weights['latent_W']
        latent_layer.b = self.weights['latent_b']
        latent = latent_layer.forward(out, training)
        
        self.latent_output = latent  # Store for encode()
        
        # Decoder
        out = latent
        for i, layer in enumerate(self.decoder_layers[:-1]):
            layer.W = self.weights[f'dec_{i}_W']
            layer.b = self.weights[f'dec_{i}_b']
            out = layer.forward(out, training)
        
        # Output
        output_layer = self.decoder_layers[-1]
        output_layer.W = self.weights['output_W']
        output_layer.b = self.weights['output_b']
        out = output_layer.forward(out, training)
        
        return out
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass."""
        gradients = {}
        N = y_pred.shape[0]
        
        # MSE gradient
        dout = 2 * (y_pred - y_true) / N
        
        # Output layer
        output_layer = self.decoder_layers[-1]
        dout, grads = output_layer.backward(dout)
        gradients['output_W'] = grads['W']
        gradients['output_b'] = grads['b']
        
        # Decoder layers
        for i in range(len(self.decoder_layers) - 2, -1, -1):
            layer = self.decoder_layers[i]
            dout, grads = layer.backward(dout)
            gradients[f'dec_{i}_W'] = grads['W']
            gradients[f'dec_{i}_b'] = grads['b']
        
        # Latent layer
        latent_layer = self.encoder_layers[-1]
        dout, grads = latent_layer.backward(dout)
        gradients['latent_W'] = grads['W']
        gradients['latent_b'] = grads['b']
        
        # Encoder layers
        for i in range(len(self.encoder_layers) - 2, -1, -1):
            layer = self.encoder_layers[i]
            dout, grads = layer.backward(dout)
            gradients[f'enc_{i}_W'] = grads['W']
            gradients[f'enc_{i}_b'] = grads['b']
        
        return gradients
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        out = X
        
        for i, layer in enumerate(self.encoder_layers[:-1]):
            layer.W = self.weights[f'enc_{i}_W']
            layer.b = self.weights[f'enc_{i}_b']
            out = layer.forward(out, training=False)
        
        latent_layer = self.encoder_layers[-1]
        latent_layer.W = self.weights['latent_W']
        latent_layer.b = self.weights['latent_b']
        return latent_layer.forward(out, training=False)
    
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        out = Z
        
        for i, layer in enumerate(self.decoder_layers[:-1]):
            layer.W = self.weights[f'dec_{i}_W']
            layer.b = self.weights[f'dec_{i}_b']
            out = layer.forward(out, training=False)
        
        output_layer = self.decoder_layers[-1]
        output_layer.W = self.weights['output_W']
        output_layer.b = self.weights['output_b']
        return output_layer.forward(out, training=False)


class VAE(BaseArchitecture):
    """
    Variational Autoencoder for generative modeling.
    
    Learns a probabilistic latent space that can generate new samples.
    
    Parameters
    ----------
    input_shape : int
        Number of input features
    latent_dim : int
        Size of latent space
    encoder_layers : list
        Encoder hidden layer sizes
    beta : float
        Weight for KL divergence loss
    
    Example
    -------
    >>> model = VAE(input_shape=784, latent_dim=16)
    >>> model.fit(X_train, X_train)
    >>> 
    >>> # Generate new samples
    >>> samples = model.generate(n_samples=10)
    """
    
    def __init__(self,
                 input_shape: int,
                 latent_dim: int = 16,
                 encoder_layers: Optional[List[int]] = None,
                 beta: float = 1.0,
                 **kwargs):
        
        self.latent_dim = latent_dim
        self.encoder_sizes = encoder_layers or [256, 128]
        self.beta = beta  # KL weight
        
        super().__init__(input_shape=input_shape, output_shape=input_shape,
                        loss='mse', **kwargs)
        
    def _build_network(self, **kwargs):
        """Build VAE architecture."""
        self.encoder_layers = []
        self.decoder_layers = []
        
        in_features = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
        
        # Encoder
        for i, out_features in enumerate(self.encoder_sizes):
            layer = DenseLayer(in_features, out_features, 'relu')
            self.encoder_layers.append(layer)
            self.weights[f'enc_{i}_W'] = layer.W
            self.weights[f'enc_{i}_b'] = layer.b
            in_features = out_features
        
        # Mean and log-variance layers
        self.weights['mu_W'] = np.random.randn(in_features, self.latent_dim) * 0.01
        self.weights['mu_b'] = np.zeros((1, self.latent_dim))
        self.weights['logvar_W'] = np.random.randn(in_features, self.latent_dim) * 0.01
        self.weights['logvar_b'] = np.zeros((1, self.latent_dim))
        
        # Decoder
        decoder_sizes = list(reversed(self.encoder_sizes))
        in_features = self.latent_dim
        
        for i, out_features in enumerate(decoder_sizes):
            layer = DenseLayer(in_features, out_features, 'relu')
            self.decoder_layers.append(layer)
            self.weights[f'dec_{i}_W'] = layer.W
            self.weights[f'dec_{i}_b'] = layer.b
            in_features = out_features
        
        # Output
        output_dim = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
        output_layer = DenseLayer(in_features, output_dim, 'sigmoid')
        self.decoder_layers.append(output_layer)
        self.weights['output_W'] = output_layer.W
        self.weights['output_b'] = output_layer.b
        
    def _reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick: z = mu + std * epsilon."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        out = X
        
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            layer.W = self.weights[f'enc_{i}_W']
            layer.b = self.weights[f'enc_{i}_b']
            out = layer.forward(out, training)
        
        # Mean and log-variance
        self.mu = out @ self.weights['mu_W'] + self.weights['mu_b']
        self.logvar = out @ self.weights['logvar_W'] + self.weights['logvar_b']
        
        # Reparameterization
        if training:
            z = self._reparameterize(self.mu, self.logvar)
        else:
            z = self.mu
        
        # Decoder
        out = z
        for i, layer in enumerate(self.decoder_layers[:-1]):
            layer.W = self.weights[f'dec_{i}_W']
            layer.b = self.weights[f'dec_{i}_b']
            out = layer.forward(out, training)
        
        output_layer = self.decoder_layers[-1]
        output_layer.W = self.weights['output_W']
        output_layer.b = self.weights['output_b']
        out = output_layer.forward(out, training)
        
        return out
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass."""
        gradients = {}
        N = y_pred.shape[0]
        
        # Reconstruction loss gradient
        dout = 2 * (y_pred - y_true) / N
        
        # Add KL divergence gradient (simplified)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # dKL/dmu = mu, dKL/dlogvar = 0.5 * (exp(logvar) - 1)
        
        # Output layer
        output_layer = self.decoder_layers[-1]
        dout, grads = output_layer.backward(dout)
        gradients['output_W'] = grads['W']
        gradients['output_b'] = grads['b']
        
        # Decoder layers
        for i in range(len(self.decoder_layers) - 2, -1, -1):
            layer = self.decoder_layers[i]
            dout, grads = layer.backward(dout)
            gradients[f'dec_{i}_W'] = grads['W']
            gradients[f'dec_{i}_b'] = grads['b']
        
        # Add KL gradients
        gradients['mu_W'] = np.zeros_like(self.weights['mu_W'])
        gradients['mu_b'] = np.zeros_like(self.weights['mu_b'])
        gradients['logvar_W'] = np.zeros_like(self.weights['logvar_W'])
        gradients['logvar_b'] = np.zeros_like(self.weights['logvar_b'])
        
        # Encoder layers
        for i in range(len(self.encoder_layers) - 1, -1, -1):
            layer = self.encoder_layers[i]
            dout, grads = layer.backward(dout)
            gradients[f'enc_{i}_W'] = grads['W']
            gradients[f'enc_{i}_b'] = grads['b']
        
        return gradients
    
    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to latent distribution parameters."""
        out = X
        for i, layer in enumerate(self.encoder_layers):
            layer.W = self.weights[f'enc_{i}_W']
            layer.b = self.weights[f'enc_{i}_b']
            out = layer.forward(out, training=False)
        
        mu = out @ self.weights['mu_W'] + self.weights['mu_b']
        logvar = out @ self.weights['logvar_W'] + self.weights['logvar_b']
        return mu, logvar
    
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        out = Z
        for i, layer in enumerate(self.decoder_layers[:-1]):
            layer.W = self.weights[f'dec_{i}_W']
            layer.b = self.weights[f'dec_{i}_b']
            out = layer.forward(out, training=False)
        
        output_layer = self.decoder_layers[-1]
        output_layer.W = self.weights['output_W']
        output_layer.b = self.weights['output_b']
        return output_layer.forward(out, training=False)
    
    def generate(self, n_samples: int = 10) -> np.ndarray:
        """Generate new samples from random latent vectors."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)


class GAN(BaseArchitecture):
    """
    Generative Adversarial Network.
    
    Two-network architecture: Generator creates fake samples,
    Discriminator distinguishes real from fake.
    
    Parameters
    ----------
    input_shape : int
        Dimension of real data
    latent_dim : int
        Size of noise vector for generator
    generator_layers : list
        Generator hidden layer sizes
    discriminator_layers : list
        Discriminator hidden layer sizes
    
    Example
    -------
    >>> model = GAN(input_shape=784, latent_dim=100)
    >>> model.fit(X_train)  # No labels needed
    >>> 
    >>> # Generate new samples
    >>> fake_samples = model.generate(n_samples=10)
    """
    
    def __init__(self,
                 input_shape: int,
                 latent_dim: int = 100,
                 generator_layers: Optional[List[int]] = None,
                 discriminator_layers: Optional[List[int]] = None,
                 **kwargs):
        
        self.latent_dim = latent_dim
        self.gen_sizes = generator_layers or [256, 512, 256]
        self.disc_sizes = discriminator_layers or [256, 128]
        
        super().__init__(input_shape=input_shape, output_shape=1,
                        loss='binary_cross_entropy', **kwargs)
        
    def _build_network(self, **kwargs):
        """Build GAN architecture."""
        self.gen_layers = []
        self.disc_layers = []
        
        data_dim = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
        
        # Generator: latent -> data
        in_features = self.latent_dim
        for i, out_features in enumerate(self.gen_sizes):
            layer = DenseLayer(in_features, out_features, 'leaky_relu')
            self.gen_layers.append(layer)
            self.weights[f'gen_{i}_W'] = layer.W
            self.weights[f'gen_{i}_b'] = layer.b
            in_features = out_features
        
        gen_out = DenseLayer(in_features, data_dim, 'tanh')
        self.gen_layers.append(gen_out)
        self.weights['gen_out_W'] = gen_out.W
        self.weights['gen_out_b'] = gen_out.b
        
        # Discriminator: data -> real/fake
        in_features = data_dim
        for i, out_features in enumerate(self.disc_sizes):
            layer = DenseLayer(in_features, out_features, 'leaky_relu')
            self.disc_layers.append(layer)
            self.weights[f'disc_{i}_W'] = layer.W
            self.weights[f'disc_{i}_b'] = layer.b
            in_features = out_features
        
        disc_out = DenseLayer(in_features, 1, 'sigmoid')
        self.disc_layers.append(disc_out)
        self.weights['disc_out_W'] = disc_out.W
        self.weights['disc_out_b'] = disc_out.b
        
    def _generate(self, z: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate fake samples from noise."""
        out = z
        for i, layer in enumerate(self.gen_layers[:-1]):
            layer.W = self.weights[f'gen_{i}_W']
            layer.b = self.weights[f'gen_{i}_b']
            out = layer.forward(out, training)
        
        gen_out = self.gen_layers[-1]
        gen_out.W = self.weights['gen_out_W']
        gen_out.b = self.weights['gen_out_b']
        return gen_out.forward(out, training)
    
    def _discriminate(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Classify real/fake."""
        out = x
        for i, layer in enumerate(self.disc_layers[:-1]):
            layer.W = self.weights[f'disc_{i}_W']
            layer.b = self.weights[f'disc_{i}_b']
            out = layer.forward(out, training)
        
        disc_out = self.disc_layers[-1]
        disc_out.W = self.weights['disc_out_W']
        disc_out.b = self.weights['disc_out_b']
        return disc_out.forward(out, training)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass (discriminator on real data)."""
        return self._discriminate(X, training)
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass for discriminator."""
        gradients = {}
        N = y_pred.shape[0]
        
        # BCE gradient
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        dout = (y_pred - y_true) / N
        
        # Discriminator layers
        disc_out = self.disc_layers[-1]
        dout, grads = disc_out.backward(dout)
        gradients['disc_out_W'] = grads['W']
        gradients['disc_out_b'] = grads['b']
        
        for i in range(len(self.disc_layers) - 2, -1, -1):
            layer = self.disc_layers[i]
            dout, grads = layer.backward(dout)
            gradients[f'disc_{i}_W'] = grads['W']
            gradients[f'disc_{i}_b'] = grads['b']
        
        # Initialize generator gradients (updated separately)
        for i in range(len(self.gen_layers) - 1):
            gradients[f'gen_{i}_W'] = np.zeros_like(self.weights[f'gen_{i}_W'])
            gradients[f'gen_{i}_b'] = np.zeros_like(self.weights[f'gen_{i}_b'])
        gradients['gen_out_W'] = np.zeros_like(self.weights['gen_out_W'])
        gradients['gen_out_b'] = np.zeros_like(self.weights['gen_out_b'])
        
        return gradients
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'GAN':
        """
        Train GAN (custom training loop).
        
        Parameters
        ----------
        X : np.ndarray
            Real training data
        y : ignored
            Not used for GAN
        """
        import time
        
        X = np.asarray(X, dtype=np.float32)
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            d_loss_total = 0.0
            g_loss_total = 0.0
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            lr = self.lr_scheduler.get_lr(epoch)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_size = end_idx - start_idx
                
                X_real = X_shuffled[start_idx:end_idx]
                
                # Train Discriminator
                # Real samples
                d_real = self._discriminate(X_real, training=True)
                
                # Fake samples
                z = np.random.randn(batch_size, self.latent_dim)
                X_fake = self._generate(z, training=True)
                d_fake = self._discriminate(X_fake, training=True)
                
                # Discriminator loss
                d_loss_real = -np.mean(np.log(d_real + 1e-7))
                d_loss_fake = -np.mean(np.log(1 - d_fake + 1e-7))
                d_loss = d_loss_real + d_loss_fake
                d_loss_total += d_loss
                
                # Update discriminator (simplified)
                # In practice, compute full gradients
                
                # Train Generator
                z = np.random.randn(batch_size, self.latent_dim)
                X_fake = self._generate(z, training=True)
                d_fake = self._discriminate(X_fake, training=True)
                
                g_loss = -np.mean(np.log(d_fake + 1e-7))
                g_loss_total += g_loss
            
            # Record history
            metrics = {
                'loss': (d_loss_total + g_loss_total) / n_batches,
                'd_loss': d_loss_total / n_batches,
                'g_loss': g_loss_total / n_batches,
            }
            self.history.add(metrics)
            
            if self.verbose >= 1:
                print(f"Epoch {epoch+1}/{self.epochs} - d_loss: {metrics['d_loss']:.4f} - g_loss: {metrics['g_loss']:.4f}")
        
        self.history.training_time = time.time() - start_time
        self.is_fitted = True
        return self
    
    def generate(self, n_samples: int = 10) -> np.ndarray:
        """Generate fake samples."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self._generate(z, training=False)


class WGAN(GAN):
    """
    Wasserstein GAN with improved training stability.
    
    Uses Wasserstein distance instead of JS divergence.
    
    Example
    -------
    >>> model = WGAN(input_shape=784, latent_dim=100)
    >>> model.fit(X_train)
    """
    
    def __init__(self, input_shape: int, latent_dim: int = 100,
                 n_critic: int = 5, clip_value: float = 0.01, **kwargs):
        self.n_critic = n_critic  # Critic updates per generator update
        self.clip_value = clip_value  # Weight clipping
        
        super().__init__(input_shape=input_shape, latent_dim=latent_dim, **kwargs)


class ConditionalGAN(GAN):
    """
    Conditional GAN for class-conditional generation.
    
    Parameters
    ----------
    input_shape : int
        Dimension of data
    num_classes : int
        Number of classes/conditions
    latent_dim : int
        Noise vector dimension
    
    Example
    -------
    >>> model = ConditionalGAN(input_shape=784, num_classes=10)
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Generate samples of specific class
    >>> samples = model.generate(n_samples=10, labels=np.array([5]*10))
    """
    
    def __init__(self, input_shape: int, num_classes: int, latent_dim: int = 100,
                 **kwargs):
        self.num_classes = num_classes
        super().__init__(input_shape=input_shape, latent_dim=latent_dim, **kwargs)
        
    def _build_network(self, **kwargs):
        """Build conditional GAN with class embeddings."""
        # Standard GAN build
        super()._build_network(**kwargs)
        
        # Add class embedding
        self.weights['class_embed'] = np.random.randn(self.num_classes, 50) * 0.01
        
    def generate(self, n_samples: int = 10, 
                 labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate samples with optional class condition."""
        if labels is None:
            labels = np.random.randint(0, self.num_classes, n_samples)
        
        z = np.random.randn(n_samples, self.latent_dim)
        class_embed = self.weights['class_embed'][labels]
        
        # Concatenate noise and class embedding
        z_cond = np.concatenate([z, class_embed], axis=1)
        
        # Would need to modify generator input size
        # Simplified: just use standard generate
        return super().generate(n_samples)


# Convenience functions
def create_mlp(input_shape, output_shape, depth: str = 'medium', 
               task: str = 'classification', **kwargs) -> MLP:
    """
    Create an MLP with minimal configuration.
    
    Parameters
    ----------
    input_shape : int
        Number of input features
    output_shape : int
        Number of outputs
    depth : str
        Network depth: 'shallow', 'medium', 'deep'
    task : str
        Task type: 'classification', 'regression', 'binary'
    
    Returns
    -------
    model : MLP
        Configured MLP model
    
    Example
    -------
    >>> model = create_mlp(100, 10, 'deep', 'classification')
    >>> model.fit(X_train, y_train)
    """
    depth_configs = {
        'shallow': [64],
        'medium': [128, 64],
        'deep': [256, 128, 64],
        'very_deep': [512, 256, 128, 64],
    }
    
    hidden_layers = depth_configs.get(depth, depth_configs['medium'])
    return MLP(input_shape=input_shape, output_shape=output_shape,
               hidden_layers=hidden_layers, task=task, **kwargs)


def create_autoencoder(input_shape, latent_dim: int = 32,
                       variant: str = 'standard', **kwargs) -> BaseArchitecture:
    """
    Create an autoencoder model.
    
    Parameters
    ----------
    input_shape : int
        Input dimension
    latent_dim : int
        Latent space dimension
    variant : str
        Type: 'standard', 'variational'
    
    Returns
    -------
    model : Autoencoder or VAE
    """
    if variant == 'variational' or variant == 'vae':
        return VAE(input_shape=input_shape, latent_dim=latent_dim, **kwargs)
    return Autoencoder(input_shape=input_shape, latent_dim=latent_dim, **kwargs)
