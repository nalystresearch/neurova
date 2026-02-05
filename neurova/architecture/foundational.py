# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Foundational Neural Network Architectures for Neurova

Basic building blocks that form the foundation of all deep learning:
- Perceptron (single neuron)
- Single-Layer Neural Network
- Multi-Layer Perceptron (MLP)
- Feedforward Neural Network (FNN)
- DenseNet (fully connected deep network)

These are the ancestors - every other model builds on these ideas.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from .base import BaseArchitecture, ParameterValidator, TrainingHistory


# Activation Functions

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    return 1 - np.tanh(x) ** 2

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation (identity)."""
    return x

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x > 0, x, alpha * x)

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation function."""
    return x * sigmoid(x)

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    'softmax': softmax,
    'linear': linear,
    'leaky_relu': leaky_relu,
    'elu': elu,
    'swish': swish,
    'gelu': gelu,
}

ACTIVATION_DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'relu': relu_derivative,
    'tanh': tanh_derivative,
}


# Perceptron

class Perceptron:
    """
    Single Perceptron - the simplest neural network unit.
    
    A perceptron is a single artificial neuron that computes a weighted sum
    of inputs and applies a step function to produce a binary output.
    
    Parameters
    ----------
    input_size : int
        Number of input features
    learning_rate : float
        Learning rate for weight updates
    activation : str
        Activation function: 'step', 'sigmoid', 'linear'
    max_iter : int
        Maximum number of training iterations
    
    Example
    -------
    >>> perceptron = Perceptron(input_size=2)
    >>> # AND gate
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y = np.array([0, 0, 0, 1])
    >>> perceptron.fit(X, y)
    >>> perceptron.predict([[1, 1]])
    array([1])
    """
    
    def __init__(self, 
                 input_size: int,
                 learning_rate: float = 0.01,
                 activation: str = 'step',
                 max_iter: int = 1000,
                 random_state: Optional[int] = None):
        
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.max_iter = max_iter
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        
        self.is_fitted = False
        self.n_iter_ = 0
        self.errors_ = []
        
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'step':
            return np.where(x >= 0, 1, 0)
        elif self.activation == 'sigmoid':
            return sigmoid(x)
        else:  # linear
            return x
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, input_size)
            
        Returns
        -------
        predictions : np.ndarray
            Predicted labels
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activate(linear_output)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """
        Train the perceptron using the perceptron learning rule.
        
        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, input_size)
        y : np.ndarray
            Training labels of shape (n_samples,)
            
        Returns
        -------
        self : Perceptron
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        for iteration in range(self.max_iter):
            errors = 0
            
            for xi, target in zip(X, y):
                prediction = self.predict(xi.reshape(1, -1))[0]
                
                if self.activation == 'step':
                    prediction = int(prediction)
                
                error = target - prediction
                
                # Update weights using perceptron learning rule
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
                
                errors += int(error != 0)
            
            self.errors_.append(errors)
            self.n_iter_ = iteration + 1
            
            # Converged
            if errors == 0:
                break
        
        self.is_fitted = True
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        predictions = self.predict(X)
        if self.activation == 'step':
            predictions = predictions.astype(int)
            y = np.asarray(y).astype(int)
        return np.mean(predictions == y)


# Single-Layer Neural Network

class SingleLayerNetwork:
    """
    Single-Layer Neural Network (also known as ADALINE).
    
    A network with one layer of neurons, each computing a weighted sum
    and applying an activation function. Uses gradient descent for training.
    
    Parameters
    ----------
    input_size : int
        Number of input features
    output_size : int
        Number of output neurons
    activation : str
        Activation function
    learning_rate : float
        Learning rate for gradient descent
    
    Example
    -------
    >>> net = SingleLayerNetwork(input_size=4, output_size=3)
    >>> net.fit(X_train, y_train)
    >>> predictions = net.predict(X_test)
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'sigmoid',
                 learning_rate: float = 0.01,
                 epochs: int = 100,
                 batch_size: int = 32,
                 random_state: Optional[int] = None):
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros(output_size)
        
        self.is_fitted = False
        self.history = {'loss': []}
        
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        return ACTIVATIONS.get(self.activation, sigmoid)(x)
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        linear = np.dot(X, self.weights) + self.bias
        return self._activate(linear)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._forward(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict(X)
        return np.argmax(probs, axis=1)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SingleLayerNetwork':
        """Train the network using gradient descent."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        # One-hot encode if needed
        if y.ndim == 1:
            y_onehot = np.zeros((len(y), self.output_size))
            y_onehot[np.arange(len(y)), y.astype(int)] = 1
            y = y_onehot
        
        n_samples = len(X)
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                output = self._forward(X_batch)
                
                # Compute loss (cross-entropy)
                loss = -np.mean(y_batch * np.log(output + 1e-10))
                epoch_loss += loss
                
                # Backward pass
                error = output - y_batch
                grad_w = np.dot(X_batch.T, error) / len(X_batch)
                grad_b = np.mean(error, axis=0)
                
                # Update weights
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b
            
            self.history['loss'].append(epoch_loss / n_batches)
        
        self.is_fitted = True
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        predictions = self.predict_classes(X)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        return np.mean(predictions == y)


# Multi-Layer Perceptron (MLP)

class Layer:
    """A single fully connected layer."""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: str = 'relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # He initialization for ReLU, Xavier for others
        if activation == 'relu':
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(2.0 / (input_size + output_size))
        
        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros((1, output_size))
        
        # Cache for backprop
        self.input = None
        self.output = None
        self.z = None  # pre-activation
        
        # Gradients
        self.dW = None
        self.db = None
        
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through layer."""
        self.input = X
        self.z = np.dot(X, self.W) + self.b
        self.output = ACTIVATIONS.get(self.activation, relu)(self.z)
        return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass through layer."""
        # Apply activation derivative
        if self.activation == 'relu':
            dout = dout * (self.z > 0)
        elif self.activation == 'sigmoid':
            s = sigmoid(self.z)
            dout = dout * s * (1 - s)
        elif self.activation == 'tanh':
            dout = dout * (1 - np.tanh(self.z) ** 2)
        # softmax derivative is handled separately
        
        # Compute gradients
        self.dW = np.dot(self.input.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient for next layer
        return np.dot(dout, self.W.T)


class MultiLayerPerceptron(BaseArchitecture):
    """
    Multi-Layer Perceptron (MLP) - Universal Function Approximator.
    
    A feedforward neural network with multiple hidden layers, capable of
    learning complex non-linear relationships.
    
    Parameters
    ----------
    input_shape : int or tuple
        Input feature dimension
    output_shape : int or tuple
        Output dimension (number of classes for classification)
    hidden_layers : list
        List of hidden layer sizes, e.g., [256, 128, 64]
    activation : str
        Activation function for hidden layers
    output_activation : str
        Activation for output layer
    dropout : float
        Dropout rate for regularization
    batch_norm : bool
        Whether to use batch normalization
    l2_reg : float
        L2 regularization strength
    task : str
        'classification', 'regression', or 'binary'
    
    Example
    -------
    >>> # Classification with 3 hidden layers
    >>> model = MultiLayerPerceptron(
    ...     input_shape=784,
    ...     output_shape=10,
    ...     hidden_layers=[512, 256, 128],
    ...     dropout=0.3
    ... )
    >>> model.fit(X_train, y_train)
    >>> accuracy = model.score(X_test, y_test)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_layers': [[64], [128, 64], [256, 128], [512, 256, 128]],
        'dropout': [0.0, 0.2, 0.3, 0.5],
        'activation': ['relu', 'leaky_relu', 'elu', 'swish'],
        'batch_norm': [True, False],
        'l2_reg': [0.0, 0.0001, 0.001],
    }
    
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 output_shape: Union[int, Tuple[int, ...]],
                 hidden_layers: Optional[List[int]] = None,
                 activation: str = 'relu',
                 output_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 l2_reg: float = 0.0,
                 task: str = 'classification',
                 **kwargs):
        
        self.hidden_sizes = hidden_layers or [256, 128]
        self.activation = activation
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.task = task
        
        # Set output activation based on task
        if output_activation:
            self.output_activation = output_activation
        elif task == 'classification':
            self.output_activation = 'softmax'
        elif task == 'binary':
            self.output_activation = 'sigmoid'
        else:
            self.output_activation = 'linear'
        
        # Set loss based on task
        if task == 'classification':
            loss = 'cross_entropy'
        elif task == 'binary':
            loss = 'binary_cross_entropy'
        else:
            loss = 'mse'
        
        super().__init__(input_shape=input_shape, output_shape=output_shape,
                        loss=loss, **kwargs)
    
    def _build_network(self, **kwargs):
        """Build the MLP architecture."""
        self.layers = []
        
        # Input dimension
        in_size = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
        
        # Hidden layers
        for i, out_size in enumerate(self.hidden_sizes):
            layer = Layer(in_size, out_size, self.activation)
            self.layers.append(layer)
            self.weights[f'W{i}'] = layer.W
            self.weights[f'b{i}'] = layer.b
            in_size = out_size
        
        # Output layer
        out_size = self.output_shape[0] if isinstance(self.output_shape, tuple) else self.output_shape
        output_layer = Layer(in_size, out_size, self.output_activation)
        self.layers.append(output_layer)
        self.weights['W_out'] = output_layer.W
        self.weights['b_out'] = output_layer.b
        
        # Dropout mask
        self.dropout_masks = []
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the network."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        out = X
        self.dropout_masks = []
        
        for i, layer in enumerate(self.layers[:-1]):
            # Sync weights
            layer.W = self.weights[f'W{i}']
            layer.b = self.weights[f'b{i}']
            
            out = layer.forward(out, training)
            
            # Apply dropout
            if self.dropout_rate > 0 and training:
                mask = (np.random.rand(*out.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                out = out * mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
        
        # Output layer
        output_layer = self.layers[-1]
        output_layer.W = self.weights['W_out']
        output_layer.b = self.weights['b_out']
        out = output_layer.forward(out, training)
        
        return out
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through the network."""
        gradients = {}
        N = y_pred.shape[0]
        
        # Compute output gradient
        if self.task == 'classification':
            if y_true.ndim == 1:
                y_onehot = np.zeros_like(y_pred)
                y_onehot[np.arange(N), y_true.astype(int)] = 1
                y_true = y_onehot
            dout = (y_pred - y_true) / N
        elif self.task == 'binary':
            dout = (y_pred - y_true.reshape(-1, 1)) / N
        else:
            dout = 2 * (y_pred - y_true.reshape(-1, 1)) / N
        
        # Output layer backward
        output_layer = self.layers[-1]
        dout = output_layer.backward(dout)
        gradients['W_out'] = output_layer.dW + self.l2_reg * output_layer.W
        gradients['b_out'] = output_layer.db
        
        # Hidden layers backward (reverse order)
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            
            # Apply dropout mask
            if self.dropout_masks[i] is not None:
                dout = dout * self.dropout_masks[i]
            
            dout = layer.backward(dout)
            gradients[f'W{i}'] = layer.dW + self.l2_reg * layer.W
            gradients[f'b{i}'] = layer.db
        
        return gradients


# Feedforward Neural Network (FNN)

class FeedforwardNetwork(MultiLayerPerceptron):
    """
    Feedforward Neural Network (FNN).
    
    A neural network where information flows only in one direction,
    from input to output, without any loops or cycles.
    This is essentially the same as MLP but emphasizes the feedforward nature.
    
    Parameters
    ----------
    Same as MultiLayerPerceptron
    
    Example
    -------
    >>> model = FeedforwardNetwork(
    ...     input_shape=100,
    ...     output_shape=10,
    ...     hidden_layers=[64, 32]
    ... )
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Alias
FNN = FeedforwardNetwork


# DenseNet (Fully Connected Deep Network)

class DenseBlock:
    """Dense block with skip connections."""
    
    def __init__(self, input_size: int, growth_rate: int, 
                 n_layers: int = 4, activation: str = 'relu'):
        self.input_size = input_size
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.activation = activation
        
        self.layers = []
        current_size = input_size
        
        for i in range(n_layers):
            layer = Layer(current_size, growth_rate, activation)
            self.layers.append(layer)
            current_size += growth_rate
        
        self.output_size = current_size
        
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with dense connections."""
        features = [X]
        
        for layer in self.layers:
            # Concatenate all previous features
            concat = np.concatenate(features, axis=-1)
            out = layer.forward(concat, training)
            features.append(out)
        
        return np.concatenate(features, axis=-1)


class DenseNet(BaseArchitecture):
    """
    DenseNet - Densely Connected Neural Network.
    
    A network architecture where each layer receives feature maps from all
    preceding layers, promoting feature reuse and gradient flow.
    
    Parameters
    ----------
    input_shape : int
        Input feature dimension
    output_shape : int
        Output dimension
    growth_rate : int
        Number of features added by each layer
    n_blocks : int
        Number of dense blocks
    layers_per_block : int
        Number of layers in each dense block
    compression : float
        Compression factor for transition layers
    
    Example
    -------
    >>> model = DenseNet(
    ...     input_shape=784,
    ...     output_shape=10,
    ...     growth_rate=32,
    ...     n_blocks=3
    ... )
    >>> model.fit(X_train, y_train)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'growth_rate': [16, 32, 48],
        'n_blocks': [2, 3, 4],
        'layers_per_block': [3, 4, 6],
    }
    
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 output_shape: Union[int, Tuple[int, ...]],
                 growth_rate: int = 32,
                 n_blocks: int = 3,
                 layers_per_block: int = 4,
                 compression: float = 0.5,
                 activation: str = 'relu',
                 **kwargs):
        
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.layers_per_block = layers_per_block
        self.compression = compression
        self.activation = activation
        
        super().__init__(input_shape=input_shape, output_shape=output_shape,
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build DenseNet architecture."""
        self.dense_blocks = []
        self.transition_layers = []
        
        in_size = self.input_shape[0] if isinstance(self.input_shape, tuple) else self.input_shape
        
        # Dense blocks with transitions
        for i in range(self.n_blocks):
            block = DenseBlock(in_size, self.growth_rate, 
                             self.layers_per_block, self.activation)
            self.dense_blocks.append(block)
            in_size = block.output_size
            
            # Transition layer (compression)
            if i < self.n_blocks - 1:
                out_size = int(in_size * self.compression)
                transition = Layer(in_size, out_size, self.activation)
                self.transition_layers.append(transition)
                self.weights[f'trans{i}_W'] = transition.W
                self.weights[f'trans{i}_b'] = transition.b
                in_size = out_size
        
        # Output layer
        out_size = self.output_shape[0] if isinstance(self.output_shape, tuple) else self.output_shape
        self.output_layer = Layer(in_size, out_size, 'softmax')
        self.weights['out_W'] = self.output_layer.W
        self.weights['out_b'] = self.output_layer.b
        
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through DenseNet."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        out = X
        
        for i, block in enumerate(self.dense_blocks):
            out = block.forward(out, training)
            
            if i < len(self.transition_layers):
                trans = self.transition_layers[i]
                trans.W = self.weights[f'trans{i}_W']
                trans.b = self.weights[f'trans{i}_b']
                out = trans.forward(out, training)
        
        # Output
        self.output_layer.W = self.weights['out_W']
        self.output_layer.b = self.weights['out_b']
        out = self.output_layer.forward(out, training)
        
        return out
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass (simplified for DenseNet)."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_onehot
        
        dout = (y_pred - y_true) / N
        
        # Output layer gradients
        dout = self.output_layer.backward(dout)
        gradients['out_W'] = self.output_layer.dW
        gradients['out_b'] = self.output_layer.db
        
        # Transition layers gradients (reverse order)
        for i in range(len(self.transition_layers) - 1, -1, -1):
            trans = self.transition_layers[i]
            dout = trans.backward(dout)
            gradients[f'trans{i}_W'] = trans.dW
            gradients[f'trans{i}_b'] = trans.db
        
        return gradients


# Convenience Functions

def create_perceptron(input_size: int, **kwargs) -> Perceptron:
    """Create a Perceptron."""
    return Perceptron(input_size=input_size, **kwargs)

def create_mlp(input_shape: int, output_shape: int, 
               depth: str = 'medium', **kwargs) -> MultiLayerPerceptron:
    """
    Create an MLP with preset configurations.
    
    Parameters
    ----------
    input_shape : int
        Input feature dimension
    output_shape : int
        Output dimension
    depth : str
        'shallow', 'medium', 'deep', or 'very_deep'
    """
    depth_configs = {
        'shallow': [64, 32],
        'medium': [256, 128, 64],
        'deep': [512, 256, 128, 64],
        'very_deep': [1024, 512, 256, 128, 64],
    }
    
    hidden_layers = depth_configs.get(depth, depth_configs['medium'])
    
    return MultiLayerPerceptron(
        input_shape=input_shape,
        output_shape=output_shape,
        hidden_layers=hidden_layers,
        **kwargs
    )

def create_feedforward(input_shape: int, output_shape: int, 
                       **kwargs) -> FeedforwardNetwork:
    """Create a Feedforward Neural Network."""
    return FeedforwardNetwork(input_shape=input_shape, 
                             output_shape=output_shape, **kwargs)

def create_densenet(input_shape: int, output_shape: int,
                    **kwargs) -> DenseNet:
    """Create a DenseNet."""
    return DenseNet(input_shape=input_shape, output_shape=output_shape, **kwargs)


# Aliases for common usage
MLP = MultiLayerPerceptron
ANN = MultiLayerPerceptron  # Artificial Neural Network
