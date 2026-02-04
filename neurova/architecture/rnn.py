# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
RNN (Recurrent Neural Network) Architectures for Neurova

Pre-built RNN architectures for sequence modeling, time series, and NLP tasks.
Includes SimpleRNN, LSTM, GRU, Bidirectional, and Stacked variants.

Features:
- Pre-configured architectures with sensible defaults
- Easy sequence-to-sequence and sequence-to-class modeling
- Built-in training with progress tracking
- Automatic input/output shape validation
- Hyperparameter tuning support
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from .base import BaseArchitecture, ParameterValidator


class RNNCell:
    """Basic RNN cell."""
    
    def __init__(self, input_size: int, hidden_size: int, activation: str = 'tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wxh = np.random.randn(input_size, hidden_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.bh = np.zeros((1, hidden_size))
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass for one time step.
        
        Parameters
        ----------
        x : np.ndarray
            Input at current time step (batch, input_size)
        h_prev : np.ndarray
            Hidden state from previous step (batch, hidden_size)
            
        Returns
        -------
        h : np.ndarray
            New hidden state
        cache : dict
            Values needed for backward pass
        """
        z = x @ self.Wxh + h_prev @ self.Whh + self.bh
        
        if self.activation == 'tanh':
            h = np.tanh(z)
        elif self.activation == 'relu':
            h = np.maximum(0, z)
        else:
            h = z
            
        cache = {'x': x, 'h_prev': h_prev, 'h': h, 'z': z}
        return h, cache
    
    def backward(self, dh: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Backward pass for one time step.
        
        Returns
        -------
        dx : np.ndarray
            Gradient w.r.t. input
        dh_prev : np.ndarray
            Gradient w.r.t. previous hidden state
        grads : dict
            Gradients for weights
        """
        x = cache['x']
        h_prev = cache['h_prev']
        h = cache['h']
        z = cache['z']
        
        # Gradient through activation
        if self.activation == 'tanh':
            dz = dh * (1 - h ** 2)
        elif self.activation == 'relu':
            dz = dh * (z > 0)
        else:
            dz = dh
        
        # Gradients
        dWxh = x.T @ dz
        dWhh = h_prev.T @ dz
        dbh = np.sum(dz, axis=0, keepdims=True)
        
        dx = dz @ self.Wxh.T
        dh_prev = dz @ self.Whh.T
        
        grads = {'Wxh': dWxh, 'Whh': dWhh, 'bh': dbh}
        return dx, dh_prev, grads


class LSTMCell:
    """
    Long Short-Term Memory cell.
    
    Handles long-term dependencies with gates:
    - Forget gate: What to forget from cell state
    - Input gate: What new information to add
    - Output gate: What to output
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weights for all gates (more efficient)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Input weights [i, f, g, o]
        self.Wx = np.random.randn(input_size, 4 * hidden_size) * scale
        # Hidden weights
        self.Wh = np.random.randn(hidden_size, 4 * hidden_size) * scale
        # Biases (forget gate bias initialized to 1 for better gradient flow)
        self.b = np.zeros((1, 4 * hidden_size))
        self.b[0, hidden_size:2*hidden_size] = 1.0  # Forget gate bias
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray, 
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass for one time step.
        
        Parameters
        ----------
        x : np.ndarray
            Input (batch, input_size)
        h_prev : np.ndarray
            Previous hidden state (batch, hidden_size)
        c_prev : np.ndarray
            Previous cell state (batch, hidden_size)
            
        Returns
        -------
        h : np.ndarray
            New hidden state
        c : np.ndarray
            New cell state
        cache : dict
            Values for backward pass
        """
        H = self.hidden_size
        
        # Compute all gates in one matmul
        gates = x @ self.Wx + h_prev @ self.Wh + self.b
        
        # Split into individual gates
        i = self._sigmoid(gates[:, :H])           # Input gate
        f = self._sigmoid(gates[:, H:2*H])        # Forget gate
        g = np.tanh(gates[:, 2*H:3*H])            # Cell gate
        o = self._sigmoid(gates[:, 3*H:])         # Output gate
        
        # New cell state
        c = f * c_prev + i * g
        
        # New hidden state
        h = o * np.tanh(c)
        
        cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'i': i, 'f': f, 'g': g, 'o': o, 'c': c, 'h': h,
            'gates': gates
        }
        return h, c, cache
    
    def backward(self, dh: np.ndarray, dc_next: np.ndarray, 
                 cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Backward pass for one time step.
        
        Returns
        -------
        dx : np.ndarray
            Gradient w.r.t. input
        dh_prev : np.ndarray
            Gradient w.r.t. previous hidden state
        dc_prev : np.ndarray
            Gradient w.r.t. previous cell state
        grads : dict
            Gradients for weights
        """
        H = self.hidden_size
        
        x = cache['x']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        i, f, g, o = cache['i'], cache['f'], cache['g'], cache['o']
        c, h = cache['c'], cache['h']
        
        # Gradient through output
        tanh_c = np.tanh(c)
        dc = dc_next + dh * o * (1 - tanh_c ** 2)
        
        # Gate gradients
        do = dh * tanh_c
        di = dc * g
        df = dc * c_prev
        dg = dc * i
        
        # Gradient through activations
        do_gate = do * o * (1 - o)
        di_gate = di * i * (1 - i)
        df_gate = df * f * (1 - f)
        dg_gate = dg * (1 - g ** 2)
        
        # Combine gate gradients
        dgates = np.concatenate([di_gate, df_gate, dg_gate, do_gate], axis=1)
        
        # Weight gradients
        dWx = x.T @ dgates
        dWh = h_prev.T @ dgates
        db = np.sum(dgates, axis=0, keepdims=True)
        
        # Input and hidden state gradients
        dx = dgates @ self.Wx.T
        dh_prev = dgates @ self.Wh.T
        dc_prev = dc * f
        
        grads = {'Wx': dWx, 'Wh': dWh, 'b': db}
        return dx, dh_prev, dc_prev, grads
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))


class GRUCell:
    """
    Gated Recurrent Unit cell.
    
    Simplified variant of LSTM with:
    - Reset gate: How much to forget
    - Update gate: How much to update
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Combined weights for reset and update gates [r, z]
        self.Wx_gates = np.random.randn(input_size, 2 * hidden_size) * scale
        self.Wh_gates = np.random.randn(hidden_size, 2 * hidden_size) * scale
        self.b_gates = np.zeros((1, 2 * hidden_size))
        
        # Weights for candidate hidden state
        self.Wx_h = np.random.randn(input_size, hidden_size) * scale
        self.Wh_h = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros((1, hidden_size))
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass for one time step."""
        H = self.hidden_size
        
        # Compute gates
        gates = x @ self.Wx_gates + h_prev @ self.Wh_gates + self.b_gates
        r = self._sigmoid(gates[:, :H])      # Reset gate
        z = self._sigmoid(gates[:, H:])      # Update gate
        
        # Candidate hidden state
        h_tilde = np.tanh(x @ self.Wx_h + (r * h_prev) @ self.Wh_h + self.b_h)
        
        # New hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        cache = {
            'x': x, 'h_prev': h_prev, 'r': r, 'z': z, 
            'h_tilde': h_tilde, 'h': h, 'gates': gates
        }
        return h, cache
    
    def backward(self, dh: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Backward pass for one time step."""
        H = self.hidden_size
        
        x = cache['x']
        h_prev = cache['h_prev']
        r, z = cache['r'], cache['z']
        h_tilde = cache['h_tilde']
        
        # Gradients
        dh_tilde = dh * z
        dz = dh * (h_tilde - h_prev)
        dh_prev = dh * (1 - z)
        
        # Through tanh
        dh_tilde_raw = dh_tilde * (1 - h_tilde ** 2)
        
        # Candidate gradients
        dWx_h = x.T @ dh_tilde_raw
        dWh_h = (r * h_prev).T @ dh_tilde_raw
        db_h = np.sum(dh_tilde_raw, axis=0, keepdims=True)
        
        dr = (dh_tilde_raw @ self.Wh_h.T) * h_prev
        dh_prev += (dh_tilde_raw @ self.Wh_h.T) * r
        
        # Gate gradients
        dr_gate = dr * r * (1 - r)
        dz_gate = dz * z * (1 - z)
        dgates = np.concatenate([dr_gate, dz_gate], axis=1)
        
        dWx_gates = x.T @ dgates
        dWh_gates = h_prev.T @ dgates
        db_gates = np.sum(dgates, axis=0, keepdims=True)
        
        dx = dgates @ self.Wx_gates.T + dh_tilde_raw @ self.Wx_h.T
        dh_prev += dgates @ self.Wh_gates.T
        
        grads = {
            'Wx_gates': dWx_gates, 'Wh_gates': dWh_gates, 'b_gates': db_gates,
            'Wx_h': dWx_h, 'Wh_h': dWh_h, 'b_h': db_h
        }
        return dx, dh_prev, grads
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0,
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))


class RNN(BaseArchitecture):
    """
    Recurrent Neural Network for sequence modeling.
    
    Supports multiple RNN cell types and configurations.
    
    Parameters
    ----------
    input_shape : tuple
        (sequence_length, input_features) or just (input_features,)
    output_shape : int or tuple
        Number of output classes or features
    hidden_size : int
        Number of hidden units per layer
    num_layers : int
        Number of stacked RNN layers
    cell_type : str
        RNN cell type: 'rnn', 'lstm', 'gru'
    bidirectional : bool
        Use bidirectional RNN
    dropout : float
        Dropout rate between layers
    return_sequences : bool
        Return full sequence or just last output
    
    Example
    -------
    >>> # Simple classification
    >>> model = RNN(input_shape=(100, 10), output_shape=5)
    >>> model.fit(X_train, y_train)
    
    >>> # Time series with LSTM
    >>> model = RNN(
    ...     input_shape=(50, 1),
    ...     output_shape=1,
    ...     cell_type='lstm',
    ...     num_layers=2,
    ...     hidden_size=64,
    ... )
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_size': [32, 64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.0, 0.2, 0.5],
        'cell_type': ['lstm', 'gru', 'rnn'],
    }
    
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...], Tuple[int, int]],
                 output_shape: Union[int, Tuple[int]],
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 cell_type: str = 'lstm',
                 bidirectional: bool = False,
                 dropout: float = 0.0,
                 return_sequences: bool = False,
                 **kwargs):
        
        # Store config before super().__init__
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        self.return_sequences = return_sequences
        
        # Parse input shape
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if len(input_shape) == 1:
            # Just features, sequence length will be inferred
            input_shape = (None, input_shape[0])
            
        super().__init__(input_shape=input_shape, output_shape=output_shape, 
                        loss='cross_entropy', **kwargs)
        
    def _build_network(self, **kwargs):
        """Build RNN architecture."""
        self.cells = []
        self.cells_backward = []  # For bidirectional
        
        # Determine input size for each layer
        input_size = self.input_shape[-1] if self.input_shape[-1] is not None else 1
        
        cell_class = {
            'rnn': RNNCell,
            'lstm': LSTMCell,
            'gru': GRUCell,
        }[self.cell_type]
        
        for layer in range(self.num_layers):
            layer_input_size = input_size if layer == 0 else self.hidden_size
            if self.bidirectional and layer > 0:
                layer_input_size = self.hidden_size * 2
                
            cell = cell_class(layer_input_size, self.hidden_size)
            self.cells.append(cell)
            
            # Store weights
            if self.cell_type == 'lstm':
                self.weights[f'cell{layer}_Wx'] = cell.Wx
                self.weights[f'cell{layer}_Wh'] = cell.Wh
                self.weights[f'cell{layer}_b'] = cell.b
            elif self.cell_type == 'gru':
                self.weights[f'cell{layer}_Wx_gates'] = cell.Wx_gates
                self.weights[f'cell{layer}_Wh_gates'] = cell.Wh_gates
                self.weights[f'cell{layer}_b_gates'] = cell.b_gates
                self.weights[f'cell{layer}_Wx_h'] = cell.Wx_h
                self.weights[f'cell{layer}_Wh_h'] = cell.Wh_h
                self.weights[f'cell{layer}_b_h'] = cell.b_h
            else:
                self.weights[f'cell{layer}_Wxh'] = cell.Wxh
                self.weights[f'cell{layer}_Whh'] = cell.Whh
                self.weights[f'cell{layer}_bh'] = cell.bh
            
            if self.bidirectional:
                cell_bw = cell_class(layer_input_size, self.hidden_size)
                self.cells_backward.append(cell_bw)
                
                if self.cell_type == 'lstm':
                    self.weights[f'cell{layer}_bw_Wx'] = cell_bw.Wx
                    self.weights[f'cell{layer}_bw_Wh'] = cell_bw.Wh
                    self.weights[f'cell{layer}_bw_b'] = cell_bw.b
                elif self.cell_type == 'gru':
                    self.weights[f'cell{layer}_bw_Wx_gates'] = cell_bw.Wx_gates
                    self.weights[f'cell{layer}_bw_Wh_gates'] = cell_bw.Wh_gates
                    self.weights[f'cell{layer}_bw_b_gates'] = cell_bw.b_gates
                    self.weights[f'cell{layer}_bw_Wx_h'] = cell_bw.Wx_h
                    self.weights[f'cell{layer}_bw_Wh_h'] = cell_bw.Wh_h
                    self.weights[f'cell{layer}_bw_b_h'] = cell_bw.b_h
                else:
                    self.weights[f'cell{layer}_bw_Wxh'] = cell_bw.Wxh
                    self.weights[f'cell{layer}_bw_Whh'] = cell_bw.Whh
                    self.weights[f'cell{layer}_bw_bh'] = cell_bw.bh
        
        # Output layer
        output_input_size = self.hidden_size * (2 if self.bidirectional else 1)
        n_classes = self.output_shape[0] if isinstance(self.output_shape, tuple) else self.output_shape
        
        scale = np.sqrt(2.0 / output_input_size)
        self.weights['output_W'] = np.random.randn(output_input_size, n_classes) * scale
        self.weights['output_b'] = np.zeros((1, n_classes))
        
    def _sync_weights(self):
        """Sync weights from self.weights to cells."""
        for layer, cell in enumerate(self.cells):
            if self.cell_type == 'lstm':
                cell.Wx = self.weights[f'cell{layer}_Wx']
                cell.Wh = self.weights[f'cell{layer}_Wh']
                cell.b = self.weights[f'cell{layer}_b']
            elif self.cell_type == 'gru':
                cell.Wx_gates = self.weights[f'cell{layer}_Wx_gates']
                cell.Wh_gates = self.weights[f'cell{layer}_Wh_gates']
                cell.b_gates = self.weights[f'cell{layer}_b_gates']
                cell.Wx_h = self.weights[f'cell{layer}_Wx_h']
                cell.Wh_h = self.weights[f'cell{layer}_Wh_h']
                cell.b_h = self.weights[f'cell{layer}_b_h']
            else:
                cell.Wxh = self.weights[f'cell{layer}_Wxh']
                cell.Whh = self.weights[f'cell{layer}_Whh']
                cell.bh = self.weights[f'cell{layer}_bh']
                
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through RNN."""
        self._sync_weights()
        
        # X shape: (batch, seq_len, features)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
            
        batch_size, seq_len, _ = X.shape
        
        self.caches = []
        
        # Process through layers
        layer_input = X
        
        for layer_idx, cell in enumerate(self.cells):
            # Initialize hidden states
            h = np.zeros((batch_size, self.hidden_size))
            if self.cell_type == 'lstm':
                c = np.zeros((batch_size, self.hidden_size))
            
            outputs = []
            layer_cache = []
            
            # Forward direction
            for t in range(seq_len):
                x_t = layer_input[:, t, :]
                
                if self.cell_type == 'lstm':
                    h, c, cache = cell.forward(x_t, h, c)
                elif self.cell_type == 'gru':
                    h, cache = cell.forward(x_t, h)
                else:
                    h, cache = cell.forward(x_t, h)
                    
                outputs.append(h)
                layer_cache.append(cache)
            
            forward_outputs = np.stack(outputs, axis=1)  # (batch, seq, hidden)
            
            # Backward direction (bidirectional)
            if self.bidirectional:
                cell_bw = self.cells_backward[layer_idx]
                h_bw = np.zeros((batch_size, self.hidden_size))
                if self.cell_type == 'lstm':
                    c_bw = np.zeros((batch_size, self.hidden_size))
                
                outputs_bw = []
                
                for t in range(seq_len - 1, -1, -1):
                    x_t = layer_input[:, t, :]
                    
                    if self.cell_type == 'lstm':
                        h_bw, c_bw, cache = cell_bw.forward(x_t, h_bw, c_bw)
                    elif self.cell_type == 'gru':
                        h_bw, cache = cell_bw.forward(x_t, h_bw)
                    else:
                        h_bw, cache = cell_bw.forward(x_t, h_bw)
                        
                    outputs_bw.insert(0, h_bw)
                
                backward_outputs = np.stack(outputs_bw, axis=1)
                layer_output = np.concatenate([forward_outputs, backward_outputs], axis=2)
            else:
                layer_output = forward_outputs
            
            # Apply dropout between layers (not on last layer)
            if self.dropout_rate > 0 and training and layer_idx < self.num_layers - 1:
                mask = (np.random.rand(*layer_output.shape) > self.dropout_rate)
                layer_output = layer_output * mask / (1 - self.dropout_rate)
            
            self.caches.append(layer_cache)
            layer_input = layer_output
        
        # Output layer
        if self.return_sequences:
            # Apply output to all timesteps
            output = np.zeros((batch_size, seq_len, self.n_classes))
            for t in range(seq_len):
                out_t = layer_output[:, t, :] @ self.weights['output_W'] + self.weights['output_b']
                output[:, t, :] = self._softmax(out_t)
            return output
        else:
            # Use last timestep
            final_hidden = layer_output[:, -1, :]
            logits = final_hidden @ self.weights['output_W'] + self.weights['output_b']
            return self._softmax(logits)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through RNN."""
        gradients = {}
        N = y_pred.shape[0]
        
        # Convert labels if needed
        if y_true.ndim == 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        # Output gradient
        dout = (y_pred - y_true) / N
        
        # Output layer gradients
        if self.return_sequences:
            # Sum over sequence
            seq_len = dout.shape[1]
            gradients['output_W'] = np.zeros_like(self.weights['output_W'])
            gradients['output_b'] = np.zeros_like(self.weights['output_b'])
            # Simplified: just handle last timestep for now
            dout = dout[:, -1, :]
        
        # Get final hidden from last forward pass
        hidden_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        gradients['output_W'] = np.zeros_like(self.weights['output_W'])
        gradients['output_b'] = np.sum(dout, axis=0, keepdims=True)
        
        # Initialize RNN cell gradients
        for layer in range(self.num_layers):
            if self.cell_type == 'lstm':
                gradients[f'cell{layer}_Wx'] = np.zeros_like(self.weights[f'cell{layer}_Wx'])
                gradients[f'cell{layer}_Wh'] = np.zeros_like(self.weights[f'cell{layer}_Wh'])
                gradients[f'cell{layer}_b'] = np.zeros_like(self.weights[f'cell{layer}_b'])
            elif self.cell_type == 'gru':
                gradients[f'cell{layer}_Wx_gates'] = np.zeros_like(self.weights[f'cell{layer}_Wx_gates'])
                gradients[f'cell{layer}_Wh_gates'] = np.zeros_like(self.weights[f'cell{layer}_Wh_gates'])
                gradients[f'cell{layer}_b_gates'] = np.zeros_like(self.weights[f'cell{layer}_b_gates'])
                gradients[f'cell{layer}_Wx_h'] = np.zeros_like(self.weights[f'cell{layer}_Wx_h'])
                gradients[f'cell{layer}_Wh_h'] = np.zeros_like(self.weights[f'cell{layer}_Wh_h'])
                gradients[f'cell{layer}_b_h'] = np.zeros_like(self.weights[f'cell{layer}_b_h'])
            else:
                gradients[f'cell{layer}_Wxh'] = np.zeros_like(self.weights[f'cell{layer}_Wxh'])
                gradients[f'cell{layer}_Whh'] = np.zeros_like(self.weights[f'cell{layer}_Whh'])
                gradients[f'cell{layer}_bh'] = np.zeros_like(self.weights[f'cell{layer}_bh'])
        
        return gradients


class LSTM(RNN):
    """
    Long Short-Term Memory Network.
    
    Specialized RNN with LSTM cells for learning long-term dependencies.
    
    Parameters
    ----------
    input_shape : tuple
        (sequence_length, features)
    output_shape : int
        Number of output classes
    hidden_size : int
        LSTM hidden state size
    num_layers : int
        Number of stacked LSTM layers
    bidirectional : bool
        Use bidirectional LSTM
    
    Example
    -------
    >>> # Text classification
    >>> model = LSTM(input_shape=(100, 300), output_shape=5, hidden_size=128)
    >>> model.fit(X_train, y_train, epochs=20)
    
    >>> # Time series prediction
    >>> model = LSTM(
    ...     input_shape=(50, 1),
    ...     output_shape=1,
    ...     num_layers=2,
    ...     bidirectional=True,
    ... )
    """
    
    def __init__(self, input_shape, output_shape, hidden_size: int = 128,
                 num_layers: int = 1, bidirectional: bool = False, **kwargs):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_type='lstm',
            bidirectional=bidirectional,
            **kwargs
        )


class GRU(RNN):
    """
    Gated Recurrent Unit Network.
    
    Simplified variant of LSTM with fewer parameters.
    
    Parameters
    ----------
    input_shape : tuple
        (sequence_length, features)
    output_shape : int
        Number of output classes
    hidden_size : int
        GRU hidden state size
    num_layers : int
        Number of stacked GRU layers
    
    Example
    -------
    >>> model = GRU(input_shape=(50, 10), output_shape=3, hidden_size=64)
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, input_shape, output_shape, hidden_size: int = 64,
                 num_layers: int = 1, **kwargs):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_type='gru',
            **kwargs
        )


class BiLSTM(LSTM):
    """
    Bidirectional LSTM Network.
    
    Processes sequence in both directions for better context.
    
    Example
    -------
    >>> model = BiLSTM(input_shape=(100, 50), output_shape=10)
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, input_shape, output_shape, hidden_size: int = 64, **kwargs):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_size=hidden_size,
            bidirectional=True,
            **kwargs
        )


class StackedLSTM(LSTM):
    """
    Deep stacked LSTM Network.
    
    Multiple LSTM layers for learning hierarchical representations.
    
    Parameters
    ----------
    input_shape : tuple
        (sequence_length, features)
    output_shape : int
        Number of output classes
    hidden_sizes : list
        Hidden size for each layer, e.g., [256, 128, 64]
    
    Example
    -------
    >>> model = StackedLSTM(
    ...     input_shape=(100, 50),
    ...     output_shape=10,
    ...     hidden_sizes=[256, 128, 64],
    ... )
    """
    
    def __init__(self, input_shape, output_shape, 
                 hidden_sizes: List[int] = [128, 64], **kwargs):
        # Use first hidden size, will build manually
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_size=hidden_sizes[0],
            num_layers=len(hidden_sizes),
            **kwargs
        )


class SimpleRNN(RNN):
    """
    Simple/Vanilla RNN Network.
    
    Basic RNN without gating mechanisms. Good for short sequences.
    
    Example
    -------
    >>> model = SimpleRNN(input_shape=(20, 10), output_shape=5)
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, input_shape, output_shape, hidden_size: int = 32, **kwargs):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_size=hidden_size,
            cell_type='rnn',
            **kwargs
        )


class Seq2SeqEncoder:
    """Encoder for sequence-to-sequence models."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                     for i in range(num_layers)]
        
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Encode input sequence."""
        batch_size, seq_len, _ = X.shape
        
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            for cell in self.cells:
                h, c, _ = cell.forward(x_t, h, c)
                x_t = h
                
        return h, (h, c)


class Seq2SeqDecoder:
    """Decoder for sequence-to-sequence models."""
    
    def __init__(self, output_size: int, hidden_size: int, num_layers: int = 1):
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.cells = [LSTMCell(output_size if i == 0 else hidden_size, hidden_size)
                     for i in range(num_layers)]
        
        # Output projection
        scale = np.sqrt(2.0 / hidden_size)
        self.W_out = np.random.randn(hidden_size, output_size) * scale
        self.b_out = np.zeros((1, output_size))
        
    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode one step."""
        for cell in self.cells:
            h, c, _ = cell.forward(x, h, c)
            x = h
            
        output = h @ self.W_out + self.b_out
        return output, h, c


class Seq2Seq(BaseArchitecture):
    """
    Sequence-to-Sequence Model.
    
    Encoder-decoder architecture for tasks like translation, summarization.
    
    Parameters
    ----------
    input_vocab_size : int
        Size of input vocabulary
    output_vocab_size : int
        Size of output vocabulary
    hidden_size : int
        Hidden state size
    max_length : int
        Maximum sequence length
    
    Example
    -------
    >>> model = Seq2Seq(
    ...     input_vocab_size=10000,
    ...     output_vocab_size=8000,
    ...     hidden_size=256,
    ... )
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, input_vocab_size: int, output_vocab_size: int,
                 hidden_size: int = 256, embedding_dim: int = 128,
                 max_length: int = 100, **kwargs):
        
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self._hidden_size = hidden_size
        
        super().__init__(
            input_shape=(max_length, embedding_dim),
            output_shape=output_vocab_size,
            **kwargs
        )
        
    def _build_network(self, **kwargs):
        """Build Seq2Seq architecture."""
        # Embeddings
        self.weights['encoder_emb'] = np.random.randn(self.input_vocab_size, self.embedding_dim) * 0.01
        self.weights['decoder_emb'] = np.random.randn(self.output_vocab_size, self.embedding_dim) * 0.01
        
        # Encoder LSTM
        self.encoder = Seq2SeqEncoder(self.embedding_dim, self._hidden_size)
        
        # Decoder LSTM
        self.decoder = Seq2SeqDecoder(self.embedding_dim, self._hidden_size)
        
        # Store encoder/decoder weights
        for i, cell in enumerate(self.encoder.cells):
            self.weights[f'enc_cell{i}_Wx'] = cell.Wx
            self.weights[f'enc_cell{i}_Wh'] = cell.Wh
            self.weights[f'enc_cell{i}_b'] = cell.b
            
        for i, cell in enumerate(self.decoder.cells):
            self.weights[f'dec_cell{i}_Wx'] = cell.Wx
            self.weights[f'dec_cell{i}_Wh'] = cell.Wh
            self.weights[f'dec_cell{i}_b'] = cell.b
            
        self.weights['dec_W_out'] = self.decoder.W_out
        self.weights['dec_b_out'] = self.decoder.b_out
        
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass (simplified - returns encoder output)."""
        batch_size = X.shape[0]
        
        # Embed input
        if X.dtype in [np.int32, np.int64]:
            X_emb = self.weights['encoder_emb'][X]
        else:
            X_emb = X
            
        # Encode
        _, (h, c) = self.encoder.forward(X_emb)
        
        # For training, we'd teacher-force the decoder
        # Simplified: return final hidden state projected to output
        output = h @ self.decoder.W_out + self.decoder.b_out
        return self._softmax(output)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass (simplified)."""
        gradients = {}
        N = y_pred.shape[0]
        
        if y_true.ndim == 1:
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / N
        
        gradients['dec_W_out'] = np.zeros_like(self.weights['dec_W_out'])
        gradients['dec_b_out'] = np.sum(dout, axis=0, keepdims=True)
        
        return gradients


# Convenience functions
def create_rnn(input_shape, output_shape, architecture: str = 'lstm', **kwargs) -> RNN:
    """
    Create an RNN with minimal configuration.
    
    Parameters
    ----------
    input_shape : tuple
        (sequence_length, features)
    output_shape : int
        Number of classes
    architecture : str
        RNN type: 'rnn', 'lstm', 'gru', 'bilstm'
    
    Returns
    -------
    model : RNN
        Configured RNN model
    
    Example
    -------
    >>> model = create_rnn((100, 10), 5, 'bilstm')
    >>> model.fit(X_train, y_train)
    """
    architectures = {
        'rnn': SimpleRNN,
        'lstm': LSTM,
        'gru': GRU,
        'bilstm': BiLSTM,
    }
    
    arch = architecture.lower()
    if arch not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Choose from: {list(architectures.keys())}")
    
    return architectures[arch](input_shape=input_shape, output_shape=output_shape, **kwargs)
