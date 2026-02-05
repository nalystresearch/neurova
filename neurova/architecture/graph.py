# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Graph Neural Networks for Neurova

Complete implementation of graph-based architectures:
- GNN (Basic Graph Neural Network)
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE (Sample and Aggregate)
- MPNN (Message Passing Neural Network)
- GIN (Graph Isomorphism Network)
- EdgeConv (Edge Convolution)
- ChebNet (Chebyshev Spectral Graph Convolution)

All implementations use pure NumPy for educational purposes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from .base import BaseArchitecture


# Utility Functions

def normalize_adjacency(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    
    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (N, N)
    add_self_loops : bool
        Whether to add self-loops (identity)
    
    Returns
    -------
    A_norm : np.ndarray
        Normalized adjacency matrix
    """
    if add_self_loops:
        A = A + np.eye(A.shape[0])
    
    # Degree matrix
    D = np.diag(np.sum(A, axis=1))
    
    # D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
    
    return D_inv_sqrt @ A @ D_inv_sqrt


def compute_laplacian(A: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
    Compute graph Laplacian.
    
    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix
    normalized : bool
        Whether to compute normalized Laplacian
    
    Returns
    -------
    L : np.ndarray
        Graph Laplacian
    """
    D = np.diag(np.sum(A, axis=1))
    
    if normalized:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
        return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        return D - A


def edge_index_to_adjacency(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Convert edge index to adjacency matrix.
    
    Parameters
    ----------
    edge_index : np.ndarray
        Edge indices of shape (2, E)
    n_nodes : int
        Number of nodes
    
    Returns
    -------
    A : np.ndarray
        Adjacency matrix
    """
    A = np.zeros((n_nodes, n_nodes))
    A[edge_index[0], edge_index[1]] = 1
    return A


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x > 0, x, alpha * x)


# Layer Components

class GraphConvLayer:
    """
    Graph Convolutional Layer.
    
    H' = sigma(D^(-1/2) * A * D^(-1/2) * H * W)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, activation: str = 'relu'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features) if bias else None
    
    def forward(self, X: np.ndarray, A_norm: np.ndarray, 
                training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : np.ndarray
            Node features (N, in_features)
        A_norm : np.ndarray
            Normalized adjacency matrix (N, N)
        
        Returns
        -------
        H : np.ndarray
            Output node features (N, out_features)
        """
        # Aggregate neighbor features
        H = A_norm @ X @ self.W
        
        if self.b is not None:
            H += self.b
        
        # Apply activation
        if self.activation == 'relu':
            H = relu(H)
        elif self.activation == 'leaky_relu':
            H = leaky_relu(H)
        elif self.activation == 'tanh':
            H = np.tanh(H)
        
        return H


class GraphAttentionLayer:
    """
    Graph Attention Layer (GAT).
    
    Uses attention mechanism to weight neighbor contributions.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 1, concat: bool = True,
                 dropout: float = 0.0, alpha: float = 0.2):
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha
        
        # Weight matrices for each head
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(n_heads, in_features, out_features) * scale
        
        # Attention parameters for each head
        self.a = np.random.randn(n_heads, 2 * out_features) * 0.01
    
    def forward(self, X: np.ndarray, A: np.ndarray, 
                training: bool = True) -> np.ndarray:
        """
        Forward pass with attention.
        
        Parameters
        ----------
        X : np.ndarray
            Node features (N, in_features)
        A : np.ndarray
            Adjacency matrix (N, N) - used for masking
        
        Returns
        -------
        H : np.ndarray
            Output node features
        """
        N = X.shape[0]
        heads_out = []
        
        for head in range(self.n_heads):
            # Linear transformation
            Wh = X @ self.W[head]  # (N, out_features)
            
            # Compute attention coefficients
            # a_input = [Wh_i || Wh_j] for all pairs
            a_input = np.zeros((N, N, 2 * self.out_features))
            for i in range(N):
                for j in range(N):
                    a_input[i, j] = np.concatenate([Wh[i], Wh[j]])
            
            # Attention scores
            e = leaky_relu(a_input @ self.a[head], self.alpha)  # (N, N)
            
            # Mask out non-neighbors (set to -inf before softmax)
            mask = np.where(A > 0, 0, -1e9)
            e = e + mask
            
            # Softmax over neighbors
            attention = softmax(e, axis=1)  # (N, N)
            
            # Apply dropout
            if training and self.dropout > 0:
                attention = attention * (np.random.rand(*attention.shape) > self.dropout)
            
            # Aggregate
            h = attention @ Wh  # (N, out_features)
            heads_out.append(h)
        
        if self.concat:
            return np.concatenate(heads_out, axis=-1)
        else:
            return np.mean(heads_out, axis=0)


class SAGEConvLayer:
    """
    GraphSAGE Convolution Layer.
    
    Aggregates neighbor features via mean/max/LSTM pooling.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 aggregator: str = 'mean', normalize: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.normalize = normalize
        
        scale = np.sqrt(2.0 / (in_features + out_features))
        
        # Self and neighbor transformations
        self.W_self = np.random.randn(in_features, out_features) * scale
        self.W_neigh = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
    
    def _aggregate(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Aggregate neighbor features."""
        if self.aggregator == 'mean':
            # Mean aggregation
            D = np.sum(A, axis=1, keepdims=True)
            D = np.where(D > 0, D, 1)  # Avoid division by zero
            return (A @ X) / D
        elif self.aggregator == 'max':
            # Max pooling aggregation
            N = X.shape[0]
            agg = np.zeros((N, X.shape[1]))
            for i in range(N):
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) > 0:
                    agg[i] = np.max(X[neighbors], axis=0)
            return agg
        elif self.aggregator == 'sum':
            return A @ X
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
    
    def forward(self, X: np.ndarray, A: np.ndarray,
                training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : np.ndarray
            Node features (N, in_features)
        A : np.ndarray
            Adjacency matrix (N, N)
        
        Returns
        -------
        H : np.ndarray
            Output node features (N, out_features)
        """
        # Aggregate neighbor features
        neigh_agg = self._aggregate(X, A)
        
        # Combine self and neighbor features
        h_self = X @ self.W_self
        h_neigh = neigh_agg @ self.W_neigh
        
        H = h_self + h_neigh + self.b
        H = relu(H)
        
        # L2 normalize
        if self.normalize:
            H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        
        return H


class MessagePassingLayer:
    """
    Message Passing Neural Network Layer.
    
    General framework for graph neural networks.
    """
    
    def __init__(self, node_features: int, edge_features: int,
                 hidden_dim: int, out_features: int):
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        
        scale = np.sqrt(2.0 / hidden_dim)
        
        # Message function
        self.W_msg = np.random.randn(node_features + edge_features, hidden_dim) * scale
        self.b_msg = np.zeros(hidden_dim)
        
        # Update function
        self.W_update = np.random.randn(node_features + hidden_dim, out_features) * scale
        self.b_update = np.zeros(out_features)
    
    def message(self, X_i: np.ndarray, X_j: np.ndarray, 
                edge_attr: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute messages from neighbors."""
        if edge_attr is not None:
            msg_input = np.concatenate([X_j, edge_attr], axis=-1)
        else:
            msg_input = X_j
        
        msg = msg_input @ self.W_msg + self.b_msg
        return relu(msg)
    
    def aggregate(self, messages: List[np.ndarray]) -> np.ndarray:
        """Aggregate messages (sum)."""
        if len(messages) == 0:
            return np.zeros(self.hidden_dim)
        return np.sum(messages, axis=0)
    
    def update(self, X_i: np.ndarray, agg_msg: np.ndarray) -> np.ndarray:
        """Update node features."""
        update_input = np.concatenate([X_i, agg_msg], axis=-1)
        h = update_input @ self.W_update + self.b_update
        return relu(h)
    
    def forward(self, X: np.ndarray, edge_index: np.ndarray,
                edge_attr: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : np.ndarray
            Node features (N, node_features)
        edge_index : np.ndarray
            Edge indices (2, E)
        edge_attr : np.ndarray, optional
            Edge features (E, edge_features)
        
        Returns
        -------
        H : np.ndarray
            Updated node features (N, out_features)
        """
        N = X.shape[0]
        H = np.zeros((N, self.out_features))
        
        for i in range(N):
            # Find neighbors
            neighbor_mask = edge_index[0] == i
            neighbors = edge_index[1, neighbor_mask]
            
            # Compute messages
            messages = []
            for idx, j in enumerate(neighbors):
                e_attr = edge_attr[neighbor_mask][idx] if edge_attr is not None else None
                msg = self.message(X[i], X[j], e_attr)
                messages.append(msg)
            
            # Aggregate and update
            agg_msg = self.aggregate(messages)
            H[i] = self.update(X[i], agg_msg)
        
        return H


class GINLayer:
    """
    Graph Isomorphism Network Layer.
    
    More expressive than GCN for graph classification.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 hidden_dim: int, eps: float = 0.0, train_eps: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.train_eps = train_eps
        
        scale = np.sqrt(2.0 / hidden_dim)
        
        # MLP
        self.W1 = np.random.randn(in_features, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, out_features) * scale
        self.b2 = np.zeros(out_features)
    
    def forward(self, X: np.ndarray, A: np.ndarray,
                training: bool = True) -> np.ndarray:
        """
        Forward pass: h_i = MLP((1 + eps) * h_i + sum_j h_j)
        """
        # Aggregate: (1 + eps) * self + neighbors
        agg = (1 + self.eps) * X + A @ X
        
        # MLP
        h = agg @ self.W1 + self.b1
        h = relu(h)
        h = h @ self.W2 + self.b2
        
        return relu(h)


class EdgeConvLayer:
    """
    Edge Convolution Layer (from DGCNN).
    
    Uses edge features for dynamic graph construction.
    """
    
    def __init__(self, in_features: int, out_features: int, k: int = 20):
        self.in_features = in_features
        self.out_features = out_features
        self.k = k  # Number of nearest neighbors
        
        scale = np.sqrt(2.0 / (2 * in_features))
        
        self.W = np.random.randn(2 * in_features, out_features) * scale
        self.b = np.zeros(out_features)
    
    def _knn_graph(self, X: np.ndarray) -> np.ndarray:
        """Compute k-nearest neighbors graph."""
        N = X.shape[0]
        
        # Compute pairwise distances
        dist = np.sum(X ** 2, axis=1, keepdims=True) + \
               np.sum(X ** 2, axis=1, keepdims=True).T - \
               2 * X @ X.T
        
        # Get k nearest neighbors (excluding self)
        np.fill_diagonal(dist, np.inf)
        knn_idx = np.argsort(dist, axis=1)[:, :self.k]
        
        # Build adjacency
        A = np.zeros((N, N))
        for i in range(N):
            A[i, knn_idx[i]] = 1
        
        return A
    
    def forward(self, X: np.ndarray, A: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        If A is None, compute KNN graph from features.
        """
        if A is None:
            A = self._knn_graph(X)
        
        N = X.shape[0]
        H = np.zeros((N, self.out_features))
        
        for i in range(N):
            neighbors = np.where(A[i] > 0)[0]
            
            if len(neighbors) == 0:
                # Self-loop only
                edge_feat = np.concatenate([X[i], X[i] - X[i]])
                H[i] = relu(edge_feat @ self.W + self.b)
            else:
                # Edge features: [x_i, x_j - x_i]
                edge_feats = []
                for j in neighbors:
                    ef = np.concatenate([X[i], X[j] - X[i]])
                    edge_feats.append(ef)
                
                edge_feats = np.array(edge_feats)
                h = relu(edge_feats @ self.W + self.b)
                H[i] = np.max(h, axis=0)  # Max aggregation
        
        return H


class ChebConvLayer:
    """
    Chebyshev Spectral Graph Convolution.
    
    Uses Chebyshev polynomials for spectral filtering.
    """
    
    def __init__(self, in_features: int, out_features: int, K: int = 3):
        self.in_features = in_features
        self.out_features = out_features
        self.K = K  # Order of Chebyshev polynomials
        
        scale = np.sqrt(2.0 / (in_features + out_features))
        
        # Weight for each Chebyshev polynomial
        self.W = np.random.randn(K, in_features, out_features) * scale
        self.b = np.zeros(out_features)
    
    def forward(self, X: np.ndarray, L: np.ndarray,
                training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : np.ndarray
            Node features (N, in_features)
        L : np.ndarray
            Normalized graph Laplacian (N, N)
        
        Returns
        -------
        H : np.ndarray
            Output features (N, out_features)
        """
        # Scale Laplacian to [-1, 1]
        # L_scaled = 2L / lambda_max - I
        lambda_max = np.max(np.abs(np.linalg.eigvalsh(L)))
        L_scaled = (2 * L / lambda_max) - np.eye(L.shape[0])
        
        # Chebyshev polynomials
        T = [X]  # T_0(L) * X = X
        if self.K > 1:
            T.append(L_scaled @ X)  # T_1(L) * X = L * X
        
        for k in range(2, self.K):
            # T_k(L) = 2 * L * T_{k-1}(L) - T_{k-2}(L)
            T.append(2 * L_scaled @ T[-1] - T[-2])
        
        # Combine with weights
        H = np.zeros((X.shape[0], self.out_features))
        for k in range(self.K):
            H += T[k] @ self.W[k]
        
        H += self.b
        
        return relu(H)


# Full Architecture Implementations

class GNN(BaseArchitecture):
    """
    Basic Graph Neural Network.
    
    Multi-layer graph convolution network.
    
    Parameters
    ----------
    input_dim : int
        Input node feature dimension
    hidden_dims : list
        Hidden layer dimensions
    output_dim : int
        Output dimension
    task : str
        'node' for node classification, 'graph' for graph classification
    
    Example
    -------
    >>> gnn = GNN(input_dim=32, hidden_dims=[64, 64], output_dim=7)
    >>> gnn.fit(X, A, y)
    >>> predictions = gnn.predict(X, A)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001, 0.01],
        'hidden_dims': [[32], [64, 32], [128, 64]],
    }
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 output_dim: int = 2,
                 task: str = 'node',
                 **kwargs):
        
        self.hidden_dims = hidden_dims or [64, 32]
        self.output_dim = output_dim
        self.task = task
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build GNN layers."""
        input_dim = self.input_shape[0]
        
        self.layers = []
        in_dim = input_dim
        
        for h_dim in self.hidden_dims:
            layer = GraphConvLayer(in_dim, h_dim)
            self.layers.append(layer)
            in_dim = h_dim
        
        # Output layer
        self.output_layer = GraphConvLayer(in_dim, self.output_dim, activation='linear')
    
    def _forward(self, X: np.ndarray, A: np.ndarray = None,
                 training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Parameters
        ----------
        X : np.ndarray
            Node features (N, input_dim) or (batch, N, input_dim)
        A : np.ndarray
            Adjacency matrix (N, N)
        """
        # Normalize adjacency
        if A is None:
            A = np.eye(X.shape[-2])
        
        A_norm = normalize_adjacency(A)
        
        # Handle batched input
        if X.ndim == 3:
            batch_size = X.shape[0]
            outputs = []
            for i in range(batch_size):
                out = self._forward_single(X[i], A_norm, training)
                outputs.append(out)
            return np.stack(outputs)
        
        return self._forward_single(X, A_norm, training)
    
    def _forward_single(self, X: np.ndarray, A_norm: np.ndarray,
                       training: bool) -> np.ndarray:
        """Forward pass for single graph."""
        H = X
        
        for layer in self.layers:
            H = layer.forward(H, A_norm, training)
        
        H = self.output_layer.forward(H, A_norm, training)
        
        if self.task == 'graph':
            # Global mean pooling for graph classification
            H = np.mean(H, axis=0, keepdims=True)
        
        # Softmax
        exp_H = np.exp(H - np.max(H, axis=-1, keepdims=True))
        return exp_H / np.sum(exp_H, axis=-1, keepdims=True)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through GNN."""
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


class GCN(GNN):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    
    Semi-supervised classification using graph convolutions.
    
    Example
    -------
    >>> gcn = GCN(input_dim=1433, hidden_dims=[16], output_dim=7)
    >>> gcn.fit(features, adjacency, labels)
    """
    pass  # Same as GNN with GraphConvLayer


class GAT(BaseArchitecture):
    """
    Graph Attention Network.
    
    Uses attention mechanism for neighbor aggregation.
    
    Parameters
    ----------
    input_dim : int
        Input node feature dimension
    hidden_dim : int
        Hidden dimension per head
    output_dim : int
        Output dimension
    n_heads : int
        Number of attention heads
    
    Example
    -------
    >>> gat = GAT(input_dim=1433, hidden_dim=8, output_dim=7, n_heads=8)
    >>> gat.fit(features, adjacency, labels)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001, 0.005],
        'hidden_dim': [8, 16, 32],
        'n_heads': [4, 8],
    }
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 8,
                 output_dim: int = 7,
                 n_heads: int = 8,
                 dropout: float = 0.6,
                 **kwargs):
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dropout = dropout
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build GAT layers."""
        input_dim = self.input_shape[0]
        
        # First attention layer (multi-head, concatenate)
        self.gat1 = GraphAttentionLayer(
            input_dim, self.hidden_dim, self.n_heads,
            concat=True, dropout=self.dropout
        )
        
        # Second attention layer (single head for output)
        self.gat2 = GraphAttentionLayer(
            self.hidden_dim * self.n_heads, self.output_dim, 1,
            concat=False, dropout=self.dropout
        )
    
    def _forward(self, X: np.ndarray, A: np.ndarray = None,
                 training: bool = True) -> np.ndarray:
        """Forward pass."""
        if A is None:
            A = np.eye(X.shape[0])
        
        # Add self-loops
        A = A + np.eye(A.shape[0])
        
        # GAT layers
        H = self.gat1.forward(X, A, training)
        H = relu(H)
        
        H = self.gat2.forward(H, A, training)
        
        # Softmax
        exp_H = np.exp(H - np.max(H, axis=-1, keepdims=True))
        return exp_H / np.sum(exp_H, axis=-1, keepdims=True)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through GAT."""
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


class GraphSAGE(BaseArchitecture):
    """
    GraphSAGE - Inductive Representation Learning.
    
    Learns to aggregate neighbor features via sampling.
    
    Parameters
    ----------
    input_dim : int
        Input node feature dimension
    hidden_dims : list
        Hidden layer dimensions
    output_dim : int
        Output dimension
    aggregator : str
        'mean', 'max', or 'sum'
    
    Example
    -------
    >>> sage = GraphSAGE(input_dim=128, hidden_dims=[128, 64], output_dim=10)
    >>> sage.fit(features, adjacency, labels)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001, 0.01],
        'aggregator': ['mean', 'max', 'sum'],
        'hidden_dims': [[64], [128, 64]],
    }
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 output_dim: int = 10,
                 aggregator: str = 'mean',
                 **kwargs):
        
        self.hidden_dims = hidden_dims or [128, 64]
        self.output_dim = output_dim
        self.aggregator = aggregator
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build GraphSAGE layers."""
        input_dim = self.input_shape[0]
        
        self.layers = []
        in_dim = input_dim
        
        for h_dim in self.hidden_dims:
            layer = SAGEConvLayer(in_dim, h_dim, self.aggregator)
            self.layers.append(layer)
            in_dim = h_dim
        
        # Output projection
        self.output_W = np.random.randn(in_dim, self.output_dim) * 0.01
        self.output_b = np.zeros(self.output_dim)
    
    def _forward(self, X: np.ndarray, A: np.ndarray = None,
                 training: bool = True) -> np.ndarray:
        """Forward pass."""
        if A is None:
            A = np.eye(X.shape[0])
        
        H = X
        
        for layer in self.layers:
            H = layer.forward(H, A, training)
        
        # Output projection
        H = H @ self.output_W + self.output_b
        
        # Softmax
        exp_H = np.exp(H - np.max(H, axis=-1, keepdims=True))
        return exp_H / np.sum(exp_H, axis=-1, keepdims=True)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through GraphSAGE."""
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


class MPNN(BaseArchitecture):
    """
    Message Passing Neural Network.
    
    General framework for graph neural networks with message passing.
    
    Parameters
    ----------
    node_features : int
        Node feature dimension
    edge_features : int
        Edge feature dimension
    hidden_dim : int
        Hidden dimension
    output_dim : int
        Output dimension
    n_layers : int
        Number of message passing layers
    
    Example
    -------
    >>> mpnn = MPNN(node_features=32, edge_features=4, output_dim=10)
    >>> predictions = mpnn.forward(node_feats, edge_index, edge_feats)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'hidden_dim': [64, 128],
        'n_layers': [2, 3, 4],
    }
    
    def __init__(self,
                 node_features: int,
                 edge_features: int = 0,
                 hidden_dim: int = 64,
                 output_dim: int = 10,
                 n_layers: int = 3,
                 task: str = 'graph',
                 **kwargs):
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.task = task
        
        super().__init__(input_shape=(node_features,),
                        output_shape=(output_dim,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build MPNN layers."""
        self.layers = []
        
        in_dim = self.node_features
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else self.hidden_dim
            layer = MessagePassingLayer(
                in_dim, self.edge_features, self.hidden_dim, out_dim
            )
            self.layers.append(layer)
            in_dim = out_dim
        
        # Readout for graph-level prediction
        self.readout_W = np.random.randn(self.hidden_dim, self.output_dim) * 0.01
        self.readout_b = np.zeros(self.output_dim)
    
    def _forward(self, X: np.ndarray, edge_index: np.ndarray = None,
                 edge_attr: np.ndarray = None,
                 training: bool = True) -> np.ndarray:
        """Forward pass."""
        if edge_index is None:
            # Fully connected graph
            N = X.shape[0]
            edge_index = np.array([
                [i for i in range(N) for j in range(N) if i != j],
                [j for i in range(N) for j in range(N) if i != j]
            ])
        
        H = X
        
        for layer in self.layers:
            H = layer.forward(H, edge_index, edge_attr, training)
        
        if self.task == 'graph':
            # Global sum pooling
            H = np.sum(H, axis=0, keepdims=True)
        
        # Output projection
        out = H @ self.readout_W + self.readout_b
        
        # Softmax
        exp_out = np.exp(out - np.max(out, axis=-1, keepdims=True))
        return exp_out / np.sum(exp_out, axis=-1, keepdims=True)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through MPNN."""
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


class GIN(BaseArchitecture):
    """
    Graph Isomorphism Network.
    
    More expressive than GCN for graph-level tasks.
    
    Parameters
    ----------
    input_dim : int
        Input node feature dimension
    hidden_dim : int
        Hidden dimension
    output_dim : int
        Output dimension
    n_layers : int
        Number of GIN layers
    
    Example
    -------
    >>> gin = GIN(input_dim=32, hidden_dim=64, output_dim=2, n_layers=5)
    >>> gin.fit(graphs, labels)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001, 0.01],
        'hidden_dim': [32, 64, 128],
        'n_layers': [3, 4, 5],
    }
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 2,
                 n_layers: int = 5,
                 train_eps: bool = True,
                 **kwargs):
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.train_eps = train_eps
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build GIN layers."""
        input_dim = self.input_shape[0]
        
        self.layers = []
        in_dim = input_dim
        
        for i in range(self.n_layers):
            layer = GINLayer(in_dim, self.hidden_dim, self.hidden_dim,
                           train_eps=self.train_eps)
            self.layers.append(layer)
            in_dim = self.hidden_dim
        
        # Graph-level readout (sum of all layer representations)
        self.readout_W = np.random.randn(
            self.hidden_dim * self.n_layers, self.output_dim
        ) * 0.01
        self.readout_b = np.zeros(self.output_dim)
    
    def _forward(self, X: np.ndarray, A: np.ndarray = None,
                 training: bool = True) -> np.ndarray:
        """Forward pass with jumping knowledge."""
        if A is None:
            A = np.eye(X.shape[0])
        
        layer_outputs = []
        H = X
        
        for layer in self.layers:
            H = layer.forward(H, A, training)
            layer_outputs.append(np.sum(H, axis=0))  # Graph-level sum
        
        # Concatenate all layer outputs
        graph_rep = np.concatenate(layer_outputs)
        
        # Readout
        out = graph_rep @ self.readout_W + self.readout_b
        
        # Softmax
        exp_out = np.exp(out - np.max(out))
        return exp_out / np.sum(exp_out)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through GIN."""
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


class ChebNet(BaseArchitecture):
    """
    Chebyshev Spectral Graph Convolution Network.
    
    Uses Chebyshev polynomials for spectral filtering.
    
    Parameters
    ----------
    input_dim : int
        Input node feature dimension
    hidden_dims : list
        Hidden layer dimensions
    output_dim : int
        Output dimension
    K : int
        Order of Chebyshev polynomials
    
    Example
    -------
    >>> chebnet = ChebNet(input_dim=32, hidden_dims=[64], output_dim=7, K=3)
    >>> predictions = chebnet.forward(features, laplacian)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001, 0.01],
        'K': [2, 3, 5],
        'hidden_dims': [[32], [64, 32]],
    }
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 output_dim: int = 7,
                 K: int = 3,
                 **kwargs):
        
        self.hidden_dims = hidden_dims or [64]
        self.output_dim = output_dim
        self.K = K
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(output_dim,),
                        loss='cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build ChebNet layers."""
        input_dim = self.input_shape[0]
        
        self.layers = []
        in_dim = input_dim
        
        for h_dim in self.hidden_dims:
            layer = ChebConvLayer(in_dim, h_dim, self.K)
            self.layers.append(layer)
            in_dim = h_dim
        
        # Output layer
        self.output_layer = ChebConvLayer(in_dim, self.output_dim, self.K)
    
    def _forward(self, X: np.ndarray, A: np.ndarray = None,
                 training: bool = True) -> np.ndarray:
        """Forward pass."""
        if A is None:
            A = np.eye(X.shape[0])
        
        # Compute Laplacian
        L = compute_laplacian(A, normalized=True)
        
        H = X
        
        for layer in self.layers:
            H = layer.forward(H, L, training)
        
        H = self.output_layer.forward(H, L, training)
        
        # Softmax
        exp_H = np.exp(H - np.max(H, axis=-1, keepdims=True))
        return exp_H / np.sum(exp_H, axis=-1, keepdims=True)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through ChebNet."""
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


# Graph Pooling

class GlobalMeanPool:
    """Global mean pooling for graph-level representations."""
    
    def forward(self, X: np.ndarray, batch: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Pool node features to graph-level.
        
        Parameters
        ----------
        X : np.ndarray
            Node features (N, F)
        batch : np.ndarray, optional
            Batch indices for each node
        
        Returns
        -------
        graph_features : np.ndarray
            Graph-level features
        """
        if batch is None:
            return np.mean(X, axis=0, keepdims=True)
        
        # Pool per graph in batch
        unique_batches = np.unique(batch)
        pooled = []
        for b in unique_batches:
            mask = batch == b
            pooled.append(np.mean(X[mask], axis=0))
        return np.array(pooled)


class GlobalSumPool:
    """Global sum pooling for graph-level representations."""
    
    def forward(self, X: np.ndarray, batch: Optional[np.ndarray] = None) -> np.ndarray:
        if batch is None:
            return np.sum(X, axis=0, keepdims=True)
        
        unique_batches = np.unique(batch)
        pooled = []
        for b in unique_batches:
            mask = batch == b
            pooled.append(np.sum(X[mask], axis=0))
        return np.array(pooled)


class GlobalMaxPool:
    """Global max pooling for graph-level representations."""
    
    def forward(self, X: np.ndarray, batch: Optional[np.ndarray] = None) -> np.ndarray:
        if batch is None:
            return np.max(X, axis=0, keepdims=True)
        
        unique_batches = np.unique(batch)
        pooled = []
        for b in unique_batches:
            mask = batch == b
            pooled.append(np.max(X[mask], axis=0))
        return np.array(pooled)


# Factory Function

def create_gnn(architecture: str, input_dim: int, output_dim: int,
               **kwargs) -> BaseArchitecture:
    """
    Factory function to create graph neural networks.
    
    Parameters
    ----------
    architecture : str
        Architecture name: 'gnn', 'gcn', 'gat', 'graphsage', 'mpnn', 'gin', 'chebnet'
    input_dim : int
        Input node feature dimension
    output_dim : int
        Output dimension
    **kwargs
        Architecture-specific parameters
    
    Returns
    -------
    model : BaseArchitecture
        The requested GNN model
    
    Example
    -------
    >>> gcn = create_gnn('gcn', input_dim=1433, output_dim=7)
    >>> gat = create_gnn('gat', input_dim=1433, output_dim=7, n_heads=8)
    """
    architectures = {
        'gnn': GNN,
        'gcn': GCN,
        'gat': GAT,
        'graphsage': GraphSAGE,
        'sage': GraphSAGE,
        'mpnn': MPNN,
        'gin': GIN,
        'chebnet': ChebNet,
        'chebyshev': ChebNet,
    }
    
    arch_name = architecture.lower()
    if arch_name not in architectures:
        available = list(architectures.keys())
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {available}")
    
    return architectures[arch_name](input_dim=input_dim, output_dim=output_dim, **kwargs)
