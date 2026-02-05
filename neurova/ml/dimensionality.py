# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Dimensionality reduction algorithms

All implementations use NumPy only.
"""

import numpy as np
from typing import Optional
from neurova.core.errors import ValidationError


class PCA:
    """
    Principal Component Analysis (PCA)
    
    Examples:
        >>> pca = PCA(n_components=2)
        >>> X_reduced = pca.fit_transform(X)
    """
    
    def __init__(self, n_components: Optional[int] = None, whiten: bool = False):
        """
        Args:
            n_components: Number of components to keep
            whiten: Whether to whiten the data
        """
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """Fit PCA on data"""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # compute covariance matrix
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # select number of components
        if self.n_components is None:
            n_components = n_features
        else:
            n_components = min(self.n_components, n_features)
        
        # store components
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]
        
        # calculate explained variance ratio
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:n_components] / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to PC space"""
        if self.components_ is None:
            raise ValidationError('pca', 'not fitted', 'fitted PCA')
        
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T
        
        if self.whiten and self.explained_variance_ is not None:
            X_transformed /= np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform data back to original space"""
        if self.components_ is None:
            raise ValidationError('pca', 'not fitted', 'fitted PCA')
        
        X_result = X_transformed.copy()
        if self.whiten and self.explained_variance_ is not None:
            X_result = X_result * np.sqrt(self.explained_variance_)
        
        return X_result @ self.components_ + self.mean_


class LDA:
    """
    Linear Discriminant Analysis (LDA)
    
    Examples:
        >>> lda = LDA(n_components=2)
        >>> X_reduced = lda.fit_transform(X, y)
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Args:
            n_components: Number of components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDA':
        """Fit LDA on data"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # overall mean
        self.mean_ = X.mean(axis=0)
        
        # within-class scatter matrix
        S_W = np.zeros((n_features, n_features))
        
        # between-class scatter matrix
        S_B = np.zeros((n_features, n_features))
        
        for cls in classes:
            X_cls = X[y == cls]
            n_cls = X_cls.shape[0]
            
            # class mean
            mean_cls = X_cls.mean(axis=0)
            
            # within-class scatter
            X_centered = X_cls - mean_cls
            S_W += X_centered.T @ X_centered
            
            # between-class scatter
            mean_diff = (mean_cls - self.mean_).reshape(-1, 1)
            S_B += n_cls * (mean_diff @ mean_diff.T)
        
        # solve generalized eigenvalue problem
        # s_W^{-1} S_B
        try:
            S_W_inv = np.linalg.inv(S_W)
            matrix = S_W_inv @ S_B
        except np.linalg.LinAlgError:
            # if S_W is singular, use pseudo-inverse
            S_W_inv = np.linalg.pinv(S_W)
            matrix = S_W_inv @ S_B
        
        # eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # sort by eigenvalues (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        
        # select number of components
        if self.n_components is None:
            n_components = min(n_features, n_classes - 1)
        else:
            n_components = min(self.n_components, n_features, n_classes - 1)
        
        # store components
        self.components_ = eigenvectors[:, :n_components].T
        
        # calculate explained variance ratio
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:n_components] / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to LD space"""
        if self.components_ is None:
            raise ValidationError('lda', 'not fitted', 'fitted LDA')
        
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


class TSNE:
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE).
    
    Nonlinear dimensionality reduction for visualization.
    
    Parameters
    ----------
    n_components : int, default=2
        Dimension of embedded space
    perplexity : float, default=30.0
        Related to number of nearest neighbors
    early_exaggeration : float, default=12.0
        Controls tightness of clusters
    learning_rate : float, default=200.0
        Learning rate for gradient descent
    n_iter : int, default=1000
        Number of iterations
    random_state : int, optional
        Random state
        
    Attributes
    ----------
    embedding_ : ndarray
        Stored embedding
    kl_divergence_ : float
        Final KL divergence
        
    Examples
    --------
    >>> tsne = TSNE(n_components=2, perplexity=30)
    >>> X_embedded = tsne.fit_transform(X)
    """
    
    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 early_exaggeration: float = 12.0, learning_rate: float = 200.0,
                 n_iter: int = 1000, random_state: Optional[int] = None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.embedding_ = None
        self.kl_divergence_ = None
    
    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise squared Euclidean distances."""
        sum_X = np.sum(X ** 2, axis=1)
        D = sum_X + sum_X[:, np.newaxis] - 2 * X @ X.T
        return np.maximum(D, 0)
    
    def _compute_joint_probabilities(self, D: np.ndarray) -> np.ndarray:
        """Compute joint probabilities P_ij from distances."""
        n = D.shape[0]
        P = np.zeros((n, n))
        target_entropy = np.log(self.perplexity)
        
        for i in range(n):
            # Binary search for sigma
            Di = D[i, :]
            Di = np.delete(Di, i)  # Remove self-distance
            
            sigma_min, sigma_max = 1e-10, 1e10
            sigma = 1.0
            
            for _ in range(50):  # Max iterations for binary search
                exp_D = np.exp(-Di / (2 * sigma ** 2))
                sum_exp = np.sum(exp_D)
                
                if sum_exp == 0:
                    Pi = np.ones(n - 1) / (n - 1)
                else:
                    Pi = exp_D / sum_exp
                
                # Compute entropy
                entropy = -np.sum(Pi * np.log(Pi + 1e-10))
                
                if abs(entropy - target_entropy) < 1e-5:
                    break
                
                if entropy > target_entropy:
                    sigma_max = sigma
                    sigma = (sigma + sigma_min) / 2
                else:
                    sigma_min = sigma
                    sigma = (sigma + sigma_max) / 2
            
            # Insert probabilities (excluding i-th position)
            idx = np.arange(n) != i
            P[i, idx] = Pi
        
        # Symmetrize and normalize
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_q_distribution(self, Y: np.ndarray) -> tuple:
        """Compute student-t distribution Q_ij."""
        D = self._compute_pairwise_distances(Y)
        Q = 1.0 / (1.0 + D)
        np.fill_diagonal(Q, 0)
        sum_Q = np.sum(Q)
        Q = np.maximum(Q / sum_Q, 1e-12)
        return Q, D
    
    def _compute_gradient(self, P: np.ndarray, Q: np.ndarray, 
                          Y: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Compute gradient of KL divergence."""
        n = Y.shape[0]
        PQ_diff = P - Q
        
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum(PQ_diff[i, :, np.newaxis] * diff / 
                                 (1 + D[i, :, np.newaxis]), axis=0)
        
        return grad
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        rng = np.random.default_rng(self.random_state)
        
        # Compute pairwise distances and joint probabilities
        D = self._compute_pairwise_distances(X)
        P = self._compute_joint_probabilities(D)
        
        # Initialize embedding randomly
        Y = rng.normal(0, 1e-4, (n_samples, self.n_components))
        
        # Gradient descent with momentum
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)
        
        for iteration in range(self.n_iter):
            # Apply early exaggeration
            if iteration < 250:
                P_iter = P * self.early_exaggeration
            else:
                P_iter = P
            
            # Compute Q distribution
            Q, D_Y = self._compute_q_distribution(Y)
            
            # Compute gradient
            grad = self._compute_gradient(P_iter, Q, Y, D_Y)
            
            # Update gains
            gains = np.where(np.sign(grad) != np.sign(velocity),
                            gains + 0.2, gains * 0.8)
            gains = np.maximum(gains, 0.01)
            
            # Update velocity and position
            momentum = 0.5 if iteration < 250 else 0.8
            velocity = momentum * velocity - self.learning_rate * gains * grad
            Y = Y + velocity
            
            # Center embedding
            Y = Y - np.mean(Y, axis=0)
        
        # Compute final KL divergence
        Q, _ = self._compute_q_distribution(Y)
        self.kl_divergence_ = np.sum(P * np.log(P / Q))
        
        self.embedding_ = Y
        return Y
    
    def fit(self, X: np.ndarray) -> 'TSNE':
        """Fit the model."""
        self.fit_transform(X)
        return self


class Isomap:
    """
    Isometric Mapping (Isomap).
    
    Nonlinear dimensionality reduction using geodesic distances.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions
    n_neighbors : int, default=5
        Number of neighbors for graph construction
        
    Attributes
    ----------
    embedding_ : ndarray
        Transformed data
    dist_matrix_ : ndarray
        Geodesic distance matrix
        
    Examples
    --------
    >>> isomap = Isomap(n_components=2, n_neighbors=10)
    >>> X_embedded = isomap.fit_transform(X)
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.embedding_ = None
        self.dist_matrix_ = None
    
    def _compute_neighbors_graph(self, X: np.ndarray) -> np.ndarray:
        """Compute k-nearest neighbors distance graph."""
        n_samples = X.shape[0]
        
        # Compute all pairwise distances
        D = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        
        # Create neighbor graph (infinity for non-neighbors)
        graph = np.full((n_samples, n_samples), np.inf)
        
        for i in range(n_samples):
            # Find k nearest neighbors (excluding self)
            neighbors = np.argsort(D[i])[1:self.n_neighbors + 1]
            graph[i, neighbors] = D[i, neighbors]
            graph[neighbors, i] = D[i, neighbors]  # Symmetrize
        
        return graph
    
    def _floyd_warshall(self, graph: np.ndarray) -> np.ndarray:
        """Compute shortest paths using Floyd-Warshall algorithm."""
        dist = graph.copy()
        n = dist.shape[0]
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        return dist
    
    def _mds(self, D: np.ndarray) -> np.ndarray:
        """Classical MDS on distance matrix."""
        n = D.shape[0]
        
        # Double centering
        D_sq = D ** 2
        row_mean = np.mean(D_sq, axis=1, keepdims=True)
        col_mean = np.mean(D_sq, axis=0, keepdims=True)
        total_mean = np.mean(D_sq)
        
        B = -0.5 * (D_sq - row_mean - col_mean + total_mean)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top components
        eigenvalues = np.maximum(eigenvalues[:self.n_components], 0)
        Y = eigenvectors[:, :self.n_components] * np.sqrt(eigenvalues)
        
        return Y
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        X = np.asarray(X, dtype=np.float64)
        
        # Build neighbor graph
        graph = self._compute_neighbors_graph(X)
        
        # Compute geodesic distances
        self.dist_matrix_ = self._floyd_warshall(graph)
        
        # Check for disconnected components
        if np.any(np.isinf(self.dist_matrix_)):
            # Replace inf with large value
            max_dist = np.max(self.dist_matrix_[~np.isinf(self.dist_matrix_)])
            self.dist_matrix_[np.isinf(self.dist_matrix_)] = max_dist * 2
        
        # Apply MDS
        self.embedding_ = self._mds(self.dist_matrix_)
        
        return self.embedding_
    
    def fit(self, X: np.ndarray) -> 'Isomap':
        """Fit the model."""
        self.fit_transform(X)
        return self


class MDS:
    """
    Multidimensional Scaling.
    
    Finds a low-dimensional representation that preserves pairwise distances.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions
    metric : bool, default=True
        If True, use metric MDS (classical). If False, use non-metric.
    n_init : int, default=4
        Number of initializations for non-metric MDS
    max_iter : int, default=300
        Maximum iterations for non-metric MDS
    random_state : int, optional
        Random state
        
    Attributes
    ----------
    embedding_ : ndarray
        Transformed data
    stress_ : float
        Final stress value (for non-metric MDS)
        
    Examples
    --------
    >>> mds = MDS(n_components=2)
    >>> X_embedded = mds.fit_transform(X)
    """
    
    def __init__(self, n_components: int = 2, metric: bool = True,
                 n_init: int = 4, max_iter: int = 300,
                 random_state: Optional[int] = None):
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.embedding_ = None
        self.stress_ = None
    
    def _classical_mds(self, D: np.ndarray) -> np.ndarray:
        """Classical (metric) MDS."""
        n = D.shape[0]
        
        # Double centering
        D_sq = D ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D_sq @ H
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top components
        eigenvalues = np.maximum(eigenvalues[:self.n_components], 0)
        Y = eigenvectors[:, :self.n_components] * np.sqrt(eigenvalues)
        
        return Y
    
    def _compute_stress(self, D: np.ndarray, Y: np.ndarray) -> float:
        """Compute stress value."""
        D_Y = np.linalg.norm(Y[:, np.newaxis] - Y[np.newaxis, :], axis=2)
        return np.sqrt(np.sum((D - D_Y) ** 2) / np.sum(D ** 2))
    
    def _smacof(self, D: np.ndarray, rng: np.random.Generator) -> tuple:
        """SMACOF algorithm for non-metric MDS."""
        n = D.shape[0]
        
        # Initialize randomly
        Y = rng.standard_normal((n, self.n_components))
        
        for iteration in range(self.max_iter):
            # Compute current distances
            D_Y = np.linalg.norm(Y[:, np.newaxis] - Y[np.newaxis, :], axis=2)
            D_Y[D_Y == 0] = 1e-10
            
            # Compute B matrix
            B = -D / D_Y
            np.fill_diagonal(B, 0)
            np.fill_diagonal(B, -np.sum(B, axis=1))
            
            # Update Y
            Y_new = B @ Y / n
            
            # Check convergence
            if np.linalg.norm(Y_new - Y) < 1e-6:
                break
            
            Y = Y_new
        
        stress = self._compute_stress(D, Y)
        return Y, stress
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        X = np.asarray(X, dtype=np.float64)
        
        # Compute distance matrix
        D = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        
        if self.metric:
            self.embedding_ = self._classical_mds(D)
            self.stress_ = self._compute_stress(D, self.embedding_)
        else:
            rng = np.random.default_rng(self.random_state)
            
            best_Y = None
            best_stress = np.inf
            
            for _ in range(self.n_init):
                Y, stress = self._smacof(D, rng)
                if stress < best_stress:
                    best_stress = stress
                    best_Y = Y
            
            self.embedding_ = best_Y
            self.stress_ = best_stress
        
        return self.embedding_
    
    def fit(self, X: np.ndarray) -> 'MDS':
        """Fit the model."""
        self.fit_transform(X)
        return self


class LocallyLinearEmbedding:
    """
    Locally Linear Embedding (LLE).
    
    Preserves local linear structure in low dimensions.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions
    n_neighbors : int, default=5
        Number of neighbors
    reg : float, default=1e-3
        Regularization constant
        
    Attributes
    ----------
    embedding_ : ndarray
        Transformed data
        
    Examples
    --------
    >>> lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
    >>> X_embedded = lle.fit_transform(X)
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 5,
                 reg: float = 1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.embedding_ = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Find neighbors
        D = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        neighbors = np.argsort(D, axis=1)[:, 1:self.n_neighbors + 1]
        
        # Compute reconstruction weights
        W = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            Z = X[neighbors[i]] - X[i]
            C = Z @ Z.T
            C = C + self.reg * np.trace(C) * np.eye(self.n_neighbors)
            
            try:
                w = np.linalg.solve(C, np.ones(self.n_neighbors))
            except np.linalg.LinAlgError:
                w = np.ones(self.n_neighbors) / self.n_neighbors
            
            w = w / np.sum(w)
            W[i, neighbors[i]] = w
        
        # Compute embedding
        M = (np.eye(n_samples) - W).T @ (np.eye(n_samples) - W)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        
        # Skip the zero eigenvalue, take next n_components
        idx = np.argsort(eigenvalues)
        self.embedding_ = eigenvectors[:, idx[1:self.n_components + 1]]
        
        return self.embedding_
    
    def fit(self, X: np.ndarray) -> 'LocallyLinearEmbedding':
        """Fit the model."""
        self.fit_transform(X)
        return self


class SpectralEmbedding:
    """
    Spectral Embedding for nonlinear dimensionality reduction.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions
    affinity : str, default='nearest_neighbors'
        Affinity type: 'nearest_neighbors' or 'rbf'
    n_neighbors : int, default=5
        Number of neighbors (for nearest_neighbors)
    gamma : float, default=None
        RBF kernel coefficient (for rbf)
        
    Attributes
    ----------
    embedding_ : ndarray
        Transformed data
    affinity_matrix_ : ndarray
        Affinity matrix
        
    Examples
    --------
    >>> se = SpectralEmbedding(n_components=2, n_neighbors=10)
    >>> X_embedded = se.fit_transform(X)
    """
    
    def __init__(self, n_components: int = 2, affinity: str = 'nearest_neighbors',
                 n_neighbors: int = 5, gamma: float = None):
        self.n_components = n_components
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.embedding_ = None
        self.affinity_matrix_ = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Compute affinity matrix
        D = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        
        if self.affinity == 'rbf':
            gamma = self.gamma if self.gamma else 1.0 / X.shape[1]
            self.affinity_matrix_ = np.exp(-gamma * D ** 2)
        else:  # nearest_neighbors
            self.affinity_matrix_ = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                neighbors = np.argsort(D[i])[1:self.n_neighbors + 1]
                self.affinity_matrix_[i, neighbors] = 1
                self.affinity_matrix_[neighbors, i] = 1
        
        # Compute normalized Laplacian
        degrees = np.sum(self.affinity_matrix_, axis=1)
        D_sqrt_inv = np.diag(np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0))
        L = np.eye(n_samples) - D_sqrt_inv @ self.affinity_matrix_ @ D_sqrt_inv
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Take smallest eigenvalues (skip zero)
        idx = np.argsort(eigenvalues)
        self.embedding_ = eigenvectors[:, idx[1:self.n_components + 1]]
        
        return self.embedding_
    
    def fit(self, X: np.ndarray) -> 'SpectralEmbedding':
        """Fit the model."""
        self.fit_transform(X)
        return self


class TruncatedSVD:
    """
    Truncated Singular Value Decomposition (LSA).
    
    Dimensionality reduction using truncated SVD.
    Unlike PCA, this doesn't center the data.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of components
        
    Attributes
    ----------
    components_ : ndarray
        Computed components
    explained_variance_ : ndarray
        Variance explained by each component
    explained_variance_ratio_ : ndarray
        Ratio of variance explained
    singular_values_ : ndarray
        Singular values
        
    Examples
    --------
    >>> svd = TruncatedSVD(n_components=5)
    >>> X_reduced = svd.fit_transform(X)
    """
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
    
    def fit(self, X: np.ndarray) -> 'TruncatedSVD':
        """Fit the model."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Take top components
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = s[:self.n_components]
        
        # Explained variance
        self.explained_variance_ = (s[:self.n_components] ** 2) / (n_samples - 1)
        total_var = (s ** 2).sum() / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        return X_transformed @ self.components_


class FactorAnalysis:
    """
    Factor Analysis.
    
    Linear generative model with Gaussian latent factors.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of latent factors
    max_iter : int, default=1000
        Maximum EM iterations
    tol : float, default=1e-2
        Convergence tolerance
        
    Attributes
    ----------
    components_ : ndarray
        Factor loading matrix
    noise_variance_ : ndarray
        Noise variance for each feature
    mean_ : ndarray
        Feature means
        
    Examples
    --------
    >>> fa = FactorAnalysis(n_components=5)
    >>> X_reduced = fa.fit_transform(X)
    """
    
    def __init__(self, n_components: int = None, max_iter: int = 1000,
                 tol: float = 1e-2, random_state: Optional[int] = None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.components_ = None
        self.noise_variance_ = None
        self.mean_ = None
    
    def fit(self, X: np.ndarray) -> 'FactorAnalysis':
        """Fit the model using EM."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        rng = np.random.default_rng(self.random_state)
        
        n_components = self.n_components or n_features
        
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Initialize with PCA
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:n_components] * s[:n_components, np.newaxis] / np.sqrt(n_samples)
        self.noise_variance_ = np.var(X_centered, axis=0) - np.sum(self.components_ ** 2, axis=0)[:n_features]
        self.noise_variance_ = np.maximum(self.noise_variance_, 1e-10)
        
        # EM iterations
        for iteration in range(self.max_iter):
            old_ll = self._log_likelihood(X_centered)
            
            # E-step
            psi_inv = 1.0 / self.noise_variance_
            W = self.components_.T
            M = np.eye(n_components) + W.T * psi_inv @ W
            M_inv = np.linalg.inv(M)
            
            # Expected latent factors
            Ez = X_centered * psi_inv @ W @ M_inv
            
            # M-step
            Ezz = n_samples * M_inv + Ez.T @ Ez
            
            self.components_ = (np.linalg.solve(Ezz, Ez.T @ X_centered)).T
            self.noise_variance_ = np.mean(X_centered ** 2, axis=0) - \
                                   np.mean(X_centered * (Ez @ self.components_), axis=0)
            self.noise_variance_ = np.maximum(self.noise_variance_, 1e-10)
            
            # Check convergence
            new_ll = self._log_likelihood(X_centered)
            if abs(new_ll - old_ll) < self.tol:
                break
        
        self.components_ = self.components_.T
        return self
    
    def _log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood."""
        n_samples = X.shape[0]
        W = self.components_.T
        cov = W @ W.T + np.diag(self.noise_variance_)
        
        try:
            cov_inv = np.linalg.inv(cov)
            log_det = np.linalg.slogdet(cov)[1]
        except np.linalg.LinAlgError:
            return -np.inf
        
        ll = -0.5 * n_samples * log_det
        ll -= 0.5 * np.sum(X @ cov_inv * X)
        return ll
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to latent space."""
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        
        W = self.components_.T
        psi_inv = 1.0 / self.noise_variance_
        M = np.eye(self.components_.shape[0]) + W.T * psi_inv @ W
        M_inv = np.linalg.inv(M)
        
        return X_centered * psi_inv @ W @ M_inv
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


class NMF:
    """
    Non-negative Matrix Factorization.
    
    Finds W, H >= 0 such that X ≈ W @ H.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of components
    max_iter : int, default=200
        Maximum iterations
    tol : float, default=1e-4
        Convergence tolerance
    init : str, default='random'
        Initialization: 'random' or 'nndsvd'
    random_state : int, optional
        Random state
        
    Attributes
    ----------
    components_ : ndarray
        H matrix (n_components, n_features)
    reconstruction_err_ : float
        Frobenius norm of residual
        
    Examples
    --------
    >>> nmf = NMF(n_components=5)
    >>> W = nmf.fit_transform(X)
    >>> H = nmf.components_
    """
    
    def __init__(self, n_components: int = None, max_iter: int = 200,
                 tol: float = 1e-4, init: str = 'random',
                 random_state: Optional[int] = None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.components_ = None
        self.reconstruction_err_ = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return transformed data."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        n_components = self.n_components or min(n_samples, n_features)
        
        rng = np.random.default_rng(self.random_state)
        
        # Initialize W and H
        if self.init == 'nndsvd':
            # Use SVD-based initialization
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            W = np.abs(U[:, :n_components]) * np.sqrt(s[:n_components])
            H = np.abs(Vt[:n_components]) * np.sqrt(s[:n_components])[:, np.newaxis]
        else:
            avg = np.sqrt(np.mean(X) / n_components)
            W = rng.uniform(0, avg, (n_samples, n_components))
            H = rng.uniform(0, avg, (n_components, n_features))
        
        eps = 1e-10
        
        # Multiplicative update rules
        for iteration in range(self.max_iter):
            # Update H
            WtX = W.T @ X
            WtWH = W.T @ W @ H + eps
            H = H * WtX / WtWH
            
            # Update W
            XHt = X @ H.T
            WHHt = W @ H @ H.T + eps
            W = W * XHt / WHHt
            
            # Check convergence
            if iteration % 10 == 0:
                err = np.linalg.norm(X - W @ H)
                if hasattr(self, '_prev_err') and abs(self._prev_err - err) < self.tol:
                    break
                self._prev_err = err
        
        self.components_ = H
        self.reconstruction_err_ = np.linalg.norm(X - W @ H)
        
        return W
    
    def fit(self, X: np.ndarray) -> 'NMF':
        """Fit the model."""
        self.fit_transform(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_components = self.components_.shape[0]
        
        rng = np.random.default_rng(self.random_state)
        W = rng.uniform(0, 1, (n_samples, n_components))
        
        eps = 1e-10
        H = self.components_
        
        for _ in range(self.max_iter):
            XHt = X @ H.T
            WHHt = W @ H @ H.T + eps
            W = W * XHt / WHHt
        
        return W


class KernelPCA(PCA):
    """
    Kernel PCA for nonlinear dimensionality reduction.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of components
    kernel : str, default='rbf'
        Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    gamma : float, default=None
        Kernel coefficient for rbf, poly, sigmoid
    degree : int, default=3
        Degree for polynomial kernel
    coef0 : float, default=1
        Independent term for poly and sigmoid
        
    Attributes
    ----------
    lambdas_ : ndarray
        Eigenvalues of kernel matrix
    alphas_ : ndarray
        Eigenvectors of kernel matrix
        
    Examples
    --------
    >>> kpca = KernelPCA(n_components=2, kernel='rbf')
    >>> X_reduced = kpca.fit_transform(X)
    """
    
    def __init__(self, n_components: int = None, kernel: str = 'rbf',
                 gamma: float = None, degree: int = 3, coef0: float = 1):
        super().__init__(n_components=n_components)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.X_fit_ = None
        self.lambdas_ = None
        self.alphas_ = None
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """Compute kernel matrix."""
        if Y is None:
            Y = X
        
        if self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'poly':
            gamma = self.gamma if self.gamma else 1.0 / X.shape[1]
            return (gamma * X @ Y.T + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma else 1.0 / X.shape[1]
            XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
            YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
            D = XX + YY - 2 * X @ Y.T
            return np.exp(-gamma * D)
        elif self.kernel == 'sigmoid':
            gamma = self.gamma if self.gamma else 1.0 / X.shape[1]
            return np.tanh(gamma * X @ Y.T + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray) -> 'KernelPCA':
        """Fit Kernel PCA."""
        X = np.asarray(X, dtype=np.float64)
        self.X_fit_ = X
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self._compute_kernel(X)
        
        # Center kernel matrix
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        n_components = self.n_components or n_samples
        
        self.lambdas_ = eigenvalues[:n_components]
        self.alphas_ = eigenvectors[:, :n_components]
        
        # Normalize eigenvectors
        self.alphas_ = self.alphas_ / np.sqrt(np.maximum(self.lambdas_, 1e-10))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        X = np.asarray(X, dtype=np.float64)
        
        # Compute kernel with training data
        K = self._compute_kernel(X, self.X_fit_)
        
        # Center
        n_train = self.X_fit_.shape[0]
        K_train = self._compute_kernel(self.X_fit_)
        
        K_centered = K - np.mean(K_train, axis=0) - \
                     np.mean(K, axis=1, keepdims=True) + \
                     np.mean(K_train)
        
        return K_centered @ self.alphas_


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.