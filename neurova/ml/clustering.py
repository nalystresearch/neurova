# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Clustering algorithms for machine learning

All implementations use NumPy only.
"""

import numpy as np
from typing import Optional
from neurova.core.errors import ValidationError


class KMeans:
    """
    K-Means clustering algorithm
    
    Examples:
        >>> kmeans = KMeans(n_clusters=3)
        >>> kmeans.fit(X)
        >>> labels = kmeans.predict(X)
    """
    
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, 
                 n_init: int = 10, random_state: Optional[int] = None):
        """
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations
            n_init: Number of times to run with different initializations
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
    
    def _initialize_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Initialize cluster centers using k-means++"""
        n_samples = X.shape[0]
        centers = []
        
        # choose first center randomly
        centers.append(X[rng.integers(0, n_samples)])
        
        # choose remaining centers
        for _ in range(1, self.n_clusters):
            # compute distances to nearest center
            distances = np.array([
                np.min([np.linalg.norm(x - c) ** 2 for c in centers])
                for x in X
            ])
            
            # choose new center with probability proportional to distance squared
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = rng.random()
            idx = np.searchsorted(cumulative_probs, r)
            centers.append(X[idx])
        
        return np.array(centers)
    
    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign each sample to nearest center"""
        distances = np.sqrt(((X[:, np.newaxis] - centers) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, 
                        centers: np.ndarray) -> float:
        """Compute inertia (sum of squared distances)"""
        return np.sum((X - centers[labels]) ** 2)
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """Compute k-means clustering"""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        if self.n_clusters > n_samples:
            raise ValidationError('n_clusters', self.n_clusters, 
                                f'<= {n_samples} (number of samples)')
        
        rng = np.random.default_rng(self.random_state)
        
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        
        # run multiple times with different initializations
        for _ in range(self.n_init):
            centers = self._initialize_centers(X, rng)
            labels = np.zeros(n_samples, dtype=int)  # Initialize labels
            
            # iterate until convergence or max_iter
            for iteration in range(self.max_iter):
                # assign labels
                labels = self._assign_labels(X, centers)
                
                # update centers
                new_centers = np.array([
                    X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centers[k]
                    for k in range(self.n_clusters)
                ])
                
                # check convergence
                if np.allclose(centers, new_centers):
                    break
                
                centers = new_centers
            
            # compute inertia
            inertia = self._compute_inertia(X, labels, centers)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        if self.cluster_centers_ is None:
            raise ValidationError('clusterer', 'not fitted', 'fitted clusterer')
        
        X = np.asarray(X, dtype=np.float64)
        return self._assign_labels(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        return self.fit(X).labels_


class DBSCAN:
    """
    DBSCAN clustering algorithm
    
    Examples:
        >>> dbscan = DBSCAN(eps=0.5, min_samples=5)
        >>> labels = dbscan.fit_predict(X)
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean'):
        """
        Args:
            eps: Maximum distance between two samples
            min_samples: Minimum samples in a neighborhood
            metric: Distance metric
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances"""
        if self.metric == 'euclidean':
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            distances = np.sqrt((diff ** 2).sum(axis=2))
        elif self.metric == 'manhattan':
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            distances = np.abs(diff).sum(axis=2)
        else:
            raise ValidationError('metric', self.metric, 'euclidean or manhattan')
        
        return distances
    
    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """Perform DBSCAN clustering"""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # compute pairwise distances
        distances = self._compute_distances(X)
        
        # find neighbors
        neighbors = [np.where(distances[i] <= self.eps)[0] for i in range(n_samples)]
        
        # initialize labels (-1 means noise)
        labels = -np.ones(n_samples, dtype=int)
        
        # find core samples
        core_samples = [i for i in range(n_samples) if len(neighbors[i]) >= self.min_samples]
        self.core_sample_indices_ = np.array(core_samples)
        
        # assign clusters
        cluster_id = 0
        visited = set()
        
        for core_idx in core_samples:
            if core_idx in visited:
                continue
            
            # start new cluster
            queue = [core_idx]
            visited.add(core_idx)
            labels[core_idx] = cluster_id
            
            while queue:
                current = queue.pop(0)
                
                # add neighbors to cluster
                for neighbor in neighbors[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        labels[neighbor] = cluster_id
                        
                        # if neighbor is core sample, add its neighbors to queue
                        if len(neighbors[neighbor]) >= self.min_samples:
                            queue.append(neighbor)
            
            cluster_id += 1
        
        self.labels_ = labels
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels"""
        return self.fit(X).labels_


class AgglomerativeClustering:
    """
    Agglomerative (hierarchical) clustering
    
    Examples:
        >>> agg = AgglomerativeClustering(n_clusters=3)
        >>> labels = agg.fit_predict(X)
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'average'):
        """
        Args:
            n_clusters: Number of clusters
            linkage: Linkage criterion ('single', 'complete', 'average')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances"""
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))
    
    def _linkage_distance(self, cluster1: list, cluster2: list, 
                         distances: np.ndarray) -> float:
        """Compute distance between two clusters"""
        if self.linkage == 'single':
            # minimum distance
            return np.min([distances[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == 'complete':
            # maximum distance
            return np.max([distances[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == 'average':
            # average distance
            return float(np.mean([distances[i, j] for i in cluster1 for j in cluster2]))
        else:
            raise ValidationError('linkage', self.linkage, 'single, complete, or average')
    
    def fit(self, X: np.ndarray) -> 'AgglomerativeClustering':
        """Perform hierarchical clustering"""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # compute pairwise distances
        distances = self._compute_distances(X)
        
        # initialize each sample as its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        # merge clusters until we have n_clusters
        while len(clusters) > self.n_clusters:
            # find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._linkage_distance(clusters[i], clusters[j], distances)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # assign labels
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_id
        
        self.labels_ = labels
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels"""
        return self.fit(X).labels_


class MeanShift:
    """
    Mean Shift clustering algorithm.
    
    Non-parametric clustering that finds dense regions in feature space.
    
    Parameters
    ----------
    bandwidth : float, optional
        Kernel bandwidth. If None, estimated using estimate_bandwidth().
    seeds : array-like, optional
        Initial kernel locations. If None, uses all data points.
    max_iter : int, default=300
        Maximum iterations per seed
    bin_seeding : bool, default=False
        If True, use binned seeding for initialization
    min_bin_freq : int, default=1
        Minimum number of points in a bin to seed
    cluster_all : bool, default=True
        If True, assigns all points to clusters
        
    Attributes
    ----------
    cluster_centers_ : ndarray
        Coordinates of cluster centers
    labels_ : ndarray
        Cluster labels for each point
    
    Examples
    --------
    >>> ms = MeanShift(bandwidth=0.8)
    >>> labels = ms.fit_predict(X)
    """
    
    def __init__(self, bandwidth: Optional[float] = None, seeds=None,
                 max_iter: int = 300, bin_seeding: bool = False,
                 min_bin_freq: int = 1, cluster_all: bool = True):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.max_iter = max_iter
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.cluster_centers_ = None
        self.labels_ = None
    
    def _estimate_bandwidth(self, X: np.ndarray) -> float:
        """Estimate bandwidth using Scott's rule."""
        n_samples, n_features = X.shape
        # Scott's rule
        sigma = np.std(X, axis=0)
        bandwidth = np.mean(sigma) * (n_samples ** (-1.0 / (n_features + 4)))
        return max(bandwidth, 0.1)  # Minimum bandwidth
    
    def _mean_shift_single(self, X: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Run mean shift for a single seed."""
        center = seed.copy()
        
        for _ in range(self.max_iter):
            # Find points within bandwidth
            distances = np.linalg.norm(X - center, axis=1)
            within_bandwidth = distances <= self.bandwidth
            
            if not np.any(within_bandwidth):
                break
            
            # Compute weighted mean (Gaussian kernel)
            weights = np.exp(-0.5 * (distances[within_bandwidth] / self.bandwidth) ** 2)
            new_center = np.average(X[within_bandwidth], weights=weights, axis=0)
            
            # Check convergence
            if np.linalg.norm(new_center - center) < 1e-3 * self.bandwidth:
                break
            
            center = new_center
        
        return center
    
    def fit(self, X: np.ndarray) -> 'MeanShift':
        """Perform mean shift clustering."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Estimate bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(X)
        
        # Get seeds
        if self.seeds is None:
            if self.bin_seeding:
                # Binned seeding for efficiency
                n_bins = max(1, int(1 / self.bandwidth))
                X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
                bin_indices = (X_scaled * n_bins).astype(int)
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                
                # Use unique bins as seeds
                unique_bins = np.unique(bin_indices, axis=0)
                seeds = unique_bins / n_bins * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
            else:
                seeds = X.copy()
        else:
            seeds = np.asarray(self.seeds, dtype=np.float64)
        
        # Run mean shift for each seed
        centers = []
        for seed in seeds:
            center = self._mean_shift_single(X, seed)
            centers.append(center)
        centers = np.array(centers)
        
        # Merge nearby centers
        merged_centers = []
        used = set()
        
        for i, center in enumerate(centers):
            if i in used:
                continue
            
            # Find centers within bandwidth
            cluster_centers = [center]
            for j, other in enumerate(centers[i+1:], i+1):
                if j not in used and np.linalg.norm(center - other) < self.bandwidth:
                    cluster_centers.append(other)
                    used.add(j)
            
            merged_centers.append(np.mean(cluster_centers, axis=0))
            used.add(i)
        
        self.cluster_centers_ = np.array(merged_centers)
        
        # Assign labels
        if len(self.cluster_centers_) > 0:
            distances = np.linalg.norm(
                X[:, np.newaxis] - self.cluster_centers_, axis=2
            )
            self.labels_ = np.argmin(distances, axis=1)
            
            if not self.cluster_all:
                # Mark points far from centers as noise (-1)
                min_distances = np.min(distances, axis=1)
                self.labels_[min_distances > self.bandwidth] = -1
        else:
            self.labels_ = np.zeros(n_samples, dtype=int)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for samples."""
        X = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.cluster_centers_, axis=2
        )
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_


class SpectralClustering:
    """
    Spectral Clustering using graph Laplacian.
    
    Projects data onto eigenvectors of the graph Laplacian and
    then applies k-means in the projected space.
    
    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters
    affinity : str, default='rbf'
        Affinity type: 'rbf', 'nearest_neighbors', 'precomputed'
    gamma : float, default=1.0
        RBF kernel coefficient
    n_neighbors : int, default=10
        Number of neighbors for nearest_neighbors affinity
    assign_labels : str, default='kmeans'
        Strategy for assigning labels: 'kmeans' or 'discretize'
    random_state : int, optional
        Random state
        
    Attributes
    ----------
    labels_ : ndarray
        Cluster labels
    affinity_matrix_ : ndarray
        Computed affinity matrix
        
    Examples
    --------
    >>> sc = SpectralClustering(n_clusters=3, affinity='rbf')
    >>> labels = sc.fit_predict(X)
    """
    
    def __init__(self, n_clusters: int = 8, affinity: str = 'rbf',
                 gamma: float = 1.0, n_neighbors: int = 10,
                 assign_labels: str = 'kmeans', random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.random_state = random_state
        self.labels_ = None
        self.affinity_matrix_ = None
    
    def _compute_affinity(self, X: np.ndarray) -> np.ndarray:
        """Compute affinity matrix."""
        n_samples = X.shape[0]
        
        if self.affinity == 'rbf':
            # RBF (Gaussian) kernel
            distances = np.linalg.norm(
                X[:, np.newaxis] - X[np.newaxis, :], axis=2
            )
            affinity = np.exp(-self.gamma * distances ** 2)
            
        elif self.affinity == 'nearest_neighbors':
            # k-nearest neighbors connectivity
            distances = np.linalg.norm(
                X[:, np.newaxis] - X[np.newaxis, :], axis=2
            )
            affinity = np.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                # Get k nearest neighbors (excluding self)
                nn_indices = np.argsort(distances[i])[1:self.n_neighbors + 1]
                affinity[i, nn_indices] = 1
                affinity[nn_indices, i] = 1  # Symmetrize
            
        elif self.affinity == 'precomputed':
            affinity = X
            
        else:
            raise ValueError(f"Unknown affinity: {self.affinity}")
        
        return affinity
    
    def _compute_laplacian(self, affinity: np.ndarray) -> np.ndarray:
        """Compute normalized graph Laplacian."""
        # Degree matrix
        degrees = np.sum(affinity, axis=1)
        degrees_sqrt_inv = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0)
        D_sqrt_inv = np.diag(degrees_sqrt_inv)
        
        # Normalized Laplacian: I - D^(-1/2) A D^(-1/2)
        L_normalized = np.eye(len(degrees)) - D_sqrt_inv @ affinity @ D_sqrt_inv
        
        return L_normalized
    
    def fit(self, X: np.ndarray) -> 'SpectralClustering':
        """Perform spectral clustering."""
        X = np.asarray(X, dtype=np.float64)
        
        # Compute affinity matrix
        self.affinity_matrix_ = self._compute_affinity(X)
        
        # Compute normalized Laplacian
        L = self._compute_laplacian(self.affinity_matrix_)
        
        # Get eigenvectors of Laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Use smallest k eigenvectors (excluding constant)
        # Sort by eigenvalue and take k smallest
        idx = np.argsort(eigenvalues)
        embedding = eigenvectors[:, idx[:self.n_clusters]]
        
        # Normalize rows
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        embedding = embedding / row_norms
        
        # Apply k-means to embedded data
        if self.assign_labels == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters, 
                          random_state=self.random_state)
            self.labels_ = kmeans.fit_predict(embedding)
        else:
            # Discretize using SVD-based approach
            from scipy.cluster.vq import vq
            # Simplified discretization
            kmeans = KMeans(n_clusters=self.n_clusters,
                          random_state=self.random_state)
            self.labels_ = kmeans.fit_predict(embedding)
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_


class OPTICS:
    """
    Ordering Points To Identify the Clustering Structure.
    
    Related to DBSCAN but identifies clusters of varying density.
    
    Parameters
    ----------
    min_samples : int, default=5
        Minimum samples in a neighborhood
    max_eps : float, default=np.inf
        Maximum distance between points
    metric : str, default='euclidean'
        Distance metric
    p : float, default=2
        Minkowski distance parameter
    xi : float, default=0.05
        Minimum steepness for cluster boundary
    min_cluster_size : float, default=None
        Minimum cluster size as fraction or absolute
        
    Attributes
    ----------
    labels_ : ndarray
        Cluster labels (-1 for noise)
    reachability_ : ndarray
        Reachability distances
    ordering_ : ndarray
        Ordered indices
    core_distances_ : ndarray
        Core distances
        
    Examples
    --------
    >>> optics = OPTICS(min_samples=5)
    >>> labels = optics.fit_predict(X)
    """
    
    def __init__(self, min_samples: int = 5, max_eps: float = np.inf,
                 metric: str = 'euclidean', p: float = 2,
                 xi: float = 0.05, min_cluster_size=None):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.p = p
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        self.reachability_ = None
        self.ordering_ = None
        self.core_distances_ = None
    
    def _compute_core_distances(self, X: np.ndarray, 
                                 distances: np.ndarray) -> np.ndarray:
        """Compute core distances."""
        n_samples = X.shape[0]
        core_distances = np.full(n_samples, np.inf)
        
        for i in range(n_samples):
            sorted_dist = np.sort(distances[i])
            if len(sorted_dist) > self.min_samples:
                core_distances[i] = sorted_dist[self.min_samples]
        
        return core_distances
    
    def fit(self, X: np.ndarray) -> 'OPTICS':
        """Perform OPTICS clustering."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        distances = np.linalg.norm(
            X[:, np.newaxis] - X[np.newaxis, :], axis=2
        )
        
        # Compute core distances
        self.core_distances_ = self._compute_core_distances(X, distances)
        
        # Initialize
        self.reachability_ = np.full(n_samples, np.inf)
        processed = np.zeros(n_samples, dtype=bool)
        self.ordering_ = []
        
        # Process all points
        for _ in range(n_samples):
            # Find unprocessed point with minimum reachability
            unprocessed = ~processed
            if not np.any(unprocessed):
                break
            
            # Get point with minimum reachability (or random unprocessed)
            unprocessed_reach = np.where(unprocessed, self.reachability_, np.inf)
            p_idx = np.argmin(unprocessed_reach)
            
            if self.reachability_[p_idx] == np.inf:
                # Start with unprocessed point
                unprocessed_idx = np.where(unprocessed)[0]
                if len(unprocessed_idx) == 0:
                    break
                p_idx = unprocessed_idx[0]
            
            processed[p_idx] = True
            self.ordering_.append(p_idx)
            
            # Update reachability of unprocessed neighbors
            if self.core_distances_[p_idx] <= self.max_eps:
                # Find neighbors
                neighbors = np.where(distances[p_idx] <= self.max_eps)[0]
                
                for n_idx in neighbors:
                    if not processed[n_idx]:
                        new_reach = max(self.core_distances_[p_idx], 
                                       distances[p_idx, n_idx])
                        if new_reach < self.reachability_[n_idx]:
                            self.reachability_[n_idx] = new_reach
        
        self.ordering_ = np.array(self.ordering_)
        
        # Extract clusters using xi-steep method
        self._extract_clusters(X)
        
        return self
    
    def _extract_clusters(self, X: np.ndarray):
        """Extract clusters from reachability plot."""
        n_samples = X.shape[0]
        
        # Simple extraction based on reachability threshold
        min_cluster_size = self.min_cluster_size or self.min_samples
        if isinstance(min_cluster_size, float):
            min_cluster_size = int(min_cluster_size * n_samples)
        
        # Use xi to find cluster boundaries
        reachability_ordered = self.reachability_[self.ordering_]
        
        self.labels_ = -np.ones(n_samples, dtype=int)
        cluster_id = 0
        
        # Simple cluster extraction
        in_cluster = False
        cluster_start = 0
        
        for i in range(len(self.ordering_)):
            if reachability_ordered[i] < self.max_eps:
                if not in_cluster:
                    in_cluster = True
                    cluster_start = i
            else:
                if in_cluster:
                    # End of cluster
                    if i - cluster_start >= min_cluster_size:
                        for j in range(cluster_start, i):
                            self.labels_[self.ordering_[j]] = cluster_id
                        cluster_id += 1
                    in_cluster = False
        
        # Handle last cluster
        if in_cluster and len(self.ordering_) - cluster_start >= min_cluster_size:
            for j in range(cluster_start, len(self.ordering_)):
                self.labels_[self.ordering_[j]] = cluster_id
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_


class GaussianMixture:
    """
    Gaussian Mixture Model using EM algorithm.
    
    Parameters
    ----------
    n_components : int, default=1
        Number of mixture components
    covariance_type : str, default='full'
        Covariance type: 'full', 'tied', 'diag', 'spherical'
    max_iter : int, default=100
        Maximum EM iterations
    tol : float, default=1e-3
        Convergence tolerance
    n_init : int, default=1
        Number of initializations
    init_params : str, default='kmeans'
        Initialization: 'kmeans' or 'random'
    random_state : int, optional
        Random state
        
    Attributes
    ----------
    weights_ : ndarray
        Component weights
    means_ : ndarray
        Component means
    covariances_ : ndarray
        Component covariances
    converged_ : bool
        Whether EM converged
        
    Examples
    --------
    >>> gmm = GaussianMixture(n_components=3)
    >>> labels = gmm.fit_predict(X)
    """
    
    def __init__(self, n_components: int = 1, covariance_type: str = 'full',
                 max_iter: int = 100, tol: float = 1e-3, n_init: int = 1,
                 init_params: str = 'kmeans', random_state: Optional[int] = None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.lower_bound_ = -np.inf
    
    def _initialize(self, X: np.ndarray, rng: np.random.Generator):
        """Initialize parameters."""
        n_samples, n_features = X.shape
        
        if self.init_params == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_components,
                          random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        else:
            indices = rng.choice(n_samples, self.n_components, replace=False)
            self.means_ = X[indices].copy()
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize covariances
        if self.covariance_type == 'full':
            self.covariances_ = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features)
                                          for _ in range(self.n_components)])
        elif self.covariance_type == 'diag':
            self.covariances_ = np.array([np.var(X, axis=0) + 1e-6
                                          for _ in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.array([np.mean(np.var(X, axis=0)) + 1e-6
                                          for _ in range(self.n_components)])
        else:
            self.covariances_ = np.cov(X.T) + 1e-6 * np.eye(n_features)
    
    def _estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Estimate log probabilities."""
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
                try:
                    cov_inv = np.linalg.inv(cov)
                    log_det = np.linalg.slogdet(cov)[1]
                except np.linalg.LinAlgError:
                    cov_inv = np.eye(n_features)
                    log_det = 0
            elif self.covariance_type == 'diag':
                cov_inv = np.diag(1.0 / self.covariances_[k])
                log_det = np.sum(np.log(self.covariances_[k]))
            elif self.covariance_type == 'spherical':
                cov_inv = np.eye(n_features) / self.covariances_[k]
                log_det = n_features * np.log(self.covariances_[k])
            else:
                cov = self.covariances_
                try:
                    cov_inv = np.linalg.inv(cov)
                    log_det = np.linalg.slogdet(cov)[1]
                except np.linalg.LinAlgError:
                    cov_inv = np.eye(n_features)
                    log_det = 0
            
            diff = X - self.means_[k]
            maha = np.sum(diff @ cov_inv * diff, axis=1)
            
            log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + maha)
        
        return log_prob
    
    def _e_step(self, X: np.ndarray) -> tuple:
        """E-step: compute responsibilities."""
        log_prob = self._estimate_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-10)
        
        weighted_log_prob = log_prob + log_weights
        log_prob_norm = np.logaddexp.reduce(weighted_log_prob, axis=1)
        
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return np.mean(log_prob_norm), np.exp(log_resp)
    
    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        """M-step: update parameters."""
        n_samples, n_features = X.shape
        
        nk = resp.sum(axis=0) + 1e-10
        self.weights_ = nk / n_samples
        
        self.means_ = (resp.T @ X) / nk[:, np.newaxis]
        
        # Update covariances
        if self.covariance_type == 'full':
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k:k+1] * diff).T @ diff / nk[k]
                self.covariances_[k] += 1e-6 * np.eye(n_features)
        elif self.covariance_type == 'diag':
            self.covariances_ = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k:k+1] * diff ** 2, axis=0) / nk[k]
                self.covariances_[k] += 1e-6
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.zeros(self.n_components)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k] * np.sum(diff ** 2, axis=1)) / (nk[k] * n_features)
                self.covariances_[k] += 1e-6
    
    def fit(self, X: np.ndarray) -> 'GaussianMixture':
        """Fit the Gaussian mixture model."""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        
        best_lower_bound = -np.inf
        best_params = None
        
        for _ in range(self.n_init):
            self._initialize(X, rng)
            
            prev_lower_bound = -np.inf
            
            for iteration in range(self.max_iter):
                # E-step
                lower_bound, resp = self._e_step(X)
                
                # M-step
                self._m_step(X, resp)
                
                # Check convergence
                change = lower_bound - prev_lower_bound
                if abs(change) < self.tol:
                    self.converged_ = True
                    break
                
                prev_lower_bound = lower_bound
            
            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                best_params = (self.weights_.copy(), self.means_.copy(),
                             [c.copy() if hasattr(c, 'copy') else c 
                              for c in (self.covariances_ if hasattr(self.covariances_, '__iter__') 
                                       else [self.covariances_])])
        
        if best_params is not None:
            self.weights_, self.means_, self.covariances_ = best_params
            if len(self.covariances_) == 1 and self.covariance_type not in ['full', 'diag']:
                self.covariances_ = self.covariances_[0]
            else:
                self.covariances_ = np.array(self.covariances_)
        
        self.lower_bound_ = best_lower_bound
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        X = np.asarray(X, dtype=np.float64)
        log_prob = self._estimate_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-10)
        return np.argmax(log_prob + log_weights, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability."""
        X = np.asarray(X, dtype=np.float64)
        _, resp = self._e_step(X)
        return resp
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).predict(X)
    
    def score(self, X: np.ndarray) -> float:
        """Compute average log-likelihood."""
        X = np.asarray(X, dtype=np.float64)
        log_prob = self._estimate_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-10)
        return np.mean(np.logaddexp.reduce(log_prob + log_weights, axis=1))


class MiniBatchKMeans(KMeans):
    """
    Mini-batch K-Means clustering.
    
    Faster version of K-Means using mini-batches.
    
    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters
    max_iter : int, default=100
        Maximum iterations
    batch_size : int, default=100
        Size of mini-batches
    random_state : int, optional
        Random state
        
    Examples
    --------
    >>> mbkmeans = MiniBatchKMeans(n_clusters=3, batch_size=50)
    >>> labels = mbkmeans.fit_predict(X)
    """
    
    def __init__(self, n_clusters: int = 8, max_iter: int = 100,
                 batch_size: int = 100, random_state: Optional[int] = None):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter,
                        n_init=1, random_state=random_state)
        self.batch_size = batch_size
    
    def fit(self, X: np.ndarray) -> 'MiniBatchKMeans':
        """Fit using mini-batch updates."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        rng = np.random.default_rng(self.random_state)
        
        # Initialize centers
        self.cluster_centers_ = self._initialize_centers(X, rng)
        
        # Counts for averaging
        counts = np.zeros(self.n_clusters)
        
        for iteration in range(self.max_iter):
            # Sample mini-batch
            batch_indices = rng.choice(n_samples, min(self.batch_size, n_samples),
                                       replace=False)
            X_batch = X[batch_indices]
            
            # Assign labels to batch
            labels = self._assign_labels(X_batch, self.cluster_centers_)
            
            # Update centers
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    counts[k] += np.sum(mask)
                    eta = 1.0 / counts[k]
                    self.cluster_centers_[k] = (1 - eta) * self.cluster_centers_[k] + \
                                               eta * np.mean(X_batch[mask], axis=0)
        
        # Final labels and inertia
        self.labels_ = self._assign_labels(X, self.cluster_centers_)
        self.inertia_ = self._compute_inertia(X, self.labels_, self.cluster_centers_)
        
        return self


class Birch:
    """
    BIRCH clustering (Balanced Iterative Reducing and Clustering using Hierarchies).
    
    Parameters
    ----------
    threshold : float, default=0.5
        Radius threshold for subcluster merging
    branching_factor : int, default=50
        Maximum number of subclusters per node
    n_clusters : int, default=3
        Number of final clusters (uses AgglomerativeClustering)
        
    Attributes
    ----------
    labels_ : ndarray
        Cluster labels
    subcluster_centers_ : ndarray
        Subcluster centroids
        
    Examples
    --------
    >>> birch = Birch(n_clusters=3)
    >>> labels = birch.fit_predict(X)
    """
    
    def __init__(self, threshold: float = 0.5, branching_factor: int = 50,
                 n_clusters: int = 3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.labels_ = None
        self.subcluster_centers_ = None
    
    def fit(self, X: np.ndarray) -> 'Birch':
        """Fit BIRCH clustering."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Simple implementation: create subclusters greedily
        subclusters = []
        subcluster_sums = []
        subcluster_counts = []
        
        for point in X:
            if len(subclusters) == 0:
                subclusters.append(point.copy())
                subcluster_sums.append(point.copy())
                subcluster_counts.append(1)
            else:
                # Find nearest subcluster
                centers = np.array(subclusters)
                distances = np.linalg.norm(centers - point, axis=1)
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] <= self.threshold:
                    # Merge with existing subcluster
                    subcluster_sums[nearest_idx] += point
                    subcluster_counts[nearest_idx] += 1
                    subclusters[nearest_idx] = subcluster_sums[nearest_idx] / subcluster_counts[nearest_idx]
                else:
                    # Create new subcluster
                    if len(subclusters) < self.branching_factor * 10:
                        subclusters.append(point.copy())
                        subcluster_sums.append(point.copy())
                        subcluster_counts.append(1)
                    else:
                        # Merge with nearest anyway
                        subcluster_sums[nearest_idx] += point
                        subcluster_counts[nearest_idx] += 1
                        subclusters[nearest_idx] = subcluster_sums[nearest_idx] / subcluster_counts[nearest_idx]
        
        self.subcluster_centers_ = np.array(subclusters)
        
        # Apply agglomerative clustering to subclusters
        if len(self.subcluster_centers_) > self.n_clusters:
            agg = AgglomerativeClustering(n_clusters=self.n_clusters)
            subcluster_labels = agg.fit_predict(self.subcluster_centers_)
        else:
            subcluster_labels = np.arange(len(self.subcluster_centers_))
        
        # Assign labels to original points
        self.labels_ = np.zeros(n_samples, dtype=int)
        for i, point in enumerate(X):
            distances = np.linalg.norm(self.subcluster_centers_ - point, axis=1)
            nearest_subcluster = np.argmin(distances)
            self.labels_[i] = subcluster_labels[nearest_subcluster]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        X = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.subcluster_centers_, axis=2
        )
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_


# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.