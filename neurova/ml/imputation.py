# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Advanced preprocessing and feature engineering."""

from __future__ import annotations
import numpy as np
from typing import Optional, Literal


class RobustScaler:
    """Scale features using statistics robust to outliers.
    
    Uses median and interquartile range instead of mean and std.
    
    Args:
        with_centering: Whether to center data
        with_scaling: Whether to scale data
        quantile_range: Quantile range for IQR calculation
        
    Examples:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> RobustScaler:
        """Compute median and IQR."""
        X = np.asarray(X)
        
        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        
        if self.with_scaling:
            q_min, q_max = self.quantile_range
            quantiles = np.percentile(X, [q_min, q_max], axis=0)
            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_[self.scale_ == 0] = 1.0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features using median and IQR."""
        X = np.asarray(X).copy()
        
        if self.with_centering and self.center_ is not None:
            X -= self.center_
        
        if self.with_scaling and self.scale_ is not None:
            X /= self.scale_
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


class MaxAbsScaler:
    """Scale features by maximum absolute value.
    
    Scales data to [-1, 1] range. Sparse data friendly.
    
    Examples:
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(self):
        self.max_abs_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> MaxAbsScaler:
        """Compute max absolute value."""
        X = np.asarray(X)
        self.max_abs_ = np.max(np.abs(X), axis=0)
        self.scale_ = self.max_abs_.copy()
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale by max absolute value."""
        X = np.asarray(X)
        if self.scale_ is None:
            raise RuntimeError("Scaler not fitted")
        return X / self.scale_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


class QuantileTransformer:
    """Transform features using quantiles.
    
    Maps data to uniform or normal distribution.
    
    Args:
        n_quantiles: Number of quantiles
        output_distribution: "uniform" or "normal"
        
    Examples:
        qt = QuantileTransformer(output_distribution="normal")
        X_transformed = qt.fit_transform(X)
    """
    
    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
    ):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        
        self.quantiles_: Optional[np.ndarray] = None
        self.references_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> QuantileTransformer:
        """Compute quantiles."""
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # compute quantiles for each feature
        quantile_levels = np.linspace(0, 100, self.n_quantiles)
        self.quantiles_ = np.percentile(X, quantile_levels, axis=0)
        
        # compute reference distribution
        if self.output_distribution == "uniform":
            self.references_ = np.linspace(0, 1, self.n_quantiles)
        else:  # normal
            # standard normal quantiles
            from neurova.ml.stats import _norm_cdf
            uniform = np.linspace(0.001, 0.999, self.n_quantiles)
            # inverse CDF (approximation)
            self.references_ = self._norm_ppf(uniform)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using quantiles."""
        X = np.asarray(X)
        
        if self.quantiles_ is None or self.references_ is None:
            raise RuntimeError("Transformer not fitted")
        
        X_transformed = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            # interpolate to find quantile
            X_transformed[:, i] = np.interp(
                X[:, i],
                self.quantiles_[:, i],
                self.references_,
            )
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)
    
    def _norm_ppf(self, q: np.ndarray) -> np.ndarray:
        """Approximate inverse normal CDF."""
        # rational approximation
        return np.sqrt(2) * self._erfinv(2 * q - 1)
    
    def _erfinv(self, x: np.ndarray) -> np.ndarray:
        """Inverse error function (approximation)."""
        # simple approximation
        a = 0.147
        b = 2 / (np.pi * a)
        c = np.log(1 - x**2)
        
        sign = np.sign(x)
        v1 = b + c / 2
        v2 = c / a
        
        return sign * np.sqrt(np.sqrt(v1**2 - v2) - v1)


class PowerTransformer:
    """Apply power transform for normality.
    
    Supports Box-Cox and Yeo-Johnson transformations.
    
    Args:
        method: "yeo-johnson" or "box-cox"
        standardize: Whether to standardize after transform
        
    Examples:
        pt = PowerTransformer(method="yeo-johnson")
        X_transformed = pt.fit_transform(X)
    """
    
    def __init__(
        self,
        method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
        standardize: bool = True,
    ):
        self.method = method
        self.standardize = standardize
        
        self.lambdas_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> PowerTransformer:
        """Fit power transformer."""
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        self.lambdas_ = np.zeros(n_features)
        
        # find optimal lambda for each feature
        for i in range(n_features):
            self.lambdas_[i] = self._optimize_lambda(X[:, i])
        
        # transform and compute statistics if standardizing
        if self.standardize:
            X_transformed = self._transform_with_lambdas(X, self.lambdas_)
            self.mean_ = np.mean(X_transformed, axis=0)
            self.std_ = np.std(X_transformed, axis=0)
            self.std_[self.std_ == 0] = 1.0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply power transformation."""
        X = np.asarray(X)
        
        if self.lambdas_ is None:
            raise RuntimeError("Transformer not fitted")
        
        X_transformed = self._transform_with_lambdas(X, self.lambdas_)
        
        if self.standardize and self.mean_ is not None and self.std_ is not None:
            X_transformed = (X_transformed - self.mean_) / self.std_
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)
    
    def _optimize_lambda(self, x: np.ndarray) -> float:
        """Find optimal lambda using grid search."""
        lambdas = np.linspace(-2, 2, 41)
        best_lambda = 0.0
        best_score = -np.inf
        
        for lam in lambdas:
            x_trans = self._transform_single(x, lam)
            # score based on normality (simplified: use variance)
            score = -np.var(x_trans)
            if score > best_score:
                best_score = score
                best_lambda = lam
        
        return best_lambda
    
    def _transform_single(self, x: np.ndarray, lam: float) -> np.ndarray:
        """Transform single feature."""
        if self.method == "box-cox":
            # box-Cox (requires positive data)
            x = np.clip(x, 1e-10, None)
            if abs(lam) < 1e-10:
                return np.log(x)
            else:
                return (np.power(x, lam) - 1) / lam
        else:  # yeo-johnson
            # yeo-Johnson (works with any data)
            x_transformed = np.zeros_like(x)
            
            pos_mask = x >= 0
            neg_mask = ~pos_mask
            
            # positive values
            if np.any(pos_mask):
                if abs(lam) < 1e-10:
                    x_transformed[pos_mask] = np.log1p(x[pos_mask])
                else:
                    x_transformed[pos_mask] = (np.power(x[pos_mask] + 1, lam) - 1) / lam
            
            # negative values
            if np.any(neg_mask):
                if abs(lam - 2) < 1e-10:
                    x_transformed[neg_mask] = -np.log1p(-x[neg_mask])
                else:
                    x_transformed[neg_mask] = -(np.power(-x[neg_mask] + 1, 2 - lam) - 1) / (2 - lam)
            
            return x_transformed
    
    def _transform_with_lambdas(self, X: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
        """Transform all features with their lambdas."""
        X_transformed = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_transformed[:, i] = self._transform_single(X[:, i], lambdas[i])
        return X_transformed


class SimpleImputer:
    """Impute missing values.
    
    Args:
        strategy: "mean", "median", "most_frequent", or "constant"
        fill_value: Value to use when strategy="constant"
        
    Examples:
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
    """
    
    def __init__(
        self,
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
        fill_value: Optional[float] = None,
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        
        self.statistics_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> SimpleImputer:
        """Compute imputation statistics."""
        X = np.asarray(X)
        
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            # most frequent value per feature
            self.statistics_ = np.array([
                self._most_frequent(X[:, i]) for i in range(X.shape[1])
            ])
        elif self.strategy == "constant":
            if self.fill_value is None:
                self.fill_value = 0.0
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values."""
        X = np.asarray(X, dtype=np.float64).copy()
        
        if self.statistics_ is None:
            raise RuntimeError("Imputer not fitted")
        
        # replace NaN with statistics
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = self.statistics_[i]
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)
    
    def _most_frequent(self, x: np.ndarray) -> float:
        """Get most frequent value."""
        x_valid = x[~np.isnan(x)]
        if len(x_valid) == 0:
            return 0.0
        values, counts = np.unique(x_valid, return_counts=True)
        return values[np.argmax(counts)]


class KNNImputer:
    """Impute using k-Nearest Neighbors.
    
    Args:
        n_neighbors: Number of neighbors to use
        weights: "uniform" or "distance"
        
    Examples:
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X)
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        
        self.X_fit_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> KNNImputer:
        """Store training data."""
        self.X_fit_ = np.asarray(X, dtype=np.float64)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute using KNN."""
        X = np.asarray(X, dtype=np.float64).copy()
        
        if self.X_fit_ is None:
            raise RuntimeError("Imputer not fitted")
        
        # for each sample with missing values
        for i in range(X.shape[0]):
            if np.any(np.isnan(X[i])):
                # find k nearest neighbors (ignoring NaN in distance)
                distances = self._nan_euclidean_distances(X[i], self.X_fit_)
                nearest_idx = np.argsort(distances)[:self.n_neighbors]
                
                # impute each missing feature
                for j in range(X.shape[1]):
                    if np.isnan(X[i, j]):
                        neighbor_values = self.X_fit_[nearest_idx, j]
                        neighbor_values = neighbor_values[~np.isnan(neighbor_values)]
                        
                        if len(neighbor_values) > 0:
                            if self.weights == "uniform":
                                X[i, j] = np.mean(neighbor_values)
                            else:  # distance
                                neighbor_dists = distances[nearest_idx]
                                neighbor_dists = neighbor_dists[~np.isnan(self.X_fit_[nearest_idx, j])]
                                weights = 1 / (neighbor_dists + 1e-10)
                                X[i, j] = np.average(neighbor_values, weights=weights)
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)
    
    def _nan_euclidean_distances(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute distances ignoring NaN."""
        distances = np.zeros(Y.shape[0])
        
        for i in range(Y.shape[0]):
            # only use features that are non-NaN in both
            valid_mask = ~(np.isnan(x) | np.isnan(Y[i]))
            if np.sum(valid_mask) > 0:
                distances[i] = np.sqrt(np.sum((x[valid_mask] - Y[i, valid_mask])**2))
            else:
                distances[i] = np.inf
        
        return distances


__all__ = [
    "RobustScaler",
    "MaxAbsScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "SimpleImputer",
    "KNNImputer",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.