# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Feature selection methods."""

from __future__ import annotations
import numpy as np
from typing import Optional, Literal


class SelectKBest:
    """Select features according to k highest scores.
    
    Args:
        score_func: Function taking X, y and returning scores
        k: Number of top features to select
        
    Examples:
        from neurova.ml.feature_selection import f_classif
        selector = SelectKBest(score_func=f_classif, k=10)
        X_new = selector.fit_transform(X, y)
    """
    
    def __init__(self, score_func=None, k: int = 10):
        self.score_func = score_func or f_classif
        self.k = k
        
        self.scores_: Optional[np.ndarray] = None
        self.pvalues_: Optional[np.ndarray] = None
        self.selected_features_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> SelectKBest:
        """Fit feature selector."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        # compute scores
        self.scores_, self.pvalues_ = self.score_func(X, y)
        
        # select k best
        self.selected_features_ = np.argsort(self.scores_)[-self.k:]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform by selecting features."""
        X = np.asarray(X)
        
        if self.selected_features_ is None:
            raise RuntimeError("Selector not fitted")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)


class SelectPercentile:
    """Select features in highest percentile.
    
    Args:
        score_func: Scoring function
        percentile: Percentile of features to keep
        
    Examples:
        selector = SelectPercentile(percentile=10)
        X_new = selector.fit_transform(X, y)
    """
    
    def __init__(self, score_func=None, percentile: int = 10):
        self.score_func = score_func or f_classif
        self.percentile = percentile
        
        self.scores_: Optional[np.ndarray] = None
        self.selected_features_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> SelectPercentile:
        """Fit selector."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.scores_, _ = self.score_func(X, y)
        
        # select percentile
        threshold = np.percentile(self.scores_, 100 - self.percentile)
        self.selected_features_ = np.where(self.scores_ >= threshold)[0]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features."""
        X = np.asarray(X)
        
        if self.selected_features_ is None:
            raise RuntimeError("Selector not fitted")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)


class RFE:
    """Recursive Feature Elimination.
    
    Args:
        estimator: Base estimator with fit and coef_/feature_importances_
        n_features_to_select: Number of features to select
        step: Number of features to remove at each iteration
        
    Examples:
        from neurova.ml import LogisticRegression
        rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
        X_new = rfe.fit_transform(X, y)
    """
    
    def __init__(
        self,
        estimator,
        n_features_to_select: Optional[int] = None,
        step: int = 1,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        
        self.support_: Optional[np.ndarray] = None
        self.ranking_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> RFE:
        """Fit RFE."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        if self.n_features_to_select is None:
            self.n_features_to_select = n_features // 2
        
        # initialize
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        
        # iteratively remove features
        n_iterations = 0
        while np.sum(support) > self.n_features_to_select:
            # fit estimator
            features = np.where(support)[0]
            self.estimator.fit(X[:, features], y)
            
            # get feature importances
            if hasattr(self.estimator, 'coef_'):
                importances = np.abs(self.estimator.coef_).ravel()
            elif hasattr(self.estimator, 'feature_importances_'):
                importances = self.estimator.feature_importances_
            else:
                raise ValueError("Estimator must have coef_ or feature_importances_")
            
            # rank features
            ranks = np.argsort(importances)
            
            # remove worst features
            n_to_remove = min(self.step, np.sum(support) - self.n_features_to_select)
            threshold = importances[ranks[n_to_remove - 1]]
            
            removed = (importances <= threshold) & support[features]
            support[features[removed]] = False
            ranking[features[removed]] = n_iterations + 1
            
            n_iterations += 1
        
        self.support_ = support
        self.ranking_ = ranking
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features."""
        X = np.asarray(X)
        
        if self.support_ is None:
            raise RuntimeError("RFE not fitted")
        
        return X[:, self.support_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)


class VarianceThreshold:
    """Remove low-variance features.
    
    Args:
        threshold: Features with variance below this are removed
        
    Examples:
        selector = VarianceThreshold(threshold=0.01)
        X_new = selector.fit_transform(X)
    """
    
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        
        self.variances_: Optional[np.ndarray] = None
        self.selected_features_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y=None) -> VarianceThreshold:
        """Compute variances."""
        X = np.asarray(X)
        
        variances = np.var(X, axis=0)
        self.variances_ = variances
        selected = variances > self.threshold
        self.selected_features_ = np.where(selected)[0]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Remove low-variance features."""
        X = np.asarray(X)
        
        if self.selected_features_ is None:
            raise RuntimeError("Selector not fitted")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)


def f_classif(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ANOVA F-value for classification.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        Tuple of (F-scores, p-values)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    classes = np.unique(y)
    n_features = X.shape[1]
    
    f_scores = np.zeros(n_features)
    p_values = np.zeros(n_features)
    
    for i in range(n_features):
        # compute F-statistic for each feature
        groups = [X[y == c, i] for c in classes]
        
        # between-group variance
        grand_mean = np.mean(X[:, i])
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        df_between = len(classes) - 1
        
        # within-group variance
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
        df_within = len(y) - len(classes)
        
        # f-statistic
        ms_between = ss_between / df_between
        ms_within = ss_within / (df_within + 1e-10)
        f_scores[i] = ms_between / (ms_within + 1e-10)
        
        # approximate p-value
        p_values[i] = _f_pvalue_approx(f_scores[i], df_between, df_within)
    
    return f_scores, p_values


def f_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute F-value for regression.
    
    Args:
        X: Feature matrix
        y: Target values
        
    Returns:
        Tuple of (F-scores, p-values)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    n_samples, n_features = X.shape
    
    f_scores = np.zeros(n_features)
    p_values = np.zeros(n_features)
    
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean)**2)
    
    for i in range(n_features):
        # compute correlation
        x_centered = X[:, i] - np.mean(X[:, i])
        y_centered = y - y_mean
        
        corr = np.sum(x_centered * y_centered) / (
            np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)) + 1e-10
        )
        
        # f-statistic from correlation
        r_squared = corr**2
        f_scores[i] = (r_squared * (n_samples - 2)) / ((1 - r_squared) + 1e-10)
        
        # approximate p-value
        p_values[i] = _f_pvalue_approx(f_scores[i], 1, n_samples - 2)
    
    return f_scores, p_values


def mutual_info_classif(X: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> np.ndarray:
    """Estimate mutual information for classification.
    
    Args:
        X: Feature matrix
        y: Target labels
        n_neighbors: Number of neighbors for estimation
        
    Returns:
        Mutual information scores
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)
    
    for i in range(n_features):
        # discretize feature
        x = X[:, i]
        x_discrete = _discretize(x, n_bins=10)
        
        # compute mutual information
        mi_scores[i] = _mutual_information(x_discrete, y)
    
    return mi_scores


def _discretize(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretize continuous variable."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(x, quantiles)
    bins = np.unique(bins)
    return np.digitize(x, bins[1:-1])


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between discrete variables."""
    # joint distribution
    xy = np.stack([x, y], axis=1)
    unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
    p_xy = counts_xy / len(x)
    
    # marginal distributions
    unique_x, counts_x = np.unique(x, return_counts=True)
    p_x = counts_x / len(x)
    
    unique_y, counts_y = np.unique(y, return_counts=True)
    p_y = counts_y / len(y)
    
    # mutual information
    mi = 0.0
    for i, (xi, yi) in enumerate(unique_xy):
        p_xy_i = p_xy[i]
        p_x_i = p_x[unique_x == xi][0]
        p_y_i = p_y[unique_y == yi][0]
        
        if p_xy_i > 0:
            mi += p_xy_i * np.log(p_xy_i / (p_x_i * p_y_i + 1e-10) + 1e-10)
    
    return max(0.0, mi)


def _f_pvalue_approx(f_stat: float, df1: int, df2: int) -> float:
    """Approximate F-distribution p-value."""
    if f_stat < 1.0:
        return 0.90
    elif f_stat < 2.0:
        return 0.20
    elif f_stat < 3.0:
        return 0.10
    elif f_stat < 4.0:
        return 0.05
    elif f_stat < 6.0:
        return 0.02
    else:
        return 0.01


__all__ = [
    "SelectKBest",
    "SelectPercentile",
    "RFE",
    "VarianceThreshold",
    "f_classif",
    "f_regression",
    "mutual_info_classif",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.