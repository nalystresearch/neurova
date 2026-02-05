# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Preprocessing utilities for machine learning

All implementations use NumPy only.
"""

import numpy as np
from typing import Optional, Union
from neurova.core.errors import ValidationError


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance
    
    Examples:
        >>> scaler = StandardScaler()
        >>> X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute mean and std to be used for scaling
        
        Args:
            X: Training data, shape (n_samples, n_features)
            
        Returns:
            self
        """
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        self.n_features_ = X.shape[1] if X.ndim > 1 else 1
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features
        
        Args:
            X: Data to transform
            
        Returns:
            Standardized data
        """
        if self.mean_ is None:
            raise ValidationError('scaler', 'not fitted', 'fitted scaler')
        
        X = np.asarray(X)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardization"""
        X = np.asarray(X)
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range (default [0, 1])
    
    Examples:
        >>> scaler = MinMaxScaler()
        >>> X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Compute min and max for scaling"""
        X = np.asarray(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0
        
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to feature_range"""
        if self.scale_ is None:
            raise ValidationError('scaler', 'not fitted', 'fitted scaler')
        
        X = np.asarray(X)
        return X * self.scale_ + self.min_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling"""
        X = np.asarray(X)
        return (X - self.min_) / self.scale_


class Normalizer:
    """
    Normalize samples individually to unit norm
    
    Examples:
        >>> normalizer = Normalizer(norm='l2')
        >>> X_normalized = normalizer.fit_transform(X)
    """
    
    def __init__(self, norm: str = 'l2'):
        """
        Args:
            norm: Norm to use ('l1', 'l2', or 'max')
        """
        if norm not in ('l1', 'l2', 'max'):
            raise ValidationError('norm', norm, 'l1, l2, or max')
        self.norm = norm
    
    def fit(self, X: np.ndarray) -> 'Normalizer':
        """No-op, exists for API compatibility"""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize each sample"""
        X = np.asarray(X, dtype=np.float64)
        
        if self.norm == 'l1':
            norms = np.abs(X).sum(axis=1, keepdims=True)
        elif self.norm == 'l2':
            norms = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        else:  # max
            norms = np.abs(X).max(axis=1, keepdims=True)
        
        norms[norms == 0] = 1.0
        return X / norms
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1
    
    Examples:
        >>> encoder = LabelEncoder()
        >>> y_encoded = encoder.fit_transform(['cat', 'dog', 'cat', 'bird'])
    """
    
    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None
    
    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """Find unique classes"""
        self.classes_ = np.unique(y)
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels to normalized encoding"""
        if self.classes_ is None or self.class_to_index_ is None:
            raise ValidationError('encoder', 'not fitted', 'fitted encoder')
        
        return np.array([self.class_to_index_[label] for label in y])
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels back to original encoding"""
        if self.classes_ is None:
            raise ValidationError('encoder', 'not fitted', 'fitted encoder')
        
        return np.array([self.classes_[idx] for idx in y])


class OneHotEncoder:
    """
    Encode categorical features as one-hot numeric array
    
    Examples:
        >>> encoder = OneHotEncoder()
        >>> X_encoded = encoder.fit_transform([[0], [1], [2], [1]])
    """
    
    def __init__(self, sparse: bool = False):
        self.sparse = sparse
        self.categories_ = None
        self.n_features_ = None
    
    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """Find categories"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_ = X.shape[1]
        self.categories_ = [np.unique(X[:, i]) for i in range(self.n_features_)]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to one-hot encoding"""
        if self.categories_ is None:
            raise ValidationError('encoder', 'not fitted', 'fitted encoder')
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        n_features_out = sum(len(cats) for cats in self.categories_)
        
        result = np.zeros((n_samples, n_features_out), dtype=np.float64)
        
        feature_idx = 0
        n_features = self.n_features_ if self.n_features_ is not None else 0
        for i in range(n_features):
            for j, category in enumerate(self.categories_[i]):
                mask = X[:, i] == category
                result[mask, feature_idx + j] = 1
            feature_idx += len(self.categories_[i])
        
        return result
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


def polynomial_features(X: np.ndarray, degree: int = 2, 
                       include_bias: bool = True) -> np.ndarray:
    """
    Generate polynomial features
    
    Args:
        X: Input data, shape (n_samples, n_features)
        degree: Degree of polynomial features
        include_bias: Whether to include bias column
        
    Returns:
        Polynomial features
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    
    features = [X]
    
    for d in range(2, degree + 1):
        features.append(X ** d)
    
    result = np.hstack(features)
    
    if include_bias:
        bias = np.ones((n_samples, 1))
        result = np.hstack([bias, result])
    
    return result
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.