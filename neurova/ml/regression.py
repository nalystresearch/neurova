# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Linear Regression Models for Machine Learning.

Provides comprehensive regression implementations including:
- Linear Regression (OLS)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet Regression (L1 + L2 regularization)
- Polynomial Regression
- Bayesian Ridge Regression
- Huber Regression (robust to outliers)
- Quantile Regression
- Lars (Least Angle Regression)
- Orthogonal Matching Pursuit

All implementations follow scikit-learn conventions.
"""

import numpy as np
from typing import Optional, Union, Tuple, List
from abc import ABC, abstractmethod


class BaseRegressor(ABC):
    """
    Abstract base class for regression models.
    
    All regressors inherit from this class and implement
    fit() and predict() methods.
    """
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseRegressor':
        """Fit the model to data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score (coefficient of determination).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,)
            True values
            
        Returns
        -------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)


class LinearRegression(BaseRegressor):
    """
    Ordinary Least Squares Linear Regression.
    
    Fits a linear model y = X @ w + b to minimize the residual 
    sum of squares between observed and predicted targets.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten
    normalize : bool, default=False
        If True, normalize X before regression (deprecated, use StandardScaler)
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Estimated coefficients
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in the linear model
    rank_ : int
        Rank of matrix X
    singular_ : ndarray of shape (min(X.shape),)
        Singular values of X
        
    Examples
    --------
    >>> from neurova.ml.regression import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """
    
    def __init__(self, fit_intercept: bool = True, copy_X: bool = True,
                 normalize: bool = False):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.normalize = normalize
        self.rank_ = None
        self.singular_ = None
        self._X_mean = None
        self._X_scale = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values
            
        Returns
        -------
        self : LinearRegression
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.copy_X:
            X = X.copy()
        
        self.n_features_in_ = X.shape[1]
        
        # Center data
        if self.fit_intercept:
            self._X_mean = np.mean(X, axis=0)
            X = X - self._X_mean
            y_mean = np.mean(y, axis=0)
            y = y - y_mean
        
        # Normalize if requested
        if self.normalize:
            self._X_scale = np.linalg.norm(X, axis=0)
            self._X_scale[self._X_scale == 0] = 1
            X = X / self._X_scale
        
        # Solve using SVD for numerical stability
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.rank_ = np.sum(s > 1e-10 * s[0])
        self.singular_ = s
        
        # Compute coefficients
        s_inv = np.zeros_like(s)
        s_inv[s > 1e-10] = 1 / s[s > 1e-10]
        self.coef_ = Vt.T @ np.diag(s_inv) @ U.T @ y
        
        # Undo normalization
        if self.normalize:
            self.coef_ = self.coef_ / self._X_scale
        
        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(self._X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class Ridge(BaseRegressor):
    """
    Ridge Regression (L2 regularization).
    
    Minimizes ||y - Xw||² + alpha * ||w||²
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Must be positive.
    fit_intercept : bool, default=True
        Whether to calculate the intercept
    normalize : bool, default=False
        Deprecated, use StandardScaler
    copy_X : bool, default=True
        If True, X will be copied
    max_iter : int, default=None
        Maximum iterations for solver
    tol : float, default=1e-4
        Precision of the solution
    solver : str, default='auto'
        Solver: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weight vector
    intercept_ : float
        Independent term
        
    Examples
    --------
    >>> from neurova.ml.regression import Ridge
    >>> X = np.array([[0, 0], [0, 0], [1, 1]])
    >>> y = np.array([0, 0.1, 1])
    >>> clf = Ridge(alpha=1.0)
    >>> clf.fit(X, y)
    >>> clf.coef_
    array([0.34545455, 0.34545455])
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 normalize: bool = False, copy_X: bool = True,
                 max_iter: Optional[int] = None, tol: float = 1e-4,
                 solver: str = 'auto'):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Ridge':
        """
        Fit Ridge regression model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : Ridge
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.copy_X:
            X = X.copy()
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Center data
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            X = X - X_mean
            y_mean = np.mean(y)
            y = y - y_mean
        
        # Solve (X.T @ X + alpha * I) @ w = X.T @ y
        A = X.T @ X + self.alpha * np.eye(n_features)
        b = X.T @ y
        
        self.coef_ = np.linalg.solve(A, b)
        
        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Ridge model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class Lasso(BaseRegressor):
    """
    Lasso Regression (L1 regularization).
    
    Minimizes (1/2n) * ||y - Xw||² + alpha * ||w||₁
    
    Uses coordinate descent optimization.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Must be positive.
    fit_intercept : bool, default=True
        Whether to calculate the intercept
    normalize : bool, default=False
        Deprecated, use StandardScaler
    max_iter : int, default=1000
        Maximum iterations
    tol : float, default=1e-4
        Convergence tolerance
    warm_start : bool, default=False
        Reuse solution of previous fit
    positive : bool, default=False
        Force coefficients to be positive
    selection : str, default='cyclic'
        Coefficient selection: 'cyclic' or 'random'
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weight vector (sparse)
    intercept_ : float
        Independent term
    n_iter_ : int
        Number of iterations run
        
    Examples
    --------
    >>> from neurova.ml.regression import Lasso
    >>> clf = Lasso(alpha=0.1)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    >>> clf.coef_
    array([0.85, 0.  ])
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 normalize: bool = False, max_iter: int = 1000,
                 tol: float = 1e-4, warm_start: bool = False,
                 positive: bool = False, selection: str = 'cyclic'):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.selection = selection
        self.n_iter_ = 0
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator for L1 regularization."""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Lasso':
        """
        Fit Lasso model using coordinate descent.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : Lasso
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Center data
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            X = X - X_mean
            y_mean = np.mean(y)
            y = y - y_mean
        
        # Initialize coefficients
        if self.warm_start and self.coef_ is not None:
            coef = self.coef_.copy()
        else:
            coef = np.zeros(n_features)
        
        # Precompute column norms
        col_norms = np.sum(X ** 2, axis=0)
        
        # Coordinate descent
        residual = y - X @ coef
        
        for iteration in range(self.max_iter):
            coef_old = coef.copy()
            
            # Select coordinate order
            if self.selection == 'random':
                feature_order = np.random.permutation(n_features)
            else:
                feature_order = range(n_features)
            
            for j in feature_order:
                # Update residual for feature j
                residual += X[:, j] * coef[j]
                
                # Compute gradient component
                rho = np.dot(X[:, j], residual)
                
                # Apply soft thresholding
                if col_norms[j] > 1e-10:
                    z = self._soft_threshold(rho, n_samples * self.alpha)
                    coef[j] = z / col_norms[j]
                    
                    if self.positive and coef[j] < 0:
                        coef[j] = 0.0
                else:
                    coef[j] = 0.0
                
                # Update residual
                residual -= X[:, j] * coef[j]
            
            # Check convergence
            max_change = np.max(np.abs(coef - coef_old))
            if max_change < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        self.coef_ = coef
        
        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Lasso model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class ElasticNet(BaseRegressor):
    """
    ElasticNet Regression (L1 + L2 regularization).
    
    Minimizes (1/2n) * ||y - Xw||² + alpha * l1_ratio * ||w||₁ 
               + 0.5 * alpha * (1 - l1_ratio) * ||w||²
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength
    l1_ratio : float, default=0.5
        Mix of L1 and L2. l1_ratio=1 is Lasso, l1_ratio=0 is Ridge.
    fit_intercept : bool, default=True
        Whether to calculate the intercept
    normalize : bool, default=False
        Deprecated, use StandardScaler
    max_iter : int, default=1000
        Maximum iterations
    tol : float, default=1e-4
        Convergence tolerance
    warm_start : bool, default=False
        Reuse solution
    positive : bool, default=False
        Force positive coefficients
    selection : str, default='cyclic'
        'cyclic' or 'random'
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weight vector
    intercept_ : float
        Independent term
    n_iter_ : int
        Number of iterations
        
    Examples
    --------
    >>> from neurova.ml.regression import ElasticNet
    >>> clf = ElasticNet(alpha=0.1, l1_ratio=0.5)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    """
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 fit_intercept: bool = True, normalize: bool = False,
                 max_iter: int = 1000, tol: float = 1e-4,
                 warm_start: bool = False, positive: bool = False,
                 selection: str = 'cyclic'):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.selection = selection
        self.n_iter_ = 0
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator."""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNet':
        """
        Fit ElasticNet model using coordinate descent.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : ElasticNet
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Compute L1 and L2 penalty weights
        l1_penalty = self.alpha * self.l1_ratio * n_samples
        l2_penalty = self.alpha * (1 - self.l1_ratio)
        
        # Center data
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            X = X - X_mean
            y_mean = np.mean(y)
            y = y - y_mean
        
        # Initialize coefficients
        if self.warm_start and self.coef_ is not None:
            coef = self.coef_.copy()
        else:
            coef = np.zeros(n_features)
        
        # Precompute column norms with L2 penalty
        col_norms = np.sum(X ** 2, axis=0) + l2_penalty * n_samples
        
        # Coordinate descent
        residual = y - X @ coef
        
        for iteration in range(self.max_iter):
            coef_old = coef.copy()
            
            # Select coordinate order
            if self.selection == 'random':
                feature_order = np.random.permutation(n_features)
            else:
                feature_order = range(n_features)
            
            for j in feature_order:
                # Update residual for feature j
                residual += X[:, j] * coef[j]
                
                # Compute gradient component
                rho = np.dot(X[:, j], residual)
                
                # Apply soft thresholding
                if col_norms[j] > 1e-10:
                    z = self._soft_threshold(rho, l1_penalty)
                    coef[j] = z / col_norms[j]
                    
                    if self.positive and coef[j] < 0:
                        coef[j] = 0.0
                else:
                    coef[j] = 0.0
                
                # Update residual
                residual -= X[:, j] * coef[j]
            
            # Check convergence
            max_change = np.max(np.abs(coef - coef_old))
            if max_change < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        self.coef_ = coef
        
        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ElasticNet model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.
    
    Parameters
    ----------
    degree : int, default=2
        The degree of polynomial features
    interaction_only : bool, default=False
        If True, only interaction features
    include_bias : bool, default=True
        If True, include a bias column of all ones
        
    Examples
    --------
    >>> from neurova.ml.regression import PolynomialFeatures
    >>> X = np.arange(6).reshape(3, 2)
    >>> poly = PolynomialFeatures(2)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.,  0.,  1.],
           [ 1.,  2.,  3.,  4.,  6.,  9.],
           [ 1.,  4.,  5., 16., 20., 25.]])
    """
    
    def __init__(self, degree: int = 2, interaction_only: bool = False,
                 include_bias: bool = True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_features_in_ = None
        self.n_output_features_ = None
    
    def fit(self, X: np.ndarray, y=None) -> 'PolynomialFeatures':
        """
        Compute the output feature count.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : PolynomialFeatures
        """
        X = np.asarray(X)
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        
        # Calculate number of output features
        from math import comb
        if self.interaction_only:
            n_output = sum(comb(n_features, k) for k in range(self.degree + 1))
        else:
            n_output = comb(n_features + self.degree, self.degree)
        
        if not self.include_bias:
            n_output -= 1
        
        self.n_output_features_ = n_output
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to polynomial features.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_poly : ndarray of shape (n_samples, n_output_features_)
            Transformed data
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Build polynomial features iteratively
        features = []
        
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        
        # Degree 1 features
        features.append(X)
        
        # Higher degree features
        for d in range(2, self.degree + 1):
            for i in range(n_features):
                for j in range(i, n_features):
                    if self.interaction_only and i == j:
                        continue
                    if d == 2:
                        features.append((X[:, i] * X[:, j]).reshape(-1, 1))
                    else:
                        # For higher degrees, multiply existing features
                        features.append((X[:, i] ** d).reshape(-1, 1))
        
        return np.hstack(features)
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


class BayesianRidge(BaseRegressor):
    """
    Bayesian Ridge Regression with automatic relevance determination.
    
    Parameters
    ----------
    n_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-3
        Convergence threshold
    alpha_1 : float, default=1e-6
        Shape parameter for Gamma prior over alpha
    alpha_2 : float, default=1e-6
        Rate parameter for Gamma prior over alpha
    lambda_1 : float, default=1e-6
        Shape parameter for Gamma prior over lambda
    lambda_2 : float, default=1e-6
        Rate parameter for Gamma prior over lambda
    compute_score : bool, default=False
        Compute log marginal likelihood at each iteration
    fit_intercept : bool, default=True
        Whether to calculate the intercept
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients of the regression model
    alpha_ : float
        Estimated precision of the noise
    lambda_ : float
        Estimated precision of the weights
    sigma_ : ndarray of shape (n_features, n_features)
        Estimated variance-covariance matrix of the weights
    scores_ : list
        Log marginal likelihood scores (if compute_score=True)
    """
    
    def __init__(self, n_iter: int = 300, tol: float = 1e-3,
                 alpha_1: float = 1e-6, alpha_2: float = 1e-6,
                 lambda_1: float = 1e-6, lambda_2: float = 1e-6,
                 compute_score: bool = False, fit_intercept: bool = True):
        super().__init__()
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.alpha_ = None
        self.lambda_ = None
        self.sigma_ = None
        self.scores_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianRidge':
        """
        Fit the Bayesian ridge model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : BayesianRidge
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Center data
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            X = X - X_mean
            y_mean = np.mean(y)
            y = y - y_mean
        
        # Compute eigenvalues
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        eigenvalues = S ** 2
        
        # Initialize hyperparameters
        self.alpha_ = 1.0 / np.var(y)
        self.lambda_ = 1.0
        
        # EM-like iterations
        coef = np.zeros(n_features)
        
        for iteration in range(self.n_iter):
            # Store old values for convergence check
            alpha_old = self.alpha_
            lambda_old = self.lambda_
            
            # Compute posterior covariance
            A = self.alpha_ * (X.T @ X) + self.lambda_ * np.eye(n_features)
            A_inv = np.linalg.inv(A)
            self.sigma_ = A_inv / self.alpha_
            
            # Compute posterior mean
            coef = self.alpha_ * A_inv @ X.T @ y
            
            # Update hyperparameters
            gamma = np.sum(eigenvalues / (eigenvalues + self.lambda_ / self.alpha_))
            
            self.lambda_ = (gamma + 2 * self.lambda_1) / (np.dot(coef, coef) + 2 * self.lambda_2)
            
            residual = y - X @ coef
            self.alpha_ = (n_samples - gamma + 2 * self.alpha_1) / \
                         (np.dot(residual, residual) + 2 * self.alpha_2)
            
            # Compute score if requested
            if self.compute_score:
                score = self._log_marginal_likelihood(X, y, eigenvalues)
                self.scores_.append(score)
            
            # Check convergence
            if np.abs(self.alpha_ - alpha_old) < self.tol and \
               np.abs(self.lambda_ - lambda_old) < self.tol:
                break
        
        self.coef_ = coef
        
        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def _log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray, 
                                  eigenvalues: np.ndarray) -> float:
        """Compute log marginal likelihood."""
        n_samples, n_features = X.shape
        
        log_det = np.sum(np.log(eigenvalues + self.lambda_ / self.alpha_))
        residual = y - X @ self.coef_
        
        score = 0.5 * n_features * np.log(self.lambda_)
        score += 0.5 * n_samples * np.log(self.alpha_)
        score -= 0.5 * log_det
        score -= 0.5 * self.alpha_ * np.dot(residual, residual)
        score -= 0.5 * self.lambda_ * np.dot(self.coef_, self.coef_)
        
        return score
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Predict using the Bayesian Ridge model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples
        return_std : bool, default=False
            If True, return standard deviation of predictions
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Mean of predictive distribution
        y_std : ndarray of shape (n_samples,) (optional)
            Standard deviation of predictive distribution
        """
        X = np.asarray(X, dtype=np.float64)
        y_mean = X @ self.coef_ + self.intercept_
        
        if return_std:
            # Predictive variance: 1/alpha + x.T @ Sigma @ x
            X_centered = X - np.mean(X, axis=0) if self.fit_intercept else X
            y_var = 1.0 / self.alpha_ + np.sum(X_centered @ self.sigma_ * X_centered, axis=1)
            return y_mean, np.sqrt(y_var)
        
        return y_mean


class HuberRegressor(BaseRegressor):
    """
    Huber Regression - robust to outliers.
    
    Uses Huber loss instead of squared loss, making it more
    robust to outliers than standard linear regression.
    
    Parameters
    ----------
    epsilon : float, default=1.35
        Threshold for Huber loss transition
    max_iter : int, default=100
        Maximum iterations
    alpha : float, default=0.0001
        Regularization strength (L2)
    warm_start : bool, default=False
        Reuse solution
    fit_intercept : bool, default=True
        Whether to fit intercept
    tol : float, default=1e-5
        Convergence tolerance
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients
    intercept_ : float
        Intercept
    scale_ : float
        Estimated scale of the residuals
    n_iter_ : int
        Number of iterations
    """
    
    def __init__(self, epsilon: float = 1.35, max_iter: int = 100,
                 alpha: float = 0.0001, warm_start: bool = False,
                 fit_intercept: bool = True, tol: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.scale_ = None
        self.n_iter_ = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None) -> 'HuberRegressor':
        """
        Fit the Huber regression model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        sample_weight : ndarray of shape (n_samples,), optional
            Individual weights for each sample
            
        Returns
        -------
        self : HuberRegressor
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        
        # Initialize with OLS
        if not (self.warm_start and self.coef_ is not None):
            lr = LinearRegression(fit_intercept=self.fit_intercept)
            lr.fit(X, y)
            self.coef_ = lr.coef_
            self.intercept_ = lr.intercept_
        
        # Iteratively reweighted least squares
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            # Compute residuals
            y_pred = X @ self.coef_ + self.intercept_
            residuals = y - y_pred
            
            # Estimate scale using MAD
            self.scale_ = np.median(np.abs(residuals - np.median(residuals))) / 0.6745
            self.scale_ = max(self.scale_, 1e-10)
            
            # Compute Huber weights
            scaled_residuals = residuals / self.scale_
            weights = np.ones(n_samples)
            outlier_mask = np.abs(scaled_residuals) > self.epsilon
            weights[outlier_mask] = self.epsilon / np.abs(scaled_residuals[outlier_mask])
            
            # Combined weights
            W = sample_weight * weights
            
            # Weighted least squares with L2 regularization
            XtW = X.T * W
            XtWX = XtW @ X + self.alpha * np.eye(n_features)
            XtWy = XtW @ y
            
            if self.fit_intercept:
                X_mean = np.sum(W[:, np.newaxis] * X, axis=0) / np.sum(W)
                y_mean = np.sum(W * y) / np.sum(W)
                X_centered = X - X_mean
                y_centered = y - y_mean
                
                XtW_c = X_centered.T * W
                XtWX_c = XtW_c @ X_centered + self.alpha * np.eye(n_features)
                XtWy_c = XtW_c @ y_centered
                
                self.coef_ = np.linalg.solve(XtWX_c, XtWy_c)
                self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
            else:
                self.coef_ = np.linalg.solve(XtWX, XtWy)
                self.intercept_ = 0.0
            
            # Check convergence
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Huber model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class QuantileRegressor(BaseRegressor):
    """
    Quantile Regression.
    
    Fits a linear model that predicts a specific quantile of the target.
    
    Parameters
    ----------
    quantile : float, default=0.5
        The quantile to predict (0.5 = median)
    alpha : float, default=1.0
        Regularization strength (L1)
    fit_intercept : bool, default=True
        Whether to fit intercept
    max_iter : int, default=1000
        Maximum iterations
    tol : float, default=1e-4
        Convergence tolerance
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients
    intercept_ : float
        Intercept
    n_iter_ : int
        Number of iterations
    """
    
    def __init__(self, quantile: float = 0.5, alpha: float = 1.0,
                 fit_intercept: bool = True, max_iter: int = 1000,
                 tol: float = 1e-4):
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_ = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        """
        Fit the quantile regression model.
        
        Uses iteratively reweighted least squares.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : QuantileRegressor
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(n_samples), X])
            n_coefs = n_features + 1
        else:
            X_aug = X
            n_coefs = n_features
        
        # Initialize with OLS
        coef = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        
        # IRLS for quantile regression
        for iteration in range(self.max_iter):
            coef_old = coef.copy()
            
            # Compute residuals
            residuals = y - X_aug @ coef
            
            # Quantile weights (check function)
            weights = np.where(residuals >= 0, self.quantile, 1 - self.quantile)
            weights = weights / (np.abs(residuals) + 1e-6)
            
            # Weighted least squares with L1-like penalty
            W = np.diag(weights)
            XtWX = X_aug.T @ W @ X_aug + self.alpha * np.eye(n_coefs)
            XtWy = X_aug.T @ W @ y
            
            coef = np.linalg.solve(XtWX, XtWy)
            
            # Check convergence
            if np.max(np.abs(coef - coef_old)) < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using quantile model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class Lars(BaseRegressor):
    """
    Least Angle Regression.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit intercept
    normalize : bool, default=True
        Whether to normalize features
    n_nonzero_coefs : int, default=500
        Maximum number of non-zero coefficients
    eps : float, default=np.finfo(float).eps
        Machine precision
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients
    intercept_ : float
        Intercept
    alphas_ : ndarray
        Maximum absolute correlations at each step
    active_ : list
        Indices of active features at end
    coef_path_ : ndarray
        Coefficients at each step
    """
    
    def __init__(self, fit_intercept: bool = True, normalize: bool = True,
                 n_nonzero_coefs: int = 500, eps: float = np.finfo(float).eps):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.alphas_ = None
        self.active_ = None
        self.coef_path_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Lars':
        """
        Fit LARS model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : Lars
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Center and normalize
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            X = X - X_mean
            y_mean = np.mean(y)
            y = y - y_mean
        
        if self.normalize:
            norms = np.linalg.norm(X, axis=0)
            norms[norms == 0] = 1
            X = X / norms
        
        # Initialize
        coef = np.zeros(n_features)
        residual = y.copy()
        active = []
        
        alphas = []
        coef_path = [coef.copy()]
        
        max_features = min(n_features, self.n_nonzero_coefs)
        
        for k in range(max_features):
            # Compute correlations
            c = X.T @ residual
            
            # Find most correlated feature
            c_abs = np.abs(c)
            j = np.argmax(c_abs)
            
            if j in active:
                # All features already active or numerical issues
                break
            
            active.append(j)
            alphas.append(c_abs[j])
            
            # Sign of correlation
            s = np.sign(c[active])
            
            # Compute equiangular direction
            X_active = X[:, active] * s
            G = X_active.T @ X_active
            
            try:
                G_inv = np.linalg.inv(G + self.eps * np.eye(len(active)))
            except np.linalg.LinAlgError:
                break
            
            ones = np.ones(len(active))
            A_k = 1 / np.sqrt(ones @ G_inv @ ones)
            w = A_k * G_inv @ ones
            u = X_active @ w
            
            # Compute step size
            a = X.T @ u
            
            gamma = float('inf')
            for j_inactive in range(n_features):
                if j_inactive in active:
                    continue
                
                c_j = c[j_inactive]
                a_j = a[j_inactive]
                
                if abs(A_k - a_j) > self.eps:
                    g1 = (alphas[-1] - c_j) / (A_k - a_j)
                    if g1 > self.eps and g1 < gamma:
                        gamma = g1
                
                if abs(A_k + a_j) > self.eps:
                    g2 = (alphas[-1] + c_j) / (A_k + a_j)
                    if g2 > self.eps and g2 < gamma:
                        gamma = g2
            
            if gamma == float('inf'):
                gamma = alphas[-1] / A_k
            
            # Update coefficients
            coef[active] = coef[active] + gamma * s * w
            residual = residual - gamma * u
            coef_path.append(coef.copy())
            
            if np.linalg.norm(residual) < self.eps:
                break
        
        self.coef_ = coef
        self.alphas_ = np.array(alphas)
        self.active_ = active
        self.coef_path_ = np.array(coef_path)
        
        if self.normalize:
            self.coef_ = self.coef_ / norms
        
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using LARS model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class OrthogonalMatchingPursuit(BaseRegressor):
    """
    Orthogonal Matching Pursuit regression.
    
    Parameters
    ----------
    n_nonzero_coefs : int, default=None
        Number of non-zero coefficients. If None, use tol.
    tol : float, default=None
        Residual norm threshold. If None, use n_nonzero_coefs.
    fit_intercept : bool, default=True
        Whether to fit intercept
    normalize : bool, default=True
        Whether to normalize features
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients
    intercept_ : float
        Intercept
    n_iter_ : int
        Number of selected features
    n_nonzero_coefs_ : int
        Number of non-zero coefficients
    """
    
    def __init__(self, n_nonzero_coefs: Optional[int] = None,
                 tol: Optional[float] = None, fit_intercept: bool = True,
                 normalize: bool = True):
        super().__init__()
        if n_nonzero_coefs is None and tol is None:
            n_nonzero_coefs = 1
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.n_iter_ = 0
        self.n_nonzero_coefs_ = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OrthogonalMatchingPursuit':
        """
        Fit OMP model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : OrthogonalMatchingPursuit
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Center and normalize
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            X = X - X_mean
            y_mean = np.mean(y)
            y = y - y_mean
        
        if self.normalize:
            norms = np.linalg.norm(X, axis=0)
            norms[norms == 0] = 1
            X = X / norms
        
        # Initialize
        coef = np.zeros(n_features)
        residual = y.copy()
        active = []
        
        n_target = self.n_nonzero_coefs if self.n_nonzero_coefs else n_features
        
        for k in range(n_target):
            # Find most correlated feature
            correlations = np.abs(X.T @ residual)
            correlations[active] = -1  # Exclude already selected
            j = np.argmax(correlations)
            active.append(j)
            
            # Solve least squares on active set
            X_active = X[:, active]
            coef_active = np.linalg.lstsq(X_active, y, rcond=None)[0]
            
            # Update residual
            residual = y - X_active @ coef_active
            
            # Check tolerance
            if self.tol is not None and np.linalg.norm(residual) <= self.tol:
                break
        
        self.n_iter_ = len(active)
        self.n_nonzero_coefs_ = len(active)
        
        # Set coefficients
        coef[active] = coef_active
        self.coef_ = coef
        
        if self.normalize:
            self.coef_ = self.coef_ / norms
        
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using OMP model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


# Convenience functions for regression
def make_regression_pipeline(X: np.ndarray, y: np.ndarray,
                             degree: int = 1,
                             regularization: Optional[str] = None,
                             alpha: float = 1.0) -> Tuple[BaseRegressor, float]:
    """
    Create and fit a regression pipeline.
    
    Parameters
    ----------
    X : ndarray
        Features
    y : ndarray
        Targets
    degree : int, default=1
        Polynomial degree
    regularization : str, optional
        'l1', 'l2', or 'elasticnet'
    alpha : float, default=1.0
        Regularization strength
        
    Returns
    -------
    model : BaseRegressor
        Fitted model
    score : float
        R² score
    """
    if degree > 1:
        poly = PolynomialFeatures(degree=degree)
        X = poly.fit_transform(X)
    
    if regularization == 'l2':
        model = Ridge(alpha=alpha)
    elif regularization == 'l1':
        model = Lasso(alpha=alpha)
    elif regularization == 'elasticnet':
        model = ElasticNet(alpha=alpha)
    else:
        model = LinearRegression()
    
    model.fit(X, y)
    score = model.score(X, y)
    
    return model, score


# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.
