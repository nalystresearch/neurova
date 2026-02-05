# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Gaussian Process models for regression and classification."""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Callable


class GaussianProcessRegressor:
    """Gaussian Process Regression.
    
    Args:
        kernel: Kernel function or string ("rbf", "linear", "matern")
        alpha: Noise level (nugget term)
        n_restarts_optimizer: Number of optimizer restarts
        normalize_y: Whether to normalize target values
        
    Examples:
        gp = GaussianProcessRegressor(kernel="rbf", alpha=1e-10)
        gp.fit(X_train, y_train)
        y_pred, std = gp.predict(X_test, return_std=True)
    """
    
    def __init__(
        self,
        kernel: str | Callable = "rbf",
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
    ):
        self.kernel_name = kernel if isinstance(kernel, str) else "custom"
        self.kernel_func = kernel if callable(kernel) else None
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.y_train_mean_: float = 0.0
        self.y_train_std_: float = 1.0
        self.L_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        
        # kernel hyperparameters
        self.length_scale: float = 1.0
        self.nu: float = 1.5  # For Matern kernel
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Squared Exponential) kernel."""
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_norm + X2_norm - 2 * X1 @ X2.T
        return np.exp(-0.5 * distances_sq / (self.length_scale ** 2))
    
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Linear kernel."""
        return X1 @ X2.T
    
    def _matern_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern kernel (nu=1.5 version)."""
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances = np.sqrt(np.maximum(X1_norm + X2_norm - 2 * X1 @ X2.T, 0))
        
        sqrt3 = np.sqrt(3)
        scaled_dist = sqrt3 * distances / self.length_scale
        return (1.0 + scaled_dist) * np.exp(-scaled_dist)
    
    def _get_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Get kernel matrix between X1 and X2."""
        if self.kernel_func is not None:
            return self.kernel_func(X1, X2)
        elif self.kernel_name == "rbf":
            return self._rbf_kernel(X1, X2)
        elif self.kernel_name == "linear":
            return self._linear_kernel(X1, X2)
        elif self.kernel_name == "matern":
            return self._matern_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit Gaussian Process model.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.X_train_ = X
        
        # normalize y if requested
        if self.normalize_y:
            self.y_train_mean_ = float(np.mean(y))
            self.y_train_std_ = float(np.std(y))
            if self.y_train_std_ == 0:
                self.y_train_std_ = 1.0
            self.y_train_ = (y - self.y_train_mean_) / self.y_train_std_
        else:
            self.y_train_ = y
        
        # compute kernel matrix
        K = self._get_kernel(X, X)
        
        # add noise term (nugget)
        K[np.diag_indices_from(K)] += self.alpha
        
        # cholesky decomposition for numerical stability
        try:
            self.L_ = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # fall back to adding more noise
            K[np.diag_indices_from(K)] += 1e-6
            self.L_ = np.linalg.cholesky(K)
        
        # solve L * L.T * alpha = y
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, self.y_train_))
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        return_cov: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict using Gaussian Process.
        
        Args:
            X: Test data (n_samples, n_features)
            return_std: Whether to return standard deviation
            return_cov: Whether to return covariance
            
        Returns:
            Predictions, and optionally standard deviations or covariance
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.X_train_ is None or self.L_ is None or self.alpha_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        # compute kernel between test and training points
        K_trans = self._get_kernel(X, self.X_train_)
        
        # mean prediction
        y_mean = K_trans @ self.alpha_
        
        # denormalize if needed
        if self.normalize_y:
            y_mean = self.y_train_std_ * y_mean + self.y_train_mean_
        
        if not return_std and not return_cov:
            return y_mean
        
        # compute variance
        v = np.linalg.solve(self.L_, K_trans.T)
        
        if return_cov:
            # full covariance
            K_test = self._get_kernel(X, X)
            y_cov = K_test - v.T @ v
            
            if self.normalize_y:
                y_cov = self.y_train_std_ ** 2 * y_cov
            
            return y_mean, y_cov
        else:
            # standard deviation
            y_var = np.ones(X.shape[0]) - np.sum(v ** 2, axis=0)
            y_var = np.maximum(y_var, 0)  # Numerical stability
            y_std = np.sqrt(y_var)
            
            if self.normalize_y:
                y_std = self.y_train_std_ * y_std
            
            return y_mean, y_std
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R^2 score.
        
        Args:
            X: Test data
            y: True values
            
        Returns:
            R^2 score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)


class GaussianProcessClassifier:
    """Gaussian Process Classification (using Laplace approximation).
    
    Args:
        kernel: Kernel function or string
        n_restarts_optimizer: Number of optimizer restarts
        max_iter_predict: Maximum iterations for prediction
        
    Examples:
        gpc = GaussianProcessClassifier(kernel="rbf")
        gpc.fit(X_train, y_train)
        predictions = gpc.predict(X_test)
        probas = gpc.predict_proba(X_test)
    """
    
    def __init__(
        self,
        kernel: str | Callable = "rbf",
        n_restarts_optimizer: int = 0,
        max_iter_predict: int = 100,
    ):
        self.kernel_name = kernel if isinstance(kernel, str) else "custom"
        self.kernel_func = kernel if callable(kernel) else None
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        
        self.X_train_: Optional[np.ndarray] = None
        self.pi_: Optional[np.ndarray] = None
        self.W_sr_: Optional[np.ndarray] = None
        self.L_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        
        self.length_scale: float = 1.0
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel."""
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_norm + X2_norm - 2 * X1 @ X2.T
        return np.exp(-0.5 * distances_sq / (self.length_scale ** 2))
    
    def _get_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Get kernel matrix."""
        if self.kernel_func is not None:
            return self.kernel_func(X1, X2)
        elif self.kernel_name == "rbf":
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessClassifier:
        """Fit Gaussian Process classifier using Laplace approximation.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GPC only supports binary classification")
        
        # convert to -1/+1
        self.y_train_ = np.where(y == self.classes_[0], -1, 1).astype(np.float64)
        self.X_train_ = X
        
        # compute kernel
        K = self._get_kernel(X, X)
        
        # laplace approximation using Newton's method
        f = np.zeros(len(y))
        
        for _ in range(self.max_iter_predict):
            # compute pi (sigmoid probabilities)
            pi = self._sigmoid(self.y_train_ * f)
            W = pi * (1 - pi)  # Diagonal of Hessian
            W_sr = np.sqrt(W)
            
            # compute gradient
            grad = self.y_train_ * (0.5 - pi)
            
            # solve for f update
            # (K^{-1} + W) * delta_f = grad
            W_sr_K = W_sr[:, np.newaxis] * K
            B = np.eye(len(y)) + W_sr_K * W_sr[np.newaxis, :]
            
            try:
                L = np.linalg.cholesky(B)
                b = W * f + grad
                a = b - W_sr * np.linalg.solve(L.T, np.linalg.solve(L, W_sr * (K @ b)))
                f_new = K @ a
                
                # check convergence
                if np.max(np.abs(f_new - f)) < 1e-6:
                    break
                
                f = f_new
            except np.linalg.LinAlgError:
                break
        
        # store final values
        self.pi_ = self._sigmoid(self.y_train_ * f)
        W = self.pi_ * (1 - self.pi_)
        self.W_sr_ = np.sqrt(W)
        
        W_sr_K = self.W_sr_[:, np.newaxis] * K
        B = np.eye(len(y)) + W_sr_K * self.W_sr_[np.newaxis, :]
        self.L_ = np.linalg.cholesky(B)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, 2)
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.X_train_ is None or self.pi_ is None or self.y_train_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        K_trans = self._get_kernel(X, self.X_train_)
        
        # mean prediction
        f_star = K_trans @ (self.y_train_ * (self.pi_ - 0.5))
        
        # convert to probabilities
        p1 = self._sigmoid(f_star)
        p0 = 1 - p1
        
        return np.column_stack([p0, p1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        if self.classes_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        proba = self.predict_proba(X)
        predictions = np.where(proba[:, 1] > 0.5, self.classes_[1], self.classes_[0])
        return predictions


__all__ = ["GaussianProcessRegressor", "GaussianProcessClassifier"]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.