# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Support Vector Machines and kernel methods."""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Callable


class SVC:
    """Support Vector Classification using SMO algorithm (simplified).
    
    Args:
        C: Regularization parameter
        kernel: Kernel type ("linear", "rbf", "poly")
        gamma: Kernel coefficient for rbf/poly
        degree: Degree for polynomial kernel
        tol: Tolerance for stopping criterion
        max_iter: Maximum iterations
        
    Examples:
        svm = SVC(kernel="rbf", C=1.0)
        svm.fit(X_train, y_train)
        predictions = svm.predict(X_test)
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "rbf", "poly"] = "rbf",
        gamma: float | Literal["scale", "auto"] = "scale",
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = 1000,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma_param = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.gamma: float = 0.0
        self.classes_: Optional[np.ndarray] = None
    
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "rbf":
            # rBF kernel: exp(-gamma * ||x1 - x2||^2)
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * X1 @ X2.T
            return np.exp(-self.gamma * distances)
        elif self.kernel == "poly":
            # polynomial kernel: (gamma * <x1, x2> + 1)^degree
            return (self.gamma * (X1 @ X2.T) + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> SVC:
        """Fit SVM classifier using simplified SMO algorithm.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target labels (n_samples,) - must be 0/1 or -1/+1
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVC only supports binary classification")
        
        # convert labels to -1/+1
        y_coded = np.where(y == self.classes_[0], -1.0, 1.0)
        
        n_samples, n_features = X.shape
        
        # set gamma
        if self.gamma_param == "scale":
            self.gamma = 1.0 / (n_features * X.var())
        elif self.gamma_param == "auto":
            self.gamma = 1.0 / n_features
        else:
            self.gamma = float(self.gamma_param)
        
        # compute kernel matrix
        K = self._kernel_function(X, X)
        
        # initialize alphas
        alphas = np.zeros(n_samples)
        b = 0.0
        
        # simplified SMO (Sequential Minimal Optimization)
        for iteration in range(self.max_iter):
            alpha_changed = 0
            
            for i in range(n_samples):
                # calculate prediction
                f_i = np.sum(alphas * y_coded * K[i, :]) + b
                E_i = f_i - y_coded[i]
                
                # check KKT conditions
                if ((y_coded[i] * E_i < -self.tol and alphas[i] < self.C) or
                    (y_coded[i] * E_i > self.tol and alphas[i] > 0)):
                    
                    # select second alpha randomly
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    f_j = np.sum(alphas * y_coded * K[j, :]) + b
                    E_j = f_j - y_coded[j]
                    
                    # save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    
                    # compute bounds
                    if y_coded[i] != y_coded[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:
                        continue
                    
                    # compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # update alpha_j
                    alphas[j] -= y_coded[j] * (E_i - E_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # update alpha_i
                    alphas[i] += y_coded[i] * y_coded[j] * (alpha_j_old - alphas[j])
                    
                    # update bias
                    b1 = b - E_i - y_coded[i] * (alphas[i] - alpha_i_old) * K[i, i] - \
                         y_coded[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y_coded[i] * (alphas[i] - alpha_i_old) * K[i, j] - \
                         y_coded[j] * (alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    
                    alpha_changed += 1
            
            if alpha_changed == 0:
                break
        
        # store support vectors
        sv_indices = alphas > 1e-5
        self.support_ = np.where(sv_indices)[0]
        self.support_vectors_ = X[sv_indices]
        self.dual_coef_ = (alphas[sv_indices] * y_coded[sv_indices]).reshape(1, -1)
        self.intercept_ = b
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Decision function values (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.support_vectors_ is None or self.dual_coef_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        K = self._kernel_function(X, self.support_vectors_)
        return np.sum(K * self.dual_coef_, axis=1) + self.intercept_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        if self.classes_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        decision = self.decision_function(X)
        predictions = np.where(decision >= 0, self.classes_[1], self.classes_[0])
        return predictions


class SVR:
    """Support Vector Regression.
    
    Args:
        C: Regularization parameter
        epsilon: Epsilon in epsilon-SVR model
        kernel: Kernel type ("linear", "rbf", "poly")
        gamma: Kernel coefficient
        degree: Degree for polynomial kernel
        tol: Tolerance
        max_iter: Maximum iterations
        
    Examples:
        svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
        svr.fit(X_train, y_train)
        predictions = svr.predict(X_test)
    """
    
    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: Literal["linear", "rbf", "poly"] = "rbf",
        gamma: float | Literal["scale", "auto"] = "scale",
        degree: int = 3,
        tol: float = 1e-3,
        max_iter: int = 1000,
    ):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma_param = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.gamma: float = 0.0
    
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "rbf":
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * X1 @ X2.T
            return np.exp(-self.gamma * distances)
        elif self.kernel == "poly":
            return (self.gamma * (X1 @ X2.T) + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> SVR:
        """Fit SVR model (simplified implementation).
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        # set gamma
        if self.gamma_param == "scale":
            self.gamma = 1.0 / (n_features * X.var())
        elif self.gamma_param == "auto":
            self.gamma = 1.0 / n_features
        else:
            self.gamma = float(self.gamma_param)
        
        # compute kernel matrix
        K = self._kernel_function(X, X)
        
        # simplified: Use subset of points as support vectors
        # in practice, would solve QP problem
        # here we use a heuristic: points with large residuals
        
        # initial prediction (mean)
        y_mean = np.mean(y)
        residuals = np.abs(y - y_mean)
        
        # select points with residuals > epsilon
        sv_mask = residuals > self.epsilon
        if np.sum(sv_mask) == 0:
            # no support vectors needed, use mean prediction
            sv_mask[0] = True
        
        self.support_ = np.where(sv_mask)[0]
        self.support_vectors_ = X[sv_mask]
        
        # solve for dual coefficients (simplified)
        K_sv = K[np.ix_(sv_mask, sv_mask)]
        y_sv = y[sv_mask]
        
        # ridge regression on kernel matrix
        lambda_reg = 1.0 / (2 * self.C)
        coef = np.linalg.solve(K_sv + lambda_reg * np.eye(len(y_sv)), y_sv)
        
        self.dual_coef_ = coef.reshape(1, -1)
        self.intercept_ = y_mean - np.mean(K_sv @ coef)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.support_vectors_ is None or self.dual_coef_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        K = self._kernel_function(X, self.support_vectors_)
        return np.sum(K * self.dual_coef_, axis=1) + self.intercept_


class KernelRidge:
    """Kernel Ridge Regression.
    
    Args:
        alpha: Regularization strength
        kernel: Kernel type ("linear", "rbf", "poly")
        gamma: Kernel coefficient
        degree: Degree for polynomial kernel
        
    Examples:
        kr = KernelRidge(kernel="rbf", alpha=1.0)
        kr.fit(X_train, y_train)
        predictions = kr.predict(X_test)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        kernel: Literal["linear", "rbf", "poly"] = "linear",
        gamma: Optional[float] = None,
        degree: int = 3,
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        
        self.dual_coef_: Optional[np.ndarray] = None
        self.X_fit_: Optional[np.ndarray] = None
    
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "rbf":
            gamma = self.gamma if self.gamma is not None else 1.0 / X1.shape[1]
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * X1 @ X2.T
            return np.exp(-gamma * distances)
        elif self.kernel == "poly":
            gamma = self.gamma if self.gamma is not None else 1.0
            return (gamma * (X1 @ X2.T) + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> KernelRidge:
        """Fit kernel ridge regression.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.X_fit_ = X
        
        # compute kernel matrix
        K = self._kernel_function(X, X)
        
        # solve (K + alpha*I) * coef = y
        n = K.shape[0]
        self.dual_coef_ = np.linalg.solve(K + self.alpha * np.eye(n), y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.X_fit_ is None or self.dual_coef_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        K = self._kernel_function(X, self.X_fit_)
        return K @ self.dual_coef_


__all__ = ["SVC", "SVR", "KernelRidge"]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.