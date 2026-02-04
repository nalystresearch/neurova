# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Classification algorithms for machine learning

All implementations use NumPy only.
"""

import numpy as np
from typing import Optional, Union
from neurova.core.errors import ValidationError


class KNearestNeighbors:
    """
    K-Nearest Neighbors classifier
    
    Examples:
        >>> knn = KNearestNeighbors(n_neighbors=5)
        >>> knn.fit(X_train, y_train)
        >>> y_pred = knn.predict(X_test)
    """
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        """
        Args:
            n_neighbors: Number of neighbors to use
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train_ = None
        self.y_train_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestNeighbors':
        """Store training data"""
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        return self
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute distances between X and training data"""
        if self.X_train_ is None:
            raise ValidationError('classifier', 'not fitted', 'fitted classifier')
        
        if self.metric == 'euclidean':
            # vectorized euclidean distance
            distances = np.sqrt(((X[:, np.newaxis] - self.X_train_) ** 2).sum(axis=2))
        elif self.metric == 'manhattan':
            distances = np.abs(X[:, np.newaxis] - self.X_train_).sum(axis=2)
        elif self.metric == 'cosine':
            # cosine similarity
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            train_norm = self.X_train_ / (np.sqrt((self.X_train_ ** 2).sum(axis=1, keepdims=True)) + 1e-8)
            similarities = X_norm @ train_norm.T
            distances = 1 - similarities
        else:
            raise ValidationError('metric', self.metric, 'euclidean, manhattan, or cosine')
        
        return distances
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if self.X_train_ is None or self.y_train_ is None:
            raise ValidationError('classifier', 'not fitted', 'fitted classifier')
        
        X = np.asarray(X)
        distances = self._compute_distances(X)
        
        # get k nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self.y_train_[nearest_indices]
        
        # majority vote
        predictions = np.array([
            np.bincount(labels).argmax() for labels in nearest_labels
        ])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.y_train_ is None:
            raise ValidationError('classifier', 'not fitted', 'fitted classifier')
        
        X = np.asarray(X)
        distances = self._compute_distances(X)
        
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self.y_train_[nearest_indices]
        
        n_classes = len(np.unique(self.y_train_))
        n_samples = X.shape[0]
        
        probabilities = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            counts = np.bincount(nearest_labels[i], minlength=n_classes)
            probabilities[i] = counts / self.n_neighbors
        
        return probabilities


class LogisticRegression:
    """
    Logistic Regression classifier
    
    Examples:
        >>> lr = LogisticRegression()
        >>> lr.fit(X_train, y_train)
        >>> y_pred = lr.predict(X_test)
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: float = 0.0):
        """
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations
            regularization: L2 regularization parameter
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights_ = None
        self.bias_ = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """Train the model using gradient descent"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        # initialize weights
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        
        # gradient descent
        for _ in range(self.n_iterations):
            # forward pass
            linear_model = X @ self.weights_ + self.bias_
            y_pred = self._sigmoid(linear_model)
            
            # compute gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y)) + self.regularization * self.weights_
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # update parameters
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.weights_ is None or self.bias_ is None:
            raise ValidationError('classifier', 'not fitted', 'fitted classifier')
        
        X = np.asarray(X, dtype=np.float64)
        linear_model = X @ self.weights_ + self.bias_
        probabilities = self._sigmoid(linear_model)
        
        return np.vstack([1 - probabilities, probabilities]).T
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)


class NaiveBayes:
    """
    Gaussian Naive Bayes classifier
    
    Examples:
        >>> nb = NaiveBayes()
        >>> nb.fit(X_train, y_train)
        >>> y_pred = nb.predict(X_test)
    """
    
    def __init__(self):
        self.classes_ = None
        self.class_prior_ = None
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayes':
        """Fit Gaussian Naive Bayes"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        
        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.mean_[idx] = X_cls.mean(axis=0)
            self.var_[idx] = X_cls.var(axis=0) + 1e-9  # Add small value for numerical stability
            self.class_prior_[idx] = X_cls.shape[0] / X.shape[0]
        
        return self
    
    def _gaussian_probability(self, x: np.ndarray, mean: np.ndarray, 
                             var: np.ndarray) -> np.ndarray:
        """Calculate Gaussian probability"""
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return exponent / np.sqrt(2 * np.pi * var)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.classes_ is None or self.mean_ is None or self.var_ is None or self.class_prior_ is None:
            raise ValidationError('classifier', 'not fitted', 'fitted classifier')
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        probabilities = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            prior = np.log(self.class_prior_[idx])
            likelihood = np.sum(np.log(self._gaussian_probability(
                X, self.mean_[idx], self.var_[idx]
            )), axis=1)
            probabilities[:, idx] = prior + likelihood
        
        # convert log probabilities to probabilities
        probabilities = np.exp(probabilities - probabilities.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if self.classes_ is None:
            raise ValidationError('classifier', 'not fitted', 'fitted classifier')
        
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class DecisionTreeClassifier:
    """
    Simple Decision Tree classifier using Gini impurity
    
    Examples:
        >>> tree = DecisionTreeClassifier(max_depth=5)
        >>> tree.fit(X_train, y_train)
        >>> y_pred = tree.predict(X_test)
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        """
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _split(self, X: np.ndarray, y: np.ndarray, feature: int, threshold: float):
        """Split dataset"""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the best split"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask, right_mask = self._split(X, y, feature, threshold)
                
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                
                # calculate weighted Gini impurity
                n = len(y)
                gini = (np.sum(left_mask) / n) * self._gini_impurity(y[left_mask]) + \
                       (np.sum(right_mask) / n) * self._gini_impurity(y[right_mask])
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Recursively build the tree"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # stopping criteria
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            return {'class': np.bincount(y).argmax()}
        
        # find best split
        feature, threshold = self._find_best_split(X, y)
        
        if feature is None or threshold is None:
            return {'class': np.bincount(y).argmax()}
        
        # split dataset
        left_mask, right_mask = self._split(X, y, feature, threshold)
        
        # build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """Build decision tree"""
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        self.tree_ = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, tree: dict) -> int:
        """Predict single sample"""
        if 'class' in tree:
            return tree['class']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if self.tree_ is None:
            raise ValidationError('classifier', 'not fitted', 'fitted classifier')
        
        X = np.asarray(X)
        return np.array([self._predict_sample(x, self.tree_) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (placeholder for ensemble compatibility)."""
        if self.tree_ is None:
            raise ValidationError('classifier', 'not fitted', 'fitted classifier')
        
        X = np.asarray(X)
        predictions = self.predict(X)
        n_samples = X.shape[0]
        
        # Return one-hot encoded probabilities for compatibility
        proba = np.zeros((n_samples, 2))
        proba[np.arange(n_samples), predictions] = 1.0
        return proba


class DecisionTreeRegressor:
    """
    Decision Tree regressor using MSE impurity.
    
    A decision tree regressor that recursively partitions the feature space
    to minimize mean squared error at each node.
    
    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
        
    Attributes
    ----------
    tree_ : dict
        The learned decision tree structure
        
    Examples
    --------
    >>> tree = DecisionTreeRegressor(max_depth=5)
    >>> tree.fit(X_train, y_train)
    >>> y_pred = tree.predict(X_test)
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.n_features_in_ = None
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error (variance) for a node."""
        if len(y) == 0:
            return 0.0
        return np.var(y)
    
    def _split(self, X: np.ndarray, y: np.ndarray, feature: int, threshold: float):
        """Split dataset at the given feature and threshold."""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the best split that minimizes MSE."""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            # Get unique sorted values as potential thresholds
            thresholds = np.unique(X[:, feature])
            
            # Use midpoints between consecutive values
            if len(thresholds) > 1:
                thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            for threshold in thresholds:
                left_mask, right_mask = self._split(X, y, feature, threshold)
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                # Check minimum samples constraints
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                if n_left < self.min_samples_split or n_right < self.min_samples_split:
                    continue
                
                # Calculate weighted MSE
                mse = (n_left / n_samples) * self._mse(y[left_mask]) + \
                      (n_right / n_samples) * self._mse(y[right_mask])
                
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        """Recursively build the regression tree."""
        n_samples = len(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            np.var(y) < 1e-10):  # Pure node
            return {'value': np.mean(y), 'n_samples': n_samples}
        
        # Find best split
        feature, threshold = self._find_best_split(X, y)
        
        if feature is None or threshold is None:
            return {'value': np.mean(y), 'n_samples': n_samples}
        
        # Split dataset
        left_mask, right_mask = self._split(X, y, feature, threshold)
        
        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'n_samples': n_samples,
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """Build decision tree regressor from training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, tree: dict) -> float:
        """Predict value for a single sample."""
        if 'value' in tree:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values
        """
        if self.tree_ is None:
            raise ValidationError('regressor', 'not fitted', 'fitted regressor')
        
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_sample(x, self.tree_) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
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


# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.