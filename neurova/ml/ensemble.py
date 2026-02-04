# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Ensemble learning methods for Neurova."""

from __future__ import annotations
import numpy as np
from typing import Optional, Literal
from neurova.ml.classification import DecisionTreeClassifier
from neurova.core.errors import ValidationError


class RandomForestClassifier:
    """Random Forest classifier using bagging and feature randomness.
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples required to split
        max_features: Number of features to consider for splits ("sqrt", "log2", or int)
        bootstrap: Whether to use bootstrap samples
        random_state: Random seed
        
    Examples:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: str | int = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.trees: list[DecisionTreeClassifier] = []
        self.n_classes: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_n_features(self, n_total: int) -> int:
        """Get number of features to use."""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_total)
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_total))
        elif self.max_features == "log2":
            return int(np.log2(n_total))
        else:
            return n_total
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Fit random forest classifier.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        self.n_classes = len(np.unique(y))
        self.trees = []
        
        # calculate max features
        max_feat = self._get_n_features(n_features)
        
        # build trees
        for _ in range(self.n_estimators):
            # bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            # random feature subset
            feature_indices = np.random.choice(n_features, max_feat, replace=False)
            X_subset = X_sample[:, feature_indices]
            
            # train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_subset, y_sample)
            tree.feature_indices = feature_indices  # Store which features were used
            self.trees.append(tree)
        
        # calculate feature importances (average across trees)
        self._compute_feature_importances(n_features)
        
        return self
    
    def _compute_feature_importances(self, n_features: int) -> None:
        """Compute feature importances from all trees."""
        importances = np.zeros(n_features)
        
        for tree in self.trees:
            if hasattr(tree, 'feature_importances_') and tree.feature_importances_ is not None:
                # map tree importances back to original feature indices
                for idx, feat_idx in enumerate(tree.feature_indices):
                    if idx < len(tree.feature_importances_):
                        importances[feat_idx] += tree.feature_importances_[idx]
        
        # normalize
        if np.sum(importances) > 0:
            importances /= np.sum(importances)
        
        self.feature_importances_ = importances
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # aggregate predictions from all trees
        all_proba = np.zeros((n_samples, self.n_classes))
        
        for tree in self.trees:
            # get predictions on relevant features
            X_subset = X[:, tree.feature_indices]
            tree_proba = tree.predict_proba(X_subset)
            all_proba += tree_proba
        
        # average
        all_proba /= self.n_estimators
        
        return all_proba


class GradientBoostingClassifier:
    """Gradient Boosting classifier.
    
    Builds an additive model in a forward stage-wise fashion.
    
    Args:
        n_estimators: Number of boosting stages
        learning_rate: Shrinks contribution of each tree
        max_depth: Maximum depth of individual trees
        min_samples_split: Minimum samples to split
        subsample: Fraction of samples for fitting trees
        random_state: Random seed
        
    Examples:
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        gb.fit(X_train, y_train)
        predictions = gb.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        
        self.trees: list[DecisionTreeClassifier] = []
        self.init_pred: float = 0.0
        self.n_classes: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifier:
        """Fit gradient boosting classifier.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        self.n_classes = len(np.unique(y))
        
        # binary classification for now
        if self.n_classes > 2:
            raise NotImplementedError("Multi-class gradient boosting not yet implemented")
        
        # initialize with log-odds
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        self.init_pred = np.log((n_pos + 1e-10) / (n_neg + 1e-10))
        
        # initialize predictions
        F = np.full(n_samples, self.init_pred)
        
        # build trees
        self.trees = []
        for _ in range(self.n_estimators):
            # compute negative gradient (residuals)
            p = self._sigmoid(F)
            residuals = y - p
            
            # subsample
            if self.subsample < 1.0:
                n_subsample = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, n_subsample, replace=False)
                X_sub = X[indices]
                res_sub = residuals[indices]
            else:
                X_sub = X
                res_sub = residuals
            
            # fit tree to residuals
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            # use residuals as pseudo-targets (convert to binary for tree)
            y_pseudo = (res_sub > 0).astype(int)
            tree.fit(X_sub, y_pseudo)
            self.trees.append(tree)
            
            # update predictions
            tree_pred = tree.predict_proba(X)[:, 1] - 0.5  # Center around 0
            F += self.learning_rate * tree_pred * 2  # Scale
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, 2)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # initialize with base prediction
        F = np.full(n_samples, self.init_pred)
        
        # add tree predictions
        for tree in self.trees:
            tree_pred = tree.predict_proba(X)[:, 1] - 0.5
            F += self.learning_rate * tree_pred * 2
        
        # convert to probabilities
        p1 = self._sigmoid(F)
        p0 = 1 - p1
        
        return np.column_stack([p0, p1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class AdaBoostClassifier:
    """AdaBoost (Adaptive Boosting) classifier.
    
    Args:
        n_estimators: Number of boosting stages
        learning_rate: Weight applied to each classifier
        random_state: Random seed
        
    Examples:
        ada = AdaBoostClassifier(n_estimators=50)
        ada.fit(X_train, y_train)
        predictions = ada.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.estimators: list[DecisionTreeClassifier] = []
        self.estimator_weights: list[float] = []
        self.n_classes: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> AdaBoostClassifier:
        """Fit AdaBoost classifier.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target labels (n_samples,) - must be 0/1
            
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        self.n_classes = len(np.unique(y))
        if self.n_classes != 2:
            raise ValidationError("n_classes", self.n_classes, "2 (binary classification only)")
        
        # convert labels to -1/+1
        y_coded = 2 * y - 1
        
        # initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples
        
        self.estimators = []
        self.estimator_weights = []
        
        for _ in range(self.n_estimators):
            # train weak learner (decision stump)
            tree = DecisionTreeClassifier(max_depth=1)  # Stump
            
            # sample with weights (bootstrap with replacement)
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )
            tree.fit(X[indices], y[indices])
            
            # predict on full dataset
            y_pred = tree.predict(X)
            y_pred_coded = 2 * y_pred - 1
            
            # calculate error
            incorrect = (y_pred_coded != y_coded)
            error = np.sum(sample_weights * incorrect)
            
            # avoid division by zero
            if error >= 0.5:
                break
            
            # calculate estimator weight
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))
            
            # update sample weights
            sample_weights *= np.exp(-estimator_weight * y_coded * y_pred_coded)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators.append(tree)
            self.estimator_weights.append(estimator_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # weighted vote
        predictions = np.zeros(n_samples)
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            y_pred = estimator.predict(X)
            y_pred_coded = 2 * y_pred - 1
            predictions += weight * y_pred_coded
        
        # convert back to 0/1
        return (predictions > 0).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, 2)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # weighted vote
        scores = np.zeros(n_samples)
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            y_pred = estimator.predict(X)
            y_pred_coded = 2 * y_pred - 1
            scores += weight * y_pred_coded
        
        # convert to probabilities using sigmoid
        p1 = 1.0 / (1.0 + np.exp(-scores))
        p0 = 1 - p1
        
        return np.column_stack([p0, p1])


class BaggingClassifier:
    """Bagging (Bootstrap Aggregating) classifier.
    
    Args:
        base_estimator: Base estimator to use (default: DecisionTreeClassifier)
        n_estimators: Number of estimators
        max_samples: Number/fraction of samples to draw
        max_features: Number/fraction of features to draw
        bootstrap: Whether to use bootstrap samples
        random_state: Random seed
        
    Examples:
        bag = BaggingClassifier(n_estimators=10)
        bag.fit(X_train, y_train)
        predictions = bag.predict(X_test)
    """
    
    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.estimators: list = []
        self.n_classes: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaggingClassifier:
        """Fit bagging classifier.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        self.n_classes = len(np.unique(y))
        
        # calculate sample and feature sizes
        if self.max_samples <= 1.0:
            n_samples_est = int(self.max_samples * n_samples)
        else:
            n_samples_est = int(self.max_samples)
        
        if self.max_features <= 1.0:
            n_features_est = int(self.max_features * n_features)
        else:
            n_features_est = int(self.max_features)
        
        self.estimators = []
        
        for _ in range(self.n_estimators):
            # bootstrap samples
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, n_samples_est, replace=True)
            else:
                sample_indices = np.random.choice(n_samples, n_samples_est, replace=False)
            
            # random features
            feature_indices = np.random.choice(n_features, n_features_est, replace=False)
            
            # create estimator
            if self.base_estimator is None:
                estimator = DecisionTreeClassifier()
            else:
                # clone base estimator
                estimator = self.base_estimator.__class__(**self.base_estimator.__dict__)
            
            # train on subset
            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]
            estimator.fit(X_subset, y_subset)
            
            # store with feature indices
            estimator.feature_indices = feature_indices
            self.estimators.append(estimator)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        all_proba = np.zeros((n_samples, self.n_classes))
        
        for estimator in self.estimators:
            X_subset = X[:, estimator.feature_indices]
            proba = estimator.predict_proba(X_subset)
            all_proba += proba
        
        all_proba /= self.n_estimators
        
        return all_proba


class RandomForestRegressor:
    """
    Random Forest regressor using bootstrap aggregation.
    
    A random forest is a meta estimator that fits a number of decision tree
    regressors on various sub-samples of the dataset and uses averaging
    to improve predictive accuracy and control over-fitting.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=10
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    max_features : str or float, default='sqrt'
        Number of features to consider for each split:
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - float: fraction of features
    bootstrap : bool, default=True
        Whether to use bootstrap samples
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes
    ----------
    estimators_ : list
        List of fitted decision tree regressors
    feature_importances_ : ndarray of shape (n_features,)
        Impurity-based feature importances
        
    Examples
    --------
    >>> rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    >>> rf.fit(X_train, y_train)
    >>> y_pred = rf.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: Union[str, float] = 'sqrt',
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.estimators_: list = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_n_features(self, n_features: int) -> int:
        """Calculate number of features to select."""
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == 'log2':
                return max(1, int(np.log2(n_features)))
            elif self.max_features == 'auto':
                return n_features
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return n_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        """Build a forest of trees from training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : RandomForestRegressor
            Fitted estimator
        """
        from neurova.ml.classification import DecisionTreeRegressor
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        self.n_features_in_ = n_features
        n_features_split = self._get_n_features(n_features)
        
        self.estimators_ = []
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                sample_indices = np.arange(n_samples)
            
            # Random feature subset for this tree
            feature_indices = np.random.choice(n_features, n_features_split, replace=False)
            
            # Create and train tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            
            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]
            tree.fit(X_subset, y_subset)
            
            # Store feature indices for prediction
            tree.feature_indices_ = feature_indices
            self.estimators_.append(tree)
        
        # Calculate feature importances (based on how often features are used)
        self._calculate_feature_importances(n_features)
        
        return self
    
    def _calculate_feature_importances(self, n_features: int):
        """Calculate impurity-based feature importances."""
        importances = np.zeros(n_features)
        for tree in self.estimators_:
            for idx in tree.feature_indices_:
                importances[idx] += 1
        importances = importances / importances.sum()
        self.feature_importances_ = importances
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values (average of all trees)
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        predictions = np.zeros((n_samples, self.n_estimators))
        
        for i, tree in enumerate(self.estimators_):
            X_subset = X[:, tree.feature_indices_]
            predictions[:, i] = tree.predict(X_subset)
        
        return np.mean(predictions, axis=1)
    
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


class GradientBoostingRegressor:
    """
    Gradient Boosting regressor for regression problems.
    
    Gradient Boosting builds an additive model in a forward stage-wise fashion.
    At each stage, a decision tree is fit on the negative gradient of the loss
    function (mean squared error).
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages
    learning_rate : float, default=0.1
        Shrinkage factor for each tree's contribution
    max_depth : int, default=3
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    subsample : float, default=1.0
        Fraction of samples for fitting each tree
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes
    ----------
    estimators_ : list
        List of fitted decision tree regressors
    train_score_ : ndarray of shape (n_estimators,)
        Training scores at each iteration
    init_ : float
        Initial prediction (mean of training targets)
        
    Examples
    --------
    >>> gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    >>> gb.fit(X_train, y_train)
    >>> y_pred = gb.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        
        self.estimators_: list = []
        self.train_score_: list = []
        self.init_: float = 0.0
        self.n_features_in_: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRegressor':
        """Build gradient boosting model from training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : GradientBoostingRegressor
            Fitted estimator
        """
        from neurova.ml.classification import DecisionTreeRegressor
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        self.n_features_in_ = n_features
        
        # Initialize with mean
        self.init_ = np.mean(y)
        F = np.full(n_samples, self.init_)
        
        self.estimators_ = []
        self.train_score_ = []
        
        for _ in range(self.n_estimators):
            # Compute negative gradient (residuals for squared error loss)
            residuals = y - F
            
            # Subsample
            if self.subsample < 1.0:
                n_subsample = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, n_subsample, replace=False)
                X_sub = X[indices]
                res_sub = residuals[indices]
            else:
                X_sub = X
                res_sub = residuals
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_sub, res_sub)
            self.estimators_.append(tree)
            
            # Update predictions
            F += self.learning_rate * tree.predict(X)
            
            # Track training score
            mse = np.mean((y - F) ** 2)
            self.train_score_.append(mse)
        
        self.train_score_ = np.array(self.train_score_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        F = np.full(n_samples, self.init_)
        
        # Add tree predictions
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        
        return F
    
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
    
    def staged_predict(self, X: np.ndarray):
        """Yield predictions at each stage.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Yields
        ------
        y : ndarray of shape (n_samples,)
            Predicted values at each stage
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        F = np.full(n_samples, self.init_)
        
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
            yield F.copy()


class BaggingRegressor:
    """
    Bagging regressor using bootstrap aggregation.
    
    A bagging regressor fits base regressors on random subsets of the
    original dataset and then aggregates predictions by averaging.
    
    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on subsets. If None, uses DecisionTreeRegressor
    n_estimators : int, default=10
        Number of base estimators
    max_samples : float, default=1.0
        Number of samples to draw (as fraction)
    max_features : float, default=1.0
        Number of features to draw (as fraction)
    bootstrap : bool, default=True
        Whether to use bootstrap samples
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Examples
    --------
    >>> bag = BaggingRegressor(n_estimators=10)
    >>> bag.fit(X_train, y_train)
    >>> y_pred = bag.predict(X_test)
    """
    
    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.estimators_: list = []
        self.n_features_in_: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingRegressor':
        """Build a bagging ensemble from training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : BaggingRegressor
            Fitted estimator
        """
        from neurova.ml.classification import DecisionTreeRegressor
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        self.n_features_in_ = n_features
        
        # Calculate sample and feature sizes
        if self.max_samples <= 1.0:
            n_samples_est = int(self.max_samples * n_samples)
        else:
            n_samples_est = int(self.max_samples)
        
        if self.max_features <= 1.0:
            n_features_est = int(self.max_features * n_features)
        else:
            n_features_est = int(self.max_features)
        
        self.estimators_ = []
        
        for _ in range(self.n_estimators):
            # Bootstrap samples
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, n_samples_est, replace=True)
            else:
                sample_indices = np.random.choice(n_samples, n_samples_est, replace=False)
            
            # Random features
            feature_indices = np.random.choice(n_features, n_features_est, replace=False)
            
            # Create estimator
            if self.base_estimator is None:
                estimator = DecisionTreeRegressor()
            else:
                estimator = self.base_estimator.__class__(**self.base_estimator.__dict__)
            
            # Train on subset
            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]
            estimator.fit(X_subset, y_subset)
            
            # Store with feature indices
            estimator.feature_indices_ = feature_indices
            self.estimators_.append(estimator)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values (average of all estimators)
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        predictions = np.zeros((n_samples, self.n_estimators))
        
        for i, estimator in enumerate(self.estimators_):
            X_subset = X[:, estimator.feature_indices_]
            predictions[:, i] = estimator.predict(X_subset)
        
        return np.mean(predictions, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)


class AdaBoostRegressor:
    """
    AdaBoost regressor using the AdaBoost.R2 algorithm.
    
    An AdaBoost regressor that combines weak learners to create
    a stronger regressor, using sample weighting.
    
    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting stages
    learning_rate : float, default=1.0
        Weight applied to each regressor
    max_depth : int, default=3
        Maximum depth of each decision tree
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes
    ----------
    estimators_ : list
        List of fitted weak learners
    estimator_weights_ : ndarray of shape (n_estimators,)
        Weights assigned to each estimator
        
    Examples
    --------
    >>> ada = AdaBoostRegressor(n_estimators=50)
    >>> ada.fit(X_train, y_train)
    >>> y_pred = ada.predict(X_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 3,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.estimators_: list = []
        self.estimator_weights_: list = []
        self.n_features_in_: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostRegressor':
        """Build AdaBoost regressor from training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : AdaBoostRegressor
            Fitted estimator
        """
        from neurova.ml.classification import DecisionTreeRegressor
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        self.n_features_in_ = n_features
        
        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.estimator_weights_ = []
        
        for _ in range(self.n_estimators):
            # Train weak learner with weighted samples
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            
            # Sample with weights (bootstrap)
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )
            tree.fit(X[indices], y[indices])
            
            # Predict
            y_pred = tree.predict(X)
            
            # Calculate weighted error
            error = np.abs(y - y_pred)
            max_error = np.max(error) + 1e-10
            relative_error = error / max_error  # Normalize to [0, 1]
            
            # Weighted average loss
            weighted_loss = np.sum(sample_weights * relative_error)
            
            # Avoid division by zero
            if weighted_loss >= 0.5:
                # Too much error, stop
                break
            
            # Estimator weight (beta)
            beta = weighted_loss / (1 - weighted_loss + 1e-10)
            estimator_weight = self.learning_rate * np.log(1 / (beta + 1e-10))
            
            # Update sample weights
            sample_weights *= np.power(beta, 1 - relative_error)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators_.append(tree)
            self.estimator_weights_.append(estimator_weight)
        
        self.estimator_weights_ = np.array(self.estimator_weights_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.
        
        Uses weighted median of all estimator predictions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Collect all predictions
        predictions = np.array([est.predict(X) for est in self.estimators_]).T
        
        # Weighted average prediction
        weights = self.estimator_weights_ / np.sum(self.estimator_weights_)
        y_pred = np.sum(predictions * weights, axis=1)
        
        return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)


__all__ = [
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "AdaBoostClassifier",
    "BaggingClassifier",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "AdaBoostRegressor",
    "BaggingRegressor",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.