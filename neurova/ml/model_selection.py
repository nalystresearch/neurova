# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Model selection utilities for machine learning

All implementations use NumPy only.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from neurova.core.errors import ValidationError


def train_test_split(X: np.ndarray, y: Optional[np.ndarray] = None,
                    test_size: float = 0.25, random_state: Optional[int] = None,
                    shuffle: bool = True) -> Tuple:
    """
    Split arrays into random train and test subsets
    
    Args:
        X: Features array
        y: Target array (optional)
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility
        shuffle: Whether to shuffle data before splitting
        
    Returns:
        Tuple of train-test splits
    
    Examples:
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    if y is not None:
        y = np.asarray(y)
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, X_test, y_train, y_test
    
    return X_train, X_test


def cross_validate(estimator, X: np.ndarray, y: np.ndarray, 
                  cv: int = 5, scoring: str = 'accuracy') -> Dict[str, np.ndarray]:
    """
    Evaluate metric(s) by cross-validation
    
    Args:
        estimator: Estimator object with fit and predict methods
        X: Features array
        y: Target array
        cv: Number of folds
        scoring: Scoring metric ('accuracy', 'precision', 'recall', 'f1')
        
    Returns:
        Dictionary with test scores and fit times
    
    Examples:
        >>> from neurova.ml import KNearestNeighbors
        >>> knn = KNearestNeighbors(n_neighbors=5)
        >>> scores = cross_validate(knn, X, y, cv=5)
    """
    from neurova.ml.metrics import (
        accuracy_score, precision_score, recall_score, f1_score
    )
    import time
    
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]
    
    # create fold indices
    fold_size = n_samples // cv
    indices = np.arange(n_samples)
    
    test_scores = []
    fit_times = []
    
    for fold in range(cv):
        # create train/test split for this fold
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < cv - 1 else n_samples
        
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # fit and predict
        start_time = time.time()
        estimator.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        y_pred = estimator.predict(X_test)
        
        # calculate score
        if scoring == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif scoring == 'precision':
            score = precision_score(y_test, y_pred, average='macro')
        elif scoring == 'recall':
            score = recall_score(y_test, y_pred, average='macro')
        elif scoring == 'f1':
            score = f1_score(y_test, y_pred, average='macro')
        else:
            raise ValidationError('scoring', scoring, 'accuracy, precision, recall, or f1')
        
        test_scores.append(score)
        fit_times.append(fit_time)
    
    return {
        'test_score': np.array(test_scores),
        'fit_time': np.array(fit_times),
    }


class GridSearchCV:
    """
    Exhaustive search over specified parameter values
    
    Examples:
        >>> from neurova.ml import KNearestNeighbors
        >>> param_grid = {'n_neighbors': [3, 5, 7]}
        >>> grid = GridSearchCV(KNearestNeighbors(), param_grid, cv=5)
        >>> grid.fit(X, y)
        >>> print(grid.best_params_)
    """
    
    def __init__(self, estimator, param_grid: Dict[str, List],
                 cv: int = 5, scoring: str = 'accuracy'):
        """
        Args:
            estimator: Estimator object (must be class, not instance)
            param_grid: Dictionary with parameters as keys and lists of values
            cv: Number of cross-validation folds
            scoring: Scoring metric
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None
    
    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all combinations of parameters"""
        import itertools
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GridSearchCV':
        """Run grid search"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        param_combinations = self._generate_param_combinations()
        
        results = []
        best_score = -np.inf
        best_params = None
        
        for params in param_combinations:
            # create estimator with these parameters
            estimator = self.estimator(**params)
            
            # cross-validate
            cv_results = cross_validate(estimator, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = cv_results['test_score'].mean()
            
            results.append({
                'params': params,
                'mean_test_score': mean_score,
                'std_test_score': cv_results['test_score'].std(),
                'mean_fit_time': cv_results['fit_time'].mean(),
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        # fit best estimator on full dataset
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = self.estimator(**best_params)
        self.best_estimator_.fit(X, y)
        self.cv_results_ = results
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best estimator"""
        if self.best_estimator_ is None:
            raise ValidationError('grid_search', 'not fitted', 'fitted grid search')
        
        return self.best_estimator_.predict(X)


class RandomizedSearchCV:
    """
    Randomized search on hyperparameters.
    
    Samples parameter settings from distributions and evaluates
    performance using cross-validation.
    
    Parameters
    ----------
    estimator : estimator object
        Estimator class (must be class, not instance)
    param_distributions : dict
        Dictionary with parameters as keys and distributions or lists as values.
        For lists, values are sampled uniformly. For distributions, the rvs
        method is called for sampling.
    n_iter : int, default=10
        Number of parameter settings sampled
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='accuracy'
        Scoring metric ('accuracy', 'precision', 'recall', 'f1')
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes
    ----------
    best_params_ : dict
        Parameter setting with best score
    best_score_ : float
        Mean cross-validated score of best_estimator
    best_estimator_ : estimator
        Estimator fitted on full data with best parameters
    cv_results_ : list of dict
        Results for each parameter setting
        
    Examples
    --------
    >>> from neurova.ml import KNearestNeighbors
    >>> from neurova.ml.model_selection import RandomizedSearchCV
    >>> param_dist = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
    >>> search = RandomizedSearchCV(KNearestNeighbors, param_dist, n_iter=5, cv=3)
    >>> search.fit(X, y)
    >>> print(search.best_params_)
    """
    
    def __init__(
        self,
        estimator,
        param_distributions: Dict[str, Any],
        n_iter: int = 10,
        cv: int = 5,
        scoring: str = 'accuracy',
        random_state: Optional[int] = None,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.best_estimator_: Optional[Any] = None
        self.cv_results_: Optional[List[Dict]] = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sample_params(self) -> Dict[str, Any]:
        """Sample a single parameter setting."""
        params = {}
        for key, dist in self.param_distributions.items():
            if hasattr(dist, 'rvs'):
                # scipy distribution - call rvs
                params[key] = dist.rvs()
            elif isinstance(dist, (list, tuple, np.ndarray)):
                # list - sample uniformly
                params[key] = np.random.choice(dist)
            else:
                # single value
                params[key] = dist
        return params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomizedSearchCV':
        """Run randomized search.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : RandomizedSearchCV
            Fitted search object
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        results = []
        best_score = -np.inf
        best_params = None
        
        for _ in range(self.n_iter):
            # Sample parameters
            params = self._sample_params()
            
            # Create estimator with these parameters
            estimator = self.estimator(**params)
            
            # Cross-validate
            cv_results = cross_validate(estimator, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = cv_results['test_score'].mean()
            
            results.append({
                'params': params,
                'mean_test_score': mean_score,
                'std_test_score': cv_results['test_score'].std(),
                'mean_fit_time': cv_results['fit_time'].mean(),
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        # Fit best estimator on full dataset
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = self.estimator(**best_params)
        self.best_estimator_.fit(X, y)
        self.cv_results_ = results
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y : ndarray
            Predicted values
        """
        if self.best_estimator_ is None:
            raise ValidationError('random_search', 'not fitted', 'fitted search')
        
        return self.best_estimator_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return score of best estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        y : array-like of shape (n_samples,)
            True values
            
        Returns
        -------
        score : float
            Score of best estimator
        """
        if self.best_estimator_ is None:
            raise ValidationError('random_search', 'not fitted', 'fitted search')
        
        return self.best_estimator_.score(X, y)


class KFold:
    """
    K-Fold cross-validation iterator.
    
    Provides train/test indices to split data into train/test sets.
    The dataset is divided into k consecutive folds.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting
    random_state : int or None, default=None
        Random seed for shuffling
        
    Examples
    --------
    >>> from neurova.ml.model_selection import KFold
    >>> kf = KFold(n_splits=5, shuffle=True, random_state=42)
    >>> for train_idx, test_idx in kf.split(X):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target variable (not used, for API compatibility)
        groups : array-like of shape (n_samples,), optional
            Group labels (not used, for API compatibility)
            
        Yields
        ------
        train : ndarray
            Training indices for this fold
        test : ndarray
            Test indices for this fold
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_splits - 1 else n_samples
            
            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, 
                     groups: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Fold cross-validation iterator.
    
    Provides train/test indices to split data into train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage
    of samples for each class.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting
    random_state : int or None, default=None
        Random seed for shuffling
        
    Examples
    --------
    >>> from neurova.ml.model_selection import StratifiedKFold
    >>> skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> for train_idx, test_idx in skf.split(X, y):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target variable (used for stratification)
        groups : array-like of shape (n_samples,), optional
            Group labels (not used)
            
        Yields
        ------
        train : ndarray
            Training indices for this fold
        test : ndarray
            Test indices for this fold
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        # Get unique classes and their indices
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        
        # Get indices for each class
        class_indices = [np.where(y_indices == i)[0] for i in range(n_classes)]
        
        # Shuffle within each class if requested
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            for indices in class_indices:
                rng.shuffle(indices)
        
        # Build test indices for each fold
        test_folds = [[] for _ in range(self.n_splits)]
        
        for cls_indices in class_indices:
            n_cls = len(cls_indices)
            fold_sizes = np.full(self.n_splits, n_cls // self.n_splits)
            fold_sizes[:n_cls % self.n_splits] += 1
            
            current = 0
            for fold, size in enumerate(fold_sizes):
                test_folds[fold].extend(cls_indices[current:current + size])
                current += size
        
        # Generate train/test splits
        for fold in range(self.n_splits):
            test_indices = np.array(test_folds[fold])
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits


class GroupKFold:
    """
    K-fold iterator variant with non-overlapping groups.
    
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        
    Examples
    --------
    >>> from neurova.ml.model_selection import GroupKFold
    >>> gkf = GroupKFold(n_splits=3)
    >>> # Groups could be patient IDs, user IDs, etc.
    >>> groups = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 4])
    >>> for train_idx, test_idx in gkf.split(X, y, groups):
    ...     print(f"Train groups: {np.unique(groups[train_idx])}")
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: np.ndarray = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target variable (not used)
        groups : array-like of shape (n_samples,)
            Group labels (samples with same group stay together)
            
        Yields
        ------
        train : ndarray
            Training indices for this fold
        test : ndarray
            Test indices for this fold
        """
        if groups is None:
            raise ValueError("The 'groups' parameter must be specified")
        
        X = np.asarray(X)
        groups = np.asarray(groups)
        n_samples = X.shape[0]
        
        # Get unique groups
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_splits:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} "
                f"greater than the number of groups: {n_groups}"
            )
        
        # Assign groups to folds
        group_to_fold = {}
        for i, group in enumerate(unique_groups):
            group_to_fold[group] = i % self.n_splits
        
        # Generate train/test splits
        for fold in range(self.n_splits):
            test_mask = np.array([group_to_fold[g] == fold for g in groups])
            test_indices = np.where(test_mask)[0]
            train_indices = np.where(~test_mask)[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits


class LeaveOneOut:
    """
    Leave-One-Out cross-validation iterator.
    
    Provides train/test indices where each sample is used once as a test set.
    
    Examples
    --------
    >>> from neurova.ml.model_selection import LeaveOneOut
    >>> loo = LeaveOneOut()
    >>> for train_idx, test_idx in loo.split(X):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
              groups: Optional[np.ndarray] = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Not used (for API compatibility)
        groups : array-like, optional
            Not used (for API compatibility)
            
        Yields
        ------
        train : ndarray
            Training indices
        test : ndarray
            Test indices (single sample)
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        for i in range(n_samples):
            test_indices = np.array([i])
            train_indices = np.concatenate([indices[:i], indices[i+1:]])
            yield train_indices, test_indices
    
    def get_n_splits(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return X.shape[0]


class LeaveOneGroupOut:
    """
    Leave-One-Group-Out cross-validation iterator.
    
    Provides train/test indices where each group is left out exactly once.
    
    Examples
    --------
    >>> from neurova.ml.model_selection import LeaveOneGroupOut
    >>> logo = LeaveOneGroupOut()
    >>> groups = np.array([1, 1, 2, 2, 3, 3])
    >>> for train_idx, test_idx in logo.split(X, y, groups):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
              groups: np.ndarray = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Not used (for API compatibility)
        groups : array-like of shape (n_samples,)
            Group labels
            
        Yields
        ------
        train : ndarray
            Training indices
        test : ndarray
            Test indices (one group)
        """
        if groups is None:
            raise ValueError("The 'groups' parameter must be specified")
        
        X = np.asarray(X)
        groups = np.asarray(groups)
        
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            test_mask = groups == group
            test_indices = np.where(test_mask)[0]
            train_indices = np.where(~test_mask)[0]
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     groups: np.ndarray = None) -> int:
        """Return number of splits."""
        if groups is None:
            raise ValueError("The 'groups' parameter must be specified")
        return len(np.unique(groups))


class ShuffleSplit:
    """
    Random permutation cross-validator.
    
    Yields indices to split data into a training and test set.
    
    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations
    test_size : float, default=0.1
        Proportion of dataset to include in test split
    train_size : float, default=None
        Proportion of dataset to include in train split
    random_state : int or None, default=None
        Random seed
        
    Examples
    --------
    >>> from neurova.ml.model_selection import ShuffleSplit
    >>> ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    >>> for train_idx, test_idx in ss.split(X):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """
    
    def __init__(
        self,
        n_splits: int = 10,
        test_size: float = 0.1,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Not used (for API compatibility)
        groups : array-like, optional
            Not used (for API compatibility)
            
        Yields
        ------
        train : ndarray
            Training indices
        test : ndarray
            Test indices
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        n_test = int(n_samples * self.test_size)
        if self.train_size is not None:
            n_train = int(n_samples * self.train_size)
        else:
            n_train = n_samples - n_test
        
        rng = np.random.default_rng(self.random_state)
        
        for _ in range(self.n_splits):
            permutation = rng.permutation(n_samples)
            test_indices = permutation[:n_test]
            train_indices = permutation[n_test:n_test + n_train]
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits


class StratifiedShuffleSplit:
    """
    Stratified ShuffleSplit cross-validator.
    
    Provides train/test indices with preserved class distribution.
    
    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations
    test_size : float, default=0.1
        Proportion of dataset to include in test split
    train_size : float, default=None
        Proportion of dataset to include in train split
    random_state : int or None, default=None
        Random seed
        
    Examples
    --------
    >>> from neurova.ml.model_selection import StratifiedShuffleSplit
    >>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    >>> for train_idx, test_idx in sss.split(X, y):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """
    
    def __init__(
        self,
        n_splits: int = 10,
        test_size: float = 0.1,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target variable (used for stratification)
        groups : array-like, optional
            Not used (for API compatibility)
            
        Yields
        ------
        train : ndarray
            Training indices
        test : ndarray
            Test indices
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        # Get class information
        classes, y_indices = np.unique(y, return_inverse=True)
        class_indices = [np.where(y_indices == i)[0] for i in range(len(classes))]
        
        rng = np.random.default_rng(self.random_state)
        
        for _ in range(self.n_splits):
            test_indices_list = []
            train_indices_list = []
            
            for cls_indices in class_indices:
                n_cls = len(cls_indices)
                n_cls_test = max(1, int(n_cls * self.test_size))
                if self.train_size is not None:
                    n_cls_train = max(1, int(n_cls * self.train_size))
                else:
                    n_cls_train = n_cls - n_cls_test
                
                permutation = rng.permutation(cls_indices)
                test_indices_list.extend(permutation[:n_cls_test])
                train_indices_list.extend(permutation[n_cls_test:n_cls_test + n_cls_train])
            
            yield np.array(train_indices_list), np.array(test_indices_list)
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits


class TimeSeriesSplit:
    """
    Time Series cross-validator.
    
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits
    max_train_size : int or None, default=None
        Maximum size for a single training set
    test_size : int or None, default=None
        Number of samples to use for test set
    gap : int, default=0
        Number of samples to exclude between train and test
        
    Examples
    --------
    >>> from neurova.ml.model_selection import TimeSeriesSplit
    >>> tscv = TimeSeriesSplit(n_splits=5)
    >>> for train_idx, test_idx in tscv.split(X):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Not used (for API compatibility)
        groups : array-like, optional
            Not used (for API compatibility)
            
        Yields
        ------
        train : ndarray
            Training indices
        test : ndarray
            Test indices
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Calculate test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = test_start - self.gap
            
            if train_end <= 0:
                continue
            
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            yield indices[train_start:train_end], indices[test_start:test_end]
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits


__all__ = [
    'train_test_split',
    'cross_validate',
    'GridSearchCV',
    'RandomizedSearchCV',
    'KFold',
    'StratifiedKFold',
    'GroupKFold',
    'LeaveOneOut',
    'LeaveOneGroupOut',
    'ShuffleSplit',
    'StratifiedShuffleSplit',
    'TimeSeriesSplit',
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.