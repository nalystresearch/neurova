# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Pipeline utilities for composing estimators.

Provides:
- Pipeline: Sequential application of transforms and estimator
- ColumnTransformer: Apply different transformers to different columns
- FeatureUnion: Concatenate outputs of multiple transformers
- make_pipeline: Convenience function for creating pipelines
- make_union: Convenience function for creating feature unions

All implementations follow scikit-learn conventions.
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from abc import ABC, abstractmethod


class TransformerMixin(ABC):
    """Mixin class for all transformers."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TransformerMixin':
        """Fit the transformer."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class EstimatorMixin(ABC):
    """Mixin class for all estimators."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EstimatorMixin':
        """Fit the estimator."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass


class Pipeline:
    """
    Pipeline of transforms with a final estimator.
    
    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is,
    they must implement fit and transform methods. The final estimator
    only needs to implement fit and predict.
    
    Parameters
    ----------
    steps : list of (name, transformer) tuples
        List of (name, transform) tuples that are chained in order.
        The last object must be an estimator.
    memory : None
        Used for caching (not implemented, for API compatibility)
    verbose : bool, default=False
        If True, print pipeline progress
        
    Attributes
    ----------
    named_steps : dict
        Dictionary-like object to access steps by name
        
    Examples
    --------
    >>> from neurova.ml.pipeline import Pipeline
    >>> from neurova.ml import StandardScaler, PCA, LogisticRegression
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('pca', PCA(n_components=2)),
    ...     ('classifier', LogisticRegression())
    ... ])
    >>> pipe.fit(X_train, y_train)
    >>> y_pred = pipe.predict(X_test)
    """
    
    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        memory: Optional[Any] = None,
        verbose: bool = False,
    ):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        
        self._validate_steps()
    
    def _validate_steps(self):
        """Validate pipeline steps."""
        names, estimators = zip(*self.steps) if self.steps else ([], [])
        
        # Check for duplicate names
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique")
        
        # Check that all but last step have transform method
        for name, est in self.steps[:-1]:
            if not hasattr(est, 'transform'):
                raise TypeError(
                    f"All intermediate steps should implement transform. "
                    f"'{name}' (type {type(est)}) doesn't."
                )
    
    @property
    def named_steps(self) -> Dict[str, Any]:
        """Access the steps by name."""
        return dict(self.steps)
    
    def __getitem__(self, ind: Union[str, int, slice]):
        """Get a step by index, slice, or name."""
        if isinstance(ind, str):
            return self.named_steps[ind]
        elif isinstance(ind, slice):
            return Pipeline(self.steps[ind])
        else:
            return self.steps[ind][1]
    
    def __len__(self) -> int:
        """Return number of steps."""
        return len(self.steps)
    
    def _fit_transform_steps(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params
    ) -> np.ndarray:
        """Fit and transform all but the last step."""
        Xt = X
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if self.verbose:
                print(f"[Pipeline] Fitting and transforming step '{name}'...")
            
            if hasattr(transformer, 'fit_transform'):
                # Try with y first, fall back to without y
                try:
                    Xt = transformer.fit_transform(Xt, y)
                except TypeError:
                    Xt = transformer.fit_transform(Xt)
            else:
                try:
                    Xt = transformer.fit(Xt, y).transform(Xt)
                except TypeError:
                    Xt = transformer.fit(Xt).transform(Xt)
        
        return Xt
    
    def _transform_steps(self, X: np.ndarray) -> np.ndarray:
        """Transform through all but the last step."""
        Xt = X
        for name, transformer in self.steps[:-1]:
            Xt = transformer.transform(Xt)
        return Xt
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> 'Pipeline':
        """Fit the pipeline.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Training targets
        **fit_params : dict
            Parameters passed to the fit method of each step
            
        Returns
        -------
        self : Pipeline
            Fitted pipeline
        """
        Xt = self._fit_transform_steps(X, y, **fit_params)
        
        # Fit final estimator
        if self.steps:
            final_name, final_estimator = self.steps[-1]
            if self.verbose:
                print(f"[Pipeline] Fitting final step '{final_name}'...")
            
            if y is not None:
                final_estimator.fit(Xt, y)
            else:
                final_estimator.fit(Xt)
        
        return self
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params
    ) -> np.ndarray:
        """Fit and transform the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Training targets
            
        Returns
        -------
        Xt : ndarray
            Transformed data
        """
        Xt = self._fit_transform_steps(X, y, **fit_params)
        
        # Fit and transform final step if it's a transformer
        if self.steps:
            final_name, final_estimator = self.steps[-1]
            if hasattr(final_estimator, 'fit_transform'):
                if y is not None:
                    Xt = final_estimator.fit_transform(Xt, y)
                else:
                    Xt = final_estimator.fit_transform(Xt)
            elif hasattr(final_estimator, 'transform'):
                if y is not None:
                    final_estimator.fit(Xt, y)
                else:
                    final_estimator.fit(Xt)
                Xt = final_estimator.transform(Xt)
            else:
                if y is not None:
                    final_estimator.fit(Xt, y)
                else:
                    final_estimator.fit(Xt)
        
        return Xt
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        Xt : ndarray
            Transformed data
        """
        Xt = self._transform_steps(X)
        
        # Transform with final step if it's a transformer
        if self.steps and hasattr(self.steps[-1][1], 'transform'):
            Xt = self.steps[-1][1].transform(Xt)
        
        return Xt
    
    def predict(self, X: np.ndarray, **predict_params) -> np.ndarray:
        """Predict using the final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        y : ndarray
            Predicted values
        """
        Xt = self._transform_steps(X)
        return self.steps[-1][1].predict(Xt, **predict_params)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        Xt = self._transform_steps(X)
        return self.steps[-1][1].predict_proba(Xt)
    
    def score(self, X: np.ndarray, y: np.ndarray, **score_params) -> float:
        """Return score of the final estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        y : array-like of shape (n_samples,)
            True values
            
        Returns
        -------
        score : float
            Score of the final estimator
        """
        Xt = self._transform_steps(X)
        return self.steps[-1][1].score(Xt, y, **score_params)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters of the pipeline."""
        params = {'memory': self.memory, 'verbose': self.verbose}
        
        if deep:
            for name, step in self.steps:
                if hasattr(step, 'get_params'):
                    for key, value in step.get_params(deep=True).items():
                        params[f'{name}__{key}'] = value
        
        return params
    
    def set_params(self, **params) -> 'Pipeline':
        """Set parameters of the pipeline."""
        for key, value in params.items():
            if '__' in key:
                step_name, param_name = key.split('__', 1)
                for name, step in self.steps:
                    if name == step_name and hasattr(step, 'set_params'):
                        step.set_params(**{param_name: value})
            else:
                setattr(self, key, value)
        return self


class ColumnTransformer:
    """
    Apply transformers to columns of an array.
    
    This estimator allows different columns or column subsets of the input
    to be transformed separately and the features generated by each
    transformer will be concatenated to form a single feature space.
    
    Parameters
    ----------
    transformers : list of (name, transformer, columns) tuples
        List of transformer specifications:
        - name: name of the transformer
        - transformer: transformer object or 'passthrough' or 'drop'
        - columns: indices or slice specifying which columns to apply
    remainder : {'drop', 'passthrough'} or transformer, default='drop'
        By default, only the specified columns are transformed and combined.
        - 'drop': remaining columns are not included
        - 'passthrough': remaining columns are passed through unchanged
        - transformer: transformer object to apply to remaining columns
    sparse_threshold : float, default=0.3
        Not used (for API compatibility)
    n_jobs : int, default=None
        Not used (for API compatibility)
    verbose : bool, default=False
        If True, print progress
        
    Examples
    --------
    >>> from neurova.ml.pipeline import ColumnTransformer
    >>> from neurova.ml import StandardScaler, OneHotEncoder
    >>> ct = ColumnTransformer([
    ...     ('num', StandardScaler(), [0, 1, 2]),
    ...     ('cat', OneHotEncoder(), [3, 4])
    ... ])
    >>> X_transformed = ct.fit_transform(X)
    """
    
    def __init__(
        self,
        transformers: List[Tuple[str, Any, Union[List[int], slice]]],
        remainder: Union[str, Any] = 'drop',
        sparse_threshold: float = 0.3,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ):
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self._remainder_columns: List[int] = []
        self._fitted_transformers: List[Tuple[str, Any, List[int]]] = []
        self.n_features_in_: int = 0
    
    def _get_column_indices(self, columns: Union[List[int], slice], n_features: int) -> List[int]:
        """Convert column specification to list of indices."""
        if isinstance(columns, slice):
            return list(range(*columns.indices(n_features)))
        elif isinstance(columns, list):
            return columns
        else:
            return [columns]
    
    def _get_remainder_columns(self, n_features: int) -> List[int]:
        """Get columns not covered by any transformer."""
        specified_columns = set()
        for name, transformer, columns in self.transformers:
            indices = self._get_column_indices(columns, n_features)
            specified_columns.update(indices)
        
        return [i for i in range(n_features) if i not in specified_columns]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ColumnTransformer':
        """Fit all transformers.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Training targets (not used, for API compatibility)
            
        Returns
        -------
        self : ColumnTransformer
            Fitted transformer
        """
        X = np.asarray(X)
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        
        self._fitted_transformers = []
        
        for name, transformer, columns in self.transformers:
            indices = self._get_column_indices(columns, n_features)
            
            if self.verbose:
                print(f"[ColumnTransformer] Fitting '{name}' on columns {indices}...")
            
            if transformer == 'passthrough':
                self._fitted_transformers.append((name, 'passthrough', indices))
            elif transformer == 'drop':
                self._fitted_transformers.append((name, 'drop', indices))
            else:
                X_subset = X[:, indices]
                # Try with y first, fall back to without y
                try:
                    transformer.fit(X_subset, y)
                except TypeError:
                    transformer.fit(X_subset)
                self._fitted_transformers.append((name, transformer, indices))
        
        # Handle remainder
        self._remainder_columns = self._get_remainder_columns(n_features)
        if self.remainder != 'drop' and self._remainder_columns:
            if self.remainder == 'passthrough':
                pass  # Will just pass through
            elif hasattr(self.remainder, 'fit'):
                X_remainder = X[:, self._remainder_columns]
                try:
                    self.remainder.fit(X_remainder, y)
                except TypeError:
                    self.remainder.fit(X_remainder)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        Xt : ndarray
            Transformed data with features concatenated
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        transformed_parts = []
        
        for name, transformer, indices in self._fitted_transformers:
            X_subset = X[:, indices]
            
            if transformer == 'passthrough':
                transformed_parts.append(X_subset)
            elif transformer == 'drop':
                continue  # Skip dropped columns
            else:
                Xt = transformer.transform(X_subset)
                # Ensure 2D
                if Xt.ndim == 1:
                    Xt = Xt.reshape(-1, 1)
                transformed_parts.append(Xt)
        
        # Handle remainder
        if self._remainder_columns:
            X_remainder = X[:, self._remainder_columns]
            if self.remainder == 'passthrough':
                transformed_parts.append(X_remainder)
            elif self.remainder != 'drop':
                Xt = self.remainder.transform(X_remainder)
                if Xt.ndim == 1:
                    Xt = Xt.reshape(-1, 1)
                transformed_parts.append(Xt)
        
        if not transformed_parts:
            return np.empty((n_samples, 0))
        
        return np.hstack(transformed_parts)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Training targets
            
        Returns
        -------
        Xt : ndarray
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names (placeholder)."""
        names = []
        for name, transformer, indices in self._fitted_transformers:
            if transformer == 'drop':
                continue
            for idx in indices:
                names.append(f'{name}__{idx}')
        return names


class FeatureUnion:
    """
    Concatenates results of multiple transformer objects.
    
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results.
    
    Parameters
    ----------
    transformer_list : list of (name, transformer) tuples
        List of transformer objects to be applied in parallel
    n_jobs : int, default=None
        Not used (for API compatibility)
    verbose : bool, default=False
        If True, print progress
        
    Examples
    --------
    >>> from neurova.ml.pipeline import FeatureUnion
    >>> from neurova.ml import PCA, StandardScaler
    >>> union = FeatureUnion([
    ...     ('pca', PCA(n_components=2)),
    ...     ('scaled', StandardScaler())
    ... ])
    >>> X_transformed = union.fit_transform(X)
    """
    
    def __init__(
        self,
        transformer_list: List[Tuple[str, Any]],
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    @property
    def named_transformers(self) -> Dict[str, Any]:
        """Access transformers by name."""
        return dict(self.transformer_list)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureUnion':
        """Fit all transformers.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Training targets
            
        Returns
        -------
        self : FeatureUnion
            Fitted transformer
        """
        for name, transformer in self.transformer_list:
            if self.verbose:
                print(f"[FeatureUnion] Fitting '{name}'...")
            transformer.fit(X, y)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data with all transformers and concatenate.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        Xt : ndarray
            Concatenated transformed data
        """
        transformed_parts = []
        
        for name, transformer in self.transformer_list:
            Xt = transformer.transform(X)
            if Xt.ndim == 1:
                Xt = Xt.reshape(-1, 1)
            transformed_parts.append(Xt)
        
        return np.hstack(transformed_parts)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Training targets
            
        Returns
        -------
        Xt : ndarray
            Transformed data
        """
        return self.fit(X, y).transform(X)


def make_pipeline(*steps, memory=None, verbose=False) -> Pipeline:
    """
    Construct a Pipeline from the given estimators.
    
    This is a shorthand for creating a Pipeline with auto-generated names.
    
    Parameters
    ----------
    *steps : list of estimators
        Transformers and final estimator
    memory : None
        Not used (for API compatibility)
    verbose : bool, default=False
        If True, print progress
        
    Returns
    -------
    p : Pipeline
        Pipeline with auto-generated step names
        
    Examples
    --------
    >>> from neurova.ml.pipeline import make_pipeline
    >>> from neurova.ml import StandardScaler, PCA, LogisticRegression
    >>> pipe = make_pipeline(StandardScaler(), PCA(2), LogisticRegression())
    """
    named_steps = []
    for step in steps:
        name = type(step).__name__.lower()
        named_steps.append((name, step))
    
    return Pipeline(named_steps, memory=memory, verbose=verbose)


def make_union(*transformers, n_jobs=None, verbose=False) -> FeatureUnion:
    """
    Construct a FeatureUnion from the given transformers.
    
    This is a shorthand for creating a FeatureUnion with auto-generated names.
    
    Parameters
    ----------
    *transformers : list of transformers
        Transformer objects to concatenate
    n_jobs : int, default=None
        Not used (for API compatibility)
    verbose : bool, default=False
        If True, print progress
        
    Returns
    -------
    fu : FeatureUnion
        FeatureUnion with auto-generated step names
        
    Examples
    --------
    >>> from neurova.ml.pipeline import make_union
    >>> from neurova.ml import PCA, StandardScaler
    >>> union = make_union(PCA(2), StandardScaler())
    """
    named_transformers = []
    for transformer in transformers:
        name = type(transformer).__name__.lower()
        named_transformers.append((name, transformer))
    
    return FeatureUnion(named_transformers, n_jobs=n_jobs, verbose=verbose)


class FunctionTransformer:
    """
    Constructs a transformer from an arbitrary callable.
    
    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. If None, uses identity.
    inverse_func : callable, default=None
        The callable to use for the inverse transformation
    validate : bool, default=False
        Whether to validate input (convert to array)
    kw_args : dict, default=None
        Dictionary of keyword arguments to pass to func
        
    Examples
    --------
    >>> from neurova.ml.pipeline import FunctionTransformer
    >>> transformer = FunctionTransformer(np.log1p)
    >>> X_transformed = transformer.fit_transform(X)
    """
    
    def __init__(
        self,
        func: Optional[Callable] = None,
        inverse_func: Optional[Callable] = None,
        validate: bool = False,
        kw_args: Optional[Dict] = None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.kw_args = kw_args or {}
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FunctionTransformer':
        """Fit (does nothing, for API compatibility).
        
        Parameters
        ----------
        X : array-like
            Training data (ignored)
        y : array-like, optional
            Training targets (ignored)
            
        Returns
        -------
        self : FunctionTransformer
        """
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using the callable.
        
        Parameters
        ----------
        X : array-like
            Data to transform
            
        Returns
        -------
        Xt : ndarray
            Transformed data
        """
        if self.validate:
            X = np.asarray(X)
        
        if self.func is None:
            return X
        
        return self.func(X, **self.kw_args)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform X using the inverse callable.
        
        Parameters
        ----------
        X : array-like
            Data to inverse transform
            
        Returns
        -------
        Xt : ndarray
            Inverse transformed data
        """
        if self.validate:
            X = np.asarray(X)
        
        if self.inverse_func is None:
            return X
        
        return self.inverse_func(X)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


__all__ = [
    'Pipeline',
    'ColumnTransformer',
    'FeatureUnion',
    'FunctionTransformer',
    'make_pipeline',
    'make_union',
    'TransformerMixin',
    'EstimatorMixin',
]
