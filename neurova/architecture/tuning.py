# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Hyperparameter Tuning for Neurova Architecture

Provides automated hyperparameter tuning for architecture models.
Includes grid search, random search, and Bayesian optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed


class HyperparameterSpace:
    """
    Define hyperparameter search space.
    
    Example
    -------
    >>> space = HyperparameterSpace()
    >>> space.add('learning_rate', 'log_uniform', low=1e-5, high=1e-1)
    >>> space.add('hidden_layers', 'choice', options=[[64], [128, 64], [256, 128, 64]])
    >>> space.add('dropout', 'uniform', low=0.0, high=0.5)
    """
    
    def __init__(self):
        self.params = {}
        
    def add(self, name: str, param_type: str, **kwargs):
        """
        Add a hyperparameter to the search space.
        
        Parameters
        ----------
        name : str
            Parameter name
        param_type : str
            Type: 'choice', 'uniform', 'log_uniform', 'int_uniform', 'categorical'
        **kwargs
            Type-specific arguments
        """
        self.params[name] = {'type': param_type, **kwargs}
        return self
    
    def add_learning_rate(self, low: float = 1e-5, high: float = 1e-1):
        """Add learning rate parameter."""
        return self.add('learning_rate', 'log_uniform', low=low, high=high)
    
    def add_batch_size(self, options: List[int] = [16, 32, 64, 128]):
        """Add batch size parameter."""
        return self.add('batch_size', 'choice', options=options)
    
    def add_hidden_layers(self, options: List[List[int]] = None):
        """Add hidden layers parameter."""
        if options is None:
            options = [[64], [128, 64], [256, 128, 64], [512, 256, 128]]
        return self.add('hidden_layers', 'choice', options=options)
    
    def add_dropout(self, low: float = 0.0, high: float = 0.5):
        """Add dropout parameter."""
        return self.add('dropout', 'uniform', low=low, high=high)
    
    def add_optimizer(self, options: List[str] = ['adam', 'sgd', 'rmsprop']):
        """Add optimizer parameter."""
        return self.add('optimizer', 'choice', options=options)
    
    def sample(self) -> Dict[str, Any]:
        """Sample a random configuration from the space."""
        config = {}
        
        for name, spec in self.params.items():
            param_type = spec['type']
            
            if param_type == 'choice' or param_type == 'categorical':
                # Use random.randint instead of np.random.choice for complex types
                options = spec['options']
                idx = np.random.randint(0, len(options))
                config[name] = options[idx]
            elif param_type == 'uniform':
                config[name] = np.random.uniform(spec['low'], spec['high'])
            elif param_type == 'log_uniform':
                log_low = np.log(spec['low'])
                log_high = np.log(spec['high'])
                config[name] = np.exp(np.random.uniform(log_low, log_high))
            elif param_type == 'int_uniform':
                config[name] = np.random.randint(spec['low'], spec['high'] + 1)
            elif param_type == 'quniform':
                q = spec.get('q', 1)
                val = np.random.uniform(spec['low'], spec['high'])
                config[name] = round(val / q) * q
                
        return config
    
    def grid(self) -> List[Dict[str, Any]]:
        """Generate all grid combinations."""
        param_grids = {}
        
        for name, spec in self.params.items():
            param_type = spec['type']
            
            if param_type in ['choice', 'categorical']:
                param_grids[name] = spec['options']
            elif param_type == 'uniform':
                n_points = spec.get('n_points', 3)
                param_grids[name] = np.linspace(spec['low'], spec['high'], n_points).tolist()
            elif param_type == 'log_uniform':
                n_points = spec.get('n_points', 3)
                param_grids[name] = np.logspace(
                    np.log10(spec['low']), np.log10(spec['high']), n_points
                ).tolist()
            elif param_type == 'int_uniform':
                step = spec.get('step', 1)
                param_grids[name] = list(range(spec['low'], spec['high'] + 1, step))
        
        # Generate all combinations
        keys = list(param_grids.keys())
        combinations = list(product(*[param_grids[k] for k in keys]))
        
        return [dict(zip(keys, combo)) for combo in combinations]


class TuningResult:
    """Container for tuning results."""
    
    def __init__(self):
        self.trials = []
        self.best_params = None
        self.best_score = float('-inf')
        self.best_model = None
        self.total_time = 0.0
        
    def add_trial(self, params: Dict, score: float, model=None, time_taken: float = 0.0):
        """Add a trial result."""
        self.trials.append({
            'params': params,
            'score': score,
            'time': time_taken,
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            self.best_model = model
            
    def summary(self) -> str:
        """Get tuning summary."""
        lines = [
            "=" * 60,
            "HYPERPARAMETER TUNING RESULTS",
            "=" * 60,
            f"Total trials: {len(self.trials)}",
            f"Total time: {self.total_time:.2f}s",
            f"Best score: {self.best_score:.4f}",
            "",
            "Best parameters:",
        ]
        
        for key, value in self.best_params.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("")
        lines.append("Top 5 trials:")
        
        sorted_trials = sorted(self.trials, key=lambda x: x['score'], reverse=True)[:5]
        for i, trial in enumerate(sorted_trials):
            lines.append(f"  {i+1}. Score: {trial['score']:.4f} - {trial['params']}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame (if available)."""
        try:
            import pandas as pd
            
            rows = []
            for trial in self.trials:
                row = trial['params'].copy()
                row['score'] = trial['score']
                row['time'] = trial['time']
                rows.append(row)
            
            return pd.DataFrame(rows)
        except ImportError:
            print("pandas required for DataFrame conversion")
            return None


class GridSearchCV:
    """
    Exhaustive search over parameter grid.
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    param_grid : dict or HyperparameterSpace
        Parameters to search
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    verbose : int
        Verbosity level
    
    Example
    -------
    >>> from neurova.architecture import MLP, GridSearchCV
    >>> 
    >>> param_grid = {
    ...     'learning_rate': [0.001, 0.01],
    ...     'hidden_layers': [[128], [256, 128]],
    ...     'dropout': [0.0, 0.3],
    ... }
    >>> 
    >>> search = GridSearchCV(MLP, param_grid, cv=3)
    >>> search.fit(X_train, y_train, input_shape=100, output_shape=10)
    >>> 
    >>> print(f"Best params: {search.best_params_}")
    >>> best_model = search.best_model_
    """
    
    def __init__(self, model_class, param_grid: Union[Dict, HyperparameterSpace],
                 cv: int = 5, scoring: str = 'accuracy', verbose: int = 1,
                 n_jobs: int = 1):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        
        self.results_ = TuningResult()
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **fixed_params):
        """
        Run grid search.
        
        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        **fixed_params
            Fixed parameters passed to model (e.g., input_shape)
        """
        start_time = time.time()
        
        # Generate parameter combinations
        if isinstance(self.param_grid, HyperparameterSpace):
            param_combinations = self.param_grid.grid()
        else:
            # Convert dict to list of dicts
            keys = list(self.param_grid.keys())
            values = [self.param_grid[k] for k in keys]
            param_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
        
        total = len(param_combinations)
        
        if self.verbose:
            print(f"Grid Search: {total} parameter combinations")
            print("-" * 50)
        
        for i, params in enumerate(param_combinations):
            trial_start = time.time()
            
            # Merge with fixed params
            all_params = {**fixed_params, **params}
            
            # Cross-validation
            score = self._cross_validate(X, y, all_params)
            
            trial_time = time.time() - trial_start
            
            # Create model with best params for storage
            model = self.model_class(**all_params)
            self.results_.add_trial(params, score, model, trial_time)
            
            if self.verbose:
                print(f"[{i+1}/{total}] Score: {score:.4f} - {params}")
        
        self.results_.total_time = time.time() - start_time
        self.best_params_ = self.results_.best_params
        self.best_score_ = self.results_.best_score
        
        # Retrain best model on full data
        best_all_params = {**fixed_params, **self.best_params_}
        self.best_model_ = self.model_class(**best_all_params)
        self.best_model_.fit(X, y)
        
        if self.verbose:
            print("\n" + self.results_.summary())
        
        return self
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       params: Dict) -> float:
        """Perform k-fold cross-validation."""
        n_samples = len(X)
        fold_size = n_samples // self.cv
        indices = np.random.permutation(n_samples)
        
        scores = []
        
        for fold in range(self.cv):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.cv - 1 else n_samples
            
            val_idx = indices[start_idx:end_idx]
            train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = self.model_class(**params)
            model.verbose = 0  # Suppress output
            model.fit(X_train, y_train)
            
            # Evaluate
            score = model.score(X_val, y_val)
            scores.append(score)
        
        return np.mean(scores)


class RandomSearchCV:
    """
    Random search over parameter space.
    
    More efficient than grid search for large spaces.
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    param_space : dict or HyperparameterSpace
        Parameters to search
    n_iter : int
        Number of random samples
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Example
    -------
    >>> space = HyperparameterSpace()
    >>> space.add_learning_rate()
    >>> space.add_hidden_layers()
    >>> space.add_dropout()
    >>> 
    >>> search = RandomSearchCV(MLP, space, n_iter=50, cv=3)
    >>> search.fit(X_train, y_train, input_shape=100, output_shape=10)
    """
    
    def __init__(self, model_class, param_space: Union[Dict, HyperparameterSpace],
                 n_iter: int = 20, cv: int = 5, scoring: str = 'accuracy',
                 verbose: int = 1, random_state: Optional[int] = None):
        self.model_class = model_class
        self.param_space = param_space
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.results_ = TuningResult()
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **fixed_params):
        """Run random search."""
        start_time = time.time()
        
        # Convert dict to HyperparameterSpace if needed
        if isinstance(self.param_space, dict):
            space = HyperparameterSpace()
            for name, values in self.param_space.items():
                if isinstance(values, list):
                    space.add(name, 'choice', options=values)
                elif isinstance(values, tuple) and len(values) == 2:
                    space.add(name, 'uniform', low=values[0], high=values[1])
        else:
            space = self.param_space
        
        if self.verbose:
            print(f"Random Search: {self.n_iter} iterations")
            print("-" * 50)
        
        for i in range(self.n_iter):
            trial_start = time.time()
            
            # Sample parameters
            params = space.sample()
            all_params = {**fixed_params, **params}
            
            # Cross-validation
            score = self._cross_validate(X, y, all_params)
            
            trial_time = time.time() - trial_start
            
            model = self.model_class(**all_params)
            self.results_.add_trial(params, score, model, trial_time)
            
            if self.verbose:
                print(f"[{i+1}/{self.n_iter}] Score: {score:.4f} - {params}")
        
        self.results_.total_time = time.time() - start_time
        self.best_params_ = self.results_.best_params
        self.best_score_ = self.results_.best_score
        
        # Retrain best model
        best_all_params = {**fixed_params, **self.best_params_}
        self.best_model_ = self.model_class(**best_all_params)
        self.best_model_.fit(X, y)
        
        if self.verbose:
            print("\n" + self.results_.summary())
        
        return self
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray,
                       params: Dict) -> float:
        """Perform k-fold cross-validation."""
        n_samples = len(X)
        fold_size = n_samples // self.cv
        indices = np.random.permutation(n_samples)
        
        scores = []
        
        for fold in range(self.cv):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.cv - 1 else n_samples
            
            val_idx = indices[start_idx:end_idx]
            train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.model_class(**params)
            model.verbose = 0
            model.fit(X_train, y_train)
            
            score = model.score(X_val, y_val)
            scores.append(score)
        
        return np.mean(scores)


class BayesianOptimization:
    """
    Bayesian optimization for hyperparameter tuning.
    
    Uses Gaussian Process surrogate model to efficiently
    explore the parameter space.
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    param_space : HyperparameterSpace
        Parameters to search
    n_iter : int
        Number of optimization iterations
    n_initial : int
        Number of initial random samples
    
    Example
    -------
    >>> space = HyperparameterSpace()
    >>> space.add_learning_rate()
    >>> space.add('hidden_size', 'int_uniform', low=32, high=256)
    >>> 
    >>> opt = BayesianOptimization(MLP, space, n_iter=30)
    >>> opt.fit(X_train, y_train, input_shape=100, output_shape=10)
    """
    
    def __init__(self, model_class, param_space: HyperparameterSpace,
                 n_iter: int = 30, n_initial: int = 5, cv: int = 3,
                 verbose: int = 1, random_state: Optional[int] = None):
        self.model_class = model_class
        self.param_space = param_space
        self.n_iter = n_iter
        self.n_initial = n_initial
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.results_ = TuningResult()
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        
        # Gaussian Process components
        self.X_observed = []
        self.y_observed = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **fixed_params):
        """Run Bayesian optimization."""
        start_time = time.time()
        
        if self.verbose:
            print(f"Bayesian Optimization: {self.n_iter} iterations")
            print(f"Initial random samples: {self.n_initial}")
            print("-" * 50)
        
        # Initial random samples
        for i in range(self.n_initial):
            params = self.param_space.sample()
            all_params = {**fixed_params, **params}
            
            score = self._cross_validate(X, y, all_params)
            
            self.X_observed.append(self._params_to_vector(params))
            self.y_observed.append(score)
            
            self.results_.add_trial(params, score, None, 0)
            
            if self.verbose:
                print(f"[Init {i+1}/{self.n_initial}] Score: {score:.4f} - {params}")
        
        # Bayesian optimization iterations
        for i in range(self.n_iter - self.n_initial):
            # Find next point to evaluate using acquisition function
            params = self._suggest_next()
            all_params = {**fixed_params, **params}
            
            score = self._cross_validate(X, y, all_params)
            
            self.X_observed.append(self._params_to_vector(params))
            self.y_observed.append(score)
            
            self.results_.add_trial(params, score, None, 0)
            
            if self.verbose:
                print(f"[{self.n_initial + i + 1}/{self.n_iter}] Score: {score:.4f} - {params}")
        
        self.results_.total_time = time.time() - start_time
        self.best_params_ = self.results_.best_params
        self.best_score_ = self.results_.best_score
        
        # Retrain best model
        best_all_params = {**fixed_params, **self.best_params_}
        self.best_model_ = self.model_class(**best_all_params)
        self.best_model_.fit(X, y)
        
        if self.verbose:
            print("\n" + self.results_.summary())
        
        return self
    
    def _params_to_vector(self, params: Dict) -> np.ndarray:
        """Convert parameter dict to numeric vector."""
        vector = []
        for name in sorted(self.param_space.params.keys()):
            value = params[name]
            spec = self.param_space.params[name]
            
            if spec['type'] in ['choice', 'categorical']:
                # Encode as index
                idx = spec['options'].index(value)
                vector.append(idx / len(spec['options']))
            else:
                # Normalize to [0, 1]
                low = spec.get('low', 0)
                high = spec.get('high', 1)
                if spec['type'] == 'log_uniform':
                    value = np.log(value)
                    low = np.log(low)
                    high = np.log(high)
                vector.append((value - low) / (high - low + 1e-10))
        
        return np.array(vector)
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict:
        """Convert numeric vector back to parameter dict."""
        params = {}
        for i, name in enumerate(sorted(self.param_space.params.keys())):
            spec = self.param_space.params[name]
            value = vector[i]
            
            if spec['type'] in ['choice', 'categorical']:
                idx = int(value * len(spec['options']))
                idx = min(idx, len(spec['options']) - 1)
                params[name] = spec['options'][idx]
            else:
                low = spec.get('low', 0)
                high = spec.get('high', 1)
                
                if spec['type'] == 'log_uniform':
                    low = np.log(low)
                    high = np.log(high)
                    params[name] = np.exp(low + value * (high - low))
                elif spec['type'] == 'int_uniform':
                    params[name] = int(low + value * (high - low))
                else:
                    params[name] = low + value * (high - low)
        
        return params
    
    def _suggest_next(self) -> Dict:
        """Suggest next parameters using acquisition function."""
        # Simplified: use Upper Confidence Bound (UCB)
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        
        best_acquisition = float('-inf')
        best_params = None
        
        # Random sampling for acquisition optimization
        for _ in range(100):
            candidate = self.param_space.sample()
            x = self._params_to_vector(candidate)
            
            # Simple acquisition: exploitation + exploration
            # Use kernel-based estimate
            if len(X_obs) > 0:
                distances = np.sum((X_obs - x) ** 2, axis=1)
                weights = np.exp(-distances)
                weights = weights / (np.sum(weights) + 1e-10)
                
                mean = np.sum(weights * y_obs)
                var = np.sum(weights * (y_obs - mean) ** 2)
                std = np.sqrt(var + 1e-10)
                
                # UCB
                acquisition = mean + 2.0 * std
            else:
                acquisition = np.random.random()
            
            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_params = candidate
        
        return best_params or self.param_space.sample()
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray,
                       params: Dict) -> float:
        """Perform k-fold cross-validation."""
        n_samples = len(X)
        fold_size = n_samples // self.cv
        indices = np.random.permutation(n_samples)
        
        scores = []
        
        for fold in range(self.cv):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.cv - 1 else n_samples
            
            val_idx = indices[start_idx:end_idx]
            train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                model = self.model_class(**params)
                model.verbose = 0
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
            except Exception as e:
                score = 0.0  # Invalid configuration
            
            scores.append(score)
        
        return np.mean(scores)


class AutoML:
    """
    Automated Machine Learning for architecture selection and tuning.
    
    Automatically selects the best architecture and hyperparameters.
    
    Parameters
    ----------
    architectures : list
        List of architecture classes to try
    time_budget : int
        Maximum time in seconds
    metric : str
        Optimization metric
    
    Example
    -------
    >>> from neurova.architecture import AutoML, MLP, CNN, LSTM
    >>> 
    >>> automl = AutoML(architectures=[MLP, CNN], time_budget=300)
    >>> automl.fit(X_train, y_train, input_shape=100, output_shape=10)
    >>> 
    >>> best_model = automl.best_model_
    >>> predictions = best_model.predict(X_test)
    """
    
    def __init__(self, architectures: List = None, time_budget: int = 300,
                 metric: str = 'accuracy', cv: int = 3, verbose: int = 1):
        self.architectures = architectures
        self.time_budget = time_budget
        self.metric = metric
        self.cv = cv
        self.verbose = verbose
        
        self.results_ = {}
        self.best_model_ = None
        self.best_score_ = float('-inf')
        self.best_architecture_ = None
        self.best_params_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **fixed_params):
        """
        Run AutoML.
        
        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        **fixed_params
            Fixed parameters (e.g., input_shape, output_shape)
        """
        start_time = time.time()
        
        # Default architectures
        if self.architectures is None:
            from .mlp import MLP
            self.architectures = [MLP]
        
        n_archs = len(self.architectures)
        time_per_arch = self.time_budget / n_archs
        
        if self.verbose:
            print("=" * 60)
            print("AUTOML - Automated Architecture Selection & Tuning")
            print("=" * 60)
            print(f"Architectures: {[a.__name__ for a in self.architectures]}")
            print(f"Time budget: {self.time_budget}s ({time_per_arch:.0f}s per architecture)")
            print("-" * 60)
        
        for arch in self.architectures:
            arch_name = arch.__name__
            arch_start = time.time()
            
            if self.verbose:
                print(f"\nTrying architecture: {arch_name}")
            
            # Get default hyperparameter space
            if hasattr(arch, 'PARAM_SPACE'):
                param_space = arch.PARAM_SPACE
            else:
                param_space = {
                    'learning_rate': [0.001, 0.01],
                    'batch_size': [32, 64],
                }
            
            # Estimate number of iterations possible
            n_iter = max(5, int(time_per_arch / 10))  # Rough estimate
            
            # Random search for this architecture
            search = RandomSearchCV(
                arch, param_space, n_iter=n_iter, cv=self.cv, verbose=0
            )
            
            try:
                search.fit(X, y, **fixed_params)
                
                self.results_[arch_name] = {
                    'best_score': search.best_score_,
                    'best_params': search.best_params_,
                    'best_model': search.best_model_,
                }
                
                if search.best_score_ > self.best_score_:
                    self.best_score_ = search.best_score_
                    self.best_model_ = search.best_model_
                    self.best_architecture_ = arch
                    self.best_params_ = search.best_params_
                
                if self.verbose:
                    print(f"  Best score: {search.best_score_:.4f}")
                    print(f"  Best params: {search.best_params_}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Failed: {str(e)}")
            
            # Check time budget
            if time.time() - start_time > self.time_budget:
                if self.verbose:
                    print("\nTime budget exceeded, stopping...")
                break
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("AUTOML RESULTS")
            print("=" * 60)
            print(f"Total time: {total_time:.2f}s")
            print(f"Best architecture: {self.best_architecture_.__name__ if self.best_architecture_ else 'None'}")
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best params: {self.best_params_}")
            print("=" * 60)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using best model."""
        if self.best_model_ is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.best_model_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate best model."""
        if self.best_model_ is None:
            raise RuntimeError("Must call fit() before score()")
        return self.best_model_.score(X, y)


# Convenience functions
def tune_model(model_class, X: np.ndarray, y: np.ndarray,
               param_space: Union[Dict, HyperparameterSpace],
               method: str = 'random', n_iter: int = 20, cv: int = 5,
               **fixed_params):
    """
    Quick hyperparameter tuning function.
    
    Parameters
    ----------
    model_class : class
        Architecture class
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels
    param_space : dict or HyperparameterSpace
        Parameters to search
    method : str
        Search method: 'grid', 'random', 'bayesian'
    n_iter : int
        Number of iterations (for random/bayesian)
    cv : int
        Cross-validation folds
    **fixed_params
        Fixed model parameters
    
    Returns
    -------
    best_model, best_params, best_score
    
    Example
    -------
    >>> from neurova.architecture import MLP, tune_model
    >>> 
    >>> params = {'learning_rate': [0.001, 0.01], 'dropout': [0.0, 0.3]}
    >>> model, params, score = tune_model(MLP, X, y, params, input_shape=100, output_shape=10)
    """
    if method == 'grid':
        search = GridSearchCV(model_class, param_space, cv=cv)
    elif method == 'bayesian':
        if isinstance(param_space, dict):
            space = HyperparameterSpace()
            for k, v in param_space.items():
                space.add(k, 'choice', options=v)
            param_space = space
        search = BayesianOptimization(model_class, param_space, n_iter=n_iter, cv=cv)
    else:  # random
        search = RandomSearchCV(model_class, param_space, n_iter=n_iter, cv=cv)
    
    search.fit(X, y, **fixed_params)
    
    return search.best_model_, search.best_params_, search.best_score_
