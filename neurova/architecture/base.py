# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Base Architecture Module for Neurova

Provides the foundational classes for all neural network architectures.
Features:
- Automatic parameter validation
- Built-in training with progress tracking
- Plotting for loss, accuracy, and metrics
- Hyperparameter tuning integration
- Model evaluation and testing
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import warnings


class ParameterValidator:
    """Validates architecture parameters automatically."""
    
    @staticmethod
    def validate_positive_int(value: Any, name: str) -> int:
        """Validate that value is a positive integer."""
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return int(value)
    
    @staticmethod
    def validate_positive_float(value: Any, name: str) -> float:
        """Validate that value is a positive float."""
        if not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return float(value)
    
    @staticmethod
    def validate_float_range(value: Any, name: str, min_val: float = 0.0, 
                            max_val: float = 1.0) -> float:
        """Validate that value is a float within range."""
        if not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        if value < min_val or value > max_val:
            raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")
        return float(value)
    
    @staticmethod
    def validate_choice(value: Any, name: str, choices: List[Any]) -> Any:
        """Validate that value is one of the allowed choices."""
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}, got {value}")
        return value
    
    @staticmethod
    def validate_shape(shape: Any, name: str, min_dims: int = 1, 
                      max_dims: int = 4) -> Tuple[int, ...]:
        """Validate input/output shape."""
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"{name} must be a tuple or int, got {type(shape).__name__}")
        shape = tuple(shape)
        if len(shape) < min_dims or len(shape) > max_dims:
            raise ValueError(f"{name} must have {min_dims}-{max_dims} dimensions, got {len(shape)}")
        for i, dim in enumerate(shape):
            if not isinstance(dim, (int, np.integer)) or dim <= 0:
                raise ValueError(f"{name}[{i}] must be a positive integer, got {dim}")
        return tuple(int(d) for d in shape)


class TrainingHistory:
    """Tracks training metrics during model training."""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.epochs: int = 0
        self.best_epoch: int = 0
        self.best_val_loss: float = float('inf')
        self.training_time: float = 0.0
        
    def add(self, metrics: Dict[str, float]):
        """Add metrics for an epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        self.epochs += 1
        
        # Track best validation loss
        if 'val_loss' in metrics and metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            self.best_epoch = self.epochs
            
    def get(self, metric: str) -> List[float]:
        """Get history for a specific metric."""
        return self.history.get(metric, [])
    
    def summary(self) -> str:
        """Get training summary."""
        lines = ["Training Summary:", "-" * 40]
        lines.append(f"Total Epochs: {self.epochs}")
        lines.append(f"Training Time: {self.training_time:.2f}s")
        if self.best_epoch > 0:
            lines.append(f"Best Epoch: {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")
        
        # Final metrics
        lines.append("\nFinal Metrics:")
        for key, values in self.history.items():
            if values:
                lines.append(f"  {key}: {values[-1]:.4f}")
        return "\n".join(lines)


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 restore_best_weights: bool = True, monitor: str = 'val_loss',
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, epoch: int, metrics: Dict[str, float], 
                 model: 'BaseArchitecture') -> bool:
        """Check if training should stop. Returns True to stop."""
        score = metrics.get(self.monitor)
        if score is None:
            return False
            
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.get_weights()
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.get_weights()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                model.set_weights(self.best_weights)
            return True
        return False


class LearningRateScheduler:
    """Learning rate scheduling strategies."""
    
    def __init__(self, schedule: str = 'constant', initial_lr: float = 0.001,
                 decay_rate: float = 0.1, decay_steps: int = 100,
                 warmup_steps: int = 0, min_lr: float = 1e-6):
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        
    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch."""
        # Warmup phase
        if epoch < self.warmup_steps:
            return self.initial_lr * (epoch + 1) / self.warmup_steps
            
        # Adjusted epoch after warmup
        adj_epoch = epoch - self.warmup_steps
        
        if self.schedule == 'constant':
            lr = self.initial_lr
        elif self.schedule == 'step':
            lr = self.initial_lr * (self.decay_rate ** (adj_epoch // self.decay_steps))
        elif self.schedule == 'exponential':
            lr = self.initial_lr * (self.decay_rate ** adj_epoch)
        elif self.schedule == 'cosine':
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * adj_epoch / self.decay_steps))
        elif self.schedule == 'linear':
            lr = self.initial_lr * (1 - adj_epoch / self.decay_steps)
        else:
            lr = self.initial_lr
            
        return max(lr, self.min_lr)


class BaseArchitecture(ABC):
    """
    Base class for all neural network architectures.
    
    Provides:
    - Automatic parameter validation
    - Built-in training loop with callbacks
    - Metrics tracking and visualization
    - Hyperparameter tuning interface
    
    Subclasses must implement:
    - _build_network(): Construct the neural network
    - _forward(): Forward pass
    - _backward(): Backward pass
    """
    
    # Class-level hyperparameter space for tuning
    PARAM_SPACE: Dict[str, Any] = {}
    
    def __init__(self, 
                 input_shape: Union[int, Tuple[int, ...]],
                 output_shape: Union[int, Tuple[int, ...]],
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 optimizer: str = 'adam',
                 loss: str = 'cross_entropy',
                 metrics: Optional[List[str]] = None,
                 validation_split: float = 0.2,
                 early_stopping: bool = True,
                 patience: int = 10,
                 lr_schedule: str = 'constant',
                 random_state: Optional[int] = None,
                 verbose: int = 1,
                 **kwargs):
        """
        Initialize base architecture.
        
        Parameters
        ----------
        input_shape : int or tuple
            Shape of input data (excluding batch dimension)
        output_shape : int or tuple
            Shape of output (number of classes for classification)
        learning_rate : float
            Initial learning rate
        batch_size : int
            Training batch size
        epochs : int
            Maximum training epochs
        optimizer : str
            Optimizer: 'adam', 'sgd', 'rmsprop', 'adagrad'
        loss : str
            Loss function: 'cross_entropy', 'mse', 'mae', 'binary_cross_entropy'
        metrics : list
            Metrics to track: 'accuracy', 'precision', 'recall', 'f1'
        validation_split : float
            Fraction of data for validation
        early_stopping : bool
            Enable early stopping
        patience : int
            Early stopping patience
        lr_schedule : str
            Learning rate schedule: 'constant', 'step', 'exponential', 'cosine'
        random_state : int
            Random seed for reproducibility
        verbose : int
            Verbosity level: 0=silent, 1=progress, 2=detailed
        """
        # Validate parameters
        self.validator = ParameterValidator()
        
        self.input_shape = self.validator.validate_shape(input_shape, 'input_shape')
        self.output_shape = self.validator.validate_shape(output_shape, 'output_shape')
        self.learning_rate = self.validator.validate_positive_float(learning_rate, 'learning_rate')
        self.batch_size = self.validator.validate_positive_int(batch_size, 'batch_size')
        self.epochs = self.validator.validate_positive_int(epochs, 'epochs')
        self.optimizer_name = self.validator.validate_choice(
            optimizer, 'optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamw']
        )
        self.loss_name = self.validator.validate_choice(
            loss, 'loss', ['cross_entropy', 'mse', 'mae', 'binary_cross_entropy', 'huber']
        )
        self.validation_split = self.validator.validate_float_range(
            validation_split, 'validation_split', 0.0, 0.5
        )
        self.early_stopping_enabled = early_stopping
        self.patience = self.validator.validate_positive_int(patience, 'patience')
        self.lr_schedule = self.validator.validate_choice(
            lr_schedule, 'lr_schedule', ['constant', 'step', 'exponential', 'cosine', 'linear']
        )
        self.verbose = verbose
        
        # Set random state
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize metrics
        self.metrics_names = metrics or ['accuracy']
        
        # Training state
        self.history = TrainingHistory()
        self.is_fitted = False
        self.weights = {}
        self.n_classes = self.output_shape[0] if len(self.output_shape) == 1 else self.output_shape[-1]
        
        # Build network
        self._build_network(**kwargs)
        
        # Setup training components
        self._setup_optimizer()
        self._setup_loss()
        self._setup_lr_scheduler()
        
    @abstractmethod
    def _build_network(self, **kwargs):
        """Build the neural network architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the network."""
        pass
    
    @abstractmethod
    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass to compute gradients."""
        pass
    
    def _setup_optimizer(self):
        """Setup optimizer with momentum/adaptive learning rate."""
        self.optimizer_state = {
            'm': {},  # First moment (for Adam)
            'v': {},  # Second moment (for Adam)
            't': 0,   # Time step
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        }
        
    def _setup_loss(self):
        """Setup loss function."""
        self.loss_functions = {
            'cross_entropy': self._cross_entropy_loss,
            'mse': self._mse_loss,
            'mae': self._mae_loss,
            'binary_cross_entropy': self._bce_loss,
            'huber': self._huber_loss,
        }
        self.loss_fn = self.loss_functions[self.loss_name]
        
    def _setup_lr_scheduler(self):
        """Setup learning rate scheduler."""
        self.lr_scheduler = LearningRateScheduler(
            schedule=self.lr_schedule,
            initial_lr=self.learning_rate,
            decay_rate=0.1,
            decay_steps=self.epochs // 3 + 1,
        )
        
    def _cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cross-entropy loss for classification."""
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        if y_true.ndim == 1:
            # Sparse labels
            return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true.astype(int)]))
        else:
            # One-hot labels
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def _mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error loss."""
        return np.mean((y_true - y_pred) ** 2)
    
    def _mae_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error loss."""
        return np.mean(np.abs(y_true - y_pred))
    
    def _bce_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _huber_loss(self, y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
        """Huber loss (smooth L1)."""
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic ** 2 + delta * linear)
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        # Get predicted classes
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_class = np.argmax(y_pred, axis=1)
        else:
            y_pred_class = (y_pred.flatten() > 0.5).astype(int)
            
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_class = np.argmax(y_true, axis=1)
        else:
            y_true_class = y_true.flatten().astype(int)
        
        for metric_name in self.metrics_names:
            if metric_name == 'accuracy':
                metrics['accuracy'] = np.mean(y_true_class == y_pred_class)
            elif metric_name == 'precision':
                tp = np.sum((y_pred_class == 1) & (y_true_class == 1))
                fp = np.sum((y_pred_class == 1) & (y_true_class == 0))
                metrics['precision'] = tp / (tp + fp + 1e-10)
            elif metric_name == 'recall':
                tp = np.sum((y_pred_class == 1) & (y_true_class == 1))
                fn = np.sum((y_pred_class == 0) & (y_true_class == 1))
                metrics['recall'] = tp / (tp + fn + 1e-10)
            elif metric_name == 'f1':
                tp = np.sum((y_pred_class == 1) & (y_true_class == 1))
                fp = np.sum((y_pred_class == 1) & (y_true_class == 0))
                fn = np.sum((y_pred_class == 0) & (y_true_class == 1))
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                metrics['f1'] = 2 * precision * recall / (precision + recall + 1e-10)
                
        return metrics
    
    def _update_weights(self, gradients: Dict[str, np.ndarray], lr: float):
        """Update weights using the selected optimizer."""
        self.optimizer_state['t'] += 1
        t = self.optimizer_state['t']
        
        for name, grad in gradients.items():
            if name not in self.weights:
                continue
                
            if self.optimizer_name == 'sgd':
                self.weights[name] -= lr * grad
                
            elif self.optimizer_name == 'adam' or self.optimizer_name == 'adamw':
                if name not in self.optimizer_state['m']:
                    self.optimizer_state['m'][name] = np.zeros_like(grad)
                    self.optimizer_state['v'][name] = np.zeros_like(grad)
                    
                m = self.optimizer_state['m'][name]
                v = self.optimizer_state['v'][name]
                beta1 = self.optimizer_state['beta1']
                beta2 = self.optimizer_state['beta2']
                eps = self.optimizer_state['epsilon']
                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                
                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                
                if self.optimizer_name == 'adamw':
                    update += lr * 0.01 * self.weights[name]  # Weight decay
                    
                self.weights[name] -= update
                self.optimizer_state['m'][name] = m
                self.optimizer_state['v'][name] = v
                
            elif self.optimizer_name == 'rmsprop':
                if name not in self.optimizer_state['v']:
                    self.optimizer_state['v'][name] = np.zeros_like(grad)
                    
                v = self.optimizer_state['v'][name]
                beta2 = 0.99
                eps = 1e-8
                
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                self.weights[name] -= lr * grad / (np.sqrt(v) + eps)
                self.optimizer_state['v'][name] = v
                
            elif self.optimizer_name == 'adagrad':
                if name not in self.optimizer_state['v']:
                    self.optimizer_state['v'][name] = np.zeros_like(grad)
                    
                v = self.optimizer_state['v'][name]
                v += grad ** 2
                self.weights[name] -= lr * grad / (np.sqrt(v) + 1e-8)
                self.optimizer_state['v'][name] = v
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            callbacks: Optional[List[Callable]] = None) -> 'BaseArchitecture':
        """
        Train the neural network.
        
        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels/targets
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels/targets
        callbacks : list, optional
            Custom callbacks
            
        Returns
        -------
        self : BaseArchitecture
            Fitted model
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        # Validation split if not provided
        if X_val is None and self.validation_split > 0:
            n_val = int(len(X) * self.validation_split)
            indices = np.random.permutation(len(X))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            X_val, y_val = X[val_idx], y[val_idx]
            X, y = X[train_idx], y[train_idx]
        
        # Setup early stopping
        early_stopper = None
        if self.early_stopping_enabled:
            early_stopper = EarlyStopping(patience=self.patience)
            
        # Training loop
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_metrics = {m: 0.0 for m in self.metrics_names}
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Get learning rate for epoch
            lr = self.lr_scheduler.get_lr(epoch)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self._forward(X_batch, training=True)
                
                # Compute loss
                batch_loss = self.loss_fn(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Compute metrics
                batch_metrics = self._compute_metrics(y_batch, y_pred)
                for m, v in batch_metrics.items():
                    epoch_metrics[m] += v
                
                # Backward pass
                gradients = self._backward(y_batch, y_pred)
                
                # Update weights
                self._update_weights(gradients, lr)
            
            # Average metrics
            epoch_loss /= n_batches
            for m in epoch_metrics:
                epoch_metrics[m] /= n_batches
            
            # Validation
            val_loss = 0.0
            val_metrics = {}
            if X_val is not None:
                y_val_pred = self._forward(X_val, training=False)
                val_loss = self.loss_fn(y_val, y_val_pred)
                val_metrics = self._compute_metrics(y_val, y_val_pred)
            
            # Record history
            history_entry = {'loss': epoch_loss, 'lr': lr}
            history_entry.update(epoch_metrics)
            if X_val is not None:
                history_entry['val_loss'] = val_loss
                for m, v in val_metrics.items():
                    history_entry[f'val_{m}'] = v
            self.history.add(history_entry)
            
            # Print progress
            if self.verbose >= 1:
                self._print_progress(epoch, history_entry)
                
            # Early stopping check
            if early_stopper and early_stopper(epoch, history_entry, self):
                if self.verbose >= 1:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
                
            # Custom callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, history_entry, self)
        
        self.history.training_time = time.time() - start_time
        self.is_fitted = True
        
        if self.verbose >= 1:
            print(f"\n{self.history.summary()}")
            
        return self
    
    def _print_progress(self, epoch: int, metrics: Dict[str, float]):
        """Print training progress."""
        msg = f"Epoch {epoch + 1}/{self.epochs} - "
        msg += " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(msg)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
            
        Returns
        -------
        predictions : np.ndarray
            Model predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        X = np.asarray(X, dtype=np.float32)
        return self._forward(X, training=False)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict(X)
        if probs.ndim > 1 and probs.shape[1] > 1:
            return np.argmax(probs, axis=1)
        return (probs.flatten() > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : np.ndarray
            Test features
        y : np.ndarray
            True labels
            
        Returns
        -------
        metrics : dict
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        loss = self.loss_fn(y, y_pred)
        metrics = self._compute_metrics(y, y_pred)
        metrics['loss'] = loss
        return metrics
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy on the given test data.
        
        Parameters
        ----------
        X : np.ndarray
            Test features
        y : np.ndarray
            True labels
            
        Returns
        -------
        accuracy : float
            Classification accuracy
        """
        metrics = self.evaluate(X, y)
        return metrics.get('accuracy', 1.0 - metrics.get('loss', 0.0))
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get a copy of model weights."""
        return {k: v.copy() for k, v in self.weights.items()}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights."""
        for k, v in weights.items():
            if k in self.weights:
                self.weights[k] = v.copy()
    
    def summary(self) -> str:
        """
        Get model summary.
        
        Returns
        -------
        summary : str
            Model architecture summary
        """
        lines = [
            f"{'='*60}",
            f"Model: {self.__class__.__name__}",
            f"{'='*60}",
            f"Input Shape: {self.input_shape}",
            f"Output Shape: {self.output_shape}",
            f"",
            f"Training Configuration:",
            f"  Optimizer: {self.optimizer_name}",
            f"  Learning Rate: {self.learning_rate}",
            f"  LR Schedule: {self.lr_schedule}",
            f"  Batch Size: {self.batch_size}",
            f"  Max Epochs: {self.epochs}",
            f"  Loss: {self.loss_name}",
            f"  Metrics: {self.metrics_names}",
            f"  Early Stopping: {self.early_stopping_enabled} (patience={self.patience})",
            f"",
            f"Layers:",
        ]
        
        total_params = 0
        for name, weight in self.weights.items():
            params = weight.size
            total_params += params
            lines.append(f"  {name}: {weight.shape} ({params:,} params)")
            
        lines.append(f"")
        lines.append(f"Total Parameters: {total_params:,}")
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)
    
    def plot_history(self, metrics: Optional[List[str]] = None, 
                     figsize: Tuple[int, int] = (12, 4),
                     save_path: Optional[str] = None):
        """
        Plot training history.
        
        Parameters
        ----------
        metrics : list, optional
            Metrics to plot. Defaults to ['loss', 'accuracy']
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        if not self.history.epochs:
            print("No training history to plot")
            return
            
        metrics = metrics or ['loss'] + self.metrics_names
        n_plots = len(metrics)
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics):
            if metric in self.history.history:
                ax.plot(self.history.history[metric], label=f'Train {metric}')
            if f'val_{metric}' in self.history.history:
                ax.plot(self.history.history[f'val_{metric}'], label=f'Val {metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Test features
        y : np.ndarray
            True labels
        class_names : list, optional
            Names of classes
        save_path : str, optional
            Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting")
            return
        
        y_pred = self.predict_classes(X)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        
        n_classes = self.n_classes
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y.astype(int), y_pred):
            cm[true, pred] += 1
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
            
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=class_names,
               yticklabels=class_names,
               xlabel='Predicted',
               ylabel='True',
               title='Confusion Matrix')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @classmethod
    def get_param_space(cls) -> Dict[str, Any]:
        """
        Get hyperparameter search space for tuning.
        
        Returns
        -------
        param_space : dict
            Dictionary defining hyperparameter ranges
        """
        return cls.PARAM_SPACE
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters (for sklearn compatibility)."""
        return {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer': self.optimizer_name,
            'loss': self.loss_name,
            'validation_split': self.validation_split,
            'early_stopping': self.early_stopping_enabled,
            'patience': self.patience,
            'lr_schedule': self.lr_schedule,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }
    
    def set_params(self, **params) -> 'BaseArchitecture':
        """Set model parameters (for sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def save(self, path: str):
        """
        Save model to file.
        
        Parameters
        ----------
        path : str
            Path to save the model
        """
        import pickle
        model_data = {
            'class': self.__class__.__name__,
            'params': self.get_params(),
            'weights': self.weights,
            'history': self.history.history,
            'is_fitted': self.is_fitted,
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseArchitecture':
        """
        Load model from file.
        
        Parameters
        ----------
        path : str
            Path to load the model from
            
        Returns
        -------
        model : BaseArchitecture
            Loaded model
        """
        import pickle
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(**model_data['params'])
        model.weights = model_data['weights']
        model.history.history = model_data['history']
        model.is_fitted = model_data['is_fitted']
        return model
