# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Training Utilities and Visualization for Neurova Architecture

Provides helper functions for training, evaluation, and visualization.
Includes callbacks, metrics, and plotting utilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time


# Callbacks

class Callback:
    """Base callback class."""
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        pass
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None):
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.
    
    Parameters
    ----------
    callbacks : list
        List of Callback objects
    
    Example
    -------
    >>> callbacks = CallbackList([
    ...     EarlyStopping(patience=5),
    ...     ModelCheckpoint('model.pkl'),
    ... ])
    >>> callbacks.on_epoch_end(epoch=5, logs={'loss': 0.5})
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    
    Parameters
    ----------
    monitor : str
        Metric to monitor
    patience : int
        Number of epochs with no improvement before stopping
    min_delta : float
        Minimum change to qualify as improvement
    restore_best_weights : bool
        Restore model weights from best epoch
    mode : str
        'min' or 'max' - direction of improvement
    
    Example
    -------
    >>> early_stop = EarlyStopping(monitor='val_loss', patience=10)
    >>> model.fit(X, y, callbacks=[early_stop])
    """
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 min_delta: float = 1e-4, restore_best_weights: bool = True,
                 mode: str = 'min', verbose: int = 1):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = None
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.best_value is None:
            self.best_value = current
            return
        
        if self.mode == 'min':
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current
            self.counter = 0
            # Store weights (would need model reference)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")


class ModelCheckpoint(Callback):
    """
    Save the model after every epoch (if improved).
    
    Parameters
    ----------
    filepath : str
        Path to save the model
    monitor : str
        Metric to monitor
    save_best_only : bool
        Only save if metric improved
    mode : str
        'min' or 'max'
    
    Example
    -------
    >>> checkpoint = ModelCheckpoint('best_model.pkl', monitor='val_accuracy', mode='max')
    >>> model.fit(X, y, callbacks=[checkpoint])
    """
    
    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 save_best_only: bool = True, mode: str = 'min',
                 verbose: int = 1):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.best_value = None
        self.model = None
        
    def set_model(self, model):
        self.model = model
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        elif self.best_value is None:
            should_save = True
            self.best_value = current
        elif self.mode == 'min' and current < self.best_value:
            should_save = True
            self.best_value = current
        elif self.mode == 'max' and current > self.best_value:
            should_save = True
            self.best_value = current
        
        if should_save and self.model is not None:
            self.model.save(self.filepath)
            if self.verbose:
                print(f"\nModel saved to {self.filepath}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback.
    
    Parameters
    ----------
    schedule : callable
        Function that takes epoch index and returns learning rate
    
    Example
    -------
    >>> def lr_schedule(epoch):
    ...     return 0.001 * (0.9 ** epoch)
    >>> lr_scheduler = LearningRateScheduler(lr_schedule)
    """
    
    def __init__(self, schedule: Callable[[int], float], verbose: int = 0):
        self.schedule = schedule
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        lr = self.schedule(epoch)
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.6f}")


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when metric plateaus.
    
    Parameters
    ----------
    monitor : str
        Metric to monitor
    factor : float
        Factor to reduce LR by
    patience : int
        Epochs to wait before reducing
    min_lr : float
        Minimum learning rate
    
    Example
    -------
    >>> reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    """
    
    def __init__(self, monitor: str = 'val_loss', factor: float = 0.1,
                 patience: int = 10, min_lr: float = 1e-6,
                 mode: str = 'min', verbose: int = 1):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = None
        self.counter = 0


class TensorBoardLogger(Callback):
    """
    Log training metrics for visualization.
    
    Parameters
    ----------
    log_dir : str
        Directory to save logs
    
    Example
    -------
    >>> logger = TensorBoardLogger('./logs')
    >>> model.fit(X, y, callbacks=[logger])
    """
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = log_dir
        self.logs = []
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        logs = logs or {}
        logs['epoch'] = epoch
        self.logs.append(logs.copy())
    
    def save_logs(self):
        """Save logs to CSV file."""
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        
        filepath = os.path.join(self.log_dir, 'training_log.csv')
        
        if not self.logs:
            return
        
        keys = list(self.logs[0].keys())
        
        with open(filepath, 'w') as f:
            f.write(','.join(keys) + '\n')
            for log in self.logs:
                values = [str(log.get(k, '')) for k in keys]
                f.write(','.join(values) + '\n')


class ProgressBar(Callback):
    """
    Display training progress bar.
    
    Example
    -------
    >>> progress = ProgressBar()
    >>> model.fit(X, y, callbacks=[progress])
    """
    
    def __init__(self, width: int = 30):
        self.width = width
        self.n_batches = 0
        
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if self.n_batches > 0:
            progress = (batch + 1) / self.n_batches
            filled = int(self.width * progress)
            bar = '=' * filled + '>' + '.' * (self.width - filled - 1)
            print(f'\r[{bar}] {progress*100:.1f}%', end='', flush=True)


# Metrics

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_class = np.argmax(y_pred, axis=1)
    else:
        y_pred_class = (y_pred.flatten() > 0.5).astype(int)
    
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_class = np.argmax(y_true, axis=1)
    else:
        y_true_class = y_true.flatten().astype(int)
    
    return np.mean(y_true_class == y_pred_class)


def precision(y_true: np.ndarray, y_pred: np.ndarray, 
              average: str = 'binary') -> float:
    """Compute precision score."""
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if average == 'binary':
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp + 1e-10)
    else:
        # Macro average
        classes = np.unique(y_true)
        precisions = []
        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            precisions.append(tp / (tp + fp + 1e-10))
        return np.mean(precisions)


def recall(y_true: np.ndarray, y_pred: np.ndarray,
           average: str = 'binary') -> float:
    """Compute recall score."""
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if average == 'binary':
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn + 1e-10)
    else:
        classes = np.unique(y_true)
        recalls = []
        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fn = np.sum((y_pred != c) & (y_true == c))
            recalls.append(tp / (tp + fn + 1e-10))
        return np.mean(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             average: str = 'binary') -> float:
    """Compute F1 score."""
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return 2 * p * r / (p + r + 1e-10)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    n_classes : int, optional
        Number of classes
    
    Returns
    -------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes)
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None) -> str:
    """
    Generate classification report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Names of classes
    
    Returns
    -------
    report : str
        Formatted classification report
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [str(c) for c in classes]
    
    lines = []
    lines.append(f"{'':15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
    lines.append("")
    
    total_support = 0
    macro_p, macro_r, macro_f1 = 0, 0, 0
    
    for i, c in enumerate(classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = np.sum(y_true == c)
        
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        
        lines.append(f"{class_names[i]:15s} {p:10.4f} {r:10.4f} {f1:10.4f} {support:10d}")
        
        total_support += support
        macro_p += p
        macro_r += r
        macro_f1 += f1
    
    lines.append("")
    lines.append(f"{'macro avg':15s} {macro_p/n_classes:10.4f} {macro_r/n_classes:10.4f} "
                f"{macro_f1/n_classes:10.4f} {total_support:10d}")
    
    acc = accuracy(y_true, y_pred)
    lines.append(f"{'accuracy':15s} {'':10s} {'':10s} {acc:10.4f} {total_support:10d}")
    
    return "\n".join(lines)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


# Plotting Utilities

def plot_training_history(history: Dict[str, List[float]],
                         metrics: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (12, 4),
                         save_path: Optional[str] = None):
    """
    Plot training history curves.
    
    Parameters
    ----------
    history : dict
        Training history dictionary
    metrics : list, optional
        Metrics to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Example
    -------
    >>> plot_training_history(model.history.history, ['loss', 'accuracy'])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting. Install: pip install matplotlib")
        return
    
    if metrics is None:
        # Get unique base metrics (without val_ prefix)
        base_metrics = set()
        for key in history.keys():
            if key.startswith('val_'):
                base_metrics.add(key[4:])
            else:
                base_metrics.add(key)
        metrics = list(base_metrics)[:4]  # Limit to 4 plots
    
    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}', linewidth=2)
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label=f'Val {metric}', 
                   linewidth=2, linestyle='--')
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Training', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False,
                         cmap: str = 'Blues',
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : list, optional
        Class labels
    normalize : bool
        Normalize values
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add text annotations
    thresh = cm.max() / 2
    fmt = '.2f' if normalize else 'd'
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                  figsize: Tuple[int, int] = (8, 6),
                  save_path: Optional[str] = None):
    """
    Plot ROC curve for binary classification.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Predicted probabilities
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    if y_scores.ndim > 1:
        y_scores = y_scores[:, 1]  # Probability of positive class
    
    # Compute ROC curve
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tpr.append(tp / (tp + fn + 1e-10))
        fpr.append(fp / (fp + tn + 1e-10))
    
    # Compute AUC
    auc = np.trapz(tpr[::-1], fpr[::-1])
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_learning_curves(train_sizes: np.ndarray,
                        train_scores: np.ndarray,
                        val_scores: np.ndarray,
                        xlabel: str = 'Training Set Size',
                        ylabel: str = 'Score',
                        title: str = 'Learning Curves',
                        figsize: Tuple[int, int] = (8, 6),
                        save_path: Optional[str] = None):
    """
    Plot learning curves.
    
    Parameters
    ----------
    train_sizes : np.ndarray
        Training set sizes
    train_scores : np.ndarray
        Training scores
    val_scores : np.ndarray
        Validation scores
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_sizes, train_scores, 'o-', linewidth=2, label='Training Score')
    ax.plot(train_sizes, val_scores, 'o-', linewidth=2, label='Validation Score')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           top_k: int = 20,
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None):
    """
    Plot feature importance.
    
    Parameters
    ----------
    importance : np.ndarray
        Feature importance values
    feature_names : list, optional
        Feature names
    top_k : int
        Number of top features to show
    figsize : tuple
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]
    
    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_k]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importance[indices], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_weights(weights: np.ndarray,
                     shape: Optional[Tuple[int, int]] = None,
                     n_show: int = 16,
                     figsize: Tuple[int, int] = (10, 10),
                     cmap: str = 'viridis',
                     save_path: Optional[str] = None):
    """
    Visualize learned weights (e.g., first layer of CNN).
    
    Parameters
    ----------
    weights : np.ndarray
        Weight matrix or filters
    shape : tuple, optional
        Shape to reshape weights for visualization
    n_show : int
        Number of filters/weights to show
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    n_show = min(n_show, len(weights))
    n_cols = int(np.ceil(np.sqrt(n_show)))
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_show):
        w = weights[i]
        if shape is not None:
            w = w.reshape(shape)
        axes[i].imshow(w, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')
    
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# Data Utilities

def train_test_split(X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.2,
                    random_state: Optional[int] = None,
                    stratify: bool = False) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    test_size : float
        Fraction for test set
    random_state : int, optional
        Random seed
    stratify : bool
        Maintain class proportions
    
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    if stratify and y.ndim == 1:
        # Stratified split
        classes = np.unique(y)
        train_idx = []
        test_idx = []
        
        for c in classes:
            c_idx = np.where(y == c)[0]
            np.random.shuffle(c_idx)
            n_c_test = max(1, int(len(c_idx) * test_size))
            test_idx.extend(c_idx[:n_c_test])
            train_idx.extend(c_idx[n_c_test:])
        
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    else:
        indices = np.random.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def cross_val_score(model, X: np.ndarray, y: np.ndarray,
                   cv: int = 5, scoring: str = 'accuracy') -> np.ndarray:
    """
    Evaluate model using cross-validation.
    
    Parameters
    ----------
    model : BaseArchitecture
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int
        Number of folds
    scoring : str
        Scoring metric
    
    Returns
    -------
    scores : np.ndarray
        Scores for each fold
    """
    n_samples = len(X)
    fold_size = n_samples // cv
    indices = np.random.permutation(n_samples)
    
    scores = []
    
    for i in range(cv):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < cv - 1 else n_samples
        
        val_idx = indices[start_idx:end_idx]
        train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone model (simplified - just reset weights)
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train, y_train)
        
        if scoring == 'accuracy':
            score = accuracy(y_val, model_copy.predict(X_val))
        elif scoring == 'mse':
            score = -mse(y_val, model_copy.predict(X_val))
        else:
            score = model_copy.score(X_val, y_val)
        
        scores.append(score)
    
    return np.array(scores)


def one_hot_encode(y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode labels.
    
    Parameters
    ----------
    y : np.ndarray
        Integer labels
    n_classes : int, optional
        Number of classes
    
    Returns
    -------
    y_onehot : np.ndarray
        One-hot encoded labels
    """
    y = y.astype(int)
    if n_classes is None:
        n_classes = y.max() + 1
    
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot


def standardize(X: np.ndarray, 
                mean: Optional[np.ndarray] = None,
                std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    mean : np.ndarray, optional
        Pre-computed mean
    std : np.ndarray, optional
        Pre-computed std
    
    Returns
    -------
    X_scaled, mean, std
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0) + 1e-10
    
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def normalize(X: np.ndarray,
              min_val: Optional[np.ndarray] = None,
              max_val: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to [0, 1] range.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    
    Returns
    -------
    X_scaled, min_val, max_val
    """
    if min_val is None:
        min_val = np.min(X, axis=0)
    if max_val is None:
        max_val = np.max(X, axis=0)
    
    X_scaled = (X - min_val) / (max_val - min_val + 1e-10)
    return X_scaled, min_val, max_val
