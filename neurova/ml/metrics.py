# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Evaluation metrics for machine learning

All implementations use NumPy only.
"""

import numpy as np
from typing import Optional, Union, List


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, 
                   average: str = 'binary', zero_division: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate precision score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary', 'micro', 'macro', or 'weighted'
        zero_division: Value to return when there is zero division
        
    Returns:
        Precision score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else zero_division
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        precisions.append(precision)
    
    precisions = np.array(precisions)
    
    if average == 'micro':
        tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
        fp_total = np.sum([np.sum((y_true != cls) & (y_pred == cls)) for cls in classes])
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else zero_division
    elif average == 'macro':
        return float(np.mean(precisions))
    elif average == 'weighted':
        weights = np.array([np.sum(y_true == cls) for cls in classes])
        return np.average(precisions, weights=weights)
    else:
        return precisions


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, 
                average: str = 'binary', zero_division: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate recall score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary', 'micro', 'macro', or 'weighted'
        zero_division: Value to return when there is zero division
        
    Returns:
        Recall score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else zero_division
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        recalls.append(recall)
    
    recalls = np.array(recalls)
    
    if average == 'micro':
        tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
        fn_total = np.sum([np.sum((y_true == cls) & (y_pred != cls)) for cls in classes])
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else zero_division
    elif average == 'macro':
        return float(np.mean(recalls))
    elif average == 'weighted':
        weights = np.array([np.sum(y_true == cls) for cls in classes])
        return np.average(recalls, weights=weights)
    else:
        return recalls


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, 
            average: str = 'binary', zero_division: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate F1 score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary', 'micro', 'macro', or 'weighted'
        zero_division: Value to return when there is zero division
        
    Returns:
        F1 score(s)
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    
    if isinstance(precision, np.ndarray):
        f1 = np.zeros_like(precision)
        mask = (precision + recall) > 0
        f1[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
        f1[~mask] = zero_division
        return f1
    else:
        if (precision + recall) > 0:
            return 2 * (precision * recall) / (precision + recall)
        return zero_division


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: Optional[np.ndarray] = None) -> dict:
    """
    Build classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        
    Returns:
        Dictionary with metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if labels is None:
        labels = classes
    
    report = {}
    
    for i, cls in enumerate(classes):
        label = labels[i] if i < len(labels) else cls
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        report[label] = {
            'precision': precision_score(y_true_binary, y_pred_binary, average='binary'),
            'recall': recall_score(y_true_binary, y_pred_binary, average='binary'),
            'f1-score': f1_score(y_true_binary, y_pred_binary, average='binary'),
            'support': np.sum(y_true == cls)
        }
    
    # add overall metrics
    report['accuracy'] = accuracy_score(y_true, y_pred)
    report['macro avg'] = {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1-score': f1_score(y_true, y_pred, average='macro'),
        'support': len(y_true)
    }
    report['weighted avg'] = {
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1-score': f1_score(y_true, y_pred, average='weighted'),
        'support': len(y_true)
    }
    
    return report


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean squared error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Mean squared error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean absolute error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Mean absolute error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.