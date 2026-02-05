# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Loss functions for training neural networks.

Neurova implementation loss functions.
"""

from __future__ import annotations
import numpy as np
from neurova.nn.layers import Module
from neurova.nn.autograd import Tensor


class MSELoss(Module):
    """
    Mean Squared Error loss.
    
    Neurova implementation
    
    L = (1/n) * sum((y_pred - y_true)^2)
    """
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute MSE loss."""
        diff = pred - target
        return (diff * diff).mean()


class L1Loss(Module):
    """
    Mean Absolute Error loss.
    
    Neurova implementation
    
    L = (1/n) * sum(|y_pred - y_true|)
    """
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute L1 loss."""
        diff = pred - target
        # absolute value via sqrt(x^2)
        return (diff * diff).sum() ** 0.5 / pred.size


class CrossEntropyLoss(Module):
    """
    Cross entropy loss for classification.
    
    Neurova implementation
    Combines LogSoftmax and NLLLoss.
    """
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute cross entropy loss.
        
        Parameters
        ----------
        pred : Tensor
            Predictions of shape (N, C) where C is number of classes
        target : Tensor
            Target class indices of shape (N,)
        """
        # log softmax
        exp_x = Tensor(np.exp(pred.data - np.max(pred.data, axis=-1, keepdims=True)))
        log_softmax = pred - Tensor(np.log(exp_x.data.sum(axis=-1, keepdims=True)))
        
        # negative log likelihood
        N = pred.shape[0]
        C = pred.shape[1]
        
        # create one-hot encoding
        target_data = target.data.astype(int).flatten()
        one_hot = np.zeros((N, C), dtype=np.float32)
        one_hot[np.arange(N), target_data] = 1.0
        
        # compute loss
        loss = -(log_softmax * Tensor(one_hot)).sum() / N
        return loss


class NLLLoss(Module):
    """
    Negative Log Likelihood loss.
    
    Neurova implementation
    """
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute NLL loss."""
        N = pred.shape[0]
        C = pred.shape[1]
        
        # create one-hot encoding
        target_data = target.data.astype(int).flatten()
        one_hot = np.zeros((N, C), dtype=np.float32)
        one_hot[np.arange(N), target_data] = 1.0
        
        # compute loss
        loss = -(pred * Tensor(one_hot)).sum() / N
        return loss


class BCELoss(Module):
    """
    Binary Cross Entropy loss.
    
    Neurova implementation
    """
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute BCE loss."""
        eps = 1e-7
        pred_clipped = Tensor(np.clip(pred.data, eps, 1 - eps))
        
        # bCE = -[y*log(p) + (1-y)*log(1-p)]
        term1 = target * Tensor(np.log(pred_clipped.data))
        term2 = (1 - target) * Tensor(np.log(1 - pred_clipped.data))
        return -(term1 + term2).mean()


class BCEWithLogitsLoss(Module):
    """
    Binary Cross Entropy with Logits loss (more numerically stable).
    
    Neurova implementation
    """
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute BCE with logits loss."""
        # numerically stable: max(x,0) - x*y + log(1 + exp(-|x|))
        max_val = Tensor(np.maximum(pred.data, 0))
        loss = max_val - pred * target + Tensor(np.log(1 + np.exp(-np.abs(pred.data))))
        return loss.mean()


class SmoothL1Loss(Module):
    """
    Smooth L1 loss (Huber loss).
    
    Neurova implementation
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute smooth L1 loss."""
        diff = pred - target
        abs_diff = Tensor(np.abs(diff.data))
        
        # use L2 when |diff| < beta, L1 otherwise
        l2_loss = 0.5 * (diff * diff) / self.beta
        l1_loss = abs_diff - 0.5 * self.beta
        
        # combine using mask
        mask = (abs_diff.data < self.beta).astype(np.float32)
        loss = Tensor(mask) * l2_loss + Tensor(1 - mask) * l1_loss
        return loss.mean()


class HuberLoss(Module):
    """
    Huber loss.
    
    Neurova implementation
    """
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Huber loss."""
        diff = pred - target
        abs_diff = Tensor(np.abs(diff.data))
        
        # quadratic when |diff| <= delta, linear otherwise
        quadratic = 0.5 * (diff * diff)
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        
        mask = (abs_diff.data <= self.delta).astype(np.float32)
        loss = Tensor(mask) * quadratic + Tensor(1 - mask) * linear
        return loss.mean()


class KLDivLoss(Module):
    """
    Kullback-Leibler divergence loss.
    
    Neurova implementation
    """
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute KL divergence loss."""
        eps = 1e-7
        target_clipped = Tensor(np.clip(target.data, eps, 1))
        pred_clipped = Tensor(np.clip(pred.data, eps, 1))
        
        # kL(P||Q) = sum(P * log(P/Q))
        loss = target * (Tensor(np.log(target_clipped.data)) - Tensor(np.log(pred_clipped.data)))
        return loss.sum()


class PoissonNLLLoss(Module):
    """
    Poisson Negative Log Likelihood loss.
    
    Neurova implementation
    """
    
    def __init__(self, log_input: bool = True):
        super().__init__()
        self.log_input = log_input
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Poisson NLL loss."""
        if self.log_input:
            # pred is log(lambda)
            loss = Tensor(np.exp(pred.data)) - target * pred
        else:
            # pred is lambda
            eps = 1e-7
            loss = pred - target * Tensor(np.log(pred.data + eps))
        
        return loss.mean()


class CTCLoss(Module):
    """
    Connectionist Temporal Classification loss (simplified).
    
    Neurova implementation (simplified version).
    Note: Full CTC requires dynamic programming - this is a placeholder.
    """
    
    def forward(self, pred: Tensor, target: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        """
        Compute CTC loss (simplified).
        
        Note: This is a placeholder. Full CTC implementation requires
        complex dynamic programming algorithms.
        """
        raise NotImplementedError("Full CTC loss requires complex DP - use specialized library")


class TripletMarginLoss(Module):
    """
    Triplet Margin Loss for learning embeddings.
    
    L(a, p, n) = max(0, d(a, p) - d(a, n) + margin)
    
    where d is the distance function (typically L2 or cosine).
    
    Parameters
    ----------
    margin : float, default=1.0
        Margin for triplet loss
    p : float, default=2.0
        The norm degree for distance calculation
    eps : float, default=1e-6
        Small value for numerical stability
    swap : bool, default=False
        Whether to use the swap variant
    reduction : str, default='mean'
        Reduction type: 'none', 'mean', 'sum'
    """
    
    def __init__(self, margin: float = 1.0, p: float = 2.0, 
                 eps: float = 1e-6, swap: bool = False,
                 reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction
    
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """
        Compute triplet margin loss.
        
        Parameters
        ----------
        anchor : Tensor
            Anchor samples of shape (N, D)
        positive : Tensor
            Positive samples of shape (N, D)
        negative : Tensor
            Negative samples of shape (N, D)
        """
        a = anchor.data if hasattr(anchor, 'data') else anchor
        p = positive.data if hasattr(positive, 'data') else positive
        n = negative.data if hasattr(negative, 'data') else negative
        
        # Compute distances
        d_ap = np.linalg.norm(a - p, ord=self.p, axis=-1)
        d_an = np.linalg.norm(a - n, ord=self.p, axis=-1)
        
        if self.swap:
            d_pn = np.linalg.norm(p - n, ord=self.p, axis=-1)
            d_an = np.minimum(d_an, d_pn)
        
        # Triplet loss
        losses = np.maximum(d_ap - d_an + self.margin, 0)
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class TripletMarginWithDistanceLoss(Module):
    """
    Triplet Margin Loss with custom distance function.
    
    Parameters
    ----------
    distance_function : callable, optional
        Custom distance function. If None, uses Euclidean.
    margin : float, default=1.0
        Margin for triplet loss
    swap : bool, default=False
        Whether to use the swap variant
    reduction : str, default='mean'
        Reduction type: 'none', 'mean', 'sum'
    """
    
    def __init__(self, distance_function=None, margin: float = 1.0,
                 swap: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.distance_function = distance_function or self._euclidean
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
    
    def _euclidean(self, x, y):
        return np.linalg.norm(x - y, axis=-1)
    
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Compute triplet margin loss with custom distance."""
        a = anchor.data if hasattr(anchor, 'data') else anchor
        p = positive.data if hasattr(positive, 'data') else positive
        n = negative.data if hasattr(negative, 'data') else negative
        
        d_ap = self.distance_function(a, p)
        d_an = self.distance_function(a, n)
        
        if self.swap:
            d_pn = self.distance_function(p, n)
            d_an = np.minimum(d_an, d_pn)
        
        losses = np.maximum(d_ap - d_an + self.margin, 0)
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class CosineEmbeddingLoss(Module):
    """
    Cosine Embedding Loss.
    
    Measures whether two inputs are similar using cosine similarity.
    
    L = 1 - cos(x1, x2), if y = 1
    L = max(0, cos(x1, x2) - margin), if y = -1
    
    Parameters
    ----------
    margin : float, default=0.0
        Margin for dissimilar pairs
    reduction : str, default='mean'
        Reduction type: 'none', 'mean', 'sum'
    """
    
    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """
        Compute cosine embedding loss.
        
        Parameters
        ----------
        input1 : Tensor of shape (N, D)
        input2 : Tensor of shape (N, D)
        target : Tensor of shape (N,) with values 1 or -1
        """
        x1 = input1.data if hasattr(input1, 'data') else input1
        x2 = input2.data if hasattr(input2, 'data') else input2
        y = target.data if hasattr(target, 'data') else target
        
        # Cosine similarity
        cos_sim = np.sum(x1 * x2, axis=-1) / (
            np.linalg.norm(x1, axis=-1) * np.linalg.norm(x2, axis=-1) + 1e-8
        )
        
        # Loss for similar pairs (y = 1): 1 - cos_sim
        # Loss for dissimilar pairs (y = -1): max(0, cos_sim - margin)
        loss_similar = 1 - cos_sim
        loss_dissimilar = np.maximum(0, cos_sim - self.margin)
        
        losses = np.where(y == 1, loss_similar, loss_dissimilar)
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class MarginRankingLoss(Module):
    """
    Margin Ranking Loss.
    
    Creates a criterion that measures the loss of ranking between inputs.
    
    L = max(0, -y * (x1 - x2) + margin)
    
    Parameters
    ----------
    margin : float, default=0.0
        Margin value
    reduction : str, default='mean'
        Reduction type: 'none', 'mean', 'sum'
    """
    
    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """
        Compute margin ranking loss.
        
        Parameters
        ----------
        input1 : Tensor
            First input
        input2 : Tensor
            Second input
        target : Tensor
            Target values (1 or -1)
        """
        x1 = input1.data if hasattr(input1, 'data') else input1
        x2 = input2.data if hasattr(input2, 'data') else input2
        y = target.data if hasattr(target, 'data') else target
        
        losses = np.maximum(0, -y * (x1 - x2) + self.margin)
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class HingeEmbeddingLoss(Module):
    """
    Hinge Embedding Loss.
    
    Measures loss for learning nonlinear embeddings.
    
    L = x, if y = 1
    L = max(0, margin - x), if y = -1
    
    Parameters
    ----------
    margin : float, default=1.0
        Margin for negative samples
    reduction : str, default='mean'
        Reduction type: 'none', 'mean', 'sum'
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute hinge embedding loss.
        
        Parameters
        ----------
        input : Tensor
            Input tensor
        target : Tensor
            Target values (1 or -1)
        """
        x = input.data if hasattr(input, 'data') else input
        y = target.data if hasattr(target, 'data') else target
        
        loss_pos = x
        loss_neg = np.maximum(0, self.margin - x)
        
        losses = np.where(y == 1, loss_pos, loss_neg)
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class MultiMarginLoss(Module):
    """
    Multi-class Margin Loss (SVM-like).
    
    L = sum_j max(0, margin - x[y] + x[j])^p / C
    
    Parameters
    ----------
    p : int, default=1
        The norm degree
    margin : float, default=1.0
        Margin value
    weight : array-like, optional
        Class weights
    reduction : str, default='mean'
        Reduction type
    """
    
    def __init__(self, p: int = 1, margin: float = 1.0, 
                 weight=None, reduction: str = 'mean'):
        super().__init__()
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute multi-margin loss."""
        x = input.data if hasattr(input, 'data') else input
        y = target.data if hasattr(target, 'data') else target
        
        n_samples, n_classes = x.shape
        y_int = y.astype(int).flatten()
        
        losses = np.zeros(n_samples)
        for i in range(n_samples):
            correct_score = x[i, y_int[i]]
            for j in range(n_classes):
                if j != y_int[i]:
                    margin_loss = np.maximum(0, self.margin - correct_score + x[i, j])
                    if self.p == 2:
                        margin_loss = margin_loss ** 2
                    if self.weight is not None:
                        margin_loss *= self.weight[y_int[i]]
                    losses[i] += margin_loss
        
        losses /= n_classes
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class MultiLabelMarginLoss(Module):
    """
    Multi-Label Margin Loss.
    
    For multi-label classification tasks.
    
    Parameters
    ----------
    reduction : str, default='mean'
        Reduction type
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute multi-label margin loss."""
        x = input.data if hasattr(input, 'data') else input
        y = target.data if hasattr(target, 'data') else target
        
        n_samples, n_classes = x.shape
        losses = np.zeros(n_samples)
        
        for i in range(n_samples):
            positive_indices = np.where(y[i] >= 0)[0]
            negative_indices = np.where(y[i] < 0)[0]
            
            for pos_idx in positive_indices:
                if y[i, pos_idx] == -1:
                    break
                for neg_idx in negative_indices:
                    margin_loss = np.maximum(0, 1 - x[i, pos_idx] + x[i, neg_idx])
                    losses[i] += margin_loss
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class MultiLabelSoftMarginLoss(Module):
    """
    Multi-Label Soft Margin Loss using sigmoid and binary cross-entropy.
    
    Parameters
    ----------
    weight : array-like, optional
        Class weights
    reduction : str, default='mean'
        Reduction type
    """
    
    def __init__(self, weight=None, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute multi-label soft margin loss."""
        x = input.data if hasattr(input, 'data') else input
        y = target.data if hasattr(target, 'data') else target
        
        # Sigmoid
        sigmoid_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        # Binary cross-entropy per class
        eps = 1e-7
        bce = -y * np.log(sigmoid_x + eps) - (1 - y) * np.log(1 - sigmoid_x + eps)
        
        if self.weight is not None:
            bce = bce * self.weight
        
        # Sum over classes
        losses = np.sum(bce, axis=-1)
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class SoftMarginLoss(Module):
    """
    Soft Margin Loss (two-class classification).
    
    L = log(1 + exp(-y * x))
    
    Parameters
    ----------
    reduction : str, default='mean'
        Reduction type
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute soft margin loss."""
        x = input.data if hasattr(input, 'data') else input
        y = target.data if hasattr(target, 'data') else target
        
        # log(1 + exp(-y * x))
        yx = -y * x
        losses = np.log1p(np.exp(np.clip(yx, -500, 500)))
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class FocalLoss(Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Parameters
    ----------
    alpha : float, default=1.0
        Weighting factor
    gamma : float, default=2.0
        Focusing parameter
    reduction : str, default='mean'
        Reduction type
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute focal loss."""
        x = input.data if hasattr(input, 'data') else input
        y = target.data if hasattr(target, 'data') else target
        
        # Get class probabilities
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        n_samples = x.shape[0]
        n_classes = x.shape[1]
        y_int = y.astype(int).flatten()
        
        # Get p_t (probability of correct class)
        p_t = probs[np.arange(n_samples), y_int]
        
        # Focal loss
        losses = -self.alpha * ((1 - p_t) ** self.gamma) * np.log(p_t + 1e-8)
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class GaussianNLLLoss(Module):
    """
    Gaussian Negative Log Likelihood Loss.
    
    For regression with uncertainty estimation.
    
    Parameters
    ----------
    full : bool, default=False
        Include constant term
    eps : float, default=1e-6
        Minimum variance
    reduction : str, default='mean'
        Reduction type
    """
    
    def __init__(self, full: bool = False, eps: float = 1e-6,
                 reduction: str = 'mean'):
        super().__init__()
        self.full = full
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        """
        Compute Gaussian NLL loss.
        
        Parameters
        ----------
        input : Tensor
            Predicted mean
        target : Tensor
            Target values
        var : Tensor
            Predicted variance
        """
        x = input.data if hasattr(input, 'data') else input
        y = target.data if hasattr(target, 'data') else target
        v = var.data if hasattr(var, 'data') else var
        
        v = np.maximum(v, self.eps)
        
        # NLL: 0.5 * (log(var) + (y - x)^2 / var)
        losses = 0.5 * (np.log(v) + (y - x) ** 2 / v)
        
        if self.full:
            losses = losses + 0.5 * np.log(2 * np.pi)
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class DiceLoss(Module):
    """
    Dice Loss for segmentation tasks.
    
    L = 1 - (2 * |X ∩ Y| / (|X| + |Y|))
    
    Parameters
    ----------
    smooth : float, default=1.0
        Smoothing factor to prevent division by zero
    reduction : str, default='mean'
        Reduction type
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute Dice loss."""
        x = input.data if hasattr(input, 'data') else input
        y = target.data if hasattr(target, 'data') else target
        
        # Flatten to (N, -1)
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        
        # Dice coefficient
        intersection = np.sum(x_flat * y_flat, axis=1)
        union = np.sum(x_flat, axis=1) + np.sum(y_flat, axis=1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        losses = 1 - dice
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class IoULoss(Module):
    """
    Intersection over Union (IoU) Loss for bounding box regression.
    
    Parameters
    ----------
    reduction : str, default='mean'
        Reduction type
    eps : float, default=1e-6
        Small constant for numerical stability
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute IoU loss for bounding boxes.
        
        Parameters
        ----------
        pred : Tensor of shape (N, 4)
            Predicted boxes [x1, y1, x2, y2]
        target : Tensor of shape (N, 4)
            Target boxes [x1, y1, x2, y2]
        """
        p = pred.data if hasattr(pred, 'data') else pred
        t = target.data if hasattr(target, 'data') else target
        
        # Intersection
        inter_x1 = np.maximum(p[:, 0], t[:, 0])
        inter_y1 = np.maximum(p[:, 1], t[:, 1])
        inter_x2 = np.minimum(p[:, 2], t[:, 2])
        inter_y2 = np.minimum(p[:, 3], t[:, 3])
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Areas
        pred_area = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
        target_area = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
        
        # Union
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + self.eps)
        losses = 1 - iou
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


class ContrastiveLoss(Module):
    """
    Contrastive Loss for metric learning.
    
    L = (1 - y) * 0.5 * D^2 + y * 0.5 * max(0, margin - D)^2
    
    Parameters
    ----------
    margin : float, default=1.0
        Margin for dissimilar pairs
    reduction : str, default='mean'
        Reduction type
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """
        Compute contrastive loss.
        
        Parameters
        ----------
        input1, input2 : Tensor
            Embedding pairs
        target : Tensor
            Labels (0 for similar, 1 for dissimilar)
        """
        x1 = input1.data if hasattr(input1, 'data') else input1
        x2 = input2.data if hasattr(input2, 'data') else input2
        y = target.data if hasattr(target, 'data') else target
        
        # Euclidean distance
        d = np.linalg.norm(x1 - x2, axis=-1)
        
        # Contrastive loss
        loss_similar = (1 - y) * 0.5 * d ** 2
        loss_dissimilar = y * 0.5 * np.maximum(0, self.margin - d) ** 2
        losses = loss_similar + loss_dissimilar
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.