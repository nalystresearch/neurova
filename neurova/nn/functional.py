# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Functional API.

Neurova implementation of neural network functional operations
for building and training deep learning models.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union, List
import math


# Activation Functions

def relu(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    if inplace:
        np.maximum(x, 0, out=x)
        return x
    return np.maximum(0, x)


def relu6(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """ReLU6 activation: min(max(0, x), 6)."""
    if inplace:
        np.clip(x, 0, 6, out=x)
        return x
    return np.clip(x, 0, 6)


def elu(x: np.ndarray, alpha: float = 1.0, inplace: bool = False) -> np.ndarray:
    """Exponential Linear Unit activation."""
    result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    if inplace:
        x[:] = result
        return x
    return result


def selu(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Scaled Exponential Linear Unit activation."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    result = scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    if inplace:
        x[:] = result
        return x
    return result


def celu(x: np.ndarray, alpha: float = 1.0, inplace: bool = False) -> np.ndarray:
    """Continuously Differentiable Exponential Linear Unit."""
    result = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    if inplace:
        x[:] = result
        return x
    return result


def leaky_relu(x: np.ndarray, negative_slope: float = 0.01, inplace: bool = False) -> np.ndarray:
    """Leaky ReLU activation."""
    result = np.where(x >= 0, x, negative_slope * x)
    if inplace:
        x[:] = result
        return x
    return result


def gelu(x: np.ndarray, approximate: str = 'none') -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    if approximate == 'tanh':
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    else:
        from scipy import special
        return 0.5 * x * (1 + special.erf(x / np.sqrt(2)))


def glu(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """Gated Linear Unit: a * sigmoid(b)."""
    a, b = np.split(x, 2, axis=dim)
    return a * sigmoid(b)


def silu(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Sigmoid Linear Unit (Swish): x * sigmoid(x)."""
    result = x * sigmoid(x)
    if inplace:
        x[:] = result
        return x
    return result


def mish(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Mish activation: x * tanh(softplus(x))."""
    result = x * np.tanh(softplus(x))
    if inplace:
        x[:] = result
        return x
    return result


def hardtanh(x: np.ndarray, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> np.ndarray:
    """Hard Tanh activation."""
    if inplace:
        np.clip(x, min_val, max_val, out=x)
        return x
    return np.clip(x, min_val, max_val)


def hardswish(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Hard Swish activation: x * ReLU6(x+3) / 6."""
    result = x * relu6(x + 3) / 6
    if inplace:
        x[:] = result
        return x
    return result


def hardsigmoid(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Hard Sigmoid activation: ReLU6(x+3) / 6."""
    result = relu6(x + 3) / 6
    if inplace:
        x[:] = result
        return x
    return result


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)


def softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """Softplus activation: (1/beta) * log(1 + exp(beta*x))."""
    scaled = beta * x
    return np.where(scaled > threshold, x, np.log1p(np.exp(scaled)) / beta)


def softshrink(x: np.ndarray, lambd: float = 0.5) -> np.ndarray:
    """Soft shrinkage function."""
    return np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0))


def softsign(x: np.ndarray) -> np.ndarray:
    """Softsign activation: x / (1 + |x|)."""
    return x / (1 + np.abs(x))


def tanhshrink(x: np.ndarray) -> np.ndarray:
    """Tanhshrink activation: x - tanh(x)."""
    return x - np.tanh(x)


def threshold(x: np.ndarray, threshold_val: float, value: float, inplace: bool = False) -> np.ndarray:
    """Thresholding function."""
    result = np.where(x > threshold_val, x, value)
    if inplace:
        x[:] = result
        return x
    return result


def softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """Softmax function."""
    x_max = np.max(x, axis=dim, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=dim, keepdims=True)


def softmin(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """Softmin function: Softmax(-x)."""
    return softmax(-x, dim=dim)


def log_softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """Log-Softmax function."""
    x_max = np.max(x, axis=dim, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=dim, keepdims=True))


def gumbel_softmax(logits: np.ndarray, tau: float = 1.0, hard: bool = False, dim: int = -1) -> np.ndarray:
    """Gumbel-Softmax sampling."""
    gumbels = -np.log(-np.log(np.random.uniform(1e-10, 1, logits.shape)))
    y = softmax((logits + gumbels) / tau, dim=dim)
    if hard:
        indices = np.argmax(y, axis=dim)
        y_hard = np.zeros_like(y)
        np.put_along_axis(y_hard, np.expand_dims(indices, dim), 1, axis=dim)
        return y_hard - y.copy() + y
    return y


# Normalization Functions

def batch_norm(x: np.ndarray, running_mean: Optional[np.ndarray], running_var: Optional[np.ndarray],
               weight: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None,
               training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> np.ndarray:
    """Batch Normalization."""
    if training:
        if x.ndim == 2:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
        else:
            axes = tuple(i for i in range(x.ndim) if i != 1)
            mean = np.mean(x, axis=axes)
            var = np.var(x, axis=axes)
        if running_mean is not None:
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean
        if running_var is not None:
            running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mean, var = running_mean, running_var
    
    if x.ndim == 2:
        x_norm = (x - mean) / np.sqrt(var + eps)
        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias
    else:
        shape = [1, -1] + [1] * (x.ndim - 2)
        x_norm = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)
        if weight is not None:
            x_norm = x_norm * weight.reshape(shape)
        if bias is not None:
            x_norm = x_norm + bias.reshape(shape)
    return x_norm


def layer_norm(x: np.ndarray, normalized_shape: Tuple[int, ...],
               weight: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None,
               eps: float = 1e-5) -> np.ndarray:
    """Layer Normalization."""
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    ndim = len(normalized_shape)
    axes = tuple(range(-ndim, 0))
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


def group_norm(x: np.ndarray, num_groups: int, weight: Optional[np.ndarray] = None,
               bias: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
    """Group Normalization."""
    N, C = x.shape[:2]
    spatial_shape = x.shape[2:]
    x_grouped = x.reshape(N, num_groups, C // num_groups, *spatial_shape)
    axes = tuple(range(2, x_grouped.ndim))
    mean = np.mean(x_grouped, axis=axes, keepdims=True)
    var = np.var(x_grouped, axis=axes, keepdims=True)
    x_norm = (x_grouped - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, *spatial_shape)
    if weight is not None:
        shape = [1, -1] + [1] * len(spatial_shape)
        x_norm = x_norm * weight.reshape(shape)
    if bias is not None:
        shape = [1, -1] + [1] * len(spatial_shape)
        x_norm = x_norm + bias.reshape(shape)
    return x_norm


def instance_norm(x: np.ndarray, running_mean: Optional[np.ndarray] = None,
                  running_var: Optional[np.ndarray] = None, weight: Optional[np.ndarray] = None,
                  bias: Optional[np.ndarray] = None, use_input_stats: bool = True,
                  momentum: float = 0.1, eps: float = 1e-5) -> np.ndarray:
    """Instance Normalization."""
    axes = tuple(range(2, x.ndim))
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        shape = [1, -1] + [1] * (x.ndim - 2)
        x_norm = x_norm * weight.reshape(shape)
    if bias is not None:
        shape = [1, -1] + [1] * (x.ndim - 2)
        x_norm = x_norm + bias.reshape(shape)
    return x_norm


def rms_norm(x: np.ndarray, normalized_shape: Tuple[int, ...],
             weight: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
    """Root Mean Square Normalization."""
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    ndim = len(normalized_shape)
    axes = tuple(range(-ndim, 0))
    rms = np.sqrt(np.mean(x ** 2, axis=axes, keepdims=True) + eps)
    x_norm = x / rms
    if weight is not None:
        x_norm = x_norm * weight
    return x_norm


# Dropout Functions

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True, inplace: bool = False) -> np.ndarray:
    """Dropout regularization."""
    if not training or p == 0:
        return x
    mask = np.random.binomial(1, 1 - p, x.shape) / (1 - p)
    result = x * mask
    if inplace:
        x[:] = result
        return x
    return result


def dropout2d(x: np.ndarray, p: float = 0.5, training: bool = True, inplace: bool = False) -> np.ndarray:
    """2D Dropout (drops entire channels)."""
    if not training or p == 0:
        return x
    N, C, H, W = x.shape
    mask = np.random.binomial(1, 1 - p, (N, C, 1, 1)) / (1 - p)
    result = x * mask
    if inplace:
        x[:] = result
        return x
    return result


def dropout3d(x: np.ndarray, p: float = 0.5, training: bool = True, inplace: bool = False) -> np.ndarray:
    """3D Dropout (drops entire channels)."""
    if not training or p == 0:
        return x
    N, C, D, H, W = x.shape
    mask = np.random.binomial(1, 1 - p, (N, C, 1, 1, 1)) / (1 - p)
    result = x * mask
    if inplace:
        x[:] = result
        return x
    return result


def alpha_dropout(x: np.ndarray, p: float = 0.5, training: bool = True, inplace: bool = False) -> np.ndarray:
    """Alpha Dropout for SELU activations."""
    if not training or p == 0:
        return x
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    alpha_p = -alpha * scale
    a = ((1 - p) * (1 + p * alpha_p ** 2)) ** -0.5
    b = -a * alpha_p * p
    mask = np.random.binomial(1, 1 - p, x.shape)
    result = a * (mask * x + (1 - mask) * alpha_p) + b
    if inplace:
        x[:] = result
        return x
    return result


# Loss Functions

def l1_loss(input: np.ndarray, target: np.ndarray, reduction: str = 'mean') -> np.ndarray:
    """L1 (Mean Absolute Error) Loss."""
    loss = np.abs(input - target)
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    return loss


def mse_loss(input: np.ndarray, target: np.ndarray, reduction: str = 'mean') -> np.ndarray:
    """Mean Squared Error Loss."""
    loss = (input - target) ** 2
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    return loss


def cross_entropy(input: np.ndarray, target: np.ndarray, weight: Optional[np.ndarray] = None,
                  ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> np.ndarray:
    """Cross Entropy Loss."""
    log_probs = log_softmax(input, dim=1 if input.ndim > 1 else -1)
    N = input.shape[0]
    C = input.shape[1] if input.ndim > 1 else input.shape[0]
    if input.ndim == 2:
        nll = -log_probs[np.arange(N), target.astype(int)]
    else:
        target_flat = target.reshape(-1).astype(int)
        log_probs_flat = log_probs.transpose((0, *range(2, input.ndim), 1)).reshape(-1, C)
        nll = -log_probs_flat[np.arange(len(target_flat)), target_flat]
        nll = nll.reshape(target.shape)
    if label_smoothing > 0:
        smooth_loss = -np.mean(log_probs, axis=1 if input.ndim > 1 else -1)
        nll = (1 - label_smoothing) * nll + label_smoothing * smooth_loss
    if weight is not None:
        nll = nll * weight[target.astype(int)]
    if ignore_index >= 0:
        mask = target != ignore_index
        nll = nll * mask
    if reduction == 'mean':
        if ignore_index >= 0:
            return np.sum(nll) / np.sum(mask)
        return np.mean(nll)
    elif reduction == 'sum':
        return np.sum(nll)
    return nll


def nll_loss(input: np.ndarray, target: np.ndarray, weight: Optional[np.ndarray] = None,
             ignore_index: int = -100, reduction: str = 'mean') -> np.ndarray:
    """Negative Log Likelihood Loss."""
    N = input.shape[0]
    nll = -input[np.arange(N), target.astype(int)]
    if weight is not None:
        nll = nll * weight[target.astype(int)]
    if ignore_index >= 0:
        mask = target != ignore_index
        nll = nll * mask
    if reduction == 'mean':
        if ignore_index >= 0:
            return np.sum(nll) / np.sum(mask)
        return np.mean(nll)
    elif reduction == 'sum':
        return np.sum(nll)
    return nll


def binary_cross_entropy(input: np.ndarray, target: np.ndarray, weight: Optional[np.ndarray] = None,
                         reduction: str = 'mean') -> np.ndarray:
    """Binary Cross Entropy Loss."""
    eps = 1e-7
    input = np.clip(input, eps, 1 - eps)
    loss = -target * np.log(input) - (1 - target) * np.log(1 - input)
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    return loss


def binary_cross_entropy_with_logits(input: np.ndarray, target: np.ndarray, weight: Optional[np.ndarray] = None,
                                     reduction: str = 'mean', pos_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Binary Cross Entropy with Logits Loss."""
    max_val = np.maximum(-input, 0)
    if pos_weight is not None:
        log_weight = (pos_weight - 1) * target + 1
        loss = (1 - target) * input + log_weight * (max_val + np.log(np.exp(-max_val) + np.exp(-input - max_val)))
    else:
        loss = (1 - target) * input + max_val + np.log(np.exp(-max_val) + np.exp(-input - max_val))
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    return loss


def smooth_l1_loss(input: np.ndarray, target: np.ndarray, beta: float = 1.0, reduction: str = 'mean') -> np.ndarray:
    """Smooth L1 Loss."""
    diff = np.abs(input - target)
    loss = np.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    return loss


def huber_loss(input: np.ndarray, target: np.ndarray, delta: float = 1.0, reduction: str = 'mean') -> np.ndarray:
    """Huber Loss."""
    diff = np.abs(input - target)
    loss = np.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    return loss


def kl_div(input: np.ndarray, target: np.ndarray, reduction: str = 'mean', log_target: bool = False) -> np.ndarray:
    """Kullback-Leibler Divergence Loss."""
    if log_target:
        loss = np.exp(target) * (target - input)
    else:
        loss = target * (np.log(target + 1e-10) - input)
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    elif reduction == 'batchmean':
        return np.sum(loss) / input.shape[0]
    return loss


def triplet_margin_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray,
                        margin: float = 1.0, p: float = 2.0, reduction: str = 'mean') -> np.ndarray:
    """Triplet Margin Loss."""
    d_pos = np.linalg.norm(anchor - positive, ord=p, axis=-1)
    d_neg = np.linalg.norm(anchor - negative, ord=p, axis=-1)
    loss = np.maximum(0, d_pos - d_neg + margin)
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    return loss


def cosine_embedding_loss(input1: np.ndarray, input2: np.ndarray, target: np.ndarray,
                          margin: float = 0.0, reduction: str = 'mean') -> np.ndarray:
    """Cosine Embedding Loss."""
    cos_sim = cosine_similarity(input1, input2, dim=1)
    loss = np.where(target == 1, 1 - cos_sim, np.maximum(0, cos_sim - margin))
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    return loss


# Pooling Functions

def max_pool2d(x: np.ndarray, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0,
               return_indices: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """2D Max Pooling."""
    if isinstance(kernel_size, int):
        kH, kW = kernel_size, kernel_size
    else:
        kH, kW = kernel_size
    if stride is None:
        sH, sW = kH, kW
    elif isinstance(stride, int):
        sH, sW = stride, stride
    else:
        sH, sW = stride
    if isinstance(padding, int):
        pH, pW = padding, padding
    else:
        pH, pW = padding
    
    N, C, H, W = x.shape
    if pH > 0 or pW > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant', constant_values=-np.inf)
    H_padded, W_padded = x.shape[2], x.shape[3]
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1
    output = np.zeros((N, C, H_out, W_out))
    indices = np.zeros((N, C, H_out, W_out), dtype=np.int64) if return_indices else None
    
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i * sH, j * sW
            pool_region = x[:, :, h_start:h_start + kH, w_start:w_start + kW]
            output[:, :, i, j] = np.max(pool_region, axis=(2, 3))
            if return_indices:
                flat_idx = np.argmax(pool_region.reshape(N, C, -1), axis=2)
                indices[:, :, i, j] = (h_start + flat_idx // kW) * W_padded + (w_start + flat_idx % kW)
    
    if return_indices:
        return output, indices
    return output


def avg_pool2d(x: np.ndarray, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0) -> np.ndarray:
    """2D Average Pooling."""
    if isinstance(kernel_size, int):
        kH, kW = kernel_size, kernel_size
    else:
        kH, kW = kernel_size
    if stride is None:
        sH, sW = kH, kW
    elif isinstance(stride, int):
        sH, sW = stride, stride
    else:
        sH, sW = stride
    if isinstance(padding, int):
        pH, pW = padding, padding
    else:
        pH, pW = padding
    
    N, C, H, W = x.shape
    if pH > 0 or pW > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')
    H_padded, W_padded = x.shape[2], x.shape[3]
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1
    output = np.zeros((N, C, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i * sH, j * sW
            pool_region = x[:, :, h_start:h_start + kH, w_start:w_start + kW]
            output[:, :, i, j] = np.mean(pool_region, axis=(2, 3))
    return output


def adaptive_avg_pool2d(x: np.ndarray, output_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """Adaptive 2D Average Pooling."""
    if isinstance(output_size, int):
        oH, oW = output_size, output_size
    else:
        oH, oW = output_size
    N, C, H, W = x.shape
    output = np.zeros((N, C, oH, oW))
    for i in range(oH):
        for j in range(oW):
            h_start = int(i * H / oH)
            h_end = int((i + 1) * H / oH)
            w_start = int(j * W / oW)
            w_end = int((j + 1) * W / oW)
            output[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
    return output


def adaptive_max_pool2d(x: np.ndarray, output_size: Union[int, Tuple[int, int]],
                        return_indices: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Adaptive 2D Max Pooling."""
    if isinstance(output_size, int):
        oH, oW = output_size, output_size
    else:
        oH, oW = output_size
    N, C, H, W = x.shape
    output = np.zeros((N, C, oH, oW))
    indices = np.zeros((N, C, oH, oW), dtype=np.int64) if return_indices else None
    
    for i in range(oH):
        for j in range(oW):
            h_start = int(i * H / oH)
            h_end = int((i + 1) * H / oH)
            w_start = int(j * W / oW)
            w_end = int((j + 1) * W / oW)
            region = x[:, :, h_start:h_end, w_start:w_end]
            output[:, :, i, j] = np.max(region, axis=(2, 3))
            if return_indices:
                flat_idx = np.argmax(region.reshape(N, C, -1), axis=2)
                rH, rW = h_end - h_start, w_end - w_start
                indices[:, :, i, j] = (h_start + flat_idx // rW) * W + (w_start + flat_idx % rW)
    
    if return_indices:
        return output, indices
    return output


# Convolution Functions

def conv2d(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
           stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
           dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1) -> np.ndarray:
    """2D Convolution."""
    if isinstance(stride, int):
        sH, sW = stride, stride
    else:
        sH, sW = stride
    if isinstance(padding, int):
        pH, pW = padding, padding
    else:
        pH, pW = padding
    if isinstance(dilation, int):
        dH, dW = dilation, dilation
    else:
        dH, dW = dilation
    
    N, C_in, H, W = x.shape
    C_out, C_in_g, kH, kW = weight.shape
    
    if pH > 0 or pW > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')
    
    H_out = (x.shape[2] - dH * (kH - 1) - 1) // sH + 1
    W_out = (x.shape[3] - dW * (kW - 1) - 1) // sW + 1
    output = np.zeros((N, C_out, H_out, W_out))
    
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    
    for g in range(groups):
        x_g = x[:, g * C_in_per_group:(g + 1) * C_in_per_group]
        w_g = weight[g * C_out_per_group:(g + 1) * C_out_per_group]
        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * sH, j * sW
                x_col = x_g[:, :, h_start:h_start + kH * dH:dH, w_start:w_start + kW * dW:dW]
                output[:, g * C_out_per_group:(g + 1) * C_out_per_group, i, j] = \
                    np.einsum('nchw,ochw->no', x_col, w_g)
    
    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)
    return output


# Utility Functions

def normalize(x: np.ndarray, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Normalize tensor along dimension."""
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    return x / np.maximum(norm, eps)


def pairwise_distance(x1: np.ndarray, x2: np.ndarray, p: float = 2.0, eps: float = 1e-6,
                      keepdim: bool = False) -> np.ndarray:
    """Compute pairwise distance between two tensors."""
    diff = x1 - x2
    dist = np.linalg.norm(diff, ord=p, axis=-1)
    if keepdim:
        dist = dist[..., np.newaxis]
    return dist


def cosine_similarity(x1: np.ndarray, x2: np.ndarray, dim: int = 1, eps: float = 1e-8) -> np.ndarray:
    """Compute cosine similarity."""
    x1_norm = x1 / np.maximum(np.linalg.norm(x1, axis=dim, keepdims=True), eps)
    x2_norm = x2 / np.maximum(np.linalg.norm(x2, axis=dim, keepdims=True), eps)
    return np.sum(x1_norm * x2_norm, axis=dim)


def pdist(x: np.ndarray, p: float = 2.0) -> np.ndarray:
    """Compute pairwise distances in batch."""
    N = x.shape[0]
    distances = []
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(x[i] - x[j], ord=p)
            distances.append(dist)
    return np.array(distances)


def one_hot(indices: np.ndarray, num_classes: int = -1) -> np.ndarray:
    """One-hot encoding."""
    indices = np.asarray(indices, dtype=np.int64)
    if num_classes < 0:
        num_classes = int(np.max(indices)) + 1
    shape = indices.shape + (num_classes,)
    output = np.zeros(shape)
    flat_indices = indices.flatten()
    flat_output = output.reshape(-1, num_classes)
    flat_output[np.arange(len(flat_indices)), flat_indices] = 1
    return output


def embedding(input: np.ndarray, weight: np.ndarray, padding_idx: Optional[int] = None,
              max_norm: Optional[float] = None, norm_type: float = 2.0) -> np.ndarray:
    """Embedding lookup."""
    if max_norm is not None:
        norms = np.linalg.norm(weight, ord=norm_type, axis=1, keepdims=True)
        weight = weight * np.minimum(1, max_norm / (norms + 1e-10))
    return weight[input.astype(int)]


def pad(x: np.ndarray, pad_sizes: Tuple[int, ...], mode: str = 'constant', value: float = 0) -> np.ndarray:
    """Pad tensor."""
    pad_pairs = [(0, 0)] * (x.ndim - len(pad_sizes) // 2)
    for i in range(0, len(pad_sizes), 2):
        dim_idx = x.ndim - 1 - i // 2
        pad_pairs.insert(dim_idx, (pad_sizes[i], pad_sizes[i + 1]))
    pad_width = pad_pairs[:x.ndim]
    if mode == 'constant':
        return np.pad(x, pad_width, mode='constant', constant_values=value)
    elif mode == 'reflect':
        return np.pad(x, pad_width, mode='reflect')
    elif mode == 'replicate':
        return np.pad(x, pad_width, mode='edge')
    elif mode == 'circular':
        return np.pad(x, pad_width, mode='wrap')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")


def interpolate(x: np.ndarray, size: Optional[Union[int, Tuple[int, ...]]] = None,
                scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
                mode: str = 'nearest') -> np.ndarray:
    """Interpolate tensor."""
    from scipy import ndimage
    N, C = x.shape[:2]
    spatial_dims = x.ndim - 2
    if size is not None:
        if isinstance(size, int):
            size = (size,) * spatial_dims
        scale = tuple(s / x.shape[i + 2] for i, s in enumerate(size))
    elif scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale = (scale_factor,) * spatial_dims
        else:
            scale = scale_factor
    else:
        raise ValueError("Either size or scale_factor must be specified")
    full_scale = (1.0, 1.0) + scale
    order = 0 if mode == 'nearest' else 1
    return ndimage.zoom(x, full_scale, order=order)


def pixel_shuffle(x: np.ndarray, upscale_factor: int) -> np.ndarray:
    """Rearrange elements for upscaling."""
    N, C, H, W = x.shape
    r = upscale_factor
    C_out = C // (r * r)
    x = x.reshape(N, C_out, r, r, H, W)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(N, C_out, H * r, W * r)


def pixel_unshuffle(x: np.ndarray, downscale_factor: int) -> np.ndarray:
    """Rearrange elements for downscaling."""
    N, C, H, W = x.shape
    r = downscale_factor
    H_out, W_out = H // r, W // r
    x = x.reshape(N, C, H_out, r, W_out, r)
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(N, C * r * r, H_out, W_out)


def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray,
                                  attn_mask: Optional[np.ndarray] = None, dropout_p: float = 0.0,
                                  is_causal: bool = False, scale: Optional[float] = None) -> np.ndarray:
    """Scaled Dot-Product Attention."""
    E = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(E)
    scores = np.matmul(query, np.swapaxes(key, -2, -1)) * scale
    if is_causal:
        L, S = scores.shape[-2], scores.shape[-1]
        causal_mask = np.triu(np.ones((L, S)), k=1).astype(bool)
        scores = np.where(causal_mask, -np.inf, scores)
    if attn_mask is not None:
        if attn_mask.dtype == bool:
            scores = np.where(attn_mask, -np.inf, scores)
        else:
            scores = scores + attn_mask
    attn_weights = softmax(scores, dim=-1)
    if dropout_p > 0:
        attn_weights = dropout(attn_weights, p=dropout_p, training=True)
    return np.matmul(attn_weights, value)


def linear(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Linear transformation: y = x @ W.T + b."""
    output = x @ weight.T
    if bias is not None:
        output = output + bias
    return output


def bilinear(input1: np.ndarray, input2: np.ndarray, weight: np.ndarray,
             bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Bilinear transformation."""
    output = np.einsum('...i,oij,...j->...o', input1, weight, input2)
    if bias is not None:
        output = output + bias
    return output


# Export all functions
__all__ = [
    # Activations
    'relu', 'relu6', 'elu', 'selu', 'celu', 'leaky_relu', 'gelu', 'glu', 'silu', 'mish',
    'hardtanh', 'hardswish', 'hardsigmoid', 'sigmoid', 'tanh', 'softplus', 'softshrink',
    'softsign', 'tanhshrink', 'threshold', 'softmax', 'softmin', 'log_softmax', 'gumbel_softmax',
    # Normalization
    'batch_norm', 'layer_norm', 'group_norm', 'instance_norm', 'rms_norm',
    # Dropout
    'dropout', 'dropout2d', 'dropout3d', 'alpha_dropout',
    # Loss functions
    'l1_loss', 'mse_loss', 'cross_entropy', 'nll_loss', 'binary_cross_entropy',
    'binary_cross_entropy_with_logits', 'smooth_l1_loss', 'huber_loss', 'kl_div',
    'triplet_margin_loss', 'cosine_embedding_loss',
    # Pooling
    'max_pool2d', 'avg_pool2d', 'adaptive_avg_pool2d', 'adaptive_max_pool2d',
    # Convolution
    'conv2d',
    # Utilities
    'normalize', 'pairwise_distance', 'cosine_similarity', 'pdist',
    'one_hot', 'embedding', 'pad', 'interpolate', 'pixel_shuffle', 'pixel_unshuffle',
    # Attention
    'scaled_dot_product_attention',
    # Linear
    'linear', 'bilinear',
]
