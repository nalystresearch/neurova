# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Neurova neural: functional helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from neurova.neural.tensor import Tensor


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(logits: Tensor, target: np.ndarray, *, reduction: str = "mean") -> Tensor:
    """Cross entropy for integer class indices.

    Args:
        logits: (N, C)
        target: (N,) int64/int32
    """

    if logits.data.ndim != 2:
        raise ValueError("cross_entropy expects logits with shape (N, C)")
    if target.ndim != 1:
        raise ValueError("cross_entropy expects target with shape (N,)")
    if target.shape[0] != logits.data.shape[0]:
        raise ValueError("target length must match logits batch size")

    N, C = logits.data.shape
    probs = softmax(logits.data, axis=1)
    # numerical safety
    eps = 1e-12
    p = probs[np.arange(N), target.astype(int)]
    losses = -np.log(np.clip(p, eps, 1.0))

    if reduction == "mean":
        out_data = losses.mean()
    elif reduction == "sum":
        out_data = losses.sum()
    elif reduction == "none":
        out_data = losses
    else:
        raise ValueError("reduction must be one of: mean, sum, none")

    out = Tensor(np.asarray(out_data, dtype=logits.data.dtype), requires_grad=logits.requires_grad)
    out._prev = {logits}
    out._op = "cross_entropy"

    def _backward() -> None:
        if out.grad is None:
            return
        if not logits.requires_grad:
            return

        g_out = out.grad
        grad_logits = probs
        grad_logits[np.arange(N), target.astype(int)] -= 1.0

        if reduction == "mean":
            grad_logits = grad_logits / float(N)
        elif reduction == "sum":
            pass
        elif reduction == "none":
            # for per-sample, g_out has shape (N,). Scale each row.
            grad_logits = grad_logits * g_out.reshape(N, 1)
            logits._accumulate_grad(grad_logits)
            return

        logits._accumulate_grad(grad_logits * float(g_out))

    out._backward = _backward
    return out


def mse_loss(pred: Tensor, target: np.ndarray, *, reduction: str = "mean") -> Tensor:
    if pred.data.shape != target.shape:
        raise ValueError("mse_loss expects target shape to match pred")
    diff = pred - target
    sq = diff * diff
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    raise ValueError("reduction must be one of: mean, sum, none")
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.