# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Neurova neural: loss modules."""

from __future__ import annotations

import numpy as np

from neurova.neural.functional import cross_entropy, mse_loss
from neurova.neural.module import Module
from neurova.neural.tensor import Tensor


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def forward(self, pred: Tensor, target: np.ndarray) -> Tensor:
        return mse_loss(pred, target, reduction=self.reduction)


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def forward(self, logits: Tensor, target: np.ndarray) -> Tensor:
        return cross_entropy(logits, target, reduction=self.reduction)
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.