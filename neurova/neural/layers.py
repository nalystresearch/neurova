# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Neurova neural: core layers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from neurova.neural.module import Module, Parameter
from neurova.neural.tensor import Tensor, tensor


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # kaiming-like uniform init (simple)
        limit = np.sqrt(1.0 / max(1, in_features))
        w = rng.uniform(-limit, limit, size=(out_features, in_features)).astype(np.float32)
        self.weight = Parameter(w)
        if bias:
            b = np.zeros((out_features,), dtype=np.float32)
            self.bias = Parameter(b)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers: List[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.