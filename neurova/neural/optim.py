# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Neurova neural: optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from neurova.neural.module import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        self.params = list(params)
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-2, momentum: float = 0.0) -> None:
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self._v: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.momentum > 0.0:
                vid = id(p)
                v = self._v.get(vid)
                if v is None:
                    v = np.zeros_like(g)
                v = self.momentum * v + g
                self._v[vid] = v
                g = v
            p.data = p.data - self.lr * g


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            pid = id(p)

            m = self.m.get(pid)
            v = self.v.get(pid)
            if m is None:
                m = np.zeros_like(g)
            if v is None:
                v = np.zeros_like(g)

            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            m_hat = m / (1.0 - (self.beta1 ** self.t))
            v_hat = v / (1.0 - (self.beta2 ** self.t))

            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.m[pid] = m
            self.v[pid] = v
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.