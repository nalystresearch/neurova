# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Neurova neural: module system.

Provides a small neural module/parameter structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from neurova.neural.tensor import Tensor, tensor


class Parameter(Tensor):
    """A trainable tensor."""

    def __init__(self, data: np.ndarray):
        super().__init__(data=data, requires_grad=True)


class Module:
    """Base class for neural modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []

        def visit(obj: Any) -> None:
            if isinstance(obj, Parameter):
                params.append(obj)
            elif isinstance(obj, Module):
                for v in obj.__dict__.values():
                    visit(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    visit(v)
            elif isinstance(obj, dict):
                for v in obj.values():
                    visit(v)

        visit(self)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.