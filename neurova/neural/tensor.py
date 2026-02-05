# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Neurova neural: NumPy-first tensors with autograd.

This is a lightweight automatic differentiation engine implemented from scratch.
It is designed to keep Neurova's core dependency footprint minimal (NumPy only).

Design goals:
- Small surface area, easy to audit.
- Reasonable broadcasting support for common ops.
- Sufficient building block for simple neural networks (MLP/CNN scaffolding later).

Notes:
- This is not intended to be a replacement for other frameworks.
- The API is intentionally compact and Neurova-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Set, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, float, int, Sequence[float], Sequence[int]]


def _as_array(x: ArrayLike, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _sum_to_shape(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum grad so that it matches a given target shape (undo broadcasting)."""

    if grad.shape == shape:
        return grad

    target_shape = shape

    # add leading singleton axes if needed.
    padded_shape = shape
    while len(padded_shape) < grad.ndim:
        padded_shape = (1,) + padded_shape

    # sum over axes where target has size 1 but grad has size > 1.
    axes = tuple(i for i, (g, s) in enumerate(zip(grad.shape, padded_shape)) if s == 1 and g != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)

    # finally reshape to the exact requested shape.
    return grad.reshape(target_shape)


@dataclass(eq=False)
class Tensor:
    """A NumPy-backed tensor supporting reverse-mode autodiff."""

    data: np.ndarray
    requires_grad: bool = False

    grad: Optional[np.ndarray] = None
    _backward: Callable[[], None] = lambda: None
    _prev: Set["Tensor"] = None  # type: ignore[assignment]
    _op: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            self.data = _as_array(self.data)
        if self._prev is None:
            self._prev = set()

    @staticmethod
    def from_array(x: ArrayLike, *, requires_grad: bool = False, dtype: Optional[np.dtype] = None) -> "Tensor":
        return Tensor(_as_array(x, dtype=dtype), requires_grad=requires_grad)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def numpy(self) -> np.ndarray:
        return self.data

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def _accumulate_grad(self, g: np.ndarray) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = g
        else:
            self.grad = self.grad + g

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() requires grad for non-scalar tensors")
            grad_arr = np.ones_like(self.data)
        else:
            grad_arr = _as_array(grad, dtype=self.data.dtype)
            if grad_arr.shape != self.data.shape:
                raise ValueError(f"grad shape {grad_arr.shape} does not match tensor shape {self.data.shape}")

        self._accumulate_grad(grad_arr)

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(v: Tensor) -> None:
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        for v in reversed(topo):
            v._backward()

    # basic representations
    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, requires_grad={self.requires_grad})"

    # identity semantics for graph bookkeeping
    __hash__ = object.__hash__

    # helpers
    def _ensure_tensor(self, other: Any) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor.from_array(other, requires_grad=False, dtype=self.data.dtype)

    # unary ops
    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "neg"

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            # grad already accumulated on out; propagate to self
            self._accumulate_grad(-out.grad)  # type: ignore[arg-type]

        out._backward = _backward
        return out

    # binary ops
    def __add__(self, other: Any) -> "Tensor":
        other_t = self._ensure_tensor(other)
        out = Tensor(self.data + other_t.data, requires_grad=self.requires_grad or other_t.requires_grad)
        out._prev = {self, other_t}
        out._op = "add"

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_sum_to_shape(out.grad, self.data.shape))
            if other_t.requires_grad:
                other_t._accumulate_grad(_sum_to_shape(out.grad, other_t.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        return self.__add__(-self._ensure_tensor(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return (-self).__add__(other)

    def __mul__(self, other: Any) -> "Tensor":
        other_t = self._ensure_tensor(other)
        out = Tensor(self.data * other_t.data, requires_grad=self.requires_grad or other_t.requires_grad)
        out._prev = {self, other_t}
        out._op = "mul"

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_sum_to_shape(out.grad * other_t.data, self.data.shape))
            if other_t.requires_grad:
                other_t._accumulate_grad(_sum_to_shape(out.grad * self.data, other_t.data.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        other_t = self._ensure_tensor(other)
        out = Tensor(self.data / other_t.data, requires_grad=self.requires_grad or other_t.requires_grad)
        out._prev = {self, other_t}
        out._op = "div"

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_sum_to_shape(out.grad / other_t.data, self.data.shape))
            if other_t.requires_grad:
                other_t._accumulate_grad(_sum_to_shape(out.grad * (-self.data) / (other_t.data ** 2), other_t.data.shape))

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Any) -> "Tensor":
        return self._ensure_tensor(other).__truediv__(self)

    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "pow"

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * (power * (self.data ** (power - 1))))

        out._backward = _backward
        return out

    # matrix multiplication
    def matmul(self, other: Any) -> "Tensor":
        other_t = self._ensure_tensor(other)
        out = Tensor(self.data @ other_t.data, requires_grad=self.requires_grad or other_t.requires_grad)
        out._prev = {self, other_t}
        out._op = "matmul"

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other_t.data.T)
            if other_t.requires_grad:
                other_t._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other: Any) -> "Tensor":
        return self.matmul(other)

    # reductions / shape ops
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "sum"

        def _backward() -> None:
            if out.grad is None:
                return
            if not self.requires_grad:
                return
            g = out.grad
            if axis is None:
                g_full = np.ones_like(self.data) * g
            else:
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis
                if not keepdims:
                    for ax in sorted(axes):
                        g = np.expand_dims(g, axis=ax)
                g_full = np.ones_like(self.data) * g
            self._accumulate_grad(g_full)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        denom = self.data.size if axis is None else np.prod(np.asarray(self.data.shape)[list(axis if isinstance(axis, tuple) else (axis,))])
        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "reshape"

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "transpose"

        inv = np.argsort(np.asarray(axes))

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad.transpose(*inv))

        out._backward = _backward
        return out

    T = property(lambda self: self.transpose())

    # elementwise non-linearities
    def relu(self) -> "Tensor":
        out_data = np.maximum(self.data, 0)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "relu"

        mask = (self.data > 0).astype(self.data.dtype)

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * mask)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "tanh"

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (1 - t ** 2))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "sigmoid"

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (s * (1.0 - s)))

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "exp"

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * e)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out_data = np.log(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "log"

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad / self.data)

        out._backward = _backward
        return out


def tensor(x: ArrayLike, *, requires_grad: bool = False, dtype: Optional[np.dtype] = None) -> Tensor:
    return Tensor.from_array(x, requires_grad=requires_grad, dtype=dtype)
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.