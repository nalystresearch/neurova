# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Binary morphology for Neurova."""

from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np

from neurova.core.constants import KernelShape
from neurova.core.errors import ValidationError
from neurova.filters.convolution import convolve2d


BorderModeStr = Literal["constant", "reflect", "replicate", "wrap"]
IntOrPair = Union[int, Tuple[int, int]]


def structuring_element(shape: KernelShape, ksize: IntOrPair) -> np.ndarray:
    """Create a binary structuring element."""

    if isinstance(ksize, int):
        kh = kw = int(ksize)
    else:
        if len(ksize) != 2:
            raise ValidationError("ksize", ksize, "int or (kh, kw)")
        kh, kw = int(ksize[0]), int(ksize[1])

    if kh <= 0 or kw <= 0:
        raise ValidationError("ksize", ksize, "positive")
    if (kh % 2) == 0 or (kw % 2) == 0:
        raise ValidationError("ksize", ksize, "odd")

    if shape == KernelShape.RECT:
        return np.ones((kh, kw), dtype=np.uint8)

    if shape == KernelShape.CROSS:
        k = np.zeros((kh, kw), dtype=np.uint8)
        k[kh // 2, :] = 1
        k[:, kw // 2] = 1
        return k

    if shape == KernelShape.ELLIPSE:
        yy, xx = np.meshgrid(np.arange(kh), np.arange(kw), indexing="ij")
        cy = (kh - 1) / 2.0
        cx = (kw - 1) / 2.0
        ry = kh / 2.0
        rx = kw / 2.0
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        return mask.astype(np.uint8)

    raise ValidationError("shape", shape, "KernelShape")


def binary_erode(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 1,
    border_mode: BorderModeStr = "constant",
    constant_value: int = 0,
) -> np.ndarray:
    """Binary erosion."""

    return _binary_morph(image, kernel, op="erode", iterations=iterations, border_mode=border_mode, constant_value=constant_value)


def binary_dilate(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 1,
    border_mode: BorderModeStr = "constant",
    constant_value: int = 0,
) -> np.ndarray:
    """Binary dilation."""

    return _binary_morph(image, kernel, op="dilate", iterations=iterations, border_mode=border_mode, constant_value=constant_value)


def binary_open(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 1,
) -> np.ndarray:
    """Binary opening (erode then dilate)."""

    out = binary_erode(image, kernel, iterations=iterations)
    out = binary_dilate(out, kernel, iterations=iterations)
    return out


def binary_close(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 1,
) -> np.ndarray:
    """Binary closing (dilate then erode)."""

    out = binary_dilate(image, kernel, iterations=iterations)
    out = binary_erode(out, kernel, iterations=iterations)
    return out


def binary_gradient(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    iterations: int = 1,
) -> np.ndarray:
    """Binary morphological gradient (dilate - erode)."""

    dil = binary_dilate(image, kernel, iterations=iterations)
    ero = binary_erode(image, kernel, iterations=iterations)

    if dil.dtype == np.bool_:
        return np.logical_and(dil, np.logical_not(ero))

    return np.clip(dil.astype(np.int16) - ero.astype(np.int16), 0, 255).astype(dil.dtype)


def _binary_morph(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    op: Literal["erode", "dilate"],
    iterations: int,
    border_mode: BorderModeStr,
    constant_value: int,
) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValidationError("image", img.shape, "2D array")

    it = int(iterations)
    if it <= 0:
        raise ValidationError("iterations", iterations, "> 0")

    k = np.asarray(kernel)
    if k.ndim != 2:
        raise ValidationError("kernel", k.shape, "2D array")
    if (k.shape[0] % 2) == 0 or (k.shape[1] % 2) == 0:
        raise ValidationError("kernel", k.shape, "odd kernel dimensions")

    kbin = (k != 0).astype(np.float64)
    ksum = float(kbin.sum())
    if ksum <= 0.0:
        raise ValidationError("kernel", k.shape, "kernel with at least one non-zero")

    is_bool = img.dtype == np.bool_
    if is_bool:
        work = img.astype(np.float64)
    else:
        # treat any non-zero as 1.
        work = (img != 0).astype(np.float64)

    out = work
    for _ in range(it):
        conv = convolve2d(out, kbin, border_mode=border_mode, constant_value=float(constant_value))
        if op == "erode":
            out = (conv >= (ksum - 1e-9)).astype(np.float64)
        else:
            out = (conv > 0.0).astype(np.float64)

    if is_bool:
        return out.astype(bool)

    return (out * 255.0).astype(img.dtype)
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.