# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Drawing utilities for Neurova.

These helpers are intentionally simple and NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from neurova.core.errors import ValidationError
from neurova.features.types import Keypoint


Color = Tuple[int, int, int]
Point = Tuple[int, int]


@dataclass(frozen=True)
class Match:
    """A minimal descriptor match linking indices between keypoint sets."""

    query_idx: int
    train_idx: int


def draw_rectangle(
    image: np.ndarray,
    pt1: Point,
    pt2: Point,
    *,
    color: Color = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    img = _as_drawable(image)
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])

    if thickness <= 0:
        raise ValidationError("thickness", thickness, "> 0")

    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))

    for t in range(thickness):
        _draw_hline(img, y_min + t, x_min, x_max, color)
        _draw_hline(img, y_max - t, x_min, x_max, color)
        _draw_vline(img, x_min + t, y_min, y_max, color)
        _draw_vline(img, x_max - t, y_min, y_max, color)

    return img


def draw_line(
    image: np.ndarray,
    pt1: Point,
    pt2: Point,
    *,
    color: Color = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    img = _as_drawable(image)
    x0, y0 = int(pt1[0]), int(pt1[1])
    x1, y1 = int(pt2[0]), int(pt2[1])

    if thickness <= 0:
        raise ValidationError("thickness", thickness, "> 0")

    # bresenham.
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        _draw_disk(img, (x0, y0), radius=max(0, thickness - 1), color=color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return img


def draw_circle(
    image: np.ndarray,
    center: Point,
    radius: int,
    *,
    color: Color = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    img = _as_drawable(image)
    r = int(radius)
    if r <= 0:
        raise ValidationError("radius", radius, "> 0")

    if thickness <= 0:
        raise ValidationError("thickness", thickness, "> 0")

    # simple: draw as many rings as thickness.
    for t in range(thickness):
        _draw_circle_outline(img, center, r - t, color)

    return img


def draw_keypoints(
    image: np.ndarray,
    keypoints: Sequence[Keypoint],
    *,
    color: Color = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    img = _as_drawable(image)
    for kp in keypoints:
        _draw_disk(img, (int(round(kp.x)), int(round(kp.y))), radius=radius, color=color)
    return img


def draw_matches(
    img1: np.ndarray,
    kp1: Sequence[Keypoint],
    img2: np.ndarray,
    kp2: Sequence[Keypoint],
    matches: Sequence[Match] | Sequence[tuple[int, int]],
    *,
    color: Color = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    """Visualize matches by concatenating images and drawing connecting lines."""

    a = _as_drawable(img1)
    b = _as_drawable(img2)

    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + b.shape[1]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] : a.shape[1] + b.shape[1]] = b

    offset_x = a.shape[1]

    # convert tuple matches to Match.
    mlist: List[Match] = []
    for m in matches:
        if isinstance(m, Match):
            mlist.append(m)
        else:
            mlist.append(Match(query_idx=int(m[0]), train_idx=int(m[1])))

    for m in mlist:
        if m.query_idx < 0 or m.query_idx >= len(kp1):
            continue
        if m.train_idx < 0 or m.train_idx >= len(kp2):
            continue

        p1 = (int(round(kp1[m.query_idx].x)), int(round(kp1[m.query_idx].y)))
        p2 = (
            int(round(kp2[m.train_idx].x)) + offset_x,
            int(round(kp2[m.train_idx].y)),
        )

        _draw_disk(canvas, p1, radius=radius, color=color)
        _draw_disk(canvas, p2, radius=radius, color=color)
        draw_line(canvas, p1, p2, color=color, thickness=1)

    return canvas


def _as_drawable(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr3 = np.stack([arr, arr, arr], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr3 = arr
    else:
        raise ValidationError("image", arr.shape, "(H,W) or (H,W,3) array")

    # convert to uint8 for drawing.
    if arr3.dtype != np.uint8:
        a = arr3.astype(np.float64)
        if a.size > 0 and float(a.max()) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0.0, 255.0)
        arr3 = a.round().astype(np.uint8)

    return arr3.copy()


def _put_pixel(img: np.ndarray, x: int, y: int, color: Color) -> None:
    h, w = img.shape[0], img.shape[1]
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    img[y, x, 0] = np.uint8(color[0])
    img[y, x, 1] = np.uint8(color[1])
    img[y, x, 2] = np.uint8(color[2])


def _draw_hline(img: np.ndarray, y: int, x0: int, x1: int, color: Color) -> None:
    if y < 0 or y >= img.shape[0]:
        return
    x0, x1 = int(min(x0, x1)), int(max(x0, x1))
    x0 = max(0, x0)
    x1 = min(img.shape[1] - 1, x1)
    img[y, x0 : x1 + 1, :] = np.array(color, dtype=np.uint8)


def _draw_vline(img: np.ndarray, x: int, y0: int, y1: int, color: Color) -> None:
    if x < 0 or x >= img.shape[1]:
        return
    y0, y1 = int(min(y0, y1)), int(max(y0, y1))
    y0 = max(0, y0)
    y1 = min(img.shape[0] - 1, y1)
    img[y0 : y1 + 1, x, :] = np.array(color, dtype=np.uint8)


def _draw_disk(img: np.ndarray, center: Point, *, radius: int, color: Color) -> None:
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    if r <= 0:
        _put_pixel(img, cx, cy, color)
        return

    y0 = max(0, cy - r)
    y1 = min(img.shape[0] - 1, cy + r)
    x0 = max(0, cx - r)
    x1 = min(img.shape[1] - 1, cx + r)

    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= r * r
    patch = img[y0 : y1 + 1, x0 : x1 + 1]
    patch[mask] = np.array(color, dtype=np.uint8)


def _draw_circle_outline(img: np.ndarray, center: Point, radius: int, color: Color) -> None:
    r = int(radius)
    if r <= 0:
        return

    cx, cy = int(center[0]), int(center[1])
    x = r
    y = 0
    err = 0

    while x >= y:
        pts = [
            (cx + x, cy + y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx - x, cy + y),
            (cx - x, cy - y),
            (cx - y, cy - x),
            (cx + y, cy - x),
            (cx + x, cy - y),
        ]
        for px, py in pts:
            _put_pixel(img, px, py, color)

        y += 1
        err += 1 + 2 * y
        if 2 * (err - x) + 1 > 0:
            x -= 1
            err += 1 - 2 * x
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.