# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""neurova.detection.haar_cascade

Pure-Python Haar cascade detector compatible with Neurova's XML cascade files.

This implements a small subset of Neurova's CascadeClassifier behavior:
- load Neurova Haar XML
- `detect(...)` producing (x, y, w, h, confidence)

It is designed for educational/research use and for keeping Neurova independent
from Neurova, while still supporting common Haar cascade assets.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class _WeakClassifier:
    feature_idx: int
    threshold: float
    left_val: float
    right_val: float


@dataclass
class _Stage:
    threshold: float
    classifiers: List[_WeakClassifier]


class HaarCascadeClassifier:
    """Load an Neurova Haar cascade XML and run detection."""

    def __init__(self, cascade_path: str):
        self.cascade_path = str(cascade_path)
        self.window_size: Tuple[int, int] = (24, 24)
        self.features: List[List[dict]] = []
        self.stages: List[_Stage] = []
        self.is_loaded: bool = False

        # Defaults close to Neurova behavior
        self.equalize_hist: bool = True
        self.scale_factor: float = 1.1
        self.min_neighbors: int = 5
        self.min_size: Tuple[int, int] = (30, 30)

        # Speed/quality knobs
        self.max_detection_size: int = 320
        self.step_divisor: int = 6
        self.max_scales: Optional[int] = 12
        self.nms_threshold: float = 0.3

        self._load_cascade(self.cascade_path)

    def _equalize_hist_uint8(self, gray: np.ndarray) -> np.ndarray:
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8, copy=False)
        hist = np.bincount(gray.ravel(), minlength=256)
        cdf = hist.cumsum()
        nonzero = cdf[cdf > 0]
        if nonzero.size == 0:
            return gray
        cdf_min = int(nonzero[0])
        denom = int(cdf[-1] - cdf_min)
        if denom <= 0:
            return gray
        lut = ((cdf - cdf_min) * 255 // denom).astype(np.uint8)
        return lut[gray]

    def _preprocess_gray(self, gray: np.ndarray) -> np.ndarray:
        if self.equalize_hist:
            return self._equalize_hist_uint8(gray)
        return gray

    def _to_gray_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] >= 3:
            # Accept BGR/BGRA/RGB – we just interpret channels [0,1,2] as B,G,R.
            b = image[:, :, 0].astype(np.uint16)
            g = image[:, :, 1].astype(np.uint16)
            r = image[:, :, 2].astype(np.uint16)
            gray = ((29 * r + 150 * g + 77 * b + 128) >> 8).astype(np.uint8)
        else:
            raise ValueError("Unsupported image shape")

        if gray.dtype != np.uint8:
            if np.issubdtype(gray.dtype, np.floating) and gray.max() <= 1.0:
                gray = (gray * 255.0).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        return self._preprocess_gray(gray)

    def _load_cascade(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self.is_loaded = False
            return

        try:
            tree = ET.parse(str(p))
            root = tree.getroot()

            cascade = root.find('.//cascade')
            if cascade is None:
                cascade = root.find('cascade')
            if cascade is None:
                self.is_loaded = False
                return

            width = cascade.find('.//width')
            height = cascade.find('.//height')
            if width is not None and height is not None:
                self.window_size = (int(width.text), int(height.text))

            self.features = []
            features_elem = cascade.find('.//features')
            if features_elem is not None:
                for feat in features_elem.findall('_'):
                    rects_elem = feat.find('rects')
                    if rects_elem is None:
                        continue
                    rects = []
                    for rect in rects_elem.findall('_'):
                        if not rect.text:
                            continue
                        parts = rect.text.strip().split()
                        if len(parts) >= 5:
                            rects.append(
                                {
                                    'x': int(parts[0]),
                                    'y': int(parts[1]),
                                    'w': int(parts[2]),
                                    'h': int(parts[3]),
                                    'weight': float(parts[4]),
                                }
                            )
                    self.features.append(rects)

            self.stages = []
            stages_elem = cascade.find('.//stages')
            if stages_elem is not None:
                for stage in stages_elem.findall('_'):
                    stage_threshold = stage.find('stageThreshold')
                    threshold = float(stage_threshold.text) if stage_threshold is not None else 0.0

                    classifiers: List[_WeakClassifier] = []
                    wc_elem = stage.find('weakClassifiers')
                    if wc_elem is not None:
                        for wc in wc_elem.findall('_'):
                            internal = wc.find('internalNodes')
                            leaves = wc.find('leafValues')
                            if internal is None or leaves is None or not internal.text or not leaves.text:
                                continue
                            int_parts = internal.text.strip().split()
                            leaf_parts = leaves.text.strip().split()
                            if len(int_parts) >= 4 and len(leaf_parts) >= 2:
                                classifiers.append(
                                    _WeakClassifier(
                                        feature_idx=int(int_parts[2]),
                                        threshold=float(int_parts[3]),
                                        left_val=float(leaf_parts[0]),
                                        right_val=float(leaf_parts[1]),
                                    )
                                )

                    self.stages.append(_Stage(threshold=threshold, classifiers=classifiers))

            self.is_loaded = bool(self.stages and self.features)
        except Exception:
            self.is_loaded = False

    def _integral_image(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float64)
        integral = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
        integral[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
        return integral

    def _rect_sum(self, integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return 0.0
        max_x = integral.shape[1] - 1
        max_y = integral.shape[0] - 1
        if x + w > max_x or y + h > max_y:
            return 0.0
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]

    def _haar_feature(self, integral: np.ndarray, x: int, y: int, scale: float, rects: Sequence[dict]) -> float:
        total = 0.0
        h_img = integral.shape[0] - 1
        w_img = integral.shape[1] - 1

        for r in rects:
            rx = int(x + np.round(r['x'] * scale))
            ry = int(y + np.round(r['y'] * scale))
            rw = max(1, int(np.round(r['w'] * scale)))
            rh = max(1, int(np.round(r['h'] * scale)))
            if rx < 0 or ry < 0 or rx + rw > w_img or ry + rh > h_img:
                continue
            total += self._rect_sum(integral, rx, ry, rw, rh) * float(r['weight'])
        return total

    def _evaluate(self, integral: np.ndarray, sq_integral: np.ndarray, x: int, y: int, scale: float) -> Tuple[bool, float]:
        win_w = max(1, int(np.round(self.window_size[0] * scale)))
        win_h = max(1, int(np.round(self.window_size[1] * scale)))
        area = win_w * win_h
        if area <= 0:
            return False, 0.0

        mean = self._rect_sum(integral, x, y, win_w, win_h) / area
        sq_mean = self._rect_sum(sq_integral, x, y, win_w, win_h) / area
        var = sq_mean - mean * mean
        std = float(np.sqrt(max(var, 0.0)))
        if std < 1e-9:
            std = 1.0
        inv_area = 1.0 / float(area)

        total_score = 0.0
        for stage in self.stages:
            stage_score = 0.0
            for wc in stage.classifiers:
                if wc.feature_idx >= len(self.features):
                    continue
                feat_val = self._haar_feature(integral, x, y, scale, self.features[wc.feature_idx])
                feat_val_norm = feat_val * inv_area
                thresh = wc.threshold * std
                if feat_val_norm < thresh:
                    stage_score += wc.left_val
                else:
                    stage_score += wc.right_val
            if stage_score < stage.threshold:
                return False, 0.0
            total_score += stage_score

        conf = min(1.0, max(0.0, (total_score + 50.0) / 100.0))
        return True, conf

    def _resize_nn(self, image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        old_h, old_w = image.shape[:2]
        x_ratio = old_w / float(new_w)
        y_ratio = old_h / float(new_h)
        x = (np.arange(new_w) * x_ratio).astype(int)
        y = (np.arange(new_h) * y_ratio).astype(int)
        x = np.clip(x, 0, old_w - 1)
        y = np.clip(y, 0, old_h - 1)
        if image.ndim == 3:
            return image[y[:, None], x[None, :], :]
        return image[y[:, None], x[None, :]]

    def _iou(self, a, b) -> float:
        x1, y1, w1, h1 = a
        x2, y2, w2, h2 = b
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return float(inter) / max(float(union), 1.0)

    def _group(self, rects: Sequence[Tuple[float, float, float, float, float]], min_neighbors: int) -> List[Tuple[float, float, float, float, float]]:
        if not rects:
            return []
        rects = list(rects)
        used = [False] * len(rects)
        groups = []
        for i in range(len(rects)):
            if used[i]:
                continue
            grp = [rects[i]]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(rects)):
                    if used[j]:
                        continue
                    for g in grp:
                        if self._iou(g[:4], rects[j][:4]) > 0.2:
                            grp.append(rects[j])
                            used[j] = True
                            changed = True
                            break
            if len(grp) >= int(min_neighbors):
                arr = np.array(grp, dtype=np.float64)
                avg = np.mean(arr, axis=0)
                avg[4] = np.max(arr[:, 4])
                groups.append(tuple(avg))
        return groups

    def _nms(self, boxes: Sequence[Tuple[float, float, float, float, float]]) -> List[Tuple[float, float, float, float, float]]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep: List[Tuple[float, float, float, float, float]] = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best[:4], b[:4]) < float(self.nms_threshold)]
        return keep

    def detect(
        self,
        image: np.ndarray,
        scale_factor: Optional[float] = None,
        min_neighbors: Optional[int] = None,
        min_size: Optional[Tuple[int, int]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Tuple[float, float, float, float, float]]:
        if not self.is_loaded:
            return []

        scale_factor = float(scale_factor) if scale_factor is not None else float(self.scale_factor)
        min_neighbors = int(min_neighbors) if min_neighbors is not None else int(self.min_neighbors)
        min_size = tuple(min_size) if min_size is not None else tuple(self.min_size)

        orig_h, orig_w = image.shape[:2]

        # Downscale for speed
        scale_down = 1.0
        if max(orig_h, orig_w) > int(self.max_detection_size):
            scale_down = float(self.max_detection_size) / float(max(orig_h, orig_w))
            new_w = max(1, int(orig_w * scale_down))
            new_h = max(1, int(orig_h * scale_down))
            image_small = self._resize_nn(image, new_w, new_h)
        else:
            image_small = image
            new_w, new_h = orig_w, orig_h

        gray = self._to_gray_uint8(image_small)

        roi_small = None
        if roi is not None:
            x1, y1, x2, y2 = roi
            x1 = int(np.floor(float(x1) * scale_down))
            y1 = int(np.floor(float(y1) * scale_down))
            x2 = int(np.ceil(float(x2) * scale_down))
            y2 = int(np.ceil(float(y2) * scale_down))
            x1 = max(0, min(x1, new_w))
            y1 = max(0, min(y1, new_h))
            x2 = max(0, min(x2, new_w))
            y2 = max(0, min(y2, new_h))
            if x2 > x1 and y2 > y1:
                roi_small = (x1, y1, x2, y2)

        integral = self._integral_image(gray)
        sq_integral = self._integral_image(gray.astype(np.float64) ** 2)

        img_h, img_w = gray.shape
        min_w, min_h = int(min_size[0]), int(min_size[1])

        candidates: List[Tuple[float, float, float, float, float]] = []
        win_w, win_h = self.window_size

        scale = 1.0
        scales_checked = 0
        while int(win_w * scale) <= img_w and int(win_h * scale) <= img_h:
            scaled_w = max(1, int(np.round(win_w * scale)))
            scaled_h = max(1, int(np.round(win_h * scale)))
            if scaled_w < min_w or scaled_h < min_h:
                scale *= scale_factor
                continue

            scales_checked += 1
            if self.max_scales is not None and scales_checked > int(self.max_scales):
                break

            step = max(2, scaled_w // max(2, int(self.step_divisor)))

            if roi_small is None:
                x_start, y_start = 0, 0
                x_end = img_w - scaled_w
                y_end = img_h - scaled_h
            else:
                rx1, ry1, rx2, ry2 = roi_small
                x_start, y_start = rx1, ry1
                x_end = min(img_w - scaled_w, rx2 - scaled_w)
                y_end = min(img_h - scaled_h, ry2 - scaled_h)

            if x_end < x_start or y_end < y_start:
                scale *= scale_factor
                continue

            for y in range(y_start, y_end + 1, step):
                for x in range(x_start, x_end + 1, step):
                    ok, conf = self._evaluate(integral, sq_integral, x, y, scale)
                    if ok:
                        candidates.append((float(x), float(y), float(scaled_w), float(scaled_h), float(conf)))

            scale *= scale_factor

        grouped = self._group(candidates, min_neighbors=min_neighbors)
        final = self._nms(grouped)

        if scale_down < 1.0:
            scale_up = 1.0 / scale_down
            final = [(x * scale_up, y * scale_up, w * scale_up, h * scale_up, c) for (x, y, w, h, c) in final]

        return final


__all__ = ["HaarCascadeClassifier"]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.