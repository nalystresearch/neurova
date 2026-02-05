# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Object detection algorithms for Neurova."""

from __future__ import annotations
import numpy as np
from typing import Optional, Union
from neurova.core.errors import ValidationError, DetectionError
from neurova.core.array_ops import ensure_2d, ensure_3d
from neurova.filters.convolution import convolve2d


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    method: str = "ncc",
) -> np.ndarray:
    """Template matching using various correlation methods.
    
    Args:
        image: Input image (grayscale or color)
        template: Template to match (same number of channels as image)
        method: Matching method:
            - "ncc": Normalized cross-correlation (default, range [-1, 1])
            - "ssd": Sum of squared differences (lower is better)
            - "sad": Sum of absolute differences (lower is better)
            - "ccorr": Cross-correlation (unnormalized)
            
    Returns:
        Response map (same size as image), values depend on method
    """
    if method not in ("ncc", "ssd", "sad", "ccorr"):
        raise ValidationError("method", method, "ncc, ssd, sad, or ccorr")
    
    # ensure same dimensions
    if image.ndim == 2:
        image = ensure_2d(image).astype(np.float64)
        template = ensure_2d(template).astype(np.float64)
    elif image.ndim == 3:
        image = ensure_3d(image).astype(np.float64)
        template = ensure_3d(template).astype(np.float64)
        if image.shape[2] != template.shape[2]:
            raise ValidationError(
                "template",
                f"channels={template.shape[2]}",
                f"channels={image.shape[2]}"
            )
    else:
        raise ValidationError("image", f"ndim={image.ndim}", "ndim=2 or 3")
    
    th, tw = template.shape[:2]
    ih, iw = image.shape[:2]
    
    if th > ih or tw > iw:
        raise ValidationError(
            "template",
            f"shape={template.shape}",
            f"smaller than image {image.shape}"
        )
    
    # pad image for convolution
    pad_h = th // 2
    pad_w = tw // 2
    
    if method == "ncc":
        # normalized cross-correlation
        result = _match_template_ncc(image, template)
    elif method == "ssd":
        # sum of squared differences
        result = _match_template_ssd(image, template)
    elif method == "sad":
        # sum of absolute differences
        result = _match_template_sad(image, template)
    else:  # ccorr
        # cross-correlation
        result = _match_template_ccorr(image, template)
    
    return result


def _match_template_ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation."""
    th, tw = template.shape[:2]
    ih, iw = image.shape[:2]
    
    # normalize template
    template_mean = template.mean()
    template_std = template.std()
    if template_std == 0:
        template_std = 1.0
    template_norm = (template - template_mean) / template_std
    
    # result map
    result = np.zeros((ih, iw), dtype=np.float64)
    
    # slide template over image
    for i in range(ih - th + 1):
        for j in range(iw - tw + 1):
            patch = image[i:i+th, j:j+tw]
            patch_mean = patch.mean()
            patch_std = patch.std()
            if patch_std == 0:
                ncc = 0.0
            else:
                patch_norm = (patch - patch_mean) / patch_std
                ncc = (template_norm * patch_norm).mean()
            result[i + th//2, j + tw//2] = ncc
    
    return result


def _match_template_ssd(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sum of squared differences."""
    th, tw = template.shape[:2]
    ih, iw = image.shape[:2]
    result = np.full((ih, iw), np.inf, dtype=np.float64)
    
    for i in range(ih - th + 1):
        for j in range(iw - tw + 1):
            patch = image[i:i+th, j:j+tw]
            ssd = ((patch - template) ** 2).sum()
            result[i + th//2, j + tw//2] = ssd
    
    return result


def _match_template_sad(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sum of absolute differences."""
    th, tw = template.shape[:2]
    ih, iw = image.shape[:2]
    result = np.full((ih, iw), np.inf, dtype=np.float64)
    
    for i in range(ih - th + 1):
        for j in range(iw - tw + 1):
            patch = image[i:i+th, j:j+tw]
            sad = np.abs(patch - template).sum()
            result[i + th//2, j + tw//2] = sad
    
    return result


def _match_template_ccorr(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cross-correlation."""
    th, tw = template.shape[:2]
    ih, iw = image.shape[:2]
    result = np.zeros((ih, iw), dtype=np.float64)
    
    for i in range(ih - th + 1):
        for j in range(iw - tw + 1):
            patch = image[i:i+th, j:j+tw]
            ccorr = (patch * template).sum()
            result[i + th//2, j + tw//2] = ccorr
    
    return result


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Non-maximum suppression for bounding boxes.
    
    Args:
        boxes: Bounding boxes (Nx4 array of [x1, y1, x2, y2])
        scores: Confidence scores (N array)
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of kept boxes
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    boxes = boxes.astype(np.float64)
    scores = scores.astype(np.float64)
    
    # get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # compute areas
    areas = (x2 - x1) * (y2 - y1)
    
    # sort by scores
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-10)
        
        # keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)


def sliding_window_detection(
    image: np.ndarray,
    window_size: tuple[int, int],
    stride: int = 8,
    scales: Optional[list[float]] = None,
) -> list[tuple[int, int, int, int, float]]:
    """Sliding window detection framework.
    
    This is a helper function that generates sliding window proposals.
    Actual detection requires a classifier function.
    
    Args:
        image: Input image
        window_size: (height, width) of sliding window
        stride: Step size for sliding window
        scales: List of scale factors for multi-scale detection
        
    Returns:
        List of window proposals as (x1, y1, x2, y2, scale) tuples
    """
    if scales is None:
        scales = [1.0]
    
    window_h, window_w = window_size
    proposals = []
    
    for scale in scales:
        # resize image
        if scale != 1.0:
            from neurova.transform import resize
            scaled_h = int(image.shape[0] * scale)
            scaled_w = int(image.shape[1] * scale)
            scaled_image = resize(image, (scaled_h, scaled_w))
        else:
            scaled_image = image
        
        h, w = scaled_image.shape[:2]
        
        # slide window
        for y in range(0, h - window_h + 1, stride):
            for x in range(0, w - window_w + 1, stride):
                # convert back to original image coordinates
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + window_w) / scale)
                y2 = int((y + window_h) / scale)
                proposals.append((x1, y1, x2, y2, scale))
    
    return proposals


class TemplateDetector:
    """Template-based object detector.
    
    Examples:
        detector = TemplateDetector(template, threshold=0.8)
        boxes, scores = detector.detect(image)
    """
    
    def __init__(
        self,
        template: np.ndarray,
        threshold: float = 0.7,
        method: str = "ncc",
        nms_threshold: float = 0.3,
    ):
        """Initialize template detector.
        
        Args:
            template: Template image
            threshold: Detection threshold
            method: Matching method (see match_template)
            nms_threshold: NMS IoU threshold
        """
        self.template = template.astype(np.float64)
        self.threshold = threshold
        self.method = method
        self.nms_threshold = nms_threshold
        self.template_h, self.template_w = template.shape[:2]
    
    def detect(
        self,
        image: np.ndarray,
        scales: Optional[list[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect template in image.
        
        Args:
            image: Input image
            scales: Optional list of scales for multi-scale detection
            
        Returns:
            Tuple of (boxes, scores) where:
            - boxes: Nx4 array of [x1, y1, x2, y2]
            - scores: N array of confidence scores
        """
        if scales is None:
            scales = [1.0]
        
        all_boxes = []
        all_scores = []
        
        for scale in scales:
            # resize template
            if scale != 1.0:
                from neurova.transform import resize
                scaled_h = int(self.template_h * scale)
                scaled_w = int(self.template_w * scale)
                scaled_template = resize(self.template, (scaled_h, scaled_w))
            else:
                scaled_template = self.template
            
            # match template
            response = match_template(image, scaled_template, method=self.method)
            
            # find peaks
            if self.method in ("ncc", "ccorr"):
                # higher is better
                mask = response >= self.threshold
            else:
                # lower is better (SSD, SAD)
                mask = response <= self.threshold
            
            coords = np.column_stack(np.where(mask))
            if len(coords) == 0:
                continue
            
            # convert to boxes
            th, tw = scaled_template.shape[:2]
            for r, c in coords:
                x1 = c - tw // 2
                y1 = r - th // 2
                x2 = x1 + tw
                y2 = y1 + th
                score = response[r, c]
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
        
        if len(all_boxes) == 0:
            return np.array([]).reshape(0, 4), np.array([])
        
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        
        # apply NMS
        keep = non_max_suppression(boxes, scores, self.nms_threshold)
        return boxes[keep], scores[keep]


__all__ = [
    "match_template",
    "non_max_suppression",
    "sliding_window_detection",
    "TemplateDetector",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.