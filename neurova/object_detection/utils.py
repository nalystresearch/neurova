# Neurova Library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)

"""
Object Detection Utility Functions.

Provides IoU computation, NMS, box conversions, and evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Box Format Conversions

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2).
    
    Args:
        boxes: Array of shape (N, 4) in xywh format
        
    Returns:
        Array of shape (N, 4) in xyxy format
    """
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (x1, y1, x2, y2) to (x_center, y_center, width, height).
    
    Args:
        boxes: Array of shape (N, 4) in xyxy format
        
    Returns:
        Array of shape (N, 4) in xywh format
    """
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    
    return np.stack([x_center, y_center, w, h], axis=1)


def normalize_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
    format: str = "xywh",
) -> np.ndarray:
    """
    Normalize box coordinates to [0, 1] range.
    
    Args:
        boxes: Box coordinates (absolute pixel values)
        image_width: Image width in pixels
        image_height: Image height in pixels
        format: Box format ('xywh' or 'xyxy')
        
    Returns:
        Normalized box coordinates
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    
    boxes = boxes.copy()
    
    if format == "xywh":
        boxes[:, 0] /= image_width   # x_center
        boxes[:, 1] /= image_height  # y_center
        boxes[:, 2] /= image_width   # width
        boxes[:, 3] /= image_height  # height
    elif format == "xyxy":
        boxes[:, 0] /= image_width   # x1
        boxes[:, 1] /= image_height  # y1
        boxes[:, 2] /= image_width   # x2
        boxes[:, 3] /= image_height  # y2
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return boxes


def denormalize_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
    format: str = "xywh",
) -> np.ndarray:
    """
    Denormalize box coordinates from [0, 1] to absolute pixel values.
    
    Args:
        boxes: Normalized box coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        format: Box format ('xywh' or 'xyxy')
        
    Returns:
        Absolute box coordinates
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    
    boxes = boxes.copy()
    
    if format == "xywh":
        boxes[:, 0] *= image_width   # x_center
        boxes[:, 1] *= image_height  # y_center
        boxes[:, 2] *= image_width   # width
        boxes[:, 3] *= image_height  # height
    elif format == "xyxy":
        boxes[:, 0] *= image_width   # x1
        boxes[:, 1] *= image_height  # y1
        boxes[:, 2] *= image_width   # x2
        boxes[:, 3] *= image_height  # y2
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return boxes


def clip_boxes(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """
    Clip boxes to image boundaries (for xyxy format).
    
    Args:
        boxes: Box coordinates in xyxy format
        image_width: Image width
        image_height: Image height
        
    Returns:
        Clipped boxes
    """
    boxes = np.asarray(boxes).copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height)
    return boxes


# IoU Computation

def compute_iou(box1: np.ndarray, box2: np.ndarray, format: str = "xyxy") -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: First box coordinates
        box2: Second box coordinates
        format: Box format ('xywh' or 'xyxy')
        
    Returns:
        IoU value (0-1)
    """
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    
    if format == "xywh":
        box1 = xywh_to_xyxy(box1.reshape(1, -1))[0]
        box2 = xywh_to_xyxy(box2.reshape(1, -1))[0]
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def compute_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray, format: str = "xyxy") -> np.ndarray:
    """
    Compute IoU between all pairs of two sets of boxes.
    
    Args:
        boxes1: First set of boxes (N, 4)
        boxes2: Second set of boxes (M, 4)
        format: Box format ('xywh' or 'xyxy')
        
    Returns:
        IoU matrix of shape (N, M)
    """
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    
    if boxes1.ndim == 1:
        boxes1 = boxes1.reshape(1, -1)
    if boxes2.ndim == 1:
        boxes2 = boxes2.reshape(1, -1)
    
    if format == "xywh":
        boxes1 = xywh_to_xyxy(boxes1)
        boxes2 = xywh_to_xyxy(boxes2)
    
    # Compute intersection
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1[:, None] + area2[None, :] - intersection
    
    iou = np.where(union > 0, intersection / union, 0.0)
    
    return iou


# Non-Maximum Suppression

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_detections: int = 300,
    format: str = "xyxy",
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping boxes.
    
    Args:
        boxes: Box coordinates (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum score to keep
        max_detections: Maximum number of detections to keep
        format: Box format ('xywh' or 'xyxy')
        
    Returns:
        Indices of boxes to keep
    """
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    if format == "xywh":
        boxes = xywh_to_xyxy(boxes)
    
    # Filter by score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    original_indices = np.where(mask)[0]
    
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Sort by score (descending)
    order = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(order) > 0 and len(keep) < max_detections:
        idx = order[0]
        keep.append(idx)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        iou = compute_iou_batch(boxes[idx:idx+1], boxes[order[1:]])[0]
        
        # Keep boxes with IoU below threshold
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    
    keep = np.array(keep, dtype=np.int32)
    return original_indices[keep]


def nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_detections: int = 300,
) -> np.ndarray:
    """
    Apply NMS separately for each class.
    
    Args:
        boxes: Box coordinates (N, 4) in xyxy format
        scores: Confidence scores (N,)
        class_ids: Class IDs (N,)
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum score to keep
        max_detections: Maximum number of detections to keep
        
    Returns:
        Indices of boxes to keep
    """
    unique_classes = np.unique(class_ids)
    keep_indices = []
    
    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        
        cls_keep = non_max_suppression(
            cls_boxes,
            cls_scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        
        keep_indices.extend(cls_indices[cls_keep])
    
    keep_indices = np.array(keep_indices, dtype=np.int32)
    
    # Sort by score and limit total detections
    if len(keep_indices) > max_detections:
        keep_scores = scores[keep_indices]
        top_k = np.argsort(keep_scores)[::-1][:max_detections]
        keep_indices = keep_indices[top_k]
    
    return keep_indices


# Evaluation Metrics

def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
    use_07_metric: bool = False,
) -> float:
    """
    Compute Average Precision (AP) from recall-precision curve.
    
    Args:
        recalls: Recall values at each threshold
        precisions: Precision values at each threshold
        use_07_metric: Use VOC 2007 11-point interpolation
        
    Returns:
        Average Precision value
    """
    recalls = np.asarray(recalls)
    precisions = np.asarray(precisions)
    
    # Add sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    if use_07_metric:
        # VOC 2007 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if np.any(mask):
                ap += np.max(precisions[mask])
        ap /= 11
    else:
        # VOC 2010+ area under curve
        i = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    
    return ap


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = None,
) -> Dict[str, float]:
    """
    Compute Mean Average Precision (mAP) across all classes.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'class_ids', 'image_id'
        ground_truths: List of dicts with 'boxes', 'class_ids', 'image_id'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
        
    Returns:
        Dict with 'mAP' and per-class APs
    """
    if num_classes is None:
        all_classes = set()
        for gt in ground_truths:
            all_classes.update(gt.get('class_ids', []))
        num_classes = max(all_classes) + 1 if all_classes else 0
    
    # Organize ground truths by image and class
    gt_by_image: Dict[Any, Dict[int, List]] = {}
    for gt in ground_truths:
        img_id = gt['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = {}
        
        boxes = np.asarray(gt.get('boxes', []))
        class_ids = np.asarray(gt.get('class_ids', []))
        
        for i, cls_id in enumerate(class_ids):
            cls_id = int(cls_id)
            if cls_id not in gt_by_image[img_id]:
                gt_by_image[img_id][cls_id] = {'boxes': [], 'matched': []}
            gt_by_image[img_id][cls_id]['boxes'].append(boxes[i])
            gt_by_image[img_id][cls_id]['matched'].append(False)
    
    # Collect predictions per class
    preds_by_class: Dict[int, List] = {i: [] for i in range(num_classes)}
    
    for pred in predictions:
        img_id = pred['image_id']
        boxes = np.asarray(pred.get('boxes', []))
        scores = np.asarray(pred.get('scores', []))
        class_ids = np.asarray(pred.get('class_ids', []))
        
        for i, cls_id in enumerate(class_ids):
            cls_id = int(cls_id)
            if cls_id < num_classes:
                preds_by_class[cls_id].append({
                    'box': boxes[i],
                    'score': scores[i],
                    'image_id': img_id,
                })
    
    # Compute AP for each class
    aps = {}
    
    for cls_id in range(num_classes):
        preds = preds_by_class[cls_id]
        
        if len(preds) == 0:
            aps[cls_id] = 0.0
            continue
        
        # Sort predictions by score
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        # Count total ground truths for this class
        n_gt = sum(
            len(gt_by_image.get(img_id, {}).get(cls_id, {}).get('boxes', []))
            for img_id in gt_by_image
        )
        
        if n_gt == 0:
            aps[cls_id] = 0.0
            continue
        
        # Match predictions to ground truths
        for i, pred in enumerate(preds):
            img_id = pred['image_id']
            
            if img_id not in gt_by_image or cls_id not in gt_by_image[img_id]:
                fp[i] = 1
                continue
            
            gt_data = gt_by_image[img_id][cls_id]
            gt_boxes = gt_data['boxes']
            matched = gt_data['matched']
            
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            
            # Find best matching ground truth
            best_iou = 0
            best_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if not matched[j]:
                    iou = compute_iou(pred['box'], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
            
            if best_iou >= iou_threshold and best_idx >= 0:
                tp[i] = 1
                matched[best_idx] = True
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        aps[cls_id] = compute_ap(recalls, precisions)
    
    # Compute mAP
    valid_aps = [ap for ap in aps.values() if ap > 0 or True]
    mAP = np.mean(valid_aps) if valid_aps else 0.0
    
    return {
        'mAP': mAP,
        'per_class_ap': aps,
    }


# Drawing Utilities

def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
    format: str = "xyxy",
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.
    
    Args:
        image: Input image (H, W, C) in RGB or BGR
        boxes: Box coordinates (N, 4)
        class_ids: Class IDs (N,)
        scores: Confidence scores (N,)
        class_names: List of class names
        colors: List of colors per class (R, G, B)
        thickness: Box line thickness
        font_scale: Font scale for labels
        format: Box format ('xywh' or 'xyxy')
        
    Returns:
        Image with drawn detections
    """
    image = image.copy()
    boxes = np.asarray(boxes)
    
    if len(boxes) == 0:
        return image
    
    if format == "xywh":
        boxes = xywh_to_xyxy(boxes)
    
    # Generate default colors
    if colors is None:
        np.random.seed(42)
        n_colors = max(20, len(np.unique(class_ids)) if class_ids is not None else 1)
        colors = [(int(r), int(g), int(b)) for r, g, b in 
                  np.random.randint(0, 255, size=(n_colors, 3))]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        cls_id = int(class_ids[i]) if class_ids is not None else 0
        color = colors[cls_id % len(colors)]
        
        # Draw box
        image[y1:y1+thickness, x1:x2] = color
        image[y2-thickness:y2, x1:x2] = color
        image[y1:y2, x1:x1+thickness] = color
        image[y1:y2, x2-thickness:x2] = color
        
        # Create label
        label = ""
        if class_names is not None and cls_id < len(class_names):
            label = class_names[cls_id]
        else:
            label = f"class_{cls_id}"
        
        if scores is not None:
            label += f" {scores[i]:.2f}"
        
        # Draw label background
        label_h = int(15 * font_scale * 2)
        label_w = len(label) * int(8 * font_scale * 2)
        
        ly1 = max(0, y1 - label_h)
        ly2 = y1
        lx1 = x1
        lx2 = min(image.shape[1], x1 + label_w)
        
        image[ly1:ly2, lx1:lx2] = color
    
    return image


# Augmentation for Detection

def augment_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    rotation_range: float = 0.0,
    scale_range: Tuple[float, float] = (1.0, 1.0),
    brightness_range: Tuple[float, float] = (1.0, 1.0),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply augmentation to image and bounding boxes.
    
    Args:
        image: Input image (H, W, C)
        boxes: Box coordinates (N, 4) in xywh normalized format
        class_ids: Class IDs (N,)
        horizontal_flip: Apply random horizontal flip
        vertical_flip: Apply random vertical flip
        rotation_range: Maximum rotation angle in degrees
        scale_range: (min_scale, max_scale) for random scaling
        brightness_range: (min, max) brightness multiplier
        seed: Random seed
        
    Returns:
        Tuple of (augmented_image, augmented_boxes, class_ids)
    """
    rng = np.random.default_rng(seed)
    
    image = image.copy().astype(np.float32)
    boxes = boxes.copy()
    h, w = image.shape[:2]
    
    # Horizontal flip
    if horizontal_flip and rng.random() > 0.5:
        image = image[:, ::-1].copy()
        boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip x_center
    
    # Vertical flip
    if vertical_flip and rng.random() > 0.5:
        image = image[::-1, :].copy()
        boxes[:, 1] = 1.0 - boxes[:, 1]  # Flip y_center
    
    # Brightness adjustment
    if brightness_range != (1.0, 1.0):
        brightness = rng.uniform(brightness_range[0], brightness_range[1])
        image = np.clip(image * brightness, 0, 255)
    
    # Scale augmentation
    if scale_range != (1.0, 1.0):
        scale = rng.uniform(scale_range[0], scale_range[1])
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Simple resize using slicing
        if scale > 1.0:
            # Crop to original size
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            # Boxes don't need adjustment for center crop
        else:
            # Pad to original size
            pass  # Keep boxes as is
    
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image, boxes, class_ids

# Neurova Library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)
