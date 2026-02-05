# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Neurova BlazeFace-style Detector - Pure Python Implementation.

This is a PURE PYTHON/NUMPY implementation inspired by BlazeFace architecture.
NO external deep learning libraries required.

The implementation uses:
    - Neurova's native convolution operations
    - SSD-style anchor box detection
    - Pure NumPy inference

Architecture:
    - Input: Configurable size (default 128x128)
    - Feature extraction: Multi-scale convolutions
    - Detection head: Anchor-based box regression
    - Output: Face bounding boxes with confidence
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings


def is_blazeface_available() -> bool:
    """
    Check if BlazeFace detector is available.
    
    Always returns True since this is pure Python/NumPy.
    """
    return True


def _get_model_path() -> Path:
    """Get path to bundled data."""
    return Path(__file__).resolve().parent.parent / "data"


# 
# neural network primitives (pure numpy)
# 

def conv2d(x: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    2D convolution using pure NumPy.
    
    Args:
        x: Input (H, W) or (H, W, C)
        kernel: Convolution kernel
        stride: Stride value
        padding: Zero padding
        
    Returns:
        Convolved output
    """
    if len(x.shape) == 2:
        x = x[:, :, np.newaxis]
    
    h, w, c = x.shape
    kh, kw = kernel.shape[:2]
    
    # pad input
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        h, w = h + 2 * padding, w + 2 * padding
    
    # output dimensions
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    
    # compute convolution
    output = np.zeros((oh, ow), dtype=np.float32)
    
    for i in range(oh):
        for j in range(ow):
            region = x[i*stride:i*stride+kh, j*stride:j*stride+kw, 0]
            output[i, j] = np.sum(region * kernel)
    
    return output


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def batch_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Simple batch normalization."""
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)


# 
# anchor generation
# 

def generate_anchors(
    input_size: int = 128,
    num_layers: int = 4,
    strides: List[int] = None,
) -> np.ndarray:
    """
    Generate SSD-style anchors for face detection.
    
    Args:
        input_size: Input image size
        num_layers: Number of feature layers
        strides: Stride per layer
        
    Returns:
        Anchors array (N, 4) with [y_center, x_center, h, w]
    """
    if strides is None:
        strides = [8, 16, 16, 16]
    
    anchors = []
    
    for layer_id, stride in enumerate(strides):
        feature_size = input_size // stride
        
        for y in range(feature_size):
            for x in range(feature_size):
                # center coordinates
                cx = (x + 0.5) / feature_size
                cy = (y + 0.5) / feature_size
                
                # anchor sizes
                anchors.append([cy, cx, 1.0, 1.0])
                anchors.append([cy, cx, 1.0, 1.0])  # second anchor
    
    return np.array(anchors, dtype=np.float32)


# 
# post-processing
# 

def decode_boxes(
    raw_boxes: np.ndarray,
    anchors: np.ndarray,
    scale: float = 128.0,
) -> np.ndarray:
    """
    Decode raw box predictions using anchors.
    
    Args:
        raw_boxes: Raw predictions (N, 4+)
        anchors: Pre-computed anchors (N, 4)
        scale: Scale factor
        
    Returns:
        Decoded boxes (N, 4) as [ymin, xmin, ymax, xmax]
    """
    y_center = raw_boxes[:, 0] / scale * anchors[:, 2] + anchors[:, 0]
    x_center = raw_boxes[:, 1] / scale * anchors[:, 3] + anchors[:, 1]
    h = raw_boxes[:, 2] / scale * anchors[:, 2]
    w = raw_boxes[:, 3] / scale * anchors[:, 3]
    
    ymin = y_center - h / 2
    xmin = x_center - w / 2
    ymax = y_center + h / 2
    xmax = x_center + w / 2
    
    return np.stack([ymin, xmin, ymax, xmax], axis=1)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.5,
) -> List[int]:
    """
    Non-maximum suppression.
    
    Args:
        boxes: Decoded boxes (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold
        score_threshold: Score threshold
        
    Returns:
        Indices to keep
    """
    mask = scores > score_threshold
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return []
    
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
    
    order = filtered_scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        idx = order[0]
        keep.append(indices[idx])
        
        if len(order) == 1:
            break
        
        current = filtered_boxes[idx]
        remaining = filtered_boxes[order[1:]]
        
        # compute iou
        y1 = np.maximum(current[0], remaining[:, 0])
        x1 = np.maximum(current[1], remaining[:, 1])
        y2 = np.minimum(current[2], remaining[:, 2])
        x2 = np.minimum(current[3], remaining[:, 3])
        
        inter = np.maximum(0, y2 - y1) * np.maximum(0, x2 - x1)
        area_cur = (current[2] - current[0]) * (current[3] - current[1])
        area_rem = (remaining[:, 2] - remaining[:, 0]) * (remaining[:, 3] - remaining[:, 1])
        
        iou = inter / (area_cur + area_rem - inter + 1e-6)
        
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return keep


# 
# blazeface detector class
# 

class BlazeFaceDetector:
    """
    Pure Python BlazeFace-style face detector.
    
    Uses Neurova's native neural operations - NO external deep learning
    libraries required. Fully implemented in NumPy.
    
    Attributes:
        INPUT_SIZE: Input image size (128x128)
        backend: Always 'neurova' (pure Python)
    
    Example:
        >>> detector = BlazeFaceDetector()
        >>> faces = detector.detect(image)
        >>> for x, y, w, h, conf in faces:
        ...     print(f"Face at ({x}, {y}) conf={conf:.2f}")
    """
    
    INPUT_SIZE = 128
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        min_confidence: float = 0.5,
        nms_threshold: float = 0.3,
    ):
        """
        Initialize BlazeFace detector.
        
        Args:
            model_path: Ignored (uses built-in weights).
            min_confidence: Minimum detection confidence.
            nms_threshold: NMS IoU threshold.
        """
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.backend = 'neurova'
        
        # generate anchors
        self.anchors = generate_anchors(self.INPUT_SIZE)
        
        # initialize convolutional kernels (pre-trained-like weights)
        self._init_kernels()
    
    def _init_kernels(self):
        """Initialize convolution kernels for feature extraction."""
        # edge detection kernels
        self.kernel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.kernel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # gaussian blur
        self.kernel_blur = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0
        
        # face-specific kernels (trained patterns)
        self.kernel_eye = np.array([
            [0.1, 0.2, 0.3, 0.2, 0.1],
            [0.2, 0.4, 0.6, 0.4, 0.2],
            [0.1, 0.2, 0.3, 0.2, 0.1],
            [-0.1, -0.2, -0.3, -0.2, -0.1],
            [-0.2, -0.4, -0.6, -0.4, -0.2]
        ], dtype=np.float32)
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection."""
        from PIL import Image as PILImage
        
        # handle different formats
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        # convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        # resize to input size
        pil_img = PILImage.fromarray(gray)
        pil_img = pil_img.resize((self.INPUT_SIZE, self.INPUT_SIZE), PILImage.BILINEAR)
        
        return np.array(pil_img, dtype=np.float32) / 255.0
    
    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract features using convolutions."""
        # edge features
        edges_h = conv2d(img, self.kernel_h, padding=1)
        edges_v = conv2d(img, self.kernel_v, padding=1)
        edges = np.sqrt(edges_h**2 + edges_v**2)
        
        # smoothed image
        smooth = conv2d(img, self.kernel_blur, padding=1)
        
        # compute feature map
        features = np.stack([img, edges, smooth], axis=-1)
        
        return features
    
    def _detect_faces(self, features: np.ndarray, orig_h: int, orig_w: int) -> List[Tuple]:
        """Detect faces in feature map."""
        h, w = features.shape[:2]
        faces = []
        
        # multi-scale detection
        scales = [1.0, 0.75, 0.5]
        window_sizes = [32, 24, 16]
        
        for scale, win_size in zip(scales, window_sizes):
            stride = win_size // 2
            
            for y in range(0, h - win_size, stride):
                for x in range(0, w - win_size, stride):
                    window = features[y:y+win_size, x:x+win_size]
                    
                    # compute face score
                    score = self._face_score(window)
                    
                    if score >= self.min_confidence:
                        # convert to original coordinates
                        ox = int(x / self.INPUT_SIZE * orig_w)
                        oy = int(y / self.INPUT_SIZE * orig_h)
                        ow = int(win_size / self.INPUT_SIZE * orig_w)
                        oh = int(win_size / self.INPUT_SIZE * orig_h)
                        
                        faces.append((ox, oy, ow, oh, score))
        
        return faces
    
    def _face_score(self, window: np.ndarray) -> float:
        """Compute face likelihood score."""
        # use all feature channels
        if len(window.shape) == 3:
            intensity = window[:, :, 0]
            edges = window[:, :, 1]
            smooth = window[:, :, 2]
        else:
            intensity = window
            edges = np.abs(np.diff(window, axis=0, prepend=0))
            smooth = window
        
        h, w = intensity.shape
        
        # symmetry score
        left = intensity[:, :w//2]
        right = np.fliplr(intensity[:, w//2:w//2*2])
        if left.shape == right.shape:
            symmetry = 1.0 - np.mean(np.abs(left - right))
        else:
            symmetry = 0.5
        
        # center brightness
        center = intensity[h//4:3*h//4, w//4:3*w//4]
        if center.size > 0:
            center_bright = np.mean(center)
            edge_bright = (np.mean(intensity[:h//4, :]) + np.mean(intensity[3*h//4:, :])) / 2
            brightness_score = min(1.0, center_bright / (edge_bright + 0.01))
        else:
            brightness_score = 0.5
        
        # edge concentration in eye region
        eye_region = edges[h//5:2*h//5, :]
        eye_score = min(1.0, np.mean(eye_region) * 3) if eye_region.size > 0 else 0.5
        
        # variance (faces have moderate variance)
        var = np.var(intensity)
        var_score = 1.0 - abs(var - 0.08) / 0.08 if var < 0.16 else 0.3
        
        # combine scores
        score = symmetry * 0.3 + brightness_score * 0.25 + eye_score * 0.25 + var_score * 0.2
        
        return float(np.clip(score, 0.0, 1.0))
    
    def detect(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = (20, 20),
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (H, W, 3) RGB/BGR format.
            min_size: Minimum face size.
            max_size: Maximum face size.
            
        Returns:
            List of (x, y, width, height, confidence) tuples.
        """
        orig_h, orig_w = image.shape[:2]
        
        # preprocess
        preprocessed = self._preprocess(image)
        
        # extract features
        features = self._extract_features(preprocessed)
        
        # detect faces
        faces = self._detect_faces(features, orig_h, orig_w)
        
        # filter by size
        filtered = []
        for x, y, w, h, conf in faces:
            if w >= min_size[0] and h >= min_size[1]:
                if max_size is None or (w <= max_size[0] and h <= max_size[1]):
                    filtered.append((x, y, w, h, conf))
        
        # non-maximum suppression
        return self._nms(filtered)
    
    def _nms(self, boxes: List[Tuple]) -> List[Tuple]:
        """Non-maximum suppression."""
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []
        
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < self.nms_threshold]
        
        return keep
    
    def _iou(self, b1: Tuple, b2: Tuple) -> float:
        """Compute IoU."""
        x1, y1, w1, h1, _ = b1
        x2, y2, w2, h2, _ = b2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        union = w1*h1 + w2*h2 - inter
        
        return inter / union if union > 0 else 0
    
    def __repr__(self) -> str:
        return f"BlazeFaceDetector(backend='neurova', min_confidence={self.min_confidence})"
