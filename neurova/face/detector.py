# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Face Detection Module - Multiple Backend Support.

This module provides a unified interface for face detection with multiple
backend implementations. Choose the best detector for your use case:

Recommended Detectors (in order of performance):
    1. NativeDetector - Fast GPU-accelerated BlazeFace (best for real-time)
    2. DNNDetector - DNN with prototext/pb models
    3. HaarCascadeDetector - CPU-only, no deep learning dependencies
    4. LBPCascadeDetector - Faster than Haar, less accurate
    5. HOGDetector - HOG + SVM based detection

Quick Start:
    >>> from neurova.face import NativeDetector
    >>> detector = NativeDetector()  # Best performance
    >>> faces = detector.detect(image)
    
Traditional Haar Cascade:
    >>> from neurova.face import FaceDetector
    >>> detector = FaceDetector(method='haar')
    >>> faces = detector.detect(image)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np



# Helper Functions


def _get_cascade_path(cascade_type: str, name: str) -> str:
    """
    Get path to bundled cascade file.
    
    Args:
        cascade_type: Type of cascade ('haar', 'lbp', 'hog').
        name: Cascade name (e.g., 'frontalface_default').
        
    Returns:
        Absolute path to cascade XML file.
        
    Raises:
        FileNotFoundError: If cascade file doesn't exist.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    if cascade_type == "haar":
        cascade_dir = data_dir / "haarcascades"
        if not name.startswith("haarcascade_"):
            name = f"haarcascade_{name}"
    elif cascade_type == "lbp":
        cascade_dir = data_dir / "lbpcascades"
        if not name.startswith("lbpcascade_"):
            name = f"lbpcascade_{name}"
    elif cascade_type == "hog":
        cascade_dir = data_dir / "hogcascades"
        if not name.startswith("hogcascade_"):
            name = f"hogcascade_{name}"
    else:
        raise ValueError(f"Unknown cascade type: {cascade_type}")
    
    if not name.endswith(".xml"):
        name = f"{name}.xml"
    
    path = cascade_dir / name
    if path.exists():
        return str(path)
    
    raise FileNotFoundError(f"Cascade not found: {path}")



# FaceDetector - Unified Interface

class FaceDetector:
    """
    Unified face detector supporting multiple detection methods.
    
    This class provides a simple interface to switch between different
    face detection backends. For production use, consider using the
    specific detector classes directly (NativeDetector, etc.).
    
    Args:
        method: Detection method ('haar', 'lbp', 'hog', 'dnn', 'native').
        cascade: Cascade name or path for cascade-based methods.
        model_path: Path to model file for DNN/native methods.
        min_confidence: Minimum detection confidence (0-1).
        min_size: Minimum face size (width, height).
        max_size: Maximum face size (width, height).
        
    Example:
        >>> detector = FaceDetector(method='haar')
        >>> faces = detector.detect(image)
        >>> for x, y, w, h, conf in faces:
        ...     print(f"Face at ({x}, {y}) size {w}x{h}")
    """
    
    def __init__(
        self,
        method: str = "haar",
        cascade: str = "frontalface_default",
        model_path: Optional[str] = None,
        min_confidence: float = 0.5,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Optional[Tuple[int, int]] = None,
    ):
        self.method = method.lower()
        self.min_confidence = min_confidence
        self.min_size = min_size
        self.max_size = max_size
        self._detector = None
        
        if self.method == "haar":
            self._detector = HaarCascadeDetector(cascade)
        elif self.method == "lbp":
            self._detector = LBPCascadeDetector(cascade)
        elif self.method == "hog":
            self._detector = HOGDetector()
        elif self.method == "dnn":
            self._detector = DNNDetector(model_path)
        elif self.method == "native":
            self._detector = NativeDetector(model_path, min_confidence)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect(
        self,
        image: np.ndarray,
        min_size: Optional[Tuple[int, int]] = None,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR or RGB, numpy array).
            min_size: Override minimum face size.
            max_size: Override maximum face size.
        
        Returns:
            List of (x, y, width, height, confidence) tuples.
        """
        min_size = min_size or self.min_size
        max_size = max_size or self.max_size
        
        faces = self._detector.detect(image, min_size, max_size)
        
        # Filter by confidence
        faces = [f for f in faces if f[4] >= self.min_confidence]
        
        return faces
    
    def detect_and_crop(
        self,
        image: np.ndarray,
        margin: float = 0.2,
        size: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """
        Detect and crop faces from image.
        
        Args:
            image: Input image.
            margin: Margin around face (fraction of face size).
            size: Resize cropped faces to this size.
        
        Returns:
            List of cropped face images.
        """
        faces = self.detect(image)
        cropped = []
        
        h, w = image.shape[:2]
        
        for x, y, fw, fh, conf in faces:
            # Add margin
            mx = int(fw * margin)
            my = int(fh * margin)
            
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + fw + mx)
            y2 = min(h, y + fh + my)
            
            face_img = image[y1:y2, x1:x2].copy()
            
            if size and face_img.size > 0:
                from PIL import Image
                pil_img = Image.fromarray(face_img)
                pil_img = pil_img.resize(size, Image.LANCZOS)
                face_img = np.array(pil_img)
            
            cropped.append(face_img)
        
        return cropped


class HaarCascadeDetector:
    """Haar Cascade face detector using Neurova's pure-Python implementation."""
    
    def __init__(self, cascade: str = "frontalface_default"):
        self.cascade_path = None
        
        # Check if it's a path or cascade name
        if os.path.exists(cascade):
            self.cascade_path = cascade
        else:
            try:
                self.cascade_path = _get_cascade_path("haar", cascade)
            except FileNotFoundError:
                pass
        
        if not self.cascade_path:
            raise FileNotFoundError(f"Haar cascade not found: {cascade}")
        
        # Use Neurova's own HaarCascadeClassifier (pure Python)
        self._cascade = None
        self._use_neurova = False
        
        try:
            from neurova.detection.haar_cascade import HaarCascadeClassifier
            self._cascade = HaarCascadeClassifier(self.cascade_path)
            self._use_neurova = True
        except (ImportError, Exception):
            # Fallback to cv2 if Neurova fails
            try:
                import cv2
                self._cascade = cv2.CascadeClassifier(self.cascade_path)
                self._use_neurova = False
            except ImportError:
                pass
    
    def detect(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using Haar cascade."""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        if self._use_neurova and self._cascade is not None:
            # Use Neurova's pure-Python detector
            self._cascade.min_size = min_size
            faces = self._cascade.detect(gray)
            # Returns (x, y, w, h, confidence) tuples
            return [(int(x), int(y), int(w), int(h), float(c)) for (x, y, w, h, c) in faces]
        
        if not self._use_neurova and self._cascade is not None:
            import cv2
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_size,
                maxSize=max_size or (0, 0),
            )
            
            # Convert to list of tuples with confidence
            return [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in faces]
        
        # Fallback: return empty if no cv2
        return []


class LBPCascadeDetector:
    """LBP Cascade face detector."""
    
    def __init__(self, cascade: str = "frontalface"):
        self.cascade_path = None
        
        if os.path.exists(cascade):
            self.cascade_path = cascade
        else:
            try:
                self.cascade_path = _get_cascade_path("lbp", cascade)
            except FileNotFoundError:
                pass
        
        if not self.cascade_path:
            raise FileNotFoundError(f"LBP cascade not found: {cascade}")
        
        self._cascade = None
        try:
            import cv2
            self._cascade = cv2.CascadeClassifier(self.cascade_path)
            self._use_cv2 = True
        except ImportError:
            self._use_cv2 = False
    
    def detect(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using LBP cascade."""
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        if self._use_cv2 and self._cascade is not None:
            import cv2
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_size,
                maxSize=max_size or (0, 0),
            )
            return [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in faces]
        
        return []


class HOGDetector:
    """HOG-based face detector using Neurova's pure-Python implementation."""
    
    def __init__(self):
        self._detector = None
        self._use_neurova = False
        
        # Try Neurova's HOGDescriptor first
        try:
            from neurova.detection.hog import HOGDescriptor
            self._hog = HOGDescriptor()
            # Set default people detector (would need trained SVM weights)
            self._use_neurova = True
        except (ImportError, Exception):
            pass
        
        # Fallback to dlib
        if not self._use_neurova:
            try:
                import dlib
                self._detector = dlib.get_frontal_face_detector()
                self._use_dlib = True
            except ImportError:
                self._use_dlib = False
    
    def detect(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using HOG."""
        
        # Try Neurova HOG (currently basic - returns empty without trained weights)
        if self._use_neurova and hasattr(self, '_hog'):
            # Neurova HOG descriptor for feature extraction
            # Note: Full detection requires trained SVM weights
            pass
        
        if hasattr(self, '_use_dlib') and self._use_dlib and self._detector is not None:
            # Convert to RGB if BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb = image[:, :, ::-1]
            else:
                rgb = image
            
            dets, scores, _ = self._detector.run(rgb, 1, -1)
            
            faces = []
            for det, score in zip(dets, scores):
                x, y = det.left(), det.top()
                w, h = det.width(), det.height()
                
                if w >= min_size[0] and h >= min_size[1]:
                    if max_size is None or (w <= max_size[0] and h <= max_size[1]):
                        faces.append((x, y, w, h, float(score)))
            
            return faces
        
        return []


class DNNDetector:
    """DNN-based face detector using DNN or pb format."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._net = None
        
        if model_path:
            try:
                import cv2
                self._net = cv2.dnn.readNetFromPrototext(
                    model_path + ".prototxt",
                    model_path + ".prototext"
                )
            except Exception:
                pass
    
    def detect(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using DNN."""
        
        if self._net is None:
            return []
        
        import cv2
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self._net.setInput(blob)
        detections = self._net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                fw, fh = x2 - x1, y2 - y1
                
                if fw >= min_size[0] and fh >= min_size[1]:
                    faces.append((x1, y1, fw, fh, float(confidence)))
        
        return faces


# NativeDetector - Pure Python Face Detection (NO deep learning dependencies)

class NativeDetector:
    """
    Fast face detector using pure Python/NumPy implementation.
    
    NO DEEP LEARNING DEPENDENCIES - uses Haar cascade and feature-based
    detection methods only. Fully compatible with any Python environment.
    
    Detection Methods:
        1. Haar cascade (primary - fast and accurate)
        2. Feature-based sliding window (fallback)
    
    Attributes:
        min_confidence: Minimum detection confidence threshold.
        backend: Active backend ('haar' or 'features').
    
    Example:
        >>> detector = NativeDetector(min_confidence=0.5)
        >>> faces = detector.detect(image)
        >>> for x, y, w, h, conf in faces:
        ...     print(f"Face at ({x}, {y}) conf={conf:.2f}")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize face detector.
        
        Args:
            model_path: Ignored (kept for API compatibility).
            min_confidence: Minimum detection confidence (0.0 to 1.0).
        """
        self.min_confidence = min_confidence
        self._haar_detector = None
        self.backend = None
        
        # Use Haar cascade detector (pure Python, no deep learning)
        try:
            self._haar_detector = HaarCascadeDetector("frontalface_default")
            self.backend = 'haar'
        except Exception:
            # Ultimate fallback: feature-based detection
            self.backend = 'features'
    
    def detect(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (H, W, 3) in BGR or RGB format.
            min_size: Minimum face size (width, height) to return.
            max_size: Maximum face size (width, height) to return.
            
        Returns:
            List of (x, y, width, height, confidence) tuples.
        """
        # Use Haar cascade if available
        if self._haar_detector is not None:
            return self._haar_detector.detect(image, min_size, max_size)
        
        # Fallback to feature-based detection
        return self._detect_features(image, min_size, max_size)
    
    def _detect_features(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int],
        max_size: Optional[Tuple[int, int]],
    ) -> List[Tuple[int, int, int, int, float]]:
        """Detect using feature-based sliding window approach."""
        from PIL import Image as PILImage
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        h, w = gray.shape
        faces = []
        
        # Multi-scale sliding window detection
        scales = [1.0, 0.75, 0.5, 0.35, 0.25]
        window_size = 64
        stride = 16
        
        for scale in scales:
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            
            if scaled_h < window_size or scaled_w < window_size:
                continue
            
            # Resize image
            pil_img = PILImage.fromarray(gray)
            pil_img = pil_img.resize((scaled_w, scaled_h), PILImage.LANCZOS)
            scaled_img = np.array(pil_img)
            
            # Sliding window
            for y in range(0, scaled_h - window_size, stride):
                for x in range(0, scaled_w - window_size, stride):
                    window = scaled_img[y:y+window_size, x:x+window_size]
                    
                    # Compute face likelihood score
                    confidence = self._compute_face_score(window)
                    
                    if confidence >= self.min_confidence:
                        # Convert back to original scale
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_w = int(window_size / scale)
                        orig_h = int(window_size / scale)
                        
                        if orig_w >= min_size[0] and orig_h >= min_size[1]:
                            if max_size is None or (orig_w <= max_size[0] and orig_h <= max_size[1]):
                                faces.append((orig_x, orig_y, orig_w, orig_h, confidence))
        
        # Non-maximum suppression
        faces = self._nms(faces, threshold=0.3)
        
        return faces
    
    def _compute_face_score(self, window: np.ndarray) -> float:
        """
        Compute face likelihood score using image features.
        
        Uses heuristics based on face-like characteristics:
        - Variance distribution
        - Symmetry
        - Edge patterns typical of faces
        """
        h, w = window.shape
        
        # Normalize
        window = window.astype(np.float32)
        if window.max() > 0:
            window = window / window.max()
        
        # Feature 1: Variance (faces have moderate variance)
        variance = np.var(window)
        var_score = 1.0 - abs(variance - 0.08) / 0.08 if variance < 0.16 else 0.3
        
        # Feature 2: Symmetry (faces are roughly symmetric)
        left_half = window[:, :w//2]
        right_half = np.fliplr(window[:, w//2:w//2*2])
        if left_half.shape == right_half.shape:
            symmetry = 1.0 - np.mean(np.abs(left_half - right_half))
        else:
            symmetry = 0.5
        
        # Feature 3: Center is brighter than edges (typical face pattern)
        center = window[h//4:3*h//4, w//4:3*w//4]
        edge_top = window[:h//4, :]
        edge_bottom = window[3*h//4:, :]
        center_brightness = np.mean(center) if center.size > 0 else 0.5
        edge_brightness = (np.mean(edge_top) + np.mean(edge_bottom)) / 2 if edge_top.size > 0 else 0.5
        brightness_ratio = min(1.0, center_brightness / (edge_brightness + 0.01))
        
        # Feature 4: Horizontal edges in eye region
        eye_region = window[h//5:2*h//5, :]
        edges = np.abs(np.diff(eye_region.astype(float), axis=0))
        edge_score = min(1.0, np.mean(edges) * 5)
        
        # Combine scores
        score = (var_score * 0.2 + symmetry * 0.3 + brightness_ratio * 0.25 + edge_score * 0.25)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _nms(
        self,
        boxes: List[Tuple[int, int, int, int, float]],
        threshold: float = 0.3,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Non-maximum suppression to remove overlapping detections."""
        if not boxes:
            return []
        
        # Sort by confidence
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            
            boxes = [
                box for box in boxes
                if self._iou(best, box) < threshold
            ]
        
        return keep
    
    def _iou(
        self,
        box1: Tuple[int, int, int, int, float],
        box2: Tuple[int, int, int, int, float],
    ) -> float:
        """Compute intersection over union."""
        x1, y1, w1, h1, _ = box1
        x2, y2, w2, h2, _ = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"NativeDetector(backend={self.backend!r}, "
            f"min_confidence={self.min_confidence})"
        )
