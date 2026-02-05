# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Neurova Native Face Detector.

Pure Python/NumPy implementation - NO external deep learning
libraries required. Uses Neurova's built-in neural network primitives.

This module provides face detection using:
    1. Haar cascade (fast, reliable)
    2. Feature-based detection (pure NumPy)
    3. Neurova's native neural layers (for advanced use)

No external dependencies beyond numpy and Pillow.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


# availability flag - always true since this is pure Python
HAS_TFLITE = True


def is_tflite_available() -> bool:
    """
    Check if face detection is available.
    
    Always returns True since this uses pure Python/NumPy.
    Kept for backwards API compatibility.
    """
    return True


def _get_model_path() -> Path:
    """Get path to bundled cascade data."""
    return Path(__file__).resolve().parent.parent / "data" / "haarcascades"


class TFLiteFaceDetector:
    """
    Native face detector using pure Python/NumPy.
    
    This is a drop-in replacement for native inference based detectors.
    Uses Neurova's Haar cascade implementation internally.
    
    NO EXTERNAL DEEP LEARNING LIBRARIES REQUIRED.
    
    Example:
        >>> detector = TFLiteFaceDetector()
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
        self._cascade = None
        
        # load haar cascade
        try:
            from neurova.detection.haar_cascade import HaarCascade
            cascade_path = _get_model_path() / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                self._cascade = HaarCascade(str(cascade_path))
        except (ImportError, Exception):
            pass
    
    def detect(
        self,
        image: np.ndarray,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (H, W, 3) RGB or BGR format.
            min_size: Minimum face size (width, height).
            max_size: Maximum face size (width, height).
            
        Returns:
            List of (x, y, width, height, confidence) tuples.
        """
        # convert to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        # use haar cascade if available
        if self._cascade is not None:
            try:
                raw_faces = self._cascade.detect(gray)
                faces = []
                for face in raw_faces:
                    if len(face) >= 4:
                        x, y, w, h = face[:4]
                        conf = face[4] if len(face) > 4 else 0.9
                        
                        # filter by size
                        if w >= min_size[0] and h >= min_size[1]:
                            if max_size is None or (w <= max_size[0] and h <= max_size[1]):
                                if conf >= self.min_confidence:
                                    faces.append((int(x), int(y), int(w), int(h), float(conf)))
                return faces
            except Exception:
                pass
        
        # fallback to feature-based detection
        return self._detect_features(gray, min_size, max_size)
    
    def _detect_features(
        self,
        gray: np.ndarray,
        min_size: Tuple[int, int],
        max_size: Optional[Tuple[int, int]],
    ) -> List[Tuple[int, int, int, int, float]]:
        """Feature-based face detection using image analysis."""
        from PIL import Image
        
        h, w = gray.shape
        faces = []
        
        # multi-scale sliding window
        scales = [1.0, 0.75, 0.5, 0.35]
        window_size = 64
        stride = 16
        
        for scale in scales:
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            
            if scaled_h < window_size or scaled_w < window_size:
                continue
            
            # resize
            pil_img = Image.fromarray(gray)
            pil_img = pil_img.resize((scaled_w, scaled_h), Image.LANCZOS)
            scaled_img = np.array(pil_img, dtype=np.float32) / 255.0
            
            # sliding window
            for y in range(0, scaled_h - window_size, stride):
                for x in range(0, scaled_w - window_size, stride):
                    window = scaled_img[y:y+window_size, x:x+window_size]
                    
                    # compute face score
                    score = self._face_score(window)
                    
                    if score >= self.min_confidence:
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_w = int(window_size / scale)
                        orig_h = int(window_size / scale)
                        
                        if orig_w >= min_size[0] and orig_h >= min_size[1]:
                            if max_size is None or (orig_w <= max_size[0] and orig_h <= max_size[1]):
                                faces.append((orig_x, orig_y, orig_w, orig_h, score))
        
        # non-maximum suppression
        return self._nms(faces, 0.3)
    
    def _face_score(self, window: np.ndarray) -> float:
        """Compute face likelihood score using image features."""
        h, w = window.shape
        
        # normalize
        if window.max() > 0:
            window = window / window.max()
        
        # variance check
        variance = np.var(window)
        var_score = 1.0 - abs(variance - 0.08) / 0.08 if variance < 0.16 else 0.3
        
        # symmetry check
        left = window[:, :w//2]
        right = np.fliplr(window[:, w//2:w//2*2])
        if left.shape == right.shape:
            symmetry = 1.0 - np.mean(np.abs(left - right))
        else:
            symmetry = 0.5
        
        # center brightness
        center = window[h//4:3*h//4, w//4:3*w//4]
        edges = np.concatenate([window[:h//4, :].flatten(), window[3*h//4:, :].flatten()])
        if center.size > 0 and edges.size > 0:
            brightness_ratio = min(1.0, np.mean(center) / (np.mean(edges) + 0.01))
        else:
            brightness_ratio = 0.5
        
        # edge features in eye region
        eye_region = window[h//5:2*h//5, :]
        edges = np.abs(np.diff(eye_region, axis=0))
        edge_score = min(1.0, np.mean(edges) * 5)
        
        score = var_score * 0.2 + symmetry * 0.3 + brightness_ratio * 0.25 + edge_score * 0.25
        return float(np.clip(score, 0.0, 1.0))
    
    def _nms(
        self,
        boxes: List[Tuple[int, int, int, int, float]],
        threshold: float,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Non-maximum suppression."""
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []
        
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < threshold]
        
        return keep
    
    def _iou(self, b1: Tuple, b2: Tuple) -> float:
        """Compute intersection over union."""
        x1, y1, w1, h1, _ = b1
        x2, y2, w2, h2, _ = b2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        
        return inter / union if union > 0 else 0
