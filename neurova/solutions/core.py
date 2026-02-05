# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova inference engine - core abstractions.

this module provides the foundational building blocks for neural inference
pipelines. uses a plugin-based architecture for runtime flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class InferenceBackend(Enum):
    """supported inference backends for model execution."""
    LITERT = auto()      # ai-edge-litert (preferred)
    TFLITE = auto()      # tflite-runtime
    FALLBACK = auto()  # fallback backend
    NONE = auto()        # no backend available


@dataclass(frozen=True, slots=True)
class Point3D:
    """
    immutable 3d point with confidence metrics.
    
    uses slots for memory efficiency when tracking many landmarks.
    frozen for hashability in sets/dicts.
    """
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0
    visible: bool = True
    
    def scaled(self, sx: float, sy: float, sz: float = 1.0) -> 'Point3D':
        """return new point scaled by factors."""
        return Point3D(
            x=self.x * sx,
            y=self.y * sy,
            z=self.z * sz,
            confidence=self.confidence,
            visible=self.visible,
        )
    
    def offset(self, dx: float, dy: float, dz: float = 0.0) -> 'Point3D':
        """return new point offset by deltas."""
        return Point3D(
            x=self.x + dx,
            y=self.y + dy,
            z=self.z + dz,
            confidence=self.confidence,
            visible=self.visible,
        )
    
    def distance_to(self, other: 'Point3D') -> float:
        """euclidean distance to another point."""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def as_array(self) -> np.ndarray:
        """return as numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass(slots=True)
class BoundingBox:
    """
    axis-aligned bounding box with normalized coordinates.
    
    coordinates are in [0, 1] range relative to image dimensions.
    """
    x: float       # left edge
    y: float       # top edge
    width: float   # box width
    height: float  # box height
    score: float = 0.0
    class_id: int = 0
    anchors: List[Point3D] = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[float, float]:
        """return center point (cx, cy)."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> float:
        """return box area."""
        return self.width * self.height
    
    def to_pixels(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """convert to pixel coordinates (x, y, w, h)."""
        return (
            int(self.x * img_w),
            int(self.y * img_h),
            int(self.width * img_w),
            int(self.height * img_h),
        )
    
    def to_xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """convert to corner format (x1, y1, x2, y2)."""
        x1 = int(self.x * img_w)
        y1 = int(self.y * img_h)
        x2 = int((self.x + self.width) * img_w)
        y2 = int((self.y + self.height) * img_h)
        return (x1, y1, x2, y2)
    
    def iou(self, other: 'BoundingBox') -> float:
        """compute intersection over union with another box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class InferenceOutput:
    """
    container for inference pipeline results.
    
    flexible structure to hold various output types from different models.
    """
    boxes: List[BoundingBox] = field(default_factory=list)
    keypoints: List[List[Point3D]] = field(default_factory=list)
    masks: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    raw_tensors: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # metadata
    frame_width: int = 0
    frame_height: int = 0
    inference_time_ms: float = 0.0
    backend: InferenceBackend = InferenceBackend.NONE
    
    def __bool__(self) -> bool:
        """true if any detections found."""
        return len(self.boxes) > 0 or len(self.keypoints) > 0
    
    def keypoints_as_array(self, idx: int = 0) -> np.ndarray:
        """get keypoints at index as (n, 3) array."""
        if idx >= len(self.keypoints):
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([p.as_array() for p in self.keypoints[idx]])


# legacy compatibility - map old names to new classes
class Landmark(Point3D):
    """legacy alias for point3d."""
    pass


class NormalizedLandmark(Point3D):
    """legacy alias for point3d with normalized coordinates."""
    
    @property
    def visibility(self) -> float:
        """legacy property."""
        return self.confidence
    
    @property
    def presence(self) -> float:
        """legacy property."""
        return 1.0 if self.visible else 0.0


class Detection(BoundingBox):
    """legacy alias for boundingbox."""
    
    @property
    def xmin(self) -> float:
        return self.x
    
    @property
    def ymin(self) -> float:
        return self.y
    
    @property
    def label(self) -> int:
        return self.class_id
    
    @property
    def keypoints(self) -> List[Point3D]:
        return self.anchors


class SolutionResult(InferenceOutput):
    """legacy alias for inferenceoutput."""
    
    @property
    def detections(self) -> List[BoundingBox]:
        return self.boxes
    
    @property
    def landmarks(self) -> List[List[Point3D]]:
        return self.keypoints
    
    @property
    def segmentation_mask(self) -> Optional[np.ndarray]:
        return self.masks
    
    @property
    def image_width(self) -> int:
        return self.frame_width
    
    @property
    def image_height(self) -> int:
        return self.frame_height


class RuntimeLoader:
    """
    lazy loader for inference runtimes.
    
    NOTE: The solutions module requires a TFLite runtime for neural network
    inference. This is OPTIONAL - the core neurova library works without it.
    
    If you need FaceMesh, Hands, Pose, etc., install one of:
        pip install ai-edge-litert  (recommended, ~13MB)
    """
    
    _interpreter_factory: Optional[Callable] = None
    _backend: InferenceBackend = InferenceBackend.NONE
    
    @classmethod
    def get_interpreter(cls, model_path: str) -> Tuple[Any, InferenceBackend]:
        """
        load interpreter for given model file.
        
        returns (interpreter, backend_type) tuple.
        raises importerror if no backend available.
        
        NOTE: This requires ai-edge-litert for neural network inference.
        The core neurova face detection works WITHOUT this using Haar cascade.
        """
        # try ai-edge-litert (lightweight runtime ~13MB)
        try:
            from ai_edge_litert import interpreter as litert
            interp = litert.Interpreter(model_path=model_path)
            return interp, InferenceBackend.LITERT
        except ImportError:
            pass
        
        raise ImportError(
            "neurova.solutions requires ai-edge-litert for neural inference.\n"
            "install with: pip install ai-edge-litert\n\n"
            "NOTE: Core neurova face detection works without this!\n"
            "use: from neurova.face import FaceDetector\n"
            "     detector = FaceDetector(method='haar')"
        )


class NeuralPipeline(ABC):
    """
    abstract base for neural inference pipelines.
    
    provides common infrastructure for model loading, preprocessing,
    inference, and postprocessing. subclasses implement specific
    detection/landmark algorithms.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        smoothing_factor: float = 0.0,
    ):
        """
        initialize pipeline.
        
        args:
            model_path: path to tflite model, uses bundled if none
            confidence_threshold: minimum confidence for detections
            smoothing_factor: temporal smoothing (0=none, 1=max)
        """
        self.model_path = model_path
        self.min_detection_confidence = confidence_threshold
        self.min_tracking_confidence = confidence_threshold
        self.smoothing_factor = smoothing_factor
        
        # runtime state
        self._interpreter = None
        self._backend = InferenceBackend.NONE
        self.input_details = None
        self.output_details = None
        self._initialized = False
        
        # optional smoothing state
        self._prev_output: Optional[InferenceOutput] = None
    
    @property
    def backend(self) -> str:
        """return current backend name."""
        return self._backend.name.lower()
    
    @property
    def interpreter(self):
        """legacy property for compatibility."""
        return self._interpreter
    
    @abstractmethod
    def _get_default_model_path(self) -> Path:
        """return path to the bundled model file."""
        pass
    
    @abstractmethod
    def _get_model_url(self) -> str:
        """return url for downloading model if not bundled."""
        pass
    
    def _load_interpreter(self, path: Path) -> bool:
        """load inference runtime for model."""
        try:
            self._interpreter, self._backend = RuntimeLoader.get_interpreter(str(path))
            self._interpreter.allocate_tensors()
            self.input_details = self._interpreter.get_input_details()
            self.output_details = self._interpreter.get_output_details()
            return True
        except Exception:
            self._interpreter = None
            self._backend = InferenceBackend.NONE
            return False
    
    def initialize(self) -> bool:
        """
        prepare pipeline for inference.
        
        loads model and allocates resources. call before process().
        returns true if ready.
        """
        if self._initialized:
            return True
        
        # determine model path
        if self.model_path:
            path = Path(self.model_path)
        else:
            path = self._get_default_model_path()
        
        # download if missing
        if not path.exists():
            try:
                from neurova.solutions.model_manager import download_model
                model_name = path.stem
                download_model(model_name)
            except Exception:
                return False
        
        # load runtime
        if not self._load_interpreter(path):
            return False
        
        self._initialized = True
        return True
    
    def _invoke(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """run inference on input tensor, return output tensors."""
        if not self._initialized:
            raise RuntimeError("pipeline not initialized, call initialize() first")
        
        # set input
        self._interpreter.set_tensor(
            self.input_details[0]['index'],
            input_tensor
        )
        
        # run inference
        self._interpreter.invoke()
        
        # collect outputs
        outputs = []
        for detail in self.output_details:
            outputs.append(self._interpreter.get_tensor(detail['index']))
        
        return outputs
    
    @abstractmethod
    def process(self, frame: np.ndarray) -> InferenceOutput:
        """
        run inference on input frame.
        
        args:
            frame: input image as (h, w, 3) rgb uint8 array
            
        returns:
            inference output with detections/keypoints
        """
        pass
    
    def close(self):
        """release resources."""
        self._interpreter = None
        self._initialized = False
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, *args):
        self.close()


# legacy alias
Solution = NeuralPipeline


# 
# utility functions
# 

def sigmoid(x: np.ndarray) -> np.ndarray:
    """apply sigmoid with numerical stability."""
    x_safe = np.clip(x, -88, 88)  # prevent overflow
    return 1.0 / (1.0 + np.exp(-x_safe))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """apply softmax normalization."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def nms_boxes(
    boxes: List[BoundingBox],
    iou_threshold: float = 0.5,
) -> List[BoundingBox]:
    """apply non-maximum suppression to boxes."""
    if not boxes:
        return []
    
    # sort by score descending
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    
    keep = []
    while sorted_boxes:
        best = sorted_boxes.pop(0)
        keep.append(best)
        
        # remove boxes with high iou
        sorted_boxes = [
            b for b in sorted_boxes
            if best.iou(b) < iou_threshold
        ]
    
    return keep


def smooth_keypoints(
    current: List[Point3D],
    previous: Optional[List[Point3D]],
    alpha: float = 0.5,
) -> List[Point3D]:
    """apply exponential smoothing to keypoints."""
    if previous is None or alpha <= 0:
        return current
    
    if len(current) != len(previous):
        return current
    
    smoothed = []
    for curr, prev in zip(current, previous):
        smoothed.append(Point3D(
            x=alpha * curr.x + (1 - alpha) * prev.x,
            y=alpha * curr.y + (1 - alpha) * prev.y,
            z=alpha * curr.z + (1 - alpha) * prev.z,
            confidence=curr.confidence,
            visible=curr.visible,
        ))
    
    return smoothed


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
