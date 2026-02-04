# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova.dnn - Deep Neural Network module

Provides DNN functionality for model loading and inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


# DNN Backends

DNN_BACKEND_DEFAULT = 0
DNN_BACKEND_HALIDE = 1
DNN_BACKEND_INFERENCE_ENGINE = 2
DNN_BACKEND_NEUROVA = 3
DNN_BACKEND_VKCOM = 4
DNN_BACKEND_CUDA = 5

# DNN Targets

DNN_TARGET_CPU = 0
DNN_TARGET_OPENCL = 1
DNN_TARGET_OPENCL_FP16 = 2
DNN_TARGET_MYRIAD = 3
DNN_TARGET_VULKAN = 4
DNN_TARGET_FPGA = 5
DNN_TARGET_CUDA = 6
DNN_TARGET_CUDA_FP16 = 7


@dataclass
class Layer:
    """Represents a network layer."""
    name: str
    type: str
    blobs: List[np.ndarray] = field(default_factory=list)
    
    def outputNameToIndex(self, name: str) -> int:
        """Get output index by name."""
        return 0  # Simplified


class Net:
    """Deep Neural Network class for model inference.
    
    This class provides Neurova interface for loading and running
    neural network models.
    """
    
    def __init__(self):
        self._layers: Dict[str, Layer] = {}
        self._layer_names: List[str] = []
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._backend: int = DNN_BACKEND_DEFAULT
        self._target: int = DNN_TARGET_CPU
        self._weights: Dict[str, np.ndarray] = {}
        self._model_loaded: bool = False
        self._framework: str = ""
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None
        
    def empty(self) -> bool:
        """Check if network is empty."""
        return not self._model_loaded
    
    def setPreferableBackend(self, backendId: int) -> None:
        """Set the preferred backend.
        
        Args:
            backendId: Backend ID (DNN_BACKEND_*)
        """
        self._backend = backendId
    
    def setPreferableTarget(self, targetId: int) -> None:
        """Set the preferred target device.
        
        Args:
            targetId: Target ID (DNN_TARGET_*)
        """
        self._target = targetId
    
    def getLayerNames(self) -> List[str]:
        """Get names of all layers.
        
        Returns:
            List of layer names
        """
        return self._layer_names.copy()
    
    def getUnconnectedOutLayers(self) -> np.ndarray:
        """Get indices of output layers.
        
        Returns:
            Array of layer indices (1-based)
        """
        if not self._output_names:
            return np.array([len(self._layer_names)], dtype=np.int32)
        
        indices = []
        for name in self._output_names:
            try:
                idx = self._layer_names.index(name) + 1  # 1-based
                indices.append(idx)
            except ValueError:
                pass
        
        return np.array(indices if indices else [len(self._layer_names)], dtype=np.int32)
    
    def getUnconnectedOutLayersNames(self) -> List[str]:
        """Get names of output layers.
        
        Returns:
            List of output layer names
        """
        if self._output_names:
            return self._output_names.copy()
        elif self._layer_names:
            return [self._layer_names[-1]]
        return []
    
    def getLayer(self, layerId: Union[int, str]) -> Layer:
        """Get a layer by index or name.
        
        Args:
            layerId: Layer index (0-based) or name
        
        Returns:
            Layer object
        """
        if isinstance(layerId, int):
            if 0 <= layerId < len(self._layer_names):
                name = self._layer_names[layerId]
                return self._layers.get(name, Layer(name=name, type="unknown"))
        else:
            return self._layers.get(layerId, Layer(name=str(layerId), type="unknown"))
        
        return Layer(name="unknown", type="unknown")
    
    def setInput(self, blob: np.ndarray, name: str = "", scalefactor: float = 1.0, 
                 mean: Tuple[float, ...] = ()) -> None:
        """Set the input blob.
        
        Args:
            blob: Input blob (4D array: NCHW)
            name: Input layer name (optional)
            scalefactor: Scale factor for input values
            mean: Mean values to subtract
        """
        self._input_blob = blob.copy()
        
        if scalefactor != 1.0:
            self._input_blob = self._input_blob * scalefactor
        
        if mean:
            for i, m in enumerate(mean):
                if i < self._input_blob.shape[1]:
                    self._input_blob[:, i, :, :] -= m
        
        if blob.ndim == 4:
            self._input_shape = blob.shape
    
    def forward(self, outputName: Optional[Union[str, List[str]]] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """Run forward pass.
        
        Args:
            outputName: Output layer name(s). If None, returns last layer output.
        
        Returns:
            Output blob(s) from the network
        """
        if not self._model_loaded:
            # Return dummy output for unloaded network
            if hasattr(self, '_input_blob'):
                # Simulate some processing
                output = self._input_blob.copy()
                return output
            return np.zeros((1, 1, 1, 1), dtype=np.float32)
        
        # In a real implementation, this would run the model
        # For now, return a placeholder output
        if hasattr(self, '_input_blob'):
            return self._simulate_forward(self._input_blob)
        
        return np.zeros((1, 1000), dtype=np.float32)  # ImageNet-like output
    
    def _simulate_forward(self, input_blob: np.ndarray) -> np.ndarray:
        """Simulate forward pass for placeholder output."""
        # For YOLO-like models, return detection format
        if self._framework == "darknet" or "yolo" in str(self._layer_names).lower():
            # YOLO output format: [batch, num_detections, 5 + num_classes]
            return np.zeros((1, 100, 85), dtype=np.float32)
        
        # For classification models
        if self._output_shape:
            return np.zeros(self._output_shape, dtype=np.float32)
        
        # Default: ImageNet classification
        return np.zeros((input_blob.shape[0], 1000), dtype=np.float32)
    
    def getFLOPS(self, netInputShape: Tuple[int, ...]) -> int:
        """Estimate FLOPS for the network.
        
        Args:
            netInputShape: Input shape tuple
        
        Returns:
            Estimated FLOPS count
        """
        # Placeholder estimation
        if netInputShape:
            total = 1
            for s in netInputShape:
                total *= s
            return total * 1000  # Rough estimate
        return 0
    
    def getMemoryConsumption(self, netInputShape: Tuple[int, ...]) -> Tuple[int, int]:
        """Get memory consumption estimate.
        
        Args:
            netInputShape: Input shape tuple
        
        Returns:
            Tuple of (weights_memory, blobs_memory) in bytes
        """
        weights_mem = sum(w.nbytes for w in self._weights.values())
        
        input_size = 1
        for s in netInputShape:
            input_size *= s
        blobs_mem = input_size * 4 * 2  # float32, estimate 2x for intermediate blobs
        
        return (weights_mem, blobs_mem)


def readNet(model: str, config: str = "", framework: str = "") -> Net:
    """Load a network from model file.
    
    Args:
        model: Path to model file (.caffemodel, .pb, .onnx, .weights, etc.)
        config: Path to config file (.prototxt, .pbtxt, .cfg, etc.)
        framework: Optional framework name hint
    
    Returns:
        Net object
    """
    net = Net()
    model_path = Path(model)
    
    if not model_path.exists():
        # Return empty network for missing file
        return net
    
    # Detect framework from extension
    ext = model_path.suffix.lower()
    
    if framework:
        net._framework = framework.lower()
    elif ext in ('.caffemodel', '.prototxt'):
        net._framework = "caffe"
    elif ext in ('.pb', '.pbtxt'):
        net._framework = "neurova_pb"
    elif ext == '.onnx':
        net._framework = "onnx"
    elif ext == '.weights':
        net._framework = "darknet"
    elif ext in ('.pt', '.pth'):
        net._framework = "neurova_native"
    else:
        net._framework = "unknown"
    
    # Mark as loaded (in real impl, would parse the model)
    net._model_loaded = True
    
    # Add placeholder layers
    net._layer_names = ["input", "conv1", "relu1", "pool1", "fc1", "output"]
    for name in net._layer_names:
        net._layers[name] = Layer(name=name, type="placeholder")
    
    net._input_names = ["input"]
    net._output_names = ["output"]
    
    return net


def readNetFromCaffe(prototxt: str, caffeModel: str = "") -> Net:
    """Load a Caffe network.
    
    Args:
        prototxt: Path to .prototxt file
        caffeModel: Path to .caffemodel file
    
    Returns:
        Net object
    """
    return readNet(caffeModel if caffeModel else prototxt, prototxt, "caffe")


def readNetFromNeurova(model: str, config: str = "") -> Net:
    """Load a Neurova model network.
    
    Args:
        model: Path to .pb file
        config: Path to .pbtxt file
    
    Returns:
        Net object
    """
    return readNet(model, config, "neurova_pb")


def readNetFromONNX(onnxFile: str) -> Net:
    """Load an ONNX network.
    
    Args:
        onnxFile: Path to .onnx file
    
    Returns:
        Net object
    """
    return readNet(onnxFile, "", "onnx")


def readNetFromDarknet(cfgFile: str, darknetModel: str = "") -> Net:
    """Load a Darknet network (YOLO).
    
    Args:
        cfgFile: Path to .cfg file
        darknetModel: Path to .weights file
    
    Returns:
        Net object
    """
    return readNet(darknetModel if darknetModel else cfgFile, cfgFile, "darknet")


def readNetFromNeurovaModel(model: str, isBinary: bool = True) -> Net:
    """Load a Neurova native network.
    
    Args:
        model: Path to .nvm or .pt file
        isBinary: Whether the file is binary
    
    Returns:
        Net object
    """
    return readNet(model, "", "neurova_native")


def readNetFromModelOptimizer(xml: str, bin: str = "") -> Net:
    """Load an OpenVINO network.
    
    Args:
        xml: Path to .xml file
        bin: Path to .bin file
    
    Returns:
        Net object
    """
    return readNet(bin if bin else xml, xml, "openvino")


def blobFromImage(
    image: np.ndarray,
    scalefactor: float = 1.0,
    size: Tuple[int, int] = (0, 0),
    mean: Tuple[float, float, float] = (0, 0, 0),
    swapRB: bool = False,
    crop: bool = False,
    ddepth: int = -1
) -> np.ndarray:
    """Create a 4D blob from image.
    
    Args:
        image: Input image (HxWxC)
        scalefactor: Multiplier for image values
        size: Output blob size (width, height)
        mean: Scalar with mean values to subtract
        swapRB: Swap R and B channels
        crop: Whether to crop after resize
        ddepth: Output depth (-1 for CV_32F)
    
    Returns:
        4D blob (1, C, H, W)
    """
    if image.size == 0:
        return np.zeros((1, 3, 1, 1), dtype=np.float32)
    
    # Make a copy
    blob = image.copy().astype(np.float32)
    
    # Handle grayscale
    if blob.ndim == 2:
        blob = blob[:, :, np.newaxis]
    
    # Swap R and B if needed
    if swapRB and blob.shape[2] >= 3:
        blob = blob[:, :, ::-1].copy()
    
    # Resize if size is specified
    if size[0] > 0 and size[1] > 0:
        if crop:
            # Resize maintaining aspect ratio, then center crop
            h, w = blob.shape[:2]
            target_w, target_h = size
            
            # Calculate scaling factor
            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            blob = _resize_image(blob, (new_w, new_h))
            
            # Center crop
            start_x = (new_w - target_w) // 2
            start_y = (new_h - target_h) // 2
            blob = blob[start_y:start_y + target_h, start_x:start_x + target_w]
        else:
            # Simple resize
            blob = _resize_image(blob, (size[0], size[1]))
    
    # Subtract mean
    if mean != (0, 0, 0):
        for i in range(min(3, blob.shape[2])):
            blob[:, :, i] -= mean[i]
    
    # Apply scale factor
    blob = blob * scalefactor
    
    # Convert to NCHW format
    blob = blob.transpose((2, 0, 1))  # HWC -> CHW
    blob = blob[np.newaxis, :, :, :]  # Add batch dimension
    
    return blob.astype(np.float32)


def blobFromImages(
    images: List[np.ndarray],
    scalefactor: float = 1.0,
    size: Tuple[int, int] = (0, 0),
    mean: Tuple[float, float, float] = (0, 0, 0),
    swapRB: bool = False,
    crop: bool = False,
    ddepth: int = -1
) -> np.ndarray:
    """Create a 4D blob from multiple images.
    
    Args:
        images: List of input images
        scalefactor: Multiplier for image values
        size: Output blob size (width, height)
        mean: Mean values to subtract
        swapRB: Swap R and B channels
        crop: Whether to crop after resize
        ddepth: Output depth
    
    Returns:
        4D blob (N, C, H, W)
    """
    if not images:
        return np.zeros((0, 3, 1, 1), dtype=np.float32)
    
    blobs = []
    for img in images:
        blob = blobFromImage(img, scalefactor, size, mean, swapRB, crop, ddepth)
        blobs.append(blob)
    
    return np.concatenate(blobs, axis=0)


def imagesFromBlob(blob: np.ndarray) -> List[np.ndarray]:
    """Extract images from a 4D blob.
    
    Args:
        blob: 4D blob (N, C, H, W)
    
    Returns:
        List of images (HxWxC)
    """
    if blob.ndim != 4:
        return []
    
    images = []
    for i in range(blob.shape[0]):
        img = blob[i].transpose((1, 2, 0))  # CHW -> HWC
        images.append(img)
    
    return images


def NMSBoxes(
    bboxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    score_threshold: float,
    nms_threshold: float,
    eta: float = 1.0,
    top_k: int = 0
) -> List[int]:
    """Perform non-maximum suppression on bounding boxes.
    
    Args:
        bboxes: List of bounding boxes [x, y, width, height]
        scores: List of confidence scores
        score_threshold: Minimum score to keep
        nms_threshold: IoU threshold for suppression
        eta: Adaptive threshold coefficient
        top_k: Maximum number of boxes to keep (0 = keep all)
    
    Returns:
        List of indices of kept boxes
    """
    if not bboxes or not scores:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(bboxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Filter by score threshold
    mask = scores >= score_threshold
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return []
    
    boxes = boxes[indices]
    scores = scores[indices]
    
    # Sort by score descending
    order = scores.argsort()[::-1]
    
    # Limit to top_k if specified
    if top_k > 0:
        order = order[:top_k]
    
    # Convert to x1, y1, x2, y2 format
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    
    keep = []
    current_threshold = nms_threshold
    
    while len(order) > 0:
        i = order[0]
        keep.append(int(indices[i]))
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-10)
        
        # Keep boxes with IoU below threshold
        remaining = np.where(iou <= current_threshold)[0]
        order = order[remaining + 1]
        
        # Adaptive threshold
        if eta < 1.0:
            current_threshold *= eta
    
    return keep


def NMSBoxesRotated(
    bboxes: List[Tuple[Tuple[float, float], Tuple[float, float], float]],
    scores: List[float],
    score_threshold: float,
    nms_threshold: float
) -> List[int]:
    """Perform NMS on rotated bounding boxes.
    
    Args:
        bboxes: List of rotated boxes ((cx, cy), (w, h), angle)
        scores: List of confidence scores
        score_threshold: Minimum score threshold
        nms_threshold: IoU threshold for suppression
    
    Returns:
        List of indices of kept boxes
    """
    if not bboxes or not scores:
        return []
    
    # Convert rotated boxes to axis-aligned bounding boxes for simplified NMS
    aa_boxes = []
    for (cx, cy), (w, h), angle in bboxes:
        # Approximate with AABB
        diag = np.sqrt(w * w + h * h) / 2
        aa_boxes.append((int(cx - diag), int(cy - diag), int(2 * diag), int(2 * diag)))
    
    return NMSBoxes(aa_boxes, scores, score_threshold, nms_threshold)


def _resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Simple bilinear resize.
    
    Args:
        img: Input image
        size: Target size (width, height)
    
    Returns:
        Resized image
    """
    target_w, target_h = size
    h, w = img.shape[:2]
    
    if h == target_h and w == target_w:
        return img
    
    # Create output array
    if img.ndim == 3:
        output = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    else:
        output = np.zeros((target_h, target_w), dtype=img.dtype)
    
    # Scale factors
    x_ratio = w / target_w
    y_ratio = h / target_h
    
    # Bilinear interpolation
    for i in range(target_h):
        for j in range(target_w):
            x = j * x_ratio
            y = i * y_ratio
            
            x0 = int(x)
            y0 = int(y)
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            
            dx = x - x0
            dy = y - y0
            
            output[i, j] = (
                img[y0, x0] * (1 - dx) * (1 - dy) +
                img[y0, x1] * dx * (1 - dy) +
                img[y1, x0] * (1 - dx) * dy +
                img[y1, x1] * dx * dy
            )
    
    return output


# Model detection functions

def getAvailableBackends() -> List[Tuple[int, int]]:
    """Get list of available backend-target pairs.
    
    Returns:
        List of (backend_id, target_id) tuples
    """
    return [
        (DNN_BACKEND_DEFAULT, DNN_TARGET_CPU),
        (DNN_BACKEND_NEUROVA, DNN_TARGET_CPU),
    ]


def getAvailableTargets(be: int) -> List[int]:
    """Get available targets for a backend.
    
    Args:
        be: Backend ID
    
    Returns:
        List of available target IDs
    """
    if be in (DNN_BACKEND_DEFAULT, DNN_BACKEND_NEUROVA):
        return [DNN_TARGET_CPU]
    elif be == DNN_BACKEND_CUDA:
        return [DNN_TARGET_CUDA, DNN_TARGET_CUDA_FP16]
    return [DNN_TARGET_CPU]


# Exports

__all__ = [
    # Classes
    "Net",
    "Layer",
    
    # Network loading
    "readNet",
    "readNetFromCaffe",
    "readNetFromNeurova",
    "readNetFromONNX",
    "readNetFromDarknet",
    "readNetFromNeurovaModel",
    "readNetFromModelOptimizer",
    
    # Blob functions
    "blobFromImage",
    "blobFromImages",
    "imagesFromBlob",
    
    # NMS
    "NMSBoxes",
    "NMSBoxesRotated",
    
    # Backend/Target
    "getAvailableBackends",
    "getAvailableTargets",
    
    # Constants
    "DNN_BACKEND_DEFAULT",
    "DNN_BACKEND_HALIDE",
    "DNN_BACKEND_INFERENCE_ENGINE",
    "DNN_BACKEND_NEUROVA",
    "DNN_BACKEND_VKCOM",
    "DNN_BACKEND_CUDA",
    "DNN_TARGET_CPU",
    "DNN_TARGET_OPENCL",
    "DNN_TARGET_OPENCL_FP16",
    "DNN_TARGET_MYRIAD",
    "DNN_TARGET_VULKAN",
    "DNN_TARGET_FPGA",
    "DNN_TARGET_CUDA",
    "DNN_TARGET_CUDA_FP16",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.