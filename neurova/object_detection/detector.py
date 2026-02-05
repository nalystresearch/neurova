# Neurova Library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)

"""
Object Detector - High-level detection API.

Provides a simple interface for object detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Detection:
    """Single detection result.
    
    Attributes:
        x1, y1, x2, y2: Bounding box coordinates (pixels)
        confidence: Detection confidence score
        class_id: Class index
        class_name: Class name (if available)
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""
    
    @property
    def box(self) -> Tuple[float, float, float, float]:
        """Get box as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    @property
    def box_xywh(self) -> Tuple[float, float, float, float]:
        """Get box as (x_center, y_center, width, height) tuple."""
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return (self.x1 + w/2, self.y1 + h/2, w, h)
    
    @property
    def area(self) -> float:
        """Get box area in pixels."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def __repr__(self) -> str:
        name = self.class_name or f"class_{self.class_id}"
        return f"Detection({name}, conf={self.confidence:.2f}, box=[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}])"


@dataclass
class DetectionResult:
    """Detection results for a single image.
    
    Attributes:
        detections: List of Detection objects
        image_size: Original image size (height, width)
        inference_time: Inference time in milliseconds
    """
    detections: List[Detection]
    image_size: Tuple[int, int]
    inference_time: float = 0.0
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def __iter__(self):
        return iter(self.detections)
    
    def __getitem__(self, idx: int) -> Detection:
        return self.detections[idx]
    
    @property
    def boxes(self) -> np.ndarray:
        """Get all boxes as (N, 4) array in xyxy format."""
        if not self.detections:
            return np.array([]).reshape(0, 4)
        return np.array([d.box for d in self.detections])
    
    @property
    def confidences(self) -> np.ndarray:
        """Get all confidence scores."""
        return np.array([d.confidence for d in self.detections])
    
    @property
    def class_ids(self) -> np.ndarray:
        """Get all class IDs."""
        return np.array([d.class_id for d in self.detections])
    
    @property
    def class_names(self) -> List[str]:
        """Get all class names."""
        return [d.class_name for d in self.detections]
    
    def filter_by_class(self, class_id: Optional[int] = None, class_name: Optional[str] = None) -> "DetectionResult":
        """Filter detections by class."""
        if class_id is not None:
            filtered = [d for d in self.detections if d.class_id == class_id]
        elif class_name is not None:
            filtered = [d for d in self.detections if d.class_name == class_name]
        else:
            filtered = self.detections
        return DetectionResult(filtered, self.image_size, self.inference_time)
    
    def filter_by_confidence(self, min_conf: float) -> "DetectionResult":
        """Filter detections by minimum confidence."""
        filtered = [d for d in self.detections if d.confidence >= min_conf]
        return DetectionResult(filtered, self.image_size, self.inference_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'boxes': self.boxes.tolist(),
            'confidences': self.confidences.tolist(),
            'class_ids': self.class_ids.tolist(),
            'class_names': self.class_names,
            'image_size': self.image_size,
            'inference_time': self.inference_time,
        }


class ObjectDetector:
    """
    High-level object detector with training and inference.
    
    Provides a simple interface for object detection.
    
    Args:
        num_classes: Number of object classes
        class_names: List of class names (optional)
        model_size: Model size ('nano', 'small', 'medium', 'large', 'xlarge')
        weights: Path to pretrained weights (optional)
        device: Device to use ('cpu', 'cuda')
        
    Example:
        >>> # Create detector
        >>> detector = ObjectDetector(num_classes=80, model_size='small')
        >>> 
        >>> # Load pretrained weights
        >>> detector.load('detector_weights.npz')
        >>> 
        >>> # Detect objects
        >>> results = detector.detect(image)
        >>> for det in results:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")
        >>> 
        >>> # Train on custom dataset
        >>> detector.train(
        ...     data_dir='./my_dataset',
        ...     epochs=100,
        ... )
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        class_names: Optional[List[str]] = None,
        model_size: str = "small",
        weights: Optional[str] = None,
        device: str = "cpu",
    ):
        from neurova.object_detection.model import DetectionModel
        
        self.num_classes = num_classes
        self.model_size = model_size
        self.device = device
        
        # Set class names
        if class_names is None:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            self.class_names = class_names
            self.num_classes = len(class_names)
        
        # Create model
        self.model = DetectionModel(
            num_classes=self.num_classes,
            model_size=model_size,
        )
        
        # Load weights if provided
        if weights:
            self.load(weights)
        
        # Default settings
        self.img_size = (640, 640)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
    
    def detect(
        self,
        image: Union[np.ndarray, str, Path, List],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Detect objects in image(s).
        
        Args:
            image: Input image(s) - numpy array, file path, or list
            conf_threshold: Confidence threshold (default: 0.25)
            iou_threshold: NMS IoU threshold (default: 0.45)
            classes: Filter to specific class IDs (optional)
            
        Returns:
            DetectionResult or list of DetectionResult
        """
        import time
        
        conf_threshold = conf_threshold or self.conf_threshold
        iou_threshold = iou_threshold or self.iou_threshold
        
        # Handle different input types
        if isinstance(image, (str, Path)):
            images = [self._load_image(image)]
            single = True
        elif isinstance(image, list):
            images = [self._load_image(img) if isinstance(img, (str, Path)) else img for img in image]
            single = False
        else:
            images = [image]
            single = True
        
        results = []
        
        for img in images:
            start_time = time.time()
            
            # Preprocess
            orig_h, orig_w = img.shape[:2]
            processed, scale, pad = self._preprocess(img)
            
            # Run inference
            from neurova.neural.tensor import tensor
            x = tensor(processed, requires_grad=False)
            outputs = self.model(x)
            
            # Decode predictions
            preds = self.model.decode_predictions(
                outputs,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                img_size=self.img_size,
            )[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            # Convert to Detection objects
            detections = []
            boxes = preds['boxes']
            confidences = preds['confidences']
            class_ids = preds['class_ids']
            
            for i in range(len(boxes)):
                # Denormalize to original image size
                x1 = boxes[i, 0] * orig_w
                y1 = boxes[i, 1] * orig_h
                x2 = boxes[i, 2] * orig_w
                y2 = boxes[i, 3] * orig_h
                
                cls_id = int(class_ids[i])
                
                # Filter by class if specified
                if classes is not None and cls_id not in classes:
                    continue
                
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(confidences[i]),
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                ))
            
            results.append(DetectionResult(
                detections=detections,
                image_size=(orig_h, orig_w),
                inference_time=inference_time,
            ))
        
        return results[0] if single else results
    
    def train(
        self,
        data_dir: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.01,
        save_dir: str = "./runs/train",
        resume: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, List]:
        """
        Train the detector on a dataset.
        
        Args:
            data_dir: Path to dataset directory
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            save_dir: Directory to save checkpoints
            resume: Path to checkpoint to resume from
            **kwargs: Additional training options
            
        Returns:
            Training history
        """
        from neurova.object_detection.dataset import DetectionDataset
        from neurova.object_detection.trainer import DetectionTrainer, TrainingConfig
        
        # Load dataset
        dataset = DetectionDataset(
            data_dir=data_dir,
            names=self.class_names,
            img_size=self.img_size,
        )
        
        # Update class names from dataset if needed
        if dataset.num_classes > 0:
            self.class_names = dataset.class_names
            self.num_classes = dataset.num_classes
            
            # Rebuild model if class count changed
            if self.model.num_classes != self.num_classes:
                from neurova.object_detection.model import DetectionModel
                self.model = DetectionModel(
                    num_classes=self.num_classes,
                    model_size=self.model_size,
                )
        
        print(dataset.summary())
        
        # Create trainer
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_dir=save_dir,
            img_size=self.img_size,
            **kwargs,
        )
        
        trainer = DetectionTrainer(self.model, dataset, config)
        
        # Train
        history = trainer.train(resume=resume)
        
        return history
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model weights and configuration."""
        path = Path(path)
        
        # Save weights
        self.model.save(str(path))
        
        # Save config
        config = {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'model_size': self.model_size,
            'img_size': self.img_size,
        }
        
        config_path = path.with_suffix('.config.npz')
        np.savez(str(config_path), **config)
    
    def load(self, path: Union[str, Path]) -> None:
        """Load model weights and configuration."""
        path = Path(path)
        
        # Load weights
        self.model.load(str(path))
        
        # Load config if exists
        config_path = path.with_suffix('.config.npz')
        if config_path.exists():
            config = np.load(str(config_path), allow_pickle=True)
            self.class_names = list(config.get('class_names', self.class_names))
            self.num_classes = int(config.get('num_classes', self.num_classes))
            self.img_size = tuple(config.get('img_size', self.img_size))
    
    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from file."""
        try:
            from PIL import Image
            img = Image.open(path)
            return np.array(img.convert('RGB'))
        except ImportError:
            from neurova.io import read_image
            return read_image(path).data
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Tuple of (processed_image, scale, padding)
        """
        h, w = image.shape[:2]
        target_h, target_w = self.img_size
        
        # Compute scale
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        try:
            from PIL import Image
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            resized = np.array(pil_img)
        except ImportError:
            # Simple resize
            y_indices = (np.arange(new_h) * h / new_h).astype(int)
            x_indices = (np.arange(new_w) * w / new_w).astype(int)
            resized = image[y_indices][:, x_indices]
        
        # Pad to target size
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top = pad_h // 2
        left = pad_w // 2
        
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[top:top+new_h, left:left+new_w] = resized
        
        # Normalize to [0, 1] and add batch dimension
        processed = padded.astype(np.float32) / 255.0
        processed = processed[np.newaxis, ...]  # (1, H, W, C)
        
        return processed, scale, (top, left)
    
    def draw(
        self,
        image: np.ndarray,
        results: Optional[DetectionResult] = None,
        conf_threshold: float = 0.25,
        show_labels: bool = True,
        show_conf: bool = True,
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image
            results: DetectionResult (if None, runs detection first)
            conf_threshold: Confidence threshold for display
            show_labels: Show class labels
            show_conf: Show confidence scores
            colors: Custom colors per class
            
        Returns:
            Image with drawn detections
        """
        if results is None:
            results = self.detect(image, conf_threshold=conf_threshold)
        
        from neurova.object_detection.utils import draw_detections
        
        return draw_detections(
            image,
            boxes=results.boxes,
            class_ids=results.class_ids,
            scores=results.confidences,
            class_names=self.class_names,
            colors=[colors.get(i, None) if colors else None for i in range(self.num_classes)],
        )
    
    def summary(self) -> str:
        """Get model summary."""
        n_params = sum(p.data.size for p in self.model.parameters())
        
        lines = [
            f"ObjectDetector:",
            f"  Model size: {self.model_size}",
            f"  Classes: {self.num_classes}",
            f"  Image size: {self.img_size}",
            f"  Parameters: {n_params:,}",
            f"  Class names: {self.class_names[:5]}{'...' if len(self.class_names) > 5 else ''}",
        ]
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return f"ObjectDetector(num_classes={self.num_classes}, model_size='{self.model_size}')"

# Neurova Library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)
