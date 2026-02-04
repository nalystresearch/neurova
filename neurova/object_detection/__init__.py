# Neurova Library
# Copyright (c) 2025 Neurova Team
# Licensed under the MIT License
# @analytics with harry

"""
Neurova Object Detection Module.

Provides YOLO-style object detection with training and inference capabilities.

Dataset Structure (YOLO format):
    datasets/
    └── your_dataset/
        ├── images/
        │   ├── train/      # Training images (.jpg, .png)
        │   └── val/        # Validation images
        └── labels/
            ├── train/      # Label files (.txt) - SAME name as images
            └── val/        # Validation labels

Label Format (YOLO):
    <class_id> <x_center> <y_center> <width> <height>
    All values normalized (0-1) relative to image dimensions.

Usage:
    from neurova import object_detection
    from neurova.object_detection import ObjectDetector, DetectionDataset, DetectionTrainer
    
    # Load dataset
    dataset = DetectionDataset(
        data_dir='./datasets/my_data',
        names=['person', 'car', 'dog'],  # Class names
    )
    
    # Create detector
    detector = ObjectDetector(num_classes=3, model_size='small')
    
    # Train
    trainer = DetectionTrainer(detector, dataset)
    trainer.train(epochs=100, batch_size=16)
    
    # Save/Load model
    detector.save('my_model.npz')
    detector.load('my_model.npz')
    
    # Detect objects
    results = detector.detect(image, conf_threshold=0.5)
    for box in results:
        print(f"Class: {box.class_name}, Confidence: {box.confidence:.2f}")
"""

from neurova.object_detection.dataset import (
    DetectionDataset,
    DataConfig,
    create_data_yaml,
    load_data_yaml,
    parse_yolo_label,
    create_yolo_label,
    split_dataset,
    verify_dataset,
)

from neurova.object_detection.detector import (
    ObjectDetector,
    Detection,
    DetectionResult,
)

from neurova.object_detection.model import (
    DetectionModel,
    DetectionHead,
    Backbone,
    FeaturePyramid,
)

from neurova.object_detection.trainer import (
    DetectionTrainer,
    TrainingConfig,
    train_detector,
)

from neurova.object_detection.utils import (
    non_max_suppression,
    compute_iou,
    compute_iou_batch,
    xywh_to_xyxy,
    xyxy_to_xywh,
    normalize_boxes,
    denormalize_boxes,
    clip_boxes,
    draw_detections,
    compute_ap,
    compute_map,
    augment_detection,
)

__all__ = [
    # Dataset
    "DetectionDataset",
    "DataConfig",
    "create_data_yaml",
    "load_data_yaml",
    "parse_yolo_label",
    "create_yolo_label",
    "split_dataset",
    "verify_dataset",
    # Detector
    "ObjectDetector",
    "Detection",
    "DetectionResult",
    # Model
    "DetectionModel",
    "DetectionHead",
    "Backbone",
    "FeaturePyramid",
    # Training
    "DetectionTrainer",
    "TrainingConfig",
    "train_detector",
    # Utils
    "non_max_suppression",
    "compute_iou",
    "compute_iou_batch",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "normalize_boxes",
    "denormalize_boxes",
    "clip_boxes",
    "draw_detections",
    "compute_ap",
    "compute_map",
    "augment_detection",
]

# Neurova Library
# Copyright (c) 2025 Neurova Team
# Licensed under the MIT License
# @analytics with harry
