#!/usr/bin/env python
# neurova library
# Copyright (c) 2025 Neurova Team
# licensed under the mit license
# @analytics with harry

"""
Neurova Object Detection Example

This example demonstrates how to use Neurova's object detection module
for training and inference, similar to YOLO.

Dataset Structure (YOLO format):
    datasets/
    └── your_dataset/
        ├── data.yaml          # Optional config file
        ├── images/
        │   ├── train/         # Training images (.jpg, .png)
        │   └── val/           # Validation images
        └── labels/
            ├── train/         # Label files (.txt)
            └── val/           # Validation labels

Label Format (per line in .txt file):
    <class_id> <x_center> <y_center> <width> <height>
    
    All coordinates normalized to 0-1 relative to image dimensions.
    Example: 0 0.5 0.5 0.3 0.4  (person at center, 30% width, 40% height)
"""

import numpy as np
from pathlib import Path

# import neurova object detection
from neurova import object_detection
from neurova.object_detection import (
    ObjectDetector,
    DetectionDataset,
    DetectionTrainer,
    Detection,
    DetectionResult,
    create_data_yaml,
    parse_yolo_label,
    create_yolo_label,
    verify_dataset,
)


def example_create_dataset():
    """Example: Create a dataset structure programmatically."""
    print("=" * 60)
    print("Example 1: Creating Dataset Structure")
    print("=" * 60)
    
# define dataset path
    dataset_dir = Path("./my_detection_dataset")
    
# create directory structure
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml config
    create_data_yaml(
        dataset_dir / "data.yaml",
        path=str(dataset_dir),
        train="images/train",
        val="images/val",
        names={
            0: "person",
            1: "car",
            2: "dog",
            3: "bicycle",
        }
    )
    
    print(f"Created dataset structure at: {dataset_dir}")
    print("\nNext steps:")
    print("  1. Add your images to images/train/ and images/val/")
    print("  2. Create label files in labels/train/ and labels/val/")
    print("  3. Each image needs a .txt label file with same name")
    print()


def example_create_labels():
    """Example: Create YOLO format labels."""
    print("=" * 60)
    print("Example 2: Creating YOLO Labels")
    print("=" * 60)
    
    # Example: Create label for an image
    labels = [
        # (class_id, x_center, y_center, width, height)
        (0, 0.5, 0.4, 0.3, 0.6),    # person at center
        (1, 0.2, 0.7, 0.15, 0.2),   # car on left
        (2, 0.8, 0.5, 0.1, 0.15),   # dog on right
    ]
    
# save to file
    label_path = Path("./example_label.txt")
    create_yolo_label(label_path, labels)
    
    print(f"Created label file: {label_path}")
    print("\nLabel content:")
    with open(label_path, 'r') as f:
        print(f.read())
    
# parse it back
    parsed = parse_yolo_label(label_path)
    print("Parsed labels:")
    for cls_id, x, y, w, h in parsed:
        print(f"  Class {cls_id}: center=({x:.2f}, {y:.2f}), size=({w:.2f}, {h:.2f})")
    
# cleanup
    label_path.unlink()
    print()


def example_load_dataset():
    """Example: Load an existing dataset."""
    print("=" * 60)
    print("Example 3: Loading Detection Dataset")
    print("=" * 60)
    
# create a small test dataset in memory
    dataset_dir = Path("./test_dataset")
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
# create a dummy image
    from PIL import Image
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    img.save(dataset_dir / "images" / "train" / "test_image.jpg")
    
# create corresponding label
    create_yolo_label(
        dataset_dir / "labels" / "train" / "test_image.txt",
        [(0, 0.5, 0.5, 0.3, 0.4)]
    )
    
# load dataset
    dataset = DetectionDataset(
        data_dir=dataset_dir,
        names=["object"],
        img_size=(640, 640),
    )
    
    print(dataset.summary())
    
# cleanup
    import shutil
    shutil.rmtree(dataset_dir)
    print()


def example_create_detector():
    """Example: Create object detector."""
    print("=" * 60)
    print("Example 4: Creating Object Detector")
    print("=" * 60)
    
# define your classes
    class_names = ["person", "car", "dog", "bicycle", "cat"]
    
# create detector with different sizes
    for size in ["nano", "small", "medium"]:
        detector = ObjectDetector(
            num_classes=len(class_names),
            class_names=class_names,
            model_size=size,
        )
        print(f"\nModel size: {size}")
        print(detector.summary())
    
    print()


def example_detect_objects():
    """Example: Run object detection on an image."""
    print("=" * 60)
    print("Example 5: Running Object Detection")
    print("=" * 60)
    
# create detector
    detector = ObjectDetector(
        num_classes=3,
        class_names=["person", "car", "dog"],
        model_size="nano",
    )
    
    # Create a dummy image (in real use, load your image)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Running detection on image...")
    results = detector.detect(image, conf_threshold=0.1)
    
    print(f"\nFound {len(results)} detections")
    print(f"Inference time: {results.inference_time:.1f}ms")
    
    for det in results:
        print(f"  - {det}")
    
# access detection properties
    if len(results) > 0:
        print("\nDetection properties:")
        print(f"  Boxes shape: {results.boxes.shape}")
        print(f"  Classes: {results.class_names}")
    
    print()


def example_training_workflow():
    """Example: Full training workflow."""
    print("=" * 60)
    print("Example 6: Training Workflow")
    print("=" * 60)
    
    print("""
# Full training example (requires actual dataset):

from neurova.object_detection import ObjectDetector, train_detector

# Option 1: Using convenience function
model, history = train_detector(
    data_dir='./datasets/my_data',
    class_names=['person', 'car', 'dog'],
    model_size='small',
    epochs=100,
    batch_size=16,
    save_dir='./runs/train',
)

# Option 2: Using ObjectDetector directly
detector = ObjectDetector(
    num_classes=3,
    class_names=['person', 'car', 'dog'],
    model_size='small',
)

history = detector.train(
    data_dir='./datasets/my_data',
    epochs=100,
    batch_size=16,
    learning_rate=0.01,
)

# save model
detector.save('my_model.npz')

# load model later
detector.load('my_model.npz')

# detect objects
results = detector.detect('test_image.jpg')
for det in results:
    print(f"{det.class_name}: {det.confidence:.2f}")
""")


def example_dataset_verification():
    """Example: Verify dataset integrity."""
    print("=" * 60)
    print("Example 7: Dataset Verification")
    print("=" * 60)
    
# create a test dataset
    dataset_dir = Path("./verify_test")
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
# create test files
    from PIL import Image
    for i in range(3):
        img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
        img.save(dataset_dir / "images" / "train" / f"img_{i}.jpg")
        
        if i < 2:  # Only create labels for first 2 images
            create_yolo_label(
                dataset_dir / "labels" / "train" / f"img_{i}.txt",
                [(0, 0.5, 0.5, 0.3, 0.3)]
            )
    
# verify dataset
    results = verify_dataset(dataset_dir)
    
    print(f"Dataset valid: {results['valid']}")
    print(f"Total images: {results['stats']['total_images']}")
    print(f"Total labels: {results['stats']['total_labels']}")
    print(f"Images without labels: {results['stats']['images_without_labels']}")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
# cleanup
    import shutil
    shutil.rmtree(dataset_dir)
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("NEUROVA OBJECT DETECTION EXAMPLES")
    print("=" * 60 + "\n")
    
# run examples
    example_create_labels()
    example_create_detector()
    example_detect_objects()
    example_training_workflow()
    
# these require filesystem access
    try:
        example_create_dataset()
        example_load_dataset()
        example_dataset_verification()
    except Exception as e:
        print(f"Skipped filesystem examples: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# neurova library
# Copyright (c) 2025 Neurova Team
# licensed under the mit license
# @analytics with harry
