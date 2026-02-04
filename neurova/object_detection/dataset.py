# Neurova Library
# Copyright (c) 2025 Neurova Team
# Licensed under the MIT License
# @analytics with harry

"""
YOLO-format Dataset Loader for Object Detection.

Supports the standard YOLO directory structure:
    datasets/
    └── your_dataset/
        ├── data.yaml          # Optional config file
        ├── images/
        │   ├── train/         # Training images
        │   └── val/           # Validation images
        └── labels/
            ├── train/         # Training labels (.txt)
            └── val/           # Validation labels (.txt)

Label format (per line):
    <class_id> <x_center> <y_center> <width> <height>
    
All coordinates are normalized (0-1) relative to image dimensions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


@dataclass
class DataConfig:
    """Dataset configuration (similar to YOLO data.yaml).
    
    Attributes:
        path: Root path to dataset
        train: Training images path (relative to path)
        val: Validation images path (relative to path)
        test: Test images path (optional)
        names: Class name dictionary {id: name}
        nc: Number of classes (auto-computed from names)
    """
    path: str
    train: str = "images/train"
    val: str = "images/val"
    test: Optional[str] = None
    names: Dict[int, str] = field(default_factory=dict)
    
    @property
    def nc(self) -> int:
        """Number of classes."""
        return len(self.names)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "DataConfig":
        """Load configuration from YAML file."""
        return load_data_yaml(yaml_path)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        create_data_yaml(yaml_path, self)


def create_data_yaml(
    yaml_path: Union[str, Path],
    config: Optional[DataConfig] = None,
    path: Optional[str] = None,
    train: str = "images/train",
    val: str = "images/val",
    test: Optional[str] = None,
    names: Optional[Dict[int, str]] = None,
) -> None:
    """
    Create a data.yaml configuration file.
    
    Args:
        yaml_path: Path to save YAML file
        config: DataConfig object (if provided, other args are ignored)
        path: Root path to dataset
        train: Training images path
        val: Validation images path
        test: Test images path
        names: Class names dictionary
    """
    if config is not None:
        path = config.path
        train = config.train
        val = config.val
        test = config.test
        names = config.names
    
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# Neurova Object Detection Dataset Config",
        f"# Auto-generated",
        f"",
        f"path: {path}",
        f"train: {train}",
        f"val: {val}",
    ]
    
    if test:
        lines.append(f"test: {test}")
    
    lines.extend([
        f"",
        f"# Classes",
        f"names:",
    ])
    
    if names:
        for idx, name in sorted(names.items()):
            lines.append(f"  {idx}: {name}")
    
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(lines))


def load_data_yaml(yaml_path: Union[str, Path]) -> DataConfig:
    """
    Load data configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        DataConfig object
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Data config not found: {yaml_path}")
    
    # Simple YAML parser (no external dependency)
    config = {}
    names = {}
    in_names = False
    
    with open(yaml_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line == 'names:':
                in_names = True
                continue
            
            if in_names:
                if ':' in line:
                    parts = line.split(':', 1)
                    try:
                        idx = int(parts[0].strip())
                        name = parts[1].strip()
                        names[idx] = name
                    except ValueError:
                        in_names = False
                else:
                    in_names = False
            
            if not in_names and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if value:  # Only set if value is not empty
                    config[key] = value
    
    return DataConfig(
        path=config.get('path', '.'),
        train=config.get('train', 'images/train'),
        val=config.get('val', 'images/val'),
        test=config.get('test'),
        names=names,
    )


def parse_yolo_label(label_path: Union[str, Path]) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a YOLO format label file.
    
    Args:
        label_path: Path to label .txt file
        
    Returns:
        List of (class_id, x_center, y_center, width, height) tuples
        All coordinates are normalized (0-1)
    """
    label_path = Path(label_path)
    
    if not label_path.exists():
        return []
    
    labels = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append((class_id, x_center, y_center, width, height))
                except ValueError:
                    continue
    
    return labels


def create_yolo_label(
    label_path: Union[str, Path],
    labels: List[Tuple[int, float, float, float, float]],
) -> None:
    """
    Create a YOLO format label file.
    
    Args:
        label_path: Path to save label file
        labels: List of (class_id, x_center, y_center, width, height) tuples
    """
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_path, 'w') as f:
        for class_id, x, y, w, h in labels:
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


@dataclass
class DetectionSample:
    """A single detection sample with image and labels.
    
    Attributes:
        image_path: Path to image file
        label_path: Path to label file
        image: Loaded image array (H, W, C)
        boxes: Box coordinates (N, 4) in xywh normalized format
        class_ids: Class IDs (N,)
    """
    image_path: str
    label_path: Optional[str] = None
    image: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    class_ids: Optional[np.ndarray] = None
    
    def load(self) -> "DetectionSample":
        """Load image and labels from disk."""
        # Load image
        try:
            from PIL import Image
            img = Image.open(self.image_path)
            self.image = np.array(img.convert('RGB'))
        except ImportError:
            from neurova.io import read_image
            self.image = read_image(self.image_path).data
        
        # Load labels
        if self.label_path:
            labels = parse_yolo_label(self.label_path)
            if labels:
                self.class_ids = np.array([l[0] for l in labels], dtype=np.int32)
                self.boxes = np.array([[l[1], l[2], l[3], l[4]] for l in labels], dtype=np.float32)
            else:
                self.class_ids = np.array([], dtype=np.int32)
                self.boxes = np.array([], dtype=np.float32).reshape(0, 4)
        else:
            self.class_ids = np.array([], dtype=np.int32)
            self.boxes = np.array([], dtype=np.float32).reshape(0, 4)
        
        return self


class DetectionDataset:
    """
    YOLO-format dataset loader for object detection.
    
    Supports the standard YOLO directory structure with automatic label discovery.
    
    Args:
        data_dir: Root directory of dataset
        names: List or dict of class names
        train: Training images subdirectory
        val: Validation images subdirectory
        test: Test images subdirectory (optional)
        img_size: Target image size (width, height)
        augment: Apply data augmentation
        cache: Cache images in memory
        
    Example:
        >>> dataset = DetectionDataset(
        ...     data_dir='./datasets/coco128',
        ...     names=['person', 'car', 'dog'],
        ... )
        >>> print(f"Training samples: {len(dataset.train_samples)}")
        >>> for sample in dataset.train_iter(batch_size=8):
        ...     images, targets = sample
        ...     # Train model
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        names: Optional[Union[List[str], Dict[int, str]]] = None,
        train: str = "images/train",
        val: str = "images/val",
        test: Optional[str] = None,
        img_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        cache: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.cache = cache
        
        # Load or create config
        config_path = self.data_dir / "data.yaml"
        if config_path.exists() and names is None:
            self.config = load_data_yaml(config_path)
            self._train_path = self.data_dir / self.config.train
            self._val_path = self.data_dir / self.config.val
            self._test_path = self.data_dir / self.config.test if self.config.test else None
        else:
            self._train_path = self.data_dir / train
            self._val_path = self.data_dir / val
            self._test_path = self.data_dir / test if test else None
            
            # Convert names to dict
            if names is None:
                names = {}
            elif isinstance(names, list):
                names = {i: name for i, name in enumerate(names)}
            
            self.config = DataConfig(
                path=str(self.data_dir),
                train=train,
                val=val,
                test=test,
                names=names,
            )
        
        # Discover samples
        self.train_samples = self._discover_samples(self._train_path)
        self.val_samples = self._discover_samples(self._val_path)
        self.test_samples = self._discover_samples(self._test_path) if self._test_path else []
        
        # Cache storage
        self._image_cache: Dict[str, np.ndarray] = {}
        
        # Auto-detect classes if not provided
        if not self.config.names:
            self._auto_detect_classes()
    
    def _discover_samples(self, images_path: Optional[Path]) -> List[DetectionSample]:
        """Discover image and label pairs."""
        if images_path is None or not images_path.exists():
            return []
        
        samples = []
        
        # Find all images
        for ext in IMAGE_EXTENSIONS:
            for img_path in images_path.glob(f"*{ext}"):
                # Find corresponding label
                label_path = self._get_label_path(img_path)
                
                samples.append(DetectionSample(
                    image_path=str(img_path),
                    label_path=str(label_path) if label_path and label_path.exists() else None,
                ))
        
        return samples
    
    def _get_label_path(self, image_path: Path) -> Optional[Path]:
        """Get label path for an image."""
        # Replace 'images' with 'labels' in path
        label_path = Path(str(image_path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
        label_path = label_path.with_suffix('.txt')
        return label_path
    
    def _auto_detect_classes(self) -> None:
        """Auto-detect class IDs from label files."""
        class_ids = set()
        
        for sample in self.train_samples + self.val_samples:
            if sample.label_path:
                labels = parse_yolo_label(sample.label_path)
                for label in labels:
                    class_ids.add(label[0])
        
        if class_ids:
            self.config.names = {i: f"class_{i}" for i in sorted(class_ids)}
    
    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return self.config.nc
    
    @property
    def class_names(self) -> List[str]:
        """List of class names."""
        return [self.config.names.get(i, f"class_{i}") for i in range(self.num_classes)]
    
    def __len__(self) -> int:
        """Total number of training samples."""
        return len(self.train_samples)
    
    def load_sample(
        self,
        sample: DetectionSample,
        augment: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess a sample.
        
        Args:
            sample: DetectionSample to load
            augment: Apply augmentation
            
        Returns:
            Tuple of (image, boxes, class_ids)
        """
        # Check cache
        if self.cache and sample.image_path in self._image_cache:
            image = self._image_cache[sample.image_path].copy()
        else:
            sample.load()
            image = sample.image
            
            if self.cache:
                self._image_cache[sample.image_path] = image.copy()
        
        boxes = sample.boxes if sample.boxes is not None else np.array([]).reshape(0, 4)
        class_ids = sample.class_ids if sample.class_ids is not None else np.array([])
        
        # Resize image
        if self.img_size:
            image, boxes = self._resize_with_boxes(image, boxes, self.img_size)
        
        # Apply augmentation
        if augment and self.augment:
            from neurova.object_detection.utils import augment_detection
            image, boxes, class_ids = augment_detection(
                image, boxes, class_ids,
                horizontal_flip=True,
                brightness_range=(0.8, 1.2),
            )
        
        return image, boxes, class_ids
    
    def _resize_with_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image while preserving normalized box coordinates."""
        try:
            from PIL import Image
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
            image = np.array(pil_img)
        except ImportError:
            # Simple resize using numpy
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate indices for resize
            y_indices = (np.arange(target_h) * h / target_h).astype(int)
            x_indices = (np.arange(target_w) * w / target_w).astype(int)
            
            image = image[y_indices][:, x_indices]
        
        # Boxes are normalized, so they don't need adjustment
        return image, boxes
    
    def train_iter(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        augment: bool = True,
    ) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        """
        Iterate over training data in batches.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Shuffle samples each epoch
            augment: Apply data augmentation
            
        Yields:
            Tuple of (batch_images, batch_targets)
            - batch_images: (B, H, W, C) array
            - batch_targets: List of dicts with 'boxes' and 'class_ids'
        """
        yield from self._batch_iter(self.train_samples, batch_size, shuffle, augment)
    
    def val_iter(
        self,
        batch_size: int = 16,
    ) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        """
        Iterate over validation data in batches.
        
        Args:
            batch_size: Number of samples per batch
            
        Yields:
            Tuple of (batch_images, batch_targets)
        """
        yield from self._batch_iter(self.val_samples, batch_size, shuffle=False, augment=False)
    
    def _batch_iter(
        self,
        samples: List[DetectionSample],
        batch_size: int,
        shuffle: bool,
        augment: bool,
    ) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        """Batch iterator implementation."""
        if shuffle:
            indices = np.random.permutation(len(samples))
        else:
            indices = np.arange(len(samples))
        
        for start_idx in range(0, len(samples), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_images = []
            batch_targets = []
            
            for idx in batch_indices:
                sample = samples[idx]
                sample.load()
                image, boxes, class_ids = self.load_sample(sample, augment=augment)
                
                batch_images.append(image)
                batch_targets.append({
                    'boxes': boxes,
                    'class_ids': class_ids,
                    'image_path': sample.image_path,
                })
            
            batch_images = np.stack(batch_images, axis=0)
            yield batch_images, batch_targets
    
    def summary(self) -> str:
        """Get dataset summary string."""
        lines = [
            f"DetectionDataset: {self.data_dir}",
            f"  Classes: {self.num_classes}",
            f"  Class names: {self.class_names}",
            f"  Train samples: {len(self.train_samples)}",
            f"  Val samples: {len(self.val_samples)}",
            f"  Test samples: {len(self.test_samples)}",
            f"  Image size: {self.img_size}",
        ]
        return '\n'.join(lines)


def split_dataset(
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> None:
    """
    Split a dataset of images into train/val/test splits.
    
    Args:
        images_dir: Directory containing images and labels
        output_dir: Output directory for split dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
    """
    import shutil
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # Find all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(image_files)
    
    # Split
    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:],
    }
    
    # Copy files
    for split_name, files in splits.items():
        if not files:
            continue
        
        img_dir = output_dir / 'images' / split_name
        lbl_dir = output_dir / 'labels' / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in files:
            # Copy image
            shutil.copy(img_path, img_dir / img_path.name)
            
            # Copy label if exists
            lbl_path = img_path.with_suffix('.txt')
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_dir / lbl_path.name)


def verify_dataset(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Verify dataset integrity and return statistics.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Dict with verification results
    """
    data_dir = Path(data_dir)
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'total_images': 0,
            'total_labels': 0,
            'images_without_labels': 0,
            'labels_without_images': 0,
            'class_distribution': {},
            'boxes_per_image': [],
        }
    }
    
    # Check structure
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    
    if not images_dir.exists():
        results['errors'].append(f"Images directory not found: {images_dir}")
        results['valid'] = False
        return results
    
    if not labels_dir.exists():
        results['warnings'].append(f"Labels directory not found: {labels_dir}")
    
    # Scan images and labels
    image_stems = set()
    label_stems = set()
    
    for split in ['train', 'val', 'test']:
        split_images = images_dir / split
        split_labels = labels_dir / split
        
        if split_images.exists():
            for ext in IMAGE_EXTENSIONS:
                for img in split_images.glob(f"*{ext}"):
                    image_stems.add(img.stem)
                    results['stats']['total_images'] += 1
        
        if split_labels.exists():
            for lbl in split_labels.glob("*.txt"):
                label_stems.add(lbl.stem)
                results['stats']['total_labels'] += 1
                
                # Parse label
                labels = parse_yolo_label(lbl)
                results['stats']['boxes_per_image'].append(len(labels))
                
                for class_id, *_ in labels:
                    results['stats']['class_distribution'][class_id] = \
                        results['stats']['class_distribution'].get(class_id, 0) + 1
    
    # Check mismatches
    results['stats']['images_without_labels'] = len(image_stems - label_stems)
    results['stats']['labels_without_images'] = len(label_stems - image_stems)
    
    if results['stats']['images_without_labels'] > 0:
        results['warnings'].append(
            f"{results['stats']['images_without_labels']} images have no corresponding labels"
        )
    
    if results['stats']['labels_without_images'] > 0:
        results['warnings'].append(
            f"{results['stats']['labels_without_images']} labels have no corresponding images"
        )
    
    return results

# Neurova Library
# Copyright (c) 2025 Neurova Team
# Licensed under the MIT License
# @analytics with harry
