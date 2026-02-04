# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Face Dataset Management for Training and Evaluation.

Provides tools to organize, load, and split face datasets.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


class FaceDataset:
    """
    Face dataset for training and evaluation.
    
    Organizes face images into train/test/validation splits with labels.
    
    Directory structure:
        root/
            train/
                person1/
                    img1.jpg
                    img2.jpg
                person2/
                    img1.jpg
            test/
                person1/
                    img1.jpg
                person2/
                    img1.jpg
            validation/  (optional)
                ...
    
    Args:
        root_dir: Root directory containing train/test/validation folders.
        train_dir: Path to training images (default: root_dir/train).
        test_dir: Path to test images (default: root_dir/test).
        val_dir: Path to validation images (optional).
        image_size: Target size for face images (width, height).
        grayscale: Convert images to grayscale.
        
    Example:
        >>> dataset = FaceDataset('./faces', image_size=(100, 100))
        >>> train_images, train_labels = dataset.load_train()
        >>> test_images, test_labels = dataset.load_test()
        >>> print(f"Classes: {dataset.classes}")
    """
    
    def __init__(
        self,
        root_dir: Optional[str] = None,
        train_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (100, 100),
        grayscale: bool = True,
    ):
        self.root_dir = Path(root_dir) if root_dir else None
        self.image_size = image_size
        self.grayscale = grayscale
        
        # Set directories
        if train_dir:
            self.train_dir = Path(train_dir)
        elif self.root_dir:
            self.train_dir = self.root_dir / "train"
        else:
            self.train_dir = None
            
        if test_dir:
            self.test_dir = Path(test_dir)
        elif self.root_dir:
            self.test_dir = self.root_dir / "test"
        else:
            self.test_dir = None
            
        if val_dir:
            self.val_dir = Path(val_dir)
        elif self.root_dir:
            self.val_dir = self.root_dir / "validation"
        else:
            self.val_dir = None
        
        # Build class mapping
        self._classes: List[str] = []
        self._class_to_idx: Dict[str, int] = {}
        self._build_class_mapping()
    
    def _build_class_mapping(self):
        """Build class name to index mapping from train directory."""
        if self.train_dir and self.train_dir.exists():
            classes = sorted([
                d.name for d in self.train_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ])
            self._classes = classes
            self._class_to_idx = {name: idx for idx, name in enumerate(classes)}
    
    @property
    def classes(self) -> List[str]:
        """List of class names (person names)."""
        return self._classes
    
    @property
    def num_classes(self) -> int:
        """Number of classes (persons)."""
        return len(self._classes)
    
    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Mapping from class name to index."""
        return self._class_to_idx
    
    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load and preprocess an image."""
        try:
            from PIL import Image
            
            img = Image.open(path)
            
            if self.grayscale:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            
            if self.image_size:
                img = img.resize(self.image_size, Image.LANCZOS)
            
            return np.array(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _load_from_dir(
        self, 
        directory: Path
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Load all images from a directory structure."""
        images = []
        labels = []
        filenames = []
        
        if not directory or not directory.exists():
            return images, labels, filenames
        
        for class_dir in sorted(directory.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            class_name = class_dir.name
            if class_name not in self._class_to_idx:
                # Add new class if not in training set
                idx = len(self._classes)
                self._classes.append(class_name)
                self._class_to_idx[class_name] = idx
            
            class_idx = self._class_to_idx[class_name]
            
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    img = self._load_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(class_idx)
                        filenames.append(str(img_path))
        
        return images, labels, filenames
    
    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training images and labels.
        
        Returns:
            (images, labels) tuple of numpy arrays.
        """
        images, labels, _ = self._load_from_dir(self.train_dir)
        if not images:
            raise ValueError(f"No training images found in {self.train_dir}")
        return np.array(images), np.array(labels)
    
    def load_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test images and labels.
        
        Returns:
            (images, labels) tuple of numpy arrays.
        """
        images, labels, _ = self._load_from_dir(self.test_dir)
        if not images:
            raise ValueError(f"No test images found in {self.test_dir}")
        return np.array(images), np.array(labels)
    
    def load_validation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load validation images and labels.
        
        Returns:
            (images, labels) tuple of numpy arrays.
        """
        images, labels, _ = self._load_from_dir(self.val_dir)
        if not images:
            raise ValueError(f"No validation images found in {self.val_dir}")
        return np.array(images), np.array(labels)
    
    def load_all(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load all splits.
        
        Returns:
            Dictionary with 'train', 'test', 'validation' keys.
        """
        result = {}
        
        if self.train_dir and self.train_dir.exists():
            result['train'] = self.load_train()
        
        if self.test_dir and self.test_dir.exists():
            result['test'] = self.load_test()
        
        if self.val_dir and self.val_dir.exists():
            result['validation'] = self.load_validation()
        
        return result
    
    def iter_train(self) -> Iterator[Tuple[np.ndarray, int, str]]:
        """Iterate over training images with (image, label, filename)."""
        images, labels, filenames = self._load_from_dir(self.train_dir)
        for img, label, fname in zip(images, labels, filenames):
            yield img, label, fname
    
    def iter_test(self) -> Iterator[Tuple[np.ndarray, int, str]]:
        """Iterate over test images with (image, label, filename)."""
        images, labels, filenames = self._load_from_dir(self.test_dir)
        for img, label, fname in zip(images, labels, filenames):
            yield img, label, fname
    
    def get_label_name(self, label_idx: int) -> str:
        """Get class name from label index."""
        if 0 <= label_idx < len(self._classes):
            return self._classes[label_idx]
        return f"unknown_{label_idx}"
    
    def summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        summary = {
            'num_classes': self.num_classes,
            'classes': self._classes,
            'image_size': self.image_size,
            'grayscale': self.grayscale,
        }
        
        if self.train_dir and self.train_dir.exists():
            images, labels, _ = self._load_from_dir(self.train_dir)
            summary['train_samples'] = len(images)
            summary['train_per_class'] = {
                self._classes[i]: labels.count(i) 
                for i in range(len(self._classes))
            }
        
        if self.test_dir and self.test_dir.exists():
            images, labels, _ = self._load_from_dir(self.test_dir)
            summary['test_samples'] = len(images)
        
        if self.val_dir and self.val_dir.exists():
            images, labels, _ = self._load_from_dir(self.val_dir)
            summary['validation_samples'] = len(images)
        
        return summary
    
    def __repr__(self) -> str:
        return (
            f"FaceDataset(classes={self.num_classes}, "
            f"image_size={self.image_size}, grayscale={self.grayscale})"
        )


def create_face_dataset(
    output_dir: str,
    class_names: Optional[List[str]] = None,
    create_splits: bool = True,
) -> FaceDataset:
    """
    Create a new face dataset directory structure.
    
    Args:
        output_dir: Root directory for the dataset.
        class_names: Optional list of class (person) names to create.
        create_splits: Create train/test/validation subdirectories.
    
    Returns:
        FaceDataset instance for the created structure.
        
    Example:
        >>> dataset = create_face_dataset(
        ...     './my_faces',
        ...     class_names=['alice', 'bob', 'charlie']
        ... )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'test', 'validation'] if create_splits else ['data']
    
    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        if class_names:
            for name in class_names:
                (split_dir / name).mkdir(exist_ok=True)
    
    # Create webcam_output folder
    (output_path / 'webcam_output').mkdir(exist_ok=True)
    
    return FaceDataset(root_dir=str(output_path))


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> FaceDataset:
    """
    Split a flat face dataset into train/test/validation.
    
    Args:
        source_dir: Directory with class subdirectories containing images.
        output_dir: Output directory for split dataset.
        train_ratio: Fraction for training (default 0.7).
        test_ratio: Fraction for testing (default 0.2).
        val_ratio: Fraction for validation (default 0.1).
        shuffle: Shuffle images before splitting.
        seed: Random seed for reproducibility.
    
    Returns:
        FaceDataset instance for the split dataset.
        
    Example:
        >>> dataset = split_dataset('./raw_faces', './split_faces')
    """
    if seed is not None:
        np.random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Normalize ratios
    total = train_ratio + test_ratio + val_ratio
    train_ratio /= total
    test_ratio /= total
    val_ratio /= total
    
    # Create output structure
    for split in ['train', 'test', 'validation']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        class_name = class_dir.name
        
        # Get all images
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        ]
        
        if shuffle:
            np.random.shuffle(images)
        
        # Calculate split indices
        n = len(images)
        train_end = int(n * train_ratio)
        test_end = train_end + int(n * test_ratio)
        
        splits = {
            'train': images[:train_end],
            'test': images[train_end:test_end],
            'validation': images[test_end:],
        }
        
        # Copy images to splits
        for split_name, split_images in splits.items():
            split_class_dir = output_path / split_name / class_name
            split_class_dir.mkdir(exist_ok=True)
            
            for img_path in split_images:
                dest = split_class_dir / img_path.name
                shutil.copy2(img_path, dest)
    
    return FaceDataset(root_dir=str(output_path))


class WebcamDataCollector:
    """
    Collect face images from webcam for dataset creation.
    
    Args:
        output_dir: Directory to save captured images.
        person_name: Name/label for the person being captured.
        detector: Face detector to use (optional).
        
    Example:
        >>> collector = WebcamDataCollector('./dataset/train', 'john')
        >>> collector.collect(num_images=50)
    """
    
    def __init__(
        self,
        output_dir: str,
        person_name: str,
        detector: Optional[Any] = None,
    ):
        self.output_dir = Path(output_dir) / person_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.person_name = person_name
        self.detector = detector
    
    def collect(
        self,
        num_images: int = 50,
        delay: float = 0.5,
        show_preview: bool = True,
        auto_crop: bool = True,
        image_size: Tuple[int, int] = (100, 100),
    ) -> int:
        """
        Collect face images from webcam.
        
        Args:
            num_images: Number of images to capture.
            delay: Delay between captures in seconds.
            show_preview: Show webcam preview window.
            auto_crop: Automatically crop detected faces.
            image_size: Size to resize cropped faces.
        
        Returns:
            Number of images successfully captured.
        """
        import time
        
        try:
            import neurova.nvc as nvc
        except ImportError:
            raise ImportError("neurova.nvc is required for webcam capture")
        
        # Initialize detector if needed
        if auto_crop and self.detector is None:
            from neurova.face.detector import FaceDetector
            self.detector = FaceDetector(method='haar')
        
        # Open webcam
        cap = nvc.VideoCapture(0)
        cap.set(nvc.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(nvc.CAP_PROP_FRAME_HEIGHT, 480)
        cap.open()
        
        print(f"Collecting {num_images} images for '{self.person_name}'")
        print("Position your face in the frame. Press 'q' to quit.")
        
        collected = 0
        frame_count = 0
        
        try:
            while collected < num_images:
                ret, frame = cap.read()
                if frame is None:
                    continue
                
                frame_count += 1
                display_frame = frame.copy()
                
                # Detect faces
                faces = []
                if self.detector:
                    faces = self.detector.detect(frame)
                
                # Draw rectangles
                for (x, y, w, h, conf) in faces:
                    nvc.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Status text
                nvc.putText(
                    display_frame, 
                    f"Collected: {collected}/{num_images}", 
                    (10, 30), 
                    nvc.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                if show_preview:
                    nvc.imshow("Collect Faces", display_frame)
                    key = nvc.waitKey(1)
                    if key == ord('q'):
                        break
                
                # Capture on interval
                if frame_count % int(delay * 30) == 0:
                    if faces and auto_crop:
                        # Save largest face
                        x, y, w, h, _ = max(faces, key=lambda f: f[2] * f[3])
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Resize
                        from PIL import Image
                        face_pil = Image.fromarray(face_img)
                        face_pil = face_pil.resize(image_size, Image.LANCZOS)
                        face_img = np.array(face_pil)
                    else:
                        face_img = frame
                    
                    # Save
                    img_path = self.output_dir / f"{self.person_name}_{collected:04d}.png"
                    nvc.imwrite(str(img_path), face_img)
                    collected += 1
                    print(f"Captured {collected}/{num_images}")
        
        finally:
            cap.release()
            if show_preview:
                nvc.destroyAllWindows()
        
        print(f"Collected {collected} images in {self.output_dir}")
        return collected
