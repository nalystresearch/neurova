# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Face Trainer for training custom face detection and recognition models.

Supports training custom Haar cascades, LBP cascades, and recognition models.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class FaceTrainer:
    """
    Train custom face detection and recognition models.
    
    This class provides methods to:
    - Train face detection models (cascade classifiers)
    - Train face recognition models (LBPH, EigenFace, etc.)
    - Fine-tune pre-trained models
    - Evaluate model performance
    
    Args:
        output_dir: Directory to save trained models.
        method: Training method ('cascade', 'recognition', 'embedding').
        
    Example:
        >>> trainer = FaceTrainer(output_dir='./models')
        >>> trainer.train_recognizer(dataset, method='lbph')
        >>> trainer.evaluate(test_dataset)
        >>> trainer.save('my_face_model')
    """
    
    def __init__(
        self,
        output_dir: str = "./models",
        method: str = "recognition",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        self._model = None
        self._history: List[Dict[str, Any]] = []
    
    def train_recognizer(
        self,
        dataset: Any,  # FaceDataset
        method: str = "lbph",
        epochs: int = 1,
        augment: bool = True,
        verbose: bool = True,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train a face recognition model.
        
        Args:
            dataset: FaceDataset with training data.
            method: Recognition method ('lbph', 'eigen', 'fisher', 'embedding').
            epochs: Number of training iterations (for embedding models).
            augment: Apply data augmentation.
            verbose: Print training progress.
            callback: Callback function called each epoch.
        
        Returns:
            Training history with metrics.
        """
        from .recognizer import FaceRecognizer
        from .dataset import FaceDataset
        
        if not isinstance(dataset, FaceDataset):
            raise ValueError("dataset must be a FaceDataset instance")
        
        if verbose:
            print(f"Training {method.upper()} face recognizer...")
            print(f"Training samples: {len(dataset.train_images)}")
            print(f"Classes: {len(dataset.class_names)}")
        
        # Create recognizer
        self._model = FaceRecognizer(method=method)
        
        # Get training data
        faces = dataset.train_images
        labels = [dataset.class_names[lbl] for lbl in dataset.train_labels]
        
        # Train
        start_time = self._get_time()
        self._model.train(faces, labels, augment=augment)
        train_time = self._get_time() - start_time
        
        # Evaluate on validation set if available
        val_accuracy = 0.0
        if dataset.val_images:
            val_accuracy = self._evaluate_accuracy(
                self._model, 
                dataset.val_images, 
                dataset.val_labels,
                dataset.class_names
            )
        
        history = {
            'method': method,
            'train_samples': len(faces),
            'classes': len(dataset.class_names),
            'train_time': train_time,
            'val_accuracy': val_accuracy,
            'augmented': augment,
        }
        
        self._history.append(history)
        
        if verbose:
            print(f"Training completed in {train_time:.2f}s")
            if val_accuracy > 0:
                print(f"Validation accuracy: {val_accuracy:.2%}")
        
        if callback:
            callback(history)
        
        return history
    
    def train_detector(
        self,
        positive_dir: str,
        negative_dir: str,
        cascade_type: str = "haar",
        num_stages: int = 20,
        min_hit_rate: float = 0.995,
        max_false_alarm: float = 0.5,
        width: int = 24,
        height: int = 24,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train a custom cascade classifier for face detection.
        
        Note: This requires the traincascade tool.
        
        Args:
            positive_dir: Directory with positive (face) images.
            negative_dir: Directory with negative (non-face) images.
            cascade_type: 'haar' or 'lbp'.
            num_stages: Number of cascade stages.
            min_hit_rate: Minimum hit rate per stage.
            max_false_alarm: Maximum false alarm rate per stage.
            width: Sample width.
            height: Sample height.
            verbose: Print training progress.
        
        Returns:
            Training results.
        """
        if verbose:
            print(f"Training {cascade_type.upper()} cascade classifier...")
        
        # Create output directory
        cascade_dir = self.output_dir / f"cascade_{cascade_type}"
        cascade_dir.mkdir(parents=True, exist_ok=True)
        
        # Count images
        pos_images = list(Path(positive_dir).glob("**/*.*"))
        neg_images = list(Path(negative_dir).glob("**/*.*"))
        
        pos_count = len([p for p in pos_images if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        neg_count = len([p for p in neg_images if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        if verbose:
            print(f"Positive samples: {pos_count}")
            print(f"Negative samples: {neg_count}")
        
        # Create annotation files
        pos_file = cascade_dir / "positives.txt"
        neg_file = cascade_dir / "negatives.txt"
        
        with open(pos_file, 'w') as f:
            for img in pos_images:
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    f.write(f"{img} 1 0 0 {width} {height}\n")
        
        with open(neg_file, 'w') as f:
            for img in neg_images:
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    f.write(f"{img}\n")
        
        # Create vec file
        vec_file = cascade_dir / "positives.vec"
        
        try:
            # Generate vector file
            cmd = [
                "nv_createsamples",
                "-info", str(pos_file),
                "-vec", str(vec_file),
                "-w", str(width),
                "-h", str(height),
                "-num", str(pos_count),
            ]
            
            if verbose:
                print("Creating sample vector...")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                if verbose:
                    print(f"Warning: nv_createsamples failed: {result.stderr}")
                    print("Trying alternative method...")
                
                # Use Python-based sample creation
                self._create_samples_python(pos_images, vec_file, width, height)
        
        except FileNotFoundError:
            if verbose:
                print("nv_createsamples not found. Using Python fallback.")
            self._create_samples_python(pos_images, vec_file, width, height)
        
        # Train cascade
        cascade_file = cascade_dir / "cascade.xml"
        
        try:
            cmd = [
                "nv_traincascade",
                "-data", str(cascade_dir),
                "-vec", str(vec_file),
                "-bg", str(neg_file),
                "-numPos", str(int(pos_count * 0.9)),
                "-numNeg", str(neg_count),
                "-numStages", str(num_stages),
                "-w", str(width),
                "-h", str(height),
                "-featureType", cascade_type.upper(),
                "-minHitRate", str(min_hit_rate),
                "-maxFalseAlarmRate", str(max_false_alarm),
            ]
            
            if verbose:
                print("Training cascade (this may take a while)...")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if verbose:
                    print(f"Cascade trained successfully: {cascade_file}")
                
                return {
                    'cascade_path': str(cascade_file),
                    'type': cascade_type,
                    'stages': num_stages,
                    'width': width,
                    'height': height,
                    'positive_samples': pos_count,
                    'negative_samples': neg_count,
                }
            else:
                raise RuntimeError(f"Training failed: {result.stderr}")
                
        except FileNotFoundError:
            if verbose:
                print("nv_traincascade not found.")
                print("Please install neurova training tools.")
            
            return {
                'error': 'nv_traincascade not found',
                'cascade_path': None,
            }
    
    def fine_tune(
        self,
        dataset: Any,  # FaceDataset
        base_model_path: str,
        epochs: int = 10,
        learning_rate: float = 0.001,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Fine-tune a pre-trained model on new data.
        
        Args:
            dataset: FaceDataset with new training data.
            base_model_path: Path to base model.
            epochs: Number of fine-tuning epochs.
            learning_rate: Learning rate.
            verbose: Print progress.
        
        Returns:
            Fine-tuning history.
        """
        from .recognizer import FaceRecognizer
        
        if verbose:
            print(f"Fine-tuning model from {base_model_path}...")
        
        # Load base model
        self._model = FaceRecognizer()
        self._model.load(base_model_path)
        
        # Get new training data
        faces = dataset.train_images
        labels = [dataset.class_names[lbl] for lbl in dataset.train_labels]
        
        # Update model with new data
        start_time = self._get_time()
        self._model.update(faces, labels)
        tune_time = self._get_time() - start_time
        
        # Evaluate
        val_accuracy = 0.0
        if dataset.val_images:
            val_accuracy = self._evaluate_accuracy(
                self._model,
                dataset.val_images,
                dataset.val_labels,
                dataset.class_names
            )
        
        history = {
            'base_model': base_model_path,
            'new_samples': len(faces),
            'tune_time': tune_time,
            'val_accuracy': val_accuracy,
        }
        
        self._history.append(history)
        
        if verbose:
            print(f"Fine-tuning completed in {tune_time:.2f}s")
            if val_accuracy > 0:
                print(f"Validation accuracy: {val_accuracy:.2%}")
        
        return history
    
    def evaluate(
        self,
        dataset: Any,  # FaceDataset
        split: str = "test",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            dataset: FaceDataset with test data.
            split: Which split to evaluate ('train', 'val', 'test').
            verbose: Print results.
        
        Returns:
            Evaluation metrics.
        """
        if self._model is None:
            raise ValueError("No model trained. Call train_recognizer first.")
        
        # Get data for split
        if split == "train":
            images = dataset.train_images
            labels = dataset.train_labels
        elif split == "val":
            images = dataset.val_images
            labels = dataset.val_labels
        elif split == "test":
            images = dataset.test_images
            labels = dataset.test_labels
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not images:
            return {'error': f'No data in {split} split'}
        
        # Predict
        predictions = []
        confidences = []
        
        for face in images:
            pred, conf = self._model.predict(face)
            # Convert label back to index
            if pred in dataset.class_names:
                pred_idx = dataset.class_names.index(pred)
            else:
                pred_idx = -1
            predictions.append(pred_idx)
            confidences.append(conf)
        
        # Calculate metrics
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels)
        
        # Per-class accuracy
        class_accuracy = {}
        for i, name in enumerate(dataset.class_names):
            class_mask = [l == i for l in labels]
            class_correct = sum(
                p == l for p, l, m in zip(predictions, labels, class_mask) if m
            )
            class_total = sum(class_mask)
            if class_total > 0:
                class_accuracy[name] = class_correct / class_total
        
        # Confusion matrix
        n_classes = len(dataset.class_names)
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for p, l in zip(predictions, labels):
            if 0 <= p < n_classes and 0 <= l < n_classes:
                confusion[l, p] += 1
        
        results = {
            'split': split,
            'samples': len(labels),
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'confusion_matrix': confusion.tolist(),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
        }
        
        if verbose:
            print(f"\nEvaluation on {split} set:")
            print(f"  Samples: {len(labels)}")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Mean confidence: {np.mean(confidences):.4f}")
            print("\n  Per-class accuracy:")
            for name, acc in class_accuracy.items():
                print(f"    {name}: {acc:.2%}")
        
        return results
    
    def cross_validate(
        self,
        dataset: Any,  # FaceDataset
        k_folds: int = 5,
        method: str = "lbph",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            dataset: FaceDataset with all data.
            k_folds: Number of folds.
            method: Recognition method.
            verbose: Print progress.
        
        Returns:
            Cross-validation results.
        """
        from .recognizer import FaceRecognizer
        
        # Combine all images
        all_images = dataset.train_images + dataset.val_images + dataset.test_images
        all_labels = dataset.train_labels + dataset.val_labels + dataset.test_labels
        
        n_samples = len(all_images)
        fold_size = n_samples // k_folds
        
        fold_accuracies = []
        
        for fold in range(k_folds):
            if verbose:
                print(f"Fold {fold + 1}/{k_folds}...")
            
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else n_samples
            
            train_images = all_images[:val_start] + all_images[val_end:]
            train_labels = all_labels[:val_start] + all_labels[val_end:]
            val_images = all_images[val_start:val_end]
            val_labels = all_labels[val_start:val_end]
            
            # Train
            model = FaceRecognizer(method=method)
            labels_str = [dataset.class_names[l] for l in train_labels]
            model.train(train_images, labels_str)
            
            # Evaluate
            accuracy = self._evaluate_accuracy(
                model, val_images, val_labels, dataset.class_names
            )
            fold_accuracies.append(accuracy)
            
            if verbose:
                print(f"  Fold {fold + 1} accuracy: {accuracy:.2%}")
        
        results = {
            'k_folds': k_folds,
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
        }
        
        if verbose:
            print(f"\nCross-validation results:")
            print(f"  Mean accuracy: {results['mean_accuracy']:.2%} ± {results['std_accuracy']:.2%}")
        
        return results
    
    def save(self, name: str) -> str:
        """
        Save the trained model.
        
        Args:
            name: Model name.
        
        Returns:
            Path to saved model.
        """
        if self._model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_path = self.output_dir / f"{name}.model"
        self._model.save(str(model_path))
        
        # Save metadata
        meta_path = self.output_dir / f"{name}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'name': name,
                'method': self._model.method,
                'history': self._history,
            }, f, indent=2)
        
        return str(model_path)
    
    def load(self, path: str) -> None:
        """Load a trained model."""
        from .recognizer import FaceRecognizer
        
        # Load metadata
        meta_path = Path(path).with_suffix('.meta.json')
        method = 'lbph'
        
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                method = meta.get('method', 'lbph')
                self._history = meta.get('history', [])
        
        self._model = FaceRecognizer(method=method)
        self._model.load(path)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self._history.copy()
    
    def _evaluate_accuracy(
        self,
        model: Any,
        images: List[np.ndarray],
        labels: List[int],
        class_names: List[str],
    ) -> float:
        """Calculate accuracy on a dataset."""
        if not images:
            return 0.0
        
        correct = 0
        for face, true_label in zip(images, labels):
            pred, _ = model.predict(face)
            if pred == class_names[true_label]:
                correct += 1
        
        return correct / len(labels)
    
    def _get_time(self) -> float:
        """Get current time."""
        import time
        return time.time()
    
    def _create_samples_python(
        self,
        images: List[Path],
        output_path: Path,
        width: int,
        height: int,
    ) -> None:
        """Create samples using Python (fallback method)."""
        from PIL import Image
        
        samples = []
        for img_path in images:
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize((width, height))
                samples.append(np.array(img))
            except Exception:
                continue
        
        # Save as pickle (simplified format)
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(samples, f)


def train_face_detector(
    positive_dir: str,
    negative_dir: str,
    output_dir: str = "./models",
    cascade_type: str = "haar",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train a face detector.
    
    Args:
        positive_dir: Directory with face images.
        negative_dir: Directory with non-face images.
        output_dir: Where to save the model.
        cascade_type: 'haar' or 'lbp'.
        **kwargs: Additional arguments for FaceTrainer.train_detector.
    
    Returns:
        Training results.
    """
    trainer = FaceTrainer(output_dir=output_dir)
    return trainer.train_detector(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        cascade_type=cascade_type,
        **kwargs,
    )


def train_face_recognizer(
    dataset: Any,
    method: str = "lbph",
    output_dir: str = "./models",
    model_name: str = "face_recognizer",
    **kwargs,
) -> str:
    """
    Convenience function to train a face recognizer.
    
    Args:
        dataset: FaceDataset with training data.
        method: Recognition method.
        output_dir: Where to save the model.
        model_name: Name for the saved model.
        **kwargs: Additional arguments for FaceTrainer.train_recognizer.
    
    Returns:
        Path to saved model.
    """
    trainer = FaceTrainer(output_dir=output_dir)
    trainer.train_recognizer(dataset, method=method, **kwargs)
    return trainer.save(model_name)
