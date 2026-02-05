# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Face Recognition using multiple methods.

Supports LBPH, EigenFace, FisherFace, and embedding-based recognition.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class FaceRecognizer:
    """
    Unified face recognizer supporting multiple recognition methods.
    
    Args:
        method: Recognition method ('lbph', 'eigen', 'fisher', 'embedding').
        model_path: Path to pre-trained model.
        threshold: Recognition threshold (lower = stricter).
        
    Example:
        >>> recognizer = FaceRecognizer(method='lbph')
        >>> recognizer.train(faces, labels)
        >>> label, confidence = recognizer.predict(test_face)
        >>> print(f"Person: {label}, Confidence: {confidence:.2f}")
    """
    
    def __init__(
        self,
        method: str = "lbph",
        model_path: Optional[str] = None,
        threshold: float = 100.0,
    ):
        self.method = method.lower()
        self.threshold = threshold
        self.model_path = model_path
        self._recognizer = None
        self._labels: Dict[int, str] = {}
        
        if self.method == "lbph":
            self._recognizer = LBPHRecognizer(threshold)
        elif self.method == "eigen":
            self._recognizer = EigenFaceRecognizer(threshold)
        elif self.method == "fisher":
            self._recognizer = FisherFaceRecognizer(threshold)
        elif self.method == "embedding":
            self._recognizer = EmbeddingRecognizer(model_path, threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(
        self,
        faces: List[np.ndarray],
        labels: Union[List[int], List[str]],
        augment: bool = False,
    ) -> None:
        """
        Train the recognizer with face images and labels.
        
        Args:
            faces: List of face images (grayscale or RGB).
            labels: List of labels (integers or strings).
            augment: Apply data augmentation.
        """
        # Convert string labels to integers
        if labels and isinstance(labels[0], str):
            unique_labels = list(set(labels))
            self._labels = {i: lbl for i, lbl in enumerate(unique_labels)}
            int_labels = [unique_labels.index(lbl) for lbl in labels]
        else:
            int_labels = list(labels)
        
        # Prepare faces
        processed_faces = []
        for face in faces:
            # Convert to grayscale if needed
            if len(face.shape) == 3:
                gray = np.mean(face, axis=2).astype(np.uint8)
            else:
                gray = face.astype(np.uint8)
            processed_faces.append(gray)
        
        if augment:
            processed_faces, int_labels = self._augment_data(processed_faces, int_labels)
        
        self._recognizer.train(processed_faces, int_labels)
    
    def predict(
        self,
        face: np.ndarray,
        return_all: bool = False,
    ) -> Union[Tuple[Union[int, str], float], List[Tuple[Union[int, str], float]]]:
        """
        Predict the identity of a face.
        
        Args:
            face: Face image to recognize.
            return_all: Return all predictions with scores.
        
        Returns:
            (label, confidence) or list of predictions.
        """
        # Convert to grayscale
        if len(face.shape) == 3:
            gray = np.mean(face, axis=2).astype(np.uint8)
        else:
            gray = face.astype(np.uint8)
        
        if return_all:
            predictions = self._recognizer.predict_all(gray)
            # Convert integer labels to strings if available
            return [
                (self._labels.get(lbl, lbl), conf) 
                for lbl, conf in predictions
            ]
        
        label, confidence = self._recognizer.predict(gray)
        
        # Convert integer label to string if available
        if self._labels and label in self._labels:
            label = self._labels[label]
        
        return label, confidence
    
    def update(
        self,
        faces: List[np.ndarray],
        labels: Union[List[int], List[str]],
    ) -> None:
        """
        Update the recognizer with new faces (incremental learning).
        
        Args:
            faces: New face images.
            labels: Labels for new faces.
        """
        # Convert string labels
        if labels and isinstance(labels[0], str):
            int_labels = []
            for lbl in labels:
                if lbl not in self._labels.values():
                    new_id = max(self._labels.keys(), default=-1) + 1
                    self._labels[new_id] = lbl
                int_labels.append(
                    next(k for k, v in self._labels.items() if v == lbl)
                )
        else:
            int_labels = list(labels)
        
        processed_faces = []
        for face in faces:
            if len(face.shape) == 3:
                gray = np.mean(face, axis=2).astype(np.uint8)
            else:
                gray = face.astype(np.uint8)
            processed_faces.append(gray)
        
        self._recognizer.update(processed_faces, int_labels)
    
    def save(self, path: str) -> None:
        """Save the recognizer model to file."""
        self._recognizer.save(path)
        
        # Save label mapping
        label_path = path + ".labels"
        with open(label_path, 'wb') as f:
            pickle.dump(self._labels, f)
    
    def load(self, path: str) -> None:
        """Load the recognizer model from file."""
        self._recognizer.load(path)
        
        # Load label mapping
        label_path = path + ".labels"
        if os.path.exists(label_path):
            with open(label_path, 'rb') as f:
                self._labels = pickle.load(f)
    
    def get_labels(self) -> Dict[int, str]:
        """Get the label mapping."""
        return self._labels.copy()
    
    def _augment_data(
        self,
        faces: List[np.ndarray],
        labels: List[int],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Apply data augmentation to faces."""
        augmented_faces = list(faces)
        augmented_labels = list(labels)
        
        for face, label in zip(faces, labels):
            # Horizontal flip
            flipped = np.fliplr(face)
            augmented_faces.append(flipped)
            augmented_labels.append(label)
            
            # Brightness variations
            brighter = np.clip(face * 1.2, 0, 255).astype(np.uint8)
            darker = np.clip(face * 0.8, 0, 255).astype(np.uint8)
            augmented_faces.extend([brighter, darker])
            augmented_labels.extend([label, label])
            
            # Slight rotation (requires scipy)
            try:
                from scipy.ndimage import rotate
                for angle in [-5, 5]:
                    rotated = rotate(face, angle, reshape=False, mode='constant')
                    augmented_faces.append(rotated.astype(np.uint8))
                    augmented_labels.append(label)
            except ImportError:
                pass
        
        return augmented_faces, augmented_labels


class LBPHRecognizer:
    """Local Binary Patterns Histograms face recognizer.
    
    Uses Neurova's pure-Python LBP implementation by default.
    Falls back to cv2 if available for better performance.
    """
    
    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold
        self._recognizer = None
        self._faces: List[np.ndarray] = []
        self._labels: List[int] = []
        self._histograms: List[np.ndarray] = []
        self._use_cv2 = False
        
        # Try cv2 for performance (optional)
        try:
            import cv2
            self._recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8,
                threshold=threshold
            )
            self._use_cv2 = True
        except (ImportError, AttributeError):
            # Use Neurova's pure-Python implementation (default)
            self._use_cv2 = False
    
    def train(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """Train the LBPH recognizer."""
        self._faces = list(faces)
        self._labels = list(labels)
        
        if self._use_cv2 and self._recognizer:
            self._recognizer.train(faces, np.array(labels))
        else:
            # Custom LBPH implementation
            self._histograms = []
            for face in faces:
                hist = self._compute_lbph(face)
                self._histograms.append(hist)
    
    def predict(self, face: np.ndarray) -> Tuple[int, float]:
        """Predict identity using LBPH."""
        if self._use_cv2 and self._recognizer:
            label, confidence = self._recognizer.predict(face)
            return int(label), float(confidence)
        
        # Custom implementation
        hist = self._compute_lbph(face)
        
        min_dist = float('inf')
        best_label = -1
        
        for i, ref_hist in enumerate(self._histograms):
            dist = np.sum((hist - ref_hist) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_label = self._labels[i]
        
        return best_label, min_dist
    
    def predict_all(self, face: np.ndarray) -> List[Tuple[int, float]]:
        """Get all predictions with scores."""
        if self._use_cv2 and self._recognizer:
            # cv2 doesn't support this, so compute manually
            pass
        
        hist = self._compute_lbph(face)
        
        predictions = []
        for i, ref_hist in enumerate(self._histograms):
            dist = np.sum((hist - ref_hist) ** 2)
            predictions.append((self._labels[i], dist))
        
        return sorted(predictions, key=lambda x: x[1])
    
    def update(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """Update with new faces."""
        self._faces.extend(faces)
        self._labels.extend(labels)
        
        if self._use_cv2 and self._recognizer:
            self._recognizer.update(faces, np.array(labels))
        else:
            for face in faces:
                hist = self._compute_lbph(face)
                self._histograms.append(hist)
    
    def save(self, path: str) -> None:
        """Save model."""
        if self._use_cv2 and self._recognizer:
            self._recognizer.write(path)
        else:
            with open(path, 'wb') as f:
                pickle.dump({
                    'histograms': self._histograms,
                    'labels': self._labels
                }, f)
    
    def load(self, path: str) -> None:
        """Load model."""
        if self._use_cv2 and self._recognizer:
            self._recognizer.read(path)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self._histograms = data['histograms']
                self._labels = data['labels']
    
    def _compute_lbph(self, image: np.ndarray) -> np.ndarray:
        """Compute LBP histogram for an image."""
        h, w = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        
        return hist


class EigenFaceRecognizer:
    """EigenFace (PCA-based) face recognizer."""
    
    def __init__(self, threshold: float = 5000.0):
        self.threshold = threshold
        self._recognizer = None
        self._mean_face = None
        self._eigenvectors = None
        self._projections = None
        self._labels = None
        
        try:
            import cv2
            self._recognizer = cv2.face.EigenFaceRecognizer_create(
                num_components=80,
                threshold=threshold
            )
            self._use_cv2 = True
        except (ImportError, AttributeError):
            self._use_cv2 = False
    
    def train(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """Train the EigenFace recognizer."""
        self._labels = np.array(labels)
        
        # Ensure all faces are same size
        target_size = faces[0].shape
        resized_faces = []
        for face in faces:
            if face.shape != target_size:
                from PIL import Image
                pil_img = Image.fromarray(face)
                pil_img = pil_img.resize((target_size[1], target_size[0]))
                face = np.array(pil_img)
            resized_faces.append(face)
        
        if self._use_cv2 and self._recognizer:
            self._recognizer.train(resized_faces, self._labels)
        else:
            # Custom PCA implementation
            flat_faces = np.array([f.flatten() for f in resized_faces])
            self._mean_face = np.mean(flat_faces, axis=0)
            centered = flat_faces - self._mean_face
            
            # Compute covariance and eigenvectors
            cov = np.dot(centered, centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Keep top k components
            k = min(80, len(faces) - 1)
            self._eigenvectors = np.dot(centered.T, eigenvectors[:, :k])
            self._eigenvectors /= np.linalg.norm(self._eigenvectors, axis=0)
            
            # Project training faces
            self._projections = np.dot(centered, self._eigenvectors)
    
    def predict(self, face: np.ndarray) -> Tuple[int, float]:
        """Predict identity using EigenFace."""
        if self._use_cv2 and self._recognizer:
            label, confidence = self._recognizer.predict(face)
            return int(label), float(confidence)
        
        # Custom implementation
        flat_face = face.flatten()
        centered = flat_face - self._mean_face
        projection = np.dot(centered, self._eigenvectors)
        
        # Find nearest neighbor
        distances = np.linalg.norm(self._projections - projection, axis=1)
        min_idx = np.argmin(distances)
        
        return int(self._labels[min_idx]), float(distances[min_idx])
    
    def predict_all(self, face: np.ndarray) -> List[Tuple[int, float]]:
        """Get all predictions with scores."""
        flat_face = face.flatten()
        centered = flat_face - self._mean_face
        projection = np.dot(centered, self._eigenvectors)
        
        distances = np.linalg.norm(self._projections - projection, axis=1)
        
        predictions = list(zip(self._labels.tolist(), distances.tolist()))
        return sorted(predictions, key=lambda x: x[1])
    
    def update(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """Update with new faces (retrain)."""
        # EigenFace requires full retrain
        all_faces = list(self._projections) + faces
        all_labels = list(self._labels) + labels
        self.train(all_faces, all_labels)
    
    def save(self, path: str) -> None:
        """Save model."""
        if self._use_cv2 and self._recognizer:
            self._recognizer.write(path)
        else:
            with open(path, 'wb') as f:
                pickle.dump({
                    'mean_face': self._mean_face,
                    'eigenvectors': self._eigenvectors,
                    'projections': self._projections,
                    'labels': self._labels
                }, f)
    
    def load(self, path: str) -> None:
        """Load model."""
        if self._use_cv2 and self._recognizer:
            self._recognizer.read(path)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self._mean_face = data['mean_face']
                self._eigenvectors = data['eigenvectors']
                self._projections = data['projections']
                self._labels = data['labels']


class FisherFaceRecognizer:
    """FisherFace (LDA-based) face recognizer."""
    
    def __init__(self, threshold: float = 5000.0):
        self.threshold = threshold
        self._recognizer = None
        
        try:
            import cv2
            self._recognizer = cv2.face.FisherFaceRecognizer_create(
                num_components=0,
                threshold=threshold
            )
            self._use_cv2 = True
        except (ImportError, AttributeError):
            self._use_cv2 = False
    
    def train(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """Train the FisherFace recognizer."""
        if self._use_cv2 and self._recognizer:
            # Ensure all faces are same size
            target_size = faces[0].shape
            resized_faces = []
            for face in faces:
                if face.shape != target_size:
                    from PIL import Image
                    pil_img = Image.fromarray(face)
                    pil_img = pil_img.resize((target_size[1], target_size[0]))
                    face = np.array(pil_img)
                resized_faces.append(face)
            
            self._recognizer.train(resized_faces, np.array(labels))
    
    def predict(self, face: np.ndarray) -> Tuple[int, float]:
        """Predict identity using FisherFace."""
        if self._use_cv2 and self._recognizer:
            label, confidence = self._recognizer.predict(face)
            return int(label), float(confidence)
        return -1, float('inf')
    
    def predict_all(self, face: np.ndarray) -> List[Tuple[int, float]]:
        """Get all predictions with scores."""
        label, conf = self.predict(face)
        return [(label, conf)]
    
    def update(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """Update with new faces."""
        pass  # FisherFace requires full retrain
    
    def save(self, path: str) -> None:
        """Save model."""
        if self._use_cv2 and self._recognizer:
            self._recognizer.write(path)
    
    def load(self, path: str) -> None:
        """Load model."""
        if self._use_cv2 and self._recognizer:
            self._recognizer.read(path)


class EmbeddingRecognizer:
    """Embedding-based face recognizer using pure Python feature extraction."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.6,
    ):
        self.threshold = threshold
        self.model_path = model_path
        self._embeddings: List[np.ndarray] = []
        self._labels: List[int] = []
    
    def train(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """Train by computing embeddings for faces."""
        self._labels = list(labels)
        self._embeddings = []
        
        for face in faces:
            embedding = self._compute_embedding(face)
            self._embeddings.append(embedding)
    
    def _compute_embedding(self, face: np.ndarray) -> np.ndarray:
        """
        Compute face embedding using pure Python feature extraction.
        
        Combines multiple feature types:
        - HOG-like gradient features
        - LBP-like texture features
        - Spatial histogram features
        """
        from PIL import Image
        
        # Resize to standard size
        pil_img = Image.fromarray(face)
        pil_img = pil_img.resize((64, 64))
        img = np.array(pil_img).astype(np.float32)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img
        
        gray = gray / 255.0
        
        features = []
        
        # Feature 1: Gradient histogram (HOG-like)
        gx = np.diff(gray, axis=1, prepend=0)
        gy = np.diff(gray, axis=0, prepend=0)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        # Compute histogram of gradients in 4x4 cells
        cell_h, cell_w = 16, 16
        for i in range(0, 64, cell_h):
            for j in range(0, 64, cell_w):
                cell_mag = mag[i:i+cell_h, j:j+cell_w].flatten()
                cell_angle = angle[i:i+cell_h, j:j+cell_w].flatten()
                hist, _ = np.histogram(cell_angle, bins=9, range=(-np.pi, np.pi), weights=cell_mag)
                features.extend(hist / (np.sum(hist) + 1e-7))
        
        # Feature 2: LBP-like texture (simplified)
        lbp = np.zeros_like(gray)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                shifted = np.roll(np.roll(gray, di, axis=0), dj, axis=1)
                lbp += (shifted > gray).astype(float)
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=9, range=(0, 9))
        features.extend(lbp_hist / (np.sum(lbp_hist) + 1e-7))
        
        # Feature 3: Spatial histogram
        for i in range(0, 64, 16):
            for j in range(0, 64, 16):
                cell = gray[i:i+16, j:j+16]
                features.extend([np.mean(cell), np.std(cell)])
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, face: np.ndarray) -> Tuple[int, float]:
        """Predict identity using embeddings."""
        embedding = self._compute_embedding(face)
        
        min_dist = float('inf')
        best_label = -1
        
        for i, ref_emb in enumerate(self._embeddings):
            dist = 1 - np.dot(embedding, ref_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_emb) + 1e-7
            )
            if dist < min_dist:
                min_dist = dist
                best_label = self._labels[i]
        
        return best_label, min_dist
    
    def predict_all(self, face: np.ndarray) -> List[Tuple[int, float]]:
        """Get all predictions with scores."""
        embedding = self._compute_embedding(face)
        
        predictions = []
        for i, ref_emb in enumerate(self._embeddings):
            dist = 1 - np.dot(embedding, ref_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_emb) + 1e-7
            )
            predictions.append((self._labels[i], dist))
        
        return sorted(predictions, key=lambda x: x[1])
    
    def update(self, faces: List[np.ndarray], labels: List[int]) -> None:
        """Update with new faces."""
        for face, label in zip(faces, labels):
            embedding = self._compute_embedding(face)
            self._embeddings.append(embedding)
            self._labels.append(label)
    
    def save(self, path: str) -> None:
        """Save embeddings."""
        with open(path, 'wb') as f:
            pickle.dump({
                'embeddings': self._embeddings,
                'labels': self._labels
            }, f)
    
    def load(self, path: str) -> None:
        """Load embeddings."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._embeddings = data['embeddings']
            self._labels = data['labels']
