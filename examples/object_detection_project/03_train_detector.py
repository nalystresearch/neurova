#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Step 3: Train Object Detector
==============================

This script trains an object detector using various methods:
1. HOG + SVM (traditional ML)
2. Cascade classifier training preparation
3. Template matching model
4. Sliding window classifier

Usage:
    python 03_train_detector.py
    python 03_train_detector.py --method hog
    python 03_train_detector.py --method cascade
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    DATA_DIR, TRAIN_DIR, MODELS_DIR, REPORTS_DIR,
    TRAINING_CONFIG, DETECTION_SETTINGS, CLASS_NAMES
)

# import neurova
try:
    from neurova import nv
    from neurova.detection import (
        detect_hog, compute_hog_features,
        detect_template, detect_sliding_window
    )
    from neurova.ml import (
        train_svm, train_random_forest,
        normalize_features, PCA
    )
    NEUROVA_AVAILABLE = True
except ImportError:
    NEUROVA_AVAILABLE = False
    print("  Neurova not available, using fallback implementations")


class HOGSVMDetector:
    """
    HOG + SVM based object detector.
    """
    
    def __init__(self, window_size=(64, 128), cell_size=8, block_size=2, nbins=9):
        self.window_size = window_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins
        self.model = None
        self.scaler = None
    
    def compute_hog(self, image):
        """Compute HOG features for an image."""
        from PIL import Image
        
# ensure correct size
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        img = img.resize(self.window_size[::-1], Image.LANCZOS)
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img)
        
        if NEUROVA_AVAILABLE:
            features = compute_hog_features(
                img_array,
                cell_size=self.cell_size,
                block_size=self.block_size,
                nbins=self.nbins
            )
        else:
            features = self._compute_hog_manual(img_array)
        
        return features
    
    def _compute_hog_manual(self, image):
        """Manual HOG computation fallback."""
# compute gradients
        gx = np.zeros_like(image, dtype=np.float64)
        gy = np.zeros_like(image, dtype=np.float64)
        
        gx[:, 1:-1] = image[:, 2:].astype(np.float64) - image[:, :-2].astype(np.float64)
        gy[1:-1, :] = image[2:, :].astype(np.float64) - image[:-2, :].astype(np.float64)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi
        orientation = (orientation + 180) % 180  # 0-180 range
        
# compute cells
        h, w = image.shape
        cells_h = h // self.cell_size
        cells_w = w // self.cell_size
        
        histograms = np.zeros((cells_h, cells_w, self.nbins))
        bin_width = 180 / self.nbins
        
        for i in range(cells_h):
            for j in range(cells_w):
                y1, y2 = i * self.cell_size, (i + 1) * self.cell_size
                x1, x2 = j * self.cell_size, (j + 1) * self.cell_size
                
                cell_mag = magnitude[y1:y2, x1:x2]
                cell_ori = orientation[y1:y2, x1:x2]
                
                for b in range(self.nbins):
                    bin_start = b * bin_width
                    bin_end = (b + 1) * bin_width
                    mask = (cell_ori >= bin_start) & (cell_ori < bin_end)
                    histograms[i, j, b] = np.sum(cell_mag[mask])
        
# block normalization
        blocks_h = cells_h - self.block_size + 1
        blocks_w = cells_w - self.block_size + 1
        
        features = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = histograms[i:i+self.block_size, j:j+self.block_size, :]
                block_vec = block.flatten()
                norm = np.linalg.norm(block_vec) + 1e-6
                features.extend(block_vec / norm)
        
        return np.array(features)
    
    def train(self, positive_samples, negative_samples):
        """Train the SVM model."""
        print("\n Training HOG + SVM detector...")
        
# compute features
        print("   Computing HOG features...")
        X_pos = [self.compute_hog(img) for img in positive_samples]
        X_neg = [self.compute_hog(img) for img in negative_samples]
        
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))
        
        print(f"   Positive samples: {len(X_pos)}")
        print(f"   Negative samples: {len(X_neg)}")
        print(f"   Feature dimension: {X.shape[1]}")
        
# normalize
        self.scaler = {
            'mean': X.mean(axis=0),
            'std': X.std(axis=0) + 1e-6
        }
        X_normalized = (X - self.scaler['mean']) / self.scaler['std']
        
# train svm
        print("   Training SVM...")
        if NEUROVA_AVAILABLE:
            self.model = train_svm(X_normalized, y, kernel='linear', C=1.0)
        else:
            self.model = self._train_svm_simple(X_normalized, y)
        
# evaluate on training data
        y_pred = self.predict_proba(X_normalized)
        accuracy = np.mean((y_pred > 0.5) == y)
        print(f"   Training accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def _train_svm_simple(self, X, y):
        """Simple linear SVM using gradient descent."""
        n_samples, n_features = X.shape
        
# initialize weights
        w = np.zeros(n_features)
        b = 0
        
# hyperparameters
        C = 1.0
        lr = 0.01
        epochs = 100
        
        # Convert labels to -1, 1
        y_svm = 2 * y - 1
        
        for _ in range(epochs):
            for i in range(n_samples):
                if y_svm[i] * (np.dot(X[i], w) + b) < 1:
                    w = w - lr * (2 * w / n_samples - C * y_svm[i] * X[i])
                    b = b + lr * C * y_svm[i]
                else:
                    w = w - lr * (2 * w / n_samples)
        
        return {'weights': w, 'bias': b}
    
    def predict_proba(self, X):
        """Predict probability scores."""
        if isinstance(X, list):
            X = np.array(X)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        if NEUROVA_AVAILABLE and hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X)
        else:
            w = self.model['weights']
            b = self.model['bias']
            scores = np.dot(X, w) + b
        
# convert to probability with sigmoid
        return 1 / (1 + np.exp(-scores))
    
    def detect(self, image, scale_factor=1.2, min_size=None, max_size=None):
        """Detect objects using sliding window."""
        from PIL import Image
        
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img
        
        detections = []
        width, height = img.size
        win_w, win_h = self.window_size
        
# multi-scale detection
        scale = 1.0
        while True:
            scaled_w = int(width / scale)
            scaled_h = int(height / scale)
            
            if min_size and min(scaled_w, scaled_h) < min_size:
                break
            if scaled_w < win_w or scaled_h < win_h:
                break
            
# resize image
            scaled_img = img_gray.resize((scaled_w, scaled_h), Image.LANCZOS)
            scaled_array = np.array(scaled_img)
            
# sliding window
            stride = DETECTION_SETTINGS['stride']
            for y in range(0, scaled_h - win_h + 1, stride):
                for x in range(0, scaled_w - win_w + 1, stride):
# extract window
                    window = scaled_array[y:y+win_h, x:x+win_w]
                    
# compute features
                    features = self.compute_hog(window)
                    features = (features - self.scaler['mean']) / self.scaler['std']
                    
# predict
                    score = self.predict_proba(features)[0]
                    
                    if score > DETECTION_SETTINGS['confidence_threshold']:
# scale back to original coordinates
                        orig_x = int(x * scale)
                        orig_y = int(y * scale)
                        orig_w = int(win_w * scale)
                        orig_h = int(win_h * scale)
                        
                        detections.append({
                            'bbox': [orig_x, orig_y, orig_w, orig_h],
                            'confidence': float(score),
                            'class': 'object'
                        })
            
            scale *= scale_factor
            if max_size and scale * min(win_w, win_h) > max_size:
                break
        
# non-maximum suppression
        detections = self._nms(detections, threshold=0.3)
        
        return detections
    
    def _nms(self, detections, threshold=0.3):
        """Non-maximum suppression."""
        if not detections:
            return []
        
# sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            remaining = []
            for det in detections:
                iou = self._compute_iou(best['bbox'], det['bbox'])
                if iou < threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
# intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
# union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def save(self, path):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'window_size': self.window_size,
                'cell_size': self.cell_size,
                'block_size': self.block_size,
                'nbins': self.nbins
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        detector = cls(
            window_size=data['window_size'],
            cell_size=data['cell_size'],
            block_size=data['block_size'],
            nbins=data['nbins']
        )
        detector.model = data['model']
        detector.scaler = data['scaler']
        
        return detector


class TemplateMatchingDetector:
    """
    Template matching based detector.
    """
    
    def __init__(self):
        self.templates = []
    
    def add_template(self, image, class_name='object'):
        """Add a template for matching."""
        if isinstance(image, np.ndarray):
            template = image
        else:
            template = np.array(image)
        
        if len(template.shape) == 3:
            template = np.mean(template, axis=2).astype(np.uint8)
        
        self.templates.append({
            'image': template,
            'class': class_name
        })
    
    def detect(self, image, threshold=0.7):
        """Detect objects using template matching."""
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = np.array(image)
        
        if len(img.shape) == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        
        detections = []
        
        for template in self.templates:
            tmpl = template['image']
            
            if NEUROVA_AVAILABLE:
                matches = detect_template(
                    img, tmpl,
                    method='ncc',
                    threshold=threshold
                )
            else:
                matches = self._template_match(img, tmpl, threshold)
            
            for match in matches:
                detections.append({
                    'bbox': match['bbox'],
                    'confidence': match.get('score', 1.0),
                    'class': template['class']
                })
        
        return detections
    
    def _template_match(self, image, template, threshold):
        """Simple template matching using NCC."""
        h, w = image.shape
        th, tw = template.shape
        
        matches = []
        
# normalize template
        tmpl_mean = template.mean()
        tmpl_std = template.std() + 1e-6
        tmpl_norm = (template - tmpl_mean) / tmpl_std
        
        for y in range(0, h - th + 1, 8):
            for x in range(0, w - tw + 1, 8):
# extract region
                region = image[y:y+th, x:x+tw].astype(np.float64)
                
# normalize
                reg_mean = region.mean()
                reg_std = region.std() + 1e-6
                reg_norm = (region - reg_mean) / reg_std
                
# ncc score
                ncc = np.mean(reg_norm * tmpl_norm)
                
                if ncc > threshold:
                    matches.append({
                        'bbox': [x, y, tw, th],
                        'score': float(ncc)
                    })
        
        return matches
    
    def save(self, path):
        """Save templates to file."""
        with open(path, 'wb') as f:
            pickle.dump({'templates': self.templates}, f)
    
    @classmethod
    def load(cls, path):
        """Load templates from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        detector = cls()
        detector.templates = data['templates']
        return detector


def load_training_samples(samples_dir: Path):
    """Load positive and negative samples."""
    from PIL import Image
    
    pos_dir = samples_dir / "positive"
    neg_dir = samples_dir / "negative"
    
    positive = []
    negative = []
    
    if pos_dir.exists():
        for img_path in pos_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = Image.open(img_path)
                positive.append(img)
    
    if neg_dir.exists():
        for img_path in neg_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = Image.open(img_path)
                negative.append(img)
    
    return positive, negative


def train_detector(method: str = 'hog'):
    """
    Main function to train detector.
    """
    print("=" * 60)
    print("STEP 3: TRAIN OBJECT DETECTOR")
    print("=" * 60)
    print(f"\n Method: {method.upper()}")
    
    samples_dir = DATA_DIR / "samples"
    
    if method == 'hog':
# load samples
        print("\n Loading training samples...")
        positive, negative = load_training_samples(samples_dir)
        
        if not positive:
            print("\n No positive samples found!")
            print("\n First run: python 02_prepare_dataset.py --extract-samples")
            return None
        
        print(f"   Positive samples: {len(positive)}")
        print(f"   Negative samples: {len(negative)}")
        
        if len(negative) < len(positive):
            print("\n  Warning: More positive than negative samples")
            print("   Consider adding more negative samples")
        
# create and train detector
        detector = HOGSVMDetector(
            window_size=(64, 64),
            cell_size=8,
            block_size=2,
            nbins=9
        )
        
        accuracy = detector.train(positive, negative)
        
# save model
        model_path = MODELS_DIR / "hog_svm_detector.pkl"
        detector.save(model_path)
        print(f"\n Model saved: {model_path}")
        
        return detector
    
    elif method == 'template':
# load templates
        templates_dir = DATA_DIR / "templates"
        
        if not templates_dir.exists():
            print("\n No templates directory found!")
            print(f"\n Add template images to: {templates_dir}")
            return None
        
        detector = TemplateMatchingDetector()
        
        from PIL import Image
        for img_path in templates_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = Image.open(img_path)
                class_name = img_path.stem.split('_')[0]
                detector.add_template(img, class_name)
                print(f"   Added template: {img_path.name} -> {class_name}")
        
        if not detector.templates:
            print("\n No templates loaded!")
            return None
        
# save model
        model_path = MODELS_DIR / "template_detector.pkl"
        detector.save(model_path)
        print(f"\n Model saved: {model_path}")
        
        return detector
    
    elif method == 'cascade':
# prepare cascade training data
        print("\n Preparing cascade training data...")
        
        pos_file = MODELS_DIR / "positives.txt"
        neg_file = MODELS_DIR / "negatives.txt"
        
        pos_dir = samples_dir / "positive"
        neg_dir = samples_dir / "negative"
        
        if not pos_dir.exists():
            print("\n No positive samples found!")
            print("\n First run: python 02_prepare_dataset.py --extract-samples")
            return None
        
# write positive samples list
        with open(pos_file, 'w') as f:
            for img_path in pos_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    from PIL import Image
                    img = Image.open(img_path)
                    w, h = img.size
                    f.write(f"{img_path} 1 0 0 {w} {h}\n")
        
# write negative samples list
        with open(neg_file, 'w') as f:
            for img_path in neg_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    f.write(f"{img_path}\n")
        
        print(f"\n Training files created:")
        print(f"   Positives: {pos_file}")
        print(f"   Negatives: {neg_file}")
        print("\n To train cascade, use OpenCV's opencv_traincascade tool:")
        print(f"   opencv_traincascade -data {MODELS_DIR}/cascade -vec positives.vec -bg {neg_file} ...")
        
        return None
    
    else:
        print(f"\n Unknown method: {method}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train object detector")
    parser.add_argument("--method", type=str, default="hog",
                       choices=['hog', 'template', 'cascade'],
                       help="Detection method to train")
    
    args = parser.parse_args()
    
    detector = train_detector(method=args.method)
    
    if detector:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print("\n Detector trained successfully!")
        print("\n Next step: python 04_evaluate_detector.py")


if __name__ == "__main__":
    main()
