#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Step 3: Train Face Recognition Model
======================================

This script trains the face recognition model using:
1. Neurova's FaceTrainer
2. LBPH, EigenFace, or FisherFace algorithm
3. Data augmentation (optional)

Usage:
    python 03_train_model.py
    python 03_train_model.py --method lbph
    python 03_train_model.py --method eigenface --no-augment
"""

import os
import sys
import json
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    TRAIN_DIR, MODELS_DIR, REPORTS_DIR,
    RECOGNITION_CONFIG, TRAINING_CONFIG
)


def load_training_data(train_dir: Path, face_size: tuple = (100, 100)):
    """
    Load training data from directory.
    
    Args:
        train_dir: Path to training directory
        face_size: Target face size
    
    Returns:
        tuple: (faces, labels, label_names)
    """
    from PIL import Image
    
    faces = []
    labels = []
    label_names = {}
    current_label = 0
    
    for person_dir in sorted(train_dir.iterdir()):
        if person_dir.is_dir():
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            
            if not images:
                continue
            
            label_names[current_label] = person_dir.name
            
            for img_path in images:
                try:
# load image
                    img = Image.open(img_path).convert('L')  # Grayscale
                    
# resize
                    img = img.resize(face_size, Image.LANCZOS)
                    
# convert to numpy
                    face = np.array(img)
                    
                    faces.append(face)
                    labels.append(current_label)
                    
                except Exception as e:
                    print(f"    Error loading {img_path}: {e}")
            
            current_label += 1
    
    return np.array(faces), np.array(labels), label_names


def augment_data(faces, labels, config):
    """
    Apply data augmentation.
    
    Args:
        faces: Array of face images
        labels: Array of labels
        config: Augmentation config
    
    Returns:
        tuple: (augmented_faces, augmented_labels)
    """
    aug_faces = list(faces)
    aug_labels = list(labels)
    
    for face, label in zip(faces, labels):
# horizontal flip
        if config.get('augment_flip', True):
            flipped = np.fliplr(face)
            aug_faces.append(flipped)
            aug_labels.append(label)
        
# brightness adjustment
        if config.get('augment_brightness', True):
# slightly brighter
            bright = np.clip(face.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
            aug_faces.append(bright)
            aug_labels.append(label)
            
# slightly darker
            dark = np.clip(face.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
            aug_faces.append(dark)
            aug_labels.append(label)
        
# add noise
        if config.get('augment_noise', False):
            noise = np.random.randn(*face.shape) * 5
            noisy = np.clip(face.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            aug_faces.append(noisy)
            aug_labels.append(label)
    
    return np.array(aug_faces), np.array(aug_labels)


def train_model(method: str = 'lbph', augment: bool = True):
    """
    Train face recognition model.
    
    Args:
        method: Recognition method (lbph, eigenface, fisherface)
        augment: Whether to apply data augmentation
    """
    print("=" * 60)
    print("STEP 3: TRAIN FACE RECOGNITION MODEL")
    print("=" * 60)
    
    # Step 1: Load training data
    print("\n Loading training data...")
    
    face_size = RECOGNITION_CONFIG['face_size']
    faces, labels, label_names = load_training_data(TRAIN_DIR, face_size)
    
    if len(faces) == 0:
        print("\n No training data found!")
        print("   Run 01_collect_faces.py or 02_prepare_dataset.py first")
        return
    
    print(f"   Loaded {len(faces)} images")
    print(f"   {len(label_names)} persons: {list(label_names.values())}")
    
    # Step 2: Data augmentation
    if augment and TRAINING_CONFIG.get('augment', True):
        print("\n Applying data augmentation...")
        original_count = len(faces)
        faces, labels = augment_data(faces, labels, TRAINING_CONFIG)
        print(f"   {original_count} -> {len(faces)} images")
    
    # Step 3: Create and train model
    print(f"\n Training {method.upper()} model...")
    start_time = time.time()
    
    from neurova.face import FaceRecognizer
    
    recognizer = FaceRecognizer(method=method)
    
# convert labels to list for training
    faces_list = [f for f in faces]
    labels_list = labels.tolist()
    
    recognizer.train(faces_list, labels_list)
    
    train_time = time.time() - start_time
    print(f"   Training completed in {train_time:.2f} seconds")
    
    # Step 4: Save model
    print("\n Saving model...")
    
    model_data = {
        'method': method,
        'face_size': face_size,
        'label_names': label_names,
        'n_classes': len(label_names),
        'n_samples': len(faces),
        'trained_at': datetime.now().isoformat(),
        'config': RECOGNITION_CONFIG
    }
    
# save recognizer state if available
    if hasattr(recognizer, 'get_model_data'):
        model_data['model_state'] = recognizer.get_model_data()
    
# save internal attributes
    if hasattr(recognizer, '_training_data'):
        model_data['training_faces'] = faces_list
        model_data['training_labels'] = labels_list
    
    model_path = MODELS_DIR / f"face_model_{method}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"   Saved to: {model_path}")
    
    # Step 5: Quick validation
    print("\n Quick validation on training data...")
    
    correct = 0
    total = min(20, len(faces))  # Test on subset
    
    for i in range(total):
        pred_label, confidence = recognizer.predict(faces[i])
        if pred_label == labels[i]:
            correct += 1
    
    accuracy = correct / total * 100
    print(f"   Training accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Step 6: Generate training report
    print("\n Generating training report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'n_classes': len(label_names),
        'n_samples': len(faces),
        'n_augmented': len(faces) if augment else 0,
        'train_time': train_time,
        'train_accuracy': accuracy,
        'label_names': label_names,
        'config': RECOGNITION_CONFIG
    }
    
    report_path = REPORTS_DIR / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"   Saved to: {report_path}")
    
# summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Method:           {method.upper()}")
    print(f"  Classes:          {len(label_names)}")
    print(f"  Training samples: {len(faces)}")
    print(f"  Face size:        {face_size}")
    print(f"  Training time:    {train_time:.2f}s")
    print(f"  Train accuracy:   {accuracy:.1f}%")
    print(f"  Model saved:      {model_path}")
    print("=" * 60)
    
    print("\n Training complete!")
    print("\n Next step: python 04_evaluate_model.py")


def main():
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--method", type=str, default='lbph',
                       choices=['lbph', 'eigenface', 'fisherface'],
                       help="Recognition method")
    parser.add_argument("--no-augment", action="store_true",
                       help="Disable data augmentation")
    
    args = parser.parse_args()
    
    train_model(method=args.method, augment=not args.no_augment)


if __name__ == "__main__":
    main()
