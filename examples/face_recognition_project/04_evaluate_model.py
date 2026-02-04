#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Step 4: Evaluate Model
=======================

This script evaluates the trained model on test and validation sets:
1. Loads the trained model
2. Runs predictions on test/validation data
3. Computes metrics (accuracy, precision, recall, F1)
4. Generates confusion matrix
5. Creates detailed report

Usage:
    python 04_evaluate_model.py
    python 04_evaluate_model.py --method lbph
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
    TEST_DIR, VAL_DIR, MODELS_DIR, REPORTS_DIR,
    RECOGNITION_CONFIG
)


def load_test_data(test_dir: Path, face_size: tuple = (100, 100)):
    """
    Load test data from directory.
    """
    from PIL import Image
    
    faces = []
    labels = []
    label_names = {}
    current_label = 0
    
    for person_dir in sorted(test_dir.iterdir()):
        if person_dir.is_dir():
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            
            if not images:
                continue
            
            label_names[current_label] = person_dir.name
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize(face_size, Image.LANCZOS)
                    face = np.array(img)
                    
                    faces.append(face)
                    labels.append(current_label)
                    
                except Exception as e:
                    print(f"    Error loading {img_path}: {e}")
            
            current_label += 1
    
    return np.array(faces), np.array(labels), label_names


def compute_metrics(y_true, y_pred, n_classes):
    """
    Compute classification metrics.
    """
# accuracy
    accuracy = np.mean(y_true == y_pred)
    
# per-class metrics
    precision = []
    recall = []
    f1 = []
    
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision.append(p)
        recall.append(r)
        f1.append(f)
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'precision_macro': np.mean(precision),
        'recall_macro': np.mean(recall),
        'f1_macro': np.mean(f1)
    }


def confusion_matrix(y_true, y_pred, n_classes):
    """
    Compute confusion matrix.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return cm


def print_confusion_matrix(cm, label_names):
    """
    Print confusion matrix.
    """
    n = len(label_names)
    
# header
    print("\n    Confusion Matrix:")
    print("    " + "-" * (12 * n + 10))
    
    header = "    " + " " * 12
    for i in range(n):
        name = label_names.get(i, str(i))[:8]
        header += f"{name:>10}"
    print(header)
    
    print("    " + "-" * (12 * n + 10))
    
# rows
    for i in range(n):
        name = label_names.get(i, str(i))[:10]
        row = f"    {name:<10}"
        for j in range(n):
            row += f"{cm[i, j]:>10}"
        print(row)
    
    print("    " + "-" * (12 * n + 10))


def evaluate_model(method: str = 'lbph'):
    """
    Evaluate trained model.
    """
    print("=" * 60)
    print("STEP 4: EVALUATE MODEL")
    print("=" * 60)
    
    # Step 1: Load model
    print("\n Loading trained model...")
    
    model_path = MODELS_DIR / f"face_model_{method}.pkl"
    
    if not model_path.exists():
        print(f"\n Model not found: {model_path}")
        print("   Run 03_train_model.py first")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"   Loaded: {model_path.name}")
    print(f"   Method: {model_data['method']}")
    print(f"   Classes: {model_data['n_classes']}")
    
# get label names from training
    train_label_names = model_data.get('label_names', {})
    face_size = model_data.get('face_size', (100, 100))
    
# recreate recognizer
    from neurova.face import FaceRecognizer
    recognizer = FaceRecognizer(method=method)
    
# retrain with saved data if available
    if 'training_faces' in model_data:
        print("   Restoring trained model...")
        recognizer.train(model_data['training_faces'], model_data['training_labels'])
    
    # Step 2: Evaluate on test set
    print("\n Evaluating on TEST set...")
    
    test_faces, test_labels, test_label_names = load_test_data(TEST_DIR, face_size)
    
    if len(test_faces) == 0:
        print("     No test data found")
        test_metrics = None
    else:
# map label names
        test_label_map = {}
        for test_label, test_name in test_label_names.items():
            for train_label, train_name in train_label_names.items():
                if test_name == train_name:
                    test_label_map[test_label] = train_label
                    break
        
# predict
        test_preds = []
        test_confs = []
        
        for face in test_faces:
            pred, conf = recognizer.predict(face)
            test_preds.append(pred)
            test_confs.append(conf)
        
        test_preds = np.array(test_preds)
        
# map test labels to training labels
        mapped_test_labels = np.array([test_label_map.get(l, -1) for l in test_labels])
        
# compute metrics
        valid_idx = mapped_test_labels >= 0
        if np.any(valid_idx):
            test_metrics = compute_metrics(
                mapped_test_labels[valid_idx], 
                test_preds[valid_idx],
                model_data['n_classes']
            )
            test_cm = confusion_matrix(
                mapped_test_labels[valid_idx],
                test_preds[valid_idx],
                model_data['n_classes']
            )
            
            print(f"   Test samples: {len(test_faces)}")
            print(f"   Accuracy: {test_metrics['accuracy']*100:.2f}%")
            print(f"   Precision: {test_metrics['precision_macro']*100:.2f}%")
            print(f"   Recall: {test_metrics['recall_macro']*100:.2f}%")
            print(f"   F1-Score: {test_metrics['f1_macro']*100:.2f}%")
            
            print_confusion_matrix(test_cm, train_label_names)
        else:
            test_metrics = None
            print("     No matching labels between train and test")
    
    # Step 3: Evaluate on validation set
    print("\n Evaluating on VALIDATION set...")
    
    val_faces, val_labels, val_label_names = load_test_data(VAL_DIR, face_size)
    
    if len(val_faces) == 0:
        print("     No validation data found")
        val_metrics = None
    else:
# map label names
        val_label_map = {}
        for val_label, val_name in val_label_names.items():
            for train_label, train_name in train_label_names.items():
                if val_name == train_name:
                    val_label_map[val_label] = train_label
                    break
        
# predict
        val_preds = []
        for face in val_faces:
            pred, conf = recognizer.predict(face)
            val_preds.append(pred)
        
        val_preds = np.array(val_preds)
        
# map labels
        mapped_val_labels = np.array([val_label_map.get(l, -1) for l in val_labels])
        
# compute metrics
        valid_idx = mapped_val_labels >= 0
        if np.any(valid_idx):
            val_metrics = compute_metrics(
                mapped_val_labels[valid_idx],
                val_preds[valid_idx],
                model_data['n_classes']
            )
            
            print(f"   Validation samples: {len(val_faces)}")
            print(f"   Accuracy: {val_metrics['accuracy']*100:.2f}%")
            print(f"   F1-Score: {val_metrics['f1_macro']*100:.2f}%")
        else:
            val_metrics = None
    
    # Step 4: Generate detailed report
    print("\n Generating evaluation report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'method': method,
        'n_classes': model_data['n_classes'],
        'label_names': train_label_names,
        'test': {
            'n_samples': len(test_faces) if len(test_faces) > 0 else 0,
            'metrics': test_metrics
        },
        'validation': {
            'n_samples': len(val_faces) if len(val_faces) > 0 else 0,
            'metrics': val_metrics
        }
    }
    
    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"   Saved to: {report_path}")
    
# summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if test_metrics:
        print(f"\n   TEST SET:")
        print(f"     Accuracy:  {test_metrics['accuracy']*100:.2f}%")
        print(f"     Precision: {test_metrics['precision_macro']*100:.2f}%")
        print(f"     Recall:    {test_metrics['recall_macro']*100:.2f}%")
        print(f"     F1-Score:  {test_metrics['f1_macro']*100:.2f}%")
    
    if val_metrics:
        print(f"\n   VALIDATION SET:")
        print(f"     Accuracy:  {val_metrics['accuracy']*100:.2f}%")
        print(f"     F1-Score:  {val_metrics['f1_macro']*100:.2f}%")
    
    print("\n" + "=" * 60)
    
    print("\n Evaluation complete!")
    print("\n Next step: python 05_test_webcam.py")


def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition model")
    parser.add_argument("--method", type=str, default='lbph',
                       choices=['lbph', 'eigenface', 'fisherface'],
                       help="Recognition method")
    
    args = parser.parse_args()
    
    evaluate_model(method=args.method)


if __name__ == "__main__":
    main()
