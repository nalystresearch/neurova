#!/usr/bin/env python3
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Step 4: Evaluate Object Detector


This script evaluates the trained object detector:
1. Computes precision, recall, F1-score
2. Computes mAP (mean Average Precision)
3. Generates confusion matrix
4. Analyzes detection performance at different IoU thresholds

Usage:
    python 04_evaluate_detector.py
    python 04_evaluate_detector.py --model hog_svm_detector.pkl
    python 04_evaluate_detector.py --iou-threshold 0.5
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    TEST_DIR, VAL_DIR, MODELS_DIR, REPORTS_DIR,
    DETECTION_SETTINGS, CLASS_NAMES
)


def load_model(model_path: Path):
    """Load trained detector model."""
    if not model_path.exists():
        return None
    
# import detector classes
    from train_detector import HOGSVMDetector, TemplateMatchingDetector
    
    if 'hog' in model_path.stem:
        return HOGSVMDetector.load(model_path)
    elif 'template' in model_path.stem:
        return TemplateMatchingDetector.load(model_path)
    else:
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)


def load_test_data(data_dir: Path):
    """Load test images and annotations."""
    from PIL import Image
    
    images_dir = data_dir / "images"
    annot_dir = data_dir / "annotations"
    
    test_data = []
    
    if not images_dir.exists():
        return test_data
    
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        
# load image
        img = Image.open(img_path)
        
# load annotation
        annot_path = annot_dir / f"{img_path.stem}.json"
        if annot_path.exists():
            with open(annot_path) as f:
                annot = json.load(f)
        else:
            annot = {'objects': []}
        
        test_data.append({
            'path': img_path,
            'image': img,
            'annotation': annot
        })
    
    return test_data


def compute_iou(box1, box2):
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1, box2: [x, y, w, h] format
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
# convert to corner format
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    
# intersection
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
# union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_detections(detections, ground_truths, iou_threshold=0.5):
    """
    Match detections to ground truth boxes.
    
    Returns:
        matched: List of (detection, gt, iou) tuples
        unmatched_det: List of unmatched detections (false positives)
        unmatched_gt: List of unmatched ground truths (false negatives)
    """
    if not detections or not ground_truths:
        return [], detections, ground_truths
    
# compute iou matrix
    n_det = len(detections)
    n_gt = len(ground_truths)
    iou_matrix = np.zeros((n_det, n_gt))
    
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            iou_matrix[i, j] = compute_iou(det['bbox'], gt['bbox'])
    
# greedy matching
    matched = []
    matched_det = set()
    matched_gt = set()
    
    while True:
# find best match
        if iou_matrix.size == 0:
            break
        
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        
        i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        
        matched.append((detections[i], ground_truths[j], max_iou))
        matched_det.add(i)
        matched_gt.add(j)
        
# remove matched from consideration
        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0
    
    unmatched_det = [d for i, d in enumerate(detections) if i not in matched_det]
    unmatched_gt = [g for j, g in enumerate(ground_truths) if j not in matched_gt]
    
    return matched, unmatched_det, unmatched_gt


def compute_precision_recall(all_detections, all_ground_truths, iou_threshold=0.5):
    """
    Compute precision-recall curve.
    """
# collect all detections with their scores and match status
    scored_detections = []
    total_positives = 0
    
    for detections, ground_truths in zip(all_detections, all_ground_truths):
        total_positives += len(ground_truths)
        
        matched, unmatched_det, unmatched_gt = match_detections(
            detections, ground_truths, iou_threshold
        )
        
# true positives
        for det, gt, iou in matched:
            scored_detections.append({
                'confidence': det['confidence'],
                'tp': True
            })
        
# false positives
        for det in unmatched_det:
            scored_detections.append({
                'confidence': det['confidence'],
                'tp': False
            })
    
    if not scored_detections:
        return [], [], 0.0
    
# sort by confidence
    scored_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
# compute precision-recall
    precisions = []
    recalls = []
    tp_count = 0
    fp_count = 0
    
    for det in scored_detections:
        if det['tp']:
            tp_count += 1
        else:
            fp_count += 1
        
        precision = tp_count / (tp_count + fp_count)
        recall = tp_count / total_positives if total_positives > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Compute AP (area under PR curve)
    ap = compute_ap(precisions, recalls)
    
    return precisions, recalls, ap


def compute_ap(precisions, recalls):
    """Compute Average Precision using 11-point interpolation."""
    if not precisions:
        return 0.0
    
# add endpoints
    precisions = [0] + precisions + [0]
    recalls = [0] + recalls + [1]
    
# make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        # Find max precision for recall >= t
        p = 0
        for i, r in enumerate(recalls):
            if r >= t:
                p = max(p, precisions[i])
        ap += p
    
    return ap / 11


def evaluate_detector(model_path: Path = None, iou_threshold: float = 0.5,
                     use_validation: bool = False):
    """
    Main evaluation function.
    """
    print("")
    print("STEP 4: EVALUATE OBJECT DETECTOR")
    print("")
    
# select data directory
    data_dir = VAL_DIR if use_validation else TEST_DIR
    
# load test data
    print(f"\n Loading {'validation' if use_validation else 'test'} data...")
    test_data = load_test_data(data_dir)
    
    if not test_data:
        print(f"\n No test data found in: {data_dir}")
        return None
    
    print(f"   Found {len(test_data)} images")
    
# load or find model
    if model_path is None:
# find latest model
        model_files = list(MODELS_DIR.glob("*.pkl"))
        if not model_files:
            print("\n No model found!")
            print(f"\n First run: python 03_train_detector.py")
            return None
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    print(f"\n Loading model: {model_path.name}")
    
    try:
# add current directory to path for imports
        sys.path.insert(0, str(Path(__file__).parent))
        detector = load_model(model_path)
    except Exception as e:
        print(f"\n Error loading model: {e}")
        return None
    
    if detector is None:
        print("\n Failed to load model!")
        return None
    
# run detection on all test images
    print("\n Running detection...")
    
    all_detections = []
    all_ground_truths = []
    
    for i, data in enumerate(test_data):
        print(f"   Processing {i+1}/{len(test_data)}: {data['path'].name}", end='\r')
        
# get detections
        detections = detector.detect(data['image'])
        all_detections.append(detections)
        
# get ground truths
        gt_objects = data['annotation'].get('objects', [])
        all_ground_truths.append(gt_objects)
    
    print(f"\n   Processed {len(test_data)} images")
    
# compute metrics
    print(f"\n Computing metrics (IoU threshold: {iou_threshold})...")
    
# per-class metrics
    class_metrics = {}
    
# overall metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for detections, ground_truths in zip(all_detections, all_ground_truths):
        matched, unmatched_det, unmatched_gt = match_detections(
            detections, ground_truths, iou_threshold
        )
        
        total_tp += len(matched)
        total_fp += len(unmatched_det)
        total_fn += len(unmatched_gt)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n   True Positives:  {total_tp}")
    print(f"   False Positives: {total_fp}")
    print(f"   False Negatives: {total_fn}")
    print(f"\n   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
# compute map
    print("\n Computing mAP...")
    precisions, recalls, ap = compute_precision_recall(
        all_detections, all_ground_truths, iou_threshold
    )
    
    print(f"   AP @ IoU={iou_threshold}: {ap:.4f}")
    
# compute map at different thresholds
    print("\n   AP @ different IoU thresholds:")
    ap_values = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        _, _, ap_t = compute_precision_recall(
            all_detections, all_ground_truths, threshold
        )
        ap_values.append(ap_t)
        print(f"     IoU={threshold}: {ap_t:.4f}")
    
    mAP = np.mean(ap_values)
    print(f"\n   mAP@[0.5:0.9]: {mAP:.4f}")
    
    # Analyze detection speeds/sizes
    print("\n Detection analysis:")
    
    det_sizes = []
    for detections in all_detections:
        for det in detections:
            w, h = det['bbox'][2], det['bbox'][3]
            det_sizes.append((w, h, w * h))
    
    if det_sizes:
        sizes = np.array(det_sizes)
        print(f"   Total detections: {len(det_sizes)}")
        print(f"   Avg size: {sizes[:, 2].mean():.1f} px²")
        print(f"   Min size: {sizes[:, 2].min():.1f} px²")
        print(f"   Max size: {sizes[:, 2].max():.1f} px²")
    
# generate report
    print("\n Generating report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': model_path.name,
        'test_set': str(data_dir),
        'num_images': len(test_data),
        'iou_threshold': iou_threshold,
        'metrics': {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ap': ap,
            'mAP': mAP
        },
        'ap_at_thresholds': {
            f'IoU={t}': v for t, v in zip([0.5, 0.6, 0.7, 0.8, 0.9], ap_values)
        }
    }
    
    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Saved: {report_path}")
    
# summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("")
    print(f"  Model:      {model_path.name}")
    print(f"  Test set:   {len(test_data)} images")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  mAP:        {mAP:.4f}")
    print("")
    
    print("\n Evaluation complete!")
    print("\n Next step: python 05_test_webcam.py")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate object detector")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model file")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for matching")
    parser.add_argument("--validation", action="store_true",
                       help="Use validation set instead of test set")
    
    args = parser.parse_args()
    
    model_path = Path(args.model) if args.model else None
    if model_path and not model_path.is_absolute():
        model_path = MODELS_DIR / model_path
    
    evaluate_detector(
        model_path=model_path,
        iou_threshold=args.iou_threshold,
        use_validation=args.validation
    )


if __name__ == "__main__":
    main()
