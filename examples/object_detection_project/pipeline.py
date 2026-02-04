#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Object Detection Pipeline
==========================

Complete pipeline that runs all steps in sequence:
1. Annotate images (if needed)
2. Prepare dataset
3. Train detector
4. Evaluate detector
5. Test on webcam (optional)

Usage:
    python pipeline.py
    python pipeline.py --skip-annotate
    python pipeline.py --webcam
    python pipeline.py --method template
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    PROJECT_ROOT, DATA_DIR, TRAIN_DIR, MODELS_DIR, REPORTS_DIR
)


def print_header(title):
    """Print a section header."""
    print("\n")
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_data_exists():
    """Check if training data exists."""
    images_dir = TRAIN_DIR / "images"
    if not images_dir.exists():
        return False
    
    return len(list(images_dir.glob("*"))) > 0


def check_samples_exist():
    """Check if extracted samples exist."""
    samples_dir = DATA_DIR / "samples"
    pos_dir = samples_dir / "positive"
    
    if not pos_dir.exists():
        return False
    
    return len(list(pos_dir.glob("*"))) > 0


def check_model_exists():
    """Check if trained model exists."""
    if not MODELS_DIR.exists():
        return False
    
    return len(list(MODELS_DIR.glob("*.pkl"))) > 0


def run_pipeline(skip_annotate=False, skip_train=False, skip_eval=False,
                webcam=False, method='hog'):
    """
    Run the complete object detection pipeline.
    """
    print_header("NEUROVA OBJECT DETECTION PIPELINE")
    
    start_time = datetime.now()
    
# track what steps ran
    steps_run = []
    
    # Step 1: Check/Annotate Data
    print_header("Step 1: Data Annotation")
    
    if skip_annotate:
        print("⏭  Skipping annotation (--skip-annotate)")
    elif check_data_exists():
        print(" Training data found")
        print(f"   Location: {TRAIN_DIR / 'images'}")
    else:
        print(" No training data found!")
        print("\n To add data:")
        print(f"   1. Add images to: {TRAIN_DIR / 'images'}")
        print(f"   2. Run: python 01_annotate_images.py")
        print("\n   Or use pre-existing images with cascades")
        
# ask user what to do
        response = input("\nContinue with cascade detection only? (y/n): ").strip().lower()
        if response != 'y':
            print("\n  Pipeline stopped. Please add training data first.")
            return
    
    steps_run.append("data_check")
    
    # Step 2: Prepare Dataset
    print_header("Step 2: Dataset Preparation")
    
    if check_data_exists():
        print(" Preparing dataset...")
        
        try:
            from prepare_dataset import prepare_dataset
            prepare_dataset(extract_samples_flag=True)
            steps_run.append("prepare_dataset")
            print(" Dataset prepared successfully")
        except Exception as e:
            print(f"  Dataset preparation failed: {e}")
            print("   Continuing with existing data...")
    else:
        print("⏭  Skipping (no data to prepare)")
    
    # Step 3: Train Detector
    print_header("Step 3: Train Detector")
    
    if skip_train:
        print("⏭  Skipping training (--skip-train)")
    elif check_samples_exist():
        print(f" Training {method.upper()} detector...")
        
        try:
            from train_detector import train_detector
            detector = train_detector(method=method)
            
            if detector:
                steps_run.append("train_detector")
                print(" Detector trained successfully")
            else:
                print("  Training returned no detector")
        except Exception as e:
            print(f"  Training failed: {e}")
            print("   Will use cascade detectors instead")
    else:
        print("⏭  No samples to train on")
        print("   Will use pre-trained cascade detectors")
    
    # Step 4: Evaluate Detector
    print_header("Step 4: Evaluate Detector")
    
    if skip_eval:
        print("⏭  Skipping evaluation (--skip-eval)")
    elif check_model_exists():
        print(" Evaluating detector...")
        
        try:
            from evaluate_detector import evaluate_detector
            report = evaluate_detector()
            
            if report:
                steps_run.append("evaluate")
                print(" Evaluation complete")
                print(f"   mAP: {report['metrics']['mAP']:.4f}")
        except Exception as e:
            print(f"  Evaluation failed: {e}")
    else:
        print("⏭  No model to evaluate")
    
    # Step 5: Webcam Test (optional)
    print_header("Step 5: Webcam Test")
    
    if webcam:
        print(" Starting webcam detection...")
        
        try:
            from test_webcam import webcam_detection
            
            # Use trained model if available, otherwise cascade
            model_path = None
            cascade_type = None
            
            if check_model_exists():
                model_files = list(MODELS_DIR.glob("*.pkl"))
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            else:
                cascade_type = 'face'
            
            webcam_detection(
                model_path=model_path,
                cascade_type=cascade_type
            )
            steps_run.append("webcam")
            
        except Exception as e:
            print(f"  Webcam test failed: {e}")
    else:
        print("⏭  Skipping webcam test")
        print("   Use --webcam to enable")
    
# summary
    print_header("PIPELINE SUMMARY")
    
    elapsed = datetime.now() - start_time
    
    print(f"  Started:    {start_time.strftime('%H:%M:%S')}")
    print(f"  Completed:  {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Duration:   {elapsed}")
    print()
    print(f"  Steps completed: {len(steps_run)}")
    for step in steps_run:
        print(f"     {step}")
    print()
    
    # What's next
    print(" What's next?")
    
    if not check_data_exists():
        print("   1. Add training images to data/train/images/")
        print("   2. Run: python 01_annotate_images.py")
        print("   3. Re-run the pipeline")
    
    if not check_model_exists():
        print("   1. Ensure you have training samples")
        print("   2. Run: python 03_train_detector.py")
    
    if 'webcam' not in steps_run:
        print("   - Try webcam detection: python 05_test_webcam.py")
        print("   - Or cascade detection: python 05_test_webcam.py --cascade face")
    
    print("\n Pipeline complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete object detection pipeline"
    )
    parser.add_argument("--skip-annotate", action="store_true",
                       help="Skip annotation step")
    parser.add_argument("--skip-train", action="store_true",
                       help="Skip training step")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--webcam", action="store_true",
                       help="Include webcam test at the end")
    parser.add_argument("--method", type=str, default="hog",
                       choices=['hog', 'template', 'cascade'],
                       help="Detection method to train")
    
    args = parser.parse_args()
    
# change to script directory
    os.chdir(Path(__file__).parent)
    
    run_pipeline(
        skip_annotate=args.skip_annotate,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        webcam=args.webcam,
        method=args.method
    )


if __name__ == "__main__":
    main()
