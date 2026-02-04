#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Complete Pipeline: Face Recognition
=====================================

This script runs the complete face recognition pipeline:
1. Collect faces (optional)
2. Prepare dataset
3. Train model
4. Evaluate model
5. Test with webcam

Usage:
    python pipeline.py                      # Full pipeline (no collection)
    python pipeline.py --collect "Person1"  # Collect + full pipeline
    python pipeline.py --skip-webcam        # Skip webcam test
    python pipeline.py --method eigenface   # Use different method
"""

import os
import sys
import argparse
from pathlib import Path

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_pipeline(
    collect_name: str = None,
    collect_count: int = 20,
    method: str = 'lbph',
    skip_webcam: bool = False,
    skip_eval: bool = False
):
    """
    Run complete face recognition pipeline.
    
    Args:
        collect_name: Person name to collect (None to skip)
        collect_count: Number of images to collect
        method: Recognition method
        skip_webcam: Skip webcam test
        skip_eval: Skip evaluation
    """
    print("=" * 70)
    print("        NEUROVA FACE RECOGNITION PIPELINE")
    print("=" * 70)
    
    steps = []
    if collect_name:
        steps.append(f"1. Collect faces for '{collect_name}'")
    steps.append("2. Prepare dataset")
    steps.append("3. Train model")
    if not skip_eval:
        steps.append("4. Evaluate model")
    if not skip_webcam:
        steps.append("5. Test with webcam")
    
    print("\n Pipeline Steps:")
    for step in steps:
        print(f"   {step}")
    print()
    
    # Step 1: Collect faces (optional)
    if collect_name:
        print("\n" + "=" * 70)
        print("PIPELINE STEP 1/5: COLLECT FACES")
        print("=" * 70)
        
        from importlib import import_module
        collect_module = import_module("01_collect_faces")
        collect_module.collect_faces(collect_name, collect_count)
        
        input("\n⏸  Press Enter to continue to next step...")
    
    # Step 2: Prepare dataset
    print("\n" + "=" * 70)
    print("PIPELINE STEP 2/5: PREPARE DATASET")
    print("=" * 70)
    
    from importlib import import_module
    prepare_module = import_module("02_prepare_dataset")
    prepare_module.prepare_dataset()
    
# check if we have data
    from config import TRAIN_DIR
    persons = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    if not persons:
        print("\n No training data found. Add images first!")
        print("\n Options:")
        print("   1. Run: python 01_collect_faces.py --name 'YourName' --count 20")
        print("   2. Manually add images to data/train/<person_name>/")
        return
    
    # Step 3: Train model
    print("\n" + "=" * 70)
    print("PIPELINE STEP 3/5: TRAIN MODEL")
    print("=" * 70)
    
    train_module = import_module("03_train_model")
    train_module.train_model(method=method, augment=True)
    
    # Step 4: Evaluate model
    if not skip_eval:
        print("\n" + "=" * 70)
        print("PIPELINE STEP 4/5: EVALUATE MODEL")
        print("=" * 70)
        
        eval_module = import_module("04_evaluate_model")
        eval_module.evaluate_model(method=method)
    
    # Step 5: Test with webcam
    if not skip_webcam:
        print("\n" + "=" * 70)
        print("PIPELINE STEP 5/5: TEST WITH WEBCAM")
        print("=" * 70)
        
        response = input("\n Start webcam test? (y/n): ")
        if response.lower() in ['y', 'yes']:
            webcam_module = import_module("05_test_webcam")
            webcam_module.test_webcam(method=method)
        else:
            print("   Skipped webcam test")
    
# final summary
    print("\n" + "=" * 70)
    print("                    PIPELINE COMPLETE!")
    print("=" * 70)
    
    print("\n Generated Files:")
    
    from config import MODELS_DIR, REPORTS_DIR
    
    model_path = MODELS_DIR / f"face_model_{method}.pkl"
    if model_path.exists():
        print(f"    Model: {model_path}")
    
    for report in REPORTS_DIR.glob("*.json"):
        print(f"    Report: {report}")
    
    print("\n Next Steps:")
    print("   • Add more people: python 01_collect_faces.py --name 'NewPerson'")
    print("   • Retrain: python 03_train_model.py")
    print("   • Test again: python 05_test_webcam.py")
    print("   • Try different method: python pipeline.py --method eigenface")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Complete Face Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                       # Run full pipeline
  python pipeline.py --collect "John"      # Collect faces + train + test
  python pipeline.py --skip-webcam         # Train and evaluate only
  python pipeline.py --method eigenface    # Use EigenFace method
        """
    )
    
    parser.add_argument("--collect", type=str, default=None,
                       help="Person name to collect faces for")
    parser.add_argument("--count", type=int, default=20,
                       help="Number of face images to collect")
    parser.add_argument("--method", type=str, default='lbph',
                       choices=['lbph', 'eigenface', 'fisherface'],
                       help="Face recognition method")
    parser.add_argument("--skip-webcam", action="store_true",
                       help="Skip webcam test at the end")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation step")
    
    args = parser.parse_args()
    
    run_pipeline(
        collect_name=args.collect,
        collect_count=args.count,
        method=args.method,
        skip_webcam=args.skip_webcam,
        skip_eval=args.skip_eval
    )


if __name__ == "__main__":
    main()
