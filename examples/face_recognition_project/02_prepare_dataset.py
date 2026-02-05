#!/usr/bin/env python3
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Step 2: Prepare Dataset


This script prepares your dataset by:
1. Scanning all images in data/train/
2. Validating images
3. Splitting into train/test/validation
4. Generating dataset statistics

Usage:
    python 02_prepare_dataset.py
    python 02_prepare_dataset.py --split-only  # If images already in place
"""

import os
import sys
import json
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR,
    TRAINING_CONFIG, REPORTS_DIR
)


def scan_images(directory: Path) -> dict:
    """
    Scan directory for images organized by person.
    
    Args:
        directory: Path to scan
    
    Returns:
        dict: {person_name: [image_paths]}
    """
    dataset = {}
    
    for person_dir in sorted(directory.iterdir()):
        if person_dir.is_dir():
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(person_dir.glob(ext))
                images.extend(person_dir.glob(ext.upper()))
            
            if images:
                dataset[person_dir.name] = sorted(images)
    
    return dataset


def validate_images(dataset: dict) -> dict:
    """
    Validate images and remove corrupted ones.
    
    Args:
        dataset: {person_name: [image_paths]}
    
    Returns:
        dict: Validated dataset
    """
    from PIL import Image
    
    validated = {}
    removed = []
    
    for person, images in dataset.items():
        valid_images = []
        
        for img_path in images:
            try:
                with Image.open(img_path) as img:
# check if image can be loaded
                    img.verify()
                
# re-open to check size
                with Image.open(img_path) as img:
                    if img.size[0] >= 50 and img.size[1] >= 50:
                        valid_images.append(img_path)
                    else:
                        removed.append((img_path, "too small"))
                        
            except Exception as e:
                removed.append((img_path, str(e)))
        
        if valid_images:
            validated[person] = valid_images
    
    return validated, removed


def split_dataset(dataset: dict, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, seed=42):
    """
    Split dataset into train/test/validation.
    
    Args:
        dataset: {person_name: [image_paths]}
        train_ratio: Training set ratio
        test_ratio: Test set ratio
        val_ratio: Validation set ratio
        seed: Random seed
    
    Returns:
        tuple: (train_set, test_set, val_set)
    """
    random.seed(seed)
    
    train_set = {}
    test_set = {}
    val_set = {}
    
    for person, images in dataset.items():
# shuffle images
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_test = max(1, int(n * test_ratio))
        n_val = n - n_train - n_test
        
        if n_val < 0:
            n_val = 0
            n_test = n - n_train
        
        train_set[person] = shuffled[:n_train]
        test_set[person] = shuffled[n_train:n_train+n_test]
        val_set[person] = shuffled[n_train+n_test:]
    
    return train_set, test_set, val_set


def copy_to_split_dirs(train_set, test_set, val_set):
    """
    Copy images to train/test/validation directories.
    """
    splits = [
        (train_set, TRAIN_DIR, "train"),
        (test_set, TEST_DIR, "test"),
        (val_set, VAL_DIR, "validation")
    ]
    
    for split_data, split_dir, split_name in splits:
        for person, images in split_data.items():
            person_dir = split_dir / person
            person_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in images:
                dest = person_dir / img_path.name
                if img_path.parent != person_dir:  # Only copy if not already there
                    shutil.copy2(img_path, dest)


def generate_report(dataset: dict, train_set: dict, test_set: dict, val_set: dict) -> dict:
    """
    Generate dataset statistics report.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_persons": len(dataset),
        "total_images": sum(len(imgs) for imgs in dataset.values()),
        "persons": {},
        "splits": {
            "train": {"persons": len(train_set), "images": sum(len(imgs) for imgs in train_set.values())},
            "test": {"persons": len(test_set), "images": sum(len(imgs) for imgs in test_set.values())},
            "validation": {"persons": len(val_set), "images": sum(len(imgs) for imgs in val_set.values())}
        }
    }
    
    for person, images in dataset.items():
        report["persons"][person] = {
            "total": len(images),
            "train": len(train_set.get(person, [])),
            "test": len(test_set.get(person, [])),
            "validation": len(val_set.get(person, []))
        }
    
    return report


def print_dataset_summary(report: dict):
    """
    Print dataset summary.
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("")
    
    print(f"\n Total Persons: {report['total_persons']}")
    print(f" Total Images: {report['total_images']}")
    
    print("\n Split Distribution:")
    for split, data in report['splits'].items():
        print(f"   {split.capitalize():12} - {data['images']:4} images ({data['persons']} persons)")
    
    print("\n Per-Person Breakdown:")
    print("")
    print(f"{'Person':<20} {'Total':>8} {'Train':>8} {'Test':>8} {'Val':>8}")
    print("")
    
    for person, data in report['persons'].items():
        print(f"{person:<20} {data['total']:>8} {data['train']:>8} {data['test']:>8} {data['validation']:>8}")
    
    print("")


def prepare_dataset(split_only: bool = False):
    """
    Main function to prepare dataset.
    """
    print("")
    print("STEP 2: PREPARE DATASET")
    print("")
    
    # Step 1: Scan for images
    print("\n Scanning for images...")
    
# check all directories
    all_images = {}
    for dir_path in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        scanned = scan_images(dir_path)
        for person, images in scanned.items():
            if person not in all_images:
                all_images[person] = []
            all_images[person].extend(images)
    
    if not all_images:
        print("\n No images found!")
        print("\n How to add images:")
        print(f"   1. Create folder: {TRAIN_DIR}/<person_name>/")
        print(f"   2. Add face images (JPG/PNG)")
        print("   3. Repeat for each person")
        print("\n   Or run: python 01_collect_faces.py --name 'Person1' --count 20")
        return
    
    print(f"   Found {sum(len(imgs) for imgs in all_images.values())} images")
    print(f"   Found {len(all_images)} persons")
    
    # Step 2: Validate images
    print("\n Validating images...")
    validated, removed = validate_images(all_images)
    
    if removed:
        print(f"     Removed {len(removed)} invalid images")
        for img, reason in removed[:5]:
            print(f"      - {img.name}: {reason}")
        if len(removed) > 5:
            print(f"      ... and {len(removed) - 5} more")
    
    # Step 3: Check minimum images
    min_images = TRAINING_CONFIG['min_images_per_person']
    valid_persons = {p: imgs for p, imgs in validated.items() if len(imgs) >= min_images}
    
    removed_persons = set(validated.keys()) - set(valid_persons.keys())
    if removed_persons:
        print(f"\n     Removed {len(removed_persons)} persons (less than {min_images} images):")
        for p in removed_persons:
            print(f"      - {p}: only {len(validated[p])} images")
    
    if not valid_persons:
        print(f"\n No persons with >= {min_images} images!")
        print(f"   Add more images or lower 'min_images_per_person' in config.py")
        return
    
    # Step 4: Split dataset
    print("\n  Splitting dataset...")
    train_set, test_set, val_set = split_dataset(
        valid_persons,
        train_ratio=TRAINING_CONFIG['train_ratio'],
        test_ratio=TRAINING_CONFIG['test_ratio'],
        val_ratio=TRAINING_CONFIG['val_ratio'],
        seed=TRAINING_CONFIG['random_seed']
    )
    
    # Step 5: Copy to directories
    print("\n Organizing files...")
    copy_to_split_dirs(train_set, test_set, val_set)
    
    # Step 6: Generate report
    print("\n Generating report...")
    report = generate_report(valid_persons, train_set, test_set, val_set)
    
# save report
    report_path = REPORTS_DIR / "dataset_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   Saved to: {report_path}")
    
# print summary
    print_dataset_summary(report)
    
    print("\n Dataset preparation complete!")
    print("\n Next step: python 03_train_model.py")


def main():
    parser = argparse.ArgumentParser(description="Prepare face recognition dataset")
    parser.add_argument("--split-only", action="store_true", help="Only split, don't copy")
    
    args = parser.parse_args()
    
    prepare_dataset(split_only=args.split_only)


if __name__ == "__main__":
    main()
