#!/usr/bin/env python3
# neurova library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)

"""
Step 2: Prepare Dataset

This script prepares your object detection dataset by doing the following:
    1. Validates images and annotations to make sure they are correct
    2. Splits data into train, test, and validation sets
    3. Generates statistics about your dataset
    4. Extracts positive and negative samples for training

How to use:
    python 02_prepare_dataset.py
    python 02_prepare_dataset.py --extract-samples
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
    REPORTS_DIR, TRAINING_CONFIG
)


def load_annotations(data_dir):
    """
    Load all annotations from a directory.
    
    Parameters:
        data_dir: Path to the data directory
    
    Returns:
        List of tuples containing image path and annotation data
    """
    annotations = []
    
    images_dir = data_dir / "images"
    annot_dir = data_dir / "annotations"
    
    if not images_dir.exists():
        return annotations
    
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
# find corresponding annotation
            annot_path = annot_dir / (img_path.stem + ".json")
            
            if annot_path.exists():
                with open(annot_path) as f:
                    annot = json.load(f)
                annotations.append((img_path, annot))
            else:
# no annotation - can use as negative sample
                annotations.append((img_path, None))
    
    return annotations


def validate_annotations(annotations):
    """
    Check annotations and filter out invalid ones.
    
    Parameters:
        annotations: List of image path and annotation tuples
    
    Returns:
        Two lists - valid annotations and invalid ones with reasons
    """
    valid = []
    invalid = []
    
    for img_path, annot in annotations:
        if not img_path.exists():
            invalid.append((img_path, "Image not found"))
            continue
        
        if annot is None:
# no annotation - still valid as negative sample
            valid.append((img_path, annot))
            continue
        
# check annotation format
        if 'objects' not in annot:
            invalid.append((img_path, "Missing objects field"))
            continue
        
# validate bounding boxes
        valid_objects = []
        for obj in annot.get('objects', []):
            if 'bbox' not in obj or 'class' not in obj:
                continue
            
            bbox = obj['bbox']
            if len(bbox) != 4:
                continue
            
            x, y, w, h = bbox
            if w > 0 and h > 0:
                valid_objects.append(obj)
        
        if valid_objects:
            annot['objects'] = valid_objects
            valid.append((img_path, annot))
        else:
# no valid objects - treat as negative
            valid.append((img_path, None))
    
    return valid, invalid


def get_class_statistics(annotations):
    """
    Get class distribution statistics.
    
    Parameters:
        annotations: List of image path and annotation tuples
    
    Returns:
        Dictionary of class counts and total number of objects
    """
    class_counts = {}
    total_objects = 0
    
    for img_path, annot in annotations:
        if annot is None:
            continue
        
        for obj in annot.get('objects', []):
            cls = obj.get('class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
            total_objects += 1
    
    return class_counts, total_objects


def split_dataset(annotations, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, seed=42):
    """
    Split annotations into train, test, and validation sets.
    
    Parameters:
        annotations: List of all annotations
        train_ratio: Percentage for training (default 70%)
        test_ratio: Percentage for testing (default 15%)
        val_ratio: Percentage for validation (default 15%)
        seed: Random seed for reproducibility
    
    Returns:
        Three lists for train, test, and validation sets
    """
    random.seed(seed)
    
# shuffle the data
    shuffled = annotations.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    
    train_set = shuffled[:n_train]
    test_set = shuffled[n_train:n_train + n_test]
    val_set = shuffled[n_train + n_test:]
    
    return train_set, test_set, val_set


def copy_to_split_dirs(train_set, test_set, val_set):
    """
    Copy images and annotations to their respective directories.
    
    Parameters:
        train_set: Training data
        test_set: Testing data
        val_set: Validation data
    """
    splits = [
        (train_set, TRAIN_DIR),
        (test_set, TEST_DIR),
        (val_set, VAL_DIR)
    ]
    
    for split_data, split_dir in splits:
        images_dir = split_dir / "images"
        annot_dir = split_dir / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annot_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, annot in split_data:
# copy image
            dest_img = images_dir / img_path.name
            if img_path != dest_img:
                shutil.copy2(img_path, dest_img)
            
# save annotation
            if annot is not None:
                annot_path = annot_dir / (img_path.stem + ".json")
                with open(annot_path, 'w') as f:
                    json.dump(annot, f, indent=2)


def extract_samples(annotations, output_dir, sample_size=(64, 64)):
    """
    Extract positive and negative samples for training.
    
    Parameters:
        annotations: List of image path and annotation tuples
        output_dir: Where to save the samples
        sample_size: Size of each sample in pixels
    
    Returns:
        Number of positive and negative samples extracted
    """
    from PIL import Image
    
    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    
    pos_count = 0
    neg_count = 0
    
    for img_path, annot in annotations:
        try:
            img = Image.open(img_path)
            
            if annot is not None:
# extract positive samples from object regions
                for obj in annot.get('objects', []):
                    x, y, w, h = obj['bbox']
                    
# crop and resize
                    crop = img.crop((x, y, x + w, y + h))
                    crop = crop.resize(sample_size, Image.LANCZOS)
                    
# convert to grayscale
                    if crop.mode != 'L':
                        crop = crop.convert('L')
                    
# save
                    pos_path = pos_dir / ("pos_" + str(pos_count).zfill(6) + ".jpg")
                    crop.save(pos_path)
                    pos_count += 1
            
# extract negative samples from random regions without objects
            width, height = img.size
            
# get object regions to avoid
            obj_regions = []
            if annot is not None:
                for obj in annot.get('objects', []):
                    obj_regions.append(obj['bbox'])
            
# random negative samples
            neg_per_image = TRAINING_CONFIG.get('neg_samples_per_image', 10)
            for _ in range(neg_per_image):
# random position
                sw, sh = sample_size
                if width <= sw or height <= sh:
                    continue
                
                rx = random.randint(0, width - sw)
                ry = random.randint(0, height - sh)
                
# check overlap with objects
                overlaps = False
                for ox, oy, ow, oh in obj_regions:
                    if (rx < ox + ow and rx + sw > ox and
                        ry < oy + oh and ry + sh > oy):
                        overlaps = True
                        break
                
                if not overlaps:
                    crop = img.crop((rx, ry, rx + sw, ry + sh))
                    if crop.mode != 'L':
                        crop = crop.convert('L')
                    
                    neg_path = neg_dir / ("neg_" + str(neg_count).zfill(6) + ".jpg")
                    crop.save(neg_path)
                    neg_count += 1
            
        except Exception as e:
            print("  Warning: Error processing " + str(img_path) + ": " + str(e))
    
    return pos_count, neg_count


def prepare_dataset(extract_samples_flag=False):
    """
    Main function to prepare the dataset.
    
    Parameters:
        extract_samples_flag: Whether to extract training samples
    """
    print("")
    print("STEP 2: PREPARE DATASET")
    print("")
    
    # Step 1: Load all annotations
    print("")
    print("Loading data...")
    
    all_annotations = []
    for data_dir in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        annotations = load_annotations(data_dir)
        all_annotations.extend(annotations)
    
    # Also check for unsplit data in DATA_DIR
    if (DATA_DIR / "images").exists():
        annotations = load_annotations(DATA_DIR)
        all_annotations.extend(annotations)
    
    if not all_annotations:
        print("")
        print("Error: No data found!")
        print("")
        print("How to add data:")
        print("   1. Add images to: " + str(TRAIN_DIR / 'images'))
        print("   2. Add annotations to: " + str(TRAIN_DIR / 'annotations'))
        print("   3. Or run: python 01_annotate_images.py")
        return
    
    print("   Found " + str(len(all_annotations)) + " images")
    
    # Step 2: Validate
    print("")
    print("Validating annotations...")
    valid, invalid = validate_annotations(all_annotations)
    
    print("   Valid: " + str(len(valid)))
    if invalid:
        print("   Invalid: " + str(len(invalid)))
        for img, reason in invalid[:5]:
            print("     - " + img.name + ": " + reason)
    
    # Step 3: Get statistics
    print("")
    print("Analyzing dataset...")
    class_counts, total_objects = get_class_statistics(valid)
    
    annotated = sum(1 for _, a in valid if a is not None)
    unannotated = len(valid) - annotated
    
    print("   Annotated images: " + str(annotated))
    print("   Unannotated (negative): " + str(unannotated))
    print("   Total objects: " + str(total_objects))
    print("")
    print("   Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print("     " + cls + ": " + str(count))
    
    # Step 4: Split dataset
    print("")
    print("Splitting dataset...")
    train_set, test_set, val_set = split_dataset(
        valid,
        train_ratio=TRAINING_CONFIG['train_ratio'],
        test_ratio=TRAINING_CONFIG['test_ratio'],
        val_ratio=TRAINING_CONFIG['val_ratio'],
        seed=TRAINING_CONFIG['random_seed']
    )
    
    print("   Train: " + str(len(train_set)) + " images")
    print("   Test: " + str(len(test_set)) + " images")
    print("   Validation: " + str(len(val_set)) + " images")
    
    # Step 5: Copy to directories
    print("")
    print("Organizing files...")
    copy_to_split_dirs(train_set, test_set, val_set)
    
    # Step 6: Extract samples (optional)
    if extract_samples_flag:
        print("")
        print("Extracting training samples...")
        samples_dir = DATA_DIR / "samples"
        pos_count, neg_count = extract_samples(train_set, samples_dir)
        print("   Positive samples: " + str(pos_count))
        print("   Negative samples: " + str(neg_count))
    
    # Step 7: Generate report
    print("")
    print("Generating report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(valid),
        'annotated': annotated,
        'unannotated': unannotated,
        'total_objects': total_objects,
        'class_distribution': class_counts,
        'splits': {
            'train': len(train_set),
            'test': len(test_set),
            'validation': len(val_set)
        }
    }
    
    report_path = REPORTS_DIR / "dataset_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("   Saved: " + str(report_path))
    
# summary
    print("")
    print("")
    print("DATASET SUMMARY")
    print("")
    print("  Total images:   " + str(len(valid)))
    print("  Total objects:  " + str(total_objects))
    print("  Classes:        " + str(len(class_counts)))
    print("  Train/Test/Val: " + str(len(train_set)) + "/" + str(len(test_set)) + "/" + str(len(val_set)))
    print("")
    
    print("")
    print("Dataset preparation complete!")
    print("")
    print("Next step: python 03_train_detector.py")


def main():
    parser = argparse.ArgumentParser(description="Prepare object detection dataset")
    parser.add_argument("--extract-samples", action="store_true",
                       help="Extract positive/negative samples for training")
    
    args = parser.parse_args()
    
    prepare_dataset(extract_samples_flag=args.extract_samples)


if __name__ == "__main__":
    main()

# neurova library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)
