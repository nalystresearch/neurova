#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Step 5: Real-Time Webcam Object Detection
==========================================

This script performs real-time object detection using:
1. Trained custom detector (HOG+SVM or Template)
2. Pre-trained Haar cascade detectors
3. HOG-based pedestrian detector

Usage:
    python 05_test_webcam.py
    python 05_test_webcam.py --model hog_svm_detector.pkl
    python 05_test_webcam.py --cascade face
    python 05_test_webcam.py --cascade pedestrian
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    MODELS_DIR, REPORTS_DIR, CASCADE_FILES,
    DETECTION_SETTINGS, WEBCAM_CONFIG
)

# try to import neurova
try:
    from neurova import nv
    from neurova.detection import (
        detect_haar_cascade, detect_hog,
        get_cascade_path
    )
    from neurova.video import VideoCapture
    NEUROVA_AVAILABLE = True
except ImportError:
    NEUROVA_AVAILABLE = False
    print("  Neurova not fully available")


def get_color_for_class(class_name):
    """Get a consistent color for each class."""
# simple hash-based color
    h = hash(class_name)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    return (b, g, r)  # BGR for OpenCV


def draw_detections(frame, detections, colors=None):
    """Draw detection boxes on frame."""
    import cv2
    
    for det in detections:
        x, y, w, h = det['bbox']
        conf = det.get('confidence', 1.0)
        cls = det.get('class', 'object')
        
# get color
        if colors and cls in colors:
            color = colors[cls]
        else:
            color = get_color_for_class(cls)
        
# draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
# draw label
        label = f"{cls}: {conf:.2f}"
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
# background for label
        cv2.rectangle(
            frame,
            (x, y - label_size[1] - 10),
            (x + label_size[0], y),
            color, -1
        )
        
        cv2.putText(
            frame, label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1
        )
    
    return frame


def run_custom_detector(detector, frame):
    """Run custom trained detector on frame."""
    import numpy as np
    from PIL import Image
    
# convert frame to pil image
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB
    else:
        img = frame
    
    return detector.detect(img)


def run_cascade_detector(frame, cascade_type='face'):
    """Run Haar cascade detector."""
    import cv2
    import numpy as np
    
# convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
# get cascade file
    cascade_file = CASCADE_FILES.get(cascade_type)
    if cascade_file is None:
        print(f"Unknown cascade type: {cascade_type}")
        return []
    
# load cascade
    if NEUROVA_AVAILABLE:
        cascade_path = get_cascade_path(cascade_file)
    else:
        cascade_path = cascade_file
    
    cascade = cv2.CascadeClassifier(cascade_path)
    
    if cascade.empty():
        print(f"Failed to load cascade: {cascade_path}")
        return []
    
# detect
    min_size = DETECTION_SETTINGS.get('min_size', 30)
    detections_raw = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size)
    )
    
# convert to our format
    detections = []
    for (x, y, w, h) in detections_raw:
        detections.append({
            'bbox': [int(x), int(y), int(w), int(h)],
            'confidence': 1.0,
            'class': cascade_type
        })
    
    return detections


def run_hog_pedestrian_detector(frame):
    """Run HOG pedestrian detector."""
    import cv2
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
# detect pedestrians
    detections_raw, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )
    
    detections = []
    for (x, y, w, h), weight in zip(detections_raw, weights):
        detections.append({
            'bbox': [int(x), int(y), int(w), int(h)],
            'confidence': float(weight[0]),
            'class': 'pedestrian'
        })
    
    return detections


def webcam_detection(model_path=None, cascade_type=None, show_fps=True, save_output=False):
    """
    Main webcam detection loop.
    """
    import cv2
    
    print("=" * 60)
    print("STEP 5: REAL-TIME WEBCAM DETECTION")
    print("=" * 60)
    
# determine detection mode
    detector = None
    
    if cascade_type:
        print(f"\n Mode: Haar Cascade ({cascade_type})")
        detect_func = lambda frame: run_cascade_detector(frame, cascade_type)
    
    elif cascade_type == 'pedestrian':
        print("\n Mode: HOG Pedestrian Detector")
        detect_func = run_hog_pedestrian_detector
    
    elif model_path:
        print(f"\n Mode: Custom Detector")
        print(f"   Model: {model_path}")
        
# load custom model
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from train_detector import HOGSVMDetector, TemplateMatchingDetector
            
            if 'hog' in model_path.stem:
                detector = HOGSVMDetector.load(model_path)
            elif 'template' in model_path.stem:
                detector = TemplateMatchingDetector.load(model_path)
            
            detect_func = lambda frame: run_custom_detector(detector, frame)
            print("    Model loaded successfully")
        except Exception as e:
            print(f"    Error loading model: {e}")
            print("   Falling back to face cascade")
            detect_func = lambda frame: run_cascade_detector(frame, 'face')
    else:
# default to face detection
        print("\n Mode: Haar Cascade (face)")
        detect_func = lambda frame: run_cascade_detector(frame, 'face')
    
# open webcam
    print("\n Opening webcam...")
    
    device_id = WEBCAM_CONFIG.get('device_id', 0)
    width = WEBCAM_CONFIG.get('width', 640)
    height = WEBCAM_CONFIG.get('height', 480)
    
    if NEUROVA_AVAILABLE:
        try:
            cap = VideoCapture(device_id, width=width, height=height)
        except:
            cap = cv2.VideoCapture(device_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print(" Failed to open webcam!")
        return
    
    print(f"   Device: {device_id}")
    print(f"   Resolution: {width}x{height}")
    
# setup video writer if saving
    writer = None
    if save_output:
        output_dir = REPORTS_DIR / "webcam_output"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"detection_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path), fourcc, 30,
            (width, height)
        )
        print(f"\n Recording to: {output_path}")
    
# stats
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    fps_list = []
    
    print("\n Starting detection...")
    print("   Press 'q' to quit")
    print("   Press 's' to save current frame")
    print("   Press 'r' to toggle recording")
    print()
    
    recording = save_output
    
    try:
        while True:
            frame_start = time.time()
            
# read frame
            if NEUROVA_AVAILABLE and hasattr(cap, 'read_frame'):
                ret, frame = cap.read_frame()
            else:
                ret, frame = cap.read()
            
            if not ret:
                print(" Failed to read frame")
                break
            
# run detection
            detections = detect_func(frame)
            total_detections += len(detections)
            
# draw detections
            frame = draw_detections(frame, detections)
            
# calculate fps
            fps = 1.0 / (time.time() - frame_start + 1e-6)
            fps_list.append(fps)
            
# draw info
            if show_fps:
                cv2.putText(
                    frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Detections: {len(detections)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
            
            if recording:
                cv2.putText(
                    frame, "[REC]",
                    (width - 80, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )
            
# save frame if recording
            if recording and writer:
                writer.write(frame)
            
# display
            cv2.imshow('Neurova Object Detection', frame)
            
# handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
# save current frame
                output_dir = REPORTS_DIR / "webcam_output"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = output_dir / f"frame_{timestamp}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f" Saved: {save_path}")
            elif key == ord('r'):
# toggle recording
                if not recording and writer is None:
                    output_dir = REPORTS_DIR / "webcam_output"
                    output_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = output_dir / f"detection_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(
                        str(output_path), fourcc, 30,
                        (width, height)
                    )
                    print(f" Started recording: {output_path}")
                
                recording = not recording
                print(f"Recording: {'ON' if recording else 'OFF'}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    
    finally:
# cleanup
        if NEUROVA_AVAILABLE and hasattr(cap, 'release'):
            cap.release()
        else:
            cap.release()
        
        if writer:
            writer.release()
        
        cv2.destroyAllWindows()
    
# stats
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"  Duration:    {elapsed:.1f} seconds")
    print(f"  Frames:      {frame_count}")
    print(f"  Avg FPS:     {avg_fps:.1f}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg per frame:    {total_detections / max(frame_count, 1):.1f}")
    print("=" * 60)
    
# save session report
    report = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': elapsed,
        'total_frames': frame_count,
        'avg_fps': avg_fps,
        'total_detections': total_detections,
        'avg_detections_per_frame': total_detections / max(frame_count, 1)
    }
    
    report_path = REPORTS_DIR / "webcam_session.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n Session report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-time webcam object detection")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to custom model file")
    parser.add_argument("--cascade", type=str, default=None,
                       choices=['face', 'eyes', 'smile', 'body', 'pedestrian'],
                       help="Use pre-trained cascade detector")
    parser.add_argument("--no-fps", action="store_true",
                       help="Hide FPS display")
    parser.add_argument("--save", action="store_true",
                       help="Save output video")
    
    args = parser.parse_args()
    
    model_path = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = MODELS_DIR / model_path
    
    webcam_detection(
        model_path=model_path,
        cascade_type=args.cascade,
        show_fps=not args.no_fps,
        save_output=args.save
    )


if __name__ == "__main__":
    main()
