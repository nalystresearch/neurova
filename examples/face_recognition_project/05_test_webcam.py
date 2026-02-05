#!/usr/bin/env python3
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Step 5: Test with Webcam


This script runs real-time face recognition using your webcam:
1. Loads the trained model
2. Opens webcam
3. Detects faces in each frame
4. Recognizes detected faces
5. Displays results with bounding boxes and names

Usage:
    python 05_test_webcam.py
    python 05_test_webcam.py --method lbph --device 0
    python 05_test_webcam.py --threshold 80
"""

import os
import sys
import pickle
import time
import argparse
import numpy as np
from pathlib import Path

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    MODELS_DIR, WEBCAM_CONFIG, DETECTION_CONFIG,
    DISPLAY_CONFIG, RECOGNITION_CONFIG
)


def test_webcam(method: str = 'lbph', device: int = 0, threshold: float = 100.0):
    """
    Test face recognition with webcam.
    
    Args:
        method: Recognition method
        device: Camera device ID
        threshold: Confidence threshold (lower = stricter)
    """
    print("")
    print("STEP 5: TEST WITH WEBCAM")
    print("")
    
    # Step 1: Load model
    print("\n Loading trained model...")
    
    model_path = MODELS_DIR / f"face_model_{method}.pkl"
    
    if not model_path.exists():
        print(f"\n Model not found: {model_path}")
        print("   Run 03_train_model.py first")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    label_names = model_data.get('label_names', {})
    face_size = model_data.get('face_size', (100, 100))
    
    print(f"   Model: {method.upper()}")
    print(f"   Classes: {list(label_names.values())}")
    
# create recognizer and train
    from neurova.face import FaceRecognizer, FaceDetector
    
    recognizer = FaceRecognizer(method=method)
    
    if 'training_faces' in model_data:
        recognizer.train(model_data['training_faces'], model_data['training_labels'])
        print("   Model loaded and ready!")
    else:
        print("     No training data in model, recognition may fail")
    
# create detector
    detector = FaceDetector(method=DETECTION_CONFIG['method'])
    
    # Step 2: Try to use cv2 for better webcam support
    try:
        import cv2
        use_cv2 = True
        print("\n Using cv2 for webcam...")
    except ImportError:
        use_cv2 = False
        print("\n Using Neurova VideoCapture...")
        print("     Install cv2 for better experience: pip install cv2")
    
    if use_cv2:
        run_webcam_nv(detector, recognizer, label_names, face_size, device, threshold)
    else:
        run_webcam_neurova(detector, recognizer, label_names, face_size, device, threshold)


def run_webcam_nv(detector, recognizer, label_names, face_size, device, threshold):
    """Run webcam test with cv2."""
    import cv2
    
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_CONFIG['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_CONFIG['height'])
    
    if not cap.isOpened():
        print(" Error: Could not open webcam")
        return
    
    print("\n" + "-" * 60)
    print("WEBCAM CONTROLS:")
    print("  'q' - Quit")
    print("  's' - Take screenshot")
    print("  'r' - Reset FPS counter")
    print("")
    print("\n Webcam running... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
# calculate fps
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        
# convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
# detect faces
        faces = detector.detect(gray)
        
# process each face
        for face_box in faces:
            x, y, w, h = face_box[:4]
            
# extract face region
            face_region = gray[y:y+h, x:x+w]
            
            if face_region.size > 0:
# resize for recognition
                face_resized = cv2.resize(face_region, face_size)
                
# recognize
                label, confidence = recognizer.predict(face_resized)
                
# get name
                if confidence < threshold:
                    name = label_names.get(label, f"Person {label}")
                    color = DISPLAY_CONFIG['face_color']
                else:
                    name = "Unknown"
                    color = DISPLAY_CONFIG['unknown_color']
                
# draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 
                             DISPLAY_CONFIG['box_thickness'])
                
# draw name and confidence
                text = f"{name} ({confidence:.1f})"
                cv2.putText(frame, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           DISPLAY_CONFIG['font_scale'],
                           color,
                           DISPLAY_CONFIG['font_thickness'])
        
# draw fps
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
# draw face count
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
# show frame
        cv2.imshow("Face Recognition - Press 'q' to quit", frame)
        
# handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
# screenshot
            screenshot_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f" Screenshot saved: {screenshot_path}")
        elif key == ord('r'):
            frame_count = 0
            start_time = time.time()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n Webcam test complete!")


def run_webcam_neurova(detector, recognizer, label_names, face_size, device, threshold):
    """Run webcam test with Neurova (no preview)."""
    from neurova.io import VideoCapture
    from PIL import Image
    
    cap = VideoCapture(device)
    
    print("\n  No live preview available without cv2")
    print("    Processing frames and printing results...")
    print("\n    Press Ctrl+C to stop\n")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
# convert to grayscale
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2).astype(np.uint8)
            else:
                gray = frame
            
# detect faces
            faces = detector.detect(gray)
            
# process each face
            for i, face_box in enumerate(faces):
                x, y, w, h = face_box[:4]
                
# extract and resize
                face_region = gray[y:y+h, x:x+w]
                
                if face_region.size > 0:
                    pil_face = Image.fromarray(face_region)
                    pil_face = pil_face.resize(face_size, Image.LANCZOS)
                    face_resized = np.array(pil_face)
                    
# recognize
                    label, confidence = recognizer.predict(face_resized)
                    
# get name
                    if confidence < threshold:
                        name = label_names.get(label, f"Person {label}")
                    else:
                        name = "Unknown"
                    
                    print(f"  Frame {frame_count}: Face {i+1} -> {name} (conf: {confidence:.1f})")
            
            # FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"\n  FPS: {fps:.1f}, Frames: {frame_count}\n")
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n\n⏹  Stopped by user")
    
    cap.release()
    print("\n Webcam test complete!")


def main():
    parser = argparse.ArgumentParser(description="Test face recognition with webcam")
    parser.add_argument("--method", type=str, default='lbph',
                       choices=['lbph', 'eigenface', 'fisherface'],
                       help="Recognition method")
    parser.add_argument("--device", type=int, default=0,
                       help="Camera device ID")
    parser.add_argument("--threshold", type=float, default=100.0,
                       help="Confidence threshold (lower = stricter)")
    
    args = parser.parse_args()
    
    test_webcam(method=args.method, device=args.device, threshold=args.threshold)


if __name__ == "__main__":
    main()
