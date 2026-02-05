#!/usr/bin/env python3
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Step 1: Collect Face Images from Webcam


This script captures face images from your webcam and saves them
to the training folder.

Usage:
    python 01_collect_faces.py --name "John" --count 20
    python 01_collect_faces.py --name "Jane" --count 15

Arguments:
    --name      Person's name (creates folder)
    --count     Number of images to capture (default: 20)
    --device    Camera device ID (default: 0)
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    TRAIN_DIR, WEBCAM_CONFIG, DETECTION_CONFIG, 
    HAARCASCADES, DISPLAY_CONFIG
)

def collect_faces(name: str, count: int = 20, device: int = 0):
    """
    Collect face images from webcam.
    
    Args:
        name: Person's name (folder name)
        count: Number of images to capture
        device: Camera device ID
    """
    print("")
    print("STEP 1: COLLECT FACE IMAGES")
    print("")
    
# create person folder
    person_dir = TRAIN_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    
# get existing images count
    existing = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
    start_num = len(existing) + 1
    
    print(f"\n Saving to: {person_dir}")
    print(f" Existing images: {len(existing)}")
    print(f" Target: {count} new images")
    
# try to import cv2 for webcam
    try:
        import cv2
        use_cv2 = True
    except ImportError:
        print("\n  cv2 not installed. Using Neurova's VideoCapture...")
        use_cv2 = False
    
# load face detector
    from neurova.face import FaceDetector
    detector = FaceDetector(method=DETECTION_CONFIG['method'])
    
    print("\n" + "-" * 60)
    print("INSTRUCTIONS:")
    print("  - Look at the camera")
    print("  - Move your head slightly between captures")
    print("  - Press 'q' to quit early")
    print("  - Press 's' to skip a frame")
    print("")
    
    if use_cv2:
        collect_with_nv(detector, person_dir, name, count, start_num, device)
    else:
        collect_with_neurova(detector, person_dir, name, count, start_num, device)


def collect_with_nv(detector, person_dir, name, count, start_num, device):
    """Collect faces using cv2."""
    import cv2
    
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_CONFIG['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_CONFIG['height'])
    
    if not cap.isOpened():
        print(" Error: Could not open webcam")
        return
    
    print("\n Webcam opened. Starting capture...")
    
    captured = 0
    last_capture = 0
    
    while captured < count:
        ret, frame = cap.read()
        if not ret:
            print(" Error reading frame")
            break
        
# convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
# detect faces
        faces = detector.detect(gray)
        
# draw faces on frame
        display_frame = frame.copy()
        for face in faces:
            x, y, w, h = face[:4]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), 
                         DISPLAY_CONFIG['face_color'], DISPLAY_CONFIG['box_thickness'])
        
# display status
        status = f"Captured: {captured}/{count}"
        cv2.putText(display_frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Person: {name}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(faces) == 0:
            cv2.putText(display_frame, "No face detected", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif len(faces) > 1:
            cv2.putText(display_frame, "Multiple faces - center one face", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        cv2.imshow("Collect Faces - Press 'q' to quit", display_frame)
        
# capture if conditions met
        current_time = time.time()
        if (len(faces) == 1 and 
            current_time - last_capture >= WEBCAM_CONFIG['capture_delay']):
            
            x, y, w, h = faces[0][:4]
            
# add padding
            pad = int(min(w, h) * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            
# extract and save face
            face_img = gray[y1:y2, x1:x2]
            
# resize to standard size
            face_img = cv2.resize(face_img, (200, 200))
            
# save
            img_path = person_dir / f"img_{start_num + captured:04d}.jpg"
            cv2.imwrite(str(img_path), face_img)
            
            captured += 1
            last_capture = current_time
            print(f"   Captured {captured}/{count}: {img_path.name}")
        
# check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹  Capture stopped by user")
            break
        elif key == ord('s'):
            print("  ⏭  Frame skipped")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n Captured {captured} images for '{name}'")
    print(f" Saved to: {person_dir}")


def collect_with_neurova(detector, person_dir, name, count, start_num, device):
    """Collect faces using Neurova's VideoCapture."""
    from neurova.io import VideoCapture
    from PIL import Image
    
    cap = VideoCapture(device)
    
    print("\n Using Neurova VideoCapture...")
    print("  No live preview (install cv2 for preview)")
    print("\nCapturing frames automatically...")
    
    captured = 0
    
    for i in range(count * 3):  # Try 3x frames
        ret, frame = cap.read()
        if not ret:
            continue
        
# convert to grayscale
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2).astype(np.uint8)
        else:
            gray = frame
        
# detect faces
        faces = detector.detect(gray)
        
        if len(faces) == 1:
            x, y, w, h = faces[0][:4]
            
# extract face
            face_img = gray[y:y+h, x:x+w]
            
# resize
            pil_img = Image.fromarray(face_img)
            pil_img = pil_img.resize((200, 200), Image.LANCZOS)
            
# save
            img_path = person_dir / f"img_{start_num + captured:04d}.jpg"
            pil_img.save(str(img_path))
            
            captured += 1
            print(f"   Captured {captured}/{count}")
            
            if captured >= count:
                break
            
            time.sleep(WEBCAM_CONFIG['capture_delay'])
    
    cap.release()
    
    print(f"\n Captured {captured} images for '{name}'")


def main():
    parser = argparse.ArgumentParser(description="Collect face images from webcam")
    parser.add_argument("--name", type=str, required=True, help="Person's name")
    parser.add_argument("--count", type=int, default=20, help="Number of images")
    parser.add_argument("--device", type=int, default=0, help="Camera device ID")
    
    args = parser.parse_args()
    
    collect_faces(args.name, args.count, args.device)


if __name__ == "__main__":
    main()
