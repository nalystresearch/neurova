#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Face Detection Demo using Neurova

This example demonstrates real-time face detection using:
- YuNet face detector
- Camera capture
- GPU acceleration (if available)
"""

import neurova as nv
import numpy as np
import time
import argparse


def draw_face(image, face, color=(0, 255, 0), thickness=2):
    """Draw face bounding box and landmarks on image."""
    
    # Bounding box
    x, y, w, h = face[:4].astype(int)
    nv.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Landmarks (right eye, left eye, nose, right mouth, left mouth)
    landmarks = face[4:14].reshape(5, 2).astype(int)
    landmark_colors = [
        (255, 0, 0),    # Right eye - Blue
        (0, 0, 255),    # Left eye - Red
        (0, 255, 0),    # Nose - Green
        (255, 255, 0),  # Right mouth - Cyan
        (0, 255, 255),  # Left mouth - Yellow
    ]
    
    for i, (lm, lm_color) in enumerate(zip(landmarks, landmark_colors)):
        nv.circle(image, tuple(lm), 3, lm_color, -1)
    
    # Confidence score
    confidence = face[14]
    label = f"{confidence:.2f}"
    nv.putText(image, label, (x, y - 10),
               nv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    parser = argparse.ArgumentParser(description='Face Detection Demo')
    parser.add_argument('--model', type=str, default='yunet.onnx',
                        help='Path to YuNet model file')
    parser.add_argument('--input', type=str, default='0',
                        help='Input source (camera ID or video file)')
    parser.add_argument('--score-threshold', type=float, default=0.9,
                        help='Score threshold for face detection')
    parser.add_argument('--nms-threshold', type=float, default=0.3,
                        help='NMS threshold')
    parser.add_argument('--top-k', type=int, default=5000,
                        help='Max number of faces to detect')
    parser.add_argument('--save', type=str, default='',
                        help='Save output video to file')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Neurova Face Detection Demo")
    print("=" * 50)
    
    # Initialize Face Detector
    
    print(f"\nLoading face detector model: {args.model}")
    
    try:
        detector = nv.dnn.FaceDetectorYN.create(
            model=args.model,
            config="",
            input_size=(320, 320),
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            top_k=args.top_k,
            backend_id=nv.dnn.DNN_BACKEND_DEFAULT,
            target_id=nv.dnn.DNN_TARGET_CPU
        )
        print("Face detector loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease download the YuNet model from the Neurova models directory.")
        return
    
    # Open Video Source
    
    # Determine input source
    if args.input.isdigit():
        source = int(args.input)
        print(f"\nOpening camera {source}")
    else:
        source = args.input
        print(f"\nOpening video file: {source}")
    
    cap = nv.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    frame_width = int(cap.get(nv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(nv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(nv.CAP_PROP_FPS) or 30
    
    print(f"Video: {frame_width}x{frame_height} @ {fps} FPS")
    
    # Set detector input size to match video
    detector.setInputSize((frame_width, frame_height))
    
    # Initialize Video Writer (optional)
    
    writer = None
    if args.save:
        fourcc = nv.VideoWriter_fourcc(*'mp4v')
        writer = nv.VideoWriter(args.save, fourcc, fps, 
                                 (frame_width, frame_height))
        print(f"Saving output to: {args.save}")
    
    # Processing Loop
    
    print("\nStarting detection loop...")
    print("Press 'q' to quit, 's' to save screenshot")
    print()
    
    frame_count = 0
    total_time = 0
    face_counts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Detect faces
        retval, faces = detector.detect(frame)
        
        detection_time = (time.time() - start_time) * 1000
        total_time += detection_time
        
        # Count faces in this frame
        num_faces = faces.shape[0] if faces is not None else 0
        face_counts.append(num_faces)
        
        # Draw faces
        if faces is not None:
            for face in faces:
                draw_face(frame, face)
        
        # Draw stats
        avg_fps = frame_count / (total_time / 1000) if total_time > 0 else 0
        
        stats = [
            f"FPS: {avg_fps:.1f}",
            f"Faces: {num_faces}",
            f"Detection: {detection_time:.1f}ms"
        ]
        
        for i, stat in enumerate(stats):
            nv.putText(frame, stat, (10, 30 + i * 25),
                       nv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        nv.imshow("Face Detection", frame)
        
        # Save video frame
        if writer is not None:
            writer.write(frame)
        
        # Handle keyboard input
        key = nv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"face_screenshot_{frame_count}.jpg"
            nv.imwrite(filename, frame)
            print(f"Saved screenshot: {filename}")
    
    # Cleanup and Statistics
    
    cap.release()
    if writer is not None:
        writer.release()
    nv.destroyAllWindows()
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Session Statistics")
    print("=" * 50)
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time / 1000:.2f} seconds")
    print(f"Average FPS: {frame_count / (total_time / 1000):.2f}")
    print(f"Average detection time: {total_time / frame_count:.2f} ms")
    
    if face_counts:
        print(f"\nFace detection statistics:")
        print(f"  Max faces in frame: {max(face_counts)}")
        print(f"  Avg faces per frame: {sum(face_counts) / len(face_counts):.2f}")
        print(f"  Frames with faces: {sum(1 for c in face_counts if c > 0)}/{len(face_counts)}")


class FaceRecognitionDemo:
    """Extended demo with face recognition capabilities."""
    
    def __init__(self, detector_model, recognizer_model):
        """Initialize detector and recognizer."""
        
        # Face detector
        self.detector = nv.dnn.FaceDetectorYN.create(
            model=detector_model,
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000
        )
        
        # Face recognizer
        self.recognizer = nv.dnn.FaceRecognizerSF.create(
            model=recognizer_model,
            config=""
        )
        
        # Database of known faces
        self.known_faces = {}  # name -> feature vector
    
    def enroll_face(self, image, name):
        """Add a face to the database."""
        
        # Detect face
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        retval, faces = self.detector.detect(image)
        
        if faces is None or len(faces) == 0:
            print(f"No face found for {name}")
            return False
        
        # Use first detected face
        face = faces[0]
        
        # Align and crop
        aligned = self.recognizer.alignCrop(image, face)
        
        # Extract features
        features = self.recognizer.feature(aligned)
        
        # Store in database
        self.known_faces[name] = features
        print(f"Enrolled {name}")
        
        return True
    
    def recognize_face(self, image, face):
        """Recognize a detected face."""
        
        # Align and crop
        aligned = self.recognizer.alignCrop(image, face)
        
        # Extract features
        features = self.recognizer.feature(aligned)
        
        # Compare with known faces
        best_match = None
        best_score = 0
        
        for name, known_features in self.known_faces.items():
            score = self.recognizer.match(features, known_features,
                                           nv.dnn.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_match = name
        
        # Threshold for recognition
        if best_score > 0.363:  # Recommended cosine threshold
            return best_match, best_score
        else:
            return None, best_score
    
    def run_recognition(self, source=0):
        """Run face recognition on video source."""
        
        cap = nv.VideoCapture(source)
        if not cap.isOpened():
            print("Could not open video source")
            return
        
        frame_width = int(cap.get(nv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(nv.CAP_PROP_FRAME_HEIGHT))
        self.detector.setInputSize((frame_width, frame_height))
        
        print("Face Recognition Mode")
        print("Press 'e' to enroll a face")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            retval, faces = self.detector.detect(frame)
            
            if faces is not None:
                for face in faces:
                    # Try to recognize
                    name, score = self.recognize_face(frame, face)
                    
                    # Draw
                    x, y, w, h = face[:4].astype(int)
                    
                    if name is not None:
                        color = (0, 255, 0)  # Green for recognized
                        label = f"{name}: {score:.2f}"
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        label = f"Unknown: {score:.2f}"
                    
                    nv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    nv.putText(frame, label, (x, y-10),
                               nv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            nv.imshow("Face Recognition", frame)
            
            key = nv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                # Enroll mode
                if faces is not None and len(faces) > 0:
                    print("Enter name for enrollment: ", end="")
                    name = input()
                    self.enroll_face(frame, name)
        
        cap.release()
        nv.destroyAllWindows()


if __name__ == "__main__":
    main()
