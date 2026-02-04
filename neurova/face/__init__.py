# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Neurova Face Detection and Recognition Module.

Provides comprehensive face detection, recognition, and training capabilities.

Usage:
    from neurova.face import FaceDetector, FaceRecognizer, FaceDataset
    
    # Quick detection with Haar cascade
    detector = FaceDetector(method='haar')
    faces = detector.detect(image)
    
    # Train custom face recognizer
    dataset = FaceDataset(train_dir='./train', test_dir='./test')
    recognizer = FaceRecognizer()
    recognizer.train(dataset)
    
    # Detect and recognize
    faces = detector.detect(image)
    for face in faces:
        name, confidence = recognizer.recognize(face)
"""

from neurova.face.detector import (
    FaceDetector,
    HaarCascadeDetector,
    LBPCascadeDetector,
    HOGDetector,
    DNNDetector,
    MediaPipeDetector,
)

from neurova.face.recognizer import (
    FaceRecognizer,
    LBPHRecognizer,
    EigenFaceRecognizer,
    FisherFaceRecognizer,
    EmbeddingRecognizer,
)

from neurova.face.dataset import (
    FaceDataset,
    WebcamDataCollector,
    create_face_dataset,
    split_dataset,
)

from neurova.face.trainer import (
    FaceTrainer,
    train_face_detector,
    train_face_recognizer,
)

from neurova.face.utils import (
    extract_faces,
    align_face,
    detect_landmarks,
    crop_face,
    draw_faces,
    save_faces,
    load_face,
    preprocess_face,
    compute_face_distance,
    verify_faces,
    augment_face,
    resize_face,
    to_grayscale,
    to_rgb,
)

__all__ = [
    # Detectors
    "FaceDetector",
    "HaarCascadeDetector",
    "LBPCascadeDetector",
    "HOGDetector",
    "DNNDetector",
    "MediaPipeDetector",
    # Recognizers
    "FaceRecognizer",
    "LBPHRecognizer",
    "EigenFaceRecognizer",
    "FisherFaceRecognizer",
    "EmbeddingRecognizer",
    # Dataset
    "FaceDataset",
    "WebcamDataCollector",
    "create_face_dataset",
    "split_dataset",
    # Training
    "FaceTrainer",
    "train_face_detector",
    "train_face_recognizer",
    # Utilities
    "extract_faces",
    "align_face",
    "detect_landmarks",
    "crop_face",
    "draw_faces",
    "save_faces",
    "load_face",
    "preprocess_face",
    "compute_face_distance",
    "verify_faces",
    "augment_face",
    "resize_face",
    "to_grayscale",
    "to_rgb",
]
