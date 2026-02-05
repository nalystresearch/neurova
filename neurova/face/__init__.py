# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Neurova Face Detection and Recognition Module.

Pure Python implementation with NO deep learning dependencies.
Uses Haar cascade and feature-based detection methods.

Detection Methods:
    1. HaarCascadeDetector - Fast Haar cascade (recommended)
    2. LBPCascadeDetector - LBP-based detection
    3. HOGDetector - HOG + SVM detection
    4. NativeDetector - Pure Python feature-based (no deep learning libraries!)

Quick Start:
    >>> from neurova.face import NativeDetector
    >>> detector = NativeDetector()
    >>> faces = detector.detect(image)
    >>> print(f"Found {len(faces)} faces")

Alternative - Unified API:
    >>> from neurova.face import FaceDetector
    >>> detector = FaceDetector(method='haar')
    >>> faces = detector.detect(image)

Face Recognition:
    >>> from neurova.face import FaceRecognizer, FaceDataset
    >>> dataset = FaceDataset(train_dir='./train', test_dir='./test')
    >>> recognizer = FaceRecognizer(method='lbph')
    >>> recognizer.train(dataset.train_images, dataset.train_labels)
    >>> label, confidence = recognizer.predict(test_face)

Requirements:
    - numpy (required)
    - Pillow (required)
    - NO or other deep learning libraries needed!
"""

from pathlib import Path

# 
# Module Paths
# 

FACE_DIR = Path(__file__).parent

# Model download URL (backup if local model is missing)
BLAZE_FACE_MODEL_URL = (
    "https://storage.googleapis.com/native-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)

from neurova.face.detector import (
    FaceDetector,
    HaarCascadeDetector,
    LBPCascadeDetector,
    HOGDetector,
    DNNDetector,
    NativeDetector,
)

from neurova.face.tflite_detector import (
    TFLiteFaceDetector,
    is_tflite_available,
)

from neurova.face.blazeface_detector import (
    BlazeFaceDetector,
    is_blazeface_available,
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

# 
# Module Exports
# 

__all__ = [
    # Constants
    "FACE_DIR",
    "BLAZE_FACE_MODEL",
    "BLAZE_FACE_MODEL_URL",
    # Detectors (in recommended priority order)
    "NativeDetector",      # Fast GPU-accelerated (recommended)
    "BlazeFaceDetector",      # Pure Python TFLite
    "TFLiteFaceDetector",     # Generic TFLite
    "FaceDetector",           # Unified API with method selection
    "HaarCascadeDetector",    # CPU-only fallback
    "LBPCascadeDetector",     # LBP cascade
    "HOGDetector",            # HOG + SVM
    "DNNDetector",            # DNN-based
    # Availability checks
    "is_blazeface_available",
    "is_tflite_available",
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
    # Convenience functions
    "get_best_detector",
    "quick_detect",
]


# 
# Convenience Functions
# 

def get_best_detector(min_confidence: float = 0.5):
    """
    Get the best available face detector.
    
    Tries detectors in order of performance:
    1. NativeDetector (GPU-accelerated)
    2. BlazeFaceDetector (TFLite)
    3. HaarCascadeDetector (CPU fallback)
    
    Args:
        min_confidence: Minimum detection confidence.
        
    Returns:
        Best available detector instance.
        
    Example:
        >>> detector = get_best_detector()
        >>> faces = detector.detect(image)
    """
    # Try native backend first (fastest)
    try:
        return NativeDetector(min_confidence=min_confidence)
    except Exception:
        pass
    
    # Try BlazeFace
    if is_blazeface_available():
        try:
            return BlazeFaceDetector(min_confidence=min_confidence)
        except Exception:
            pass
    
    # Fallback to Haar cascade
    return HaarCascadeDetector()


def quick_detect(image, min_confidence: float = 0.5):
    """
    Quick face detection using the best available detector.
    
    Args:
        image: Input image (numpy array, BGR or RGB).
        min_confidence: Minimum detection confidence.
        
    Returns:
        List of (x, y, width, height, confidence) tuples.
        
    Example:
        >>> faces = quick_detect(image)
        >>> for x, y, w, h, conf in faces:
        ...     print(f"Face at ({x}, {y}) with confidence {conf:.2f}")
    """
    detector = get_best_detector(min_confidence)
    return detector.detect(image)
