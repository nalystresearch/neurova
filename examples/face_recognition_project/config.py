# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Configuration file for Face Recognition Project


Edit these settings to customize your project.
"""

import os
from pathlib import Path

# paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VAL_DIR = DATA_DIR / "validation"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

# Neurova data directory (for cascades)
NEUROVA_DATA = PROJECT_DIR.parent.parent / "neurova" / "data"
HAARCASCADES = NEUROVA_DATA / "haarcascades"

# detection settings
DETECTION_CONFIG = {
    # Method: 'haar', 'hog', 'cnn', 'lbp'
    'method': 'haar',
    
# haar cascade file
    'cascade_file': 'haarcascade_frontalface_default.xml',
    
# minimum face size
    'min_size': (80, 80),
    
# scale factor for multi-scale detection
    'scale_factor': 1.1,
    
# minimum neighbors for detection
    'min_neighbors': 5,
}

# recognition settings
RECOGNITION_CONFIG = {
    # Method: 'lbph', 'eigenface', 'fisherface', 'embedding'
    'method': 'lbph',
    
    # Face size for recognition (resize all faces to this)
    'face_size': (100, 100),
    
# lbph parameters
    'lbph_radius': 1,
    'lbph_neighbors': 8,
    'lbph_grid_x': 8,
    'lbph_grid_y': 8,
    
    # Confidence threshold (lower = stricter)
    'confidence_threshold': 100.0,
}

# training settings
TRAINING_CONFIG = {
    # Train/Test/Validation split
    'train_ratio': 0.70,
    'test_ratio': 0.15,
    'val_ratio': 0.15,
    
# random seed for reproducibility
    'random_seed': 42,
    
# minimum images per person
    'min_images_per_person': 5,
    
# data augmentation
    'augment': True,
    'augment_flip': True,
    'augment_brightness': True,
    'augment_rotation': True,
}

# webcam settings
WEBCAM_CONFIG = {
    # Camera device ID (0 = default webcam)
    'device': 0,
    
# resolution
    'width': 640,
    'height': 480,
    
    # FPS
    'fps': 30,
    
# number of frames to capture per person
    'frames_per_person': 20,
    
    # Delay between captures (seconds)
    'capture_delay': 0.3,
}

# display settings
DISPLAY_CONFIG = {
    # Colors (BGR format)
    'face_color': (0, 255, 0),      # Green
    'unknown_color': (0, 0, 255),    # Red
    'text_color': (255, 255, 255),   # White
    
# font
    'font_scale': 0.7,
    'font_thickness': 2,
    
# box thickness
    'box_thickness': 2,
}

# Create directories if they don't exist
for dir_path in [DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
