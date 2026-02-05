# neurova library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)

"""
Configuration file for Object Detection Project

Edit these settings to customize your detector.
All paths and parameters can be changed here.
"""

import os
from pathlib import Path

# paths
# these are the folders where your data and models are stored
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VAL_DIR = DATA_DIR / "validation"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
TEMPLATES_DIR = PROJECT_DIR / "templates"

# neurova data directory for cascade files
NEUROVA_DATA = PROJECT_DIR.parent.parent / "neurova" / "data"
HAARCASCADES = NEUROVA_DATA / "haarcascades"
LBPCASCADES = NEUROVA_DATA / "lbpcascades"
HOGCASCADES = NEUROVA_DATA / "hogcascades"

# detection settings
# these control how the detector finds objects
DETECTION_CONFIG = {
    # Method to use: haar, hog, template, sliding_window, or custom
    'method': 'haar',
    
    # What to detect: face, eye, smile, body, upperbody, cat, or custom
    'target': 'face',
    
    # Minimum confidence score to accept a detection (0 to 1)
    'confidence_threshold': 0.5,
    
    # Threshold for removing duplicate detections (0 to 1)
    'nms_threshold': 0.3,
    
# smallest object size to detect in pixels
    'min_size': (30, 30),
    
    # Largest object size to detect (None means no limit)
    'max_size': None,
    
    # How much to scale the image at each step (1.1 means 10% smaller each step)
    'scale_factor': 1.1,
    
# how many neighbor detections needed to confirm a detection
    'min_neighbors': 5,
}

# cascade files
# pre-trained detectors included with neurova
CASCADE_FILES = {
    'face': 'haarcascade_frontalface_default.xml',
    'face_alt': 'haarcascade_frontalface_alt.xml',
    'face_alt2': 'haarcascade_frontalface_alt2.xml',
    'profile': 'haarcascade_profileface.xml',
    'eye': 'haarcascade_eye.xml',
    'eye_glasses': 'haarcascade_eye_tree_eyeglasses.xml',
    'smile': 'haarcascade_smile.xml',
    'body': 'haarcascade_fullbody.xml',
    'upperbody': 'haarcascade_upperbody.xml',
    'lowerbody': 'haarcascade_lowerbody.xml',
    'cat': 'haarcascade_frontalcatface.xml',
    'cat_ext': 'haarcascade_frontalcatface_extended.xml',
}

# training settings
# these control how the model learns from your data
TRAINING_CONFIG = {
    # How to split your data (should add up to 1.0)
    'train_ratio': 0.70,
    'test_ratio': 0.15,
    'val_ratio': 0.15,
    
# random seed for reproducible results
    'random_seed': 42,
    
    # HOG feature parameters (for pedestrian detection)
    'hog_win_size': (64, 128),
    'hog_block_size': (16, 16),
    'hog_block_stride': (8, 8),
    'hog_cell_size': (8, 8),
    'hog_nbins': 9,
    
# window sizes for sliding window detection
    'window_sizes': [(64, 64), (128, 128), (256, 256)],
    'window_stride': 16,
    
# number of negative samples to extract per image
    'neg_samples_per_image': 10,
    
# data augmentation options
    'augment': True,
    'augment_flip': True,
    'augment_scale': True,
}

# webcam settings
# settings for live camera detection
WEBCAM_CONFIG = {
    # Camera device number (0 is usually the built-in camera)
    'device': 0,
    
# video resolution
    'width': 640,
    'height': 480,
    
# frames per second
    'fps': 30,
}

# display settings
# how detections are shown on screen
DISPLAY_CONFIG = {
    # Colors for different classes (in BGR format)
    'colors': {
        'default': (0, 255, 0),    # Green
        'face': (0, 255, 0),       # Green
        'eye': (255, 0, 0),        # Blue
        'body': (0, 255, 255),     # Yellow
        'cat': (255, 0, 255),      # Magenta
        'unknown': (0, 0, 255),    # Red
    },
    
# text settings
    'font_scale': 0.6,
    'font_thickness': 2,
    
# box line thickness
    'box_thickness': 2,
    
# show confidence scores on detections
    'show_confidence': True,
}

# class names
# add your own class names here for custom detection
CLASS_NAMES = [
    'background',  # Class 0 is always background
    'object',      # Add your class names here
]

# create directories if they do not exist
for dir_path in [DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR, MODELS_DIR, REPORTS_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    
for subdir in ['images', 'annotations']:
    (TRAIN_DIR / subdir).mkdir(parents=True, exist_ok=True)
    (TEST_DIR / subdir).mkdir(parents=True, exist_ok=True)
    (VAL_DIR / subdir).mkdir(parents=True, exist_ok=True)

# neurova library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)
