# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
===============================================================================
                        NEUROVA LIBRARY EXAMPLES
                    Complete Feature Documentation
===============================================================================

Welcome to the Neurova library examples! This collection demonstrates all
features of the Neurova computer vision and machine learning library.

CHAPTER OVERVIEW
================

Chapter 1: Getting Started
    - Installation and imports
    - Version information
    - Device configuration
    - Array operations
    - Image loading basics

Chapter 2: Image Transforms
    - Color space conversion
    - Resizing and scaling
    - Rotation and flipping
    - Cropping and padding
    - Affine transformations

Chapter 3: Image Filters
    - Gaussian blur
    - Median filter
    - Bilateral filter
    - Sharpening
    - Edge detection (Sobel, Canny)
    - Morphological operations

Chapter 4: Feature Extraction
    - HOG descriptors
    - LBP features
    - Harris corners
    - Image gradients
    - GLCM texture features
    - Statistical features

Chapter 5: Object Detection
    - Haar cascade detection
    - LBP cascade detection
    - HOG + SVM detection
    - Multi-scale detection
    - Non-maximum suppression

Chapter 6: Face Detection & Recognition
    - FaceDetector class
    - FaceRecognizer (LBPH, EigenFace, FisherFace)
    - FaceTrainer
    - FaceDataset
    - Complete pipelines

Chapter 7: Machine Learning
    - Dataset loading
    - Data preprocessing
    - KNN, Decision Tree, Naive Bayes, Logistic Regression
    - Linear Regression
    - K-Means, Hierarchical, DBSCAN clustering
    - PCA dimensionality reduction
    - Model evaluation metrics

Chapter 8: Neural Networks
    - Dense, Conv2D, MaxPool2D layers
    - Activation functions
    - Loss functions
    - Optimizers (SGD, Adam, RMSprop)
    - Sequential models
    - CNN architectures
    - Autoencoders

Chapter 9: Dataset Management
    - Built-in datasets (iris, titanic, boston, diabetes, wine)
    - Time series data
    - Image datasets (Fashion-MNIST)
    - Custom datasets
    - Data augmentation
    - Data loaders

Chapter 10: Video Processing
    - VideoCapture
    - Frame processing
    - Motion detection
    - Background subtraction
    - Optical flow
    - Video recording

Chapter 11: Image Segmentation
    - Binary thresholding
    - Otsu's method
    - Adaptive thresholding
    - Connected components
    - K-Means segmentation
    - Watershed
    - Contour detection
    - Region growing

Chapter 12: GPU & Performance
    - Device configuration
    - Memory management
    - Benchmarking
    - Batch processing
    - Parallel processing
    - Caching
    - Optimization tips

RUNNING EXAMPLES
================

To run any chapter:

    python examples/chapter_XX_name.py

For example:
    python examples/chapter_01_getting_started.py
    python examples/chapter_07_machine_learning.py

AVAILABLE DATASETS
==================

Neurova includes these datasets for testing:

Tabular (neurova/data/tabular/):
    - iris.csv
    - titanic.csv
    - boston-housing.csv
    - diabetes.csv
    - wine.csv

Time Series (neurova/data/timeseries/):
    - air-passengers.csv
    - daily-temperatures.csv
    - sunspots.csv

Clustering (neurova/data/clustering/):
    - mall-customers.csv
    - penguins.csv

Image Data:
    - fashion-mnist/
    - movielens-100k/

Detection Data:
    - haarcascades/
    - lbpcascades/
    - hogcascades/

QUICK START
===========

    import neurova as nv
    from neurova import datasets, filters, detection

# load dataset
    iris = datasets.load_iris()
    
# load image
    image = nv.imread('photo.jpg')
    
# apply filter
    blurred = filters.gaussian_blur(image, kernel_size=5)
    
# detect faces
    from neurova.face import FaceDetector
    detector = FaceDetector(method='haar')
    faces = detector.detect(image)

Author: Neurova Team
Version: See neurova.__version__
License: MIT
===============================================================================
"""

import os
import sys

def main():
    """Run the examples index."""
    print(__doc__)
    
# list available chapters
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    chapters = sorted([
        f for f in os.listdir(examples_dir) 
        if f.startswith('chapter_') and f.endswith('.py')
    ])
    
    print("\nAvailable Chapter Files:")
    print("-" * 50)
    for chapter in chapters:
        print(f"   {chapter}")
    
    print("\n" + "-" * 50)
    print("Run with: python examples/<chapter_file>.py")
    print("=" * 50)

if __name__ == '__main__':
    main()
