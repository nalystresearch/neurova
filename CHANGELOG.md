# Changelog

All notable changes to Neurova will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.1] - 2026-02-04

**Initial Stable Release**

The first stable release of Neurova - a complete computer vision and deep learning toolkit with GPU acceleration and built-in datasets.

### Features

#### Image Processing

- Color space conversions (RGB, BGR, HSV, LAB, YCrCb, XYZ, Grayscale)
- Geometric transformations (resize, rotate, flip, crop, warp, affine, perspective)
- Filtering operations (Gaussian, median, bilateral, box blur, sharpen)
- Edge detection (Canny, Sobel, Laplacian)
- Morphological operations (erosion, dilation, opening, closing)
- Image I/O (PNG, JPEG, BMP support)

#### Computer Vision

- Feature detection (Harris corners, HOG descriptors)
- Face detection and recognition (Haar, LBP, DNN-based)
- Template matching
- Object detection with cascade classifiers
- Watershed segmentation
- Connected component labeling
- Contour detection and analysis

#### Deep Learning

**Neural Network Layers:**

- Linear (fully connected), Conv2D, Conv1D
- MaxPool2D, AvgPool2D
- Dropout, BatchNorm, LayerNorm
- RNN, LSTM, GRU
- Attention mechanisms, Embedding layers

**Activation Functions:**

- ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, Swish

**Loss Functions:**

- MSE, CrossEntropyLoss, BCE, Huber Loss

**Optimizers:**

- SGD, Adam, AdamW, RMSprop

**Automatic Differentiation:**

- Tensor class with gradient tracking
- Reverse-mode automatic differentiation
- `.backward()` for backpropagation

#### Machine Learning

**Classification:**

- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)

**Regression:**

- Linear Regression
- Ridge Regression
- Polynomial Regression

**Clustering:**

- K-Means
- DBSCAN
- Hierarchical/Agglomerative Clustering

**Dimensionality Reduction:**

- PCA (Principal Component Analysis)
- t-SNE

**Utilities:**

- Train/test split
- Cross-validation
- Model evaluation metrics

#### Built-in Datasets

**Tabular Data:**

- Iris (classification)
- Boston Housing (regression)
- Titanic (classification)
- Diabetes (prediction)
- Wine (classification)

**Time Series:**

- Air Passengers
- Daily Temperatures
- Sunspots

**Clustering:**

- Mall Customers
- Penguins

**Images:**

- Fashion-MNIST (70,000 images)
- Sample images (fruits, lena, building, baboon, chessboard, sudoku)

**Cascade Classifiers:**

- Haar cascades (face, eye, body detection)
- LBP cascades
- HOG cascades

#### GPU Acceleration

- Optional CuPy backend for NVIDIA GPU support
- Global device selection API (`set_device`, `get_device`)
- Automatic CPU fallback when GPU unavailable
- 10-100x speedups on compatible hardware

#### Examples

12 comprehensive tutorial chapters:

- Getting Started
- Image Transforms
- Filters
- Features
- Detection
- Face Detection/Recognition
- Machine Learning
- Neural Networks
- Datasets
- Video Processing
- Segmentation
- GPU Performance

Complete projects:

- Face Recognition System
- Object Detection Examples

### Documentation

- README with comprehensive feature overview
- INSTALLATION guide for all platforms
- QUICKSTART with complete API reference
- ARCHITECTURE design document
- Example chapters with working code

### Infrastructure

- MIT License
- Python 3.8 - 3.13 support
- NumPy >= 1.19.0 (only required dependency)
- Type hints throughout codebase
- PyPI package distribution

---

## Links

- **PyPI**: https://pypi.org/project/neurova/
- **GitHub**: https://github.com/nalystresearch/neurova
- **Documentation**: https://github.com/nalystresearch/neurova/blob/main/DOCS_INDEX.md

---

Made by @squid consultancy group (scg)
