# Neurova

**Python package for image processing, computer vision, deep learning, and classical machine learning**

Most people who work with images and visual data end up in the same repeating cycle: you have an idea, you want to load an image, detect something, extract some numbers, feed them into a model, and see if it works. Instead you spend half the time hunting compatible versions, rewriting the same preprocessing code, and debugging why things break when you move from laptop to GPU.

Neurova exists to break that cycle. One install, one namespace, and you can go from raw image to trained model without constantly switching tools or fighting data format mismatches.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/nalystresearch/neurova/blob/main/LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-v0.0.1-blue.svg)](https://pypi.org/project/neurova/)

**Navigation:** [Installation](#installation) · [Quick Start](#quick-start) · [Modules](#modules) · [Built-in Datasets](#built-in-datasets) · [Examples](#examples) · [Documentation](#documentation)

---

## Why Neurova

- **Single install, everything included** — face detection cascades, sample images, Fashion-MNIST, tabular datasets all bundled
- **Consistent conventions** — same array shapes, color orders, and device handling across all modules
- **Classical and deep learning together** — swap histogram features for conv features without rewriting your pipeline
- **GPU acceleration optional** — add CuPy and training gets 10-100x faster, no code changes needed
- **Pure Python** — no compilation, works on Windows, macOS, Linux, Colab, anywhere Python runs

---

## Key Features

### Image Processing

- Color space conversions (RGB, BGR, HSV, LAB, YCrCb, XYZ)
- Geometric transforms (resize, rotate, flip, affine, perspective)
- Filtering (blur, sharpen, edge detection, morphology)
- Segmentation (thresholding, watershed, contours)

### Computer Vision

- Feature detection (Harris corners, HOG descriptors)
- Face detection and recognition (Haar, LBP, DNN-based)
- Object detection with cascade classifiers
- Template matching and tracking

### Deep Learning

- Neural network layers (Linear, Conv2D, MaxPool2D, ReLU, Softmax)
- Tensor operations with automatic differentiation
- Optimizers (SGD, Adam, RMSprop)
- Loss functions (MSE, CrossEntropy, BCE)

### Machine Learning

- Classification (KNN, Naive Bayes, Decision Trees, Random Forest, SVM)
- Regression (Linear, Ridge, Polynomial)
- Clustering (KMeans, DBSCAN, Hierarchical)
- Dimensionality reduction (PCA, t-SNE)

### Built-in Datasets

- Tabular: Iris, Titanic, Boston Housing, Wine, Diabetes
- Time Series: Air Passengers, Sunspots, Daily Temperatures
- Clustering: Mall Customers, Penguins
- Images: Fashion-MNIST, sample images (fruits, lena, building, etc.)
- Cascade classifiers: Haar, LBP, HOG cascades included

### GPU Acceleration

- Optional CuPy backend for NVIDIA GPU support
- Automatic device selection
- 10-100x speedups on compatible hardware

## Installation

```bash
# Basic installation
pip install neurova

# With GPU support (NVIDIA)
pip install neurova cupy-cuda12x

# Development installation
git clone https://github.com/nalystresearch/neurova.git
cd neurova
pip install -e ".[dev]"
```

See [INSTALLATION.md](INSTALLATION.md) for platform-specific guides and GPU setup.

## Quick Start

### Image Processing

```python
import neurova as nv
from neurova import io, transform, filters, core

# Load image
img = io.read_image("photo.jpg")

# Color conversion
gray = core.to_grayscale(img)
hsv = core.convert_color_space(img, core.ColorSpace.BGR, core.ColorSpace.HSV)

# Apply filters
blurred = filters.gaussian_blur(img, kernel_size=5)
edges = filters.canny_edges(gray, low=50, high=150)

# Save result
io.write_image("edges.jpg", edges)
```

### Using Built-in Datasets

```python
from neurova import datasets

# Load sample images (options: 'fruits', 'lena', 'building', etc.)
img = datasets.load_sample_image('fruits')

# Load tabular data
iris = datasets.load_iris()
boston = datasets.load_boston_housing()

# Load Fashion-MNIST for neural network training
(train_images, train_labels), (test_images, test_labels) = datasets.load_fashion_mnist()
```

### Machine Learning

```python
from neurova import datasets
from neurova.ml import KNearestNeighbors, KMeans, PCA

# Load data
df = datasets.load_iris()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species'].astype('category').cat.codes.values

# Classification
knn = KNearestNeighbors(n_neighbors=3)
knn.fit(X, y)
predictions = knn.predict(X[:5])

# Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### Neural Networks

```python
from neurova import datasets
from neurova.neural import layers, Tensor, optim

# Load Fashion-MNIST
(train_images, train_labels), _ = datasets.load_fashion_mnist()

# Create model
model = layers.Sequential([
    layers.Linear(784, 128),
    layers.ReLU(),
    layers.Linear(128, 10),
    layers.Softmax()
])

# Training loop
optimizer = optim.SGD(model.parameters(), lr=0.01)
for batch in training_data:
    x, y = batch
    output = model.forward(x)
    loss = compute_loss(output, y)
    optimizer.step()
```

### Face Detection

```python
from neurova import datasets
from neurova.face import FaceDetector

# Load sample image
img = datasets.load_sample_image('lena')

# Detect faces
detector = FaceDetector(method='haar')
faces = detector.detect(img)

for (x, y, w, h) in faces:
    print(f"Face at ({x}, {y}) size {w}x{h}")
```

## Modules

| Module                 | Description                                   |
| ---------------------- | --------------------------------------------- |
| `neurova.core`         | Image class, color spaces, array operations   |
| `neurova.io`           | Image and video I/O                           |
| `neurova.transform`    | Geometric transformations                     |
| `neurova.filters`      | Convolution, blur, edge detection, morphology |
| `neurova.features`     | Corner, edge, keypoint detection              |
| `neurova.detection`    | Object and template detection                 |
| `neurova.face`         | Face detection and recognition                |
| `neurova.segmentation` | Thresholding, watershed, contours             |
| `neurova.neural`       | Neural network layers and training            |
| `neurova.ml`           | Machine learning algorithms                   |
| `neurova.datasets`     | Built-in datasets and sample images           |
| `neurova.augmentation` | Data augmentation pipelines                   |
| `neurova.video`        | Video processing and analysis                 |
| `neurova.calibration`  | Camera calibration and 3D geometry            |
| `neurova.nvc`          | Computer vision utility functions             |

## Built-in Datasets

Neurova includes ready-to-use datasets for testing and development:

### Tabular Data

```python
from neurova import datasets

# Iris flower classification
datasets.load_iris()
# Boston housing regression
datasets.load_boston_housing()
# Titanic survival classification
datasets.load_titanic()
# Diabetes prediction
datasets.load_diabetes()
# Wine classification
datasets.load_wine()
```

### Time Series

```python
# Monthly airline passengers
datasets.load_air_passengers()
# Daily temperature readings
datasets.load_daily_temperatures()
# Monthly sunspot counts
datasets.load_sunspots()
```

### Clustering

```python
# Customer segmentation
datasets.load_mall_customers()
# Palmer penguins dataset
datasets.load_penguins()
```

### Images

```python
# Sample RGB image
datasets.load_sample_image('fruits')
# Classic test image
datasets.load_sample_image('lena')
# Architectural features
datasets.load_sample_image('building')
# Calibration pattern
datasets.load_sample_image('chessboard')
# 70,000 fashion images
datasets.load_fashion_mnist()
```

### Cascade Classifiers

```python
# Face detection
datasets.get_haarcascade('frontalface_default')
# Eye detection
datasets.get_haarcascade('eye')
# LBP face detection
datasets.get_lbpcascade('frontalface')
```

## Examples

The `examples/` directory contains comprehensive tutorials:

| Chapter | Topic            | Description                               |
| ------- | ---------------- | ----------------------------------------- |
| 01      | Getting Started  | Basic setup and image operations          |
| 02      | Image Transforms | Color spaces, geometric transforms        |
| 03      | Filters          | Blur, sharpen, edge detection, morphology |
| 04      | Features         | Corner detection, HOG descriptors         |
| 05      | Detection        | Object and template detection             |
| 06      | Face             | Face detection and recognition            |
| 07      | Machine Learning | Classification, regression, clustering    |
| 08      | Neural Networks  | Building and training neural networks     |
| 09      | Datasets         | Using built-in datasets                   |
| 10      | Video            | Video processing and analysis             |
| 11      | Segmentation     | Thresholding, contours, watershed         |
| 12      | GPU Performance  | GPU acceleration with CuPy                |

### Running Examples

```bash
# Run any chapter example
python examples/chapter_07_machine_learning.py

# Run face detection project
python examples/face_recognition_project/01_collect_faces.py
```

## Dependencies

### Required

- **Python** >= 3.8
- **NumPy** >= 1.19.0

### Optional

- **Pillow**: Extended image format support
- **pandas**: For tabular dataset loading
- **CuPy**: GPU acceleration

## Documentation

| Document                                 | Description                      |
| ---------------------------------------- | -------------------------------- |
| [INSTALLATION.md](INSTALLATION.md)       | Installation and setup guide     |
| [QUICKSTART.md](QUICKSTART.md)           | Quick reference and API overview |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Development guide                |
| [ARCHITECTURE.md](ARCHITECTURE.md)       | Module structure and design      |
| [DOCS_INDEX.md](DOCS_INDEX.md)           | Full documentation index         |

## Performance

Neurova is optimized using:

- NumPy vectorization for fast array operations
- Efficient memory management with minimal copies
- Parallel processing for batch operations
- Optional GPU acceleration via CuPy

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community standards
- [SECURITY.md](SECURITY.md) - Security reporting
- [SUPPORT.md](SUPPORT.md) - Getting help

## License

MIT License - Copyright (c) 2025 @squid consultancy group (scg)

See [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

Maintained by Squid Consultancy Group (SCG)
