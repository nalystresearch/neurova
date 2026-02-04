# Neurova Documentation

**Copyright (c) 2025 @squid consultancy group (scg). All rights reserved.**

## Getting Started

| Document                              | Description                            |
| ------------------------------------- | -------------------------------------- |
| [README](README.md)                   | Overview, features, and quick examples |
| [INSTALLATION](INSTALLATION.md)       | Installation and setup guide           |
| [QUICKSTART](QUICKSTART.md)           | Complete API reference                 |
| [GETTING_STARTED](GETTING_STARTED.md) | Development guide                      |
| [ARCHITECTURE](ARCHITECTURE.md)       | Module structure and design            |

## Modules

### Image Processing

| Module                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `neurova.core`         | Image class, color spaces, array operations      |
| `neurova.io`           | Image and video I/O                              |
| `neurova.transform`    | Geometric transformations (resize, rotate, flip) |
| `neurova.filters`      | Blur, sharpen, edge detection, morphology        |
| `neurova.segmentation` | Thresholding, contours, watershed                |

### Computer Vision

| Module                | Description                       |
| --------------------- | --------------------------------- |
| `neurova.features`    | Corner detection, HOG descriptors |
| `neurova.detection`   | Object and template detection     |
| `neurova.face`        | Face detection and recognition    |
| `neurova.video`       | Video capture and processing      |
| `neurova.calibration` | Camera calibration, 3D geometry   |

### Machine Learning & Deep Learning

| Module           | Description                                 |
| ---------------- | ------------------------------------------- |
| `neurova.ml`     | Classification, regression, clustering, PCA |
| `neurova.neural` | Neural network layers and training          |
| `neurova.nn`     | Neural network utilities                    |
| `neurova.dnn`    | Deep neural network inference               |

### Data & Utilities

| Module                 | Description                         |
| ---------------------- | ----------------------------------- |
| `neurova.datasets`     | Built-in datasets and sample images |
| `neurova.augmentation` | Data augmentation pipelines         |
| `neurova.nvc`          | Computer vision utility functions   |
| `neurova.highgui`      | GUI functions for display           |

## Examples

The `examples/` directory contains comprehensive tutorials:

| Chapter | Topic                                                     |
| ------- | --------------------------------------------------------- |
| 01      | Getting Started - Basic setup and operations              |
| 02      | Image Transforms - Color spaces, geometric transforms     |
| 03      | Filters - Blur, sharpen, edge detection                   |
| 04      | Features - Corner detection, HOG                          |
| 05      | Detection - Object and template detection                 |
| 06      | Face - Face detection and recognition                     |
| 07      | Machine Learning - Classification, regression, clustering |
| 08      | Neural Networks - Building and training models            |
| 09      | Datasets - Using built-in datasets                        |
| 10      | Video - Video processing                                  |
| 11      | Segmentation - Thresholding, contours                     |
| 12      | GPU Performance - GPU acceleration                        |

### Projects

- `face_recognition_project/` - Complete face recognition system
- `object_detection_project/` - Object detection examples

## Built-in Datasets

### Tabular

- `load_iris()` - Iris flower classification
- `load_boston_housing()` - Housing price regression
- `load_titanic()` - Survival classification
- `load_diabetes()` - Medical prediction
- `load_wine()` - Wine classification

### Time Series

- `load_air_passengers()` - Monthly passengers
- `load_sunspots()` - Sunspot counts
- `load_daily_temperatures()` - Temperature readings

### Clustering

- `load_mall_customers()` - Customer segmentation
- `load_penguins()` - Palmer penguins

### Images

- `load_sample_image(name)` - Sample images (fruits, lena, building, etc.)
- `load_fashion_mnist()` - Fashion-MNIST dataset (70,000 images)

### Cascade Classifiers

- `get_haarcascade(name)` - Haar cascade files
- `get_lbpcascade(name)` - LBP cascade files
- `get_hogcascade(name)` - HOG cascade files

## Support

- [CONTRIBUTING](CONTRIBUTING.md) - Contribution guidelines
- [SUPPORT](SUPPORT.md) - Getting help
- [SECURITY](SECURITY.md) - Security policy
- [CHANGELOG](CHANGELOG.md) - Version history

---

Made by @squid consultancy group (scg)
