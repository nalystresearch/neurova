# Changelog

All notable changes to Neurova will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.6] - 2025-02-05

**Initial Public Release**

Complete computer vision and deep learning toolkit with GPU acceleration and built-in datasets.

### Languages and Technologies

- **Python** - Primary API and high-level interfaces (3.8 - 3.14 support)
- **C++** - Native performance modules for image processing and detection
- **CUDA** - GPU acceleration via CuPy backend for NVIDIA GPUs
- **NumPy** - Core numerical computations
- **TFLite** - Lightweight neural network inference

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
- Face detection and recognition (Haar cascades with native C++ implementation)
- BlazeFace detector with TFLite inference
- Template matching
- Object detection with cascade classifiers
- Watershed segmentation
- Connected component labeling
- Contour detection and analysis

#### Deep Learning

- Neural network layers (Linear, Conv2D, Conv1D, MaxPool2D, AvgPool2D)
- Dropout, BatchNorm, LayerNorm
- RNN, LSTM, GRU
- Attention mechanisms, Embedding layers
- Activation functions (ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU, Swish)
- Loss functions (MSE, CrossEntropyLoss, BCE, Huber Loss)
- Optimizers (SGD, Adam, AdamW, RMSprop)
- Automatic differentiation with gradient tracking

#### Machine Learning

- Classification (KNN, Naive Bayes, Decision Trees, Random Forest, SVM)
- Regression (Linear, Ridge, Polynomial)
- Clustering (K-Means, DBSCAN, Hierarchical)
- Dimensionality reduction (PCA, t-SNE)
- Model evaluation and cross-validation

#### 100+ Neural Network Architectures

- CNN: LeNet, AlexNet, VGG, ResNet, DenseNet, MobileNet, EfficientNet
- Transformers: BERT, GPT, T5, ViT, Swin Transformer
- Generative: GAN, WGAN, VAE, DCGAN, StyleGAN, Diffusion models
- RNN: LSTM, GRU, Seq2Seq, Bidirectional
- Graph: GCN, GAT, GraphSAGE
- RL: DQN, PPO, A2C, SAC

#### Built-in Datasets

- Tabular: Iris, Boston Housing, Titanic, Diabetes, Wine
- Time Series: Air Passengers, Daily Temperatures, Sunspots
- Clustering: Mall Customers, Penguins
- Images: Fashion-MNIST (70,000 images), sample images
- Cascade classifiers: Haar, LBP, HOG cascades included

#### GPU Acceleration

- Optional CuPy backend for NVIDIA GPU support
- Global device selection API
- Automatic CPU fallback when GPU unavailable
- 10-100x speedups on compatible hardware

### License

- Apache License 2.0
- Free for educational and academic use
- Commercial use requires license from Squid Consultancy Group (SCG)

---

## Links

- **PyPI**: https://pypi.org/project/neurova/
- **GitHub**: https://github.com/nalystresearch/neurova

---

Copyright 2025 Squid Consultancy Group (SCG)
