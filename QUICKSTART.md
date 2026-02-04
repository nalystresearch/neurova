# Neurova Quick Reference Guide

Complete API reference for Neurova modules and functions.

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

See [INSTALLATION.md](INSTALLATION.md) for platform-specific guides.

## Basic Usage

### Reading and Displaying Images

```python
import neurova as nv

# Read an image
img = nv.io.read_image("photo.jpg")

# Access image properties
print(f"Size: {img.width} x {img.height}")
print(f"Channels: {img.channels}")
print(f"Data type: {img.dtype}")
print(f"Color space: {img.color_space}")

# Get underlying numpy array
arr = img.data  # Read-only view
arr_copy = img.as_array()  # Copy
```

### Creating Images

```python
from neurova.core import create_blank_image, create_from_array, ColorSpace
import numpy as np

# Create blank image
blank = create_blank_image(width=640, height=480, channels=3, fill_value=255)

# Create from numpy array
arr = np.zeros((480, 640, 3), dtype=np.uint8)
img = create_from_array(arr, color_space=ColorSpace.RGB)
```

### Color Space Conversion

```python
from neurova.core.color import convert_color_space, to_grayscale
from neurova.core.constants import ColorSpace

# Convert to grayscale
gray = to_grayscale(img.data, from_space=ColorSpace.RGB)

# Convert between color spaces
hsv = convert_color_space(img.data, ColorSpace.RGB, ColorSpace.HSV)
lab = convert_color_space(img.data, ColorSpace.RGB, ColorSpace.LAB)
```

### Array Operations

```python
from neurova.core.array_ops import normalize, standardize, pad_array

# Normalize to [0, 1]
normalized = normalize(img.data, target_min=0.0, target_max=1.0)

# Standardize (zero mean, unit variance)
standardized = standardize(img.data)

# Pad array
padded = pad_array(img.data, pad_width=10, mode='reflect')
```

### Data Type Conversion

```python
from neurova.core.dtypes import convert_dtype, DataType

# Convert with scaling
float_img = convert_dtype(img.data, DataType.FLOAT32, scale=True)

# Convert without scaling
uint16_img = convert_dtype(img.data, DataType.UINT16, scale=False)
```

## Module Organization

### Core (`neurova.core`)

Image representation and color operations:

```python
from neurova.core import to_grayscale, convert_color_space, ColorSpace
from neurova.core.image import Image, create_blank_image
from neurova.core.array_ops import normalize, pad_array

# Color conversion
gray = to_grayscale(img)
hsv = convert_color_space(img, ColorSpace.BGR, ColorSpace.HSV)
lab = convert_color_space(img, ColorSpace.BGR, ColorSpace.LAB)

# Normalize array
normalized = normalize(arr, target_min=0.0, target_max=1.0)
```

### I/O (`neurova.io`)

Image and video input/output:

```python
from neurova import io

# Read/write images
img = io.read_image("photo.jpg")
io.write_image("output.png", img)

# Supported formats: PNG, JPG, BMP, PPM
```

### Transform (`neurova.transform`)

Geometric transformations:

```python
from neurova import transform

resized = transform.resize(img, (640, 480))
rotated = transform.rotate(img, angle=45)
flipped = transform.flip(img, axis='horizontal')
```

### Filters (`neurova.filters`)

Image filtering and edge detection:

```python
from neurova import filters

# Blur operations
blurred = filters.gaussian_blur(img, kernel_size=5)
blurred = filters.box_blur(img, kernel_size=3)
blurred = filters.median_blur(img, ksize=5)
blurred = filters.bilateral_filter(img, d=9)

# Edge detection
sobel_x, sobel_y = filters.sobel(gray)
edges = filters.canny_edges(gray, low=50, high=150)
laplacian = filters.laplacian(gray)

# Morphology
dilated = filters.dilate(binary_img, kernel_size=3)
eroded = filters.erode(binary_img, kernel_size=3)
opened = filters.morph_open(binary_img, kernel_size=3)
closed = filters.morph_close(binary_img, kernel_size=3)

# Sharpening
sharpened = filters.sharpen(img)
```

### Features (`neurova.features`)

Feature detection and descriptors:

```python
from neurova import features
from neurova.detection.hog import HOGDescriptor

# Corner detection
corners = features.detect_corners(gray, method='harris')

# HOG descriptors
hog = HOGDescriptor(cell_size=8, block_size=2)
descriptor = hog.compute(gray)
```

### Detection (`neurova.detection`)

Object and template detection:

```python
from neurova import detection

# Template matching
result = detection.template_match(img, template, method='correlation')

# Object detection with cascades
detector = detection.CascadeDetector('haarcascade_frontalface_default')
objects = detector.detect(gray)
```

### Face (`neurova.face`)

Face detection and recognition:

```python
from neurova.face import FaceDetector, FaceRecognizer

# Detection
detector = FaceDetector(method='haar')  # or 'lbp', 'dnn'
faces = detector.detect(img)

# Recognition
recognizer = FaceRecognizer(method='eigen')  # or 'fisher', 'lbph'
recognizer.train(face_images, labels)
label, confidence = recognizer.predict(face)
```

### Segmentation (`neurova.segmentation`)

Image segmentation:

```python
from neurova import segmentation

# Thresholding
binary = segmentation.threshold(gray, thresh=127)
binary = segmentation.adaptive_threshold(gray, block_size=11)
_, binary = segmentation.otsu_threshold(gray)

# Contours
contours = segmentation.find_contours(binary)
mask = segmentation.draw_contours(img, contours)

# Watershed
markers = segmentation.watershed(img, markers)
```

### Machine Learning (`neurova.ml`)

Classification, regression, and clustering:

```python
from neurova.ml import (
    KNearestNeighbors, NaiveBayes, DecisionTree, RandomForest, SVM,
    LinearRegression, RidgeRegression,
    KMeans, DBSCAN,
    PCA
)

# Classification
knn = KNearestNeighbors(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

# Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### Neural Networks (`neurova.neural`)

Deep learning layers and training:

```python
from neurova.neural import layers, Tensor, optim, losses

# Build model
model = layers.Sequential([
    layers.Linear(784, 256),
    layers.ReLU(),
    layers.Dropout(0.2),
    layers.Linear(256, 128),
    layers.ReLU(),
    layers.Linear(128, 10),
    layers.Softmax()
])

# Convolutional layers
conv_model = layers.Sequential([
    layers.Conv2D(in_channels=1, out_channels=32, kernel_size=3),
    layers.ReLU(),
    layers.MaxPool2D(kernel_size=2),
    layers.Flatten(),
    layers.Linear(32 * 13 * 13, 10)
])

# Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = losses.CrossEntropyLoss()

for epoch in range(epochs):
    output = model.forward(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Datasets (`neurova.datasets`)

Built-in datasets and sample images:

```python
from neurova import datasets

# Tabular data
iris = datasets.load_iris()
boston = datasets.load_boston_housing()
titanic = datasets.load_titanic()

# Time series
passengers = datasets.load_air_passengers()
sunspots = datasets.load_sunspots()

# Clustering
customers = datasets.load_mall_customers()
penguins = datasets.load_penguins()

# Images
img = datasets.load_sample_image('fruits')
img = datasets.load_sample_image('lena')
(train, test) = datasets.load_fashion_mnist()

# Cascade paths
haar_path = datasets.get_haarcascade('frontalface_default')
lbp_path = datasets.get_lbpcascade('frontalface')

# List available datasets
available = datasets.list_datasets()
```

### Augmentation (`neurova.augmentation`)

Data augmentation for training:

```python
from neurova import augmentation

# Create augmentation pipeline
pipeline = augmentation.Pipeline([
    augmentation.RandomRotation(degrees=15),
    augmentation.RandomFlip(horizontal=True),
    augmentation.RandomBrightness(factor=0.2),
    augmentation.RandomCrop(size=(224, 224))
])

augmented = pipeline(img)
```

### Video (`neurova.video`)

Video processing:

```python
from neurova.io import VideoCapture

cap = VideoCapture(0)  # Webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame
cap.release()
```

## Error Handling

```python
from neurova.core.errors import (
    NeurovaError,
    ImageError,
    IOError,
    ValidationError
)

try:
    img = nv.io.read_image("nonexistent.jpg")
except IOError as e:
    print(f"Failed to read image: {e}")

try:
    # Invalid color space
    img = Image(data, color_space="INVALID")
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Performance Tips

### Use NumPy Vectorization

```python
# Good: Vectorized operation
result = img * 1.5

# Avoid: Python loops
for i in range(height):
    for j in range(width):
        result[i, j] = img[i, j] * 1.5
```

### Minimize Copies

```python
# Use view when possible
view = img.data  # No copy

# Only copy when needed
copy = img.copy()
```

### Use Appropriate Data Types

```python
# For storage and display
uint8_img = img.astype(np.uint8)  # 0-255 range

# For computation
float32_img = img.astype(np.float32) / 255.0  # 0.0-1.0 range
```

## Examples

See the `examples/` directory for complete tutorials:

```bash
python examples/chapter_01_getting_started.py
python examples/chapter_07_machine_learning.py
python examples/chapter_08_neural_networks.py
```

## Getting Help

- Documentation: [DOCS_INDEX.md](DOCS_INDEX.md)
- GitHub Issues: https://github.com/nalystresearch/neurova/issues

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Made by @squid consultancy group (scg)
