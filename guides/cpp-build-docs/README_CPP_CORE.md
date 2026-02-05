# Neurova C++ Core

##  High-Performance Computer Vision & Deep Learning Library

The Neurova C++ Core provides blazing-fast implementations of:

- **Tensor Operations** - SIMD-optimized (ARM NEON)
- **Image Processing** - Grayscale, resize, blur, etc.
- **Neural Networks** - Activations, losses
- **Machine Learning** - Metrics, utilities

## Performance

| Operation       | Size               | Time     | FPS/GFLOPS  |
| --------------- | ------------------ | -------- | ----------- |
| RGB to Gray     | 1280×720           | 0.44 ms  | 2275 FPS    |
| Resize          | 1280×720 -> 640×360 | 1.77 ms  | 565 FPS     |
| Gaussian Blur   | 1280×720, k=5      | 8.01 ms  | 124 FPS     |
| Matrix Multiply | 512×512            | 25.27 ms | 10.6 GFLOPS |
| ReLU            | 1M elements        | 0.13 ms  | -           |

## Quick Start

```python
# Import C++ core
from neurova.cpp_core import *

# Check availability
check_cpp_core()  #  C++ core working (v0.2.0, SIMD: ARM NEON)

# Tensor operations
t = Tensor.randn([100, 100])
print(t.mean(), t.std())

# Matrix multiply
a = Tensor.rand([128, 256])
b = Tensor.rand([256, 64])
c = a.matmul(b)

# Neural network functions
x = Tensor.randn([batch, features])
h = nn.relu(x)
h = nn.sigmoid(h)
out = nn.softmax(h)

# Image processing
import numpy as np
arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
img = Image(arr)
gray = imgproc.rgb_to_gray(img)
resized = imgproc.resize(img, 320, 240)
blurred = imgproc.gaussian_blur(img, 5, 1.0)
```

## Building

```bash
cd Neurova
./build_minimal.sh
```

Requirements:

- Python 3.12+
- pybind11 (`pip install pybind11`)
- clang++ (Xcode Command Line Tools)

## API Reference

### Tensor

```python
# Creation
Tensor.zeros([M, N])
Tensor.ones([M, N])
Tensor.randn([M, N])  # Normal distribution
Tensor.rand([M, N])   # Uniform [0, 1)
Tensor.arange(start, stop, step)
Tensor.eye(N)         # Identity matrix

# Properties
t.shape, t.size, t.ndim, t.dtype

# Operations
a + b, a - b, a * b, a / b
a + scalar, a * scalar
t.sum(), t.mean(), t.max(), t.min()
t.std(), t.var(), t.argmax(), t.argmin()

# Math
t.exp(), t.log(), t.sqrt(), t.pow(n)
t.abs(), t.sin(), t.cos(), t.tanh()
t.clamp(min, max)

# Shape
t.reshape([...])
t.squeeze(), t.unsqueeze(axis)
t.transpose()

# Matrix
a.matmul(b), a.dot(b)

# NumPy
arr = t.numpy()
t = Tensor.from_numpy(arr)
```

### Image

```python
# Creation
img = Image(width, height, channels, dtype)
img = Image(numpy_array)

# Properties
img.width, img.height, img.channels, img.dtype

# Conversion
arr = img.numpy()
tensor = img.to_tensor()
img_float = img.to(DType.FLOAT32)
```

### Image Processing (imgproc)

```python
gray = imgproc.rgb_to_gray(img)
rgb = imgproc.gray_to_rgb(gray)
resized = imgproc.resize(img, new_w, new_h)
cropped = imgproc.crop(img, x, y, w, h)
flipped = imgproc.flip(img, horizontal=True)
blurred = imgproc.gaussian_blur(img, ksize=5, sigma=1.0)
```

### Neural Network (nn)

```python
# Activations
nn.relu(x)
nn.leaky_relu(x, negative_slope=0.01)
nn.sigmoid(x)
nn.tanh(x)
nn.softmax(x)
nn.gelu(x)
nn.swish(x)

# Losses
nn.mse_loss(pred, target)
nn.cross_entropy(pred, target)
```

### Machine Learning (ml)

```python
ml.accuracy_score(y_true, y_pred)
ml.mean_squared_error(y_true, y_pred)
```

## Architecture

```
neurova/
 neurova_core.cpython-312-darwin.so  # Compiled C++ module
 cpp_core.py                         # Python wrapper
 __init__.py                         # Package init

src/
 neurova_minimal.cpp                 # Self-contained C++ source
```

## SIMD Support

- **ARM NEON** (Apple Silicon M-series) - Enabled
- **AVX2/AVX** (Intel/AMD) - Supported
- **SSE2** - Fallback

The build automatically detects and enables the best SIMD support for your CPU.

## License

Apache License 2.0 - Copyright (c) 2026 Squid Consultancy Group (SCG)
