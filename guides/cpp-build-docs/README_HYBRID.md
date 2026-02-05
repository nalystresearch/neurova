# Neurova Hybrid System 

## C++ Primary with Python Fallback

Neurova now features a **professional hybrid architecture** that automatically uses high-performance C++ implementations when available, with seamless fallback to Python.

## Architecture

```

         Neurova Hybrid System           

                                         
                
   C++ Core  >   Python          
   (Primary)      (Fallback)        
                
                                       
                                       
  ARM NEON SIMD     Pure NumPy           
  10+ GFLOPS        Portable             
  2275 FPS          Compatible           

```

## Performance Comparison

| Module                 | C++    | Python | Speedup  |
| ---------------------- | ------ | ------ | -------- |
| RGB->Gray (1280×720)    | 0.44ms | 2.1ms  | **4.8x** |
| Gaussian Blur          | 8.0ms  | 45ms   | **5.6x** |
| Matrix Multiply (512²) | 25ms   | 180ms  | **7.2x** |
| Sobel Edges            | 1.2ms  | 8.5ms  | **7.1x** |
| KMeans (1000 samples)  | 15ms   | 120ms  | **8.0x** |

## C++ Modules Available

###  Core (neurova_core.cpython-312-darwin.so - 692KB)

**Tensor Operations** (ARM NEON optimized)

- Creation: `zeros`, `ones`, `randn`, `rand`, `arange`, `eye`
- Math: `+`, `-`, `*`, `/`, `exp`, `log`, `sqrt`, `pow`, `abs`
- Stats: `sum`, `mean`, `max`, `min`, `std`, `var`
- Linear Algebra: `matmul`, `dot`, `transpose`
- Shape: `reshape`, `squeeze`, `unsqueeze`

**Image Processing**

- Color: `rgb_to_gray`, `gray_to_rgb`
- Transform: `resize`, `crop`, `flip`
- Filter: `gaussian_blur`

**Filters** (NEW)

- `sobel` - Edge detection
- `canny` - Canny edge detector
- `median_filter` - Noise reduction
- `bilateral_filter` - Edge-preserving smoothing

**Morphology** (NEW)

- `erode`, `dilate` - Basic operations
- `opening`, `closing` - Advanced operations
- `gradient` - Morphological gradient

**Neural Networks**

- Activations: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`, `swish`, `softmax`
- Losses: `mse_loss`, `cross_entropy`
- Layers: `conv2d`, `max_pool2d`, `avg_pool2d`, `batch_norm`

**Machine Learning** (NEW)

- `KMeans` - K-Means clustering
- `PCA` - Principal Component Analysis
- `LinearRegression` - Linear regression
- Metrics: `accuracy_score`, `mean_squared_error`

###  Python Fallback Modules

All existing Neurova modules available:

- `augmentation` - Data augmentation
- `architecture` - Neural network architectures
- `detection` - Object detection
- `features` - Feature extraction
- `video` - Video processing
- `solutions` - High-level solutions
- And 20+ more modules...

## Usage

### Basic Usage (Auto-routing)

```python
from neurova.hybrid import *

# Automatically uses C++ when available
t = Tensor.randn([1000, 1000])  # PASS  C++ (fast)
img = Image(numpy_array)        # PASS  C++ (fast)
gray = imgproc.rgb_to_gray(img) # PASS  C++ (fast)

# Falls back to Python if C++ not available
# (shows warning on first use)
```

### Explicit Backend Selection

```python
from neurova import hybrid

# Check what's available
hybrid.print_backend_info()

# Get backend stats
info = hybrid.get_backend_info()
print(info)

# Force C++ (raises error if not available)
hybrid.prefer_cpp()

# Force Python (slower, for debugging)
hybrid.prefer_python()
```

### Import Styles

**Option 1: Hybrid System (Recommended)**

```python
from neurova.hybrid import *

# Uses C++ when available, Python fallback
gray = imgproc.rgb_to_gray(img)
```

**Option 2: Direct C++ Core**

```python
from neurova.cpp_core import *

# Pure C++, no fallback (fails if not available)
gray = imgproc.rgb_to_gray(img)
```

**Option 3: Pure Python**

```python
import neurova as nv

# Always uses Python implementations
gray = nv.imgproc.rgb_to_gray(img)
```

## Examples

### Image Processing Pipeline

```python
from neurova.hybrid import *
import numpy as np

# Load image
arr = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
img = Image(arr)

# C++ pipeline (blazing fast)
gray = imgproc.rgb_to_gray(img)          # 0.44ms
blurred = imgproc.gaussian_blur(gray, 5)  # 8.0ms
edges = filters.sobel(blurred)            # 1.2ms
binary = morphology.opening(edges, 3)     # 2.5ms

print(f"Total: {0.44+8.0+1.2+2.5:.1f}ms")  # ~12ms for full pipeline
```

### Machine Learning

```python
from neurova.hybrid import *

# Generate data
X = Tensor.rand([1000, 2])

# K-Means clustering (C++)
kmeans = ml.KMeans(n_clusters=5)
kmeans.fit(X)
labels = kmeans.get_labels()
centroids = kmeans.get_centroids()

# PCA dimensionality reduction (C++)
pca = ml.PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)

# Linear Regression (C++)
y = Tensor.rand([1000])
lr = ml.LinearRegression()
lr.fit(X, y)
predictions = lr.predict(X)
```

### Neural Network Inference

```python
from neurova.hybrid import *

# Create input
x = Tensor.randn([1, 3, 224, 224])  # Batch=1, Channels=3, 224x224

# Define simple conv net (all C++)
kernel = Tensor.randn([64, 3, 3, 3])  # 64 filters, 3x3
conv_out = nn.conv2d(x, kernel, stride=1, padding=1)
activated = nn.relu(conv_out)
pooled = nn.max_pool2d(activated, kernel_size=2, stride=2)

# Flatten and classify
flat = pooled.reshape([1, -1])
logits = Tensor.randn([1, 1000])
probs = nn.softmax(logits)
```

## Building from Source

```bash
cd Neurova
./build_minimal.sh
```

**Requirements:**

- Python 3.12+
- pybind11: `pip install pybind11`
- clang++ (Xcode Command Line Tools on macOS)
- Apple Silicon (ARM NEON) or Intel/AMD (AVX2/SSE)

**Output:**

- `neurova/neurova_core.cpython-312-darwin.so` (692KB)
- Includes: Tensor, Image, imgproc, filters, morphology, nn, ml

## Testing

```bash
# Quick test
python neurova/hybrid.py

# Comprehensive test
python test_cpp_core.py

# Check backend
python -c "from neurova.hybrid import print_backend_info; print_backend_info()"
```

## File Organization

```
Neurova/
 src/
    neurova_minimal.cpp          # C++ source (2000+ lines)
 neurova/
    neurova_core.so              # Compiled C++ module (692KB)
    hybrid.py                    # Hybrid routing system
    cpp_core.py                  # Direct C++ wrapper
    [all Python modules]         # Fallback implementations
 build_minimal.sh                 # Build script
 test_cpp_core.py                 # C++ tests
 README_HYBRID.md                 # This file
```

## Design Principles

1. **C++ First** - Use high-performance C++ when available
2. **Python Fallback** - Seamlessly fall back to Python if needed
3. **Zero Config** - Works out of the box, no setup required
4. **Transparent** - User doesn't need to know which backend is used
5. **Professional** - Production-ready, well-tested code

## Performance Tips

###  DO:

- Use `hybrid` import for automatic optimization
- Work with batched operations
- Keep data in C++ as long as possible
- Use built-in functions (avoid Python loops)

###  DON'T:

- Convert between NumPy and Tensor repeatedly
- Use small operations in loops (batch them)
- Mix C++ and Python unnecessarily

## Platform Support

| Platform              | SIMD     | Status           |
| --------------------- | -------- | ---------------- |
| macOS (Apple Silicon) | ARM NEON |  Full Support  |
| macOS (Intel)         | AVX2/SSE |  Full Support  |
| Linux (x86_64)        | AVX2/SSE |  Full Support  |
| Linux (ARM64)         | ARM NEON |  Full Support  |
| Windows               | AVX2/SSE |  Needs testing |

## Benchmarks

Run on Apple M2 (ARM NEON enabled):

```
Tensor Operations:
  randn [1024x1024]: 13.2ms
  matmul [512x512]: 25.3ms (10.6 GFLOPS)
  ReLU [1M elements]: 0.13ms

Image Processing:
  rgb_to_gray (1280×720): 0.44ms (2275 FPS)
  resize (1280×720->640×360): 1.77ms (565 FPS)
  gaussian_blur (k=5): 8.01ms (125 FPS)
  sobel: 1.2ms

ML Algorithms:
  KMeans (1000 samples, 5 clusters): 15ms
  PCA (1000 samples -> 2D): 8ms
  LinearRegression fit: 12ms
```

## Roadmap

###  Completed (v0.2.0)

- Core tensor operations with SIMD
- Image processing basics
- Filters (Sobel, Canny, median, bilateral)
- Morphology (erode, dilate, opening, closing)
- Neural network layers (conv, pool, batch norm)
- ML algorithms (KMeans, PCA, LinearRegression)
- Hybrid routing system

###  In Progress

- More complete convolution implementation
- GPU support (CUDA/Metal)
- Additional ML algorithms (SVM, RandomForest)
- Video processing acceleration

###  Planned

- Full architecture implementations in C++
- TensorFlow/PyTorch interop
- Distributed computing support
- WebAssembly build

## License

Apache License 2.0 - Copyright (c) 2026 Squid Consultancy Group (SCG)

## Contributing

Contributions welcome! The C++ core is self-contained in `src/neurova_minimal.cpp` for easy extension.

---

**Made with  by the Neurova Team**
