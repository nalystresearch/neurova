# Getting Started with Neurova Development

This guide helps you understand the Neurova codebase and contribute to development.

## Installation

```bash
# Clone repository
git clone https://github.com/nalystresearch/neurova.git
cd neurova

# Install in development mode
pip install -e ".[dev]"
```

## Project Status

Neurova is a complete computer vision and deep learning library with:

### Implemented Modules

- **Core** - Image class, color spaces, array operations
- **I/O** - Image reading/writing, video capture
- **Transform** - Resize, rotate, flip, affine transforms
- **Filters** - Blur, edge detection, morphology
- **Features** - Corner detection, HOG descriptors
- **Detection** - Object detection, template matching
- **Face** - Face detection and recognition (Haar, LBP, DNN)
- **Segmentation** - Thresholding, contours, watershed
- **Neural** - Layers, activations, losses, optimizers
- **ML** - Classification, regression, clustering, PCA
- **Datasets** - Built-in tabular, time series, image datasets
- **Augmentation** - Data augmentation pipelines
- **Video** - Video processing utilities

## Understanding the Codebase

### Core Components

#### 1. Image Class (`neurova/core/image.py`)

The fundamental data structure for images:

```python
from neurova.core.image import Image, create_blank_image

# Images are immutable
img = Image(data, color_space=ColorSpace.RGB)

# Access properties
width = img.width
height = img.height
channels = img.channels

# Get numpy array
arr = img.data  # Read-only view
arr = img.as_array()  # Copy
```

#### 2. Color Conversions (`neurova/core/color.py`)

Comprehensive color space support:

```python
from neurova.core.color import convert_color_space, to_grayscale
from neurova.core.constants import ColorSpace

# Convert between color spaces
hsv = convert_color_space(rgb_data, ColorSpace.RGB, ColorSpace.HSV)
lab = convert_color_space(rgb_data, ColorSpace.RGB, ColorSpace.LAB)

# To grayscale
gray = to_grayscale(rgb_data, from_space=ColorSpace.RGB)
```

#### 3. Array Operations (`neurova/core/array_ops.py`)

Utility functions for array manipulation:

```python
from neurova.core.array_ops import normalize, pad_array, ensure_3d

# Normalize to [0, 1]
normalized = normalize(arr, target_min=0.0, target_max=1.0)

# Pad array
padded = pad_array(arr, pad_width=10, mode='reflect')

# Ensure 3D shape (H, W, C)
arr_3d = ensure_3d(arr_2d)
```

## How to Add New Features

### Example: Adding a New Transformation

Let's say you want to implement image resizing. Here's the step-by-step process:

#### Step 1: Create the Module File

Create `neurova/transform/resize.py`:

```python
"""Image resizing operations"""

import numpy as np
from typing import Tuple, Optional
from neurova.core.constants import InterpolationMode
from neurova.core.errors import ValidationError
from neurova.core.array_ops import ensure_array


def resize(image: np.ndarray,
          size: Tuple[int, int],
          interpolation: InterpolationMode = InterpolationMode.LINEAR) -> np.ndarray:
    """
    Resize an image to specified size

    Args:
        image: Input image array (H, W) or (H, W, C)
        size: Target size as (width, height)
        interpolation: Interpolation method

    Returns:
        Resized image array
    """
    image = ensure_array(image)
    width, height = size

    if interpolation == InterpolationMode.NEAREST:
        return _resize_nearest(image, width, height)
    elif interpolation == InterpolationMode.LINEAR:
        return _resize_bilinear(image, width, height)
    else:
        raise ValidationError('interpolation', interpolation,
                            'NEAREST or LINEAR')


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest neighbor resize"""
    h, w = image.shape[:2]

    # Calculate indices
    y_ratio = h / height
    x_ratio = w / width

    # Create output array
    if image.ndim == 3:
        output = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    else:
        output = np.zeros((height, width), dtype=image.dtype)

    # Nearest neighbor sampling
    for i in range(height):
        for j in range(width):
            src_y = int(i * y_ratio)
            src_x = int(j * x_ratio)
            output[i, j] = image[src_y, src_x]

    return output


def _resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear interpolation resize"""
    h, w = image.shape[:2]

    # This is a simplified version - you would implement full bilinear interpolation
    # For now, use nearest neighbor as placeholder
    return _resize_nearest(image, width, height)
```

#### Step 2: Update Module **init**.py

Edit `neurova/transform/__init__.py`:

```python
"""Geometric transformations for Neurova"""

from neurova.transform.resize import resize

__all__ = ['resize']
```

#### Step 3: Add Tests

Create `tests/test_transform/test_resize.py`:

```python
import numpy as np
import pytest
from neurova.transform import resize
from neurova.core.constants import InterpolationMode


def test_resize_nearest():
    # Create test image
    img = np.random.rand(100, 100, 3).astype(np.float32)

    # Resize
    resized = resize(img, (50, 50), InterpolationMode.NEAREST)

    # Check shape
    assert resized.shape == (50, 50, 3)


def test_resize_upscale():
    img = np.random.rand(50, 50, 3).astype(np.float32)
    resized = resize(img, (100, 100), InterpolationMode.NEAREST)
    assert resized.shape == (100, 100, 3)
```

#### Step 4: Document the Feature

Add to documentation and update PROJECT_STATUS.md.

## Implementation Guidelines

### 1. Follow the Existing Pattern

- Look at existing modules (e.g., `core/color.py`) as examples
- Use type hints for all functions
- Write comprehensive docstrings
- Handle errors gracefully

### 2. Use NumPy Efficiently

```python
# Good: Vectorized
result = image * 1.5

# Bad: Python loops
for i in range(height):
    for j in range(width):
        result[i, j] = image[i, j] * 1.5
```

### 3. Validate Inputs

```python
from neurova.core.errors import ValidationError
from neurova.core.array_ops import validate_image_shape

def my_function(image: np.ndarray):
    """My function with validation"""
    validate_image_shape(image, "image")

    if image.dtype != np.uint8:
        raise ValidationError('dtype', image.dtype, 'uint8')
```

### 4. Return New Objects (Immutability)

```python
def process(image: np.ndarray) -> np.ndarray:
    """Always return a new array"""
    result = image.copy()  # Don't modify input
    # ... process result ...
    return result
```

## Priority Implementation Order

### Phase 1: I/O (High Priority)

```
neurova/io/
 writers.py       # Image writing (PNG, JPEG, BMP)
 video.py         # Basic video I/O
 formats.py       # Format detection
```

Start with `writers.py`:

- Implement PNG writing (reverse of reading)
- Add JPEG writing (via Pillow)
- Add BMP writing

### Phase 2: Transformations

```
neurova/transform/
 resize.py        # Start here - fundamental operation
 rotate.py        # Image rotation
 affine.py        # Affine transformations
 perspective.py   # Perspective transforms
```

### Phase 3: Filters

```
neurova/filters/
 convolution.py   # Core convolution (most important)
 blur.py          # Gaussian, median, bilateral
 edge.py          # Sobel, Canny edge detection
 morphology.py    # Erosion, dilation
```

**Key: Implement convolution first** - many filters use it.

### Phase 4: Features

```
neurova/features/
 corners.py       # Harris corner detector
 keypoints.py     # Keypoint detection
 descriptors.py   # ORB descriptor
 matching.py      # Feature matching
```

## � Testing Strategy

### Unit Tests

Each module should have corresponding tests:

```python
# tests/test_transform/test_resize.py
import numpy as np
from neurova.transform import resize

def test_resize_shape():
    """Test that resize produces correct shape"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = resize(img, (50, 50))
    assert result.shape == (50, 50, 3)

def test_resize_preserves_dtype():
    """Test that resize preserves data type"""
    img = np.zeros((100, 100), dtype=np.float32)
    result = resize(img, (50, 50))
    assert result.dtype == np.float32
```

### Run Tests

```bash
cd /Users/harrythapa/Desktop/Neurova

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# With coverage
pytest --cov=neurova --cov-report=html tests/
```

## Documentation Guidelines

### Docstring Format

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief description of function

    More detailed description if needed. Explain what the function does,
    any important algorithms used, and edge cases.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs

    Examples:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected output
    """
```

## Debugging Tips

### 1. Use Python Debugger

```python
import pdb; pdb.set_trace()  # Add breakpoint
```

### 2. Print Array Info

```python
def debug_array(arr: np.ndarray, name: str = "array"):
    """Print useful array information"""
    print(f"{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min: {arr.min()}, Max: {arr.max()}")
    print(f"  Mean: {arr.mean()}")
```

### 3. Visual Debugging

```python
# For image operations, save intermediate results
from neurova.io import write_image  # Once implemented
write_image("debug_step1.png", intermediate_result)
```

## Key Algorithms to Implement

### Priority Algorithms (with references):

1. **Convolution** (filters/convolution.py)
   - 2D convolution with kernels
   - Separable convolution for efficiency

2. **Gaussian Blur** (filters/blur.py)
   - Create Gaussian kernel
   - Apply via convolution

3. **Canny Edge Detection** (filters/edge.py)
   - Gaussian smoothing
   - Sobel gradients
   - Non-maximum suppression
   - Hysteresis thresholding

4. **Harris Corner Detection** (features/corners.py)
   - Compute gradients
   - Harris response function
   - Non-maximum suppression

5. **ORB Descriptor** (features/descriptors.py)
   - FAST keypoint detection
   - Orientation computation
   - rBRIEF descriptor

## Building and Distribution

### Build Package

```bash
cd /Users/harrythapa/Desktop/Neurova

# Build distribution
python setup.py sdist bdist_wheel

# Install locally
pip install -e .
```

### Test Installation

```bash
# Create virtual environment
python -m venv test_env
source test_env/bin/activate  # On macOS/Linux

# Install package
pip install -e .

# Test import
python -c "import neurova; print(neurova.__version__)"
```

## Contributing Workflow

1. **Pick a module** from PROJECT_STATUS.md
2. **Create the file** following existing patterns
3. **Implement the functionality** with proper typing and docstrings
4. **Write tests** for your implementation
5. **Update documentation** (README, PROJECT_STATUS)
6. **Test thoroughly** before committing

## Performance Optimization

### When to Optimize

1. **First, make it work** - Correctness first
2. **Then, make it right** - Clean code
3. **Finally, make it fast** - Optimize bottlenecks

### Common Optimizations

```python
# 1. Use NumPy vectorization
result = np.exp(-0.5 * (x ** 2))  # Fast

# Instead of:
result = np.array([math.exp(-0.5 * val**2) for val in x])  # Slow

# 2. Pre-allocate arrays
result = np.zeros((height, width, 3), dtype=np.float32)

# 3. Use in-place operations when safe
arr *= 2  # In-place
# vs
arr = arr * 2  # Creates new array

# 4. Use numba for hot loops (optional dependency)
from numba import jit

@jit(nopython=True)
def fast_loop(arr):
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i, j] = arr[i, j] * 2
    return result
```

## Learning Resources

### Computer Vision Algorithms

- Digital Image Processing by Gonzalez & Woods
- Computer Vision: Algorithms and Applications by Szeliski

### NumPy Optimization

- NumPy User Guide: https://numpy.org/doc/stable/user/
- From Python to Numpy: https://www.labri.fr/perso/nrougier/from-python-to-numpy/

### Python Best Practices

- PEP 8: https://pep8.org/
- PEP 484 (Type Hints): https://www.python.org/dev/peps/pep-0484/

## Getting Help

If you encounter issues:

1. Check existing code in `neurova/core/` for examples
2. Review ARCHITECTURE.md for design decisions
3. Check the Neurova documentation

## Conclusion

You now have a solid foundation to continue building with Neurova!

**Next immediate steps:**

1. Complete `neurova/io/writers.py` (image writing)
2. Implement `neurova/transform/resize.py` (image resizing)
3. Create `neurova/filters/convolution.py` (core convolution)

**Remember:**

- Follow existing patterns
- Write tests
- Document everything
- Use NumPy efficiently
- Keep it simple and Pythonic

Good luck building the most powerful image processing library!

---

_For questions or issues, refer to the documentation files in this directory._
