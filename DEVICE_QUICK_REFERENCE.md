# NEUROVA DEVICE SELECTION - QUICK REFERENCE CARD

## One-Line Setup

```python
import neurova as nv
nv.set_device('cuda') # Use GPU for EVERYTHING! 
```

---

## Device Selection Commands

| Command | Description | Example |
| ------------------------ | -------------------------- | ---------------------------- |
| `nv.set_device('cuda')` | Use GPU for all operations | `nv.set_device('cuda')` |
| `nv.set_device('cpu')` | Use CPU for all operations | `nv.set_device('cpu')` |
| `nv.set_device('gpu')` | Alias for 'cuda' | `nv.set_device('gpu')` |
| `nv.get_device()` | Get current device | `device = nv.get_device()` |
| `nv.cuda_is_available()` | Check if GPU available | `if nv.cuda_is_available():` |

---

## What Works on GPU

| Task | Module | Example | Speedup |
| --------------------- | -------------- | ------------------------------------ | ------- |
| **Image Filtering** | `nv.filters` | `nv.filters.gaussian_blur(img, 2.0)` | 20x |
| **Edge Detection** | `nv.filters` | `nv.filters.sobel(img)` | 20x |
| **Image Transforms** | `nv.transform` | `nv.transform.rotate(img, 45)` | 15x |
| **Video Processing** | `nv.video` | Process each frame | 50x |
| **Motion Tracking** | `nv.gpu_utils` | `optical_flow_gpu(f1, f2)` | 100x |
| **ML Training** | `nv.ml` | `RandomForest().fit(X, y)` | 6x |
| **DL Training** | `nv.nn` | `model.fit(X, y)` | 20x |
| **Feature Detection** | `nv.features` | `harris_corners(img)` | 15x |

---

## Common Patterns

### Auto-Select Best Device

```python
import neurova as nv

if nv.cuda_is_available():
 nv.set_device('cuda')
 print(f"Using GPU: {nv.get_device_name()}")
else:
 nv.set_device('cpu')
 print("Using CPU")
```

### Temporary GPU Usage

```python
import neurova as nv

nv.set_device('cpu') # Default to CPU

# Use GPU only for expensive operation
with nv.device_context('cuda'):
 result = nv.filters.gaussian_blur(huge_image, sigma=5.0)

# Back to CPU automatically
```

### Batch Processing on GPU

```python
import neurova as nv

nv.set_device('cuda')

# Process 100 images in parallel on GPU!
images = [nv.io.imread(f'img_{i}.jpg') for i in range(100)]
results = [nv.filters.gaussian_blur(img, 2.0) for img in images]
```

---

## Example: Video Processing

```python
import neurova as nv

# Enable GPU
nv.set_device('cuda')

# Open video
cap = nv.video.VideoCapture('video.mp4')

while cap.isOpened():
 ret, frame = cap.read()
 if not ret:
 break

 # All GPU-accelerated!
 gray = nv.core.rgb2gray(frame)
 blurred = nv.filters.gaussian_blur(gray, 2.0)
 edges = nv.filters.canny(blurred)

cap.release()
```

---

##  Example: Machine Learning

```python
import neurova as nv
import numpy as np

# Enable GPU
nv.set_device('cuda')

# Dataset
X = np.random.randn(10000, 100)
y = np.random.randint(0, 10, 10000)

# Train on GPU (6x faster!)
from neurova.ml import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
```

---

##   Example: Deep Learning

```python
import neurova as nv
from neurova.nn import Sequential, Linear, ReLU
from neurova.nn.autograd_gpu import Tensor

# Enable GPU
nv.set_device('cuda')

# Model
model = Sequential(
 Linear(784, 256),
 ReLU(),
 Linear(256, 10)
)

# GPU tensors
X = Tensor(train_data, device='cuda')
y = Tensor(train_labels, device='cuda')

# Training on GPU (20x faster!)
for epoch in range(100):
 pred = model(X)
 loss = criterion(pred, y)
 loss.backward() # GPU backprop!
 optimizer.step()
```

---

## Memory Management

```python
import neurova as nv

nv.set_device('cuda')

# Check memory usage
memory = nv.get_memory_usage()
print(f"GPU memory: {memory['used_gb']:.2f} GB")

# Clear GPU cache
nv.empty_cache()

# Wait for GPU operations
nv.synchronize()
```

---

## Device Info

```python
import neurova as nv

# Check availability
print(f"CUDA available: {nv.cuda_is_available()}")

# Get GPU count
print(f"GPU count: {nv.get_device_count()}")

# Get GPU name
print(f"GPU name: {nv.get_device_name()}")

# Get detailed info
info = nv.get_device_info()
print(info)
# {
# 'current_device': 'cuda',
# 'gpu_name': 'NVIDIA RTX 3080',
# 'compute_capability': '8.6',
# 'memory_total': '10.00 GB',
#...
# }
```

---

## Performance Tips

### DO

```python
# DO: Batch process multiple items
images = [img1, img2, img3,...]
results = [nv.filters.blur(img) for img in images] # GPU processes in parallel!

# DO: Keep data on GPU
nv.set_device('cuda')
img_gpu = nv.to_device(img)
filtered = nv.filters.blur(img_gpu)
edges = nv.filters.sobel(filtered) # Still on GPU!

# DO: Use GPU for large data
if image.size > 512*512:
 nv.set_device('cuda')
```

### DON'T

```python
# DON'T: Process one at a time
for img in images:
 result = nv.filters.blur(img) # Inefficient!

# DON'T: Transfer unnecessarily
img_gpu = nv.to_device(img, 'cuda')
img_cpu = nv.to_device(img_gpu, 'cpu') # Why?
img_gpu2 = nv.to_device(img_cpu, 'cuda') # Bad!

# DON'T: Use GPU for tiny data
tiny_img = np.zeros((32, 32)) # Too small, use CPU
```

---

## › Troubleshooting

| Problem | Solution |
| -------------------- | --------------------------------------- |
| "CuPy not installed" | `pip install cupy-cuda12x` |
| "No GPU detected" | Check NVIDIA GPU + CUDA installed |
| "Out of memory" | `nv.empty_cache()` or reduce batch size |
| "Slower on GPU" | Data too small, use CPU for small tasks |

---

## Installation

```bash
# Install Neurova
pip install neurova

# Install CuPy (for NVIDIA GPUs)
pip install cupy-cuda12x # CUDA 12.x
pip install cupy-cuda11x # CUDA 11.x

# For AMD GPUs
pip install cupy-rocm-5-0
```

---

## Decision Tree: GPU or CPU?

```
Is GPU available?
 No † Use CPU
‚ nv.set_device('cpu')
‚
 Yes † Is data large (>512Ã—512 or >10k samples)?
  No † Use CPU (overhead not worth it)
 ‚ nv.set_device('cpu')
 ‚
  Yes † Use GPU (10-100x faster!)
 nv.set_device('cuda')
```

---

## Summary

**3 Steps to GPU Acceleration:**

1. **Install CuPy**

 ```bash
 pip install cupy-cuda12x
 ```

2. **Set Device**

 ```python
 import neurova as nv
 nv.set_device('cuda')
 ```

3. **Done!** All operations now use GPU automatically! 

**Expected speedup: 10-100x faster!**

---

## Documentation

- Full Guide: [DEVICE_SELECTION_GUIDE.md](DEVICE_SELECTION_GUIDE.md)
- Quick Start: [GPU_QUICKSTART.md](GPU_QUICKSTART.md)
- Examples: [examples/choose_device_simple.py](examples/choose_device_simple.py)
- Tests: [tests/test_device_management.py](tests/test_device_management.py)

---

**Questions? Check the guides or run:**

```bash
python examples/choose_device_simple.py
python examples/gpu_acceleration_demo.py
```

 **Happy GPU-accelerated computing!** 
