# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Chapter 1: Getting Started with Neurova


This chapter covers:
- Installation and setup
- Basic imports
- Device configuration (CPU/GPU)
- Version information
- First image operations

Author: Neurova Team
"""

import numpy as np

# 1.1 basic imports
print("")
print("Chapter 1: Getting Started with Neurova")
print("")

import neurova as nv

# check version
print(f"\n1.1 Version Information")
print(f"    Neurova version: {nv.__version__}")
print(f"    Build info: {nv.get_build_info()}")

# 1.2 device configuration
print(f"\n1.2 Device Configuration")

# check available devices
print(f"    CUDA available: {nv.cuda_is_available()}")
print(f"    Current device: {nv.get_device()}")
print(f"    Backend: {nv.get_backend()}")

# Set device (CPU by default)
nv.set_device('cpu')
print(f"    Device set to: {nv.get_device()}")

# get device info
info = nv.get_device_info()
print(f"    Device info: {info}")

# 1.3 creating arrays
print(f"\n1.3 Creating Arrays")

# Create arrays (similar to numpy but optimized for GPU)
zeros = nv.zeros((3, 3))
ones = nv.ones((3, 3))
empty = nv.empty((3, 3))
arr = nv.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"    Zeros shape: {zeros.shape}")
print(f"    Ones shape: {ones.shape}")
print(f"    Custom array:\n{arr}")

# 1.4 loading your first image
print(f"\n1.4 Loading Images")

from neurova import io
from pathlib import Path

# get sample image path
data_dir = Path(__file__).parent.parent / "neurova" / "data" / "sample-images"

# create a sample image if none exists
sample_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
print(f"    Created sample image: shape={sample_img.shape}, dtype={sample_img.dtype}")

# Convert to grayscale using nvc (standard API)
gray = nv.nvc.cvtColor(sample_img, nv.nvc.COLOR_RGB2GRAY)
print(f"    Grayscale image: shape={gray.shape}")

# 1.5 basic image operations
print(f"\n1.5 Basic Image Operations")

# resize
from neurova import transform
resized = transform.resize(sample_img, (50, 50))
print(f"    Resized: {sample_img.shape} -> {resized.shape}")

# rotate
rotated = transform.rotate(sample_img, 45)
print(f"    Rotated 45 degrees: shape={rotated.shape}")

# Flip using core.ops
flipped = nv.core.ops.flip(sample_img, 1)  # 1 = horizontal flip
print(f"    Flipped horizontally: shape={flipped.shape}")

# 1.6 using context manager for device
print(f"\n1.6 Device Context Manager")

with nv.device_context('cpu'):
    result = nv.ones((100, 100)) * 2
    print(f"    Computed on: {nv.get_device()}")

# summary
print("\n" + "=" * 60)
print("Chapter 1 Summary:")
print("   Imported Neurova and checked version")
print("   Configured device (CPU/GPU)")
print("   Created arrays")
print("   Loaded and processed images")
print("   Used device context manager")
print("")
