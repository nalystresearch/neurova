# Neurova Installation Guide

**Complete installation instructions for all platforms and configurations**

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Basic Installation (CPU Only)](#basic-installation-cpu-only)
3. [GPU Installation (NVIDIA)](#gpu-installation-nvidia)
4. [Optional Dependencies](#optional-dependencies)
5. [Platform-Specific Instructions](#platform-specific-instructions)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)
8. [Uninstallation](#uninstallation)

---

## System Requirements

### Minimum Requirements

**Operating System:**

- Linux (Ubuntu 18.04+, CentOS 7+, or equivalent)
- Windows 10/11 (64-bit)
- macOS 10.15+ (Catalina or later)

**Python:**

- Python 3.8 or higher
- pip 20.0 or higher

**Disk Space:**

- Minimal installation: 55 MB
- Full installation (with GPU): 665 MB

**RAM:**

- Minimum: 2 GB
- Recommended: 8 GB or more

### GPU Requirements (Optional)

**For GPU Acceleration:**

- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.x or 12.x
- NVIDIA Driver 470+ (for CUDA 11.x) or 520+ (for CUDA 12.x)
- 4 GB+ GPU memory recommended

**Note:** AMD and Intel GPUs are not supported. Apple Silicon (M1/M2/M3) does not support GPU acceleration.

---

## Basic Installation (CPU Only)

### Step 1: Install Python

**Check if Python is installed:**

```bash
python --version
# or
python3 --version
```

**If Python is not installed:**

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install python3 python3-pip
```

**macOS:**

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python3
```

**Windows:**

- Download Python from https://www.python.org/downloads/
- Run the installer and check "Add Python to PATH"

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv neurova-env

# Activate virtual environment
# On Linux/macOS:
source neurova-env/bin/activate

# On Windows:
neurova-env\Scripts\activate
```

### Step 3: Install Neurova

```bash
pip install neurova
```

This installs:

- Neurova library
- NumPy (required dependency)

**Installation size:** ~55 MB

---

## GPU Installation (NVIDIA)

### Prerequisites

Before installing GPU support, ensure you have:

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Driver** installed
3. **CUDA Toolkit** installed

### Step 1: Check NVIDIA Driver

```bash
nvidia-smi
```

You should see output showing your GPU and CUDA version. If not, install NVIDIA drivers first.

### Step 2: Install CUDA Toolkit

**Linux:**

```bash
# Download from NVIDIA website or use package manager
# For Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3
```

**Windows:**

- Download CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
- Run the installer

**macOS:**
Not supported (no NVIDIA GPU support on macOS)

### Step 3: Install Neurova

```bash
pip install neurova
```

### Step 4: Install CuPy (GPU Support)

**Determine your CUDA version:**

```bash
nvcc --version
# or
nvidia-smi  # Check "CUDA Version" in output
```

**Install CuPy matching your CUDA version:**

```bash
# For CUDA 11.x:
pip install cupy-cuda11x

# For CUDA 12.x:
pip install cupy-cuda12x

# If unsure, CuPy will try to detect:
pip install cupy
```

**Installation size:** ~500 MB additional

### Step 5: Verify GPU Installation

```python
import neurova as nv

# Check if GPU is available
if nv.cuda_is_available():
    print(f"GPU Available: {nv.get_device_name()}")
    nv.set_device('cuda')
    print("GPU acceleration enabled!")
else:
    print("GPU not available, using CPU")
```

---

## Optional Dependencies

### Image I/O Support (Pillow)

For extended image format support (JPEG, PNG, TIFF, WebP, etc.):

```bash
pip install pillow
```

**Or:**

```bash
pip install neurova[io]
```

### Scientific Computing (SciPy)

For advanced filters and operations:

```bash
pip install scipy
```

**Or:**

```bash
pip install neurova[scientific]
```

### All Optional Features

To install everything at once:

```bash
pip install neurova[all]
```

This includes:

- Pillow (image I/O)
- SciPy (scientific computing)

### Development Tools

For contributors and developers:

```bash
pip install neurova[dev]
```

This includes:

- pytest (testing)
- pytest-cov (coverage)
- black (code formatter)
- flake8 (linter)
- mypy (type checker)

### Documentation Tools

```bash
pip install neurova[docs]
```

This includes:

- Sphinx (documentation generator)
- sphinx-rtd-theme (theme)

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

**Complete Installation with GPU:**

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv neurova-env
source neurova-env/bin/activate

# Install Neurova
pip install neurova

# Install CuPy for GPU (if you have NVIDIA GPU)
pip install cupy-cuda12x

# Install optional dependencies
pip install pillow scipy
```

### Windows

**Complete Installation with GPU:**

```powershell
# Install Python from python.org (if not already installed)

# Create virtual environment
python -m venv neurova-env
neurova-env\Scripts\activate

# Install Neurova
pip install neurova

# Install CuPy for GPU (if you have NVIDIA GPU)
pip install cupy-cuda12x

# Install optional dependencies
pip install pillow scipy
```

### macOS

**Complete Installation (CPU Only):**

```bash
# Install Python via Homebrew
brew install python3

# Create virtual environment
python3 -m venv neurova-env
source neurova-env/bin/activate

# Install Neurova
pip install neurova

# Install optional dependencies
pip install pillow scipy
```

**Note:** GPU acceleration is not available on macOS (no NVIDIA GPUs).

---

## Verification

### Verify Installation

```python
import neurova as nv

# Check version
print(f"Neurova version: {nv.__version__}")

# Check device
print(f"Current device: {nv.get_device()}")

# Check GPU availability
if nv.cuda_is_available():
    print(f"GPU available: {nv.get_device_name()}")
    print(f"GPU count: {nv.get_device_count()}")
else:
    print("Using CPU mode")

# Get build info
print(nv.get_build_info())
```

### Run Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run tests
pytest tests/

# Run with coverage
pytest --cov=neurova tests/
```

### Quick Functionality Test

```python
import neurova as nv
import numpy as np

# Test basic array operations
data = np.random.rand(100, 10)
result = nv.array(data)
print(f"Array created successfully: {result.shape}")

# Test device switching (if GPU available)
if nv.cuda_is_available():
    nv.set_device('cuda')
    gpu_array = nv.array(data)
    print(f"GPU array created: {type(gpu_array)}")
    nv.set_device('cpu')

print("All tests passed!")
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "ModuleNotFoundError: No module named 'neurova'"

**Solution:**

```bash
# Make sure you're in the correct virtual environment
which python
pip list | grep neurova

# Reinstall if necessary
pip install --upgrade --force-reinstall neurova
```

#### Issue 2: "ImportError: DLL load failed" (Windows)

**Solution:**

- Install Visual C++ Redistributable from Microsoft
- Install/update NumPy: `pip install --upgrade numpy`

#### Issue 3: CuPy not found or GPU not detected

**Solution:**

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall CuPy with correct CUDA version
pip uninstall cupy cupy-cuda11x cupy-cuda12x
pip install cupy-cuda12x  # Replace with your CUDA version

# Test CuPy directly
python -c "import cupy as cp; print(cp.cuda.Device(0).compute_capability)"
```

#### Issue 4: NumPy version incompatibility

**Solution:**

```bash
# For Python < 3.9
pip install "numpy>=1.19.0,<2.0"

# For Python >= 3.9
pip install "numpy>=2.0"
```

#### Issue 5: Permission denied errors

**Solution:**

```bash
# Use --user flag
pip install --user neurova

# Or use virtual environment (recommended)
python -m venv neurova-env
source neurova-env/bin/activate  # Linux/macOS
neurova-env\Scripts\activate     # Windows
pip install neurova
```

### Getting Help

If you encounter issues not covered here:

1. **Check the documentation:** See README.md and other .md files
2. **Search existing issues:** Check GitHub issues
3. **Create a new issue:** Provide:
   - Neurova version
   - Python version
   - Operating system
   - Full error message
   - Steps to reproduce

---

## Uninstallation

### Remove Neurova Only

```bash
pip uninstall neurova
```

### Remove Everything (Including Dependencies)

```bash
# Uninstall Neurova and all related packages
pip uninstall neurova cupy-cuda12x scipy pillow pytest -y

# Remove virtual environment (if used)
deactivate  # Exit virtual environment first
rm -rf neurova-env  # Linux/macOS
rmdir /s neurova-env  # Windows
```

### Clean pip Cache

```bash
pip cache purge
```

---

## Installation Summary

### Quick Reference

**Minimal (CPU only):**

```bash
pip install neurova
```

**With GPU support:**

```bash
pip install neurova cupy-cuda12x
```

**Full installation:**

```bash
pip install neurova[all] cupy-cuda12x
```

**Development:**

```bash
git clone https://github.com/neurova/neurova.git
cd neurova
pip install -e ".[dev]"
```

### Size Requirements

| Configuration   | Installation Size | Download Size |
| --------------- | ----------------- | ------------- |
| Minimal (CPU)   | ~55 MB            | ~17 MB        |
| With Pillow     | ~65 MB            | ~20 MB        |
| With SciPy      | ~155 MB           | ~47 MB        |
| With CuPy (GPU) | ~555 MB           | ~217 MB       |
| Full (all)      | ~665 MB           | ~250 MB       |

### Supported Versions

| Component | Versions                         |
| --------- | -------------------------------- |
| Python    | 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 |
| NumPy     | >= 1.19.0                        |
| CuPy      | >= 11.0 (optional)               |
| SciPy     | >= 1.7.0 (optional)              |
| Pillow    | >= 9.0.0 (optional)              |

---

## Next Steps

After installation:

1. **Read the Quick Start:** See QUICKSTART.md
2. **Explore Examples:** Check examples/ directory
3. **Learn GPU Usage:** Read GPU_QUICKSTART.md
4. **Try Device Selection:** See DEVICE_SELECTION_GUIDE.md
5. **Review API:** See README.md for API documentation

---

**Installation Complete!** You're ready to use Neurova.

For questions or issues, please refer to the documentation or open an issue on GitHub.
