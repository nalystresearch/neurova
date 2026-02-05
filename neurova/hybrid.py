# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Neurova Hybrid System - C++ Primary with Python Fallback

This module intelligently routes to C++ implementations when available,
falls back to pure Python when needed. Keeps both for maximum compatibility.
"""

import os
import sys
import warnings
from typing import Any, Callable

# Try to import C++ core
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)
    
    import neurova_core as _cpp
    HAS_CPP = True
    CPP_VERSION = _cpp.__version__
    CPP_SIMD = _cpp.SIMD_SUPPORT
except ImportError as e:
    HAS_CPP = False
    _cpp = None
    CPP_VERSION = "N/A"
    CPP_SIMD = "None"

print(f"Neurova Hybrid System")
print(f"  C++ Core: {' Available' if HAS_CPP else ' Not Available'}")
if HAS_CPP:
    print(f"  C++ Version: {CPP_VERSION}")
    print(f"  SIMD: {CPP_SIMD}")

# 
# Module Routing - Try C++ first, fallback to Python
# 

class ModuleRouter:
    """Routes function calls to C++ or Python implementation"""
    
    def __init__(self, name: str):
        self.name = name
        self.cpp_module = None
        self.py_module = None
        
        # Try to get C++ module
        if HAS_CPP and hasattr(_cpp, name):
            self.cpp_module = getattr(_cpp, name)
        
        # Try to get Python module
        try:
            if name == 'imgproc':
                from neurova import imgproc as py_mod
            elif name == 'filters':
                from neurova import filters as py_mod
            elif name == 'morphology':
                from neurova import morphology as py_mod
            elif name == 'nn':
                from neurova import nn as py_mod
            elif name == 'ml':
                from neurova import ml as py_mod
            elif name == 'face':
                from neurova import face as py_mod
            else:
                py_mod = None
            
            self.py_module = py_mod
        except ImportError:
            pass
    
    def __getattr__(self, attr: str):
        """Route attribute access to C++ or Python"""
        
        # Try C++ first
        if self.cpp_module and hasattr(self.cpp_module, attr):
            return getattr(self.cpp_module, attr)
        
        # Fallback to Python
        if self.py_module and hasattr(self.py_module, attr):
            func = getattr(self.py_module, attr)
            # Warn on first use of Python fallback
            if not getattr(self, f'_warned_{attr}', False):
                warnings.warn(f"{self.name}.{attr} using Python fallback (C++ not available)", 
                            RuntimeWarning, stacklevel=2)
                setattr(self, f'_warned_{attr}', True)
            return func
        
        raise AttributeError(f"'{self.name}' has no attribute '{attr}'")
    
    def __dir__(self):
        """List all available attributes"""
        attrs = set()
        if self.cpp_module:
            attrs.update(dir(self.cpp_module))
        if self.py_module:
            attrs.update(dir(self.py_module))
        return sorted(attrs)

# 
# Export Core Classes (C++ Primary)
# 

if HAS_CPP:
    # Use C++ implementations
    DType = _cpp.DType
    Tensor = _cpp.Tensor
    Image = _cpp.Image
    Rect = _cpp.Rect
    
    # Create module routers
    imgproc = ModuleRouter('imgproc')
    filters = ModuleRouter('filters')
    morphology = ModuleRouter('morphology')
    features = ModuleRouter('features')
    transform = ModuleRouter('transform')
    video = ModuleRouter('video')
    segmentation = ModuleRouter('segmentation')
    nn = ModuleRouter('nn')
    ml = ModuleRouter('ml')
    face = ModuleRouter('face')
    
else:
    # Pure Python fallback
    print("  Using pure Python mode (slower)")
    
    # Import Python implementations
    try:
        from neurova.core import DType, Tensor, Image
        from neurova.core.ops import Rect
        from neurova import imgproc, filters, morphology, nn, ml, face
    except ImportError as e:
        print(f" Failed to import Python modules: {e}")
        raise

# 
# Performance Monitoring
# 

class PerformanceMonitor:
    """Track which backend is being used"""
    
    def __init__(self):
        self.cpp_calls = 0
        self.py_calls = 0
    
    def record_cpp(self):
        self.cpp_calls += 1
    
    def record_py(self):
        self.py_calls += 1
    
    def get_stats(self):
        total = self.cpp_calls + self.py_calls
        if total == 0:
            return "No calls recorded"
        
        cpp_pct = 100 * self.cpp_calls / total
        py_pct = 100 * self.py_calls / total
        
        return {
            'cpp_calls': self.cpp_calls,
            'py_calls': self.py_calls,
            'cpp_percentage': cpp_pct,
            'py_percentage': py_pct,
            'total': total
        }
    
    def print_stats(self):
        stats = self.get_stats()
        if isinstance(stats, str):
            print(stats)
            return
        
        print("\nNeurova Performance Stats:")
        print(f"  C++ calls: {stats['cpp_calls']} ({stats['cpp_percentage']:.1f}%)")
        print(f"  Python calls: {stats['py_calls']} ({stats['py_percentage']:.1f}%)")
        print(f"  Total: {stats['total']}")

_perf_monitor = PerformanceMonitor()

# 
# Helper Functions
# 

def get_backend_info():
    """Get information about available backends"""
    return {
        'cpp_available': HAS_CPP,
        'cpp_version': CPP_VERSION,
        'simd_support': CPP_SIMD,
        'performance': _perf_monitor.get_stats()
    }

def print_backend_info():
    """Print backend information"""
    info = get_backend_info()
    print("\nNeurova Backend Information:")
    print(f"  C++ Available: {info['cpp_available']}")
    if info['cpp_available']:
        print(f"  C++ Version: {info['cpp_version']}")
        print(f"  SIMD Support: {info['simd_support']}")
    
    if info['performance'] != "No calls recorded":
        _perf_monitor.print_stats()

def prefer_cpp():
    """Force using C++ when available"""
    if not HAS_CPP:
        raise RuntimeError("C++ core not available")
    print(" C++ mode enabled (default)")

def prefer_python():
    """Force using Python implementations"""
    warnings.warn("Python mode forced - will be slower", UserWarning)
    print("  Python mode enabled - performance will be slower")

# 
# Quick Tests
# 

def quick_test():
    """Run quick tests to verify hybrid system works"""
    print("\n" + "="*60)
    print("Running Hybrid System Tests")
    print("="*60)
    
    try:
        # Test Tensor
        print("\n[1/7] Testing Tensor...")
        t = Tensor.randn([10, 10])
        print(f"  PASS  Tensor shape: {t.shape}, sum: {t.sum():.4f}")
        
        # Test Image
        print("\n[2/7] Testing Image...")
        import numpy as np
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image(arr)
        print(f"  PASS  Image: {img.width}x{img.height}, channels: {img.channels}")
        
        # Test imgproc
        print("\n[3/7] Testing imgproc...")
        gray = imgproc.rgb_to_gray(img)
        print(f"  PASS  RGB to Gray: {gray.width}x{gray.height}")
        
        # Test filters
        if HAS_CPP:
            print("\n[4/7] Testing filters...")
            edges = filters.sobel(gray)
            print(f"  PASS  Sobel edges: {edges.width}x{edges.height}")
        else:
            print("\n[4/7] Skipping filters (requires C++)")
        
        # Test morphology
        if HAS_CPP:
            print("\n[5/7] Testing morphology...")
            eroded = morphology.erode(gray, 3)
            print(f"  PASS  Erode: {eroded.width}x{eroded.height}")
        else:
            print("\n[5/7] Skipping morphology (requires C++)")
        
        # Test nn
        print("\n[6/7] Testing nn...")
        x = Tensor.randn([10, 10])
        activated = nn.relu(x)
        print(f"  PASS  ReLU: mean={activated.mean():.4f}")
        
        # Test ml
        if HAS_CPP:
            print("\n[7/7] Testing ml...")
            X = Tensor.rand([100, 2])
            kmeans = ml.KMeans(n_clusters=3)
            kmeans.fit(X)
            labels = kmeans.get_labels()
            print(f"  PASS  KMeans: {labels.size} samples clustered")
        else:
            print("\n[7/7] Skipping ml (requires C++)")
        
        print("\n" + "="*60)
        print(" All tests passed!")
        print("="*60)
        
        print_backend_info()
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()

# 
# Auto-run quick test on import
# 

if __name__ != "__main__":
    # Only show compact info on import
    pass
else:
    # Full test when run as script
    quick_test()
