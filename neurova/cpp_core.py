# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Neurova C++ Core Python Integration

This module provides high-level Python wrappers around the C++ core,
making it easy to use from existing Neurova code.
"""

import os
import sys

# Try to import the C++ core
try:
    # Add neurova directory to path
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)
    
    import neurova_core as _nvc
    
    # Version info
    __version__ = _nvc.__version__
    SIMD_SUPPORT = _nvc.SIMD_SUPPORT
    HAS_CPP_CORE = True
    
except ImportError as e:
    HAS_CPP_CORE = False
    __version__ = "0.2.0"
    SIMD_SUPPORT = "None (C++ core not available)"
    _nvc = None

# Re-export everything from the C++ core
if HAS_CPP_CORE:
    # Types
    DType = _nvc.DType
    Tensor = _nvc.Tensor
    Image = _nvc.Image
    Rect = _nvc.Rect
    
    # Submodules
    imgproc = _nvc.imgproc
    nn = _nvc.nn
    ml = _nvc.ml
    face = _nvc.face

# Helper functions
def get_info():
    """Get information about the C++ core."""
    return {
        'version': __version__,
        'cpp_core_available': HAS_CPP_CORE,
        'simd_support': SIMD_SUPPORT,
    }

def check_cpp_core():
    """Check if the C++ core is available and working."""
    if not HAS_CPP_CORE:
        print(" C++ core not available")
        return False
    
    try:
        t = Tensor.randn([10, 10])
        _ = t.sum()
        print(f" C++ core working (v{__version__}, SIMD: {SIMD_SUPPORT})")
        return True
    except Exception as e:
        print(f" C++ core error: {e}")
        return False

if __name__ == "__main__":
    print("Neurova C++ Core")
    print("=" * 40)
    info = get_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()
    check_cpp_core()
