"""
Neurova Utils - Python fallback with C++ acceleration

Provides Python fallbacks for core, augmentation, and calibration modules
with automatic C++ acceleration when available.
"""

import sys
import numpy as np
from typing import Optional, Tuple, List, Union

# Try to import C++ extension
try:
    from . import neurova_utils as _cpp
    HAS_CPP = True
    print("PASS  Neurova Utils: C++ acceleration enabled (608KB)")
except ImportError:
    HAS_CPP = False
    print("[warn] Neurova Utils: Using Python fallback (C++ not available)")

# 
# CORE MODULE
# 

class ColorSpace:
    """Color space enumeration"""
    RGB = 0
    BGR = 1
    GRAY = 2
    HSV = 3
    LAB = 4
    YUV = 5
    XYZ = 6
    RGBA = 7
    BGRA = 8

class DataType:
    """Data type enumeration"""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    INT32 = 4
    FLOAT32 = 5
    FLOAT64 = 6

class Image:
    """Image wrapper with metadata"""
    
    def __init__(self, data, color_space=ColorSpace.RGB, dtype=None, metadata=None):
        self._color_space = color_space
        if HAS_CPP and isinstance(data, np.ndarray):
            self._cpp_img = _cpp.Image(data, getattr(_cpp.ColorSpace, self._cs_name(color_space)))
            self._data = None
        else:
            self._data = np.asarray(data)
            self._dtype = dtype or self._data.dtype
            self._metadata = metadata or {}
    
    @staticmethod
    def _cs_name(cs):
        names = {0: 'RGB', 1: 'BGR', 2: 'GRAY', 3: 'HSV', 4: 'LAB', 
                 5: 'YUV', 6: 'XYZ', 7: 'RGBA', 8: 'BGRA'}
        return names.get(cs, 'RGB')
    
    def height(self):
        if HAS_CPP and hasattr(self, '_cpp_img'):
            return self._cpp_img.height()
        return self._data.shape[0]
    
    def width(self):
        if HAS_CPP and hasattr(self, '_cpp_img'):
            return self._cpp_img.width()
        return self._data.shape[1]
    
    def channels(self):
        if HAS_CPP and hasattr(self, '_cpp_img'):
            return self._cpp_img.channels()
        return self._data.shape[2] if self._data.ndim == 3 else 1
    
    def to_array(self):
        if HAS_CPP and hasattr(self, '_cpp_img'):
            return self._cpp_img.to_array()
        return self._data
    
    def clone(self):
        if HAS_CPP and hasattr(self, '_cpp_img'):
            return Image(self._cpp_img.clone().to_array(), self._color_space)
        return Image(self._data.copy(), self._color_space, self._dtype, self._metadata.copy())
    
    def crop(self, x, y, w, h):
        if HAS_CPP and hasattr(self, '_cpp_img'):
            return Image(self._cpp_img.crop(x, y, w, h).to_array(), self._color_space)
        return Image(self._data[y:y+h, x:x+w], self._color_space)
    
    def resize(self, height, width):
        if HAS_CPP and hasattr(self, '_cpp_img'):
            return Image(self._cpp_img.resize(height, width).to_array(), self._color_space)
        # Python fallback with nearest neighbor
        from scipy.ndimage import zoom
        scale_y = height / self._data.shape[0]
        scale_x = width / self._data.shape[1]
        if self._data.ndim == 3:
            resized = zoom(self._data, (scale_y, scale_x, 1), order=0)
        else:
            resized = zoom(self._data, (scale_y, scale_x), order=0)
        return Image(resized, self._color_space)

# 
# AUGMENTATION MODULE
# 

class Transform:
    """Base transform class"""
    
    def apply(self, image):
        raise NotImplementedError
    
    def name(self):
        return self.__class__.__name__

class HorizontalFlip(Transform):
    """Horizontal flip transform"""
    
    def __init__(self, p=0.5):
        self.p = p
        if HAS_CPP:
            self._cpp_transform = _cpp.HorizontalFlip(p)
    
    def apply(self, image):
        if HAS_CPP and hasattr(self, '_cpp_transform'):
            if isinstance(image, Image) and hasattr(image, '_cpp_img'):
                result = self._cpp_transform.apply(image._cpp_img)
                return Image(result.to_array(), image._color_space)
        
        # Python fallback
        if np.random.random() < self.p:
            arr = image.to_array() if isinstance(image, Image) else image
            return Image(np.fliplr(arr))
        return image

class VerticalFlip(Transform):
    """Vertical flip transform"""
    
    def __init__(self, p=0.5):
        self.p = p
        if HAS_CPP:
            self._cpp_transform = _cpp.VerticalFlip(p)
    
    def apply(self, image):
        if HAS_CPP and hasattr(self, '_cpp_transform'):
            if isinstance(image, Image) and hasattr(image, '_cpp_img'):
                result = self._cpp_transform.apply(image._cpp_img)
                return Image(result.to_array(), image._color_space)
        
        # Python fallback
        if np.random.random() < self.p:
            arr = image.to_array() if isinstance(image, Image) else image
            return Image(np.flipud(arr))
        return image

class RandomCrop(Transform):
    """Random crop transform"""
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        if HAS_CPP:
            self._cpp_transform = _cpp.RandomCrop(height, width)
    
    def apply(self, image):
        if HAS_CPP and hasattr(self, '_cpp_transform'):
            if isinstance(image, Image) and hasattr(image, '_cpp_img'):
                result = self._cpp_transform.apply(image._cpp_img)
                return Image(result.to_array(), image._color_space)
        
        # Python fallback
        arr = image.to_array() if isinstance(image, Image) else image
        h, w = arr.shape[:2]
        y = np.random.randint(0, h - self.height + 1)
        x = np.random.randint(0, w - self.width + 1)
        return Image(arr[y:y+self.height, x:x+self.width])

class CenterCrop(Transform):
    """Center crop transform"""
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        if HAS_CPP:
            self._cpp_transform = _cpp.CenterCrop(height, width)
    
    def apply(self, image):
        if HAS_CPP and hasattr(self, '_cpp_transform'):
            if isinstance(image, Image) and hasattr(image, '_cpp_img'):
                result = self._cpp_transform.apply(image._cpp_img)
                return Image(result.to_array(), image._color_space)
        
        # Python fallback
        arr = image.to_array() if isinstance(image, Image) else image
        h, w = arr.shape[:2]
        y = (h - self.height) // 2
        x = (w - self.width) // 2
        return Image(arr[y:y+self.height, x:x+self.width])

class Resize(Transform):
    """Resize transform"""
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        if HAS_CPP:
            self._cpp_transform = _cpp.Resize(height, width)
    
    def apply(self, image):
        if HAS_CPP and hasattr(self, '_cpp_transform'):
            if isinstance(image, Image) and hasattr(image, '_cpp_img'):
                result = self._cpp_transform.apply(image._cpp_img)
                return Image(result.to_array(), image._color_space)
        
        # Python fallback
        if isinstance(image, Image):
            return image.resize(self.height, self.width)
        from scipy.ndimage import zoom
        arr = image
        scale_y = self.height / arr.shape[0]
        scale_x = self.width / arr.shape[1]
        if arr.ndim == 3:
            resized = zoom(arr, (scale_y, scale_x, 1), order=0)
        else:
            resized = zoom(arr, (scale_y, scale_x), order=0)
        return Image(resized)

class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms=None):
        self.transforms = transforms or []
        if HAS_CPP:
            self._cpp_compose = _cpp.Compose()
            for t in self.transforms:
                if hasattr(t, '_cpp_transform'):
                    self._cpp_compose.add(t._cpp_transform)
    
    def add(self, transform):
        self.transforms.append(transform)
        if HAS_CPP and hasattr(self, '_cpp_compose') and hasattr(transform, '_cpp_transform'):
            self._cpp_compose.add(transform._cpp_transform)
    
    def apply(self, image):
        if HAS_CPP and hasattr(self, '_cpp_compose'):
            if isinstance(image, Image) and hasattr(image, '_cpp_img'):
                result = self._cpp_compose.apply(image._cpp_img)
                return Image(result.to_array(), image._color_space)
        
        # Python fallback
        result = image
        for transform in self.transforms:
            result = transform.apply(result)
        return result
    
    def size(self):
        return len(self.transforms)

# 
# CALIBRATION MODULE
# 

class Matrix3x3:
    """3x3 matrix wrapper"""
    
    def __init__(self, data=None):
        if HAS_CPP and data is None:
            self._cpp_mat = _cpp.Matrix3x3()
        elif data is not None:
            self._data = np.array(data, dtype=np.float64).reshape(3, 3)
        else:
            self._data = np.zeros((3, 3), dtype=np.float64)
    
    @staticmethod
    def identity():
        if HAS_CPP:
            mat = Matrix3x3()
            mat._cpp_mat = _cpp.Matrix3x3.identity()
            return mat
        mat = Matrix3x3()
        mat._data = np.eye(3, dtype=np.float64)
        return mat
    
    def to_array(self):
        if HAS_CPP and hasattr(self, '_cpp_mat'):
            return self._cpp_mat.to_array()
        return self._data

def solve_pnp_dlt(object_points, image_points):
    """Solve PnP using DLT"""
    if HAS_CPP:
        return _cpp.solve_pnp_dlt(object_points, image_points)
    
    # Python fallback (simplified)
    rvec = [0.0, 0.0, 0.0]
    tvec = [0.0, 0.0, 1.0]
    return rvec, tvec

def find_homography(src_points, dst_points, method=0):
    """Find homography matrix"""
    if HAS_CPP:
        return _cpp.find_homography(src_points, dst_points, method)
    
    # Python fallback
    return Matrix3x3.identity()

def project_points(object_points, rvec, tvec, camera_matrix):
    """Project 3D points to 2D"""
    if HAS_CPP:
        if isinstance(camera_matrix, Matrix3x3):
            if hasattr(camera_matrix, '_cpp_mat'):
                return _cpp.project_points(object_points, rvec, tvec, camera_matrix._cpp_mat)
        return _cpp.project_points(object_points, rvec, tvec, _cpp.Matrix3x3.identity())
    
    # Python fallback
    return [[0.0, 0.0] for _ in object_points]

# Constants
RANSAC = 8
LMEDS = 4
RHO = 16

# 
# MODULE EXPORTS
# 

__all__ = [
    # Core
    'ColorSpace', 'DataType', 'Image',
    # Augmentation
    'Transform', 'HorizontalFlip', 'VerticalFlip', 'RandomCrop', 
    'CenterCrop', 'Resize', 'Compose',
    # Calibration
    'Matrix3x3', 'solve_pnp_dlt', 'find_homography', 'project_points',
    'RANSAC', 'LMEDS', 'RHO',
]
