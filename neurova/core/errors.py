# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Custom exceptions for Neurova library"""

from typing import Optional


class NeurovaError(Exception):
    """Base exception class for all Neurova errors"""
    pass


class ImageError(NeurovaError):
    """Raised when image operations fail"""
    pass


class InvalidImageError(ImageError):
    """Raised when image data is invalid"""
    pass


class DimensionError(ImageError):
    """Raised when image dimensions are incompatible"""
    pass


class ColorSpaceError(ImageError):
    """Raised when color space operations fail"""
    pass


class IOError(NeurovaError):
    """Raised when I/O operations fail"""
    pass


class FileFormatError(IOError):
    """Raised when file format is not supported"""
    pass


class TransformError(NeurovaError):
    """Raised when transformation operations fail"""
    pass


class FilterError(NeurovaError):
    """Raised when filtering operations fail"""
    pass


class KernelError(FilterError):
    """Raised when kernel operations fail"""
    pass


class FeatureError(NeurovaError):
    """Raised when feature detection/description fails"""
    pass


class NeuralError(NeurovaError):
    """Raised when neural network operations fail"""
    pass


class LayerError(NeuralError):
    """Raised when layer operations fail"""
    pass


class OptimizationError(NeuralError):
    """Raised when optimization fails"""
    pass


class SegmentationError(NeurovaError):
    """Raised when segmentation operations fail"""
    pass


class DetectionError(NeurovaError):
    """Raised when object detection fails"""
    pass


class CalibrationError(NeurovaError):
    """Raised when camera calibration fails"""
    pass


class VideoError(NeurovaError):
    """Raised when video processing fails"""
    pass


class ValidationError(NeurovaError):
    """Raised when input validation fails"""
    
    def __init__(self, parameter: str, value, expected: str, message: Optional[str] = None):
        self.parameter = parameter
        self.value = value
        self.expected = expected
        if message is None:
            message = f"Invalid value for '{parameter}': got {value}, expected {expected}"
        super().__init__(message)


class ShapeError(NeurovaError):
    """Raised when array shapes are incompatible"""
    
    def __init__(self, expected_shape: tuple, actual_shape: tuple, message: Optional[str] = None):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        if message is None:
            message = f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        super().__init__(message)


class DataTypeError(NeurovaError):
    """Raised when data types are incompatible"""
    
    def __init__(self, expected_dtype: str, actual_dtype: str, message: Optional[str] = None):
        self.expected_dtype = expected_dtype
        self.actual_dtype = actual_dtype
        if message is None:
            message = f"Data type mismatch: expected {expected_dtype}, got {actual_dtype}"
        super().__init__(message)
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.