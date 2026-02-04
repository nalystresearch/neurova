# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Core Image class for Neurova"""

import numpy as np
from typing import Union, Tuple, Optional, Any
from dataclasses import dataclass
from neurova.core.constants import ColorSpace
from neurova.core.dtypes import DataType, get_dtype, convert_dtype
from neurova.core.errors import InvalidImageError, DimensionError
from neurova.core.array_ops import (
    ensure_array, validate_image_shape, get_spatial_shape, get_num_channels
)


@dataclass
class ImageInfo:
    """Metadata for an image"""
    width: int
    height: int
    channels: int
    dtype: np.dtype
    color_space: ColorSpace
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get full shape tuple"""
        if self.channels == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.channels)
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get (width, height) size"""
        return (self.width, self.height)
    
    @property
    def total_pixels(self) -> int:
        """Get total number of pixels"""
        return self.width * self.height


class Image:
    """
    Core Image class for Neurova
    
    Represents an image with associated metadata and operations.
    Images are immutable - operations return new Image objects.
    
    Attributes:
        data: The underlying numpy array (H, W) or (H, W, C)
        color_space: Color space of the image
        metadata: Additional metadata dictionary
    """
    
    def __init__(self, 
                 data: Union[np.ndarray, list, tuple],
                 color_space: Union[ColorSpace, str] = ColorSpace.RGB,
                 dtype: Optional[DataType] = None,
                 metadata: Optional[dict] = None):
        """
        Initialize an Image
        
        Args:
            data: Image data as numpy array or array-like
            color_space: Color space of the image
            dtype: Data type (if None, keep original)
            metadata: Optional metadata dictionary
            
        Raises:
            InvalidImageError: If data is invalid
        """
        # ensure numpy array
        self._data = ensure_array(data)
        
        # validate shape
        validate_image_shape(self._data, "Image data")
        
        # convert dtype if specified
        if dtype is not None:
            self._data = convert_dtype(self._data, dtype, scale=False)
        
        # ensure color space is enum
        if isinstance(color_space, str):
            try:
                color_space = ColorSpace[color_space.upper()]
            except KeyError:
                raise InvalidImageError(f"Unknown color space: {color_space}")
        
        self._color_space = color_space
        self._metadata = metadata or {}
        
        # validate color space matches channels
        self._validate_color_space()
    
    def _validate_color_space(self) -> None:
        """Validate that color space matches number of channels"""
        channels = get_num_channels(self._data)
        
        # expected channels for each color space
        expected_channels = {
            ColorSpace.GRAY: 1,
            ColorSpace.RGB: 3,
            ColorSpace.BGR: 3,
            ColorSpace.RGBA: 4,
            ColorSpace.BGRA: 4,
            ColorSpace.HSV: 3,
            ColorSpace.HSL: 3,
            ColorSpace.LAB: 3,
            ColorSpace.LUV: 3,
            ColorSpace.YCRCB: 3,
            ColorSpace.XYZ: 3,
        }
        
        expected = expected_channels.get(self._color_space)
        if expected and channels != expected:
            raise InvalidImageError(
                f"Color space {self._color_space.value} requires {expected} channels, "
                f"got {channels}"
            )
    
    @property
    def data(self) -> np.ndarray:
        """Get image data (read-only view)"""
        return self._data
    
    @property
    def color_space(self) -> ColorSpace:
        """Get color space"""
        return self._color_space
    
    @property
    def metadata(self) -> dict:
        """Get metadata dictionary"""
        return self._metadata.copy()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get image shape (H, W) or (H, W, C)"""
        return self._data.shape
    
    @property
    def height(self) -> int:
        """Get image height"""
        return self._data.shape[0]
    
    @property
    def width(self) -> int:
        """Get image width"""
        return self._data.shape[1]
    
    @property
    def channels(self) -> int:
        """Get number of channels"""
        return get_num_channels(self._data)
    
    @property
    def dtype(self) -> np.dtype:
        """Get data type"""
        return self._data.dtype
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get (width, height) size"""
        return (self.width, self.height)
    
    @property
    def total_pixels(self) -> int:
        """Get total number of pixels"""
        return self.height * self.width
    
    @property
    def is_grayscale(self) -> bool:
        """Check if image is grayscale"""
        return self.channels == 1 or self._color_space == ColorSpace.GRAY
    
    @property
    def is_color(self) -> bool:
        """Check if image is color"""
        return not self.is_grayscale
    
    @property
    def has_alpha(self) -> bool:
        """Check if image has alpha channel"""
        return self._color_space in (ColorSpace.RGBA, ColorSpace.BGRA)
    
    @property
    def info(self) -> ImageInfo:
        """Get image metadata"""
        return ImageInfo(
            width=self.width,
            height=self.height,
            channels=self.channels,
            dtype=self.dtype,
            color_space=self._color_space
        )
    
    def copy(self) -> 'Image':
        """Create a deep copy of the image"""
        return Image(
            data=self._data.copy(),
            color_space=self._color_space,
            metadata=self._metadata.copy()
        )
    
    def as_array(self) -> np.ndarray:
        """Get a copy of the underlying array"""
        return self._data.copy()
    
    def with_metadata(self, **kwargs) -> 'Image':
        """Create new image with updated metadata"""
        new_metadata = self._metadata.copy()
        new_metadata.update(kwargs)
        return Image(
            data=self._data.copy(),
            color_space=self._color_space,
            metadata=new_metadata
        )
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"Image(shape={self.shape}, dtype={self.dtype}, "
                f"color_space={self._color_space.value})")
    
    def __str__(self) -> str:
        """Human-readable string"""
        return self.__repr__()
    
    def __array__(self) -> np.ndarray:
        """Support numpy array interface"""
        return self._data
    
    def __getitem__(self, key) -> 'Image':
        """
        Support indexing and slicing
        
        Returns a new Image with the sliced data
        """
        sliced_data = self._data[key]
        
        # if result is not 2D or 3D, return just the array
        if sliced_data.ndim < 2:
            return sliced_data
        
        return Image(
            data=sliced_data,
            color_space=self._color_space,
            metadata=self._metadata.copy()
        )
    
    def __eq__(self, other) -> bool:
        """Check equality"""
        if not isinstance(other, Image):
            return False
        return (np.array_equal(self._data, other._data) and 
                self._color_space == other._color_space)
    
    def __ne__(self, other) -> bool:
        """Check inequality"""
        return not self.__eq__(other)


def create_blank_image(width: int, height: int, 
                      channels: int = 3,
                      dtype: DataType = DataType.UINT8,
                      color_space: ColorSpace = ColorSpace.RGB,
                      fill_value: float = 0) -> Image:
    """
    Create a blank image filled with a constant value
    
    Args:
        width: Image width
        height: Image height
        channels: Number of channels
        dtype: Data type
        color_space: Color space
        fill_value: Fill value
        
    Returns:
        New blank Image
    """
    dt = get_dtype(dtype)
    if channels == 1:
        data = np.full((height, width), fill_value, dtype=dt)
    else:
        data = np.full((height, width, channels), fill_value, dtype=dt)
    
    return Image(data, color_space)


def create_from_array(arr: np.ndarray, 
                     color_space: ColorSpace = ColorSpace.RGB) -> Image:
    """
    Create Image from numpy array
    
    Args:
        arr: Numpy array
        color_space: Color space
        
    Returns:
        New Image
    """
    return Image(arr, color_space)
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.