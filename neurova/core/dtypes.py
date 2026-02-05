# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Data type definitions and utilities for Neurova"""

import numpy as np
from enum import Enum
from typing import Union, Type


class DataType(Enum):
    """Supported data types for image arrays"""
    UINT8 = np.uint8
    UINT16 = np.uint16
    UINT32 = np.uint32
    INT8 = np.int8
    INT16 = np.int16
    INT32 = np.int32
    FLOAT16 = np.float16
    FLOAT32 = np.float32
    FLOAT64 = np.float64
    BOOL = np.bool_


# type aliases
DType = Union[DataType, Type[np.number], np.dtype]


def get_dtype(dtype: DType) -> np.dtype:
    """
    Convert various dtype representations to numpy dtype
    
    Args:
        dtype: Data type specification
        
    Returns:
        numpy dtype object
    """
    if isinstance(dtype, DataType):
        return np.dtype(dtype.value)
    elif isinstance(dtype, np.dtype):
        return dtype
    else:
        return np.dtype(dtype)


def is_integer_dtype(dtype: DType) -> bool:
    """Check if dtype is an integer type"""
    dt = get_dtype(dtype)
    return np.issubdtype(dt, np.integer)


def is_floating_dtype(dtype: DType) -> bool:
    """Check if dtype is a floating point type"""
    dt = get_dtype(dtype)
    return np.issubdtype(dt, np.floating)


def is_unsigned_dtype(dtype: DType) -> bool:
    """Check if dtype is unsigned"""
    dt = get_dtype(dtype)
    return np.issubdtype(dt, np.unsignedinteger)


def get_dtype_range(dtype: DType) -> tuple:
    """
    Get the valid range for a data type
    
    Args:
        dtype: Data type
        
    Returns:
        Tuple of (min_value, max_value)
    """
    dt = get_dtype(dtype)
    
    if is_floating_dtype(dt):
        return (0.0, 1.0)  # Standard range for float images
    elif is_integer_dtype(dt):
        info = np.iinfo(dt)
        return (info.min, info.max)
    else:
        return (0, 1)


def convert_dtype(arr: np.ndarray, target_dtype: DType, scale: bool = True) -> np.ndarray:
    """
    Convert array to target dtype with optional scaling
    
    Args:
        arr: Input array
        target_dtype: Target data type
        scale: If True, scale values to target dtype range
        
    Returns:
        Converted array
    """
    target_dt = get_dtype(target_dtype)
    
    if arr.dtype == target_dt:
        return arr
    
    if not scale:
        return arr.astype(target_dt)
    
    # get source and target ranges
    src_min, src_max = get_dtype_range(arr.dtype)
    tgt_min, tgt_max = get_dtype_range(target_dt)
    
    # normalize to [0, 1]
    if src_max != src_min:
        normalized = (arr.astype(np.float64) - src_min) / (src_max - src_min)
    else:
        normalized = arr.astype(np.float64)
    
    # scale to target range
    scaled = normalized * (tgt_max - tgt_min) + tgt_min
    
    return scaled.astype(target_dt)


def safe_cast(arr: np.ndarray, dtype: DType) -> np.ndarray:
    """
    Safely cast array to dtype, clipping values to valid range
    
    Args:
        arr: Input array
        dtype: Target data type
        
    Returns:
        Safely casted array
    """
    target_dt = get_dtype(dtype)
    min_val, max_val = get_dtype_range(target_dt)
    clipped = np.clip(arr, min_val, max_val)
    return clipped.astype(target_dt)


# common dtype shortcuts
uint8 = DataType.UINT8
uint16 = DataType.UINT16
float32 = DataType.FLOAT32
float64 = DataType.FLOAT64
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.