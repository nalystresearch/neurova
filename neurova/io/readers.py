# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Basic image file readers for Neurova

This module provides image reading functionality using minimal dependencies.
For extended format support, install pillow: pip install neurova[io]
"""

import numpy as np
import struct
import zlib
import io
from pathlib import Path
from typing import Union, Optional, Tuple
from neurova.core.image import Image
from neurova.core.constants import ColorSpace
from neurova.core.errors import IOError, FileFormatError


try:
    from neurova.io import codecs_native as _codecs_native

    _HAS_NATIVE_CODECS = True
except Exception:
    _codecs_native = None
    _HAS_NATIVE_CODECS = False

try:
    from neurova.io import codecs_extra as _codecs_extra

    _HAS_EXTRA_CODECS = True
except Exception:
    _codecs_extra = None
    _HAS_EXTRA_CODECS = False

try:
    from neurova.io import codecs_webp as _codecs_webp

    _HAS_WEBP_CODECS = True
except Exception:
    _codecs_webp = None
    _HAS_WEBP_CODECS = False


def read_image(filepath: Union[str, Path], 
              color_space: Optional[ColorSpace] = None) -> Image:
    """
    Read an image from file
    
    Args:
        filepath: Path to image file
        color_space: Desired color space (None = detect from file)
        
    Returns:
        Image object
        
    Raises:
        IOError: If file cannot be read
        FileFormatError: If format is not supported
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise IOError(f"File not found: {filepath}")
    
    # get file extension
    ext = filepath.suffix.lower()
    
    # try to read based on extension
    if ext in ('.png', ):
        if _HAS_NATIVE_CODECS:
            data = _codecs_native.imread(str(filepath))
        else:
            data = _read_png(filepath)
    elif ext in ('.bmp',):
        data = _read_bmp(filepath)
    elif ext in ('.ppm',):
        data = _read_ppm(filepath)
    elif ext in ('.jpg', '.jpeg'):
        if _HAS_NATIVE_CODECS:
            data = _codecs_native.imread(str(filepath))
        else:
            # JPEG requires external library
            try:
                from PIL import Image as PILImage

                pil_img = PILImage.open(filepath)
                data = np.array(pil_img)
            except ImportError:
                raise FileFormatError(
                    "JPEG support requires Pillow or Neurova native codecs. "
                    "Install with: pip install neurova[io]"
                )
    elif ext in ('.webp',):
        if _HAS_WEBP_CODECS:
            data = _codecs_webp.imread(str(filepath))
        else:
            raise FileFormatError("WebP support requires libwebp bindings. Set NEUROVA_BUILD_WEBP=1 and rebuild.")
    elif ext in ('.tif', '.tiff', '.exr', '.jp2', '.j2k', '.jpf'):
        if _HAS_EXTRA_CODECS:
            data = _codecs_extra.imread(str(filepath))
        else:
            raise FileFormatError(
                "Extended codecs (TIFF/EXR/JPEG2000) require Pillow or install neurova native extras"
            )
    else:
        # try Pillow as fallback
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(filepath)
            data = np.array(pil_img)
        except ImportError:
            raise FileFormatError(
                f"Unsupported format: {ext}. Install Pillow for extended format support: "
                "pip install neurova[io]"
            )
    
    # determine color space if not specified
    if color_space is None:
        if data.ndim == 2:
            color_space = ColorSpace.GRAY
        elif data.shape[2] == 3:
            color_space = ColorSpace.RGB
        elif data.shape[2] == 4:
            color_space = ColorSpace.RGBA
        else:
            color_space = ColorSpace.RGB
    
    return Image(data, color_space)


def _read_png(filepath: Path) -> np.ndarray:
    """
    Read PNG file using pure Python
    
    Args:
        filepath: Path to PNG file
        
    Returns:
        Image data as numpy array
    """
    with open(filepath, 'rb') as f:
        # verify PNG signature
        signature = f.read(8)
        if signature != b'\x89PNG\r\n\x1a\n':
            raise FileFormatError("Not a valid PNG file")
        
        width = height = bit_depth = color_type = None
        image_data = b''
        palette = None
        
        # read chunks
        while True:
            # read chunk length and type
            length_data = f.read(4)
            if len(length_data) < 4:
                break
                
            length = struct.unpack('>I', length_data)[0]
            chunk_type = f.read(4)
            chunk_data = f.read(length)
            crc = f.read(4)  # CRC (we'll skip validation for simplicity)
            
            if chunk_type == b'IHDR':
                # image header
                width, height, bit_depth, color_type, compression, filter_method, interlace = \
                    struct.unpack('>IIBBBBB', chunk_data)
                    
            elif chunk_type == b'PLTE':
                # palette
                palette = chunk_data
                
            elif chunk_type == b'IDAT':
                # image data
                image_data += chunk_data
                
            elif chunk_type == b'IEND':
                # end of image
                break
        
        if width is None or height is None:
            raise FileFormatError("Invalid PNG file: missing IHDR")
        
        # decompress image data
        try:
            raw_data = zlib.decompress(image_data)
        except zlib.error as e:
            raise FileFormatError(f"PNG decompression failed: {e}")
        
        # parse based on color type
        if color_type == 0:  # Grayscale
            channels = 1
        elif color_type == 2:  # RGB
            channels = 3
        elif color_type == 3:  # Indexed (palette)
            channels = 1
        elif color_type == 4:  # Grayscale + Alpha
            channels = 2
        elif color_type == 6:  # RGBA
            channels = 4
        else:
            raise FileFormatError(f"Unsupported PNG color type: {color_type}")
        
        # reconstruct image (simplified - no filter handling)
        bit_depth = bit_depth if bit_depth is not None else 8
        bytes_per_pixel = max(1, channels * bit_depth // 8)
        stride = width * bytes_per_pixel + 1  # +1 for filter byte
        
        # create array
        if bit_depth == 8:
            dtype = np.uint8
        elif bit_depth == 16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        
        # simple unfiltering (assumes filter type 0 - no filter)
        # for a complete implementation, we'd need to handle all filter types
        img_array = np.zeros((height, width, channels), dtype=dtype)
        
        for y in range(height):
            row_start = y * stride
            filter_type = raw_data[row_start]
            row_data = raw_data[row_start + 1:row_start + stride]
            
            if len(row_data) == width * bytes_per_pixel:
                row_array = np.frombuffer(row_data, dtype=dtype)
                if channels > 1:
                    img_array[y] = row_array.reshape(width, channels)
                else:
                    img_array[y, :, 0] = row_array
        
        if channels == 1:
            return img_array[:, :, 0]
        return img_array


def _read_bmp(filepath: Path) -> np.ndarray:
    """
    Read BMP file
    
    Args:
        filepath: Path to BMP file
        
    Returns:
        Image data as numpy array
    """
    with open(filepath, 'rb') as f:
        # read BMP file header
        header = f.read(14)
        if header[:2] != b'BM':
            raise FileFormatError("Not a valid BMP file")
        
        file_size = struct.unpack('<I', header[2:6])[0]
        pixel_offset = struct.unpack('<I', header[10:14])[0]
        
        # read DIB header
        dib_header_size = struct.unpack('<I', f.read(4))[0]
        f.seek(14)  # Go back
        dib_header = f.read(dib_header_size)
        
        if dib_header_size == 40:  # BITMAPINFOHEADER
            width, height, planes, bit_count = struct.unpack('<IiHH', dib_header[4:16])
            compression = struct.unpack('<I', dib_header[16:20])[0]
        else:
            raise FileFormatError(f"Unsupported BMP header size: {dib_header_size}")
        
        if compression != 0:
            raise FileFormatError("Compressed BMP files are not supported")
        
        # read pixel data
        f.seek(pixel_offset)
        
        # calculate row size (must be multiple of 4)
        row_size = ((bit_count * width + 31) // 32) * 4
        pixel_data = f.read()
        
        # parse pixel data
        if bit_count == 24:
            channels = 3
            img_array = np.zeros((abs(height), width, channels), dtype=np.uint8)
            
            for y in range(abs(height)):
                row_start = y * row_size
                row_data = pixel_data[row_start:row_start + width * 3]
                row_array = np.frombuffer(row_data, dtype=np.uint8, count=width * 3)
                img_array[y] = row_array.reshape(width, 3)
            
            # bMP stores BGR
            img_array = img_array[:, :, ::-1]  # Convert to RGB
            
        elif bit_count == 8:
            # grayscale
            img_array = np.zeros((abs(height), width), dtype=np.uint8)
            for y in range(abs(height)):
                row_start = y * row_size
                row_data = pixel_data[row_start:row_start + width]
                img_array[y] = np.frombuffer(row_data, dtype=np.uint8, count=width)
        else:
            raise FileFormatError(f"Unsupported BMP bit depth: {bit_count}")
        
        # bMP can store upside down
        if height > 0:
            img_array = np.flipud(img_array)
        
        return img_array


def _read_ppm(filepath: Path) -> np.ndarray:
    """
    Read PPM (Portable Pixmap) file
    
    Args:
        filepath: Path to PPM file
        
    Returns:
        Image data as numpy array
    """
    with open(filepath, 'rb') as f:
        # read magic number
        magic = f.readline().strip()
        
        if magic == b'P3':
            # aSCII format
            is_binary = False
        elif magic == b'P6':
            # binary format
            is_binary = True
        else:
            raise FileFormatError(f"Unsupported PPM format: {magic}")
        
        # skip comments
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        
        # read dimensions
        width, height = map(int, line.strip().split())
        
        # read max value
        max_val = int(f.readline().strip())
        
        if is_binary:
            # binary format
            if max_val < 256:
                dtype = np.uint8
                data = np.frombuffer(f.read(), dtype=dtype)
            else:
                dtype = np.uint16
                data = np.frombuffer(f.read(), dtype=dtype)
            
            data = data.reshape((height, width, 3))
        else:
            # aSCII format
            values = []
            for line in f:
                values.extend(map(int, line.split()))
            data = np.array(values, dtype=np.uint8).reshape((height, width, 3))
        
        return data


def read_from_bytes(data: bytes, format: str = 'auto') -> Image:
    """
    Read image from bytes
    
    Args:
        data: Image data as bytes
        format: Image format ('auto', 'png', 'jpeg', 'bmp')
        
    Returns:
        Image object
    """
    if format == 'auto':
        # try to detect format from magic bytes
        if data.startswith(b'\x89PNG'):
            format = 'png'
        elif data.startswith(b'\xff\xd8'):
            format = 'jpeg'
        elif data.startswith(b'BM'):
            format = 'bmp'
        elif data.startswith(b'P6') or data.startswith(b'P3'):
            format = 'ppm'
    
    # create temporary file-like object
    stream = io.BytesIO(data)
    
    # for now, use Pillow for bytes reading
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(stream)
        arr = np.array(pil_img)
        
        if arr.ndim == 2:
            color_space = ColorSpace.GRAY
        elif arr.shape[2] == 3:
            color_space = ColorSpace.RGB
        else:
            color_space = ColorSpace.RGBA
            
        return Image(arr, color_space)
    except ImportError:
        raise IOError("Reading from bytes requires Pillow. Install with: pip install neurova[io]")


def imread(filepath: Union[str, Path], 
          flags: Optional[int] = None) -> np.ndarray:
    """
    Read image and return numpy array (Neurova API)
    
    Args:
        filepath: Path to image file
        flags: Read flags (ignored for compatibility)
        
    Returns:
        Image data as numpy array
    """
    img = read_image(filepath)
    return img.data
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.