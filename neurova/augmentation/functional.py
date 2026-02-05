# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Functional API for image augmentation operations.

This module provides low-level functions for image transformations,
similar to torchvision.transforms.functional.
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Sequence

__all__ = [
    # Geometric transforms
    'hflip', 'vflip', 'rotate', 'affine', 'perspective',
    'resize', 'crop', 'center_crop', 'pad', 'five_crop', 'ten_crop',
    # Color transforms
    'normalize', 'adjust_brightness', 'adjust_contrast', 'adjust_saturation',
    'adjust_hue', 'adjust_gamma', 'rgb_to_grayscale', 'grayscale_to_rgb',
    'invert', 'posterize', 'solarize', 'autocontrast', 'equalize',
    # Noise and blur
    'gaussian_blur', 'gaussian_noise', 'salt_and_pepper_noise',
    # Utility functions
    'to_tensor', 'to_numpy', 'clamp',
]


# Utility Functions

def to_tensor(image: np.ndarray) -> np.ndarray:
    """
    Convert image to tensor format (C, H, W) with values in [0, 1].
    
    Args:
        image: Input image of shape (H, W) or (H, W, C)
        
    Returns:
        Image tensor of shape (C, H, W) with values in [0, 1]
    """
    if image.ndim == 2:
        # Grayscale image
        image = image[np.newaxis, :, :]
    elif image.ndim == 3:
        # Color image (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
    
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    return image.astype(np.float32)


def to_numpy(tensor: np.ndarray, scale: bool = True) -> np.ndarray:
    """
    Convert tensor format (C, H, W) to numpy image (H, W, C).
    
    Args:
        tensor: Input tensor of shape (C, H, W)
        scale: Whether to scale values to [0, 255] and convert to uint8
        
    Returns:
        Image array of shape (H, W, C) or (H, W) for grayscale
    """
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            image = tensor[0]
        else:
            image = np.transpose(tensor, (1, 2, 0))
    else:
        image = tensor
    
    if scale:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def clamp(x: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Clamp values to specified range."""
    return np.clip(x, min_val, max_val)


def _get_image_dimensions(image: np.ndarray) -> Tuple[int, int, int]:
    """Get image dimensions (C, H, W) regardless of input format."""
    if image.ndim == 2:
        return 1, image.shape[0], image.shape[1]
    elif image.ndim == 3:
        if image.shape[0] <= 4:  # Likely (C, H, W)
            return image.shape[0], image.shape[1], image.shape[2]
        else:  # Likely (H, W, C)
            return image.shape[2], image.shape[0], image.shape[1]
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")


# Geometric Transforms

def hflip(image: np.ndarray) -> np.ndarray:
    """
    Horizontally flip the given image.
    
    Args:
        image: Image of shape (C, H, W) or (H, W, C) or (H, W)
        
    Returns:
        Horizontally flipped image
    """
    if image.ndim == 2:
        return np.fliplr(image)
    elif image.ndim == 3:
        if image.shape[0] <= 4:  # (C, H, W)
            return image[:, :, ::-1].copy()
        else:  # (H, W, C)
            return np.fliplr(image)
    return image


def vflip(image: np.ndarray) -> np.ndarray:
    """
    Vertically flip the given image.
    
    Args:
        image: Image of shape (C, H, W) or (H, W, C) or (H, W)
        
    Returns:
        Vertically flipped image
    """
    if image.ndim == 2:
        return np.flipud(image)
    elif image.ndim == 3:
        if image.shape[0] <= 4:  # (C, H, W)
            return image[:, ::-1, :].copy()
        else:  # (H, W, C)
            return np.flipud(image)
    return image


def rotate(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    fill: float = 0.0
) -> np.ndarray:
    """
    Rotate the image by the given angle.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        angle: Rotation angle in degrees, counter-clockwise
        center: Center of rotation. If None, use image center
        fill: Fill value for areas outside the rotated image
        
    Returns:
        Rotated image
    """
    from scipy import ndimage
    
    if image.ndim == 2:
        return ndimage.rotate(image, angle, reshape=False, order=1, 
                             mode='constant', cval=fill)
    elif image.ndim == 3:
        if image.shape[0] <= 4:  # (C, H, W)
            result = np.zeros_like(image)
            for c in range(image.shape[0]):
                result[c] = ndimage.rotate(image[c], angle, reshape=False, 
                                          order=1, mode='constant', cval=fill)
            return result
        else:  # (H, W, C)
            return ndimage.rotate(image, angle, reshape=False, order=1,
                                 mode='constant', cval=fill)
    return image


def affine(
    image: np.ndarray,
    angle: float = 0.0,
    translate: Tuple[float, float] = (0, 0),
    scale: float = 1.0,
    shear: float = 0.0,
    fill: float = 0.0
) -> np.ndarray:
    """
    Apply affine transformation to the image.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        angle: Rotation angle in degrees
        translate: Translation (tx, ty) in pixels
        scale: Scale factor
        shear: Shear angle in degrees
        fill: Fill value for areas outside the image
        
    Returns:
        Transformed image
    """
    from scipy import ndimage
    
    # Get image dimensions
    if image.ndim == 2:
        h, w = image.shape
    elif image.shape[0] <= 4:  # (C, H, W)
        h, w = image.shape[1], image.shape[2]
    else:  # (H, W, C)
        h, w = image.shape[0], image.shape[1]
    
    # Build transformation matrix
    angle_rad = np.deg2rad(angle)
    shear_rad = np.deg2rad(shear)
    
    # Center of image
    cx, cy = w / 2, h / 2
    
    # Rotation and scale matrix
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Combined transformation matrix (rotation + scale + shear)
    matrix = np.array([
        [scale * (cos_a + shear_rad * sin_a), -scale * sin_a],
        [scale * sin_a, scale * (cos_a - shear_rad * sin_a)]
    ])
    
    # Apply transformation with translation
    offset = np.array([cy, cx]) - matrix @ np.array([cy, cx]) - np.array([translate[1], translate[0]])
    
    def _apply_affine(img_2d):
        return ndimage.affine_transform(
            img_2d, matrix, offset=offset, order=1,
            mode='constant', cval=fill
        )
    
    if image.ndim == 2:
        return _apply_affine(image)
    elif image.shape[0] <= 4:  # (C, H, W)
        result = np.zeros_like(image)
        for c in range(image.shape[0]):
            result[c] = _apply_affine(image[c])
        return result
    else:  # (H, W, C)
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _apply_affine(image[:, :, c])
        return result


def perspective(
    image: np.ndarray,
    startpoints: List[Tuple[float, float]],
    endpoints: List[Tuple[float, float]],
    fill: float = 0.0
) -> np.ndarray:
    """
    Apply perspective transformation to the image.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        startpoints: List of 4 source points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        endpoints: List of 4 destination points
        fill: Fill value for areas outside the image
        
    Returns:
        Transformed image
    """
    from scipy import ndimage
    
    # Compute perspective transformation matrix
    src = np.array(startpoints, dtype=np.float32)
    dst = np.array(endpoints, dtype=np.float32)
    
    # Build the transformation using homography
    # This is a simplified perspective transformation
    A = np.zeros((8, 8))
    b = np.zeros(8)
    
    for i in range(4):
        A[2*i] = [src[i, 0], src[i, 1], 1, 0, 0, 0, 
                  -dst[i, 0]*src[i, 0], -dst[i, 0]*src[i, 1]]
        A[2*i+1] = [0, 0, 0, src[i, 0], src[i, 1], 1,
                   -dst[i, 1]*src[i, 0], -dst[i, 1]*src[i, 1]]
        b[2*i] = dst[i, 0]
        b[2*i+1] = dst[i, 1]
    
    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return image  # Return original if transformation fails
    
    H = np.array([
        [coeffs[0], coeffs[1], coeffs[2]],
        [coeffs[3], coeffs[4], coeffs[5]],
        [coeffs[6], coeffs[7], 1]
    ])
    
    # Get image dimensions
    if image.ndim == 2:
        h, w = image.shape
    elif image.shape[0] <= 4:  # (C, H, W)
        h, w = image.shape[1], image.shape[2]
    else:  # (H, W, C)
        h, w = image.shape[0], image.shape[1]
    
    def _apply_perspective(img_2d):
        result = np.full_like(img_2d, fill)
        for y in range(h):
            for x in range(w):
                # Apply inverse homography
                denom = H[2, 0] * x + H[2, 1] * y + H[2, 2]
                if abs(denom) < 1e-8:
                    continue
                src_x = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / denom
                src_y = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / denom
                
                if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                    # Bilinear interpolation
                    x0, y0 = int(src_x), int(src_y)
                    dx, dy = src_x - x0, src_y - y0
                    result[y, x] = (
                        (1 - dx) * (1 - dy) * img_2d[y0, x0] +
                        dx * (1 - dy) * img_2d[y0, x0 + 1] +
                        (1 - dx) * dy * img_2d[y0 + 1, x0] +
                        dx * dy * img_2d[y0 + 1, x0 + 1]
                    )
        return result
    
    if image.ndim == 2:
        return _apply_perspective(image)
    elif image.shape[0] <= 4:  # (C, H, W)
        result = np.zeros_like(image)
        for c in range(image.shape[0]):
            result[c] = _apply_perspective(image[c])
        return result
    else:  # (H, W, C)
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _apply_perspective(image[:, :, c])
        return result


def resize(
    image: np.ndarray,
    size: Union[int, Tuple[int, int]],
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    Resize the image to the given size.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        size: Desired output size. If int, smaller edge will be matched.
              If tuple (h, w), output will be exactly this size.
        interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic')
        
    Returns:
        Resized image
    """
    from scipy import ndimage
    
    order_map = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}
    order = order_map.get(interpolation, 1)
    
    # Get current dimensions
    if image.ndim == 2:
        h, w = image.shape
    elif image.shape[0] <= 4:  # (C, H, W)
        h, w = image.shape[1], image.shape[2]
    else:  # (H, W, C)
        h, w = image.shape[0], image.shape[1]
    
    # Calculate target size
    if isinstance(size, int):
        if h < w:
            new_h, new_w = size, int(size * w / h)
        else:
            new_h, new_w = int(size * h / w), size
    else:
        new_h, new_w = size
    
    # Calculate zoom factors
    zoom_h, zoom_w = new_h / h, new_w / w
    
    if image.ndim == 2:
        return ndimage.zoom(image, (zoom_h, zoom_w), order=order)
    elif image.shape[0] <= 4:  # (C, H, W)
        return ndimage.zoom(image, (1, zoom_h, zoom_w), order=order)
    else:  # (H, W, C)
        return ndimage.zoom(image, (zoom_h, zoom_w, 1), order=order)


def crop(
    image: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int
) -> np.ndarray:
    """
    Crop the image at specified location.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        top: Top coordinate of crop box
        left: Left coordinate of crop box
        height: Height of crop box
        width: Width of crop box
        
    Returns:
        Cropped image
    """
    if image.ndim == 2:
        return image[top:top + height, left:left + width].copy()
    elif image.shape[0] <= 4:  # (C, H, W)
        return image[:, top:top + height, left:left + width].copy()
    else:  # (H, W, C)
        return image[top:top + height, left:left + width, :].copy()


def center_crop(image: np.ndarray, output_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Crop the image at the center.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        output_size: Desired output size (h, w) or single int for square
        
    Returns:
        Center-cropped image
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Get image dimensions
    if image.ndim == 2:
        h, w = image.shape
    elif image.shape[0] <= 4:  # (C, H, W)
        h, w = image.shape[1], image.shape[2]
    else:  # (H, W, C)
        h, w = image.shape[0], image.shape[1]
    
    crop_h, crop_w = output_size
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    
    return crop(image, top, left, crop_h, crop_w)


def pad(
    image: np.ndarray,
    padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
    fill: float = 0.0,
    padding_mode: str = 'constant'
) -> np.ndarray:
    """
    Pad the image.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        padding: Padding size. Int for all sides, tuple of 2 for (left/right, top/bottom),
                 tuple of 4 for (left, top, right, bottom)
        fill: Fill value for constant padding
        padding_mode: One of 'constant', 'edge', 'reflect', 'symmetric'
        
    Returns:
        Padded image
    """
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left, pad_top, pad_right, pad_bottom = padding
    
    mode_map = {
        'constant': 'constant',
        'edge': 'edge',
        'reflect': 'reflect',
        'symmetric': 'symmetric'
    }
    mode = mode_map.get(padding_mode, 'constant')
    
    if image.ndim == 2:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    elif image.shape[0] <= 4:  # (C, H, W)
        pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    else:  # (H, W, C)
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    
    kwargs = {'mode': mode}
    if mode == 'constant':
        kwargs['constant_values'] = fill
    
    return np.pad(image, pad_width, **kwargs)


def five_crop(
    image: np.ndarray,
    size: Union[int, Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop the image into 5 crops: 4 corners and center.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        size: Desired output size (h, w) or single int for square
        
    Returns:
        Tuple of 5 cropped images (top-left, top-right, bottom-left, bottom-right, center)
    """
    if isinstance(size, int):
        size = (size, size)
    
    # Get image dimensions
    if image.ndim == 2:
        h, w = image.shape
    elif image.shape[0] <= 4:  # (C, H, W)
        h, w = image.shape[1], image.shape[2]
    else:  # (H, W, C)
        h, w = image.shape[0], image.shape[1]
    
    crop_h, crop_w = size
    
    # Five crops
    tl = crop(image, 0, 0, crop_h, crop_w)
    tr = crop(image, 0, w - crop_w, crop_h, crop_w)
    bl = crop(image, h - crop_h, 0, crop_h, crop_w)
    br = crop(image, h - crop_h, w - crop_w, crop_h, crop_w)
    center = center_crop(image, size)
    
    return tl, tr, bl, br, center


def ten_crop(
    image: np.ndarray,
    size: Union[int, Tuple[int, int]],
    vertical_flip: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Generate ten crops for the image: 5 crops + 5 flipped versions.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        size: Desired output size
        vertical_flip: Whether to use vertical flip instead of horizontal
        
    Returns:
        Tuple of 10 cropped images
    """
    crops = five_crop(image, size)
    
    flip_fn = vflip if vertical_flip else hflip
    flipped_crops = tuple(flip_fn(c) for c in crops)
    
    return crops + flipped_crops


# Color Transforms

def normalize(
    image: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float]
) -> np.ndarray:
    """
    Normalize image with mean and standard deviation.
    
    Args:
        image: Image of shape (C, H, W) with values in [0, 1]
        mean: Mean for each channel
        std: Standard deviation for each channel
        
    Returns:
        Normalized image
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    if image.ndim == 3 and image.shape[0] <= 4:  # (C, H, W)
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
    elif image.ndim == 3:  # (H, W, C)
        mean = mean.reshape(1, 1, -1)
        std = std.reshape(1, 1, -1)
    
    return (image - mean) / std


def adjust_brightness(image: np.ndarray, brightness_factor: float) -> np.ndarray:
    """
    Adjust image brightness.
    
    Args:
        image: Image of shape (C, H, W) or (H, W) with values in [0, 1]
        brightness_factor: Factor to multiply brightness. 0 gives black,
                          1 gives original, >1 increases brightness
        
    Returns:
        Brightness-adjusted image
    """
    return clamp(image * brightness_factor)


def adjust_contrast(image: np.ndarray, contrast_factor: float) -> np.ndarray:
    """
    Adjust image contrast.
    
    Args:
        image: Image of shape (C, H, W) or (H, W) with values in [0, 1]
        contrast_factor: Factor to adjust contrast. 0 gives gray,
                        1 gives original, >1 increases contrast
        
    Returns:
        Contrast-adjusted image
    """
    # Compute mean per channel
    if image.ndim == 2:
        mean = image.mean()
    elif image.shape[0] <= 4:  # (C, H, W)
        mean = image.mean(axis=(1, 2), keepdims=True)
    else:  # (H, W, C)
        mean = image.mean(axis=(0, 1), keepdims=True)
    
    return clamp((image - mean) * contrast_factor + mean)


def adjust_saturation(image: np.ndarray, saturation_factor: float) -> np.ndarray:
    """
    Adjust image saturation.
    
    Args:
        image: RGB image of shape (3, H, W) or (H, W, 3) with values in [0, 1]
        saturation_factor: Factor to adjust saturation. 0 gives grayscale,
                          1 gives original, >1 increases saturation
        
    Returns:
        Saturation-adjusted image
    """
    if image.ndim == 2:
        return image  # Grayscale, no saturation to adjust
    
    if image.shape[0] == 3:  # (C, H, W)
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gray = np.stack([gray, gray, gray], axis=0)
    elif image.shape[2] == 3:  # (H, W, C)
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        gray = np.stack([gray, gray, gray], axis=-1)
    else:
        return image
    
    return clamp(gray + (image - gray) * saturation_factor)


def adjust_hue(image: np.ndarray, hue_factor: float) -> np.ndarray:
    """
    Adjust image hue.
    
    Args:
        image: RGB image of shape (3, H, W) or (H, W, 3) with values in [0, 1]
        hue_factor: Factor to shift hue, in range [-0.5, 0.5]
        
    Returns:
        Hue-adjusted image
    """
    if image.ndim == 2:
        return image
    
    # Convert to HSV
    is_chw = image.shape[0] == 3
    if is_chw:
        image = np.transpose(image, (1, 2, 0))
    
    # Simple RGB to HSV conversion
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c
    
    # Hue calculation
    h = np.zeros_like(r)
    mask = diff > 0
    
    mask_r = mask & (max_c == r)
    mask_g = mask & (max_c == g) & ~mask_r
    mask_b = mask & (max_c == b) & ~mask_r & ~mask_g
    
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = h / 6  # Normalize to [0, 1]
    
    # Saturation
    s = np.zeros_like(r)
    s[max_c > 0] = diff[max_c > 0] / max_c[max_c > 0]
    
    # Value
    v = max_c
    
    # Adjust hue
    h = (h + hue_factor) % 1.0
    
    # Convert back to RGB
    h = h * 6
    i = h.astype(int)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    result = np.zeros_like(image)
    
    mask0 = i == 0
    mask1 = i == 1
    mask2 = i == 2
    mask3 = i == 3
    mask4 = i == 4
    mask5 = i == 5
    
    result[mask0] = np.stack([v[mask0], t[mask0], p[mask0]], axis=-1)
    result[mask1] = np.stack([q[mask1], v[mask1], p[mask1]], axis=-1)
    result[mask2] = np.stack([p[mask2], v[mask2], t[mask2]], axis=-1)
    result[mask3] = np.stack([p[mask3], q[mask3], v[mask3]], axis=-1)
    result[mask4] = np.stack([t[mask4], p[mask4], v[mask4]], axis=-1)
    result[mask5] = np.stack([v[mask5], p[mask5], q[mask5]], axis=-1)
    
    if is_chw:
        result = np.transpose(result, (2, 0, 1))
    
    return clamp(result)


def adjust_gamma(
    image: np.ndarray,
    gamma: float,
    gain: float = 1.0
) -> np.ndarray:
    """
    Apply gamma correction.
    
    Args:
        image: Image with values in [0, 1]
        gamma: Gamma value. <1 brightens, >1 darkens
        gain: Multiplicative factor
        
    Returns:
        Gamma-corrected image
    """
    return clamp(gain * (image ** gamma))


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    Args:
        image: RGB image of shape (3, H, W) or (H, W, 3)
        
    Returns:
        Grayscale image of shape (1, H, W) or (H, W)
    """
    if image.ndim == 2:
        return image
    
    if image.shape[0] == 3:  # (C, H, W)
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        return gray[np.newaxis, :, :]
    elif image.shape[2] == 3:  # (H, W, C)
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return gray
    
    return image


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to RGB.
    
    Args:
        image: Grayscale image of shape (1, H, W) or (H, W)
        
    Returns:
        RGB image of shape (3, H, W) or (H, W, 3)
    """
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    elif image.ndim == 3:
        if image.shape[0] == 1:  # (1, H, W)
            return np.concatenate([image, image, image], axis=0)
        elif image.shape[2] == 1:  # (H, W, 1)
            return np.concatenate([image, image, image], axis=2)
    
    return image


def invert(image: np.ndarray) -> np.ndarray:
    """
    Invert the colors of an image.
    
    Args:
        image: Image with values in [0, 1]
        
    Returns:
        Inverted image
    """
    return 1.0 - image


def posterize(image: np.ndarray, bits: int) -> np.ndarray:
    """
    Reduce the number of bits for each color channel.
    
    Args:
        image: Image with values in [0, 1]
        bits: Number of bits to keep (1-8)
        
    Returns:
        Posterized image
    """
    # Convert to 8-bit, posterize, convert back
    img_255 = np.round(image * 255).astype(np.uint8)
    shift = 8 - bits
    img_255 = (img_255 >> shift) << shift
    return img_255.astype(np.float32) / 255.0


def solarize(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Solarize the image by inverting pixels above threshold.
    
    Args:
        image: Image with values in [0, 1]
        threshold: Threshold for inversion
        
    Returns:
        Solarized image
    """
    result = image.copy()
    mask = image >= threshold
    result[mask] = 1.0 - image[mask]
    return result


def autocontrast(image: np.ndarray) -> np.ndarray:
    """
    Maximize image contrast by stretching the range.
    
    Args:
        image: Image with values in [0, 1]
        
    Returns:
        Contrast-maximized image
    """
    min_val = image.min()
    max_val = image.max()
    
    if max_val - min_val < 1e-8:
        return image
    
    return (image - min_val) / (max_val - min_val)


def equalize(image: np.ndarray) -> np.ndarray:
    """
    Equalize the histogram of the image.
    
    Args:
        image: Image with values in [0, 1]
        
    Returns:
        Histogram-equalized image
    """
    def _equalize_channel(channel):
        # Convert to 256 bins
        hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 1))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        
        # Map original values to equalized values
        indices = (channel * 255).astype(int)
        indices = np.clip(indices, 0, 255)
        return cdf_normalized[indices]
    
    if image.ndim == 2:
        return _equalize_channel(image)
    elif image.shape[0] <= 4:  # (C, H, W)
        result = np.zeros_like(image)
        for c in range(image.shape[0]):
            result[c] = _equalize_channel(image[c])
        return result
    else:  # (H, W, C)
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _equalize_channel(image[:, :, c])
        return result


# Noise and Blur

def gaussian_blur(
    image: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    sigma: Union[float, Tuple[float, float]] = None
) -> np.ndarray:
    """
    Apply Gaussian blur to the image.
    
    Args:
        image: Image of shape (C, H, W) or (H, W)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation. If None, computed from kernel_size
        
    Returns:
        Blurred image
    """
    from scipy.ndimage import gaussian_filter
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    if sigma is None:
        # standard formula for default sigma
        sigma = [(k - 1) / 4 for k in kernel_size]
    elif isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)
    
    if image.ndim == 2:
        return gaussian_filter(image, sigma=sigma)
    elif image.shape[0] <= 4:  # (C, H, W)
        result = np.zeros_like(image)
        for c in range(image.shape[0]):
            result[c] = gaussian_filter(image[c], sigma=sigma)
        return result
    else:  # (H, W, C)
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = gaussian_filter(image[:, :, c], sigma=sigma)
        return result


def gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 0.1
) -> np.ndarray:
    """
    Add Gaussian noise to the image.
    
    Args:
        image: Image with values in [0, 1]
        mean: Mean of the Gaussian noise
        std: Standard deviation of the Gaussian noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(mean, std, image.shape)
    return clamp(image + noise)


def salt_and_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5
) -> np.ndarray:
    """
    Add salt and pepper noise to the image.
    
    Args:
        image: Image with values in [0, 1]
        amount: Proportion of pixels to affect
        salt_vs_pepper: Ratio of salt to pepper
        
    Returns:
        Noisy image
    """
    result = image.copy()
    
    # Salt
    num_salt = int(amount * image.size * salt_vs_pepper)
    coords = tuple(np.random.randint(0, dim, num_salt) for dim in image.shape)
    result[coords] = 1.0
    
    # Pepper
    num_pepper = int(amount * image.size * (1 - salt_vs_pepper))
    coords = tuple(np.random.randint(0, dim, num_pepper) for dim in image.shape)
    result[coords] = 0.0
    
    return result
