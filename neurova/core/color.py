# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Color space conversion operations"""

import numpy as np
from typing import Optional
from neurova.core.constants import ColorSpace, RGB_TO_GRAY_WEIGHTS
from neurova.core.errors import ColorSpaceError, ValidationError
from neurova.core.array_ops import ensure_array, clip_array


def to_grayscale(image: np.ndarray, 
                 from_space: ColorSpace = ColorSpace.RGB,
                 weights: Optional[tuple] = None) -> np.ndarray:
    """
    Convert color image to grayscale
    
    Args:
        image: Input image array (H, W, C)
        from_space: Source color space
        weights: Custom RGB weights (R, G, B), default uses standard luminance
        
    Returns:
        Grayscale image (H, W) or (H, W, 1)
    """
    if image.ndim == 2:
        return image  # Already grayscale
    
    if image.shape[2] == 1:
        return image[:, :, 0]  # Already grayscale with channel dim
    
    if weights is None:
        if from_space == ColorSpace.BGR:
            weights = RGB_TO_GRAY_WEIGHTS[::-1]
        else:
            weights = RGB_TO_GRAY_WEIGHTS

    
    if from_space in (ColorSpace.RGB, ColorSpace.BGR):
        # weighted sum
        w = weights if weights is not None else RGB_TO_GRAY_WEIGHTS
        gray = (image[:, :, 0] * w[0] + 
                image[:, :, 1] * w[1] + 
                image[:, :, 2] * w[2])
    elif from_space == ColorSpace.HSV:
        # use V channel
        gray = image[:, :, 2]
    elif from_space == ColorSpace.HSL:
        # use L channel
        gray = image[:, :, 2]
    elif from_space == ColorSpace.LAB:
        # use L channel (already in [0, 100])
        gray = image[:, :, 0] * 2.55  # Scale to [0, 255] for uint8
    else:
        # convert to RGB first
        rgb = convert_color_space(image, from_space, ColorSpace.RGB)
        return to_grayscale(rgb, ColorSpace.RGB, weights)
    
    return gray.astype(image.dtype)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to HSV color space
    
    Args:
        rgb: RGB image array (H, W, 3) in range [0, 255] or [0, 1]
        
    Returns:
        HSV image array (H, W, 3)
        H in [0, 360), S in [0, 1], V in [0, 1]
    """
    rgb = ensure_array(rgb).astype(np.float32)
    
    # normalize to [0, 1]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val
    
    # value channel
    v = max_val
    
    # saturation channel
    s = np.where(max_val == 0, 0, diff / max_val)
    
    # hue channel
    h = np.zeros_like(r)
    
    # red is max
    mask = (max_val == r) & (diff > 0)
    h[mask] = 60 * (((g[mask] - b[mask]) / diff[mask]) % 6)
    
    # green is max
    mask = (max_val == g) & (diff > 0)
    h[mask] = 60 * (((b[mask] - r[mask]) / diff[mask]) + 2)
    
    # blue is max
    mask = (max_val == b) & (diff > 0)
    h[mask] = 60 * (((r[mask] - g[mask]) / diff[mask]) + 4)
    
    hsv = np.stack([h, s, v], axis=2)
    return hsv


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert HSV to RGB color space
    
    Args:
        hsv: HSV image array (H, W, 3)
             H in [0, 360), S in [0, 1], V in [0, 1]
        
    Returns:
        RGB image array (H, W, 3) in range [0, 1]
    """
    hsv = ensure_array(hsv).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c
    
    rgb = np.zeros_like(hsv)
    
    # define regions
    regions = [
        (h >= 0) & (h < 60),
        (h >= 60) & (h < 120),
        (h >= 120) & (h < 180),
        (h >= 180) & (h < 240),
        (h >= 240) & (h < 300),
        (h >= 300) & (h < 360),
    ]
    
    # rGB values for each region
    rgb_values = [
        (c, x, 0),
        (x, c, 0),
        (0, c, x),
        (0, x, c),
        (x, 0, c),
        (c, 0, x),
    ]
    
    for region, (r_val, g_val, b_val) in zip(regions, rgb_values):
        rgb[region, 0] = r_val[region]
        rgb[region, 1] = g_val[region]
        rgb[region, 2] = b_val[region]
    
    rgb = rgb + m[:, :, np.newaxis]
    return rgb


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to LAB color space
    
    Args:
        rgb: RGB image array (H, W, 3) in range [0, 255] or [0, 1]
        
    Returns:
        LAB image array (H, W, 3)
        L in [0, 100], a in [-128, 127], b in [-128, 127]
    """
    rgb = ensure_array(rgb).astype(np.float32)
    
    # normalize to [0, 1]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    # rGB to XYZ
    xyz = rgb_to_xyz(rgb)
    
    # xYZ to LAB
    return xyz_to_lab(xyz)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert LAB to RGB color space
    
    Args:
        lab: LAB image array (H, W, 3)
        
    Returns:
        RGB image array (H, W, 3) in range [0, 1]
    """
    lab = ensure_array(lab).astype(np.float32)
    
    # lAB to XYZ
    xyz = lab_to_xyz(lab)
    
    # xYZ to RGB
    return xyz_to_rgb(xyz)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to XYZ color space
    
    Args:
        rgb: RGB image array (H, W, 3) in range [0, 1]
        
    Returns:
        XYZ image array (H, W, 3)
    """
    rgb = ensure_array(rgb).astype(np.float32)
    
    # gamma correction
    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] = rgb[~mask] / 12.92
    
    # transformation matrix (sRGB to XYZ, D65 illuminant)
    transform = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = np.dot(rgb, transform.T)
    return xyz


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to RGB color space
    
    Args:
        xyz: XYZ image array (H, W, 3)
        
    Returns:
        RGB image array (H, W, 3) in range [0, 1]
    """
    xyz = ensure_array(xyz).astype(np.float32)
    
    # inverse transformation matrix (XYZ to sRGB, D65 illuminant)
    transform = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])
    
    rgb = np.dot(xyz, transform.T)
    
    # inverse gamma correction
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * np.power(rgb[mask], 1/2.4) - 0.055
    rgb[~mask] = rgb[~mask] * 12.92
    
    return np.clip(rgb, 0, 1)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to LAB color space
    
    Args:
        xyz: XYZ image array (H, W, 3)
        
    Returns:
        LAB image array (H, W, 3)
    """
    xyz = ensure_array(xyz).astype(np.float32)
    
    # d65 illuminant reference white
    ref_white = np.array([0.95047, 1.00000, 1.08883])
    
    # normalize by reference white
    xyz = xyz / ref_white
    
    # apply f(t) function
    epsilon = 0.008856
    kappa = 903.3
    
    mask = xyz > epsilon
    xyz[mask] = np.power(xyz[mask], 1/3)
    xyz[~mask] = (kappa * xyz[~mask] + 16) / 116
    
    # calculate LAB
    lab = np.zeros_like(xyz)
    lab[:, :, 0] = 116 * xyz[:, :, 1] - 16  # L
    lab[:, :, 1] = 500 * (xyz[:, :, 0] - xyz[:, :, 1])  # a
    lab[:, :, 2] = 200 * (xyz[:, :, 1] - xyz[:, :, 2])  # b
    
    return lab


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """
    Convert LAB to XYZ color space
    
    Args:
        lab: LAB image array (H, W, 3)
        
    Returns:
        XYZ image array (H, W, 3)
    """
    lab = ensure_array(lab).astype(np.float32)
    
    # calculate intermediate values
    fy = (lab[:, :, 0] + 16) / 116
    fx = lab[:, :, 1] / 500 + fy
    fz = fy - lab[:, :, 2] / 200
    
    # apply inverse f(t) function
    epsilon = 0.008856
    kappa = 903.3
    
    xyz = np.zeros_like(lab)
    
    # x
    mask = fx ** 3 > epsilon
    xyz[mask, 0] = fx[mask] ** 3
    xyz[~mask, 0] = (116 * fx[~mask] - 16) / kappa
    
    # y
    mask = lab[:, :, 0] > kappa * epsilon
    xyz[mask, 1] = ((lab[mask, 0] + 16) / 116) ** 3
    xyz[~mask, 1] = lab[~mask, 0] / kappa
    
    # z
    mask = fz ** 3 > epsilon
    xyz[mask, 2] = fz[mask] ** 3
    xyz[~mask, 2] = (116 * fz[~mask] - 16) / kappa
    
    # d65 illuminant reference white
    ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz * ref_white
    
    return xyz


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to YCrCb color space
    
    Args:
        rgb: RGB image array (H, W, 3) in range [0, 255]
        
    Returns:
        YCrCb image array (H, W, 3)
    """
    rgb = ensure_array(rgb).astype(np.float32)
    
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    
    ycrcb = np.stack([y, cr, cb], axis=2)
    return ycrcb


def ycrcb_to_rgb(ycrcb: np.ndarray) -> np.ndarray:
    """
    Convert YCrCb to RGB color space
    
    Args:
        ycrcb: YCrCb image array (H, W, 3)
        
    Returns:
        RGB image array (H, W, 3) in range [0, 255]
    """
    ycrcb = ensure_array(ycrcb).astype(np.float32)
    
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    
    r = y + 1.403 * (cr - 128)
    g = y - 0.714 * (cr - 128) - 0.344 * (cb - 128)
    b = y + 1.773 * (cb - 128)
    
    rgb = np.stack([r, g, b], axis=2)
    return np.clip(rgb, 0, 255)


def convert_color_space(image: np.ndarray,
                       from_space: ColorSpace,
                       to_space: ColorSpace) -> np.ndarray:
    """
    Convert image between color spaces
    
    Args:
        image: Input image array
        from_space: Source color space
        to_space: Target color space
        
    Returns:
        Converted image array
        
    Raises:
        ColorSpaceError: If conversion is not supported
    """
    if from_space == to_space:
        return image.copy()
    
    # conversion routing
    conversions = {
        (ColorSpace.RGB, ColorSpace.GRAY): lambda x: to_grayscale(x, ColorSpace.RGB),
        (ColorSpace.BGR, ColorSpace.GRAY): lambda x: to_grayscale(x, ColorSpace.BGR),
        (ColorSpace.RGB, ColorSpace.HSV): rgb_to_hsv,
        (ColorSpace.HSV, ColorSpace.RGB): hsv_to_rgb,
        (ColorSpace.RGB, ColorSpace.LAB): rgb_to_lab,
        (ColorSpace.LAB, ColorSpace.RGB): lab_to_rgb,
        (ColorSpace.RGB, ColorSpace.XYZ): rgb_to_xyz,
        (ColorSpace.XYZ, ColorSpace.RGB): xyz_to_rgb,
        (ColorSpace.RGB, ColorSpace.YCRCB): rgb_to_ycrcb,
        (ColorSpace.YCRCB, ColorSpace.RGB): ycrcb_to_rgb,
        (ColorSpace.BGR, ColorSpace.RGB): lambda x: x[:, :, ::-1],
        (ColorSpace.RGB, ColorSpace.BGR): lambda x: x[:, :, ::-1],
    }
    
    key = (from_space, to_space)
    
    if key in conversions:
        return conversions[key](image)
    
    # try indirect conversion through RGB
    if from_space != ColorSpace.RGB and to_space != ColorSpace.RGB:
        rgb = convert_color_space(image, from_space, ColorSpace.RGB)
        return convert_color_space(rgb, ColorSpace.RGB, to_space)
    
    raise ColorSpaceError(
        f"Conversion from {from_space.value} to {to_space.value} is not supported"
    )
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.