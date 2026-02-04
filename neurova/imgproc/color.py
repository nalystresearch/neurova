# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova.imgproc.color - Color space conversion functions

Provides Neurova color conversion with full COLOR_* constant coverage.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

# Color Conversion Codes (Neurova)

# BGR <-> RGB
COLOR_BGR2BGRA = 0
COLOR_RGB2RGBA = COLOR_BGR2BGRA
COLOR_BGRA2BGR = 1
COLOR_RGBA2RGB = COLOR_BGRA2BGR
COLOR_BGR2RGBA = 2
COLOR_RGB2BGRA = COLOR_BGR2RGBA
COLOR_RGBA2BGR = 3
COLOR_BGRA2RGB = COLOR_RGBA2BGR
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = COLOR_BGR2RGB
COLOR_BGRA2RGBA = 5
COLOR_RGBA2BGRA = COLOR_BGRA2RGBA

# BGR <-> Gray
COLOR_BGR2GRAY = 6
COLOR_RGB2GRAY = 7
COLOR_GRAY2BGR = 8
COLOR_GRAY2RGB = COLOR_GRAY2BGR
COLOR_GRAY2BGRA = 9
COLOR_GRAY2RGBA = COLOR_GRAY2BGRA
COLOR_BGRA2GRAY = 10
COLOR_RGBA2GRAY = 11

# BGR <-> BGR565
COLOR_BGR2BGR565 = 12
COLOR_RGB2BGR565 = 13
COLOR_BGR5652BGR = 14
COLOR_BGR5652RGB = 15
COLOR_BGRA2BGR565 = 16
COLOR_RGBA2BGR565 = 17
COLOR_BGR5652BGRA = 18
COLOR_BGR5652RGBA = 19

# BGR <-> BGR555
COLOR_GRAY2BGR555 = 20
COLOR_BGR5552GRAY = 21
COLOR_BGR2BGR555 = 22
COLOR_RGB2BGR555 = 23
COLOR_BGR5552BGR = 24
COLOR_BGR5552RGB = 25
COLOR_BGRA2BGR555 = 26
COLOR_RGBA2BGR555 = 27
COLOR_BGR5552BGRA = 28
COLOR_BGR5552RGBA = 29

# BGR <-> XYZ
COLOR_BGR2XYZ = 32
COLOR_RGB2XYZ = 33
COLOR_XYZ2BGR = 34
COLOR_XYZ2RGB = 35

# BGR <-> YCrCb
COLOR_BGR2YCrCb = 36
COLOR_RGB2YCrCb = 37
COLOR_YCrCb2BGR = 38
COLOR_YCrCb2RGB = 39

# BGR <-> HSV
COLOR_BGR2HSV = 40
COLOR_RGB2HSV = 41
COLOR_HSV2BGR = 54
COLOR_HSV2RGB = 55

# BGR <-> HLS
COLOR_BGR2HLS = 52
COLOR_RGB2HLS = 53
COLOR_HLS2BGR = 60
COLOR_HLS2RGB = 61

# BGR <-> Lab
COLOR_BGR2Lab = 44
COLOR_RGB2Lab = 45
COLOR_Lab2BGR = 56
COLOR_Lab2RGB = 57

# BGR <-> Luv
COLOR_BGR2Luv = 50
COLOR_RGB2Luv = 51
COLOR_Luv2BGR = 58
COLOR_Luv2RGB = 59

# BGR <-> YUV
COLOR_BGR2YUV = 82
COLOR_RGB2YUV = 83
COLOR_YUV2BGR = 84
COLOR_YUV2RGB = 85

# YUV 4:2:0 family
COLOR_YUV2RGB_NV12 = 90
COLOR_YUV2BGR_NV12 = 91
COLOR_YUV2RGB_NV21 = 92
COLOR_YUV2BGR_NV21 = 93
COLOR_YUV420sp2RGB = COLOR_YUV2RGB_NV21
COLOR_YUV420sp2BGR = COLOR_YUV2BGR_NV21

COLOR_YUV2RGBA_NV12 = 94
COLOR_YUV2BGRA_NV12 = 95
COLOR_YUV2RGBA_NV21 = 96
COLOR_YUV2BGRA_NV21 = 97
COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21
COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21

COLOR_YUV2RGB_YV12 = 98
COLOR_YUV2BGR_YV12 = 99
COLOR_YUV2RGB_IYUV = 100
COLOR_YUV2BGR_IYUV = 101
COLOR_YUV2RGB_I420 = COLOR_YUV2RGB_IYUV
COLOR_YUV2BGR_I420 = COLOR_YUV2BGR_IYUV
COLOR_YUV420p2RGB = COLOR_YUV2RGB_YV12
COLOR_YUV420p2BGR = COLOR_YUV2BGR_YV12

# HSV/HLS full range
COLOR_BGR2HSV_FULL = 66
COLOR_RGB2HSV_FULL = 67
COLOR_HSV2BGR_FULL = 68
COLOR_HSV2RGB_FULL = 69
COLOR_BGR2HLS_FULL = 68
COLOR_RGB2HLS_FULL = 69
COLOR_HLS2BGR_FULL = 70
COLOR_HLS2RGB_FULL = 71

# Bayer patterns
COLOR_BayerBG2BGR = 46
COLOR_BayerGB2BGR = 47
COLOR_BayerRG2BGR = 48
COLOR_BayerGR2BGR = 49
COLOR_BayerBG2RGB = COLOR_BayerRG2BGR
COLOR_BayerGB2RGB = COLOR_BayerGR2BGR
COLOR_BayerRG2RGB = COLOR_BayerBG2BGR
COLOR_BayerGR2RGB = COLOR_BayerGB2BGR

COLOR_BayerBG2GRAY = 86
COLOR_BayerGB2GRAY = 87
COLOR_BayerRG2GRAY = 88
COLOR_BayerGR2GRAY = 89


def cvtColor(src: np.ndarray, code: int, dstCn: int = 0) -> np.ndarray:
    """Convert an image from one color space to another.
    
    Args:
        src: Source image
        code: Color conversion code (e.g., COLOR_BGR2GRAY)
        dstCn: Number of channels in destination (0 = automatic)
    
    Returns:
        Converted image
    """
    if src.size == 0:
        return src.copy()
    
    # BGR <-> RGB conversions
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return src[:, :, ::-1].copy()
    
    if code in (COLOR_BGRA2RGBA, COLOR_RGBA2BGRA):
        result = src.copy()
        result[:, :, 0], result[:, :, 2] = src[:, :, 2].copy(), src[:, :, 0].copy()
        return result
    
    if code == COLOR_BGR2BGRA:
        if src.ndim == 2:
            return np.dstack([src, src, src, np.full_like(src, 255)])
        alpha = np.full((src.shape[0], src.shape[1]), 255, dtype=src.dtype)
        return np.dstack([src, alpha])
    
    if code == COLOR_BGRA2BGR:
        return src[:, :, :3].copy()
    
    if code == COLOR_BGR2RGBA:
        alpha = np.full((src.shape[0], src.shape[1]), 255, dtype=src.dtype)
        return np.dstack([src[:, :, ::-1], alpha])
    
    if code == COLOR_RGBA2BGR:
        return src[:, :, 2::-1].copy()
    
    # Grayscale conversions
    if code == COLOR_BGR2GRAY:
        if src.ndim == 2:
            return src.copy()
        # ITU-R BT.601 conversion
        b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(src.dtype)
        return gray
    
    if code == COLOR_RGB2GRAY:
        if src.ndim == 2:
            return src.copy()
        r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(src.dtype)
        return gray
    
    if code == COLOR_BGRA2GRAY:
        if src.ndim == 2:
            return src.copy()
        b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(src.dtype)
        return gray
    
    if code == COLOR_RGBA2GRAY:
        if src.ndim == 2:
            return src.copy()
        r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(src.dtype)
        return gray
    
    if code == COLOR_GRAY2BGR:
        return np.dstack([src, src, src])
    
    if code == COLOR_GRAY2BGRA:
        alpha = np.full_like(src, 255)
        return np.dstack([src, src, src, alpha])
    
    # HSV conversions
    if code in (COLOR_BGR2HSV, COLOR_RGB2HSV, COLOR_BGR2HSV_FULL, COLOR_RGB2HSV_FULL):
        return _bgr_to_hsv(src, code in (COLOR_RGB2HSV, COLOR_RGB2HSV_FULL),
                          code in (COLOR_BGR2HSV_FULL, COLOR_RGB2HSV_FULL))
    
    if code in (COLOR_HSV2BGR, COLOR_HSV2RGB, COLOR_HSV2BGR_FULL, COLOR_HSV2RGB_FULL):
        return _hsv_to_bgr(src, code in (COLOR_HSV2RGB, COLOR_HSV2RGB_FULL),
                          code in (COLOR_HSV2BGR_FULL, COLOR_HSV2RGB_FULL))
    
    # HLS conversions
    if code in (COLOR_BGR2HLS, COLOR_RGB2HLS):
        return _bgr_to_hls(src, code == COLOR_RGB2HLS)
    
    if code in (COLOR_HLS2BGR, COLOR_HLS2RGB):
        return _hls_to_bgr(src, code == COLOR_HLS2RGB)
    
    # Lab conversions
    if code in (COLOR_BGR2Lab, COLOR_RGB2Lab):
        return _bgr_to_lab(src, code == COLOR_RGB2Lab)
    
    if code in (COLOR_Lab2BGR, COLOR_Lab2RGB):
        return _lab_to_bgr(src, code == COLOR_Lab2RGB)
    
    # YCrCb conversions
    if code in (COLOR_BGR2YCrCb, COLOR_RGB2YCrCb):
        return _bgr_to_ycrcb(src, code == COLOR_RGB2YCrCb)
    
    if code in (COLOR_YCrCb2BGR, COLOR_YCrCb2RGB):
        return _ycrcb_to_bgr(src, code == COLOR_YCrCb2RGB)
    
    # YUV conversions
    if code in (COLOR_BGR2YUV, COLOR_RGB2YUV):
        return _bgr_to_yuv(src, code == COLOR_RGB2YUV)
    
    if code in (COLOR_YUV2BGR, COLOR_YUV2RGB):
        return _yuv_to_bgr(src, code == COLOR_YUV2RGB)
    
    # XYZ conversions
    if code in (COLOR_BGR2XYZ, COLOR_RGB2XYZ):
        return _bgr_to_xyz(src, code == COLOR_RGB2XYZ)
    
    if code in (COLOR_XYZ2BGR, COLOR_XYZ2RGB):
        return _xyz_to_bgr(src, code == COLOR_XYZ2RGB)
    
    raise NotImplementedError(f"Color conversion code {code} not implemented")


def _bgr_to_hsv(src: np.ndarray, is_rgb: bool, full_range: bool) -> np.ndarray:
    """Convert BGR/RGB to HSV."""
    if is_rgb:
        r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    else:
        b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    
    if src.dtype == np.uint8:
        r, g, b = r / 255, g / 255, b / 255
    
    v = np.maximum(np.maximum(r, g), b)
    s = np.zeros_like(v)
    h = np.zeros_like(v)
    
    diff = v - np.minimum(np.minimum(r, g), b)
    
    # Saturation
    mask = v > 0
    s[mask] = diff[mask] / v[mask]
    
    # Hue
    mask = diff > 0
    
    mask_r = mask & (v == r)
    h[mask_r] = 60 * (g[mask_r] - b[mask_r]) / diff[mask_r]
    
    mask_g = mask & (v == g)
    h[mask_g] = 120 + 60 * (b[mask_g] - r[mask_g]) / diff[mask_g]
    
    mask_b = mask & (v == b)
    h[mask_b] = 240 + 60 * (r[mask_b] - g[mask_b]) / diff[mask_b]
    
    h[h < 0] += 360
    
    # Scale to output range
    if src.dtype == np.uint8:
        if full_range:
            h = (h / 2).astype(np.uint8)  # 0-180
        else:
            h = (h / 2).astype(np.uint8)  # 0-180
        s = (s * 255).astype(np.uint8)
        v = (v * 255).astype(np.uint8)
    
    return np.dstack([h, s, v])


def _hsv_to_bgr(src: np.ndarray, to_rgb: bool, full_range: bool) -> np.ndarray:
    """Convert HSV to BGR/RGB."""
    h, s, v = src[:, :, 0].astype(np.float32), src[:, :, 1].astype(np.float32), src[:, :, 2].astype(np.float32)
    
    if src.dtype == np.uint8:
        h = h * 2  # 0-360
        s = s / 255
        v = v / 255
    
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c
    
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    
    mask = (h < 60)
    r[mask], g[mask], b[mask] = c[mask], x[mask], 0
    
    mask = (h >= 60) & (h < 120)
    r[mask], g[mask], b[mask] = x[mask], c[mask], 0
    
    mask = (h >= 120) & (h < 180)
    r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
    
    mask = (h >= 180) & (h < 240)
    r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
    
    mask = (h >= 240) & (h < 300)
    r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
    
    mask = (h >= 300)
    r[mask], g[mask], b[mask] = c[mask], 0, x[mask]
    
    r, g, b = r + m, g + m, b + m
    
    if src.dtype == np.uint8:
        r = (r * 255).clip(0, 255).astype(np.uint8)
        g = (g * 255).clip(0, 255).astype(np.uint8)
        b = (b * 255).clip(0, 255).astype(np.uint8)
    
    if to_rgb:
        return np.dstack([r, g, b])
    return np.dstack([b, g, r])


def _bgr_to_hls(src: np.ndarray, is_rgb: bool) -> np.ndarray:
    """Convert BGR/RGB to HLS."""
    if is_rgb:
        r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    else:
        b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    r, g, b = r.astype(np.float32) / 255, g.astype(np.float32) / 255, b.astype(np.float32) / 255
    
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin
    
    l = (cmax + cmin) / 2
    
    s = np.zeros_like(l)
    mask = diff > 0
    s[mask & (l <= 0.5)] = diff[mask & (l <= 0.5)] / (cmax[mask & (l <= 0.5)] + cmin[mask & (l <= 0.5)])
    s[mask & (l > 0.5)] = diff[mask & (l > 0.5)] / (2 - cmax[mask & (l > 0.5)] - cmin[mask & (l > 0.5)])
    
    h = np.zeros_like(l)
    mask_r = mask & (cmax == r)
    h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / diff[mask_r] % 6)
    
    mask_g = mask & (cmax == g)
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g] + 2)
    
    mask_b = mask & (cmax == b)
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b] + 4)
    
    if src.dtype == np.uint8:
        h = (h / 2).clip(0, 180).astype(np.uint8)
        l = (l * 255).clip(0, 255).astype(np.uint8)
        s = (s * 255).clip(0, 255).astype(np.uint8)
    
    return np.dstack([h, l, s])


def _hls_to_bgr(src: np.ndarray, to_rgb: bool) -> np.ndarray:
    """Convert HLS to BGR/RGB."""
    h, l, s = src[:, :, 0].astype(np.float32), src[:, :, 1].astype(np.float32), src[:, :, 2].astype(np.float32)
    
    if src.dtype == np.uint8:
        h = h * 2
        l = l / 255
        s = s / 255
    
    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = l - c / 2
    
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    
    for i, (lo, hi) in enumerate([(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]):
        mask = (h >= lo) & (h < hi)
        if i == 0:
            r[mask], g[mask], b[mask] = c[mask], x[mask], 0
        elif i == 1:
            r[mask], g[mask], b[mask] = x[mask], c[mask], 0
        elif i == 2:
            r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
        elif i == 3:
            r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
        elif i == 4:
            r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
        else:
            r[mask], g[mask], b[mask] = c[mask], 0, x[mask]
    
    r, g, b = r + m, g + m, b + m
    
    if src.dtype == np.uint8:
        r = (r * 255).clip(0, 255).astype(np.uint8)
        g = (g * 255).clip(0, 255).astype(np.uint8)
        b = (b * 255).clip(0, 255).astype(np.uint8)
    
    if to_rgb:
        return np.dstack([r, g, b])
    return np.dstack([b, g, r])


def _bgr_to_lab(src: np.ndarray, is_rgb: bool) -> np.ndarray:
    """Convert BGR/RGB to CIE Lab."""
    # First convert to XYZ
    xyz = _bgr_to_xyz(src, is_rgb)
    
    # Then XYZ to Lab
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    
    if src.dtype == np.uint8:
        x, y, z = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
    
    # D65 white point
    xn, yn, zn = 95.047, 100.0, 108.883
    
    x, y, z = x / xn, y / yn, z / zn
    
    def f(t):
        delta = 6 / 29
        return np.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4 / 29)
    
    fx, fy, fz = f(x), f(y), f(z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    if src.dtype == np.uint8:
        L = (L * 255 / 100).clip(0, 255).astype(np.uint8)
        a = (a + 128).clip(0, 255).astype(np.uint8)
        b = (b + 128).clip(0, 255).astype(np.uint8)
    
    return np.dstack([L, a, b])


def _lab_to_bgr(src: np.ndarray, to_rgb: bool) -> np.ndarray:
    """Convert CIE Lab to BGR/RGB."""
    L, a, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    if src.dtype == np.uint8:
        L = L.astype(np.float32) * 100 / 255
        a = a.astype(np.float32) - 128
        b = b.astype(np.float32) - 128
    
    # D65 white point
    xn, yn, zn = 95.047, 100.0, 108.883
    
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    delta = 6 / 29
    
    def f_inv(t):
        return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))
    
    x = xn * f_inv(fx)
    y = yn * f_inv(fy)
    z = zn * f_inv(fz)
    
    xyz = np.dstack([x, y, z])
    return _xyz_to_bgr(xyz, to_rgb)


def _bgr_to_ycrcb(src: np.ndarray, is_rgb: bool) -> np.ndarray:
    """Convert BGR/RGB to YCrCb."""
    if is_rgb:
        r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    else:
        b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cr = (r - Y) * 0.713 + 128
    Cb = (b - Y) * 0.564 + 128
    
    if src.dtype == np.uint8:
        Y = Y.clip(0, 255).astype(np.uint8)
        Cr = Cr.clip(0, 255).astype(np.uint8)
        Cb = Cb.clip(0, 255).astype(np.uint8)
    
    return np.dstack([Y, Cr, Cb])


def _ycrcb_to_bgr(src: np.ndarray, to_rgb: bool) -> np.ndarray:
    """Convert YCrCb to BGR/RGB."""
    Y, Cr, Cb = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    Y, Cr, Cb = Y.astype(np.float32), Cr.astype(np.float32) - 128, Cb.astype(np.float32) - 128
    
    r = Y + 1.403 * Cr
    g = Y - 0.344 * Cb - 0.714 * Cr
    b = Y + 1.773 * Cb
    
    if src.dtype == np.uint8:
        r = r.clip(0, 255).astype(np.uint8)
        g = g.clip(0, 255).astype(np.uint8)
        b = b.clip(0, 255).astype(np.uint8)
    
    if to_rgb:
        return np.dstack([r, g, b])
    return np.dstack([b, g, r])


def _bgr_to_yuv(src: np.ndarray, is_rgb: bool) -> np.ndarray:
    """Convert BGR/RGB to YUV."""
    if is_rgb:
        r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    else:
        b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    U = -0.147 * r - 0.289 * g + 0.436 * b + 128
    V = 0.615 * r - 0.515 * g - 0.100 * b + 128
    
    if src.dtype == np.uint8:
        Y = Y.clip(0, 255).astype(np.uint8)
        U = U.clip(0, 255).astype(np.uint8)
        V = V.clip(0, 255).astype(np.uint8)
    
    return np.dstack([Y, U, V])


def _yuv_to_bgr(src: np.ndarray, to_rgb: bool) -> np.ndarray:
    """Convert YUV to BGR/RGB."""
    Y, U, V = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    Y, U, V = Y.astype(np.float32), U.astype(np.float32) - 128, V.astype(np.float32) - 128
    
    r = Y + 1.140 * V
    g = Y - 0.395 * U - 0.581 * V
    b = Y + 2.032 * U
    
    if src.dtype == np.uint8:
        r = r.clip(0, 255).astype(np.uint8)
        g = g.clip(0, 255).astype(np.uint8)
        b = b.clip(0, 255).astype(np.uint8)
    
    if to_rgb:
        return np.dstack([r, g, b])
    return np.dstack([b, g, r])


def _bgr_to_xyz(src: np.ndarray, is_rgb: bool) -> np.ndarray:
    """Convert BGR/RGB to CIE XYZ."""
    if is_rgb:
        r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    else:
        b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    
    if src.dtype == np.uint8:
        r, g, b = r / 255, g / 255, b / 255
    
    # sRGB to linear
    def linearize(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    
    r, g, b = linearize(r), linearize(g), linearize(b)
    
    # Linear RGB to XYZ (sRGB matrix)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    
    # Scale to 0-255 for uint8
    if src.dtype == np.uint8:
        x = (x * 255).clip(0, 255).astype(np.uint8)
        y = (y * 255).clip(0, 255).astype(np.uint8)
        z = (z * 255).clip(0, 255).astype(np.uint8)
    
    return np.dstack([x, y, z])


def _xyz_to_bgr(src: np.ndarray, to_rgb: bool) -> np.ndarray:
    """Convert CIE XYZ to BGR/RGB."""
    x, y, z = src[:, :, 0], src[:, :, 1], src[:, :, 2]
    
    x, y, z = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
    
    if src.dtype == np.uint8:
        x, y, z = x / 255, y / 255, z / 255
    
    # XYZ to linear RGB (inverse sRGB matrix)
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    
    # Linear to sRGB
    def gamma_correct(c):
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1/2.4)) - 0.055)
    
    r, g, b = gamma_correct(r), gamma_correct(g), gamma_correct(b)
    
    if src.dtype == np.uint8:
        r = (r * 255).clip(0, 255).astype(np.uint8)
        g = (g * 255).clip(0, 255).astype(np.uint8)
        b = (b * 255).clip(0, 255).astype(np.uint8)
    
    if to_rgb:
        return np.dstack([r, g, b])
    return np.dstack([b, g, r])


# Exports

__all__ = [
    "cvtColor",
    # BGR <-> RGB
    "COLOR_BGR2BGRA", "COLOR_RGB2RGBA", "COLOR_BGRA2BGR", "COLOR_RGBA2RGB",
    "COLOR_BGR2RGBA", "COLOR_RGB2BGRA", "COLOR_RGBA2BGR", "COLOR_BGRA2RGB",
    "COLOR_BGR2RGB", "COLOR_RGB2BGR", "COLOR_BGRA2RGBA", "COLOR_RGBA2BGRA",
    # BGR <-> Gray
    "COLOR_BGR2GRAY", "COLOR_RGB2GRAY", "COLOR_GRAY2BGR", "COLOR_GRAY2RGB",
    "COLOR_GRAY2BGRA", "COLOR_GRAY2RGBA", "COLOR_BGRA2GRAY", "COLOR_RGBA2GRAY",
    # HSV
    "COLOR_BGR2HSV", "COLOR_RGB2HSV", "COLOR_HSV2BGR", "COLOR_HSV2RGB",
    "COLOR_BGR2HSV_FULL", "COLOR_RGB2HSV_FULL", "COLOR_HSV2BGR_FULL", "COLOR_HSV2RGB_FULL",
    # HLS
    "COLOR_BGR2HLS", "COLOR_RGB2HLS", "COLOR_HLS2BGR", "COLOR_HLS2RGB",
    # Lab
    "COLOR_BGR2Lab", "COLOR_RGB2Lab", "COLOR_Lab2BGR", "COLOR_Lab2RGB",
    # Luv
    "COLOR_BGR2Luv", "COLOR_RGB2Luv", "COLOR_Luv2BGR", "COLOR_Luv2RGB",
    # YCrCb
    "COLOR_BGR2YCrCb", "COLOR_RGB2YCrCb", "COLOR_YCrCb2BGR", "COLOR_YCrCb2RGB",
    # YUV
    "COLOR_BGR2YUV", "COLOR_RGB2YUV", "COLOR_YUV2BGR", "COLOR_YUV2RGB",
    # XYZ
    "COLOR_BGR2XYZ", "COLOR_RGB2XYZ", "COLOR_XYZ2BGR", "COLOR_XYZ2RGB",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.