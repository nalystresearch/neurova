# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Advanced color and photometric transformations for image augmentation.

This module provides sophisticated color transformations including
channel operations, color space conversions, and photometric distortions.
"""

import numpy as np
from typing import Tuple, Optional, Union, Sequence

__all__ = [
    # Color space conversions
    'RGBToHSV', 'HSVToRGB', 'RGBToLAB', 'LABToRGB', 'RGBToYUV', 'YUVToRGB',
    # Channel operations
    'ChannelShuffle', 'ChannelDropout', 'RandomChannelShift',
    # Advanced color adjustments
    'CLAHE', 'RandomToneCurve', 'FancyPCA', 'ISONoise',
    'RandomBrightnessContrast', 'RandomGamma', 'HueSaturationValue',
    'RGBShift', 'ToSepia', 'Superpixels',
    # Normalization
    'ImageNormalize', 'HistogramMatching', 'PixelNormalize',
]


# Color Space Conversions

class RGBToHSV:
    """
    Convert RGB image to HSV color space.
    """
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: RGB image of shape (3, H, W) or (H, W, 3) with values in [0, 1]
            
        Returns:
            HSV image with same shape
        """
        if image.ndim == 2:
            return image
        
        is_chw = image.shape[0] == 3
        if is_chw:
            image = np.transpose(image, (1, 2, 0))
        
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c
        
        # Hue
        h = np.zeros_like(r)
        mask = diff > 0
        
        mask_r = mask & (max_c == r)
        mask_g = mask & (max_c == g) & ~mask_r
        mask_b = mask & (max_c == b) & ~mask_r & ~mask_g
        
        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
        h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g] + 2)
        h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b] + 4)
        
        h = h / 360  # Normalize to [0, 1]
        
        # Saturation
        s = np.zeros_like(r)
        s[max_c > 0] = diff[max_c > 0] / max_c[max_c > 0]
        
        # Value
        v = max_c
        
        result = np.stack([h, s, v], axis=-1)
        
        if is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class HSVToRGB:
    """
    Convert HSV image to RGB color space.
    """
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: HSV image of shape (3, H, W) or (H, W, 3) with H in [0, 1]
            
        Returns:
            RGB image with same shape
        """
        if image.ndim == 2:
            return image
        
        is_chw = image.shape[0] == 3
        if is_chw:
            image = np.transpose(image, (1, 2, 0))
        
        h, s, v = image[:, :, 0] * 6, image[:, :, 1], image[:, :, 2]
        
        i = h.astype(int) % 6
        f = h - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        result = np.zeros_like(image)
        
        for idx in range(6):
            mask = i == idx
            if idx == 0:
                result[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1)
            elif idx == 1:
                result[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1)
            elif idx == 2:
                result[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1)
            elif idx == 3:
                result[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1)
            elif idx == 4:
                result[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1)
            elif idx == 5:
                result[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1)
        
        if is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return np.clip(result, 0, 1)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class RGBToLAB:
    """
    Convert RGB image to CIE LAB color space.
    """
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: RGB image of shape (3, H, W) or (H, W, 3) with values in [0, 1]
            
        Returns:
            LAB image (L: 0-100, a: -128-127, b: -128-127)
        """
        if image.ndim == 2:
            return image
        
        is_chw = image.shape[0] == 3
        if is_chw:
            image = np.transpose(image, (1, 2, 0))
        
        # sRGB to linear RGB
        def srgb_to_linear(c):
            return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
        
        rgb_lin = srgb_to_linear(image)
        
        # Linear RGB to XYZ (D65 illuminant)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = np.dot(rgb_lin, M.T)
        
        # Normalize by D65 white point
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 2] /= 1.08883
        
        # XYZ to LAB
        def f(t):
            delta = 6/29
            return np.where(t > delta**3, t**(1/3), t / (3 * delta**2) + 4/29)
        
        fx, fy, fz = f(xyz[:, :, 0]), f(xyz[:, :, 1]), f(xyz[:, :, 2])
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        result = np.stack([L, a, b], axis=-1)
        
        if is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class LABToRGB:
    """
    Convert CIE LAB image to RGB color space.
    """
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: LAB image of shape (3, H, W) or (H, W, 3)
            
        Returns:
            RGB image with values in [0, 1]
        """
        if image.ndim == 2:
            return image
        
        is_chw = image.shape[0] == 3
        if is_chw:
            image = np.transpose(image, (1, 2, 0))
        
        L, a, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # LAB to XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        def f_inv(t):
            delta = 6/29
            return np.where(t > delta, t**3, 3 * delta**2 * (t - 4/29))
        
        x = 0.95047 * f_inv(fx)
        y = 1.0 * f_inv(fy)
        z = 1.08883 * f_inv(fz)
        
        xyz = np.stack([x, y, z], axis=-1)
        
        # XYZ to linear RGB
        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
        
        rgb_lin = np.dot(xyz, M_inv.T)
        
        # Linear RGB to sRGB
        def linear_to_srgb(c):
            return np.where(c > 0.0031308, 1.055 * c**(1/2.4) - 0.055, 12.92 * c)
        
        result = linear_to_srgb(rgb_lin)
        
        if is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return np.clip(result, 0, 1)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class RGBToYUV:
    """
    Convert RGB image to YUV color space.
    """
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        is_chw = image.ndim == 3 and image.shape[0] == 3
        if is_chw:
            image = np.transpose(image, (1, 2, 0))
        
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        
        result = np.stack([y, u, v], axis=-1)
        
        if is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class YUVToRGB:
    """
    Convert YUV image to RGB color space.
    """
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        is_chw = image.ndim == 3 and image.shape[0] == 3
        if is_chw:
            image = np.transpose(image, (1, 2, 0))
        
        y, u, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        
        result = np.stack([r, g, b], axis=-1)
        
        if is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return np.clip(result, 0, 1)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# Channel Operations

class ChannelShuffle:
    """
    Randomly shuffle the channels of an image.
    
    Args:
        p: Probability of applying the transform
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        if image.ndim != 3:
            return image
        
        is_chw = image.shape[0] <= 4
        
        if is_chw:
            indices = np.random.permutation(image.shape[0])
            return image[indices]
        else:
            indices = np.random.permutation(image.shape[2])
            return image[:, :, indices]
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class ChannelDropout:
    """
    Randomly drop (zero out) one or more channels.
    
    Args:
        channel_drop_range: Range for number of channels to drop (min, max)
        fill_value: Value to fill dropped channels with
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        channel_drop_range: Tuple[int, int] = (1, 1),
        fill_value: float = 0,
        p: float = 0.5
    ):
        self.channel_drop_range = channel_drop_range
        self.fill_value = fill_value
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        if image.ndim != 3:
            return image
        
        is_chw = image.shape[0] <= 4
        num_channels = image.shape[0] if is_chw else image.shape[2]
        
        num_drop = np.random.randint(
            min(self.channel_drop_range[0], num_channels - 1),
            min(self.channel_drop_range[1] + 1, num_channels)
        )
        
        channels_to_drop = np.random.choice(num_channels, num_drop, replace=False)
        
        result = image.copy()
        for ch in channels_to_drop:
            if is_chw:
                result[ch] = self.fill_value
            else:
                result[:, :, ch] = self.fill_value
        
        return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(channel_drop_range={self.channel_drop_range})'


class RandomChannelShift:
    """
    Randomly shift channel values.
    
    Args:
        limit: Maximum shift amount
        p: Probability of applying the transform
    """
    
    def __init__(self, limit: float = 0.05, p: float = 0.5):
        self.limit = limit
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        if image.ndim != 3:
            return image
        
        is_chw = image.shape[0] <= 4
        num_channels = image.shape[0] if is_chw else image.shape[2]
        
        shifts = np.random.uniform(-self.limit, self.limit, num_channels)
        
        result = image.copy().astype(np.float32)
        for i, shift in enumerate(shifts):
            if is_chw:
                result[i] += shift
            else:
                result[:, :, i] += shift
        
        return np.clip(result, 0, 1)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(limit={self.limit})'


# Advanced Color Adjustments

class CLAHE:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    
    Args:
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        p: float = 0.5
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p
    
    def _apply_clahe(self, channel: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a single channel."""
        h, w = channel.shape
        tile_h = h // self.tile_grid_size[0]
        tile_w = w // self.tile_grid_size[1]
        
        result = np.zeros_like(channel)
        
        for i in range(self.tile_grid_size[0]):
            for j in range(self.tile_grid_size[1]):
                y1, y2 = i * tile_h, min((i + 1) * tile_h, h)
                x1, x2 = j * tile_w, min((j + 1) * tile_w, w)
                
                tile = channel[y1:y2, x1:x2]
                
                # Compute histogram
                hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 1))
                
                # Clip histogram
                excess = 0
                clip = int(self.clip_limit * tile.size / 256)
                for k in range(256):
                    if hist[k] > clip:
                        excess += hist[k] - clip
                        hist[k] = clip
                
                # Redistribute excess
                excess_per_bin = excess // 256
                for k in range(256):
                    hist[k] += excess_per_bin
                
                # Compute CDF
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]
                
                # Apply equalization
                indices = (tile * 255).astype(int)
                indices = np.clip(indices, 0, 255)
                result[y1:y2, x1:x2] = cdf_normalized[indices]
        
        return result
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        if image.ndim == 2:
            return self._apply_clahe(image)
        
        is_chw = image.shape[0] <= 4
        
        if is_chw:
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = self._apply_clahe(image[i])
            return result
        else:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = self._apply_clahe(image[:, :, i])
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(clip_limit={self.clip_limit})'


class RandomToneCurve:
    """
    Apply random tone curve adjustment.
    
    Args:
        scale: Scale of the curve adjustment
        p: Probability of applying the transform
    """
    
    def __init__(self, scale: float = 0.1, p: float = 0.5):
        self.scale = scale
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Generate random control points for tone curve
        x_points = np.array([0, 0.25, 0.5, 0.75, 1.0])
        y_points = x_points + np.random.uniform(-self.scale, self.scale, 5)
        y_points = np.clip(y_points, 0, 1)
        y_points[0], y_points[-1] = 0, 1  # Keep endpoints fixed
        
        # Create lookup table
        from scipy.interpolate import interp1d
        f = interp1d(x_points, y_points, kind='cubic')
        lut = f(np.linspace(0, 1, 256))
        
        # Apply LUT
        indices = (image * 255).astype(int)
        indices = np.clip(indices, 0, 255)
        
        return lut[indices].astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(scale={self.scale})'


class FancyPCA:
    """
    Apply Fancy PCA augmentation (AlexNet-style color augmentation).
    
    Adds multiples of principal components of RGB pixel values.
    
    Args:
        alpha: Standard deviation of the random values
        p: Probability of applying the transform
    """
    
    def __init__(self, alpha: float = 0.1, p: float = 0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        if image.ndim != 3:
            return image
        
        is_chw = image.shape[0] == 3
        if is_chw:
            image = np.transpose(image, (1, 2, 0))
        
        # Flatten image
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        # Compute covariance matrix
        mean = pixels.mean(axis=0)
        pixels_centered = pixels - mean
        cov = np.cov(pixels_centered.T)
        
        # PCA
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Random weights
        alpha_rgb = np.random.normal(0, self.alpha, 3)
        delta = eigenvectors @ (alpha_rgb * eigenvalues)
        
        # Apply augmentation
        result = image + delta
        
        if is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'


class ISONoise:
    """
    Apply camera sensor noise (shot noise + read noise).
    
    Args:
        color_shift: Range for color channel shift
        intensity: Range for noise intensity
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        color_shift: Tuple[float, float] = (0.01, 0.05),
        intensity: Tuple[float, float] = (0.1, 0.5),
        p: float = 0.5
    ):
        self.color_shift = color_shift
        self.intensity = intensity
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        intensity = np.random.uniform(*self.intensity)
        color_shift = np.random.uniform(*self.color_shift)
        
        # Poisson noise (shot noise)
        result = image.copy()
        if intensity > 0:
            # Scale to simulate photon count
            scale = 1 / intensity
            noisy = np.random.poisson(image * scale) / scale
            result = noisy
        
        # Gaussian noise (read noise) per channel
        if image.ndim == 3:
            is_chw = image.shape[0] <= 4
            num_channels = image.shape[0] if is_chw else image.shape[2]
            
            for i in range(num_channels):
                shift = np.random.uniform(-color_shift, color_shift)
                if is_chw:
                    result[i] += shift + np.random.normal(0, intensity * 0.1, result[i].shape)
                else:
                    result[:, :, i] += shift + np.random.normal(0, intensity * 0.1, result[:, :, i].shape)
        else:
            result += np.random.normal(0, intensity * 0.1, result.shape)
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(intensity={self.intensity})'


class RandomBrightnessContrast:
    """
    Randomly change brightness and contrast.
    
    Args:
        brightness_limit: Range for brightness adjustment
        contrast_limit: Range for contrast adjustment
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        brightness_limit: Union[float, Tuple[float, float]] = 0.2,
        contrast_limit: Union[float, Tuple[float, float]] = 0.2,
        p: float = 0.5
    ):
        if isinstance(brightness_limit, float):
            self.brightness_limit = (-brightness_limit, brightness_limit)
        else:
            self.brightness_limit = brightness_limit
        if isinstance(contrast_limit, float):
            self.contrast_limit = (-contrast_limit, contrast_limit)
        else:
            self.contrast_limit = contrast_limit
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        brightness = np.random.uniform(*self.brightness_limit)
        contrast = np.random.uniform(*self.contrast_limit)
        
        # Apply contrast (around mean)
        mean = image.mean()
        result = (image - mean) * (1 + contrast) + mean
        
        # Apply brightness
        result = result + brightness
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(brightness={self.brightness_limit}, contrast={self.contrast_limit})'


class RandomGamma:
    """
    Apply random gamma correction.
    
    Args:
        gamma_limit: Range for gamma values
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        gamma_limit: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5
    ):
        self.gamma_limit = gamma_limit
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        gamma = np.random.uniform(*self.gamma_limit)
        return np.power(np.clip(image, 0, 1), gamma).astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(gamma_limit={self.gamma_limit})'


class HueSaturationValue:
    """
    Randomly change hue, saturation, and value.
    
    Args:
        hue_shift_limit: Range for hue shift
        sat_shift_limit: Range for saturation shift
        val_shift_limit: Range for value shift
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        hue_shift_limit: float = 0.1,
        sat_shift_limit: float = 0.3,
        val_shift_limit: float = 0.2,
        p: float = 0.5
    ):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        if image.ndim != 3:
            return image
        
        hue_shift = np.random.uniform(-self.hue_shift_limit, self.hue_shift_limit)
        sat_shift = np.random.uniform(-self.sat_shift_limit, self.sat_shift_limit)
        val_shift = np.random.uniform(-self.val_shift_limit, self.val_shift_limit)
        
        # Convert to HSV
        hsv_converter = RGBToHSV()
        rgb_converter = HSVToRGB()
        
        hsv = hsv_converter(image)
        
        is_chw = hsv.shape[0] == 3
        if is_chw:
            hsv[0] = (hsv[0] + hue_shift) % 1.0
            hsv[1] = np.clip(hsv[1] + sat_shift, 0, 1)
            hsv[2] = np.clip(hsv[2] + val_shift, 0, 1)
        else:
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 1.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 1)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift, 0, 1)
        
        return rgb_converter(hsv)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hue={self.hue_shift_limit}, sat={self.sat_shift_limit}, val={self.val_shift_limit})'


class RGBShift:
    """
    Randomly shift R, G, B channels.
    
    Args:
        r_shift_limit: Range for red channel shift
        g_shift_limit: Range for green channel shift
        b_shift_limit: Range for blue channel shift
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        r_shift_limit: float = 0.2,
        g_shift_limit: float = 0.2,
        b_shift_limit: float = 0.2,
        p: float = 0.5
    ):
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        if image.ndim != 3:
            return image
        
        r_shift = np.random.uniform(-self.r_shift_limit, self.r_shift_limit)
        g_shift = np.random.uniform(-self.g_shift_limit, self.g_shift_limit)
        b_shift = np.random.uniform(-self.b_shift_limit, self.b_shift_limit)
        
        result = image.copy()
        is_chw = image.shape[0] == 3
        
        if is_chw:
            result[0] = np.clip(result[0] + r_shift, 0, 1)
            result[1] = np.clip(result[1] + g_shift, 0, 1)
            result[2] = np.clip(result[2] + b_shift, 0, 1)
        else:
            result[:, :, 0] = np.clip(result[:, :, 0] + r_shift, 0, 1)
            result[:, :, 1] = np.clip(result[:, :, 1] + g_shift, 0, 1)
            result[:, :, 2] = np.clip(result[:, :, 2] + b_shift, 0, 1)
        
        return result.astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r_shift_limit}, g={self.g_shift_limit}, b={self.b_shift_limit})'


class ToSepia:
    """
    Apply sepia tone effect.
    
    Args:
        p: Probability of applying the transform
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        if image.ndim != 3:
            return image
        
        is_chw = image.shape[0] == 3
        if is_chw:
            image = np.transpose(image, (1, 2, 0))
        
        # Sepia transformation matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        result = np.dot(image, sepia_matrix.T)
        
        if is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class Superpixels:
    """
    Apply superpixel segmentation effect.
    
    Replaces each superpixel with its average color.
    
    Args:
        p_replace: Probability of replacing each superpixel
        n_segments: Number of superpixels
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        p_replace: float = 0.5,
        n_segments: int = 100,
        p: float = 0.5
    ):
        self.p_replace = p_replace
        self.n_segments = n_segments
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Simple grid-based superpixel segmentation
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        elif image.shape[0] <= 4:
            c, h, w = image.shape
            is_2d = False
            is_chw = True
            image = np.transpose(image, (1, 2, 0))
        else:
            h, w, c = image.shape
            is_2d = False
            is_chw = False
        
        # Calculate grid size
        n_rows = int(np.sqrt(self.n_segments * h / w))
        n_cols = int(self.n_segments / n_rows)
        
        cell_h = max(1, h // n_rows)
        cell_w = max(1, w // n_cols)
        
        result = image.copy()
        
        for i in range(n_rows):
            for j in range(n_cols):
                if np.random.random() < self.p_replace:
                    y1, y2 = i * cell_h, min((i + 1) * cell_h, h)
                    x1, x2 = j * cell_w, min((j + 1) * cell_w, w)
                    
                    if is_2d:
                        mean_val = result[y1:y2, x1:x2].mean()
                        result[y1:y2, x1:x2] = mean_val
                    else:
                        for ch in range(result.shape[2]):
                            mean_val = result[y1:y2, x1:x2, ch].mean()
                            result[y1:y2, x1:x2, ch] = mean_val
        
        if not is_2d and is_chw:
            result = np.transpose(result, (2, 0, 1))
        
        return result.astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n_segments={self.n_segments})'


# Normalization

class ImageNormalize:
    """
    Normalize image to zero mean and unit variance.
    
    Args:
        mean: Mean for each channel (if None, computed from image)
        std: Standard deviation for each channel (if None, computed from image)
    """
    
    def __init__(
        self,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None
    ):
        self.mean = np.array(mean) if mean is not None else None
        self.std = np.array(std) if std is not None else None
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.mean is None:
            mean = image.mean()
        else:
            mean = self.mean
        
        if self.std is None:
            std = image.std()
        else:
            std = self.std
        
        return ((image - mean) / (std + 1e-8)).astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class HistogramMatching:
    """
    Match image histogram to a reference histogram.
    
    Args:
        reference: Reference image or histogram to match
    """
    
    def __init__(self, reference: np.ndarray):
        self.reference = reference
    
    def _match_histogram(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram of source to reference."""
        # Compute CDFs
        src_values, src_counts = np.unique(
            (source * 255).astype(int).flatten(),
            return_counts=True
        )
        ref_values, ref_counts = np.unique(
            (reference * 255).astype(int).flatten(),
            return_counts=True
        )
        
        src_cdf = np.cumsum(src_counts).astype(float) / src_counts.sum()
        ref_cdf = np.cumsum(ref_counts).astype(float) / ref_counts.sum()
        
        # Build mapping
        mapping = np.zeros(256)
        for i, v in enumerate(src_values):
            idx = np.searchsorted(ref_cdf, src_cdf[i])
            idx = min(idx, len(ref_values) - 1)
            mapping[v] = ref_values[idx]
        
        # Apply mapping
        result = mapping[(source * 255).astype(int)]
        return result / 255
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return self._match_histogram(image, self.reference)
        
        is_chw = image.shape[0] <= 4
        if is_chw:
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = self._match_histogram(
                    image[i],
                    self.reference[i] if self.reference.ndim > 2 else self.reference
                )
            return result
        else:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                ref = self.reference[:, :, i] if self.reference.ndim > 2 else self.reference
                result[:, :, i] = self._match_histogram(image[:, :, i], ref)
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class PixelNormalize:
    """
    Normalize each pixel independently.
    
    Args:
        range_min: Minimum of output range
        range_max: Maximum of output range
    """
    
    def __init__(self, range_min: float = 0.0, range_max: float = 1.0):
        self.range_min = range_min
        self.range_max = range_max
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        min_val = image.min()
        max_val = image.max()
        
        if max_val - min_val < 1e-8:
            return np.full_like(image, (self.range_min + self.range_max) / 2)
        
        normalized = (image - min_val) / (max_val - min_val)
        return (normalized * (self.range_max - self.range_min) + self.range_min).astype(np.float32)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(range=[{self.range_min}, {self.range_max}])'
