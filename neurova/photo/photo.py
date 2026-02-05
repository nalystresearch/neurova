# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Computational photography functions for Neurova.

Provides HDR imaging, tone mapping, inpainting, and denoising functions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


# Inpainting methods
INPAINT_NS = 0      # Navier-Stokes based method
INPAINT_TELEA = 1   # Telea's method

# Edge-preserving filter types
RECURS_FILTER = 1
NORMCONV_FILTER = 2


def inpaint(
    src: np.ndarray,
    inpaintMask: np.ndarray,
    inpaintRadius: float,
    flags: int = INPAINT_TELEA
) -> np.ndarray:
    """Restore selected region in an image using region neighborhood.
    
    Args:
        src: Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 3-channel image
        inpaintMask: Inpainting mask (8-bit 1-channel image)
        inpaintRadius: Radius of circular neighborhood
        flags: Inpainting method (INPAINT_NS or INPAINT_TELEA)
    
    Returns:
        Output image with same size and type as src
    """
    src = np.asarray(src)
    mask = np.asarray(inpaintMask, dtype=np.uint8)
    
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    result = src.copy()
    h, w = mask.shape
    radius = int(inpaintRadius)
    
    # Get mask coordinates
    mask_coords = np.where(mask > 0)
    
    if len(mask_coords[0]) == 0:
        return result
    
    if flags == INPAINT_TELEA:
        # Fast Marching Method based inpainting
        result = _inpaint_telea(result, mask, radius)
    else:
        # Navier-Stokes based inpainting
        result = _inpaint_ns(result, mask, radius)
    
    return result


def _inpaint_telea(src: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Telea's fast marching method."""
    result = src.copy()
    h, w = mask.shape
    
    # Create distance transform
    dist = np.ones((h, w)) * np.inf
    dist[mask == 0] = 0
    
    # Simple FMM approximation
    for _ in range(radius * 2):
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if mask[y, x] > 0:
                    neighbors = [
                        dist[y-1, x], dist[y+1, x],
                        dist[y, x-1], dist[y, x+1]
                    ]
                    min_dist = min(neighbors)
                    if min_dist < np.inf:
                        dist[y, x] = min(dist[y, x], min_dist + 1)
    
    # Inpaint using weighted average
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x] > 0:
                total_weight = 0
                weighted_sum = np.zeros(src.shape[2] if len(src.shape) == 3 else 1)
                
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] == 0:
                                d = np.sqrt(dy*dy + dx*dx)
                                if d <= radius:
                                    weight = 1.0 / (d + 1e-6)
                                    total_weight += weight
                                    if len(src.shape) == 3:
                                        weighted_sum += weight * src[ny, nx]
                                    else:
                                        weighted_sum += weight * src[ny, nx]
                
                if total_weight > 0:
                    result[y, x] = weighted_sum / total_weight
    
    return result


def _inpaint_ns(src: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Navier-Stokes based inpainting."""
    # Simplified version - use same as Telea for now
    return _inpaint_telea(src, mask, radius)


def fastNlMeansDenoising(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    h: float = 3,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21
) -> np.ndarray:
    """Denoise a grayscale image using Non-local Means Denoising.
    
    Args:
        src: Input 8-bit 1-channel image
        dst: Output image (same size and type as src)
        h: Filter strength (big h removes more noise but removes detail)
        templateWindowSize: Template patch size (should be odd)
        searchWindowSize: Search window size (should be odd)
    
    Returns:
        Denoised image
    """
    src = np.asarray(src, dtype=np.float32)
    
    if len(src.shape) == 3:
        src = np.mean(src, axis=2)
    
    h_sq = h * h
    pad_t = templateWindowSize // 2
    pad_s = searchWindowSize // 2
    
    height, width = src.shape
    result = np.zeros_like(src)
    
    # Pad image
    padded = np.pad(src, pad_s + pad_t, mode='reflect')
    
    for y in range(height):
        for x in range(width):
            py, px = y + pad_s + pad_t, x + pad_s + pad_t
            
            # Get template
            template = padded[py-pad_t:py+pad_t+1, px-pad_t:px+pad_t+1]
            
            total_weight = 0
            weighted_sum = 0
            
            # Search in window
            for sy in range(-pad_s, pad_s + 1, 2):  # Step 2 for speed
                for sx in range(-pad_s, pad_s + 1, 2):
                    cy, cx = py + sy, px + sx
                    
                    # Get comparison patch
                    patch = padded[cy-pad_t:cy+pad_t+1, cx-pad_t:cx+pad_t+1]
                    
                    # Compute weight
                    diff = template - patch
                    dist_sq = np.mean(diff * diff)
                    weight = np.exp(-max(dist_sq - 2 * h_sq, 0) / h_sq)
                    
                    total_weight += weight
                    weighted_sum += weight * padded[cy, cx]
            
            if total_weight > 0:
                result[y, x] = weighted_sum / total_weight
            else:
                result[y, x] = src[y, x]
    
    return result.astype(np.uint8)


def fastNlMeansDenoisingColored(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    h: float = 3,
    hForColorComponents: float = 3,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21
) -> np.ndarray:
    """Denoise a color image using Non-local Means.
    
    Args:
        src: Input 8-bit 3-channel image
        dst: Output image
        h: Filter strength for luminance
        hForColorComponents: Filter strength for color
        templateWindowSize: Template patch size
        searchWindowSize: Search window size
    
    Returns:
        Denoised color image
    """
    src = np.asarray(src)
    
    if len(src.shape) == 2:
        return fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize)
    
    # Convert to LAB-like space (simplified)
    # Just process each channel
    result = np.zeros_like(src)
    
    for c in range(src.shape[2]):
        h_val = h if c == 0 else hForColorComponents
        result[:, :, c] = fastNlMeansDenoising(
            src[:, :, c], None, h_val, templateWindowSize, searchWindowSize)
    
    return result


class Tonemap:
    """Base class for tonemapping algorithms."""
    
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
    
    def process(self, src: np.ndarray) -> np.ndarray:
        """Process HDR image.
        
        Args:
            src: HDR image (32-bit float)
        
        Returns:
            LDR image (32-bit float, [0, 1] range)
        """
        src = np.asarray(src, dtype=np.float32)
        
        # Simple gamma tonemapping
        result = np.power(np.clip(src, 0, None), 1.0 / self.gamma)
        result = result / (result.max() + 1e-10)
        
        return result.astype(np.float32)
    
    def getGamma(self) -> float:
        return self.gamma
    
    def setGamma(self, gamma: float) -> None:
        self.gamma = gamma


class TonemapDrago(Tonemap):
    """Drago tonemapping algorithm."""
    
    def __init__(self, gamma: float = 1.0, saturation: float = 1.0, bias: float = 0.85):
        super().__init__(gamma)
        self.saturation = saturation
        self.bias = bias
    
    def process(self, src: np.ndarray) -> np.ndarray:
        src = np.asarray(src, dtype=np.float32)
        
        # Convert to grayscale luminance
        if len(src.shape) == 3:
            L = 0.299 * src[:, :, 2] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 0]
        else:
            L = src
        
        L_max = L.max()
        L_avg = np.exp(np.mean(np.log(L + 1e-10)))
        
        # Drago operator
        bias = np.log(self.bias) / np.log(0.5)
        L_d = np.log(1 + L / L_avg) / np.log(2 + 8 * np.power(L / L_max, bias))
        
        # Apply to color channels
        if len(src.shape) == 3:
            result = np.zeros_like(src)
            for c in range(3):
                result[:, :, c] = np.power(src[:, :, c] / (L + 1e-10), self.saturation) * L_d
        else:
            result = L_d
        
        # Gamma correction
        result = np.power(np.clip(result, 0, 1), 1.0 / self.gamma)
        
        return result.astype(np.float32)
    
    def getSaturation(self) -> float:
        return self.saturation
    
    def setSaturation(self, saturation: float) -> None:
        self.saturation = saturation
    
    def getBias(self) -> float:
        return self.bias
    
    def setBias(self, bias: float) -> None:
        self.bias = bias


class TonemapReinhard(Tonemap):
    """Reinhard tonemapping algorithm."""
    
    def __init__(
        self,
        gamma: float = 1.0,
        intensity: float = 0.0,
        light_adapt: float = 1.0,
        color_adapt: float = 0.0
    ):
        super().__init__(gamma)
        self.intensity = intensity
        self.light_adapt = light_adapt
        self.color_adapt = color_adapt
    
    def process(self, src: np.ndarray) -> np.ndarray:
        src = np.asarray(src, dtype=np.float32)
        
        # Luminance
        if len(src.shape) == 3:
            L = 0.299 * src[:, :, 2] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 0]
        else:
            L = src
        
        # Key value
        L_log = np.log(L + 1e-10)
        L_avg = np.exp(np.mean(L_log))
        
        key = 0.18 * np.power(4, self.intensity)
        
        # Scale luminance
        L_scaled = key * L / L_avg
        
        # Global operator
        L_white = L_scaled.max()
        L_d = L_scaled * (1 + L_scaled / (L_white * L_white)) / (1 + L_scaled)
        
        # Apply to color
        if len(src.shape) == 3:
            result = np.zeros_like(src)
            for c in range(3):
                # Blend between luminance and channel-based adaptation
                C_adapt = self.color_adapt * src[:, :, c] + (1 - self.color_adapt) * L
                C_adapt = C_adapt + 1e-10
                result[:, :, c] = (src[:, :, c] / C_adapt) * L_d
        else:
            result = L_d
        
        # Gamma
        result = np.power(np.clip(result, 0, 1), 1.0 / self.gamma)
        
        return result.astype(np.float32)
    
    def getIntensity(self) -> float:
        return self.intensity
    
    def setIntensity(self, intensity: float) -> None:
        self.intensity = intensity
    
    def getLightAdaptation(self) -> float:
        return self.light_adapt
    
    def setLightAdaptation(self, value: float) -> None:
        self.light_adapt = value
    
    def getColorAdaptation(self) -> float:
        return self.color_adapt
    
    def setColorAdaptation(self, value: float) -> None:
        self.color_adapt = value


class TonemapMantiuk(Tonemap):
    """Mantiuk tonemapping algorithm."""
    
    def __init__(self, gamma: float = 1.0, scale: float = 0.7, saturation: float = 1.0):
        super().__init__(gamma)
        self.scale = scale
        self.saturation = saturation
    
    def process(self, src: np.ndarray) -> np.ndarray:
        src = np.asarray(src, dtype=np.float32)
        
        # Luminance
        if len(src.shape) == 3:
            L = 0.299 * src[:, :, 2] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 0]
        else:
            L = src
        
        # Contrast domain
        L_log = np.log(L + 1e-10)
        
        # Gradient compression (simplified)
        from scipy import ndimage
        Gx = np.zeros_like(L_log)
        Gy = np.zeros_like(L_log)
        Gx[:, 1:] = L_log[:, 1:] - L_log[:, :-1]
        Gy[1:, :] = L_log[1:, :] - L_log[:-1, :]
        
        # Compress gradients
        G_mag = np.sqrt(Gx**2 + Gy**2 + 1e-10)
        compression = np.power(G_mag, self.scale - 1)
        Gx_c = Gx * compression
        Gy_c = Gy * compression
        
        # Integrate (simplified - Poisson solver approximation)
        L_c = ndimage.uniform_filter(L_log, size=5) * self.scale
        
        # Convert back
        L_d = np.exp(L_c)
        L_d = L_d / L_d.max()
        
        # Apply to color
        if len(src.shape) == 3:
            result = np.zeros_like(src)
            for c in range(3):
                result[:, :, c] = np.power(src[:, :, c] / (L + 1e-10), self.saturation) * L_d
        else:
            result = L_d
        
        # Gamma
        result = np.power(np.clip(result, 0, 1), 1.0 / self.gamma)
        
        return result.astype(np.float32)
    
    def getScale(self) -> float:
        return self.scale
    
    def setScale(self, scale: float) -> None:
        self.scale = scale
    
    def getSaturation(self) -> float:
        return self.saturation
    
    def setSaturation(self, saturation: float) -> None:
        self.saturation = saturation


def createTonemap(gamma: float = 1.0) -> Tonemap:
    """Create simple gamma-based tonemapper."""
    return Tonemap(gamma)


def createTonemapDrago(
    gamma: float = 1.0,
    saturation: float = 1.0,
    bias: float = 0.85
) -> TonemapDrago:
    """Create Drago tonemapper."""
    return TonemapDrago(gamma, saturation, bias)


def createTonemapReinhard(
    gamma: float = 1.0,
    intensity: float = 0.0,
    light_adapt: float = 1.0,
    color_adapt: float = 0.0
) -> TonemapReinhard:
    """Create Reinhard tonemapper."""
    return TonemapReinhard(gamma, intensity, light_adapt, color_adapt)


def createTonemapMantiuk(
    gamma: float = 1.0,
    scale: float = 0.7,
    saturation: float = 1.0
) -> TonemapMantiuk:
    """Create Mantiuk tonemapper."""
    return TonemapMantiuk(gamma, scale, saturation)


class MergeDebevec:
    """Debevec's HDR merge algorithm."""
    
    def process(
        self,
        images: List[np.ndarray],
        times: np.ndarray,
        response: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Merge exposure sequence to HDR.
        
        Args:
            images: List of LDR images
            times: Exposure times
            response: Camera response function (optional)
        
        Returns:
            HDR image
        """
        if len(images) == 0:
            raise ValueError("Empty image list")
        
        images = [np.asarray(img, dtype=np.float32) for img in images]
        times = np.asarray(times, dtype=np.float32)
        
        # Weight function (hat-shaped)
        def weight(z):
            z_min, z_max = 0, 255
            z_mid = (z_min + z_max) / 2
            if z <= z_mid:
                return z - z_min
            else:
                return z_max - z
        
        weights = np.array([weight(z) for z in range(256)])
        
        result = np.zeros_like(images[0])
        total_weight = np.zeros_like(images[0])
        
        for img, t in zip(images, times):
            w = weights[np.clip(img.astype(int), 0, 255)]
            
            # Assume linear response if not provided
            if response is None:
                log_radiance = np.log(img + 1e-10) - np.log(t)
            else:
                log_radiance = response[np.clip(img.astype(int), 0, 255)] - np.log(t)
            
            result += w * log_radiance
            total_weight += w
        
        result = np.exp(result / (total_weight + 1e-10))
        
        return result.astype(np.float32)


class MergeMertens:
    """Mertens exposure fusion algorithm."""
    
    def __init__(
        self,
        contrast_weight: float = 1.0,
        saturation_weight: float = 1.0,
        exposure_weight: float = 0.0
    ):
        self.contrast_weight = contrast_weight
        self.saturation_weight = saturation_weight
        self.exposure_weight = exposure_weight
    
    def process(self, images: List[np.ndarray]) -> np.ndarray:
        """Fuse exposure sequence.
        
        Args:
            images: List of LDR images
        
        Returns:
            Fused image
        """
        if len(images) == 0:
            raise ValueError("Empty image list")
        
        images = [np.asarray(img, dtype=np.float32) / 255.0 for img in images]
        
        n = len(images)
        h, w = images[0].shape[:2]
        
        # Compute weights for each image
        weights = []
        
        for img in images:
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            
            # Contrast (Laplacian)
            Lx = np.abs(gray[:, 2:] + gray[:, :-2] - 2 * gray[:, 1:-1])
            Ly = np.abs(gray[2:, :] + gray[:-2, :] - 2 * gray[1:-1, :])
            
            contrast = np.zeros_like(gray)
            contrast[:, 1:-1] += Lx
            contrast[1:-1, :] += Ly[:, 1:-1] if Ly.shape[1] > 2 else 0
            
            # Saturation
            if len(img.shape) == 3:
                saturation = np.std(img, axis=2)
            else:
                saturation = np.zeros_like(gray)
            
            # Well-exposedness
            exposure = np.exp(-0.5 * ((gray - 0.5) ** 2) / 0.04)
            
            # Combine
            w = (np.power(contrast + 1e-10, self.contrast_weight) *
                 np.power(saturation + 1e-10, self.saturation_weight) *
                 np.power(exposure + 1e-10, self.exposure_weight))
            
            weights.append(w + 1e-10)
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Blend
        if len(images[0].shape) == 3:
            result = np.zeros((h, w, 3), dtype=np.float32)
            for img, w in zip(images, weights):
                for c in range(3):
                    result[:, :, c] += img[:, :, c] * w
        else:
            result = np.zeros((h, w), dtype=np.float32)
            for img, w in zip(images, weights):
                result += img * w
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)


class CalibrateDebevec:
    """Debevec's camera response calibration."""
    
    def __init__(self, samples: int = 70, lambda_smooth: float = 10.0):
        self.samples = samples
        self.lambda_smooth = lambda_smooth
    
    def process(
        self,
        images: List[np.ndarray],
        times: np.ndarray
    ) -> np.ndarray:
        """Recover camera response function.
        
        Args:
            images: List of LDR images
            times: Exposure times
        
        Returns:
            Response curve (256 values)
        """
        # Simplified - return linear response
        return np.log(np.arange(256) + 1)


def createMergeDebevec() -> MergeDebevec:
    """Create Debevec HDR merge algorithm."""
    return MergeDebevec()


def createMergeMertens(
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposure_weight: float = 0.0
) -> MergeMertens:
    """Create Mertens exposure fusion algorithm."""
    return MergeMertens(contrast_weight, saturation_weight, exposure_weight)


def createCalibrateDebevec(
    samples: int = 70,
    lambda_smooth: float = 10.0
) -> CalibrateDebevec:
    """Create Debevec camera calibration."""
    return CalibrateDebevec(samples, lambda_smooth)


def edgePreservingFilter(
    src: np.ndarray,
    flags: int = RECURS_FILTER,
    sigma_s: float = 60,
    sigma_r: float = 0.4
) -> np.ndarray:
    """Smooth image while preserving edges.
    
    Args:
        src: Input 8-bit 3-channel image
        flags: Edge-preserving filter type
        sigma_s: Spatial sigma
        sigma_r: Range sigma
    
    Returns:
        Filtered image
    """
    src = np.asarray(src, dtype=np.float32)
    
    if len(src.shape) == 2:
        src = np.stack([src, src, src], axis=2)
    
    h, w, c = src.shape
    result = src.copy()
    
    # Simplified recursive filter
    for _ in range(3):
        # Horizontal pass
        for y in range(h):
            for x in range(1, w):
                diff = np.linalg.norm(result[y, x] - result[y, x-1])
                alpha = np.exp(-np.sqrt(2) / sigma_s) * np.exp(-diff / (sigma_r * 255))
                result[y, x] = alpha * result[y, x-1] + (1 - alpha) * result[y, x]
        
        # Reverse horizontal
        for y in range(h):
            for x in range(w - 2, -1, -1):
                diff = np.linalg.norm(result[y, x] - result[y, x+1])
                alpha = np.exp(-np.sqrt(2) / sigma_s) * np.exp(-diff / (sigma_r * 255))
                result[y, x] = alpha * result[y, x+1] + (1 - alpha) * result[y, x]
        
        # Vertical pass
        for x in range(w):
            for y in range(1, h):
                diff = np.linalg.norm(result[y, x] - result[y-1, x])
                alpha = np.exp(-np.sqrt(2) / sigma_s) * np.exp(-diff / (sigma_r * 255))
                result[y, x] = alpha * result[y-1, x] + (1 - alpha) * result[y, x]
    
    return result.astype(np.uint8)


def detailEnhance(
    src: np.ndarray,
    sigma_s: float = 10,
    sigma_r: float = 0.15
) -> np.ndarray:
    """Enhance image details.
    
    Args:
        src: Input 8-bit 3-channel image
        sigma_s: Spatial sigma
        sigma_r: Range sigma
    
    Returns:
        Enhanced image
    """
    src = np.asarray(src, dtype=np.float32)
    
    # Get base layer
    base = edgePreservingFilter(src, RECURS_FILTER, sigma_s, sigma_r).astype(np.float32)
    
    # Detail layer
    detail = src - base
    
    # Enhance details
    result = base + 2.0 * detail
    
    return np.clip(result, 0, 255).astype(np.uint8)


def stylization(
    src: np.ndarray,
    sigma_s: float = 60,
    sigma_r: float = 0.45
) -> np.ndarray:
    """Stylize image (cartoon-like effect).
    
    Args:
        src: Input 8-bit 3-channel image
        sigma_s: Spatial sigma
        sigma_r: Range sigma
    
    Returns:
        Stylized image
    """
    # Edge-preserving filter
    filtered = edgePreservingFilter(src, RECURS_FILTER, sigma_s, sigma_r)
    
    # Quantize colors
    result = (filtered.astype(np.float32) / 32).astype(np.uint8) * 32
    
    return result


def pencilSketch(
    src: np.ndarray,
    sigma_s: float = 60,
    sigma_r: float = 0.07,
    shade_factor: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """Create pencil sketch effect.
    
    Args:
        src: Input 8-bit 3-channel image
        sigma_s: Spatial sigma
        sigma_r: Range sigma
        shade_factor: Shading factor
    
    Returns:
        Tuple of (grayscale_sketch, color_sketch)
    """
    src = np.asarray(src)
    
    if len(src.shape) == 3:
        gray = np.mean(src, axis=2).astype(np.uint8)
    else:
        gray = src.astype(np.uint8)
    
    # Invert and blur
    inverted = 255 - gray
    
    # Gaussian blur
    from scipy import ndimage
    blurred = ndimage.gaussian_filter(inverted.astype(float), sigma=sigma_s / 10)
    
    # Blend
    sketch_gray = gray.astype(float) / (255 - blurred + 1)
    sketch_gray = np.clip(sketch_gray * 255, 0, 255).astype(np.uint8)
    
    # Color sketch
    if len(src.shape) == 3:
        sketch_color = src.copy()
        for c in range(3):
            sketch_color[:, :, c] = (src[:, :, c].astype(float) * 
                                     sketch_gray.astype(float) / 255).astype(np.uint8)
    else:
        sketch_color = sketch_gray
    
    return sketch_gray, sketch_color


__all__ = [
    # Functions
    "inpaint",
    "fastNlMeansDenoising",
    "fastNlMeansDenoisingColored",
    "edgePreservingFilter",
    "detailEnhance",
    "stylization",
    "pencilSketch",
    # Tonemap classes
    "Tonemap",
    "TonemapDrago",
    "TonemapReinhard",
    "TonemapMantiuk",
    # Factory functions
    "createTonemap",
    "createTonemapDrago",
    "createTonemapReinhard",
    "createTonemapMantiuk",
    # HDR merge classes
    "MergeDebevec",
    "MergeMertens",
    "CalibrateDebevec",
    "createMergeDebevec",
    "createMergeMertens",
    "createCalibrateDebevec",
    # Constants
    "INPAINT_NS",
    "INPAINT_TELEA",
    "RECURS_FILTER",
    "NORMCONV_FILTER",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.