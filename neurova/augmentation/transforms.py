# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Image transformation classes for data augmentation.

This module provides class-based transforms similar to torchvision.transforms,
following PyTorch/torchvision conventions for easy integration.
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Callable, Sequence, Any
from . import functional as F

__all__ = [
    # Compose and base
    'Compose', 'RandomApply', 'RandomChoice', 'RandomOrder', 'Lambda',
    # Geometric transforms
    'Resize', 'RandomCrop', 'CenterCrop', 'RandomResizedCrop',
    'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomRotation',
    'RandomAffine', 'RandomPerspective', 'Pad', 'FiveCrop', 'TenCrop',
    # Color transforms
    'Normalize', 'ColorJitter', 'RandomGrayscale', 'GaussianBlur',
    'RandomInvert', 'RandomPosterize', 'RandomSolarize', 'RandomAutocontrast',
    'RandomEqualize', 'RandomAdjustSharpness',
    # Noise
    'GaussianNoise', 'SaltAndPepperNoise',
    # Conversion
    'ToTensor', 'ToPILImage', 'ConvertImageDtype',
    # Erasing
    'RandomErasing',
]


class Compose:
    """
    Compose multiple transforms together.
    
    Args:
        transforms: List of transforms to compose
        
    Example:
        >>> transform = Compose([
        ...     Resize(256),
        ...     CenterCrop(224),
        ...     ToTensor(),
        ...     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ... ])
    """
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image
    
    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + '(']
        for t in self.transforms:
            lines.append(f'    {t},')
        lines.append(')')
        return '\n'.join(lines)


class RandomApply:
    """
    Apply a list of transforms with a given probability.
    
    Args:
        transforms: List of transforms
        p: Probability of applying the transforms
    """
    
    def __init__(self, transforms: List[Callable], p: float = 0.5):
        self.transforms = transforms
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            for t in self.transforms:
                image = t(image)
        return image


class RandomChoice:
    """
    Apply one randomly chosen transform from a list.
    
    Args:
        transforms: List of transforms to choose from
        p: Probability for each transform (uniform if None)
    """
    
    def __init__(self, transforms: List[Callable], p: Optional[List[float]] = None):
        self.transforms = transforms
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        t = np.random.choice(self.transforms, p=self.p)
        return t(image)


class RandomOrder:
    """
    Apply a list of transforms in random order.
    
    Args:
        transforms: List of transforms
    """
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        order = np.random.permutation(len(self.transforms))
        for i in order:
            image = self.transforms[i](image)
        return image


class Lambda:
    """
    Apply a user-defined function as a transform.
    
    Args:
        lambd: Lambda function to apply
    """
    
    def __init__(self, lambd: Callable):
        self.lambd = lambd
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.lambd(image)


# Conversion Transforms

class ToTensor:
    """
    Convert numpy image to tensor format (C, H, W) with values in [0, 1].
    """
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return F.to_tensor(image)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class ToPILImage:
    """
    Convert tensor to numpy image format (H, W, C) with uint8 values.
    
    Note: Returns numpy array, not actual PIL Image.
    """
    
    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        return F.to_numpy(tensor, scale=True)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class ConvertImageDtype:
    """
    Convert image to specified dtype.
    
    Args:
        dtype: Target numpy dtype
    """
    
    def __init__(self, dtype: np.dtype):
        self.dtype = dtype
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.astype(self.dtype)


# Geometric Transforms

class Resize:
    """
    Resize the image to the given size.
    
    Args:
        size: Desired output size. If int, smaller edge will be matched.
              If tuple (h, w), output will be exactly this size.
        interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic')
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = 'bilinear'
    ):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return F.resize(image, self.size, self.interpolation)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, interpolation={self.interpolation})'


class RandomCrop:
    """
    Randomly crop the image.
    
    Args:
        size: Desired output size (h, w) or single int for square
        padding: Optional padding before cropping
        pad_if_needed: Pad if image is smaller than crop size
        fill: Fill value for padding
        padding_mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]] = None,
        pad_if_needed: bool = False,
        fill: float = 0.0,
        padding_mode: str = 'constant'
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Apply padding if specified
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
        elif image.shape[0] <= 4:  # (C, H, W)
            h, w = image.shape[1], image.shape[2]
        else:  # (H, W, C)
            h, w = image.shape[0], image.shape[1]
        
        crop_h, crop_w = self.size
        
        # Pad if needed
        if self.pad_if_needed:
            if h < crop_h:
                image = F.pad(image, (0, (crop_h - h) // 2, 0, (crop_h - h + 1) // 2),
                             self.fill, self.padding_mode)
                h = crop_h
            if w < crop_w:
                image = F.pad(image, ((crop_w - w) // 2, 0, (crop_w - w + 1) // 2, 0),
                             self.fill, self.padding_mode)
                w = crop_w
        
        # Random crop position
        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))
        
        return F.crop(image, top, left, crop_h, crop_w)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, padding={self.padding})'


class CenterCrop:
    """
    Crop the image at the center.
    
    Args:
        size: Desired output size (h, w) or single int for square
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return F.center_crop(image, self.size)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class RandomResizedCrop:
    """
    Crop to random size and aspect ratio, then resize.
    
    Args:
        size: Expected output size
        scale: Range of crop size (min, max) relative to original
        ratio: Range of aspect ratio (min, max)
        interpolation: Interpolation method
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = 'bilinear'
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
        elif image.shape[0] <= 4:  # (C, H, W)
            h, w = image.shape[1], image.shape[2]
        else:  # (H, W, C)
            h, w = image.shape[0], image.shape[1]
        
        area = h * w
        
        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect_ratio = np.exp(np.random.uniform(*log_ratio))
            
            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if 0 < new_w <= w and 0 < new_h <= h:
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)
                
                image = F.crop(image, top, left, new_h, new_w)
                return F.resize(image, self.size, self.interpolation)
        
        # Fallback: center crop
        in_ratio = w / h
        if in_ratio < min(self.ratio):
            new_w = w
            new_h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            new_h = h
            new_w = int(round(h * max(self.ratio)))
        else:
            new_w, new_h = w, h
        
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        image = F.crop(image, top, left, new_h, new_w)
        return F.resize(image, self.size, self.interpolation)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, scale={self.scale}, ratio={self.ratio})'


class RandomHorizontalFlip:
    """
    Horizontally flip the image with a given probability.
    
    Args:
        p: Probability of flipping (default: 0.5)
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return F.hflip(image)
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomVerticalFlip:
    """
    Vertically flip the image with a given probability.
    
    Args:
        p: Probability of flipping (default: 0.5)
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return F.vflip(image)
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomRotation:
    """
    Rotate the image by a random angle.
    
    Args:
        degrees: Range of degrees to rotate. If float, (-degrees, +degrees).
        center: Center of rotation. If None, use image center.
        fill: Fill value for areas outside the rotated image.
    """
    
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        center: Optional[Tuple[float, float]] = None,
        fill: float = 0.0
    ):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.center = center
        self.fill = fill
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(*self.degrees)
        return F.rotate(image, angle, self.center, self.fill)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(degrees={self.degrees})'


class RandomAffine:
    """
    Apply random affine transformation.
    
    Args:
        degrees: Range of rotation degrees
        translate: Max translation as fraction of image size (tx, ty)
        scale: Range of scale factors (min, max)
        shear: Range of shear angles
        fill: Fill value
    """
    
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        translate: Optional[Tuple[float, float]] = None,
        scale: Optional[Tuple[float, float]] = None,
        shear: Optional[Union[float, Tuple[float, float]]] = None,
        fill: float = 0.0
    ):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.translate = translate
        self.scale = scale
        if isinstance(shear, (int, float)):
            self.shear = (-shear, shear)
        else:
            self.shear = shear
        self.fill = fill
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
        elif image.shape[0] <= 4:  # (C, H, W)
            h, w = image.shape[1], image.shape[2]
        else:  # (H, W, C)
            h, w = image.shape[0], image.shape[1]
        
        # Random parameters
        angle = np.random.uniform(*self.degrees)
        
        if self.translate is not None:
            tx = np.random.uniform(-self.translate[0], self.translate[0]) * w
            ty = np.random.uniform(-self.translate[1], self.translate[1]) * h
        else:
            tx, ty = 0, 0
        
        if self.scale is not None:
            scale = np.random.uniform(*self.scale)
        else:
            scale = 1.0
        
        if self.shear is not None:
            shear = np.random.uniform(*self.shear)
        else:
            shear = 0.0
        
        return F.affine(image, angle, (tx, ty), scale, shear, self.fill)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear})'


class RandomPerspective:
    """
    Apply random perspective transformation.
    
    Args:
        distortion_scale: Distortion strength
        p: Probability of applying the transform
        fill: Fill value
    """
    
    def __init__(
        self,
        distortion_scale: float = 0.5,
        p: float = 0.5,
        fill: float = 0.0
    ):
        self.distortion_scale = distortion_scale
        self.p = p
        self.fill = fill
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
        elif image.shape[0] <= 4:  # (C, H, W)
            h, w = image.shape[1], image.shape[2]
        else:  # (H, W, C)
            h, w = image.shape[0], image.shape[1]
        
        half_h, half_w = h / 2, w / 2
        
        # Original corners
        startpoints = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        
        # Randomly distorted corners
        endpoints = []
        for x, y in startpoints:
            dx = np.random.uniform(-self.distortion_scale, self.distortion_scale) * half_w
            dy = np.random.uniform(-self.distortion_scale, self.distortion_scale) * half_h
            endpoints.append((x + dx, y + dy))
        
        return F.perspective(image, startpoints, endpoints, self.fill)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(distortion_scale={self.distortion_scale}, p={self.p})'


class Pad:
    """
    Pad the image.
    
    Args:
        padding: Padding size
        fill: Fill value for constant padding
        padding_mode: Padding mode
    """
    
    def __init__(
        self,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
        fill: float = 0.0,
        padding_mode: str = 'constant'
    ):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return F.pad(image, self.padding, self.fill, self.padding_mode)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(padding={self.padding}, fill={self.fill})'


class FiveCrop:
    """
    Crop the image into 5 crops: 4 corners and center.
    
    Args:
        size: Desired output size
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        return F.five_crop(image, self.size)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class TenCrop:
    """
    Generate ten crops: 5 crops + 5 flipped versions.
    
    Args:
        size: Desired output size
        vertical_flip: Use vertical flip instead of horizontal
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]], vertical_flip: bool = False):
        self.size = size
        self.vertical_flip = vertical_flip
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        return F.ten_crop(image, self.size, self.vertical_flip)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


# Color Transforms

class Normalize:
    """
    Normalize image with mean and standard deviation.
    
    Args:
        mean: Mean for each channel
        std: Standard deviation for each channel
    """
    
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return F.normalize(image, self.mean, self.std)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class ColorJitter:
    """
    Randomly change brightness, contrast, saturation, and hue.
    
    Args:
        brightness: Brightness factor range (0, brightness) or (min, max)
        contrast: Contrast factor range
        saturation: Saturation factor range
        hue: Hue factor range [-0.5, 0.5]
    """
    
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0
    ):
        self.brightness = self._check_param(brightness, 'brightness')
        self.contrast = self._check_param(contrast, 'contrast')
        self.saturation = self._check_param(saturation, 'saturation')
        self.hue = self._check_param(hue, 'hue', center=0)
    
    @staticmethod
    def _check_param(value, name, center=1):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f'{name} must be non-negative')
            return (max(0, center - value), center + value)
        return value
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        transforms = []
        
        if self.brightness[0] != self.brightness[1]:
            factor = np.random.uniform(*self.brightness)
            transforms.append(lambda img: F.adjust_brightness(img, factor))
        
        if self.contrast[0] != self.contrast[1]:
            factor = np.random.uniform(*self.contrast)
            transforms.append(lambda img: F.adjust_contrast(img, factor))
        
        if self.saturation[0] != self.saturation[1]:
            factor = np.random.uniform(*self.saturation)
            transforms.append(lambda img: F.adjust_saturation(img, factor))
        
        if self.hue[0] != self.hue[1]:
            factor = np.random.uniform(*self.hue)
            transforms.append(lambda img: F.adjust_hue(img, factor))
        
        # Random order
        np.random.shuffle(transforms)
        
        for t in transforms:
            image = t(image)
        
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue})'


class RandomGrayscale:
    """
    Convert image to grayscale with a probability.
    
    Args:
        p: Probability of conversion
    """
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            gray = F.rgb_to_grayscale(image)
            # Convert back to RGB to maintain channel count
            return F.grayscale_to_rgb(gray)
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class GaussianBlur:
    """
    Apply Gaussian blur to the image.
    
    Args:
        kernel_size: Size of Gaussian kernel
        sigma: Range of standard deviation
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        sigma: Tuple[float, float] = (0.1, 2.0)
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        sigma = np.random.uniform(*self.sigma)
        return F.gaussian_blur(image, self.kernel_size, sigma)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})'


class RandomInvert:
    """
    Invert colors of the image with a probability.
    
    Args:
        p: Probability of inversion
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return F.invert(image)
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomPosterize:
    """
    Posterize the image with a probability.
    
    Args:
        bits: Number of bits to keep
        p: Probability of posterization
    """
    
    def __init__(self, bits: int, p: float = 0.5):
        self.bits = bits
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return F.posterize(image, self.bits)
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bits={self.bits}, p={self.p})'


class RandomSolarize:
    """
    Solarize the image with a probability.
    
    Args:
        threshold: Threshold for inversion
        p: Probability of solarization
    """
    
    def __init__(self, threshold: float, p: float = 0.5):
        self.threshold = threshold
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return F.solarize(image, self.threshold)
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(threshold={self.threshold}, p={self.p})'


class RandomAutocontrast:
    """
    Apply autocontrast with a probability.
    
    Args:
        p: Probability of applying autocontrast
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return F.autocontrast(image)
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomEqualize:
    """
    Equalize histogram with a probability.
    
    Args:
        p: Probability of equalizing
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return F.equalize(image)
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomAdjustSharpness:
    """
    Adjust sharpness with a probability.
    
    Args:
        sharpness_factor: Factor to adjust sharpness
        p: Probability of adjusting
    """
    
    def __init__(self, sharpness_factor: float, p: float = 0.5):
        self.sharpness_factor = sharpness_factor
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            # Sharpness is implemented as contrast enhancement
            blurred = F.gaussian_blur(image, 3, 1.0)
            return F.clamp(image + self.sharpness_factor * (image - blurred))
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sharpness_factor={self.sharpness_factor}, p={self.p})'


# Noise Transforms

class GaussianNoise:
    """
    Add Gaussian noise to the image.
    
    Args:
        mean: Mean of the noise
        std: Standard deviation of the noise
    """
    
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return F.gaussian_noise(image, self.mean, self.std)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class SaltAndPepperNoise:
    """
    Add salt and pepper noise to the image.
    
    Args:
        amount: Proportion of pixels to affect
        salt_vs_pepper: Ratio of salt to pepper
    """
    
    def __init__(self, amount: float = 0.05, salt_vs_pepper: float = 0.5):
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return F.salt_and_pepper_noise(image, self.amount, self.salt_vs_pepper)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(amount={self.amount})'


# Erasing

class RandomErasing:
    """
    Randomly erases a rectangular region in an image.
    
    Args:
        p: Probability of erasing
        scale: Range of proportion of image to erase
        ratio: Range of aspect ratio of erased region
        value: Erasing value ('random' for random noise, or float)
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[str, float] = 0
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
            channels = 1
        elif image.shape[0] <= 4:  # (C, H, W)
            channels, h, w = image.shape
            is_chw = True
        else:  # (H, W, C)
            h, w, channels = image.shape
            is_chw = False
        
        area = h * w
        
        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect_ratio = np.exp(np.random.uniform(*log_ratio))
            
            erase_h = int(round(np.sqrt(target_area / aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area * aspect_ratio)))
            
            if erase_h < h and erase_w < w:
                top = np.random.randint(0, h - erase_h)
                left = np.random.randint(0, w - erase_w)
                
                result = image.copy()
                
                if self.value == 'random':
                    noise_shape = (erase_h, erase_w) if image.ndim == 2 else \
                                  (channels, erase_h, erase_w) if is_chw else \
                                  (erase_h, erase_w, channels)
                    noise = np.random.random(noise_shape).astype(image.dtype)
                else:
                    noise = self.value
                
                if image.ndim == 2:
                    result[top:top + erase_h, left:left + erase_w] = noise
                elif is_chw:
                    result[:, top:top + erase_h, left:left + erase_w] = noise
                else:
                    result[top:top + erase_h, left:left + erase_w, :] = noise
                
                return result
        
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p}, scale={self.scale}, ratio={self.ratio})'
