# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Data augmentation module for Neurova.

This module provides comprehensive image transformation and augmentation
capabilities similar to torchvision.transforms and albumentations.

Submodules:
    - functional: Low-level transformation functions
    - transforms: Class-based transforms (standard-style)
    - geometric: Advanced geometric transformations
    - color: Color space and photometric transformations

Example:
    >>> from neurova.augmentation import Compose, RandomHorizontalFlip, ColorJitter, ToTensor
    >>> transform = Compose([
    ...     RandomHorizontalFlip(p=0.5),
    ...     ColorJitter(brightness=0.2, contrast=0.2),
    ...     ToTensor(),
    ... ])
    >>> augmented = transform(image)
"""

# Functional API
from .functional import (
    # Geometric transforms
    hflip, vflip, rotate, affine, perspective,
    resize, crop, center_crop, pad, five_crop, ten_crop,
    # Color transforms
    normalize, adjust_brightness, adjust_contrast, adjust_saturation,
    adjust_hue, adjust_gamma, rgb_to_grayscale, grayscale_to_rgb,
    invert, posterize, solarize, autocontrast, equalize,
    # Noise and blur
    gaussian_blur, gaussian_noise, salt_and_pepper_noise,
    # Utility functions
    to_tensor, to_numpy, clamp,
)

# Class-based transforms
from .transforms import (
    # Compose and base
    Compose, RandomApply, RandomChoice, RandomOrder, Lambda,
    # Geometric transforms
    Resize, RandomCrop, CenterCrop, RandomResizedCrop,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
    RandomAffine, RandomPerspective, Pad, FiveCrop, TenCrop,
    # Color transforms
    Normalize, ColorJitter, RandomGrayscale, GaussianBlur,
    RandomInvert, RandomPosterize, RandomSolarize, RandomAutocontrast,
    RandomEqualize, RandomAdjustSharpness,
    # Noise
    GaussianNoise, SaltAndPepperNoise,
    # Conversion
    ToTensor, ToPILImage, ConvertImageDtype,
    # Erasing
    RandomErasing,
)

# Advanced geometric transforms
from .geometric import (
    ElasticTransform, GridDistortion, OpticalDistortion,
    Erosion, Dilation, Opening, Closing,
    PiecewiseAffine, RandomGridShuffle, CoarseDropout,
    MotionBlur, ZoomBlur,
)

# Advanced color transforms
from .color import (
    # Color space conversions
    RGBToHSV, HSVToRGB, RGBToLAB, LABToRGB, RGBToYUV, YUVToRGB,
    # Channel operations
    ChannelShuffle, ChannelDropout, RandomChannelShift,
    # Advanced color adjustments
    CLAHE, RandomToneCurve, FancyPCA, ISONoise,
    RandomBrightnessContrast, RandomGamma, HueSaturationValue,
    RGBShift, ToSepia, Superpixels,
    # Normalization
    ImageNormalize, HistogramMatching, PixelNormalize,
)

__all__ = [
    # Functional API
    'hflip', 'vflip', 'rotate', 'affine', 'perspective',
    'resize', 'crop', 'center_crop', 'pad', 'five_crop', 'ten_crop',
    'normalize', 'adjust_brightness', 'adjust_contrast', 'adjust_saturation',
    'adjust_hue', 'adjust_gamma', 'rgb_to_grayscale', 'grayscale_to_rgb',
    'invert', 'posterize', 'solarize', 'autocontrast', 'equalize',
    'gaussian_blur', 'gaussian_noise', 'salt_and_pepper_noise',
    'to_tensor', 'to_numpy', 'clamp',
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
    # Advanced geometric
    'ElasticTransform', 'GridDistortion', 'OpticalDistortion',
    'Erosion', 'Dilation', 'Opening', 'Closing',
    'PiecewiseAffine', 'RandomGridShuffle', 'CoarseDropout',
    'MotionBlur', 'ZoomBlur',
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