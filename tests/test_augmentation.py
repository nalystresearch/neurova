#!/usr/bin/env python
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

# Test script for Neurova augmentation module

import numpy as np

print('Testing Neurova Augmentation Module...')
print('=' * 50)

# Test imports
from neurova.augmentation import (
    # Functional
    hflip, vflip, rotate, resize, crop, center_crop, pad,
    normalize, adjust_brightness, adjust_contrast, gaussian_blur,
    to_tensor, to_numpy,
    # Transforms
    Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
    RandomCrop, CenterCrop, Resize, RandomResizedCrop, ColorJitter,
    RandomGrayscale, GaussianBlur, Normalize, ToTensor, RandomErasing,
    # Advanced geometric
    ElasticTransform, GridDistortion, CoarseDropout, MotionBlur,
    # Color
    RGBToHSV, HSVToRGB, ChannelShuffle, CLAHE, RandomBrightnessContrast,
)

print(' All imports successful')

# Create test image (H, W, C)
image_hwc = np.random.rand(224, 224, 3).astype(np.float32)
image_chw = np.transpose(image_hwc, (2, 0, 1))

print(f'\n1. Testing Functional API...')

# Test hflip
flipped = hflip(image_hwc)
print(f'   hflip: {image_hwc.shape} -> {flipped.shape}')

# Test vflip
flipped = vflip(image_chw)
print(f'   vflip: {image_chw.shape} -> {flipped.shape}')

# Test rotate
rotated = rotate(image_chw, 45)
print(f'   rotate(45°): {image_chw.shape} -> {rotated.shape}')

# Test resize
resized = resize(image_chw, (128, 128))
print(f'   resize: {image_chw.shape} -> {resized.shape}')

# Test crop
cropped = crop(image_chw, 10, 10, 100, 100)
print(f'   crop: {image_chw.shape} -> {cropped.shape}')

# Test center_crop
center = center_crop(image_chw, 128)
print(f'   center_crop: {image_chw.shape} -> {center.shape}')

# Test pad
padded = pad(image_chw, 10)
print(f'   pad(10): {image_chw.shape} -> {padded.shape}')

# Test normalize
norm = normalize(image_chw, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
print(f'   normalize: mean={norm.mean():.4f}, std={norm.std():.4f}')

# Test adjust_brightness
bright = adjust_brightness(image_chw, 1.5)
print(f'   adjust_brightness(1.5): max={bright.max():.4f}')

# Test gaussian_blur
blurred = gaussian_blur(image_chw, 5)
print(f'   gaussian_blur: {image_chw.shape} -> {blurred.shape}')

# Test to_tensor / to_numpy
tensor = to_tensor(image_hwc)
back = to_numpy(tensor)
print(f'   to_tensor: (224,224,3) -> {tensor.shape}')
print(f'   to_numpy: {tensor.shape} -> {back.shape}')

print(f'\n2. Testing Transform Classes...')

# Test Resize
transform = Resize((128, 128))
result = transform(image_chw)
print(f'   Resize: {image_chw.shape} -> {result.shape}')

# Test RandomCrop
transform = RandomCrop(100)
result = transform(image_chw)
print(f'   RandomCrop: {image_chw.shape} -> {result.shape}')

# Test CenterCrop
transform = CenterCrop(112)
result = transform(image_chw)
print(f'   CenterCrop: {image_chw.shape} -> {result.shape}')

# Test RandomResizedCrop
transform = RandomResizedCrop(128, scale=(0.5, 1.0))
result = transform(image_chw)
print(f'   RandomResizedCrop: {image_chw.shape} -> {result.shape}')

# Test RandomHorizontalFlip
transform = RandomHorizontalFlip(p=1.0)
result = transform(image_chw)
print(f'   RandomHorizontalFlip: {image_chw.shape} -> {result.shape}')

# Test RandomRotation
transform = RandomRotation(30)
result = transform(image_chw)
print(f'   RandomRotation: {image_chw.shape} -> {result.shape}')

# Test ColorJitter
transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
result = transform(image_chw)
print(f'   ColorJitter: {image_chw.shape} -> {result.shape}')

# Test GaussianBlur
transform = GaussianBlur(5)
result = transform(image_chw)
print(f'   GaussianBlur: {image_chw.shape} -> {result.shape}')

# Test RandomErasing
transform = RandomErasing(p=1.0)
result = transform(image_chw)
print(f'   RandomErasing: {image_chw.shape} -> {result.shape}')

print(f'\n3. Testing Compose Pipeline...')

pipeline = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(15),
    ColorJitter(brightness=0.2, contrast=0.2),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
result = pipeline(image_chw)
print(f'   Pipeline: {image_chw.shape} -> {result.shape}')
print(f'   Result mean: {result.mean():.4f}, std: {result.std():.4f}')

print(f'\n4. Testing Advanced Geometric Transforms...')

# Test ElasticTransform
transform = ElasticTransform(alpha=50, sigma=5, p=1.0)
result = transform(image_chw)
print(f'   ElasticTransform: {image_chw.shape} -> {result.shape}')

# Test CoarseDropout
transform = CoarseDropout(max_holes=4, p=1.0)
result = transform(image_chw)
print(f'   CoarseDropout: {image_chw.shape} -> {result.shape}')

# Test MotionBlur
transform = MotionBlur(kernel_size=9, p=1.0)
result = transform(image_chw)
print(f'   MotionBlur: {image_chw.shape} -> {result.shape}')

print(f'\n5. Testing Color Space Conversions...')

# Test RGBToHSV
rgb_to_hsv = RGBToHSV()
hsv = rgb_to_hsv(image_chw)
print(f'   RGBToHSV: {image_chw.shape} -> {hsv.shape}')

# Test HSVToRGB
hsv_to_rgb = HSVToRGB()
rgb_back = hsv_to_rgb(hsv)
print(f'   HSVToRGB: {hsv.shape} -> {rgb_back.shape}')

# Verify roundtrip
error = np.abs(image_chw - rgb_back).mean()
print(f'   RGB->HSV->RGB roundtrip error: {error:.6f}')

# Test ChannelShuffle
transform = ChannelShuffle(p=1.0)
result = transform(image_chw)
print(f'   ChannelShuffle: {image_chw.shape} -> {result.shape}')

# Test RandomBrightnessContrast
transform = RandomBrightnessContrast(p=1.0)
result = transform(image_chw)
print(f'   RandomBrightnessContrast: {image_chw.shape} -> {result.shape}')

print(f'\n' + '=' * 50)
print(' All augmentation tests passed!')
