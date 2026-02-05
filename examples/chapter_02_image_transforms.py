# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Chapter 2: Image I/O and Transforms


This chapter covers:
- Reading and writing images
- Image transformations (resize, rotate, flip, crop)
- Color space conversions
- Geometric transformations
- Affine and perspective transforms

Author: Neurova Team
"""

import numpy as np
from pathlib import Path

print("")
print("Chapter 2: Image I/O and Transforms")
print("")

import neurova as nv
from neurova import io, transform, core, datasets

# 2.1 reading images
print(f"\n2.1 Reading Images")

# load sample images from neurova datasets
try:
    rgb_image = datasets.load_sample_image('fruits')
    if rgb_image.shape[2] == 4:  # BGRA to BGR
        rgb_image = rgb_image[:, :, :3]
    print(f"    Loaded 'fruits' sample image from Neurova")
except:
# fallback to random image if sample not available
    rgb_image = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
    print(f"    Using random image (sample images not available)")

gray_image = core.to_grayscale(rgb_image) if len(rgb_image.shape) == 3 else rgb_image

print(f"    RGB image: shape={rgb_image.shape}, dtype={rgb_image.dtype}")
print(f"    Grayscale image: shape={gray_image.shape}, dtype={gray_image.dtype}")

# using io module
# image = io.imread('path/to/image.jpg')
# io.imwrite('output.png', image)

# 2.2 color space conversions
print(f"\n2.2 Color Space Conversions")

# rgb to grayscale
gray = core.to_grayscale(rgb_image)
print(f"    RGB -> Grayscale: {rgb_image.shape} -> {gray.shape}")

# rgb to hsv
hsv = core.convert_color_space(rgb_image, core.ColorSpace.HSV, from_space=core.ColorSpace.RGB)
print(f"    RGB -> HSV: {rgb_image.shape} -> {hsv.shape}")

# Grayscale to RGB (expand dimensions)
rgb_from_gray = np.stack([gray, gray, gray], axis=-1)
print(f"    Grayscale -> RGB: {gray.shape} -> {rgb_from_gray.shape}")

# 2.3 resizing images
print(f"\n2.3 Resizing Images")

# resize to specific dimensions
resized_small = transform.resize(rgb_image, (120, 160))
resized_large = transform.resize(rgb_image, (480, 640))
print(f"    Original: {rgb_image.shape}")
print(f"    Resized small: {resized_small.shape}")
print(f"    Resized large: {resized_large.shape}")

# resize with aspect ratio preservation
half_size = transform.resize(rgb_image, (120, 160))
print(f"    Half size: {half_size.shape}")

# 2.4 rotation
print(f"\n2.4 Rotation")

# rotate by angle
rotated_45 = transform.rotate(rgb_image, 45)
rotated_90 = transform.rotate(rgb_image, 90)
rotated_180 = transform.rotate(rgb_image, 180)

print(f"    Rotated 45°: {rotated_45.shape}")
print(f"    Rotated 90°: {rotated_90.shape}")
print(f"    Rotated 180°: {rotated_180.shape}")

# 2.5 flipping
print(f"\n2.5 Flipping")

# Horizontal flip (mirror) - use numpy
flipped_h = np.flip(rgb_image, axis=1)
print(f"    Horizontal flip: {flipped_h.shape}")

# vertical flip
flipped_v = np.flip(rgb_image, axis=0)
print(f"    Vertical flip: {flipped_v.shape}")

# both axes
flipped_both = np.flip(np.flip(rgb_image, axis=0), axis=1)
print(f"    Both axes flip: {flipped_both.shape}")

# 2.6 cropping
print(f"\n2.6 Cropping")

# crop region
cropped = rgb_image[50:150, 80:240]
print(f"    Original: {rgb_image.shape}")
print(f"    Cropped [50:150, 80:240]: {cropped.shape}")

# center crop
h, w = rgb_image.shape[:2]
crop_size = 100
center_crop = rgb_image[
    h//2 - crop_size//2 : h//2 + crop_size//2,
    w//2 - crop_size//2 : w//2 + crop_size//2
]
print(f"    Center crop (100x100): {center_crop.shape}")

# 2.7 padding
print(f"\n2.7 Padding")

# add padding
padded = np.pad(rgb_image, ((20, 20), (20, 20), (0, 0)), mode='constant')
print(f"    Original: {rgb_image.shape}")
print(f"    Padded (20px): {padded.shape}")

# reflect padding
reflect_padded = np.pad(gray_image, ((10, 10), (10, 10)), mode='reflect')
print(f"    Reflect padded: {reflect_padded.shape}")

# 2.8 affine transformations
print(f"\n2.8 Affine Transformations")

# create affine transformation matrix
# translation
tx, ty = 10, 20
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
], dtype=np.float32)
print(f"    Translation matrix:\n{translation_matrix}")

# scaling
sx, sy = 1.5, 1.5
scale_matrix = np.array([
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1]
], dtype=np.float32)
print(f"    Scale matrix:\n{scale_matrix}")

# Rotation (around origin)
angle = np.radians(30)
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
], dtype=np.float32)
print(f"    Rotation matrix (30°):\n{np.round(rotation_matrix, 3)}")

# 2.9 image normalization
print(f"\n2.9 Image Normalization")

# Normalize to [0, 1]
normalized = rgb_image.astype(np.float32) / 255.0
print(f"    Normalized: min={normalized.min():.2f}, max={normalized.max():.2f}")

# Standardize (zero mean, unit variance)
mean = np.mean(rgb_image)
std = np.std(rgb_image)
standardized = (rgb_image.astype(np.float32) - mean) / std
print(f"    Standardized: mean={standardized.mean():.4f}, std={standardized.std():.4f}")

# imagenet normalization
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
imagenet_normalized = (normalized - imagenet_mean) / imagenet_std
print(f"    ImageNet normalized: shape={imagenet_normalized.shape}")

# summary
print("\n" + "=" * 60)
print("Chapter 2 Summary:")
print("   Read and write images with io module")
print("   Convert between color spaces (RGB, HSV, Grayscale)")
print("   Resize, rotate, flip images")
print("   Crop and pad images")
print("   Apply affine transformations")
print("   Normalize images for neural networks")
print("")
