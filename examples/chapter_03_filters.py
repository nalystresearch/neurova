# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Chapter 3: Image Filters and Convolutions
==========================================

This chapter covers:
- Smoothing filters (Gaussian, Mean, Median)
- Sharpening filters
- Edge detection (Sobel, Canny, Laplacian)
- Custom convolution kernels
- Morphological operations

Author: Neurova Team
"""

import numpy as np

print("=" * 60)
print("Chapter 3: Image Filters and Convolutions")
print("=" * 60)

import neurova as nv
from neurova import filters, datasets, core

# load sample image from neurova for filtering demos
try:
    rgb_sample = datasets.load_sample_image('lena')
    if rgb_sample.shape[2] == 4:  # BGRA to BGR
        rgb_sample = rgb_sample[:, :, :3]
    image = core.rgb2gray(rgb_sample).astype(np.uint8)
    print(f"Loaded 'lena' sample image from Neurova")
except:
# fallback to random if sample not available
    np.random.seed(42)
    image = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
    print(f"Using random image (sample images not available)")

# add some noise for denoising demos
noisy_image = image.copy().astype(np.float32)
noisy_image += np.random.normal(0, 25, image.shape)
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

print(f"Sample image: shape={image.shape}, dtype={image.dtype}")
print(f"Noisy image: shape={noisy_image.shape}")

# 3.1 gaussian blur
print(f"\n3.1 Gaussian Blur")

# apply gaussian blur with different kernel sizes
gaussian_3 = filters.gaussian_blur(noisy_image, kernel_size=3)
gaussian_5 = filters.gaussian_blur(noisy_image, kernel_size=5)
gaussian_7 = filters.gaussian_blur(noisy_image, kernel_size=7)

print(f"    Gaussian blur (3x3): shape={gaussian_3.shape}")
print(f"    Gaussian blur (5x5): shape={gaussian_5.shape}")
print(f"    Gaussian blur (7x7): shape={gaussian_7.shape}")

# with sigma parameter
gaussian_sigma = filters.gaussian_blur(noisy_image, kernel_size=5, sigma=1.5)
print(f"    Gaussian blur (sigma=1.5): shape={gaussian_sigma.shape}")

# 3.2 mean (box) filter
print(f"\n3.2 Mean (Box) Filter")

mean_3 = filters.box_blur(noisy_image, kernel_size=3)
mean_5 = filters.box_blur(noisy_image, kernel_size=5)

print(f"    Mean filter (3x3): shape={mean_3.shape}")
print(f"    Mean filter (5x5): shape={mean_5.shape}")

# 3.3 median filter
print(f"\n3.3 Median Filter")

# median filter is great for salt-and-pepper noise
median_3 = filters.median_blur(noisy_image, ksize=3)
median_5 = filters.median_blur(noisy_image, ksize=5)

print(f"    Median filter (3x3): shape={median_3.shape}")
print(f"    Median filter (5x5): shape={median_5.shape}")

# 3.4 bilateral filter
print(f"\n3.4 Bilateral Filter")

# bilateral filter preserves edges while smoothing
bilateral = filters.bilateralFilter(noisy_image, d=9, sigmaColor=75, sigmaSpace=75)
print(f"    Bilateral filter: shape={bilateral.shape}")

# 3.5 sharpening
print(f"\n3.5 Sharpening Filters")

# unsharp masking
sharpened = filters.sharpen(image)
print(f"    Sharpened image: shape={sharpened.shape}")

# custom sharpening kernel
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)
custom_sharp = filters.convolve2d(image.astype(np.float32), sharpen_kernel)
print(f"    Custom sharpen: shape={custom_sharp.shape}")

# 3.6 edge detection - sobel
print(f"\n3.6 Edge Detection - Sobel")

# Sobel edge detection - returns (Gx, Gy) tuple
sobel_x, sobel_y = filters.sobel(image)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

print(f"    Sobel X: min={sobel_x.min():.2f}, max={sobel_x.max():.2f}")
print(f"    Sobel Y: min={sobel_y.min():.2f}, max={sobel_y.max():.2f}")
print(f"    Sobel magnitude: min={sobel_magnitude.min():.2f}, max={sobel_magnitude.max():.2f}")

# 3.7 edge detection - canny
print(f"\n3.7 Edge Detection - Canny")

# canny edge detection with thresholds
canny_edges = filters.canny(image, low_threshold=50, high_threshold=150)
print(f"    Canny edges: shape={canny_edges.shape}")
print(f"    Edge pixels: {np.sum(canny_edges > 0)}")

# 3.8 laplacian
print(f"\n3.8 Laplacian Filter")

laplacian = filters.laplacian(image)
print(f"    Laplacian: shape={laplacian.shape}")
print(f"    Range: [{laplacian.min():.2f}, {laplacian.max():.2f}]")

# 3.9 custom convolution kernels
print(f"\n3.9 Custom Convolution Kernels")

# emboss kernel
emboss_kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
], dtype=np.float32)
embossed = filters.convolve2d(image.astype(np.float32), emboss_kernel)
print(f"    Emboss filter: shape={embossed.shape}")

# edge enhance kernel
edge_enhance = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
], dtype=np.float32)
enhanced = filters.convolve2d(image.astype(np.float32), edge_enhance)
print(f"    Edge enhance: shape={enhanced.shape}")

# identity kernel
identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)
identical = filters.convolve2d(image.astype(np.float32), identity)
print(f"    Identity filter (should be same): diff={np.abs(image - identical).max():.4f}")

# 3.10 morphological operations
print(f"\n3.10 Morphological Operations")

from neurova import morphology
from neurova.core.constants import KernelShape

# create binary image
binary = (image > 128).astype(np.uint8) * 255

# Create structuring element (3x3 rectangular kernel)
kernel = morphology.structuring_element(KernelShape.RECT, 3)

# erosion
eroded = morphology.binary_erode(binary, kernel)
print(f"    Eroded: white pixels {np.sum(binary > 0)} -> {np.sum(eroded > 0)}")

# dilation
dilated = morphology.binary_dilate(binary, kernel)
print(f"    Dilated: white pixels {np.sum(binary > 0)} -> {np.sum(dilated > 0)}")

# Opening (erosion followed by dilation)
opened = morphology.binary_open(binary, kernel)
print(f"    Opening: removes small bright spots")

# Closing (dilation followed by erosion)
closed = morphology.binary_close(binary, kernel)
print(f"    Closing: removes small dark spots")

# Gradient (dilation - erosion)
gradient = morphology.binary_gradient(binary, kernel)
print(f"    Morphological gradient: shape={gradient.shape}")

# 3.11 filter comparison
print(f"\n3.11 Filter Comparison on Noisy Image")

# compare noise reduction methods
def measure_noise(img, original):
    """Calculate noise level (MSE from original)"""
    return np.mean((img.astype(float) - original.astype(float))**2)

original_noise = measure_noise(noisy_image, image)
gaussian_noise = measure_noise(gaussian_5, image)
median_noise = measure_noise(median_5, image)
bilateral_noise = measure_noise(bilateral, image)

print(f"    Original noise (MSE): {original_noise:.2f}")
print(f"    After Gaussian 5x5: {gaussian_noise:.2f} ({100*(1-gaussian_noise/original_noise):.1f}% reduction)")
print(f"    After Median 5x5: {median_noise:.2f} ({100*(1-median_noise/original_noise):.1f}% reduction)")
print(f"    After Bilateral: {bilateral_noise:.2f} ({100*(1-bilateral_noise/original_noise):.1f}% reduction)")

# summary
print("\n" + "=" * 60)
print("Chapter 3 Summary:")
print("   Applied smoothing filters (Gaussian, Mean, Median)")
print("   Used bilateral filter for edge-preserving smoothing")
print("   Applied sharpening filters")
print("   Detected edges with Sobel, Canny, Laplacian")
print("   Created custom convolution kernels")
print("   Applied morphological operations")
print("=" * 60)
