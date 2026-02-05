# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Chapter 11: Image Segmentation with Neurova


This chapter covers:
- Thresholding methods (binary, Otsu)
- Connected component labeling
- Watershed segmentation
- Distance transforms
- Region properties measurement

Using Neurova's segmentation implementations!

Author: Neurova Team
"""

import numpy as np
from pathlib import Path

print("")
print("Chapter 11: Image Segmentation")
print("")

import neurova as nv
from neurova import datasets, core

# 11.1 loading test image
print(f"\n11.1 Loading Test Image")

# try to load a sample image from neurova
try:
    rgb_image = datasets.load_sample_image('fruits')
    test_image = core.to_grayscale(rgb_image)
    print(f"    Loaded 'fruits' sample image from Neurova")
except:
# create synthetic test image with distinct regions
    test_image = np.zeros((100, 100), dtype=np.uint8)
# add some circular regions
    y, x = np.ogrid[:100, :100]
    test_image[(x-25)**2 + (y-25)**2 < 15**2] = 200
    test_image[(x-70)**2 + (y-30)**2 < 12**2] = 180
    test_image[(x-50)**2 + (y-70)**2 < 18**2] = 220
# add gradient background
    test_image = test_image + (np.arange(100) * 0.3).astype(np.uint8)[:, np.newaxis]
    print(f"    Using synthetic test image")

print(f"    Test image shape: {test_image.shape}")
print(f"    Value range: [{test_image.min()}, {test_image.max()}]")

# 11.2 binary thresholding with neurova
print(f"\n11.2 Binary Thresholding")

from neurova.segmentation import apply_threshold, otsu_threshold
from neurova.core.constants import ThresholdMethod

# simple binary threshold
threshold_value = 100
used_thresh, binary = apply_threshold(test_image, threshold_value, method=ThresholdMethod.BINARY)

print(f"    Threshold: {used_thresh}")
print(f"    Foreground pixels: {np.sum(binary > 0)}")
print(f"    Background pixels: {np.sum(binary == 0)}")

# inverse binary threshold
_, binary_inv = apply_threshold(test_image, threshold_value, method=ThresholdMethod.BINARY_INV)
print(f"    Inverse: foreground={np.sum(binary_inv > 0)}, background={np.sum(binary_inv == 0)}")

# truncate threshold
_, truncated = apply_threshold(test_image, 150, method=ThresholdMethod.TRUNCATE)
print(f"    Truncate max value: {truncated.max()}")

# 11.3 otsu's thresholding
print(f"\n11.3 Otsu's Automatic Thresholding")

# Compute Otsu's threshold
otsu_thresh = otsu_threshold(test_image)
print(f"    Otsu's optimal threshold: {otsu_thresh:.1f}")

# apply otsu threshold using the threshold function
used_thresh, otsu_binary = apply_threshold(test_image, 0, method=ThresholdMethod.OTSU)
print(f"    Used threshold: {used_thresh:.1f}")
print(f"    Foreground pixels: {np.sum(otsu_binary > 0)}")
print(f"    Background pixels: {np.sum(otsu_binary == 0)}")

# 11.4 connected components
print(f"\n11.4 Connected Components")

from neurova.segmentation import label_connected_components

# label connected components in binary image
labeled = label_connected_components(otsu_binary)
num_components = labeled.max()  # Max label = number of components

print(f"    Found {num_components} connected components")

# analyze each component
for label in range(1, min(num_components + 1, 6)):  # Show first 5
    count = np.sum(labeled == label)
    print(f"      Component {label}: {count} pixels")

# 11.5 distance transform
print(f"\n11.5 Distance Transform")

from neurova.segmentation import distance_transform_edt

# compute euclidean distance transform
binary_mask = (otsu_binary > 0).astype(np.uint8)
distance_map = distance_transform_edt(binary_mask)

print(f"    Distance map shape: {distance_map.shape}")
print(f"    Max distance from edge: {distance_map.max():.2f}")
print(f"    Mean distance: {distance_map[binary_mask > 0].mean():.2f}")

# 11.6 watershed segmentation
print(f"\n11.6 Watershed Segmentation")

from neurova.segmentation import watershed_segmentation

# create markers for watershed
# use peaks in distance transform as markers
markers = np.zeros_like(test_image, dtype=np.int32)
threshold = distance_map.max() * 0.5
markers[distance_map > threshold] = 1

# simple watershed with gradient
# first compute gradient
from neurova.filters import sobel, gradient_magnitude
sobel_x, sobel_y = sobel(test_image)
gradient = gradient_magnitude(sobel_x, sobel_y)

try:
# watershed segmentation
    watershed_labels = watershed_segmentation(gradient.astype(np.uint8), markers)
    print(f"    Watershed labels: {np.unique(watershed_labels)}")
    print(f"    Watershed regions: {len(np.unique(watershed_labels)) - 1}")  # Exclude background
except Exception as e:
    print(f"    Watershed: {type(e).__name__} - using simple labeling instead")
    watershed_labels = labeled

# 11.7 contour finding
print(f"\n11.7 Contour Finding")

from neurova.segmentation import find_contours

# find contours in binary image
contours = find_contours(binary_mask, level=0.5)

print(f"    Found {len(contours)} contours")
for i, contour in enumerate(contours[:3]):  # Show first 3
    print(f"      Contour {i+1}: {len(contour)} points")

# 11.8 region properties
print(f"\n11.8 Region Properties")

from neurova.segmentation import regionprops, label_stats

# get properties of labeled regions
regions = regionprops(labeled)

print(f"    Analyzing {len(regions)} regions")
for i, region in enumerate(regions[:5]):  # Show first 5
    print(f"      Region {i+1}:")
    print(f"        Area: {region.area} pixels")
    print(f"        Centroid: ({region.centroid[0]:.1f}, {region.centroid[1]:.1f})")
    print(f"        Bounding box: {region.bbox}")

# 11.9 k-means color segmentation
print(f"\n11.9 K-Means Segmentation (using ML module)")

from neurova.ml import KMeans

# use grayscale intensities for clustering
pixels = test_image.flatten().reshape(-1, 1).astype(np.float64)

# Fit K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pixels)

# get cluster assignments
labels = kmeans.predict(pixels)
segment_labels = labels.reshape(test_image.shape)

print(f"    K = 3 clusters")
print(f"    Cluster centers: {kmeans.cluster_centers_.flatten()}")

for i in range(3):
    count = np.sum(segment_labels == i)
    print(f"      Cluster {i}: {count} pixels ({count/segment_labels.size*100:.1f}%)")

# 11.10 hierarchical segmentation
print(f"\n11.10 Multi-Scale Segmentation")

# create segmentations at multiple thresholds
thresholds = [50, 100, 150, 200]
print(f"    Multi-threshold segmentation:")

for thresh in thresholds:
    _, seg = apply_threshold(test_image, thresh, method=ThresholdMethod.BINARY)
    fg = np.sum(seg > 0)
    print(f"      Threshold {thresh}: {fg} foreground pixels ({fg/seg.size*100:.1f}%)")

# 11.11 region statistics
print(f"\n11.11 Region Statistics")

# compute statistics for each labeled region
stats = label_stats(labeled)

print(f"    Statistics for {len(stats)} regions:")
for i, (label, stat) in enumerate(list(stats.items())[:5]):
    print(f"      Region {label}:")
    print(f"        Area: {stat['area']} pixels")
    print(f"        Centroid: ({stat['centroid'][0]:.1f}, {stat['centroid'][1]:.1f})")
    print(f"        BBox: {stat['bbox']}")

# 11.12 complete segmentation pipeline
print(f"\n11.12 Complete Segmentation Pipeline")

def segment_image(image, method='otsu', k=3):
    """
    Complete segmentation pipeline.
    
    Args:
        image: Grayscale image
        method: 'otsu', 'kmeans', or threshold value
        k: Number of clusters for k-means
    
    Returns:
        Segmented labels, number of segments
    """
    if method == 'otsu':
        thresh = otsu_threshold(image)
        _, binary = apply_threshold(image, thresh, method=ThresholdMethod.BINARY)
        labels = label_connected_components(binary)
        n = labels.max()
        return labels, n
    
    elif method == 'kmeans':
        pixels = image.flatten().reshape(-1, 1).astype(np.float64)
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(pixels)
        labels = km.predict(pixels).reshape(image.shape)
        return labels, k
    
    else:
# treat as threshold value
        _, binary = apply_threshold(image, float(method), method=ThresholdMethod.BINARY)
        labels = label_connected_components(binary)
        n = labels.max()
        return labels, n

# run pipeline with different methods
methods = ['otsu', 'kmeans', 128]
for method in methods:
    labels, n = segment_image(test_image, method=method)
    print(f"    Method '{method}': {n} regions")

# summary
print("\n" + "=" * 60)
print("Chapter 11 Summary:")
print("   Applied binary thresholding methods")
print("   Used Otsu's automatic thresholding")
print("   Labeled connected components")
print("   Computed distance transforms")
print("   Applied watershed segmentation")
print("   Found contours in binary images")
print("   Measured region properties")
print("   Used K-Means for color segmentation")
print("   Performed multi-scale segmentation")
print("   Computed region statistics")
print("   Built complete segmentation pipeline")
print("")
