# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Test new Neurova features: segmentation, video, detection, neural conv layers."""

from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, '.')

import neurova as nv
from neurova.segmentation import (
    label_connected_components,
    watershed_segmentation,
    find_contours,
    regionprops,
)
from neurova.video import VideoCapture, VideoWriter
from neurova.detection import match_template, non_max_suppression, TemplateDetector
from neurova.neural.conv import Conv2D, MaxPool2D, Flatten
from neurova.neural import save_weights, load_weights

print("=" * 70)
print("NEUROVA ADVANCED FEATURES TEST")
print("=" * 70)

# test 1: Connected Components
print("\nTest 1: Connected Components Labeling")
binary = np.zeros((10, 10), dtype=np.uint8)
binary[1:3, 1:3] = 1  # First component
binary[5:8, 5:8] = 1  # Second component
labels = label_connected_components(binary)
num_components = len(np.unique(labels)) - 1  # Exclude background
print(f"   Found {num_components} connected components (expected: 2)")
assert num_components == 2, f"Expected 2 components, got {num_components}"

# test 2: Region Properties
print("\nTest 2: Region Properties")
props = regionprops(labels)
print(f"   Measured {len(props)} regions")
print(f"   Region 1: area={props[0].area}, centroid={props[0].centroid}")
assert len(props) == 2, f"Expected 2 regions, got {len(props)}"
assert props[0].area == 4 or props[0].area == 9, f"Unexpected area: {props[0].area}"

# test 3: Watershed (simplified test)
print("\nTest 3: Watershed Segmentation")
image = np.zeros((20, 20), dtype=np.uint8)
# create two "peaks" (low values for watersheds)
image[:] = 100
image[5, 5] = 10
image[15, 15] = 10
# create markers
markers = np.zeros_like(image, dtype=np.int32)
markers[5, 5] = 1
markers[15, 15] = 2
result = watershed_segmentation(image, markers=markers)
unique_labels = len(np.unique(result))
print(f"   Watershed produced {unique_labels} unique labels")
assert unique_labels >= 2, f"Expected at least 2 labels, got {unique_labels}"

# test 4: Contours (simplified)
print("\nTest 4: Contour Detection")
binary_square = np.zeros((20, 20), dtype=np.uint8)
binary_square[5:15, 5:15] = 255
contours = find_contours(binary_square, level=128)
print(f"   Found {len(contours)} contours")
if len(contours) > 0:
    print(f"   First contour has {len(contours[0])} points")

# test 5: Template Matching
print("\nTest 5: Template Matching")
image = np.random.rand(50, 50)
template = image[10:20, 10:20].copy()
response = match_template(image, template, method="ncc")
max_response = response.max()
print(f"   Max NCC response: {max_response:.3f} (should be close to 1.0)")
assert max_response > 0.9, f"Template matching failed, max response={max_response}"

# test 6: Non-Maximum Suppression
print("\nTest 6: Non-Maximum Suppression")
boxes = np.array([
    [10, 10, 20, 20],
    [12, 12, 22, 22],  # Overlaps with first
    [50, 50, 60, 60],  # Separate box
])
scores = np.array([0.9, 0.7, 0.8])
keep = non_max_suppression(boxes, scores, iou_threshold=0.3)
print(f"   Kept {len(keep)} boxes out of {len(boxes)} (expected: 2)")
assert len(keep) == 2, f"NMS should keep 2 boxes, kept {len(keep)}"

# test 7: Template Detector
print("\nTest 7: Template Detector")
detector = TemplateDetector(template, threshold=0.8, method="ncc")
detected_boxes, detected_scores = detector.detect(image)
print(f"   Detected {len(detected_boxes)} instances")
assert len(detected_boxes) >= 1, "Should detect at least 1 instance"

# test 8: Conv2D Layer
print("\nTest 8: Conv2D Layer")
conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1, seed=42)
x = nv.neural.tensor(np.random.randn(1, 32, 32, 3))
y = conv(x)
print(f"   Input shape: {x.data.shape}")
print(f"   Output shape: {y.data.shape}")
assert y.data.shape == (1, 32, 32, 16), f"Unexpected output shape: {y.data.shape}"

# test 9: MaxPool2D Layer
print("\nTest 9: MaxPool2D Layer")
pool = MaxPool2D(kernel_size=2, stride=2)
pooled = pool(y)
print(f"   After pooling: {pooled.data.shape}")
assert pooled.data.shape == (1, 16, 16, 16), f"Unexpected pooled shape: {pooled.data.shape}"

# test 10: Flatten Layer
print("\nTest 10: Flatten Layer")
flatten = Flatten()
flat = flatten(pooled)
print(f"   After flatten: {flat.data.shape}")
assert flat.data.ndim == 2, f"Should be 2D, got {flat.data.ndim}D"

# test 11: Save/Load Weights
print("\nTest 11: Model Weight Save/Load")
from neurova.neural.layers import Linear
model = nv.neural.layers.Sequential(
    Linear(10, 20, seed=1),
    nv.neural.layers.ReLU(),
    Linear(20, 5, seed=2),
)
original_params = [p.data.copy() for p in model.parameters()]
save_weights(model, "/tmp/test_model.npz")

# modify weights
for p in model.parameters():
    p.data[:] = np.random.randn(*p.data.shape)

# load back
load_weights(model, "/tmp/test_model.npz")
loaded_params = [p.data for p in model.parameters()]

# verify
for orig, loaded in zip(original_params, loaded_params):
    assert np.allclose(orig, loaded), "Loaded weights don't match saved weights"
print("   Weights saved and loaded correctly")

# test 12: Video (basic test with image sequence)
print("\nTest 12: Video Capture (image sequence)")
import tempfile
import os
temp_dir = tempfile.mkdtemp()
# create dummy frames
for i in range(5):
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    nv.io.write_image(os.path.join(temp_dir, f"frame_{i:04d}.png"), frame)

# test VideoCapture
cap = VideoCapture([
    os.path.join(temp_dir, f"frame_{i:04d}.png") for i in range(5)
])
frame_count = 0
for frame in cap:
    frame_count += 1
    assert frame.shape == (64, 64, 3), f"Unexpected frame shape: {frame.shape}"
print(f"   Read {frame_count} frames from image sequence")
assert frame_count == 5, f"Expected 5 frames, got {frame_count}"

# cleanup
import shutil
shutil.rmtree(temp_dir)

print("\n" + "=" * 70)
print("ALL ADVANCED FEATURES TESTS PASSED")
print("=" * 70)
print("\nSummary of new capabilities:")
print("   Advanced segmentation (watershed, connected components, contours)")
print("   Region measurement (area, perimeter, moments, properties)")
print("   Object detection (template matching, NMS, detectors)")
print("   Convolutional layers (Conv2D, MaxPool2D, Flatten)")
print("   Model save/load (NumPy weights)")
print("   Video processing (image sequences)")
print("\nNeurova now has significantly expanded capabilities!")
