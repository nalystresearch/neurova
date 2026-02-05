# neurova_advanced - Quick Reference Card

## Import

```python
import neurova_advanced as nva
```

## Photo Module

### Inpaint

```python
result = nva.photo.inpaint(src, mask, radius, flags)
# flags: nva.photo.INPAINT_TELEA or nva.photo.INPAINT_NS
```

### Detail Enhance

```python
enhanced = nva.photo.detailEnhance(src, sigma_s=10.0, sigma_r=0.15)
```

## Segmentation Module

### Threshold

```python
# Binary threshold
thresh_val, binary = nva.segmentation.threshold(gray, 127, 255, nva.segmentation.THRESH_BINARY)

# Otsu automatic threshold
thresh_val, binary = nva.segmentation.threshold(gray, 0, 255, nva.segmentation.THRESH_OTSU)

# Compute Otsu value only
t = nva.segmentation.otsu_threshold(gray)
```

### Distance Transform

```python
dist = nva.segmentation.distance_transform_edt(binary)
```

### Watershed

```python
labels = nva.segmentation.watershed(gradient, markers)
```

## Solutions Module

### Point3D

```python
p = nva.solutions.Point3D(x, y, z, confidence=0.9, visible=True)
p2 = p.scaled(2.0, 2.0, 1.0)
p3 = p.offset(5.0, 10.0, 0.0)
dist = p.distance_to(p2)
arr = p.as_array()  # [x, y, z]
```

### BoundingBox

```python
bbox = nva.solutions.BoundingBox(x, y, w, h, score, class_id)
cx, cy = bbox.center()
area = bbox.area()
x, y, w, h = bbox.to_pixels(img_w, img_h)
x1, y1, x2, y2 = bbox.to_xyxy(img_w, img_h)
iou = bbox.iou(other_bbox)
```

### Utilities

```python
# Non-maximum suppression
filtered = nva.solutions.non_max_suppression(boxes, iou_thresh=0.5, score_thresh=0.0)

# Normalize landmarks to [0,1]
normalized = nva.solutions.normalize_landmarks(landmarks, img_w, img_h)

# Denormalize to pixels
denormalized = nva.solutions.denormalize_landmarks(landmarks, img_w, img_h)

# Compute angle at p2 formed by p1-p2-p3 (degrees)
angle = nva.solutions.compute_angle(p1, p2, p3)

# Filter by confidence
filtered = nva.solutions.filter_by_confidence(landmarks, min_conf=0.7)
```

## Constants

### Photo

- `nva.photo.INPAINT_NS = 0`
- `nva.photo.INPAINT_TELEA = 1`

### Segmentation

- `nva.segmentation.THRESH_BINARY = 0`
- `nva.segmentation.THRESH_BINARY_INV = 1`
- `nva.segmentation.THRESH_TRUNCATE = 2`
- `nva.segmentation.THRESH_TO_ZERO = 3`
- `nva.segmentation.THRESH_TO_ZERO_INV = 4`
- `nva.segmentation.THRESH_OTSU = 8`

## Complete Example

```python
import neurova_advanced as nva
import numpy as np

# Photo - Inpaint damaged region
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
mask = np.zeros((480, 640), dtype=np.uint8)
mask[200:280, 300:340] = 255
restored = nva.photo.inpaint(img, mask, 5.0, nva.photo.INPAINT_TELEA)

# Segmentation - Watershed
gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
t, binary = nva.segmentation.threshold(gray, 0, 255, nva.segmentation.THRESH_OTSU)

dist = nva.segmentation.distance_transform_edt(binary)

markers = np.zeros((480, 640), dtype=np.int32)
markers[100:150, 100:150] = 1
markers[300:350, 400:450] = 2
labels = nva.segmentation.watershed(dist, markers)

# Solutions - Object detection post-processing
boxes = [
    nva.solutions.BoundingBox(0.1, 0.1, 0.2, 0.2, 0.95, 0),
    nva.solutions.BoundingBox(0.12, 0.12, 0.2, 0.2, 0.90, 0),  # Overlap
    nva.solutions.BoundingBox(0.5, 0.5, 0.15, 0.15, 0.85, 1),
]
filtered = nva.solutions.non_max_suppression(boxes, iou_threshold=0.5)
print(f"Boxes after NMS: {len(filtered)}")

# Solutions - Landmark processing
landmarks = [
    nva.solutions.Point3D(320, 240, 0, 0.95, True),
    nva.solutions.Point3D(400, 200, 0, 0.60, True),
    nva.solutions.Point3D(240, 300, 0, 0.85, True),
]
high_conf = nva.solutions.filter_by_confidence(landmarks, 0.7)
normalized = nva.solutions.normalize_landmarks(high_conf, 640, 480)
```

## Performance

- **Binary size**: 515 KB
- **SIMD**: ARM NEON enabled
- **Platform**: macOS ARM64
- **All functions**: Fully optimized C++ implementations

## See Also

- [README_ADVANCED.md](README_ADVANCED.md) - Full documentation
- [ADVANCED_SUMMARY.md](ADVANCED_SUMMARY.md) - Complete project summary
- [test_advanced.py](test_advanced.py) - Test suite
