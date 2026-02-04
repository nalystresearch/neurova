# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Chapter 5: Object Detection with Cascades
==========================================

This chapter covers:
- Haar Cascade detection
- LBP Cascade detection
- HOG + SVM detection
- Multi-scale detection
- Non-maximum suppression

Using Neurova's pure-Python implementations!

Author: Neurova Team
"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("Chapter 5: Object Detection with Cascades")
print("=" * 60)

import neurova as nv
from neurova import detection, datasets, core

# get data directory
DATA_DIR = Path(__file__).parent.parent / "neurova" / "data"
HAARCASCADES_DIR = DATA_DIR / "haarcascades"
LBPCASCADES_DIR = DATA_DIR / "lbpcascades"

# 5.1 available cascade files
print(f"\n5.1 Available Cascade Files")

print(f"\n    Haar Cascades:")
for cascade in sorted(HAARCASCADES_DIR.glob("*.xml")):
    print(f"      - {cascade.name}")

print(f"\n    LBP Cascades:")
for cascade in sorted(LBPCASCADES_DIR.glob("*.xml")):
    print(f"      - {cascade.name}")

# 5.2 haar cascade classifier
print(f"\n5.2 Haar Cascade Classifier")

from neurova.detection.haar_cascade import HaarCascadeClassifier

# load face detector
face_cascade_path = HAARCASCADES_DIR / "haarcascade_frontalface_default.xml"
face_detector = HaarCascadeClassifier(str(face_cascade_path))

print(f"    Loaded: {face_cascade_path.name}")
print(f"    Window size: {face_detector.window_size}")
print(f"    Scale factor: {face_detector.scale_factor}")
print(f"    Min neighbors: {face_detector.min_neighbors}")
print(f"    Min size: {face_detector.min_size}")

# load sample image from neurova for detection demo
try:
    rgb_sample = datasets.load_sample_image('lena')
    if len(rgb_sample.shape) == 3:
        if rgb_sample.shape[2] == 4:  # BGRA to BGR
            rgb_sample = rgb_sample[:, :, :3]
        test_image = core.rgb2gray(rgb_sample).astype(np.uint8)
    else:
        test_image = rgb_sample
    print(f"\n    Using 'lena' sample image from Neurova")
except:
# fallback to synthetic test image
    test_image = np.random.randint(100, 180, (300, 400), dtype=np.uint8)
    test_image[100:200, 150:250] = np.random.randint(180, 220, (100, 100), dtype=np.uint8)
    print(f"\n    Using synthetic test image")

print(f"    Test image shape: {test_image.shape}")

# detect faces
faces = face_detector.detect(test_image)
print(f"    Detected regions: {len(faces)}")
for i, (x, y, w, h, conf) in enumerate(faces[:5]):
    print(f"      Face {i+1}: x={x}, y={y}, w={w}, h={h}, conf={conf:.3f}")

# 5.3 detection parameters
print(f"\n5.3 Detection Parameters")

# create detector with custom parameters
custom_detector = HaarCascadeClassifier(str(face_cascade_path))
custom_detector.scale_factor = 1.2  # Larger scale steps (faster, less accurate)
custom_detector.min_neighbors = 3   # Fewer neighbors required
custom_detector.min_size = (50, 50) # Larger minimum face size

print(f"    Scale factor: {custom_detector.scale_factor}")
print(f"    Min neighbors: {custom_detector.min_neighbors}")
print(f"    Min size: {custom_detector.min_size}")

# detect with custom parameters
custom_faces = custom_detector.detect(test_image)
print(f"    Detected with custom params: {len(custom_faces)}")

# 5.4 eye detection
print(f"\n5.4 Eye Detection")

eye_cascade_path = HAARCASCADES_DIR / "haarcascade_eye.xml"
eye_detector = HaarCascadeClassifier(str(eye_cascade_path))
eye_detector.min_size = (20, 20)

print(f"    Loaded: {eye_cascade_path.name}")

# detect eyes in the whole image
eyes = eye_detector.detect(test_image)
print(f"    Eyes detected: {len(eyes)}")

# 5.5 smile detection
print(f"\n5.5 Smile Detection")

smile_cascade_path = HAARCASCADES_DIR / "haarcascade_smile.xml"
if smile_cascade_path.exists():
    smile_detector = HaarCascadeClassifier(str(smile_cascade_path))
    smile_detector.min_size = (25, 25)
    print(f"    Loaded: {smile_cascade_path.name}")
    
    smiles = smile_detector.detect(test_image)
    print(f"    Smiles detected: {len(smiles)}")

# 5.6 full body detection
print(f"\n5.6 Full Body Detection")

body_cascade_path = HAARCASCADES_DIR / "haarcascade_fullbody.xml"
if body_cascade_path.exists():
    body_detector = HaarCascadeClassifier(str(body_cascade_path))
    print(f"    Loaded: {body_cascade_path.name}")

# 5.7 hog descriptor for detection
print(f"\n5.7 HOG Descriptor")

from neurova.detection.hog import HOGDescriptor

# create hog descriptor for pedestrian detection
hog = HOGDescriptor(
    winSize=(64, 128),
    blockSize=(16, 16),
    blockStride=(8, 8),
    cellSize=(8, 8),
    nbins=9
)

print(f"    Window size: {hog.winSize}")
print(f"    Descriptor size: {hog.getDescriptorSize()}")

# compute hog features for a sample window
sample_window = np.random.randint(0, 255, (128, 64), dtype=np.uint8)
hog_features = hog.compute(sample_window)
if hog_features is not None:
    print(f"    HOG features computed: shape={hog_features.shape}")

# 5.8 multi-scale detection
print(f"\n5.8 Multi-Scale Detection")

def multi_scale_detect(image, detector, scales=[1.0, 0.75, 0.5]):
    """Detect objects at multiple scales."""
    from PIL import Image
    
    all_detections = []
    
    for scale in scales:
# resize image
        new_h = int(image.shape[0] * scale)
        new_w = int(image.shape[1] * scale)
        
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        scaled_image = np.array(pil_img)
        
# detect
        detections = detector.detect(scaled_image)
        
# scale detections back
        for x, y, w, h, conf in detections:
            all_detections.append((
                int(x / scale),
                int(y / scale),
                int(w / scale),
                int(h / scale),
                conf
            ))
    
    return all_detections

multi_detections = multi_scale_detect(test_image, face_detector, scales=[1.0, 0.8, 0.6])
print(f"    Multi-scale detections: {len(multi_detections)}")

# 5.9 non-maximum suppression
print(f"\n5.9 Non-Maximum Suppression (NMS)")

def non_max_suppression(boxes, threshold=0.3):
    """Apply NMS to remove overlapping detections."""
    if not boxes:
        return []
    
# sort by confidence
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        
# remove overlapping boxes
        remaining = []
        for box in boxes:
            if compute_iou(best, box) < threshold:
                remaining.append(box)
        boxes = remaining
    
    return keep

def compute_iou(box1, box2):
    """Compute Intersection over Union."""
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    
    return inter / union if union > 0 else 0

# apply nms
nms_detections = non_max_suppression(multi_detections, threshold=0.3)
print(f"    Before NMS: {len(multi_detections)}")
print(f"    After NMS: {len(nms_detections)}")

# 5.10 drawing detection results
print(f"\n5.10 Drawing Detection Results")

def draw_detections(image, detections, color=200, thickness=2):
    """Draw bounding boxes on image."""
    result = image.copy()
    
    for x, y, w, h, conf in detections:
        # Draw rectangle (simple line drawing)
        # Top
        result[max(0,y):max(0,y)+thickness, x:x+w] = color
# bottom
        result[min(image.shape[0],y+h-thickness):min(image.shape[0],y+h), x:x+w] = color
# left
        result[y:y+h, max(0,x):max(0,x)+thickness] = color
# right
        result[y:y+h, min(image.shape[1],x+w-thickness):min(image.shape[1],x+w)] = color
    
    return result

result_image = draw_detections(test_image, nms_detections[:5])
print(f"    Drew {min(5, len(nms_detections))} detections on image")

# 5.11 detection pipeline
print(f"\n5.11 Complete Detection Pipeline")

def detect_faces_pipeline(image, cascade_path):
    """Complete face detection pipeline."""
    # 1. Load detector
    detector = HaarCascadeClassifier(str(cascade_path))
    detector.min_size = (30, 30)
    
    # 2. Preprocess (histogram equalization)
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_norm = cdf * 255 / cdf[-1]
    processed = cdf_norm[image].astype(np.uint8)
    
    # 3. Multi-scale detection
    detections = multi_scale_detect(processed, detector, scales=[1.0, 0.7, 0.5])
    
    # 4. NMS
    final = non_max_suppression(detections, threshold=0.3)
    
    return final

pipeline_results = detect_faces_pipeline(test_image, face_cascade_path)
print(f"    Pipeline detected: {len(pipeline_results)} faces")

# summary
print("\n" + "=" * 60)
print("Chapter 5 Summary:")
print("   Loaded Haar cascade classifiers")
print("   Configured detection parameters")
print("   Detected faces, eyes, smiles, bodies")
print("   Used HOG descriptor")
print("   Applied multi-scale detection")
print("   Implemented non-maximum suppression")
print("   Built complete detection pipeline")
print("=" * 60)
