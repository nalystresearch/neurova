<!--
Neurova Library
Copyright (c) 2025 Neurova Team
Licensed under the MIT License
@analytics with harry
-->

# Object Detection Project with Neurova

Complete guide for training and testing custom object detection using the Neurova library.

## Project Structure

```
object_detection_project/
|
+-- data/                              <-- ADD YOUR DATA HERE
|   +-- train/
|   |   +-- images/                    Training images
|   |   |   +-- img_001.jpg
|   |   |   +-- ...
|   |   +-- annotations/               Bounding box annotations
|   |       +-- img_001.json
|   |       +-- ...
|   |
|   +-- test/
|   |   +-- images/
|   |   +-- annotations/
|   |
|   +-- validation/
|       +-- images/
|       +-- annotations/
|
+-- models/                            Trained models saved here
|   +-- detector_model.pkl
|
+-- reports/                           Detection reports
|   +-- evaluation_report.json
|
+-- templates/                         Cascade templates (Haar/HOG)
|
+-- 01_annotate_images.py             Step 1: Annotate bounding boxes
+-- 02_prepare_dataset.py             Step 2: Prepare and split dataset
+-- 03_train_detector.py              Step 3: Train object detector
+-- 04_evaluate_detector.py           Step 4: Evaluate with metrics
+-- 05_test_webcam.py                 Step 5: Real-time detection
+-- pipeline.py                       Complete pipeline
+-- config.py                         Configuration
+-- README.md                         This file
```

## Supported Detection Methods

| Method         | Description                 | Speed  | Accuracy             |
| -------------- | --------------------------- | ------ | -------------------- |
| haar           | Haar Cascade (pre-trained)  | Fast   | Good for faces       |
| hog            | HOG + SVM                   | Medium | Good for pedestrians |
| template       | Template Matching           | Fast   | For specific objects |
| sliding_window | Sliding Window + Classifier | Slow   | Custom objects       |

## Quick Start

### Option A: Use Pre-trained Cascades

```bash
# Detect faces (no training needed)
python 05_test_webcam.py --method haar --target face

# Detect eyes
python 05_test_webcam.py --method haar --target eye

# Detect full body
python 05_test_webcam.py --method haar --target body
```

### Option B: Train Custom Detector

```bash
# Step 1: Annotate your images
python 01_annotate_images.py

# Step 2: Prepare dataset
python 02_prepare_dataset.py

# Step 3: Train detector
python 03_train_detector.py

# Step 4: Evaluate
python 04_evaluate_detector.py

# Step 5: Test with webcam
python 05_test_webcam.py --method custom
```

### Option C: Run Complete Pipeline

```bash
python pipeline.py
```

## Annotation Format

Each image needs a JSON annotation file:

```json
{
  "image": "img_001.jpg",
  "width": 640,
  "height": 480,
  "objects": [
    {
      "class": "cat",
      "bbox": [100, 150, 200, 180]
    },
    {
      "class": "dog",
      "bbox": [300, 100, 150, 200]
    }
  ]
}
```

The bbox format is: [x, y, width, height]

## Adding Your Own Data

### Method 1: Use Annotation Tool

```bash
python 01_annotate_images.py
# Opens GUI to draw bounding boxes
```

### Method 2: Manual Annotation

1. Add images to data/train/images/
2. Create JSON files in data/train/annotations/
3. Run python 02_prepare_dataset.py

## Evaluation Metrics

- Precision: How many detections are correct
- Recall: How many objects were found
- mAP: Mean Average Precision
- IoU: Intersection over Union

## Configuration

Edit config.py to change:

- Detection method
- Confidence threshold
- NMS threshold
- Training parameters

<!--
Neurova Library
Copyright (c) 2025 Neurova Team
Licensed under the MIT License
@analytics with harry
-->
