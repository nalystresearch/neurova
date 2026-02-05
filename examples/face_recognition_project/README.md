# Face Recognition Project with Neurova

Complete guide for face detection and recognition using Neurova library.

##  Project Structure

```
face_recognition_project/

 data/                          #  ADD YOUR IMAGES HERE
    train/                     # Training images (70%)
       person_1/              # Create folder for each person
          img_001.jpg
          img_002.jpg
          ...
       person_2/
          img_001.jpg
          ...
       person_N/
   
    test/                      # Test images (15%)
       person_1/
       person_2/
       ...
   
    validation/                # Validation images (15%)
        person_1/
        person_2/
        ...

 models/                        # Trained models saved here
    face_model.pkl

 reports/                       # Training/test reports
    report.json

 01_collect_faces.py           # Step 1: Collect faces from webcam
 02_prepare_dataset.py         # Step 2: Prepare and split dataset
 03_train_model.py             # Step 3: Train face recognition model
 04_evaluate_model.py          # Step 4: Evaluate and generate report
 05_test_webcam.py             # Step 5: Test with live webcam
 pipeline.py                   # Complete pipeline (all steps)
 README.md                     # This file
```

##  Quick Start

### Step 1: Collect Face Images

```bash
python 01_collect_faces.py --name "John" --count 20
```

### Step 2: Prepare Dataset

```bash
python 02_prepare_dataset.py
```

### Step 3: Train Model

```bash
python 03_train_model.py
```

### Step 4: Evaluate Model

```bash
python 04_evaluate_model.py
```

### Step 5: Test with Webcam

```bash
python 05_test_webcam.py
```

### Or Run Complete Pipeline

```bash
python pipeline.py
```

##  Adding Your Own Images

### Option A: Use Webcam (Recommended)

```bash
python 01_collect_faces.py --name "YourName" --count 20
```

### Option B: Manual Upload

1. Create a folder in `data/train/` with person's name
2. Add 10-20 face images (JPG/PNG)
3. Repeat for each person
4. Run `02_prepare_dataset.py` to split into train/test/val

##  Image Requirements

- Clear face visible
- Good lighting
- Different angles (front, slight left/right)
- Recommended: 10-20 images per person
- Format: JPG, PNG
- Size: At least 100x100 pixels

##  Configuration

Edit `config.py` to change:

- Detection method (haar, hog, cnn)
- Recognition method (lbph, eigenface, fisherface)
- Training parameters
