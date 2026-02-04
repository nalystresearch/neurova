# Neurova Model Assets Directory

This directory contains model weights, configuration files, and shader assets for Neurova's deep learning and GPU-accelerated operations.

## Directory Structure

```
models/
├── classification/     # Image classification models
├── detection/          # Object detection models
├── segmentation/       # Semantic/instance segmentation
├── face/               # Face detection and recognition
├── text/               # OCR and text detection
├── pose/               # Human pose estimation
├── depth/              # Depth estimation models
└── shaders/            # GPU shader programs
```

## Supported Formats

### Model Formats

- `.onnx` - ONNX format (primary)
- `.pb` - Protobuf format
- `.pth` - Model checkpoints
- `.caffemodel` / `.prototxt` - Caffe format
- `.xml` / `.bin` - OpenVINO IR format
- `.tflite` - TFLite format
- `.ncnn.bin` / `.ncnn.param` - NCNN format
- `.engine` - TensorRT serialized engines

### Shader Formats

- `.spv` - SPIR-V compiled shaders
- `.ptx` - CUDA PTX assembly
- `.metallib` - Metal compiled shaders

## Pre-trained Models

### Classification

| Model           | Input Size | Parameters | Top-1 Acc |
| --------------- | ---------- | ---------- | --------- |
| mobilenet_v2    | 224x224    | 3.4M       | 71.8%     |
| resnet18        | 224x224    | 11.7M      | 69.8%     |
| efficientnet_b0 | 224x224    | 5.3M       | 77.1%     |

### Detection

| Model            | Input Size | mAP  | Speed (ms) |
| ---------------- | ---------- | ---- | ---------- |
| yolov8n          | 640x640    | 37.3 | 4.2        |
| ssd_mobilenet_v2 | 300x300    | 22.0 | 5.1        |
| nanodet          | 320x320    | 23.5 | 3.8        |

### Face Detection

| Model      | Input Size | mAP  | Speed (ms) |
| ---------- | ---------- | ---- | ---------- |
| retinaface | 640x640    | 94.2 | 8.3        |
| blazeface  | 128x128    | 89.5 | 1.2        |
| yunet      | 320x320    | 92.1 | 3.5        |

## Model Configuration

Each model has an associated YAML configuration:

```yaml
name: mobilenet_v2
version: 1.0
framework: onnx
input:
  name: input
  shape: [1, 3, 224, 224]
  dtype: float32
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  format: NCHW
output:
  - name: output
    shape: [1, 1000]
    dtype: float32
    post_process: softmax
labels: imagenet_labels.txt
```

## Downloading Models

Models are downloaded on first use:

```python
import neurova as nv

# Auto-download model
model = nv.dnn.readNet("mobilenet_v2")

# Or specify format
model = nv.dnn.readNetFromONNX("path/to/model.onnx")
```

## Custom Model Integration

```python
import neurova as nv

# Register custom model
nv.dnn.registerModel(
    name="my_model",
    path="path/to/model.onnx",
    config="path/to/config.yaml"
)
```
