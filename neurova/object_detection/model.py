# Neurova Library
# Copyright (c) 2025 Neurova Team
# Licensed under the MIT License
# @analytics with harry

"""
Object Detection Model Architecture.

Provides a simplified YOLO-like detection architecture using Neurova primitives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from neurova.neural.module import Module, Parameter
from neurova.neural.tensor import Tensor, tensor
from neurova.neural.conv import Conv2D, MaxPool2D
from neurova.neural.layers import Linear, ReLU, Sequential


# Batch Normalization

class BatchNorm2D(Module):
    """Batch Normalization for 2D inputs (images).
    
    Args:
        num_features: Number of channels
        eps: Small constant for numerical stability
        momentum: Momentum for running stats
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Parameter(np.ones(num_features, dtype=np.float32))
        self.beta = Parameter(np.zeros(num_features, dtype=np.float32))
        
        # Running statistics (not trainable)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input (batch, height, width, channels)
            
        Returns:
            Normalized output
        """
        data = x.data
        
        if self.training:
            # Compute batch statistics
            mean = data.mean(axis=(0, 1, 2), keepdims=True)
            var = data.var(axis=(0, 1, 2), keepdims=True)
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.flatten()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.flatten()
        else:
            mean = self.running_mean.reshape(1, 1, 1, -1)
            var = self.running_var.reshape(1, 1, 1, -1)
        
        # Normalize
        x_norm = (data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        gamma = self.gamma.data.reshape(1, 1, 1, -1)
        beta = self.beta.data.reshape(1, 1, 1, -1)
        out = gamma * x_norm + beta
        
        return tensor(out, requires_grad=x.requires_grad)


# Building Blocks

class ConvBlock(Module):
    """Convolution + BatchNorm + Activation block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "leaky_relu",
        seed: int = 0,
    ):
        super().__init__()
        self.conv = Conv2D(
            in_channels, out_channels, kernel_size, stride, padding, seed=seed
        )
        self.bn = BatchNorm2D(out_channels)
        self.activation = activation
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        
        if self.activation == "leaky_relu":
            # Leaky ReLU: max(0.1*x, x)
            data = x.data
            x = tensor(np.where(data > 0, data, 0.1 * data), requires_grad=x.requires_grad)
        elif self.activation == "relu":
            x = x.relu()
        elif self.activation == "silu":
            # SiLU/Swish: x * sigmoid(x)
            x = x * x.sigmoid()
        
        return x


class ResidualBlock(Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels: int, seed: int = 0):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0, seed=seed)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3, padding=1, seed=seed + 1)
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class CSPBlock(Module):
    """Cross Stage Partial block (simplified)."""
    
    def __init__(self, in_channels: int, out_channels: int, n_blocks: int = 1, seed: int = 0):
        super().__init__()
        mid_channels = out_channels // 2
        
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0, seed=seed)
        self.conv2 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0, seed=seed + 1)
        
        self.blocks = [ResidualBlock(mid_channels, seed=seed + 2 + i) for i in range(n_blocks)]
        
        self.conv3 = ConvBlock(mid_channels * 2, out_channels, kernel_size=1, padding=0, seed=seed + 2 + n_blocks)
    
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        for block in self.blocks:
            x1 = block(x1)
        
        # Concatenate along channel axis
        x = tensor(np.concatenate([x1.data, x2.data], axis=-1), requires_grad=x.requires_grad)
        x = self.conv3(x)
        
        return x


# Backbone

class Backbone(Module):
    """
    Feature extraction backbone (simplified CSP-Darknet style).
    
    Args:
        in_channels: Input channels (3 for RGB)
        depth_multiple: Depth multiplier for scaling
        width_multiple: Width multiplier for scaling
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.50,
        seed: int = 0,
    ):
        super().__init__()
        
        # Base channel widths
        base_channels = [64, 128, 256, 512, 1024]
        channels = [max(int(c * width_multiple), 16) for c in base_channels]
        
        # Number of blocks per stage
        base_depths = [1, 2, 3, 1]
        depths = [max(int(d * depth_multiple), 1) for d in base_depths]
        
        # Stem
        self.stem = ConvBlock(in_channels, channels[0], kernel_size=6, stride=2, padding=2, seed=seed)
        
        # Stages
        self.stage1 = Sequential(
            ConvBlock(channels[0], channels[1], kernel_size=3, stride=2, padding=1, seed=seed + 1),
            CSPBlock(channels[1], channels[1], n_blocks=depths[0], seed=seed + 2),
        )
        
        self.stage2 = Sequential(
            ConvBlock(channels[1], channels[2], kernel_size=3, stride=2, padding=1, seed=seed + 10),
            CSPBlock(channels[2], channels[2], n_blocks=depths[1], seed=seed + 11),
        )
        
        self.stage3 = Sequential(
            ConvBlock(channels[2], channels[3], kernel_size=3, stride=2, padding=1, seed=seed + 20),
            CSPBlock(channels[3], channels[3], n_blocks=depths[2], seed=seed + 21),
        )
        
        self.stage4 = Sequential(
            ConvBlock(channels[3], channels[4], kernel_size=3, stride=2, padding=1, seed=seed + 30),
            CSPBlock(channels[4], channels[4], n_blocks=depths[3], seed=seed + 31),
        )
        
        self.out_channels = [channels[2], channels[3], channels[4]]
    
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Returns:
            List of feature maps at 3 scales (P3, P4, P5)
        """
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)  # 1/8
        p4 = self.stage3(p3)  # 1/16
        p5 = self.stage4(p4)  # 1/32
        
        return [p3, p4, p5]


# Feature Pyramid Network

class FeaturePyramid(Module):
    """
    Feature Pyramid Network for multi-scale detection.
    
    Args:
        in_channels: List of input channel sizes from backbone [P3, P4, P5]
        out_channels: Output channel size for all levels
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        seed: int = 0,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv)
        self.lateral_p5 = ConvBlock(in_channels[2], out_channels, kernel_size=1, padding=0, seed=seed)
        self.lateral_p4 = ConvBlock(in_channels[1], out_channels, kernel_size=1, padding=0, seed=seed + 1)
        self.lateral_p3 = ConvBlock(in_channels[0], out_channels, kernel_size=1, padding=0, seed=seed + 2)
        
        # Output convolutions
        self.output_p5 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1, seed=seed + 3)
        self.output_p4 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1, seed=seed + 4)
        self.output_p3 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1, seed=seed + 5)
    
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Forward pass.
        
        Args:
            features: List of [P3, P4, P5] feature maps from backbone
            
        Returns:
            List of refined [P3, P4, P5] feature maps
        """
        p3, p4, p5 = features
        
        # Top-down pathway
        f5 = self.lateral_p5(p5)
        f4 = self.lateral_p4(p4) + self._upsample(f5, p4.data.shape[1:3])
        f3 = self.lateral_p3(p3) + self._upsample(f4, p3.data.shape[1:3])
        
        # Output
        out_p5 = self.output_p5(f5)
        out_p4 = self.output_p4(f4)
        out_p3 = self.output_p3(f3)
        
        return [out_p3, out_p4, out_p5]
    
    def _upsample(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        """Simple nearest-neighbor upsampling."""
        data = x.data
        h, w = target_size
        
        # Get current size
        _, curr_h, curr_w, c = data.shape
        
        # Compute scale factors
        scale_h = h // curr_h
        scale_w = w // curr_w
        
        # Repeat along height and width
        upsampled = np.repeat(np.repeat(data, scale_h, axis=1), scale_w, axis=2)
        
        # Crop to exact size if needed
        upsampled = upsampled[:, :h, :w, :]
        
        return tensor(upsampled, requires_grad=x.requires_grad)


# Detection Head

class DetectionHead(Module):
    """
    Detection head that predicts boxes and class scores.
    
    Args:
        in_channels: Input channels from FPN
        num_classes: Number of object classes
        num_anchors: Number of anchors per location
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Output channels: (x, y, w, h, objectness, class_scores) * num_anchors
        out_channels = num_anchors * (5 + num_classes)
        
        # Shared conv layers
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, seed=seed)
        self.conv2 = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, seed=seed + 1)
        
        # Output layer (no activation)
        self.output = Conv2D(in_channels, out_channels, kernel_size=1, padding=0, seed=seed + 2)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature map (batch, height, width, channels)
            
        Returns:
            Predictions (batch, height, width, num_anchors, 5 + num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(x)
        
        # Reshape to (batch, h, w, num_anchors, 5 + num_classes)
        batch, h, w, _ = x.data.shape
        x = tensor(
            x.data.reshape(batch, h, w, self.num_anchors, 5 + self.num_classes),
            requires_grad=x.requires_grad
        )
        
        return x


# Full Detection Model

class DetectionModel(Module):
    """
    Complete object detection model (YOLO-style).
    
    Args:
        num_classes: Number of object classes
        model_size: Model size ('nano', 'small', 'medium', 'large', 'xlarge')
        in_channels: Input image channels
        anchors: Anchor box sizes per scale
    """
    
    # Model size configurations
    SIZE_CONFIGS = {
        'nano':   {'depth': 0.33, 'width': 0.25},
        'small':  {'depth': 0.33, 'width': 0.50},
        'medium': {'depth': 0.67, 'width': 0.75},
        'large':  {'depth': 1.00, 'width': 1.00},
        'xlarge': {'depth': 1.33, 'width': 1.25},
    }
    
    # Default anchors for 640x640 images (3 scales, 3 anchors each)
    DEFAULT_ANCHORS = [
        [[10, 13], [16, 30], [33, 23]],      # P3/8
        [[30, 61], [62, 45], [59, 119]],     # P4/16
        [[116, 90], [156, 198], [373, 326]], # P5/32
    ]
    
    def __init__(
        self,
        num_classes: int,
        model_size: str = "small",
        in_channels: int = 3,
        anchors: Optional[List[List[List[int]]]] = None,
        seed: int = 0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_size = model_size
        self.anchors = anchors or self.DEFAULT_ANCHORS
        self.num_anchors = len(self.anchors[0])
        
        # Get size config
        if model_size not in self.SIZE_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(self.SIZE_CONFIGS.keys())}")
        
        config = self.SIZE_CONFIGS[model_size]
        
        # Build model
        self.backbone = Backbone(
            in_channels=in_channels,
            depth_multiple=config['depth'],
            width_multiple=config['width'],
            seed=seed,
        )
        
        self.fpn = FeaturePyramid(
            in_channels=self.backbone.out_channels,
            out_channels=int(256 * config['width']),
            seed=seed + 100,
        )
        
        # Detection heads for each scale
        fpn_out = int(256 * config['width'])
        self.heads = [
            DetectionHead(fpn_out, num_classes, self.num_anchors, seed=seed + 200 + i)
            for i in range(3)
        ]
        
        # Strides for each detection level
        self.strides = [8, 16, 32]
    
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images (batch, height, width, channels)
            
        Returns:
            List of predictions at 3 scales
        """
        # Backbone features
        features = self.backbone(x)
        
        # FPN
        features = self.fpn(features)
        
        # Detection heads
        outputs = [head(feat) for head, feat in zip(self.heads, features)]
        
        return outputs
    
    def decode_predictions(
        self,
        outputs: List[Tensor],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: Tuple[int, int] = (640, 640),
    ) -> List[Dict]:
        """
        Decode model outputs to bounding boxes.
        
        Args:
            outputs: List of model outputs at 3 scales
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            img_size: Original image size (height, width)
            
        Returns:
            List of detections per image
        """
        from neurova.object_detection.utils import non_max_suppression, xywh_to_xyxy
        
        batch_size = outputs[0].data.shape[0]
        all_detections = [[] for _ in range(batch_size)]
        
        for scale_idx, (output, stride, anchors) in enumerate(
            zip(outputs, self.strides, self.anchors)
        ):
            data = output.data  # (batch, h, w, num_anchors, 5 + num_classes)
            _, h, w, na, _ = data.shape
            
            # Create grid
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid_x = grid_x.reshape(1, h, w, 1)
            grid_y = grid_y.reshape(1, h, w, 1)
            
            # Decode predictions
            # xy = (sigmoid(xy) * 2 - 0.5 + grid) * stride
            xy = (self._sigmoid(data[..., :2]) * 2 - 0.5)
            xy[..., 0] += grid_x
            xy[..., 1] += grid_y
            xy *= stride
            
            # wh = (sigmoid(wh) * 2) ** 2 * anchor
            wh = (self._sigmoid(data[..., 2:4]) * 2) ** 2
            anchor_array = np.array(anchors).reshape(1, 1, 1, na, 2)
            wh *= anchor_array
            
            # Objectness and class scores
            obj = self._sigmoid(data[..., 4:5])
            cls = self._sigmoid(data[..., 5:])
            
            # Combine objectness and class scores
            scores = obj * cls
            
            # Flatten predictions
            for b in range(batch_size):
                boxes = np.concatenate([xy[b], wh[b]], axis=-1).reshape(-1, 4)
                box_scores = scores[b].reshape(-1, self.num_classes)
                
                # Get class with highest score
                class_ids = np.argmax(box_scores, axis=1)
                confidences = np.max(box_scores, axis=1)
                
                # Filter by confidence
                mask = confidences >= conf_threshold
                boxes = boxes[mask]
                class_ids = class_ids[mask]
                confidences = confidences[mask]
                
                for i in range(len(boxes)):
                    all_detections[b].append({
                        'box': boxes[i],  # xywh
                        'class_id': class_ids[i],
                        'confidence': confidences[i],
                    })
        
        # Apply NMS per image
        results = []
        for b in range(batch_size):
            dets = all_detections[b]
            if not dets:
                results.append({
                    'boxes': np.array([]).reshape(0, 4),
                    'class_ids': np.array([]),
                    'confidences': np.array([]),
                })
                continue
            
            boxes = np.array([d['box'] for d in dets])
            class_ids = np.array([d['class_id'] for d in dets])
            confidences = np.array([d['confidence'] for d in dets])
            
            # Convert to xyxy for NMS
            boxes_xyxy = xywh_to_xyxy(boxes)
            
            # NMS per class
            from neurova.object_detection.utils import nms_per_class
            keep = nms_per_class(
                boxes_xyxy, confidences, class_ids,
                iou_threshold=iou_threshold,
                score_threshold=conf_threshold,
            )
            
            # Normalize to 0-1
            boxes_xyxy = boxes_xyxy[keep]
            boxes_xyxy[:, [0, 2]] /= img_size[1]  # width
            boxes_xyxy[:, [1, 3]] /= img_size[0]  # height
            
            results.append({
                'boxes': boxes_xyxy,
                'class_ids': class_ids[keep],
                'confidences': confidences[keep],
            })
        
        return results
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def save(self, path: str) -> None:
        """Save model weights."""
        params = {}
        for i, param in enumerate(self.parameters()):
            params[f'param_{i}'] = param.data
        np.savez(path, **params)
    
    def load(self, path: str) -> None:
        """Load model weights."""
        data = np.load(path)
        params = self.parameters()
        for i, param in enumerate(params):
            key = f'param_{i}'
            if key in data:
                param.data = data[key]

# Neurova Library
# Copyright (c) 2025 Neurova Team
# Licensed under the MIT License
# @analytics with harry
