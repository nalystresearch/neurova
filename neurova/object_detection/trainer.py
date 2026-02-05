# Neurova Library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)

"""
Object Detection Trainer.

Provides training loop and utilities for object detection models.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from neurova.neural.tensor import Tensor, tensor
from neurova.neural.module import Module


@dataclass
class TrainingConfig:
    """Training configuration.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        momentum: SGD momentum
        warmup_epochs: Number of warmup epochs
        lr_scheduler: Learning rate scheduler ('cosine', 'step', 'none')
        save_dir: Directory to save checkpoints
        save_period: Save checkpoint every N epochs
        val_period: Validate every N epochs
        patience: Early stopping patience (0 = disabled)
        min_delta: Minimum improvement for early stopping
        mixed_precision: Use mixed precision training
        augment: Apply data augmentation
        mosaic: Use mosaic augmentation
        img_size: Training image size
        conf_threshold: Confidence threshold for validation
        iou_threshold: IoU threshold for NMS
    """
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 3
    lr_scheduler: str = "cosine"
    save_dir: str = "./runs/train"
    save_period: int = 10
    val_period: int = 1
    patience: int = 50
    min_delta: float = 0.0
    mixed_precision: bool = False
    augment: bool = True
    mosaic: bool = True
    img_size: Tuple[int, int] = (640, 640)
    conf_threshold: float = 0.001
    iou_threshold: float = 0.6


class DetectionLoss:
    """
    Object detection loss function.
    
    Combines box regression loss, objectness loss, and classification loss.
    """
    
    def __init__(
        self,
        num_classes: int,
        box_weight: float = 0.05,
        obj_weight: float = 1.0,
        cls_weight: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.label_smoothing = label_smoothing
    
    def __call__(
        self,
        predictions: List[Tensor],
        targets: List[Dict],
        model: Any,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute detection loss.
        
        Args:
            predictions: Model predictions at 3 scales
            targets: Target boxes and classes
            model: Detection model (for anchors/strides)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = predictions[0].data
        batch_size = device.shape[0]
        
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        
        for scale_idx, (pred, stride, anchors) in enumerate(
            zip(predictions, model.strides, model.anchors)
        ):
            pred_data = pred.data  # (batch, h, w, na, 5 + nc)
            _, h, w, na, _ = pred_data.shape
            
            # Build target tensor for this scale
            target_obj = np.zeros((batch_size, h, w, na, 1), dtype=np.float32)
            target_box = np.zeros((batch_size, h, w, na, 4), dtype=np.float32)
            target_cls = np.zeros((batch_size, h, w, na, self.num_classes), dtype=np.float32)
            
            for b, target in enumerate(targets):
                boxes = target.get('boxes', np.array([]))
                class_ids = target.get('class_ids', np.array([]))
                
                if len(boxes) == 0:
                    continue
                
                for box, cls_id in zip(boxes, class_ids):
                    # box is (x_center, y_center, width, height) normalized
                    x_center, y_center, box_w, box_h = box
                    
                    # Convert to grid coordinates
                    gx = x_center * w
                    gy = y_center * h
                    gw = box_w * model.img_size[1] if hasattr(model, 'img_size') else box_w * 640
                    gh = box_h * model.img_size[0] if hasattr(model, 'img_size') else box_h * 640
                    
                    gi, gj = int(gx), int(gy)
                    if gi >= w:
                        gi = w - 1
                    if gj >= h:
                        gj = h - 1
                    
                    # Find best anchor
                    anchor_sizes = np.array(anchors)
                    anchor_ratios = anchor_sizes / np.array([gw, gh]).clip(min=1e-6)
                    anchor_match = np.maximum(anchor_ratios, 1 / anchor_ratios).max(axis=1)
                    best_anchor = np.argmin(anchor_match)
                    
                    # Assign target
                    target_obj[b, gj, gi, best_anchor, 0] = 1.0
                    target_box[b, gj, gi, best_anchor] = [gx - gi, gy - gj, gw, gh]
                    target_cls[b, gj, gi, best_anchor, int(cls_id)] = 1.0
            
            # Compute losses
            pred_xy = self._sigmoid(pred_data[..., :2])
            pred_wh = pred_data[..., 2:4]
            pred_obj = self._sigmoid(pred_data[..., 4:5])
            pred_cls = self._sigmoid(pred_data[..., 5:])
            
            # Box loss (only for positive samples)
            pos_mask = target_obj[..., 0] > 0
            if np.any(pos_mask):
                # CIoU loss would be ideal, but we use simplified L1 loss
                box_loss = np.abs(pred_xy[pos_mask] - target_box[..., :2][pos_mask]).mean()
                box_loss += np.abs(pred_wh[pos_mask] - np.log(target_box[..., 2:4][pos_mask].clip(min=1e-6) / 
                                                              np.array(anchors)[np.where(pos_mask)[3]])).mean()
                total_box_loss += box_loss * self.box_weight
            
            # Objectness loss (BCE)
            obj_loss = self._bce_loss(pred_obj, target_obj)
            total_obj_loss += obj_loss * self.obj_weight
            
            # Classification loss (BCE, only for positive samples)
            if np.any(pos_mask):
                cls_loss = self._bce_loss(pred_cls[pos_mask], target_cls[pos_mask])
                total_cls_loss += cls_loss * self.cls_weight
        
        total_loss = total_box_loss + total_obj_loss + total_cls_loss
        
        loss_dict = {
            'box_loss': float(total_box_loss),
            'obj_loss': float(total_obj_loss),
            'cls_loss': float(total_cls_loss),
            'total_loss': float(total_loss),
        }
        
        return tensor(np.array(total_loss), requires_grad=True), loss_dict
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _bce_loss(self, pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
        pred = np.clip(pred, eps, 1 - eps)
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))


class DetectionTrainer:
    """
    Trainer for object detection models.
    
    Args:
        model: Detection model to train
        dataset: Detection dataset
        config: Training configuration
        
    Example:
        >>> trainer = DetectionTrainer(model, dataset)
        >>> trainer.train(epochs=100)
    """
    
    def __init__(
        self,
        model: Module,
        dataset: Any,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.config = config or TrainingConfig()
        
        # Setup save directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.loss_fn = DetectionLoss(
            num_classes=dataset.num_classes,
        )
        
        # Optimizer state
        self._velocities = {}
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'lr': [],
        }
    
    def train(
        self,
        epochs: Optional[int] = None,
        resume: Optional[str] = None,
    ) -> Dict[str, List]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs (overrides config)
            resume: Path to checkpoint to resume from
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.epochs
        
        if resume:
            self._load_checkpoint(resume)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"  Dataset: {len(self.dataset)} training samples")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Image size: {self.config.img_size}")
        print(f"  Classes: {self.dataset.num_classes}")
        print()
        
        start_epoch = self.current_epoch
        patience_counter = 0
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            
            # Get learning rate
            lr = self._get_lr(epoch, epochs)
            self.history['lr'].append(lr)
            
            # Training epoch
            train_loss = self._train_epoch(epoch, lr)
            self.history['train_loss'].append(train_loss)
            
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - lr: {lr:.6f}")
            
            # Validation
            if (epoch + 1) % self.config.val_period == 0:
                val_loss, val_map = self._validate(epoch)
                self.history['val_loss'].append(val_loss)
                self.history['val_map'].append(val_map)
                
                print(f"  val_loss: {val_loss:.4f} - mAP: {val_map:.4f}")
                
                # Check for improvement
                if val_map > self.best_map + self.config.min_delta:
                    self.best_map = val_map
                    self.best_epoch = epoch
                    patience_counter = 0
                    self._save_checkpoint('best.npz')
                else:
                    patience_counter += 1
                
                # Early stopping
                if self.config.patience > 0 and patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_period == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}.npz')
        
        # Save final model
        self._save_checkpoint('last.npz')
        
        print(f"\nTraining complete!")
        print(f"  Best mAP: {self.best_map:.4f} at epoch {self.best_epoch + 1}")
        print(f"  Checkpoints saved to: {self.save_dir}")
        
        return self.history
    
    def _train_epoch(self, epoch: int, lr: float) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch_images, batch_targets in self.dataset.train_iter(
            batch_size=self.config.batch_size,
            shuffle=True,
            augment=self.config.augment,
        ):
            # Normalize images to [0, 1]
            images = batch_images.astype(np.float32) / 255.0
            
            # Forward pass
            x = tensor(images, requires_grad=True)
            outputs = self.model(x)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(outputs, batch_targets, self.model)
            total_loss += loss_dict['total_loss']
            num_batches += 1
            
            # Backward pass
            loss.backward()
            
            # Update parameters (SGD with momentum)
            self._update_parameters(lr)
            
            # Zero gradients
            self.model.zero_grad()
        
        return total_loss / max(num_batches, 1)
    
    def _validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model."""
        from neurova.object_detection.utils import compute_map
        
        total_loss = 0.0
        num_batches = 0
        
        predictions = []
        ground_truths = []
        
        for batch_idx, (batch_images, batch_targets) in enumerate(self.dataset.val_iter(
            batch_size=self.config.batch_size,
        )):
            # Normalize images
            images = batch_images.astype(np.float32) / 255.0
            
            # Forward pass
            x = tensor(images, requires_grad=False)
            outputs = self.model(x)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(outputs, batch_targets, self.model)
            total_loss += loss_dict['total_loss']
            num_batches += 1
            
            # Decode predictions for mAP
            batch_preds = self.model.decode_predictions(
                outputs,
                conf_threshold=self.config.conf_threshold,
                iou_threshold=self.config.iou_threshold,
                img_size=self.config.img_size,
            )
            
            for i, (pred, target) in enumerate(zip(batch_preds, batch_targets)):
                img_id = f"{batch_idx}_{i}"
                
                predictions.append({
                    'image_id': img_id,
                    'boxes': pred['boxes'],
                    'scores': pred['confidences'],
                    'class_ids': pred['class_ids'],
                })
                
                ground_truths.append({
                    'image_id': img_id,
                    'boxes': target['boxes'],
                    'class_ids': target['class_ids'],
                })
        
        # Compute mAP
        map_results = compute_map(
            predictions, ground_truths,
            iou_threshold=0.5,
            num_classes=self.dataset.num_classes,
        )
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return avg_loss, map_results['mAP']
    
    def _update_parameters(self, lr: float) -> None:
        """Update parameters using SGD with momentum."""
        momentum = self.config.momentum
        weight_decay = self.config.weight_decay
        
        for i, param in enumerate(self.model.parameters()):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Weight decay
            if weight_decay > 0:
                grad = grad + weight_decay * param.data
            
            # Momentum
            if i not in self._velocities:
                self._velocities[i] = np.zeros_like(param.data)
            
            self._velocities[i] = momentum * self._velocities[i] + grad
            
            # Update
            param.data -= lr * self._velocities[i]
    
    def _get_lr(self, epoch: int, total_epochs: int) -> float:
        """Get learning rate for current epoch."""
        base_lr = self.config.learning_rate
        
        # Warmup
        if epoch < self.config.warmup_epochs:
            return base_lr * (epoch + 1) / self.config.warmup_epochs
        
        # Scheduler
        if self.config.lr_scheduler == "cosine":
            progress = (epoch - self.config.warmup_epochs) / (total_epochs - self.config.warmup_epochs)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.config.lr_scheduler == "step":
            # Decay at 70% and 90% of training
            if epoch >= 0.9 * total_epochs:
                return base_lr * 0.01
            elif epoch >= 0.7 * total_epochs:
                return base_lr * 0.1
            return base_lr
        else:
            return base_lr
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        path = self.save_dir / filename
        
        # Save model weights
        self.model.save(str(path))
        
        # Save training state
        state = {
            'epoch': self.current_epoch,
            'best_map': self.best_map,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }
        
        state_path = path.with_suffix('.state.npz')
        np.savez(str(state_path), **{k: np.array(v) if isinstance(v, list) else v for k, v in state.items()})
    
    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        self.model.load(path)
        
        state_path = Path(path).with_suffix('.state.npz')
        if state_path.exists():
            state = np.load(str(state_path), allow_pickle=True)
            self.current_epoch = int(state.get('epoch', 0))
            self.best_map = float(state.get('best_map', 0))
            self.best_epoch = int(state.get('best_epoch', 0))


def train_detector(
    data_dir: str,
    class_names: List[str],
    model_size: str = "small",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: Tuple[int, int] = (640, 640),
    save_dir: str = "./runs/train",
    **kwargs,
) -> Tuple[Module, Dict]:
    """
    Convenience function to train an object detector.
    
    Args:
        data_dir: Path to dataset directory
        class_names: List of class names
        model_size: Model size ('nano', 'small', 'medium', 'large')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Training image size
        save_dir: Directory to save results
        **kwargs: Additional TrainingConfig options
        
    Returns:
        Tuple of (trained_model, training_history)
        
    Example:
        >>> model, history = train_detector(
        ...     data_dir='./datasets/coco128',
        ...     class_names=['person', 'car', 'dog'],
        ...     epochs=50,
        ... )
    """
    from neurova.object_detection.dataset import DetectionDataset
    from neurova.object_detection.model import DetectionModel
    
    # Create dataset
    dataset = DetectionDataset(
        data_dir=data_dir,
        names=class_names,
        img_size=img_size,
    )
    
    print(dataset.summary())
    
    # Create model
    model = DetectionModel(
        num_classes=len(class_names),
        model_size=model_size,
    )
    
    # Create config
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        save_dir=save_dir,
        **kwargs,
    )
    
    # Train
    trainer = DetectionTrainer(model, dataset, config)
    history = trainer.train()
    
    return model, history

# Neurova Library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)
