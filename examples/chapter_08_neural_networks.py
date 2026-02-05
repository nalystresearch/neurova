# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Chapter 8: Neural Networks with Neurova


This chapter covers:
- Neural network layers
- Activation functions  
- Loss functions
- Optimizers
- Building custom networks
- CNN architectures
- Training neural networks

Using Neurova's pure-Python neural network implementations!

Author: Neurova Team
"""

import numpy as np
from pathlib import Path

print("")
print("Chapter 8: Neural Networks with Neurova")
print("")

import neurova as nv
from neurova import datasets

# 8.1 tensors and autograd
print(f"\n8.1 Tensors and Autograd")

from neurova.neural import Tensor, tensor

# create tensors
x = tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"    Tensor x: {x.data}")
print(f"    Requires grad: {x.requires_grad}")

# operations create computational graph
y = x * 2
z = y.sum()
print(f"    y = x * 2: {y.data}")
print(f"    z = y.sum(): {z.data}")

# backward pass
z.backward()
print(f"    x.grad after z.backward(): {x.grad}")

# 8.2 neural network layers
print(f"\n8.2 Neural Network Layers")

from neurova.neural import layers

# Linear (Dense) Layer
linear = layers.Linear(in_features=10, out_features=5)
print(f"    Linear Layer: 10 -> 5")
print(f"      Weight shape: {linear.weight.data.shape}")
if linear.bias is not None:
    print(f"      Bias shape: {linear.bias.data.shape}")

# forward pass
x_input = tensor(np.random.randn(4, 10).astype(np.float32))
output = linear(x_input)
print(f"      Input: {x_input.data.shape} -> Output: {output.data.shape}")

# activation layers
relu = layers.ReLU()
tanh = layers.Tanh()
sigmoid = layers.Sigmoid()
print(f"\n    Activation Layers: ReLU, Tanh, Sigmoid")

x_act = tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
print(f"    Input: {x_act.data}")
print(f"    ReLU: {relu(x_act).data}")
print(f"    Tanh: {np.round(tanh(x_act).data, 3)}")
print(f"    Sigmoid: {np.round(sigmoid(x_act).data, 3)}")

# 8.3 convolutional layers
print(f"\n8.3 Convolutional Layers")

from neurova.neural.conv import Conv2D, MaxPool2D, AvgPool2D, Flatten, BatchNorm2D

# Conv2D layer
conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
print(f"    Conv2D: 3 channels -> 16 channels")
print(f"      Kernel size: 3x3")
print(f"      Stride: 1, Padding: 1")

# Forward pass (NHWC format: batch, height, width, channels)
x_conv = tensor(np.random.randn(2, 32, 32, 3).astype(np.float32))
output_conv = conv(x_conv)
print(f"      Input: {x_conv.data.shape} -> Output: {output_conv.data.shape}")

# MaxPool2D layer
pool = MaxPool2D(kernel_size=2, stride=2)
print(f"\n    MaxPool2D: kernel=2, stride=2")

output_pool = pool(output_conv)
print(f"      Input: {output_conv.data.shape} -> Output: {output_pool.data.shape}")

# flatten layer
flatten = Flatten()
output_flat = flatten(output_pool)
print(f"\n    Flatten: {output_pool.data.shape} -> {output_flat.data.shape}")

# 8.4 loss functions
print(f"\n8.4 Loss Functions")

from neurova.neural.functional import cross_entropy, mse_loss, softmax

# mean squared error
y_pred = tensor(np.array([1.1, 1.9, 3.2], dtype=np.float32), requires_grad=True)
y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)

mse = mse_loss(y_pred, y_true)
print(f"    MSE Loss: {mse.data:.6f}")

# Cross-Entropy Loss (requires logits, not probabilities)
logits = tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
target = np.array([0, 1, 2, 0])  # Class indices

ce_loss = cross_entropy(logits, target)
print(f"    Cross-Entropy Loss: {ce_loss.data:.6f}")

# softmax
probs = softmax(logits.data)
print(f"    Softmax output sum (per sample): {probs.sum(axis=1)}")

# 8.5 optimizers
print(f"\n8.5 Optimizers")

from neurova.neural import optim

# create a simple model for optimizer demo
simple_linear = layers.Linear(5, 2)
params = simple_linear.parameters()

# sgd optimizer
sgd = optim.SGD(params, lr=0.01, momentum=0.9)
print(f"    SGD: lr=0.01, momentum=0.9")

# adam optimizer
adam = optim.Adam(params, lr=0.001, betas=(0.9, 0.999))
print(f"    Adam: lr=0.001, betas=(0.9, 0.999)")

# 8.6 building a neural network with sequential
print(f"\n8.6 Building a Neural Network with Sequential")

from neurova.neural.layers import Sequential

# create a simple feedforward network
model = Sequential(
    layers.Linear(784, 128),
    layers.ReLU(),
    layers.Linear(128, 64),
    layers.ReLU(),
    layers.Linear(64, 10),
)

print(f"    Model: Sequential")
print(f"    Layers: {len(model.layers)}")
for i, layer in enumerate(model.layers):
    print(f"      {i+1}. {layer.__class__.__name__}")

# forward pass
x_input = tensor(np.random.randn(8, 784).astype(np.float32))
output = model(x_input)
print(f"\n    Input: {x_input.data.shape}")
print(f"    Output: {output.data.shape}")

# 8.7 using fashion-mnist dataset
print(f"\n8.7 Using Fashion-MNIST Dataset")

try:
    (train_images, train_labels), (test_images, test_labels) = datasets.load_fashion_mnist()
    print(f"    Loaded Fashion-MNIST dataset from Neurova")
    print(f"    Train images: {train_images.shape}")
    print(f"    Train labels: {train_labels.shape}")
    print(f"    Test images: {test_images.shape}")
    print(f"    Test labels: {test_labels.shape}")
    
# prepare data for neural network
    # Normalize to [0, 1]
    X_train = train_images.astype(np.float32) / 255.0
    X_test = test_images.astype(np.float32) / 255.0
    
# flatten for mlp
    X_train_flat = X_train.reshape(-1, 28*28)
    X_test_flat = X_test.reshape(-1, 28*28)
    
    print(f"    Flattened train: {X_train_flat.shape}")
    print(f"    Normalized range: [{X_train_flat.min():.2f}, {X_train_flat.max():.2f}]")
    
except Exception as e:
    print(f"    Fashion-MNIST not available: {e}")
# create synthetic data
    X_train_flat = np.random.randn(1000, 784).astype(np.float32)
    train_labels = np.random.randint(0, 10, 1000)
    X_test_flat = np.random.randn(200, 784).astype(np.float32)
    test_labels = np.random.randint(0, 10, 200)
    print(f"    Using synthetic data instead")
    print(f"    Train: {X_train_flat.shape}")

# 8.8 training loop example
print(f"\n8.8 Training Loop Example")

# use a small subset for demo
batch_size = 32
n_samples = min(500, len(X_train_flat))
X_batch = X_train_flat[:n_samples]
y_batch = train_labels[:n_samples]

# create a small model
train_model = Sequential(
    layers.Linear(784, 64),
    layers.ReLU(),
    layers.Linear(64, 10),
)

# optimizer
optimizer = optim.SGD(train_model.parameters(), lr=0.01)

print(f"    Training on {n_samples} samples, batch_size={batch_size}")
print(f"    Model: 784 -> 64 -> 10")

# Training loop (simplified)
n_epochs = 3
for epoch in range(n_epochs):
    total_loss = 0.0
    n_batches = n_samples // batch_size
    
    for i in range(n_batches):
# get batch
        start = i * batch_size
        end = start + batch_size
        x = tensor(X_batch[start:end], requires_grad=False)
        y = y_batch[start:end]
        
# forward pass
        logits = train_model(x)
        
# compute loss
        loss = cross_entropy(logits, y)
        total_loss += loss.data
        
# backward pass
        optimizer.zero_grad()
        loss.backward()
        
# update weights
        optimizer.step()
    
    avg_loss = total_loss / n_batches
    print(f"    Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")

# 8.9 model evaluation
print(f"\n8.9 Model Evaluation")

# Evaluate on test set (small subset)
n_test = min(200, len(X_test_flat))
X_eval = X_test_flat[:n_test]
y_eval = test_labels[:n_test]

# Forward pass (no gradient needed)
x_test = tensor(X_eval, requires_grad=False)
logits_test = train_model(x_test)
predictions = np.argmax(logits_test.data, axis=1)

accuracy = (predictions == y_eval).mean() * 100
print(f"    Test samples: {n_test}")
print(f"    Accuracy: {accuracy:.2f}%")

# Per-class accuracy (show first 5 classes)
print(f"    Per-class accuracy (first 5 classes):")
for cls in range(5):
    mask = y_eval == cls
    if mask.sum() > 0:
        cls_acc = (predictions[mask] == y_eval[mask]).mean() * 100
        print(f"      Class {cls}: {cls_acc:.1f}%")

# 8.10 saving and loading weights
print(f"\n8.10 Saving and Loading Weights")

from neurova.neural import save_weights, load_weights

# Save model weights (to temporary location)
import tempfile
with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
    weights_path = f.name

save_weights(train_model, weights_path)
print(f"    Saved weights to: {weights_path}")

# create new model and load weights
new_model = Sequential(
    layers.Linear(784, 64),
    layers.ReLU(),
    layers.Linear(64, 10),
)

load_weights(new_model, weights_path)
print(f"    Loaded weights into new model")

# verify predictions match
new_logits = new_model(x_test)
new_predictions = np.argmax(new_logits.data, axis=1)
match = (predictions == new_predictions).all()
print(f"    Predictions match original: {match}")

# 8.11 custom module
print(f"\n8.11 Custom Module")

from neurova.neural import Module

class MLP(Module):
    """Custom Multi-Layer Perceptron."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers_list = []
        
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i in range(len(dims) - 1):
            self.layers_list.append(layers.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                self.layers_list.append(layers.ReLU())
        
# store as attributes for parameter discovery
        for idx, layer in enumerate(self.layers_list):
            setattr(self, f'layer_{idx}', layer)
    
    def forward(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers_list:
            params.extend(layer.parameters())
        return params

# create custom mlp
custom_mlp = MLP(input_dim=100, hidden_dims=[64, 32], output_dim=5)
print(f"    Custom MLP: 100 -> 64 -> 32 -> 5")

# test forward pass
x_custom = tensor(np.random.randn(8, 100).astype(np.float32))
out_custom = custom_mlp(x_custom)
print(f"    Input: {x_custom.data.shape} -> Output: {out_custom.data.shape}")
print(f"    Parameters: {len(custom_mlp.parameters())}")

# 8.12 model zoo
print(f"\n8.12 Model Zoo")

from neurova.neural import ModelZoo

print("    Available pre-defined architectures in ModelZoo:")
print("      - MLP (Multi-Layer Perceptron)")
print("      - CNN (Convolutional Neural Network)")
print("      - Simple classifiers")

# modelzoo provides utilities for common model architectures
# Example: Create a standard classifier
# classifier = ModelZoo.create_classifier(input_dim=784, hidden_dims=[256, 128], num_classes=10)

# summary
print("\n" + "=" * 60)
print("Chapter 8 Summary:")
print("   Used Tensors with automatic differentiation")
print("   Created Linear (Dense) layers")
print("   Applied activation functions (ReLU, Tanh, Sigmoid)")
print("   Used Conv2D, MaxPool2D, Flatten layers")
print("   Computed MSE and Cross-Entropy losses")
print("   Used SGD and Adam optimizers")
print("   Built models with Sequential")
print("   Loaded Fashion-MNIST dataset")
print("   Trained a model with gradient descent")
print("   Evaluated model accuracy")
print("   Saved and loaded model weights")
print("   Created custom Module classes")
print("")
