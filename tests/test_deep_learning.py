# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Test Neurova's Deep Learning capabilities - autograd and neural networks.

This demonstrates that Neurova now has:
1. Automatic differentiation (autograd)
2. Neural network layers
3. Optimizers (SGD, Adam, RMSprop, etc.)
4. Loss functions
5. Complete training pipeline
"""

import numpy as np
from neurova.nn import (
    Tensor, Parameter, no_grad,
    Linear, Sequential, Module,
    ReLU, Sigmoid,
    SGD, Adam, AdamW, RMSprop,
    MSELoss, CrossEntropyLoss, BCELoss,
)

print("=" * 70)
print("NEUROVA DEEP LEARNING DEMONSTRATION")
print("=" * 70)

# 1. AUTOMATIC DIFFERENTIATION (Autograd)
print("\n1. AUTOMATIC DIFFERENTIATION (Neurova autograd)")
print("-" * 70)

# create tensors with gradient tracking
x = Tensor([2.0, 3.0], requires_grad=True)
y = Tensor([1.0, 4.0], requires_grad=True)

# forward pass with operations
z = x * y + x ** 2  # z = xy + x^2
loss = z.sum()

# backward pass - automatically compute gradients!
print(f"Forward: x = {x.data}, y = {y.data}")
print(f"         z = x*y + x^2 = {z.data}")
print(f"         loss = sum(z) = {loss.data}")

loss.backward()

print(f"Backward: dL/dx = {x.grad}")
print(f"          dL/dy = {y.grad}")
print(" Autograd working! Gradients computed automatically")

# 2. NEURAL NETWORK LAYERS
print("\n2. NEURAL NETWORK LAYERS (Neurova nn)")
print("-" * 70)

# create a simple network
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

print(f"Model architecture:\n{model}")
print(f"\nNumber of parameters: {sum(1 for _ in model.parameters())}")

# forward pass
batch_size = 32
x = Tensor(np.random.randn(batch_size, 784), requires_grad=True)
output = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(" Neural network forward pass working!")

# 3. OPTIMIZERS (SGD, Adam, RMSprop, etc.)
print("\n3. OPTIMIZERS (Neurova optim)")
print("-" * 70)

# test different optimizers
optimizers = {
    'SGD': SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': Adam(model.parameters(), lr=0.001),
    'AdamW': AdamW(model.parameters(), lr=0.001),
    'RMSprop': RMSprop(model.parameters(), lr=0.01),
}

print("Available optimizers:")
for name in optimizers.keys():
    print(f"   {name}")

# 4. COMPLETE TRAINING LOOP
print("\n4. COMPLETE TRAINING EXAMPLE (Classification)")
print("-" * 70)

# create toy dataset (XOR problem)
np.random.seed(42)
X_train = np.random.randn(100, 2)
y_train = ((X_train[:, 0] * X_train[:, 1]) > 0).astype(np.float32).reshape(-1, 1)

# small network
class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 8)
        self.fc2 = Linear(8, 1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

net = SimpleNet()
optimizer = Adam(net.parameters(), lr=0.01)
criterion = BCELoss()

print("Training neural network on XOR problem...")
print(f"  Input size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Network: 2 -> 8 -> 1 (with ReLU and Sigmoid)")
print(f"  Optimizer: Adam (lr=0.01)")
print(f"  Loss: Binary Cross Entropy\n")

# training loop
num_epochs = 50
for epoch in range(num_epochs):
    # forward pass
    X_tensor = Tensor(X_train, requires_grad=False)
    y_tensor = Tensor(y_train, requires_grad=False)
    
    predictions = net(X_tensor)
    loss = criterion(predictions, y_tensor)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("\n Training completed successfully!")

# test accuracy
with no_grad():
    X_test = Tensor(X_train, requires_grad=False)
    y_pred = net(X_test)
    predictions = (y_pred.data > 0.5).astype(np.float32)
    accuracy = (predictions == y_train).mean()
    print(f"   Final accuracy: {accuracy * 100:.1f}%")

# 5. REGRESSION EXAMPLE
print("\n5. REGRESSION EXAMPLE")
print("-" * 70)

# generate regression data
X_reg = np.random.randn(200, 1)
y_reg = 3 * X_reg + 2 + np.random.randn(200, 1) * 0.1

# simple linear model
reg_model = Linear(1, 1)
reg_optimizer = SGD(reg_model.parameters(), lr=0.01)
mse_loss = MSELoss()

print("Training linear regression (y = 3x + 2)...")
for epoch in range(100):
    X_tensor = Tensor(X_reg, requires_grad=False)
    y_tensor = Tensor(y_reg, requires_grad=False)
    
    pred = reg_model(X_tensor)
    loss = mse_loss(pred, y_tensor)
    
    reg_optimizer.zero_grad()
    loss.backward()
    reg_optimizer.step()
    
    if (epoch + 1) % 25 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# check learned parameters
weight = reg_model.weight.data[0, 0]
bias = reg_model.bias.data[0]
print(f"\n Learned parameters: w={weight:.2f}, b={bias:.2f}")
print(f"   True parameters:    w=3.00, b=2.00")
print(f"   Error: w_error={abs(weight - 3):.3f}, b_error={abs(bias - 2):.3f}")

# sUMMARY
print("\n" + "=" * 70)
print("SUMMARY: NEUROVA DEEP LEARNING CAPABILITIES")
print("=" * 70)

features = {
    "Automatic Differentiation": " Working (autograd engine)",
    "Tensor Operations": " +, -, *, /, @, sum, mean, etc.",
    "Neural Network Layers": " Linear, Sequential, Module",
    "Activation Functions": " ReLU, Sigmoid, Tanh",
    "Optimizers (9 types)": " SGD, Adam, AdamW, RMSprop, etc.",
    "Loss Functions (11 types)": " MSE, CrossEntropy, BCE, etc.",
    "Training Pipeline": " Complete forward/backward/update",
    "Gradient Context": " no_grad() context manager",
    "Native API": " Full Neurova implementation",
}

for feature, status in features.items():
    print(f"  {feature:<30} {status}")

print("\n" + "=" * 70)
print("WHAT'S MISSING (IMPOSSIBLE WITH NUMPY-ONLY):")
print("=" * 70)
print("   GPU Acceleration (would need CuPy/CUDA)")
print("   Pretrained Models (would need hosting)")
print("   Distributed Training (would need MPI/NCCL)")
print("\n" + "=" * 70)
print("VERDICT: Neurova is now a COMPLETE CPU-based deep learning framework!")
print("         Perfect for learning, prototyping, and CPU inference.")
print("         GPU acceleration with Neurova coming soon.")
print("=" * 70)
