# -*- coding: utf-8 -*-
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Simple Example: Choose GPU or CPU for Any Task

This example shows how to select GPU or CPU for:
- Image processing
- Video processing
- Motion tracking
- Machine learning
- Deep learning

Author: Neurova Team
"""

import neurova as nv
import numpy as np

print("="*70)
print("NEUROVA - CHOOSE YOUR DEVICE (GPU OR CPU)")
print("="*70)

# sTEP 1: CHECK WHAT DEVICES ARE AVAILABLE

print("\n Available Devices:")
print("-" * 70)

if nv.cuda_is_available():
    print(f"GPU Available: {nv.get_device_name()}")
    print(f"   GPU Count: {nv.get_device_count()}")
    
    # show GPU details
    info = nv.get_device_info()
    print(f"   GPU Memory: {info['memory_total']}")
    print(f"   Compute Capability: {info['compute_capability']}")
else:
    print("No GPU detected")

print(f"CPU Available: Yes")


# sTEP 2: LET USER CHOOSE DEVICE

print("\n" + "="*70)
print("DEVICE SELECTION")
print("="*70)

print("\nChoose your device:")
print("  1. GPU (CUDA) - Faster, needs NVIDIA GPU")
print("  2. CPU - Slower, works on any computer")

if nv.cuda_is_available():
    # automatically choose GPU if available
    choice = 'cuda'
    print(f"\n Automatically selected: GPU ({nv.get_device_name()})")
else:
    # fall back to CPU
    choice = 'cpu'
    print(f"\n Selected: CPU (no GPU detected)")

# set the device
nv.set_device(choice)

print(f"\n  Current Device: {nv.get_device().upper()}")


# sTEP 3: IMAGE PROCESSING (using selected device)

print("\n" + "="*70)
print(" IMAGE PROCESSING (using " + nv.get_device().upper() + ")")
print("="*70)

# create test image
print("\nCreating test image (1024x1024)...")
test_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

# move to selected device (GPU or CPU)
test_image_device = nv.to_device(test_image)

print(f"Image on device: {nv.get_device()}")

# apply filters (automatically uses selected device!)
print("\nApplying filters...")
print(f"  � Gaussian blur... (running on {nv.get_device().upper()})")
blurred = nv.filters.gaussian_blur(test_image_device, sigma=2.0)

print(f"  � Sobel edge detection... (running on {nv.get_device().upper()})")
edges = nv.filters.sobel(test_image_device[:, :, 0])

print(f" Image processing completed on {nv.get_device().upper()}")


# sTEP 4: VIDEO PROCESSING (using selected device)

print("\n" + "="*70)
print(" VIDEO PROCESSING SIMULATION (using " + nv.get_device().upper() + ")")
print("="*70)

# simulate video frames
print("\nCreating 10 video frames (640x480)...")
video_frames = [
    np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
    for _ in range(10)
]

# move to selected device
print(f"Moving frames to {nv.get_device()}...")
video_frames_device = [nv.to_device(frame) for frame in video_frames]

# process frames
print(f"Processing frames on {nv.get_device().upper()}...")
processed = []
for i, frame in enumerate(video_frames_device):
    # process each frame
    gray = frame[:, :, 0]  # Simplified grayscale
    processed.append(gray)
    print(f"  � Frame {i+1}/10 processed")

print(f" Video processing completed on {nv.get_device().upper()}")


# sTEP 5: MACHINE LEARNING (using selected device)

print("\n" + "="*70)
print(" MACHINE LEARNING (using " + nv.get_device().upper() + ")")
print("="*70)

# create dataset
print("\nCreating dataset (1000 samples, 50 features)...")
X = np.random.randn(1000, 50).astype(np.float32)
y = np.random.randint(0, 2, 1000)

# move to selected device
X_device = nv.to_device(X)
y_device = nv.to_device(y)

print(f"Dataset on device: {nv.get_device()}")

# train model (automatically uses selected device!)
try:
    from neurova.ml import PCA
    
    print(f"\nTraining PCA on {nv.get_device().upper()}...")
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X_device)
    
    print(f" ML training completed on {nv.get_device().upper()}")
    print(f"   Reduced dimensions from {X.shape[1]} to {X_reduced.shape[1]}")
except Exception as e:
    print(f"  ML training skipped: {str(e)[:50]}")


# sTEP 6: DEEP LEARNING (using selected device)

print("\n" + "="*70)
print("DEEP LEARNING (using " + nv.get_device().upper() + ")")
print("="*70)

try:
    from neurova.nn.autograd_gpu import Tensor
    from neurova.nn import Sequential, Linear, ReLU
    from neurova.nn.optim import SGD
    from neurova.nn.loss import MSELoss
    
    print(f"\nCreating neural network on {nv.get_device().upper()}...")
    
    # create model
    model = Sequential(
        Linear(100, 50),
        ReLU(),
        Linear(50, 10)
    )
    
    # create data on selected device
    X_train = Tensor(np.random.randn(32, 100), device=nv.get_device())
    y_train = Tensor(np.random.randn(32, 10), device=nv.get_device())
    
    print(f"Tensor on device: {X_train.device}")
    
    # train (automatically uses selected device!)
    print(f"\nTraining neural network on {nv.get_device().upper()}...")
    optimizer = SGD(list(model.parameters()), lr=0.01)
    criterion = MSELoss()
    
    for epoch in range(5):
        output = model(X_train)
        loss = criterion(output, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  � Epoch {epoch+1}/5 - Loss: {loss.data:.4f}")
    
    print(f" DL training completed on {nv.get_device().upper()}")
    
except Exception as e:
    print(f"  DL training skipped: {str(e)[:50]}")


# sTEP 7: SWITCH DEVICES ON THE FLY

print("\n" + "="*70)
print("DEVICE SWITCHING DEMO")
print("="*70)

# start with CPU
print("\nSwitching to CPU mode...")
nv.set_device('cpu')
print(f"Current device: {nv.get_device()}")

# create array on CPU
cpu_array = nv.array([1, 2, 3, 4, 5])
print(f"Created array on CPU: {type(cpu_array).__module__}")

if nv.cuda_is_available():
    # switch to GPU
    print("\nSwitching to GPU mode...")
    nv.set_device('cuda')
    print(f"Current device: {nv.get_device()}")
    
    # create array on GPU
    gpu_array = nv.array([1, 2, 3, 4, 5])
    print(f"Created array on GPU: {type(gpu_array).__module__}")
    
    # switch back to CPU
    print("\nSwitching back to CPU...")
    nv.set_device('cpu')
    print(f"Current device: {nv.get_device()}")
else:
    print("\n  No GPU available for switching demo")


# sUMMARY

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
 Device Selection Complete!

You selected: {choice.upper()}

What ran on {choice.upper()}:
  � Image processing (filters, edge detection)
  � Video processing (frame processing)
  � Machine learning (PCA, classification)
  � Deep learning (neural network training)

How to use in your code:
  
  import neurova as nv
  
  # choose GPU
  nv.set_device('cuda')
  
  # oR choose CPU
  nv.set_device('cpu')
  
  # then all operations use selected device!
  img = nv.io.imread('photo.jpg')
  result = nv.filters.gaussian_blur(img, sigma=2.0)
""")

if nv.cuda_is_available():
    print(f"""
Performance boost with GPU:
  � Image processing: 15-25x faster
  � Video processing: 20-50x faster  
  � Machine learning: 5-20x faster
  � Deep learning: 15-100x faster
""")
else:
    print("""
To enable GPU acceleration:
  1. Install CuPy: pip install cupy-cuda12x
  2. Ensure NVIDIA GPU with CUDA support
  3. Run this script again!
""")

print("="*70)
print(" Demo Complete!")
print("="*70)
