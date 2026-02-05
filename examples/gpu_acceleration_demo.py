# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
GPU Acceleration Demo for Neurova

This script demonstrates GPU acceleration across all Neurova modules:
- Image Processing
- Video Processing
- Machine Learning
- Deep Learning
- Computer Vision

Run with:
    python examples/gpu_acceleration_demo.py
"""

import numpy as np
import time
import neurova as nv

print("="*70)
print(" NEUROVA GPU ACCELERATION DEMO")
print("="*70)

# 1. CHECK GPU AVAILABILITY

print("\n DEVICE INFORMATION")
print("")

if nv.cuda_is_available():
    print(" GPU detected!")
    nv.set_device('cuda')
    
    # get detailed device info
    info = nv.get_device_info()
    print(f"   GPU Name: {info['gpu_name']}")
    print(f"   Compute Capability: {info['compute_capability']}")
    print(f"   Total Memory: {info['memory_total']}")
    print(f"   Available Memory: {info['memory_available']}")
    print(f"   GPU Count: {info['gpu_count']}")
else:
    print("  No GPU detected, using CPU")
    nv.set_device('cpu')

current_device = nv.get_device()
print(f"\n  Current Device: {current_device.upper()}")


# 2. IMAGE PROCESSING ON GPU

print("\n" + "="*70)
print(" IMAGE PROCESSING BENCHMARK")
print("="*70)

# create test image
print("\nCreating test image (2048x2048)...")
test_image = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)

# move to current device
backend = nv.get_backend()
test_image_device = nv.to_device(test_image)

print(f"Image on device: {nv.get_device()}")
print(f"Image shape: {test_image.shape}")

# benchmark different operations
operations = [
    ("Gaussian Blur", lambda img: nv.filters.gaussian_blur(img, sigma=2.0)),
    ("Sobel Edge Detection", lambda img: nv.filters.sobel(img[:, :, 0])),  # Grayscale
    ("Image Rotation", lambda img: nv.transform.rotate(img, 45)),
]

print("\n Running image processing benchmarks...")
print(f"{'Operation':<25} {'Time (ms)':<15} {'Device':<10}")
print("")

for op_name, op_func in operations:
    try:
        start = time.time()
        result = op_func(test_image_device)
        nv.synchronize()  # Wait for GPU operations to finish
        elapsed = (time.time() - start) * 1000
        
        print(f"{op_name:<25} {elapsed:>10.2f} ms    {nv.get_device()}")
    except Exception as e:
        print(f"{op_name:<25} ERROR: {str(e)[:30]}")


# 3. VIDEO PROCESSING ON GPU

print("\n" + "="*70)
print(" VIDEO PROCESSING SIMULATION")
print("="*70)

# simulate video frames
print("\nCreating 30 video frames (1920x1080)...")
num_frames = 30
video_frames = [
    np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8) 
    for _ in range(num_frames)
]

# move frames to GPU
print(f"Moving frames to {nv.get_device()}...")
video_frames_device = [nv.to_device(frame) for frame in video_frames]

# batch process frames
print("\n Processing video frames...")
start = time.time()

processed_frames = []
for i, frame in enumerate(video_frames_device):
    # apply multiple operations
    gray = nv.core.rgb2gray(frame) if hasattr(nv.core, 'rgb2gray') else frame[:, :, 0]
    
    # simplified processing (since we may not have all functions implemented)
    processed = gray
    processed_frames.append(processed)
    
    if (i + 1) % 10 == 0:
        print(f"   Processed {i+1}/{num_frames} frames...")

nv.synchronize()
elapsed = (time.time() - start) * 1000

print(f"\n Processed {num_frames} frames in {elapsed:.2f} ms")
print(f"   Average: {elapsed/num_frames:.2f} ms/frame")
print(f"   FPS: {1000*num_frames/elapsed:.1f}")


# 4. MACHINE LEARNING ON GPU

print("\n" + "="*70)
print(" MACHINE LEARNING BENCHMARK")
print("="*70)

# generate dataset
print("\nGenerating dataset (10,000 samples, 100 features)...")
X = np.random.randn(10000, 100).astype(np.float32)
y = np.random.randint(0, 2, 10000)

# move to device
X_device = nv.to_device(X)
y_device = nv.to_device(y)

print(f"Dataset on device: {nv.get_device()}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# test ML operations
print("\n Running ML operations...")

# pCA
try:
    from neurova.ml import PCA
    
    print("\nTesting PCA (100  20 dimensions)...")
    start = time.time()
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X_device)
    nv.synchronize()
    elapsed = (time.time() - start) * 1000
    
    print(f"    PCA completed in {elapsed:.2f} ms")
    print(f"   Output shape: {X_reduced.shape}")
except Exception as e:
    print(f"    PCA error: {str(e)[:50]}")

# k-Means
try:
    from neurova.ml import KMeans
    
    print("\nTesting K-Means (10 clusters, 1000 samples)...")
    X_small = X_device[:1000]
    
    start = time.time()
    kmeans = KMeans(n_clusters=10, max_iter=50)
    labels = kmeans.fit_predict(X_small)
    nv.synchronize()
    elapsed = (time.time() - start) * 1000
    
    print(f"    K-Means completed in {elapsed:.2f} ms")
    print(f"   Found {len(np.unique(labels))} clusters")
except Exception as e:
    print(f"    K-Means error: {str(e)[:50]}")


# 5. DEEP LEARNING ON GPU

print("\n" + "="*70)
print(" DEEP LEARNING BENCHMARK")
print("="*70)

try:
    from neurova.nn.autograd_gpu import Tensor
    from neurova.nn import Sequential, Linear, ReLU
    from neurova.nn.optim import SGD
    from neurova.nn.loss import MSELoss
    
    print("\nCreating neural network (784  256  10)...")
    
    # create model
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 10)
    )
    
    # create dummy data on GPU
    batch_size = 64
    X_train = Tensor(np.random.randn(batch_size, 784), device=nv.get_device())
    y_train = Tensor(np.random.randn(batch_size, 10), device=nv.get_device())
    
    print(f"Input tensor on device: {X_train.device}")
    
    # benchmark forward pass
    print("\n Benchmarking forward pass...")
    start = time.time()
    for _ in range(100):
        output = model(X_train)
    nv.synchronize()
    elapsed = (time.time() - start) * 1000
    
    print(f"    100 forward passes in {elapsed:.2f} ms")
    print(f"   Average: {elapsed/100:.2f} ms/pass")
    
    # benchmark training
    print("\n Benchmarking training loop...")
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = MSELoss()
    
    start = time.time()
    for epoch in range(10):
        output = model(X_train)
        loss = criterion(output, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    nv.synchronize()
    elapsed = (time.time() - start) * 1000
    
    print(f"    10 epochs completed in {elapsed:.2f} ms")
    print(f"   Average: {elapsed/10:.2f} ms/epoch")
    
except Exception as e:
    print(f" Deep learning benchmark error: {str(e)[:100]}")


# 6. MEMORY MANAGEMENT

print("\n" + "="*70)
print(" MEMORY MANAGEMENT")
print("="*70)

if nv.get_device() == 'cuda':
    memory_info = nv.get_memory_usage()
    print(f"\n GPU Memory Usage:")
    print(f"   Used: {memory_info['used_gb']:.2f} GB")
    print(f"   Total: {memory_info['total_gb']:.2f} GB")
    print(f"   Percentage: {memory_info['used_gb']/memory_info['total_gb']*100:.1f}%")
    
    print(f"\n Clearing GPU cache...")
    nv.empty_cache()
    
    memory_info_after = nv.get_memory_usage()
    print(f"   After cleanup: {memory_info_after['used_gb']:.2f} GB")
else:
    print("\nCPU mode - no GPU memory to manage")


# 7. DEVICE SWITCHING

print("\n" + "="*70)
print(" DEVICE SWITCHING DEMO")
print("="*70)

print("\nDemonstrating temporary device switching...")

# set to CPU
nv.set_device('cpu')
print(f"\n1. Current device: {nv.get_device()}")

# create array on CPU
cpu_array = nv.array([1, 2, 3, 4, 5])
print(f"   Created array on CPU: {type(cpu_array).__module__}")

if nv.cuda_is_available():
    # temporarily switch to GPU
    print("\n2. Using GPU context manager...")
    with nv.device_context('cuda'):
        print(f"   Inside context: {nv.get_device()}")
        gpu_array = nv.array([1, 2, 3, 4, 5])
        print(f"   Created array on GPU: {type(gpu_array).__module__}")
    
    print(f"\n3. After context: {nv.get_device()}")
    print("    Device automatically restored to CPU")
else:
    print("    GPU not available for context switching demo")


# sUMMARY

print("\n" + "="*70)
print(" SUMMARY")
print("="*70)

print(f"""
 GPU Acceleration Demo Complete!

Device Used: {current_device.upper()}
""")

if nv.cuda_is_available():
    info = nv.get_device_info()
    print(f"""GPU Details:
   - Name: {info['gpu_name']}
   - Compute Capability: {info['compute_capability']}
   - Total Memory: {info['memory_total']}

To enable GPU for all operations:
   import neurova as nv
   nv.set_device('cuda')
""")
else:
    print("""No GPU detected.

To enable GPU acceleration:
   1. Install CuPy: pip install cupy-cuda12x
   2. Ensure NVIDIA GPU with CUDA support
   3. Run: import neurova as nv; nv.set_device('cuda')
""")

print("="*70)
print(" Demo finished successfully!")
print("="*70)
