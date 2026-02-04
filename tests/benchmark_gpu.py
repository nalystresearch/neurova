# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
GPU Acceleration Benchmark - Neurova vs NumPy

Tests the performance improvement of using CuPy (GPU) vs NumPy (CPU)
for common deep learning operations.

Requirements:
    pip install cupy-cuda12x  # For CUDA 12.x
    # oR
    pip install cupy-cuda11x  # For CUDA 11.x

If CuPy is not installed, will show that GPU acceleration is needed.
"""

import time
import numpy as np

print("=" * 70)
print("NEUROVA GPU ACCELERATION BENCHMARK")
print("=" * 70)

# check if CuPy is available
try:
    import cupy as cp
    HAS_GPU = True
    print(f"\nCuPy found: version {cp.__version__}")
    
    # get GPU info
    device_count = cp.cuda.runtime.getDeviceCount()
    for i in range(device_count):
        props = cp.cuda.runtime.getDeviceProperties(i)
        name = props['name'].decode('utf-8')
        total_mem = props['totalGlobalMem'] / 1024**3  # GB
        print(f"   GPU {i}: {name} ({total_mem:.1f} GB)")
    
except ImportError:
    HAS_GPU = False
    print("\nCuPy not found - GPU acceleration disabled")
    print("   Install with: pip install cupy-cuda12x")
    print("   (or cupy-cuda11x for CUDA 11)")

print("\n" + "=" * 70)
print("BENCHMARK: Matrix Operations")
print("=" * 70)

# test different sizes
sizes = [
    (128, 128, "Small (128x128)"),
    (512, 512, "Medium (512x512)"),
    (2048, 2048, "Large (2048x2048)"),
    (4096, 4096, "Huge (4096x4096)"),
]

if HAS_GPU:
    print("\n{:<20} {:<15} {:<15} {:<15}".format("Size", "CPU (NumPy)", "GPU (CuPy)", "Speedup"))
    print("-" * 70)
    
    for m, n, label in sizes:
        # cPU benchmark
        x_cpu = np.random.randn(m, n).astype(np.float32)
        y_cpu = np.random.randn(n, m).astype(np.float32)
        
        start = time.time()
        for _ in range(10):
            z_cpu = x_cpu @ y_cpu
        cpu_time = (time.time() - start) / 10
        
        # gPU benchmark
        x_gpu = cp.asarray(x_cpu)
        y_gpu = cp.asarray(y_cpu)
        cp.cuda.Stream.null.synchronize()  # Warm up
        
        start = time.time()
        for _ in range(10):
            z_gpu = x_gpu @ y_gpu
            cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) / 10
        
        speedup = cpu_time / gpu_time
        
        print(f"{label:<20} {cpu_time*1000:>10.2f} ms  {gpu_time*1000:>10.2f} ms  {speedup:>10.1f}x")
else:
    print("\n  Skipping GPU benchmark (CuPy not installed)")
    print("   Install CuPy to see GPU speedups:")
    print("   - CUDA 12.x: pip install cupy-cuda12x")
    print("   - CUDA 11.x: pip install cupy-cuda11x")

# test with Neurova's autograd
print("\n" + "=" * 70)
print("BENCHMARK: Neural Network Training (with Autograd)")
print("=" * 70)

if HAS_GPU:
    from neurova.nn.autograd_gpu import Tensor, cuda_is_available, cuda_get_device_name
    
    print(f"\n Using GPU: {cuda_get_device_name()}")
    
    # setup problem
    batch_size = 64
    input_size = 784
    hidden_size = 256
    output_size = 10
    
    # create weights (CPU)
    W1_data = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.01
    W2_data = np.random.randn(hidden_size, output_size).astype(np.float32) * 0.01
    X_data = np.random.randn(batch_size, input_size).astype(np.float32)
    
    # cPU forward pass
    print("\n{:<30} {:<15} {:<15} {:<15}".format("Operation", "CPU", "GPU", "Speedup"))
    print("-" * 70)
    
    # test 1: Forward pass
    start = time.time()
    for _ in range(100):
        X_cpu = Tensor(X_data, requires_grad=True, device='cpu')
        W1_cpu = Tensor(W1_data, requires_grad=True, device='cpu')
        W2_cpu = Tensor(W2_data, requires_grad=True, device='cpu')
        
        h = X_cpu @ W1_cpu
        h = h.relu()
        out = h @ W2_cpu
        loss = out.sum()
    cpu_fwd = (time.time() - start) / 100
    
    # gPU forward pass
    start = time.time()
    for _ in range(100):
        X_gpu = Tensor(X_data, requires_grad=True, device='cuda')
        W1_gpu = Tensor(W1_data, requires_grad=True, device='cuda')
        W2_gpu = Tensor(W2_data, requires_grad=True, device='cuda')
        
        h = X_gpu @ W1_gpu
        h = h.relu()
        out = h @ W2_gpu
        loss = out.sum()
        cp.cuda.Stream.null.synchronize()
    gpu_fwd = (time.time() - start) / 100
    
    speedup_fwd = cpu_fwd / gpu_fwd
    print(f"{'Forward pass':<30} {cpu_fwd*1000:>10.2f} ms  {gpu_fwd*1000:>10.2f} ms  {speedup_fwd:>10.1f}x")
    
    # test 2: Forward + Backward pass
    start = time.time()
    for _ in range(100):
        X_cpu = Tensor(X_data, requires_grad=True, device='cpu')
        W1_cpu = Tensor(W1_data, requires_grad=True, device='cpu')
        W2_cpu = Tensor(W2_data, requires_grad=True, device='cpu')
        
        h = X_cpu @ W1_cpu
        h = h.relu()
        out = h @ W2_cpu
        loss = out.sum()
        loss.backward()
    cpu_bwd = (time.time() - start) / 100
    
    start = time.time()
    for _ in range(100):
        X_gpu = Tensor(X_data, requires_grad=True, device='cuda')
        W1_gpu = Tensor(W1_data, requires_grad=True, device='cuda')
        W2_gpu = Tensor(W2_data, requires_grad=True, device='cuda')
        
        h = X_gpu @ W1_gpu
        h = h.relu()
        out = h @ W2_gpu
        loss = out.sum()
        loss.backward()
        cp.cuda.Stream.null.synchronize()
    gpu_bwd = (time.time() - start) / 100
    
    speedup_bwd = cpu_bwd / gpu_bwd
    print(f"{'Forward + Backward':<30} {cpu_bwd*1000:>10.2f} ms  {gpu_bwd*1000:>10.2f} ms  {speedup_bwd:>10.1f}x")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n GPU Acceleration is working!")
    print(f"   Average speedup: {(speedup_fwd + speedup_bwd) / 2:.1f}x")
    print(f"   Training on GPU is {speedup_bwd:.1f}x faster than CPU")
    print(f"\n To use GPU in your code:")
    print(f"   from neurova.nn.autograd_gpu import Tensor")
    print(f"   x = Tensor([1, 2, 3], device='cuda')  # Use GPU")
    print(f"   x = Tensor([1, 2, 3], device='cpu')   # Use CPU")
    
else:
    from neurova.nn.autograd import Tensor
    
    print("\n  GPU benchmark not available without CuPy")
    print("   Currently using CPU-only version")
    print("\nTo enable GPU acceleration:")
    print("   1. Install CUDA toolkit from NVIDIA")
    print("   2. Install CuPy: pip install cupy-cuda12x")
    print("   3. Run this benchmark again")
    print("\n   Expected speedup: 10-100x faster for matrix operations")

print("\n" + "=" * 70)
