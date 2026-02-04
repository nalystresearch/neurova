# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Chapter 12: GPU Acceleration & Performance
===========================================

This chapter covers:
- Device configuration (CPU/GPU)
- GPU memory management
- Performance optimization
- Benchmarking
- Parallel processing
- Batch processing

Using Neurova's GPU acceleration features!

Author: Neurova Team
"""

import numpy as np
import time
from pathlib import Path

print("=" * 60)
print("Chapter 12: GPU Acceleration & Performance")
print("=" * 60)

import neurova as nv

# 12.1 device configuration
print(f"\n12.1 Device Configuration")

from neurova import device

# get current device
current = device.get_device()
print(f"    Current device: {current}")

# Check available devices (cpu always available, cuda if GPU present)
available = ['cpu', 'cuda'] if device.cuda_is_available() else ['cpu']
print(f"    Available devices: {available}")

# check gpu availability
has_gpu = device.cuda_is_available()
print(f"    GPU available: {has_gpu}")

# 12.2 setting device
print(f"\n12.2 Setting Device")

# Set to CPU (always available)
device.set_device('cpu')
print(f"    Set device to: {device.get_device()}")

# try to set gpu if available
if has_gpu:
    device.set_device('gpu')
    print(f"    Set device to: {device.get_device()}")
else:
    print("    GPU not available, using CPU")

# Device context manager (if supported)
print("""
    # Using context manager:
    with device.device_context('gpu'):
        result = expensive_operation(data)
# automatically returns to previous device
""")

# 12.3 gpu information
print(f"\n12.3 GPU Information")

if has_gpu:
    info = device.get_device_info()
    print(f"    GPU Name: {info.get('gpu_name', 'Unknown')}")
    print(f"    Memory: {info.get('memory_total', 'Unknown')}")
    print(f"    Compute capability: {info.get('compute_capability', 'Unknown')}")
else:
    print("    No GPU available")
    print("    Running on CPU mode")

# 12.4 memory management
print(f"\n12.4 Memory Management")

class MemoryTracker:
    """Track memory usage during operations."""
    
    def __init__(self):
        self.snapshots = []
    
    def snapshot(self, label=""):
        """Take a memory snapshot."""
        import sys
        
# get python memory info
        snapshot = {
            'label': label,
            'time': time.time()
        }
        
# try to get gpu memory if available
        if has_gpu:
            try:
                gpu_mem = device.get_memory_usage()
                snapshot['gpu_allocated'] = gpu_mem
            except:
                pass
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def report(self):
        """Print memory report."""
        for snap in self.snapshots:
            print(f"      {snap['label']}: {snap}")

tracker = MemoryTracker()
tracker.snapshot("Initial")

# allocate some data
large_array = np.random.randn(1000, 1000).astype(np.float32)
tracker.snapshot("After allocation")

# process
result = large_array @ large_array.T
tracker.snapshot("After computation")

print(f"    Memory tracking: {len(tracker.snapshots)} snapshots")

# 12.5 benchmarking utilities
print(f"\n12.5 Benchmarking Utilities")

class Benchmark:
    """Benchmark execution time."""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.times = []
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.times.append(elapsed)
    
    def run(self, func, *args, n_runs=10, warmup=2, **kwargs):
        """Run benchmark multiple times."""
# warmup runs
        for _ in range(warmup):
            func(*args, **kwargs)
        
# timed runs
        for _ in range(n_runs):
            with self:
                func(*args, **kwargs)
        
        return self.stats()
    
    def stats(self):
        """Get benchmark statistics."""
        if not self.times:
            return {}
        
        times = np.array(self.times)
        return {
            'name': self.name,
            'mean': times.mean() * 1000,  # ms
            'std': times.std() * 1000,
            'min': times.min() * 1000,
            'max': times.max() * 1000,
            'n_runs': len(times)
        }

# benchmark matrix multiplication
def matrix_multiply(a, b):
    return a @ b

A = np.random.randn(500, 500).astype(np.float32)
B = np.random.randn(500, 500).astype(np.float32)

bench = Benchmark("MatMul 500x500")
stats = bench.run(matrix_multiply, A, B, n_runs=5, warmup=2)

print(f"    {stats['name']}:")
print(f"      Mean: {stats['mean']:.2f} ms")
print(f"      Std: {stats['std']:.2f} ms")
print(f"      Min: {stats['min']:.2f} ms")
print(f"      Max: {stats['max']:.2f} ms")

# 12.6 batch processing
print(f"\n12.6 Batch Processing")

def process_batch(images, operation):
    """Process a batch of images efficiently."""
    results = []
    for img in images:
        results.append(operation(img))
    return np.array(results)

def vectorized_operation(batch):
    """Vectorized batch operation."""
# process entire batch at once
    return batch * 2 + 1

# create batch of images
batch_size = 32
image_batch = np.random.randint(0, 255, (batch_size, 64, 64), dtype=np.uint8)

# time sequential vs vectorized
start = time.perf_counter()
seq_result = process_batch(image_batch, lambda x: x * 2 + 1)
seq_time = time.perf_counter() - start

start = time.perf_counter()
vec_result = vectorized_operation(image_batch)
vec_time = time.perf_counter() - start

print(f"    Batch size: {batch_size}")
print(f"    Sequential: {seq_time*1000:.2f} ms")
print(f"    Vectorized: {vec_time*1000:.2f} ms")
print(f"    Speedup: {seq_time/vec_time:.1f}x")

# 12.7 data type optimization
print(f"\n12.7 Data Type Optimization")

# compare different data types
data_fp64 = np.random.randn(1000, 1000)
data_fp32 = data_fp64.astype(np.float32)
data_fp16 = data_fp64.astype(np.float16)

print(f"    Float64 size: {data_fp64.nbytes / 1024**2:.2f} MB")
print(f"    Float32 size: {data_fp32.nbytes / 1024**2:.2f} MB")
print(f"    Float16 size: {data_fp16.nbytes / 1024**2:.2f} MB")

# benchmark different types
for dtype, data in [('fp64', data_fp64), ('fp32', data_fp32)]:
    bench = Benchmark(f"MatMul {dtype}")
    stats = bench.run(lambda d: d @ d.T, data, n_runs=3, warmup=1)
    print(f"    {dtype}: {stats['mean']:.2f} ms")

# 12.8 parallel processing
print(f"\n12.8 Parallel Processing")

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

def process_image(image):
    """CPU-bound image processing."""
# simulate processing
    result = image.copy()
    for _ in range(10):
        result = np.roll(result, 1, axis=0)
    return result

# create images
n_images = 8
images = [np.random.randint(0, 255, (100, 100), dtype=np.uint8) for _ in range(n_images)]

# sequential processing
start = time.perf_counter()
seq_results = [process_image(img) for img in images]
seq_time = time.perf_counter() - start

# Thread pool (good for I/O bound)
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as executor:
    thread_results = list(executor.map(process_image, images))
thread_time = time.perf_counter() - start

print(f"    CPU cores: {multiprocessing.cpu_count()}")
print(f"    Images: {n_images}")
print(f"    Sequential: {seq_time*1000:.2f} ms")
print(f"    ThreadPool (4 workers): {thread_time*1000:.2f} ms")

# 12.9 caching for performance
print(f"\n12.9 Caching for Performance")

from functools import lru_cache

class ImageCache:
    """Cache for processed images."""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
# remove oldest
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

# demo caching
cache = ImageCache(max_size=10)

# simulate repeated access
keys = list(range(5)) * 3  # Repeat keys
for key in keys:
    result = cache.get(key)
    if result is None:
# compute and cache
        result = np.random.randn(10, 10)
        cache.set(key, result)

stats = cache.stats()
print(f"    Cache hits: {stats['hits']}")
print(f"    Cache misses: {stats['misses']}")
print(f"    Hit rate: {stats['hit_rate']:.1%}")

# 12.10 memory-efficient operations
print(f"\n12.10 Memory-Efficient Operations")

def memory_efficient_convolve(image, kernel, chunk_size=256):
    """Process large images in chunks to save memory."""
    h, w = image.shape
    result = np.zeros_like(image)
    
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='reflect')
    
    for i in range(0, h, chunk_size):
        end_i = min(i + chunk_size, h)
        
        for j in range(0, w, chunk_size):
            end_j = min(j + chunk_size, w)
            
# process chunk
            chunk = padded[i:end_i+2*pad, j:end_j+2*pad]
            
# simple convolution on chunk
            for ci in range(end_i - i):
                for cj in range(end_j - j):
                    result[i+ci, j+cj] = np.sum(
                        chunk[ci:ci+kernel.shape[0], cj:cj+kernel.shape[1]] * kernel
                    )
    
    return result

# demo chunked processing
large_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
kernel = np.ones((3, 3)) / 9

start = time.perf_counter()
result = memory_efficient_convolve(large_image, kernel, chunk_size=128)
elapsed = time.perf_counter() - start

print(f"    Image size: {large_image.shape}")
print(f"    Chunk size: 128x128")
print(f"    Time: {elapsed*1000:.0f} ms")

# 12.11 performance profiling
print(f"\n12.11 Performance Profiling")

class Profiler:
    """Simple profiler for code sections."""
    
    def __init__(self):
        self.sections = {}
    
    def start(self, name):
        self.sections[name] = {'start': time.perf_counter()}
    
    def end(self, name):
        if name in self.sections:
            self.sections[name]['end'] = time.perf_counter()
            self.sections[name]['elapsed'] = (
                self.sections[name]['end'] - self.sections[name]['start']
            )
    
    def report(self):
        total = sum(s.get('elapsed', 0) for s in self.sections.values())
        print("    Profiling Report:")
        print("    " + "-" * 40)
        
        for name, data in sorted(self.sections.items(), 
                                  key=lambda x: x[1].get('elapsed', 0),
                                  reverse=True):
            elapsed = data.get('elapsed', 0)
            pct = elapsed / total * 100 if total > 0 else 0
            print(f"      {name:<20} {elapsed*1000:>8.2f} ms ({pct:>5.1f}%)")
        
        print("    " + "-" * 40)
        print(f"      {'Total':<20} {total*1000:>8.2f} ms")

# profile a pipeline
profiler = Profiler()

# simulate pipeline
profiler.start("load")
data = np.random.randn(500, 500)
time.sleep(0.01)  # Simulate I/O
profiler.end("load")

profiler.start("preprocess")
data = (data - data.mean()) / data.std()
profiler.end("preprocess")

profiler.start("compute")
result = data @ data.T
profiler.end("compute")

profiler.start("postprocess")
result = np.clip(result, -1, 1)
profiler.end("postprocess")

profiler.report()

# 12.12 optimization tips
print(f"\n12.12 Optimization Tips")

print("""
    Performance Optimization Checklist:
    
    1. Use appropriate data types:
        float32 instead of float64 for most cases
        uint8 for images when possible
    
    2. Vectorize operations:
        Use NumPy broadcasting
        Avoid Python loops over pixels
    
    3. Batch processing:
        Process multiple images at once
        Use batch matrix operations
    
    4. Memory management:
        Process large images in chunks
        Reuse buffers when possible
        Clear unused arrays
    
    5. GPU acceleration:
        Use nv.device.set('gpu') for large operations
        Minimize CPU-GPU transfers
        Keep data on GPU between operations
    
    6. Caching:
        Cache repeated computations
        Use LRU cache for transforms
    
    7. Parallelization:
        ThreadPool for I/O bound tasks
        ProcessPool for CPU bound tasks
        Vectorized ops for GPU parallelism
""")

# 12.13 benchmark suite
print(f"\n12.13 Benchmark Suite")

def run_benchmark_suite():
    """Run comprehensive benchmarks."""
    results = {}
    
# image sizes to test
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    
    for size in sizes:
        image = np.random.randint(0, 255, size, dtype=np.uint8).astype(np.float32)
        
# matrix multiply
        bench = Benchmark(f"MatMul {size}")
        stats = bench.run(lambda x: x @ x.T, image, n_runs=3, warmup=1)
        results[f"matmul_{size[0]}"] = stats['mean']
        
# element-wise
        bench = Benchmark(f"ElementWise {size}")
        stats = bench.run(lambda x: x * 2 + 1, image, n_runs=3, warmup=1)
        results[f"elemwise_{size[0]}"] = stats['mean']
    
    return results

suite_results = run_benchmark_suite()

print("    Benchmark Results:")
print("    " + "-" * 30)
for name, time_ms in sorted(suite_results.items()):
    print(f"      {name:<20} {time_ms:>8.2f} ms")

# summary
print("\n" + "=" * 60)
print("Chapter 12 Summary:")
print("   Configured CPU/GPU devices")
print("   Tracked memory usage")
print("   Benchmarked operations")
print("   Optimized batch processing")
print("   Compared data types for performance")
print("   Implemented parallel processing")
print("   Built caching system")
print("   Used memory-efficient algorithms")
print("   Profiled code sections")
print("   Ran benchmark suite")
print("=" * 60)
