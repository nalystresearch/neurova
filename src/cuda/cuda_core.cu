/**
 * Neurova CUDA Core Implementation
 * Basic CUDA operations and memory management
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef NEUROVA_HAVE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef NEUROVA_HAVE_CUFFT
#include <cufft.h>
#endif

namespace neurova {
namespace cuda {

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while(0)

// Device information structure
struct DeviceInfo {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
    bool unified_memory;
    bool concurrent_kernels;
};

// Get device count
__host__ int get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

// Get device info
__host__ DeviceInfo get_device_info(int device_id) {
    DeviceInfo info;
    info.device_id = device_id;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    strncpy(info.name, prop.name, 255);
    info.name[255] = '\0';
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    info.unified_memory = (prop.unifiedAddressing != 0);
    info.concurrent_kernels = (prop.concurrentKernels != 0);
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free_memory = free_mem;
    
    return info;
}

// Set current device
__host__ void set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

// Synchronize device
__host__ void synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Memory allocation
__host__ void* device_malloc(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

// Pinned memory allocation
__host__ void* host_malloc_pinned(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

// Unified memory allocation
__host__ void* unified_malloc(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    return ptr;
}

// Memory free
__host__ void device_free(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

__host__ void host_free_pinned(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

// Memory copy
__host__ void memcpy_host_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

__host__ void memcpy_device_to_host(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

__host__ void memcpy_device_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

// Async memory operations
__host__ void memcpy_async_h2d(void* dst, const void* src, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
}

__host__ void memcpy_async_d2h(void* dst, const void* src, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
}

// Stream management
__host__ cudaStream_t create_stream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

__host__ void destroy_stream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
}

__host__ void stream_synchronize(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Event management
__host__ cudaEvent_t create_event() {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    return event;
}

__host__ void destroy_event(cudaEvent_t event) {
    CUDA_CHECK(cudaEventDestroy(event));
}

__host__ void record_event(cudaEvent_t event, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(event, stream));
}

__host__ float elapsed_time(cudaEvent_t start, cudaEvent_t end) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

// ============================================================================
// Basic Kernels
// ============================================================================

// Fill kernel
template<typename T>
__global__ void fill_kernel(T* data, T value, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}

// Copy kernel
template<typename T>
__global__ void copy_kernel(T* dst, const T* src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

// Add kernel
template<typename T>
__global__ void add_kernel(T* dst, const T* a, const T* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = a[idx] + b[idx];
    }
}

// Multiply kernel
template<typename T>
__global__ void multiply_kernel(T* dst, const T* a, const T* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = a[idx] * b[idx];
    }
}

// Scale kernel
template<typename T>
__global__ void scale_kernel(T* data, T scale, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] *= scale;
    }
}

// Host functions for kernels
template<typename T>
void fill(T* data, T value, size_t count, cudaStream_t stream = 0) {
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    fill_kernel<<<grid_size, block_size, 0, stream>>>(data, value, count);
}

template<typename T>
void add(T* dst, const T* a, const T* b, size_t count, cudaStream_t stream = 0) {
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size, 0, stream>>>(dst, a, b, count);
}

template<typename T>
void multiply(T* dst, const T* a, const T* b, size_t count, cudaStream_t stream = 0) {
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    multiply_kernel<<<grid_size, block_size, 0, stream>>>(dst, a, b, count);
}

template<typename T>
void scale(T* data, T scale_val, size_t count, cudaStream_t stream = 0) {
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    scale_kernel<<<grid_size, block_size, 0, stream>>>(data, scale_val, count);
}

// Explicit instantiations
template void fill<float>(float*, float, size_t, cudaStream_t);
template void fill<double>(double*, double, size_t, cudaStream_t);
template void fill<int>(int*, int, size_t, cudaStream_t);
template void add<float>(float*, const float*, const float*, size_t, cudaStream_t);
template void add<double>(double*, const double*, const double*, size_t, cudaStream_t);
template void multiply<float>(float*, const float*, const float*, size_t, cudaStream_t);
template void multiply<double>(double*, const double*, const double*, size_t, cudaStream_t);
template void scale<float>(float*, float, size_t, cudaStream_t);
template void scale<double>(double*, double, size_t, cudaStream_t);

} // namespace cuda
} // namespace neurova
