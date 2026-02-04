/**
 * Neurova CUDA Filter Kernels
 * GPU-accelerated convolution and filtering
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace neurova {
namespace cuda {
namespace filters {

// Texture memory for fast 2D access
texture<unsigned char, 2, cudaReadModeElementType> tex_src;

// ============================================================================
// Convolution Kernels
// ============================================================================

// General 2D convolution kernel
__global__ void convolve2d_kernel(
    const unsigned char* src,
    float* dst,
    int width, int height, int step,
    const float* kernel, int kernel_size
) {
    extern __shared__ float shared_kernel[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int half = kernel_size / 2;
    int kernel_len = kernel_size * kernel_size;
    
    // Load kernel into shared memory
    int tid = ty * blockDim.x + tx;
    if (tid < kernel_len) {
        shared_kernel[tid] = kernel[tid];
    }
    __syncthreads();
    
    if (x < width && y < height) {
        float sum = 0;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int px = min(max(x + kx - half, 0), width - 1);
                int py = min(max(y + ky - half, 0), height - 1);
                sum += src[py * step + px] * shared_kernel[ky * kernel_size + kx];
            }
        }
        
        dst[y * width + x] = sum;
    }
}

// Separable convolution - horizontal pass
__global__ void separable_conv_h_kernel(
    const unsigned char* src,
    float* dst,
    int width, int height, int step,
    const float* kernel, int kernel_size
) {
    extern __shared__ float shared_kernel[];
    
    int tid = threadIdx.x;
    if (tid < kernel_size) {
        shared_kernel[tid] = kernel[tid];
    }
    __syncthreads();
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float sum = 0;
        
        for (int k = 0; k < kernel_size; k++) {
            int px = min(max(x + k - half, 0), width - 1);
            sum += src[y * step + px] * shared_kernel[k];
        }
        
        dst[y * width + x] = sum;
    }
}

// Separable convolution - vertical pass
__global__ void separable_conv_v_kernel(
    const float* src,
    unsigned char* dst,
    int width, int height, int step,
    const float* kernel, int kernel_size
) {
    extern __shared__ float shared_kernel[];
    
    int tid = threadIdx.x;
    if (tid < kernel_size) {
        shared_kernel[tid] = kernel[tid];
    }
    __syncthreads();
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float sum = 0;
        
        for (int k = 0; k < kernel_size; k++) {
            int py = min(max(y + k - half, 0), height - 1);
            sum += src[py * width + x] * shared_kernel[k];
        }
        
        dst[y * step + x] = (unsigned char)min(max(sum, 0.0f), 255.0f);
    }
}

// ============================================================================
// Bilateral Filter
// ============================================================================

__global__ void bilateral_filter_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int step,
    int kernel_size,
    float sigma_spatial,
    float sigma_range
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float center_val = src[y * step + x];
        
        float sum = 0;
        float weight_sum = 0;
        
        float spatial_coeff = -0.5f / (sigma_spatial * sigma_spatial);
        float range_coeff = -0.5f / (sigma_range * sigma_range);
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                
                float neighbor_val = src[py * step + px];
                
                // Spatial weight
                float spatial_dist = kx * kx + ky * ky;
                float spatial_weight = expf(spatial_dist * spatial_coeff);
                
                // Range weight
                float range_dist = (center_val - neighbor_val) * (center_val - neighbor_val);
                float range_weight = expf(range_dist * range_coeff);
                
                float weight = spatial_weight * range_weight;
                sum += neighbor_val * weight;
                weight_sum += weight;
            }
        }
        
        dst[y * step + x] = (unsigned char)(sum / weight_sum);
    }
}

// ============================================================================
// Median Filter
// ============================================================================

// Median filter using sorting network (efficient for small kernels)
__device__ void swap_if_greater(unsigned char& a, unsigned char& b) {
    if (a > b) {
        unsigned char temp = a;
        a = b;
        b = temp;
    }
}

__global__ void median_filter_3x3_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int step
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Load 3x3 neighborhood
        unsigned char v[9];
        int idx = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                v[idx++] = src[(y + ky) * step + (x + kx)];
            }
        }
        
        // Sorting network for 9 elements
        swap_if_greater(v[0], v[1]); swap_if_greater(v[3], v[4]); swap_if_greater(v[6], v[7]);
        swap_if_greater(v[1], v[2]); swap_if_greater(v[4], v[5]); swap_if_greater(v[7], v[8]);
        swap_if_greater(v[0], v[1]); swap_if_greater(v[3], v[4]); swap_if_greater(v[6], v[7]);
        swap_if_greater(v[0], v[3]); swap_if_greater(v[3], v[6]); swap_if_greater(v[0], v[3]);
        swap_if_greater(v[1], v[4]); swap_if_greater(v[4], v[7]); swap_if_greater(v[1], v[4]);
        swap_if_greater(v[2], v[5]); swap_if_greater(v[5], v[8]); swap_if_greater(v[2], v[5]);
        swap_if_greater(v[1], v[3]); swap_if_greater(v[5], v[7]);
        swap_if_greater(v[2], v[6]); swap_if_greater(v[4], v[6]); swap_if_greater(v[2], v[4]);
        swap_if_greater(v[2], v[3]); swap_if_greater(v[5], v[6]);
        swap_if_greater(v[3], v[4]); swap_if_greater(v[4], v[5]);
        
        dst[y * step + x] = v[4]; // Median
    } else if (x < width && y < height) {
        dst[y * step + x] = src[y * step + x];
    }
}

// General median filter using histogram
__global__ void median_filter_histogram_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int step,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        __shared__ unsigned int hist[256];
        
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        if (tid < 256) {
            hist[tid] = 0;
        }
        __syncthreads();
        
        int half = kernel_size / 2;
        int total = 0;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                atomicAdd(&hist[src[py * step + px]], 1);
                total++;
            }
        }
        __syncthreads();
        
        // Find median
        int median_pos = total / 2;
        int count = 0;
        unsigned char median_val = 0;
        for (int i = 0; i < 256; i++) {
            count += hist[i];
            if (count > median_pos) {
                median_val = i;
                break;
            }
        }
        
        dst[y * step + x] = median_val;
    }
}

// ============================================================================
// Laplacian Filter
// ============================================================================

__global__ void laplacian_kernel(
    const unsigned char* src,
    short* dst,
    int width, int height, int step,
    int kernel_type  // 0: 4-connected, 1: 8-connected
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        short result;
        
        if (kernel_type == 0) {
            // 4-connected: [0, 1, 0; 1, -4, 1; 0, 1, 0]
            result = -4 * src[y * step + x]
                   + src[(y-1) * step + x]
                   + src[(y+1) * step + x]
                   + src[y * step + (x-1)]
                   + src[y * step + (x+1)];
        } else {
            // 8-connected: [1, 1, 1; 1, -8, 1; 1, 1, 1]
            result = -8 * src[y * step + x]
                   + src[(y-1) * step + (x-1)]
                   + src[(y-1) * step + x]
                   + src[(y-1) * step + (x+1)]
                   + src[y * step + (x-1)]
                   + src[y * step + (x+1)]
                   + src[(y+1) * step + (x-1)]
                   + src[(y+1) * step + x]
                   + src[(y+1) * step + (x+1)];
        }
        
        dst[y * width + x] = result;
    }
}

// ============================================================================
// Sharpening
// ============================================================================

__global__ void unsharp_mask_kernel(
    const unsigned char* src,
    const unsigned char* blurred,
    unsigned char* dst,
    int width, int height, int step,
    float amount, float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * step + x;
        float original = src[idx];
        float blur = blurred[idx];
        float diff = original - blur;
        
        float result;
        if (fabsf(diff) > threshold) {
            result = original + amount * diff;
        } else {
            result = original;
        }
        
        dst[idx] = (unsigned char)min(max(result, 0.0f), 255.0f);
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void launch_convolve2d(
    const unsigned char* src, float* dst,
    int width, int height, int step,
    const float* kernel, int kernel_size,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    size_t shared_size = kernel_size * kernel_size * sizeof(float);
    convolve2d_kernel<<<grid, block, shared_size, stream>>>(
        src, dst, width, height, step, kernel, kernel_size
    );
}

void launch_bilateral_filter(
    const unsigned char* src, unsigned char* dst,
    int width, int height, int step,
    int kernel_size, float sigma_spatial, float sigma_range,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    bilateral_filter_kernel<<<grid, block, 0, stream>>>(
        src, dst, width, height, step, kernel_size, sigma_spatial, sigma_range
    );
}

void launch_median_filter_3x3(
    const unsigned char* src, unsigned char* dst,
    int width, int height, int step,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    median_filter_3x3_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, step);
}

} // namespace filters
} // namespace cuda
} // namespace neurova
