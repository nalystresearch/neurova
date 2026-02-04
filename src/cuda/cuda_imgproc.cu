/**
 * Neurova CUDA Image Processing Kernels
 * GPU-accelerated image processing operations
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace neurova {
namespace cuda {
namespace imgproc {

// ============================================================================
// Color Conversion Kernels
// ============================================================================

// RGB to Grayscale (BT.601 coefficients)
__global__ void rgb_to_gray_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int src_step, int dst_step
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int src_idx = y * src_step + x * 3;
        int dst_idx = y * dst_step + x;
        
        unsigned char r = src[src_idx + 0];
        unsigned char g = src[src_idx + 1];
        unsigned char b = src[src_idx + 2];
        
        // BT.601: Y = 0.299*R + 0.587*G + 0.114*B
        dst[dst_idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// BGR to RGB
__global__ void bgr_to_rgb_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int step
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * step + x * 3;
        dst[idx + 0] = src[idx + 2];  // R
        dst[idx + 1] = src[idx + 1];  // G
        dst[idx + 2] = src[idx + 0];  // B
    }
}

// RGB to HSV
__global__ void rgb_to_hsv_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int step
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * step + x * 3;
        
        float r = src[idx + 0] / 255.0f;
        float g = src[idx + 1] / 255.0f;
        float b = src[idx + 2] / 255.0f;
        
        float max_val = fmaxf(fmaxf(r, g), b);
        float min_val = fminf(fminf(r, g), b);
        float diff = max_val - min_val;
        
        float h = 0, s = 0, v = max_val;
        
        if (diff > 0) {
            s = diff / max_val;
            
            if (max_val == r) {
                h = 60.0f * fmodf((g - b) / diff, 6.0f);
            } else if (max_val == g) {
                h = 60.0f * ((b - r) / diff + 2.0f);
            } else {
                h = 60.0f * ((r - g) / diff + 4.0f);
            }
            
            if (h < 0) h += 360.0f;
        }
        
        dst[idx + 0] = (unsigned char)(h / 2.0f);     // H: 0-180
        dst[idx + 1] = (unsigned char)(s * 255.0f);    // S: 0-255
        dst[idx + 2] = (unsigned char)(v * 255.0f);    // V: 0-255
    }
}

// ============================================================================
// Filtering Kernels
// ============================================================================

// Gaussian blur (separable)
__constant__ float gaussian_kernel[32];

__global__ void gaussian_blur_horizontal_kernel(
    const unsigned char* src,
    float* dst,
    int width, int height, int step,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float sum = 0;
        
        for (int k = -half; k <= half; k++) {
            int px = min(max(x + k, 0), width - 1);
            sum += src[y * step + px] * gaussian_kernel[k + half];
        }
        
        dst[y * width + x] = sum;
    }
}

__global__ void gaussian_blur_vertical_kernel(
    const float* src,
    unsigned char* dst,
    int width, int height, int step,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float sum = 0;
        
        for (int k = -half; k <= half; k++) {
            int py = min(max(y + k, 0), height - 1);
            sum += src[py * width + x] * gaussian_kernel[k + half];
        }
        
        dst[y * step + x] = (unsigned char)min(max(sum, 0.0f), 255.0f);
    }
}

// Box filter
__global__ void box_filter_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int step,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float sum = 0;
        int count = 0;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                sum += src[py * step + px];
                count++;
            }
        }
        
        dst[y * step + x] = (unsigned char)(sum / count);
    }
}

// Sobel edge detection
__global__ void sobel_kernel(
    const unsigned char* src,
    unsigned char* dst_x,
    unsigned char* dst_y,
    int width, int height, int step
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
        int gx = -src[(y-1) * step + (x-1)] + src[(y-1) * step + (x+1)]
               - 2*src[y * step + (x-1)] + 2*src[y * step + (x+1)]
               - src[(y+1) * step + (x-1)] + src[(y+1) * step + (x+1)];
        
        // Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
        int gy = -src[(y-1) * step + (x-1)] - 2*src[(y-1) * step + x] - src[(y-1) * step + (x+1)]
               + src[(y+1) * step + (x-1)] + 2*src[(y+1) * step + x] + src[(y+1) * step + (x+1)];
        
        dst_x[y * step + x] = (unsigned char)min(abs(gx), 255);
        dst_y[y * step + x] = (unsigned char)min(abs(gy), 255);
    }
}

// Canny edge detection (non-maximum suppression)
__global__ void canny_nms_kernel(
    const float* magnitude,
    const float* direction,
    unsigned char* dst,
    int width, int height,
    float low_threshold, float high_threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        float mag = magnitude[idx];
        float angle = direction[idx];
        
        // Quantize angle to 0, 45, 90, 135 degrees
        float m1, m2;
        if (angle < 22.5f || angle >= 157.5f) {
            m1 = magnitude[idx - 1];
            m2 = magnitude[idx + 1];
        } else if (angle < 67.5f) {
            m1 = magnitude[(y-1) * width + (x+1)];
            m2 = magnitude[(y+1) * width + (x-1)];
        } else if (angle < 112.5f) {
            m1 = magnitude[(y-1) * width + x];
            m2 = magnitude[(y+1) * width + x];
        } else {
            m1 = magnitude[(y-1) * width + (x-1)];
            m2 = magnitude[(y+1) * width + (x+1)];
        }
        
        if (mag >= m1 && mag >= m2 && mag > low_threshold) {
            dst[idx] = (mag > high_threshold) ? 255 : 128;
        } else {
            dst[idx] = 0;
        }
    }
}

// ============================================================================
// Morphological Operations
// ============================================================================

__global__ void dilate_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int step,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        unsigned char max_val = 0;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                max_val = max(max_val, src[py * step + px]);
            }
        }
        
        dst[y * step + x] = max_val;
    }
}

__global__ void erode_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height, int step,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        unsigned char min_val = 255;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                min_val = min(min_val, src[py * step + px]);
            }
        }
        
        dst[y * step + x] = min_val;
    }
}

// ============================================================================
// Resize/Transform Kernels
// ============================================================================

__global__ void resize_bilinear_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int src_width, int src_height, int src_step,
    int dst_width, int dst_height, int dst_step,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_width && y < dst_height) {
        float scale_x = (float)src_width / dst_width;
        float scale_y = (float)src_height / dst_height;
        
        float src_x = x * scale_x;
        float src_y = y * scale_y;
        
        int x0 = (int)src_x;
        int y0 = (int)src_y;
        int x1 = min(x0 + 1, src_width - 1);
        int y1 = min(y0 + 1, src_height - 1);
        
        float wx = src_x - x0;
        float wy = src_y - y0;
        
        for (int c = 0; c < channels; c++) {
            float v00 = src[y0 * src_step + x0 * channels + c];
            float v01 = src[y0 * src_step + x1 * channels + c];
            float v10 = src[y1 * src_step + x0 * channels + c];
            float v11 = src[y1 * src_step + x1 * channels + c];
            
            float v = (1-wx) * (1-wy) * v00 + wx * (1-wy) * v01
                    + (1-wx) * wy * v10 + wx * wy * v11;
            
            dst[y * dst_step + x * channels + c] = (unsigned char)v;
        }
    }
}

// Warp affine
__global__ void warp_affine_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int src_width, int src_height, int src_step,
    int dst_width, int dst_height, int dst_step,
    int channels,
    float m00, float m01, float m02,
    float m10, float m11, float m12
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_width && y < dst_height) {
        float src_x = m00 * x + m01 * y + m02;
        float src_y = m10 * x + m11 * y + m12;
        
        if (src_x >= 0 && src_x < src_width - 1 && 
            src_y >= 0 && src_y < src_height - 1) {
            
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            float wx = src_x - x0;
            float wy = src_y - y0;
            
            for (int c = 0; c < channels; c++) {
                float v = (1-wx) * (1-wy) * src[y0 * src_step + x0 * channels + c]
                        + wx * (1-wy) * src[y0 * src_step + (x0+1) * channels + c]
                        + (1-wx) * wy * src[(y0+1) * src_step + x0 * channels + c]
                        + wx * wy * src[(y0+1) * src_step + (x0+1) * channels + c];
                
                dst[y * dst_step + x * channels + c] = (unsigned char)v;
            }
        } else {
            for (int c = 0; c < channels; c++) {
                dst[y * dst_step + x * channels + c] = 0;
            }
        }
    }
}

// ============================================================================
// Histogram Operations
// ============================================================================

__global__ void histogram_kernel(
    const unsigned char* src,
    unsigned int* hist,
    int width, int height, int step
) {
    __shared__ unsigned int local_hist[256];
    
    int tid = threadIdx.x;
    if (tid < 256) {
        local_hist[tid] = 0;
    }
    __syncthreads();
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        unsigned char val = src[y * step + x];
        atomicAdd(&local_hist[val], 1);
    }
    __syncthreads();
    
    if (tid < 256) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

__global__ void histogram_equalize_kernel(
    const unsigned char* src,
    unsigned char* dst,
    const unsigned char* lut,
    int width, int height, int step
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        dst[y * step + x] = lut[src[y * step + x]];
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void launch_rgb_to_gray(
    const unsigned char* src, unsigned char* dst,
    int width, int height, int src_step, int dst_step,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgb_to_gray_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, src_step, dst_step);
}

void launch_resize_bilinear(
    const unsigned char* src, unsigned char* dst,
    int src_width, int src_height, int src_step,
    int dst_width, int dst_height, int dst_step,
    int channels, cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    resize_bilinear_kernel<<<grid, block, 0, stream>>>(
        src, dst, src_width, src_height, src_step,
        dst_width, dst_height, dst_step, channels
    );
}

void launch_dilate(
    const unsigned char* src, unsigned char* dst,
    int width, int height, int step, int kernel_size,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    dilate_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, step, kernel_size);
}

void launch_erode(
    const unsigned char* src, unsigned char* dst,
    int width, int height, int step, int kernel_size,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    erode_kernel<<<grid, block, 0, stream>>>(src, dst, width, height, step, kernel_size);
}

} // namespace imgproc
} // namespace cuda
} // namespace neurova
