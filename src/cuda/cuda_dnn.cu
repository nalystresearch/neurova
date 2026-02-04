/**
 * Neurova CUDA DNN Kernels
 * GPU-accelerated deep neural network operations
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#ifdef NEUROVA_HAVE_CUDNN
#include <cudnn.h>
#endif

namespace neurova {
namespace cuda {
namespace dnn {

// ============================================================================
// Activation Functions
// ============================================================================

__global__ void relu_kernel(
    const float* input,
    float* output,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0.0f;
    }
}

__global__ void leaky_relu_kernel(
    const float* input,
    float* output,
    size_t count,
    float negative_slope
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val = input[idx];
        output[idx] = val > 0 ? val : val * negative_slope;
    }
}

__global__ void sigmoid_kernel(
    const float* input,
    float* output,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_kernel(
    const float* input,
    float* output,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        const float* in = input + batch_idx * num_classes;
        float* out = output + batch_idx * num_classes;
        
        // Find max for numerical stability
        float max_val = in[0];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, in[i]);
        }
        
        // Compute exp and sum
        float sum = 0;
        for (int i = 0; i < num_classes; i++) {
            out[i] = expf(in[i] - max_val);
            sum += out[i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            out[i] /= sum;
        }
    }
}

__global__ void gelu_kernel(
    const float* input,
    float* output,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

__global__ void silu_kernel(  // Swish
    const float* input,
    float* output,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// ============================================================================
// Batch Normalization
// ============================================================================

__global__ void batch_norm_forward_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    int batch_size,
    int channels,
    int spatial_size,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * spatial_size;
    
    if (idx < total) {
        int c = (idx / spatial_size) % channels;
        
        float mean = running_mean[c];
        float var = running_var[c];
        float g = gamma[c];
        float b = beta[c];
        
        float normalized = (input[idx] - mean) / sqrtf(var + epsilon);
        output[idx] = g * normalized + b;
    }
}

__global__ void instance_norm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon
) {
    int n = blockIdx.x;  // Batch index
    int c = blockIdx.y;  // Channel index
    
    if (n < batch_size && c < channels) {
        int spatial_size = height * width;
        const float* in = input + (n * channels + c) * spatial_size;
        float* out = output + (n * channels + c) * spatial_size;
        
        // Compute mean
        float sum = 0;
        for (int i = 0; i < spatial_size; i++) {
            sum += in[i];
        }
        float mean = sum / spatial_size;
        
        // Compute variance
        float var_sum = 0;
        for (int i = 0; i < spatial_size; i++) {
            float diff = in[i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / spatial_size;
        float inv_std = 1.0f / sqrtf(var + epsilon);
        
        // Normalize
        float g = gamma ? gamma[c] : 1.0f;
        float b = beta ? beta[c] : 0.0f;
        
        for (int i = 0; i < spatial_size; i++) {
            out[i] = g * (in[i] - mean) * inv_std + b;
        }
    }
}

// ============================================================================
// Pooling Operations
// ============================================================================

__global__ void max_pool2d_kernel(
    const float* input,
    float* output,
    int* indices,
    int batch_size,
    int channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (x < out_width && y < out_height && n < batch_size) {
        int in_y_start = y * stride_h - pad_h;
        int in_x_start = x * stride_w - pad_w;
        
        float max_val = -INFINITY;
        int max_idx = 0;
        
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_y = in_y_start + ky;
                int in_x = in_x_start + kx;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int in_idx = ((n * channels + c) * in_height + in_y) * in_width + in_x;
                    float val = input[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = in_idx;
                    }
                }
            }
        }
        
        int out_idx = ((n * channels + c) * out_height + y) * out_width + x;
        output[out_idx] = max_val;
        if (indices) {
            indices[out_idx] = max_idx;
        }
    }
}

__global__ void avg_pool2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (x < out_width && y < out_height && n < batch_size) {
        int in_y_start = y * stride_h - pad_h;
        int in_x_start = x * stride_w - pad_w;
        
        float sum = 0;
        int count = 0;
        
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_y = in_y_start + ky;
                int in_x = in_x_start + kx;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int in_idx = ((n * channels + c) * in_height + in_y) * in_width + in_x;
                    sum += input[in_idx];
                    count++;
                }
            }
        }
        
        int out_idx = ((n * channels + c) * out_height + y) * out_width + x;
        output[out_idx] = sum / count;
    }
}

__global__ void global_avg_pool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    
    if (n < batch_size && c < channels) {
        const float* in = input + (n * channels + c) * spatial_size;
        
        float sum = 0;
        for (int i = 0; i < spatial_size; i++) {
            sum += in[i];
        }
        
        output[n * channels + c] = sum / spatial_size;
    }
}

// ============================================================================
// Upsampling
// ============================================================================

__global__ void upsample_nearest_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height, int in_width,
    int out_height, int out_width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (x < out_width && y < out_height && n < batch_size) {
        float scale_h = (float)in_height / out_height;
        float scale_w = (float)in_width / out_width;
        
        int in_y = (int)(y * scale_h);
        int in_x = (int)(x * scale_w);
        
        int in_idx = ((n * channels + c) * in_height + in_y) * in_width + in_x;
        int out_idx = ((n * channels + c) * out_height + y) * out_width + x;
        
        output[out_idx] = input[in_idx];
    }
}

__global__ void upsample_bilinear_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height, int in_width,
    int out_height, int out_width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (x < out_width && y < out_height && n < batch_size) {
        float scale_h = (float)(in_height - 1) / (out_height - 1);
        float scale_w = (float)(in_width - 1) / (out_width - 1);
        
        float in_y = y * scale_h;
        float in_x = x * scale_w;
        
        int y0 = (int)in_y;
        int x0 = (int)in_x;
        int y1 = min(y0 + 1, in_height - 1);
        int x1 = min(x0 + 1, in_width - 1);
        
        float wy = in_y - y0;
        float wx = in_x - x0;
        
        const float* in = input + (n * channels + c) * in_height * in_width;
        
        float val = (1-wy) * (1-wx) * in[y0 * in_width + x0]
                  + (1-wy) * wx * in[y0 * in_width + x1]
                  + wy * (1-wx) * in[y1 * in_width + x0]
                  + wy * wx * in[y1 * in_width + x1];
        
        int out_idx = ((n * channels + c) * out_height + y) * out_width + x;
        output[out_idx] = val;
    }
}

// ============================================================================
// Dropout
// ============================================================================

__global__ void dropout_kernel(
    const float* input,
    float* output,
    const float* random,
    size_t count,
    float prob,
    float scale
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = random[idx] > prob ? input[idx] * scale : 0.0f;
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

__global__ void cross_entropy_loss_kernel(
    const float* logits,
    const int* targets,
    float* losses,
    int batch_size,
    int num_classes
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n < batch_size) {
        const float* logit = logits + n * num_classes;
        int target = targets[n];
        
        // Compute log softmax
        float max_val = logit[0];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, logit[i]);
        }
        
        float sum = 0;
        for (int i = 0; i < num_classes; i++) {
            sum += expf(logit[i] - max_val);
        }
        float log_sum = logf(sum) + max_val;
        
        losses[n] = log_sum - logit[target];
    }
}

__global__ void mse_loss_kernel(
    const float* predictions,
    const float* targets,
    float* losses,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float diff = predictions[idx] - targets[idx];
        losses[idx] = diff * diff;
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void launch_relu(const float* input, float* output, size_t count, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    relu_kernel<<<grid_size, block_size, 0, stream>>>(input, output, count);
}

void launch_sigmoid(const float* input, float* output, size_t count, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(input, output, count);
}

void launch_softmax(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream) {
    softmax_kernel<<<batch_size, 1, 0, stream>>>(input, output, batch_size, num_classes);
}

void launch_batch_norm(
    const float* input, float* output,
    const float* gamma, const float* beta,
    const float* mean, const float* var,
    int batch_size, int channels, int spatial_size,
    float epsilon, cudaStream_t stream
) {
    int total = batch_size * channels * spatial_size;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    batch_norm_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, gamma, beta, mean, var,
        batch_size, channels, spatial_size, epsilon
    );
}

void launch_max_pool2d(
    const float* input, float* output, int* indices,
    int batch_size, int channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        batch_size * channels
    );
    max_pool2d_kernel<<<grid, block, 0, stream>>>(
        input, output, indices,
        batch_size, channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );
}

} // namespace dnn
} // namespace cuda
} // namespace neurova
