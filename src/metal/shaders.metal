/**
 * Neurova Metal Shaders
 * GPU compute kernels for Apple platforms
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Core Operations
// ============================================================================

kernel void fill_kernel(
    device float* data [[buffer(0)]],
    constant float& value [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    data[idx] = value;
}

kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* dst [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    dst[idx] = a[idx] + b[idx];
}

kernel void multiply_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* dst [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    dst[idx] = a[idx] * b[idx];
}

kernel void scale_kernel(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    data[idx] *= scale;
}

// ============================================================================
// Color Conversion
// ============================================================================

kernel void rgb_to_gray_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& src_step [[buffer(4)]],
    constant int& dst_step [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int src_idx = gid.y * src_step + gid.x * 3;
    int dst_idx = gid.y * dst_step + gid.x;
    
    uchar r = src[src_idx + 0];
    uchar g = src[src_idx + 1];
    uchar b = src[src_idx + 2];
    
    dst[dst_idx] = uchar(0.299f * float(r) + 0.587f * float(g) + 0.114f * float(b));
}

kernel void bgr_to_rgb_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& step [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int idx = gid.y * step + gid.x * 3;
    dst[idx + 0] = src[idx + 2];
    dst[idx + 1] = src[idx + 1];
    dst[idx + 2] = src[idx + 0];
}

kernel void rgb_to_hsv_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& step [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int idx = gid.y * step + gid.x * 3;
    
    float r = src[idx + 0] / 255.0f;
    float g = src[idx + 1] / 255.0f;
    float b = src[idx + 2] / 255.0f;
    
    float max_val = max(max(r, g), b);
    float min_val = min(min(r, g), b);
    float diff = max_val - min_val;
    
    float h = 0, s = 0, v = max_val;
    
    if (diff > 0) {
        s = diff / max_val;
        
        if (max_val == r) {
            h = 60.0f * fmod((g - b) / diff, 6.0f);
        } else if (max_val == g) {
            h = 60.0f * ((b - r) / diff + 2.0f);
        } else {
            h = 60.0f * ((r - g) / diff + 4.0f);
        }
        
        if (h < 0) h += 360.0f;
    }
    
    dst[idx + 0] = uchar(h / 2.0f);
    dst[idx + 1] = uchar(s * 255.0f);
    dst[idx + 2] = uchar(v * 255.0f);
}

// ============================================================================
// Filtering
// ============================================================================

kernel void box_filter_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& step [[buffer(4)]],
    constant int& kernel_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int half = kernel_size / 2;
    float sum = 0;
    int count = 0;
    
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int px = clamp(int(gid.x) + kx, 0, width - 1);
            int py = clamp(int(gid.y) + ky, 0, height - 1);
            sum += src[py * step + px];
            count++;
        }
    }
    
    dst[gid.y * step + gid.x] = uchar(sum / count);
}

kernel void gaussian_blur_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant float* weights [[buffer(2)]],
    constant int& width [[buffer(3)]],
    constant int& height [[buffer(4)]],
    constant int& step [[buffer(5)]],
    constant int& kernel_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int half = kernel_size / 2;
    float sum = 0;
    
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int px = clamp(int(gid.x) + kx, 0, width - 1);
            int py = clamp(int(gid.y) + ky, 0, height - 1);
            int kidx = (ky + half) * kernel_size + (kx + half);
            sum += src[py * step + px] * weights[kidx];
        }
    }
    
    dst[gid.y * step + gid.x] = uchar(clamp(sum, 0.0f, 255.0f));
}

kernel void sobel_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst_x [[buffer(1)]],
    device uchar* dst_y [[buffer(2)]],
    constant int& width [[buffer(3)]],
    constant int& height [[buffer(4)]],
    constant int& step [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -src[(y-1) * step + (x-1)] + src[(y-1) * step + (x+1)]
               - 2*src[y * step + (x-1)] + 2*src[y * step + (x+1)]
               - src[(y+1) * step + (x-1)] + src[(y+1) * step + (x+1)];
        
        int gy = -src[(y-1) * step + (x-1)] - 2*src[(y-1) * step + x] - src[(y-1) * step + (x+1)]
               + src[(y+1) * step + (x-1)] + 2*src[(y+1) * step + x] + src[(y+1) * step + (x+1)];
        
        dst_x[y * step + x] = uchar(min(abs(gx), 255));
        dst_y[y * step + x] = uchar(min(abs(gy), 255));
    }
}

kernel void bilateral_filter_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& step [[buffer(4)]],
    constant int& kernel_size [[buffer(5)]],
    constant float& sigma_spatial [[buffer(6)]],
    constant float& sigma_range [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int half = kernel_size / 2;
    float center_val = src[gid.y * step + gid.x];
    
    float sum = 0;
    float weight_sum = 0;
    
    float spatial_coeff = -0.5f / (sigma_spatial * sigma_spatial);
    float range_coeff = -0.5f / (sigma_range * sigma_range);
    
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int px = clamp(int(gid.x) + kx, 0, width - 1);
            int py = clamp(int(gid.y) + ky, 0, height - 1);
            
            float neighbor_val = src[py * step + px];
            
            float spatial_dist = kx * kx + ky * ky;
            float spatial_weight = exp(spatial_dist * spatial_coeff);
            
            float range_dist = (center_val - neighbor_val) * (center_val - neighbor_val);
            float range_weight = exp(range_dist * range_coeff);
            
            float weight = spatial_weight * range_weight;
            sum += neighbor_val * weight;
            weight_sum += weight;
        }
    }
    
    dst[gid.y * step + gid.x] = uchar(sum / weight_sum);
}

// ============================================================================
// Morphological Operations
// ============================================================================

kernel void dilate_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& step [[buffer(4)]],
    constant int& kernel_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int half = kernel_size / 2;
    uchar max_val = 0;
    
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int px = clamp(int(gid.x) + kx, 0, width - 1);
            int py = clamp(int(gid.y) + ky, 0, height - 1);
            max_val = max(max_val, src[py * step + px]);
        }
    }
    
    dst[gid.y * step + gid.x] = max_val;
}

kernel void erode_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& step [[buffer(4)]],
    constant int& kernel_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int half = kernel_size / 2;
    uchar min_val = 255;
    
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int px = clamp(int(gid.x) + kx, 0, width - 1);
            int py = clamp(int(gid.y) + ky, 0, height - 1);
            min_val = min(min_val, src[py * step + px]);
        }
    }
    
    dst[gid.y * step + gid.x] = min_val;
}

// ============================================================================
// Geometric Transforms
// ============================================================================

kernel void resize_bilinear_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& src_width [[buffer(2)]],
    constant int& src_height [[buffer(3)]],
    constant int& src_step [[buffer(4)]],
    constant int& dst_width [[buffer(5)]],
    constant int& dst_height [[buffer(6)]],
    constant int& dst_step [[buffer(7)]],
    constant int& channels [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(dst_width) || gid.y >= uint(dst_height)) return;
    
    float scale_x = float(src_width) / dst_width;
    float scale_y = float(src_height) / dst_height;
    
    float src_x = gid.x * scale_x;
    float src_y = gid.y * scale_y;
    
    int x0 = int(src_x);
    int y0 = int(src_y);
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
        
        dst[gid.y * dst_step + gid.x * channels + c] = uchar(v);
    }
}

kernel void warp_affine_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant int& src_width [[buffer(2)]],
    constant int& src_height [[buffer(3)]],
    constant int& src_step [[buffer(4)]],
    constant int& dst_width [[buffer(5)]],
    constant int& dst_height [[buffer(6)]],
    constant int& dst_step [[buffer(7)]],
    constant int& channels [[buffer(8)]],
    constant float& m00 [[buffer(9)]],
    constant float& m01 [[buffer(10)]],
    constant float& m02 [[buffer(11)]],
    constant float& m10 [[buffer(12)]],
    constant float& m11 [[buffer(13)]],
    constant float& m12 [[buffer(14)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(dst_width) || gid.y >= uint(dst_height)) return;
    
    float src_x = m00 * gid.x + m01 * gid.y + m02;
    float src_y = m10 * gid.x + m11 * gid.y + m12;
    
    if (src_x >= 0 && src_x < src_width - 1 && 
        src_y >= 0 && src_y < src_height - 1) {
        
        int x0 = int(src_x);
        int y0 = int(src_y);
        float wx = src_x - x0;
        float wy = src_y - y0;
        
        for (int c = 0; c < channels; c++) {
            float v = (1-wx) * (1-wy) * src[y0 * src_step + x0 * channels + c]
                    + wx * (1-wy) * src[y0 * src_step + (x0+1) * channels + c]
                    + (1-wx) * wy * src[(y0+1) * src_step + x0 * channels + c]
                    + wx * wy * src[(y0+1) * src_step + (x0+1) * channels + c];
            
            dst[gid.y * dst_step + gid.x * channels + c] = uchar(v);
        }
    } else {
        for (int c = 0; c < channels; c++) {
            dst[gid.y * dst_step + gid.x * channels + c] = 0;
        }
    }
}

// ============================================================================
// DNN Operations
// ============================================================================

kernel void relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    output[idx] = max(input[idx], 0.0f);
}

kernel void sigmoid_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    output[idx] = 1.0f / (1.0f + exp(-input[idx]));
}

kernel void tanh_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    output[idx] = tanh(input[idx]);
}

kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& num_classes [[buffer(2)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    device const float* in = input + batch_idx * num_classes;
    device float* out = output + batch_idx * num_classes;
    
    float max_val = in[0];
    for (int i = 1; i < num_classes; i++) {
        max_val = max(max_val, in[i]);
    }
    
    float sum = 0;
    for (int i = 0; i < num_classes; i++) {
        out[i] = exp(in[i] - max_val);
        sum += out[i];
    }
    
    for (int i = 0; i < num_classes; i++) {
        out[i] /= sum;
    }
}

// ============================================================================
// Histogram
// ============================================================================

kernel void histogram_kernel(
    device const uchar* src [[buffer(0)]],
    device atomic_uint* hist [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& step [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    uchar val = src[gid.y * step + gid.x];
    atomic_fetch_add_explicit(&hist[val], 1, memory_order_relaxed);
}

kernel void histogram_equalize_kernel(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    device const uchar* lut [[buffer(2)]],
    constant int& width [[buffer(3)]],
    constant int& height [[buffer(4)]],
    constant int& step [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int idx = gid.y * step + gid.x;
    dst[idx] = lut[src[idx]];
}
