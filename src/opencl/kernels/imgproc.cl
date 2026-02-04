/**
 * Neurova OpenCL Image Processing Kernels
 * GPU-accelerated image operations
 */

// ============================================================================
// Color Conversion
// ============================================================================

__kernel void rgb_to_gray_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int src_step,
    int dst_step
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int src_idx = y * src_step + x * 3;
        int dst_idx = y * dst_step + x;
        
        uchar r = src[src_idx + 0];
        uchar g = src[src_idx + 1];
        uchar b = src[src_idx + 2];
        
        // BT.601 coefficients
        dst[dst_idx] = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

__kernel void bgr_to_rgb_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int idx = y * step + x * 3;
        dst[idx + 0] = src[idx + 2];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = src[idx + 0];
    }
}

__kernel void rgb_to_hsv_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int idx = y * step + x * 3;
        
        float r = src[idx + 0] / 255.0f;
        float g = src[idx + 1] / 255.0f;
        float b = src[idx + 2] / 255.0f;
        
        float max_val = fmax(fmax(r, g), b);
        float min_val = fmin(fmin(r, g), b);
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
        
        dst[idx + 0] = (uchar)(h / 2.0f);
        dst[idx + 1] = (uchar)(s * 255.0f);
        dst[idx + 2] = (uchar)(v * 255.0f);
    }
}

__kernel void rgba_to_rgb_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int src_idx = (y * width + x) * 4;
        int dst_idx = (y * width + x) * 3;
        
        dst[dst_idx + 0] = src[src_idx + 0];
        dst[dst_idx + 1] = src[src_idx + 1];
        dst[dst_idx + 2] = src[src_idx + 2];
    }
}

// ============================================================================
// Filtering
// ============================================================================

__kernel void box_filter_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    int kernel_size
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float sum = 0;
        int count = 0;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                sum += src[py * step + px];
                count++;
            }
        }
        
        dst[y * step + x] = (uchar)(sum / count);
    }
}

__kernel void gaussian_blur_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    __constant float* kernel_weights,
    int kernel_size
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float sum = 0;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                int kidx = (ky + half) * kernel_size + (kx + half);
                sum += src[py * step + px] * kernel_weights[kidx];
            }
        }
        
        dst[y * step + x] = (uchar)clamp(sum, 0.0f, 255.0f);
    }
}

__kernel void sobel_kernel(
    __global const uchar* src,
    __global uchar* dst_x,
    __global uchar* dst_y,
    int width,
    int height,
    int step
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -src[(y-1) * step + (x-1)] + src[(y-1) * step + (x+1)]
               - 2*src[y * step + (x-1)] + 2*src[y * step + (x+1)]
               - src[(y+1) * step + (x-1)] + src[(y+1) * step + (x+1)];
        
        int gy = -src[(y-1) * step + (x-1)] - 2*src[(y-1) * step + x] - src[(y-1) * step + (x+1)]
               + src[(y+1) * step + (x-1)] + 2*src[(y+1) * step + x] + src[(y+1) * step + (x+1)];
        
        dst_x[y * step + x] = (uchar)min(abs(gx), 255);
        dst_y[y * step + x] = (uchar)min(abs(gy), 255);
    }
}

__kernel void laplacian_kernel(
    __global const uchar* src,
    __global short* dst,
    int width,
    int height,
    int step
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        short result = -4 * src[y * step + x]
                     + src[(y-1) * step + x]
                     + src[(y+1) * step + x]
                     + src[y * step + (x-1)]
                     + src[y * step + (x+1)];
        
        dst[y * width + x] = result;
    }
}

__kernel void bilateral_filter_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    int kernel_size,
    float sigma_spatial,
    float sigma_range
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        float center_val = src[y * step + x];
        
        float sum = 0;
        float weight_sum = 0;
        
        float spatial_coeff = -0.5f / (sigma_spatial * sigma_spatial);
        float range_coeff = -0.5f / (sigma_range * sigma_range);
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                
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
        
        dst[y * step + x] = (uchar)(sum / weight_sum);
    }
}

// ============================================================================
// Morphological Operations
// ============================================================================

__kernel void dilate_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    int kernel_size
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        uchar max_val = 0;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                max_val = max(max_val, src[py * step + px]);
            }
        }
        
        dst[y * step + x] = max_val;
    }
}

__kernel void erode_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    int kernel_size
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int half = kernel_size / 2;
        uchar min_val = 255;
        
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = clamp(x + kx, 0, width - 1);
                int py = clamp(y + ky, 0, height - 1);
                min_val = min(min_val, src[py * step + px]);
            }
        }
        
        dst[y * step + x] = min_val;
    }
}

// ============================================================================
// Geometric Transforms
// ============================================================================

__kernel void resize_bilinear_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int src_width,
    int src_height,
    int src_step,
    int dst_width,
    int dst_height,
    int dst_step,
    int channels
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
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
            
            dst[y * dst_step + x * channels + c] = (uchar)v;
        }
    }
}

__kernel void resize_nearest_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int src_width,
    int src_height,
    int src_step,
    int dst_width,
    int dst_height,
    int dst_step,
    int channels
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < dst_width && y < dst_height) {
        float scale_x = (float)src_width / dst_width;
        float scale_y = (float)src_height / dst_height;
        
        int src_x = (int)(x * scale_x);
        int src_y = (int)(y * scale_y);
        
        for (int c = 0; c < channels; c++) {
            dst[y * dst_step + x * channels + c] = 
                src[src_y * src_step + src_x * channels + c];
        }
    }
}

__kernel void warp_affine_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int src_width,
    int src_height,
    int src_step,
    int dst_width,
    int dst_height,
    int dst_step,
    int channels,
    float m00, float m01, float m02,
    float m10, float m11, float m12
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
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
                
                dst[y * dst_step + x * channels + c] = (uchar)v;
            }
        } else {
            for (int c = 0; c < channels; c++) {
                dst[y * dst_step + x * channels + c] = 0;
            }
        }
    }
}

__kernel void flip_horizontal_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    int channels
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int src_x = width - 1 - x;
        for (int c = 0; c < channels; c++) {
            dst[y * step + x * channels + c] = src[y * step + src_x * channels + c];
        }
    }
}

__kernel void flip_vertical_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    int channels
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int src_y = height - 1 - y;
        for (int c = 0; c < channels; c++) {
            dst[y * step + x * channels + c] = src[src_y * step + x * channels + c];
        }
    }
}

// ============================================================================
// Threshold Operations
// ============================================================================

__kernel void threshold_binary_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    uchar thresh,
    uchar max_val
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int idx = y * step + x;
        dst[idx] = src[idx] > thresh ? max_val : 0;
    }
}

__kernel void threshold_otsu_kernel(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    int step,
    uchar thresh,
    uchar max_val
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int idx = y * step + x;
        dst[idx] = src[idx] > thresh ? max_val : 0;
    }
}

// ============================================================================
// Histogram
// ============================================================================

__kernel void histogram_kernel(
    __global const uchar* src,
    __global uint* hist,
    int width,
    int height,
    int step
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        uchar val = src[y * step + x];
        atomic_inc(&hist[val]);
    }
}

__kernel void histogram_equalize_kernel(
    __global const uchar* src,
    __global uchar* dst,
    __global const uchar* lut,
    int width,
    int height,
    int step
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int idx = y * step + x;
        dst[idx] = lut[src[idx]];
    }
}
