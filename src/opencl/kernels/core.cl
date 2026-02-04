/**
 * Neurova OpenCL Core Kernels
 * Basic GPU operations using OpenCL
 */

// ============================================================================
// Utility Functions
// ============================================================================

inline int clamp_int(int val, int min_val, int max_val) {
    return min(max(val, min_val), max_val);
}

inline float clamp_float(float val, float min_val, float max_val) {
    return fmin(fmax(val, min_val), max_val);
}

// ============================================================================
// Element-wise Operations
// ============================================================================

__kernel void fill_kernel(
    __global float* data,
    float value,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        data[idx] = value;
    }
}

__kernel void add_kernel(
    __global const float* a,
    __global const float* b,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = a[idx] + b[idx];
    }
}

__kernel void subtract_kernel(
    __global const float* a,
    __global const float* b,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = a[idx] - b[idx];
    }
}

__kernel void multiply_kernel(
    __global const float* a,
    __global const float* b,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = a[idx] * b[idx];
    }
}

__kernel void divide_kernel(
    __global const float* a,
    __global const float* b,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = a[idx] / b[idx];
    }
}

__kernel void scale_kernel(
    __global float* data,
    float scale,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        data[idx] *= scale;
    }
}

__kernel void add_scalar_kernel(
    __global float* data,
    float scalar,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        data[idx] += scalar;
    }
}

// ============================================================================
// Math Functions
// ============================================================================

__kernel void sqrt_kernel(
    __global const float* src,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = sqrt(src[idx]);
    }
}

__kernel void exp_kernel(
    __global const float* src,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = exp(src[idx]);
    }
}

__kernel void log_kernel(
    __global const float* src,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = log(src[idx]);
    }
}

__kernel void pow_kernel(
    __global const float* src,
    __global float* dst,
    float exponent,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = pow(src[idx], exponent);
    }
}

__kernel void abs_kernel(
    __global const float* src,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = fabs(src[idx]);
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

__kernel void reduce_sum_kernel(
    __global const float* src,
    __global float* dst,
    __local float* scratch,
    int count
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int group_size = get_local_size(0);
    
    // Load and add to local memory
    scratch[lid] = (gid < count) ? src[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction in local memory
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (lid == 0) {
        dst[get_group_id(0)] = scratch[0];
    }
}

__kernel void reduce_max_kernel(
    __global const float* src,
    __global float* dst,
    __local float* scratch,
    int count
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int group_size = get_local_size(0);
    
    scratch[lid] = (gid < count) ? src[gid] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        dst[get_group_id(0)] = scratch[0];
    }
}

__kernel void reduce_min_kernel(
    __global const float* src,
    __global float* dst,
    __local float* scratch,
    int count
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int group_size = get_local_size(0);
    
    scratch[lid] = (gid < count) ? src[gid] : INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmin(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        dst[get_group_id(0)] = scratch[0];
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

__kernel void transpose_kernel(
    __global const float* src,
    __global float* dst,
    int rows,
    int cols
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < rows && col < cols) {
        dst[col * rows + row] = src[row * cols + col];
    }
}

__kernel void gemm_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Tiled GEMM for better performance
#define TILE_SIZE 16

__kernel void gemm_tiled_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    __local float tile_A[TILE_SIZE][TILE_SIZE];
    __local float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = get_global_id(0);
    int col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    
    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        int tile_row = row;
        int tile_col = t * TILE_SIZE + local_col;
        
        if (tile_row < M && tile_col < K) {
            tile_A[local_row][local_col] = A[tile_row * K + tile_col];
        } else {
            tile_A[local_row][local_col] = 0.0f;
        }
        
        tile_row = t * TILE_SIZE + local_row;
        tile_col = col;
        
        if (tile_row < K && tile_col < N) {
            tile_B[local_row][local_col] = B[tile_row * N + tile_col];
        } else {
            tile_B[local_row][local_col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[local_row][k] * tile_B[k][local_col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ============================================================================
// Copy Operations
// ============================================================================

__kernel void copy_kernel(
    __global const float* src,
    __global float* dst,
    int count
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

__kernel void copy_strided_kernel(
    __global const float* src,
    __global float* dst,
    int count,
    int src_stride,
    int dst_stride
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx * dst_stride] = src[idx * src_stride];
    }
}

// ============================================================================
// Type Conversion
// ============================================================================

__kernel void float_to_uchar_kernel(
    __global const float* src,
    __global uchar* dst,
    int count,
    float scale
) {
    int idx = get_global_id(0);
    if (idx < count) {
        float val = src[idx] * scale;
        dst[idx] = (uchar)clamp_float(val, 0.0f, 255.0f);
    }
}

__kernel void uchar_to_float_kernel(
    __global const uchar* src,
    __global float* dst,
    int count,
    float scale
) {
    int idx = get_global_id(0);
    if (idx < count) {
        dst[idx] = (float)src[idx] * scale;
    }
}
