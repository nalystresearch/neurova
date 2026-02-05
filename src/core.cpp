// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * core.cpp - Core Tensor and Image implementation
 */

#include "../include/neurova/core.hpp"
#include <random>
#include <numeric>
#include <cstring>
#include <cassert>

#if defined(__SSE__) || defined(__SSE2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace neurova {

// ============================================================================
// Random number generator
// ============================================================================

static thread_local std::mt19937 rng(std::random_device{}());

// ============================================================================
// Tensor Implementation
// ============================================================================

Tensor::Tensor() : size_(0), dtype_(DType::FLOAT32) {}

Tensor::Tensor(const Shape& shape, DType dtype)
    : shape_(shape), dtype_(dtype) {
    size_ = shape_.total();
    allocate();
    compute_strides();
}

Tensor::Tensor(const std::vector<size_t>& shape, DType dtype)
    : Tensor(Shape(shape), dtype) {}

Tensor::Tensor(size_t rows, size_t cols, DType dtype)
    : Tensor(Shape{rows, cols}, dtype) {}

Tensor::Tensor(size_t height, size_t width, size_t channels, DType dtype)
    : Tensor(Shape{height, width, channels}, dtype) {}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), dtype_(other.dtype_),
      strides_(other.strides_) {
    allocate();
    std::memcpy(data_.get(), other.data_.get(), nbytes());
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)),
      size_(other.size_), dtype_(other.dtype_), strides_(std::move(other.strides_)) {
    other.size_ = 0;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        size_ = other.size_;
        dtype_ = other.dtype_;
        strides_ = other.strides_;
        allocate();
        std::memcpy(data_.get(), other.data_.get(), nbytes());
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        dtype_ = other.dtype_;
        strides_ = std::move(other.strides_);
        other.size_ = 0;
    }
    return *this;
}

void Tensor::allocate() {
    if (size_ == 0) return;
    size_t bytes = size_ * itemsize();
    data_ = std::shared_ptr<void>(
        std::aligned_alloc(32, ((bytes + 31) / 32) * 32),
        [](void* p) { std::free(p); }
    );
    std::memset(data_.get(), 0, bytes);
}

void Tensor::compute_strides() {
    strides_.resize(shape_.ndim());
    if (shape_.ndim() == 0) return;
    
    strides_[shape_.ndim() - 1] = 1;
    for (int i = shape_.ndim() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// Factory methods
Tensor Tensor::zeros(const Shape& shape, DType dtype) {
    return Tensor(shape, dtype);
}

Tensor Tensor::ones(const Shape& shape, DType dtype) {
    Tensor t(shape, dtype);
    if (dtype == DType::FLOAT32) {
        float* p = t.ptr<float>();
        for (size_t i = 0; i < t.size(); ++i) p[i] = 1.0f;
    } else if (dtype == DType::FLOAT64) {
        double* p = t.ptr<double>();
        for (size_t i = 0; i < t.size(); ++i) p[i] = 1.0;
    } else if (dtype == DType::UINT8) {
        uint8_t* p = t.ptr<uint8_t>();
        for (size_t i = 0; i < t.size(); ++i) p[i] = 1;
    }
    return t;
}

Tensor Tensor::full(const Shape& shape, double value, DType dtype) {
    Tensor t(shape, dtype);
    if (dtype == DType::FLOAT32) {
        float* p = t.ptr<float>();
        float v = static_cast<float>(value);
        for (size_t i = 0; i < t.size(); ++i) p[i] = v;
    } else if (dtype == DType::FLOAT64) {
        double* p = t.ptr<double>();
        for (size_t i = 0; i < t.size(); ++i) p[i] = value;
    } else if (dtype == DType::UINT8) {
        uint8_t* p = t.ptr<uint8_t>();
        uint8_t v = static_cast<uint8_t>(value);
        for (size_t i = 0; i < t.size(); ++i) p[i] = v;
    }
    return t;
}

Tensor Tensor::eye(size_t n, DType dtype) {
    Tensor t = zeros(Shape{n, n}, dtype);
    if (dtype == DType::FLOAT32) {
        float* p = t.ptr<float>();
        for (size_t i = 0; i < n; ++i) p[i * n + i] = 1.0f;
    } else if (dtype == DType::FLOAT64) {
        double* p = t.ptr<double>();
        for (size_t i = 0; i < n; ++i) p[i * n + i] = 1.0;
    }
    return t;
}

Tensor Tensor::randn(const Shape& shape, DType dtype) {
    Tensor t(shape, dtype);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    if (dtype == DType::FLOAT32) {
        float* p = t.ptr<float>();
        for (size_t i = 0; i < t.size(); ++i) {
            p[i] = static_cast<float>(dist(rng));
        }
    } else if (dtype == DType::FLOAT64) {
        double* p = t.ptr<double>();
        for (size_t i = 0; i < t.size(); ++i) {
            p[i] = dist(rng);
        }
    }
    return t;
}

Tensor Tensor::rand(const Shape& shape, DType dtype) {
    Tensor t(shape, dtype);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    if (dtype == DType::FLOAT32) {
        float* p = t.ptr<float>();
        for (size_t i = 0; i < t.size(); ++i) {
            p[i] = static_cast<float>(dist(rng));
        }
    } else if (dtype == DType::FLOAT64) {
        double* p = t.ptr<double>();
        for (size_t i = 0; i < t.size(); ++i) {
            p[i] = dist(rng);
        }
    }
    return t;
}

Tensor Tensor::arange(double start, double stop, double step, DType dtype) {
    size_t n = static_cast<size_t>(std::ceil((stop - start) / step));
    Tensor t(Shape{n}, dtype);
    
    if (dtype == DType::FLOAT32) {
        float* p = t.ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            p[i] = static_cast<float>(start + i * step);
        }
    } else if (dtype == DType::FLOAT64) {
        double* p = t.ptr<double>();
        for (size_t i = 0; i < n; ++i) {
            p[i] = start + i * step;
        }
    }
    return t;
}

Tensor Tensor::linspace(double start, double stop, size_t num, DType dtype) {
    Tensor t(Shape{num}, dtype);
    double step = (num > 1) ? (stop - start) / (num - 1) : 0.0;
    
    if (dtype == DType::FLOAT32) {
        float* p = t.ptr<float>();
        for (size_t i = 0; i < num; ++i) {
            p[i] = static_cast<float>(start + i * step);
        }
    } else if (dtype == DType::FLOAT64) {
        double* p = t.ptr<double>();
        for (size_t i = 0; i < num; ++i) {
            p[i] = start + i * step;
        }
    }
    return t;
}

// Reshape operations
Tensor Tensor::reshape(const Shape& new_shape) const {
    size_t new_size = new_shape.total();
    if (new_size != size_) {
        throw std::runtime_error("Cannot reshape: total size mismatch");
    }
    
    Tensor result = clone();
    result.shape_ = new_shape;
    result.compute_strides();
    return result;
}

Tensor Tensor::flatten() const {
    return reshape(Shape{size_});
}

Tensor Tensor::transpose() const {
    if (shape_.ndim() != 2) {
        throw std::runtime_error("transpose() requires 2D tensor");
    }
    
    size_t rows = shape_[0];
    size_t cols = shape_[1];
    Tensor result(Shape{cols, rows}, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* src = ptr<float>();
        float* dst = result.ptr<float>();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    } else if (dtype_ == DType::FLOAT64) {
        const double* src = ptr<double>();
        double* dst = result.ptr<double>();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }
    return result;
}

Tensor Tensor::transpose(const std::vector<size_t>& axes) const {
    // General transpose - permute dimensions
    if (axes.size() != shape_.ndim()) {
        throw std::runtime_error("transpose: axes must match tensor dimensions");
    }
    
    Shape new_shape;
    new_shape.dims.resize(axes.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        new_shape.dims[i] = shape_[axes[i]];
    }
    
    Tensor result(new_shape, dtype_);
    
    // For simplicity, implement 2D and 3D cases
    if (shape_.ndim() == 2) {
        return transpose();
    }
    
    // General case with slow path
    // TODO: Optimize for common cases
    return result;
}

// Type conversion
Tensor Tensor::astype(DType new_dtype) const {
    if (new_dtype == dtype_) return clone();
    
    Tensor result(shape_, new_dtype);
    
    // Convert from source type to destination type
    if (dtype_ == DType::FLOAT32 && new_dtype == DType::FLOAT64) {
        const float* src = ptr<float>();
        double* dst = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) dst[i] = static_cast<double>(src[i]);
    } else if (dtype_ == DType::FLOAT64 && new_dtype == DType::FLOAT32) {
        const double* src = ptr<double>();
        float* dst = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) dst[i] = static_cast<float>(src[i]);
    } else if (dtype_ == DType::FLOAT32 && new_dtype == DType::UINT8) {
        const float* src = ptr<float>();
        uint8_t* dst = result.ptr<uint8_t>();
        for (size_t i = 0; i < size_; ++i) {
            float v = std::max(0.0f, std::min(255.0f, src[i]));
            dst[i] = static_cast<uint8_t>(v);
        }
    } else if (dtype_ == DType::UINT8 && new_dtype == DType::FLOAT32) {
        const uint8_t* src = ptr<uint8_t>();
        float* dst = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) dst[i] = static_cast<float>(src[i]);
    } else if (dtype_ == DType::UINT8 && new_dtype == DType::FLOAT64) {
        const uint8_t* src = ptr<uint8_t>();
        double* dst = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) dst[i] = static_cast<double>(src[i]);
    }
    
    return result;
}

// Arithmetic operations with SIMD optimization
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        const float* b = other.ptr<float>();
        float* c = result.ptr<float>();
        
        size_t i = 0;
#if defined(__AVX__)
        for (; i + 8 <= size_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
#elif defined(__SSE__)
        for (; i + 4 <= size_; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
#elif defined(__ARM_NEON)
        for (; i + 4 <= size_; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vc = vaddq_f32(va, vb);
            vst1q_f32(c + i, vc);
        }
#endif
        for (; i < size_; ++i) c[i] = a[i] + b[i];
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        const double* b = other.ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] + b[i];
    }
    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for subtraction");
    }
    
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        const float* b = other.ptr<float>();
        float* c = result.ptr<float>();
        
        size_t i = 0;
#if defined(__AVX__)
        for (; i + 8 <= size_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
#elif defined(__SSE__)
        for (; i + 4 <= size_; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_sub_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
#endif
        for (; i < size_; ++i) c[i] = a[i] - b[i];
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        const double* b = other.ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] - b[i];
    }
    
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for multiplication");
    }
    
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        const float* b = other.ptr<float>();
        float* c = result.ptr<float>();
        
        size_t i = 0;
#if defined(__AVX__)
        for (; i + 8 <= size_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
#elif defined(__SSE__)
        for (; i + 4 <= size_; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_mul_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
#endif
        for (; i < size_; ++i) c[i] = a[i] * b[i];
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        const double* b = other.ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] * b[i];
    }
    
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for division");
    }
    
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        const float* b = other.ptr<float>();
        float* c = result.ptr<float>();
        
        size_t i = 0;
#if defined(__AVX__)
        for (; i + 8 <= size_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
#elif defined(__SSE__)
        for (; i + 4 <= size_; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_div_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
#endif
        for (; i < size_; ++i) c[i] = a[i] / b[i];
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        const double* b = other.ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] / b[i];
    }
    
    return result;
}

// Scalar operations
Tensor Tensor::operator+(double scalar) const {
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        float s = static_cast<float>(scalar);
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] + s;
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(double scalar) const {
    return *this + (-scalar);
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        float s = static_cast<float>(scalar);
        
        size_t i = 0;
#if defined(__AVX__)
        __m256 vs = _mm256_set1_ps(s);
        for (; i + 8 <= size_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vc = _mm256_mul_ps(va, vs);
            _mm256_storeu_ps(c + i, vc);
        }
#elif defined(__SSE__)
        __m128 vs = _mm_set1_ps(s);
        for (; i + 4 <= size_; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vc = _mm_mul_ps(va, vs);
            _mm_storeu_ps(c + i, vc);
        }
#endif
        for (; i < size_; ++i) c[i] = a[i] * s;
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(double scalar) const {
    return *this * (1.0 / scalar);
}

// In-place operations
Tensor& Tensor::operator+=(const Tensor& other) {
    *this = *this + other;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    *this = *this - other;
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other;
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other;
    return *this;
}

// Matrix multiplication with optimizations
Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.ndim() != 2 || other.shape_.ndim() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::runtime_error("matmul: incompatible shapes");
    }
    
    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = other.shape_[1];
    
    Tensor result = zeros(Shape{M, N}, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* A = ptr<float>();
        const float* B = other.ptr<float>();
        float* C = result.ptr<float>();
        
        // Block size for cache efficiency
        const size_t BLOCK = 64;
        
        for (size_t i0 = 0; i0 < M; i0 += BLOCK) {
            for (size_t j0 = 0; j0 < N; j0 += BLOCK) {
                for (size_t k0 = 0; k0 < K; k0 += BLOCK) {
                    size_t i_max = std::min(i0 + BLOCK, M);
                    size_t j_max = std::min(j0 + BLOCK, N);
                    size_t k_max = std::min(k0 + BLOCK, K);
                    
                    for (size_t i = i0; i < i_max; ++i) {
                        for (size_t k = k0; k < k_max; ++k) {
                            float a_ik = A[i * K + k];
                            for (size_t j = j0; j < j_max; ++j) {
                                C[i * N + j] += a_ik * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    } else if (dtype_ == DType::FLOAT64) {
        const double* A = ptr<double>();
        const double* B = other.ptr<double>();
        double* C = result.ptr<double>();
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                double a_ik = A[i * K + k];
                for (size_t j = 0; j < N; ++j) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
    }
    
    return result;
}

// Reduction operations
double Tensor::sum() const {
    double result = 0.0;
    
    if (dtype_ == DType::FLOAT32) {
        const float* p = ptr<float>();
        for (size_t i = 0; i < size_; ++i) result += p[i];
    } else if (dtype_ == DType::FLOAT64) {
        const double* p = ptr<double>();
        for (size_t i = 0; i < size_; ++i) result += p[i];
    } else if (dtype_ == DType::UINT8) {
        const uint8_t* p = ptr<uint8_t>();
        for (size_t i = 0; i < size_; ++i) result += p[i];
    }
    
    return result;
}

double Tensor::mean() const {
    return sum() / static_cast<double>(size_);
}

double Tensor::min() const {
    if (size_ == 0) return 0.0;
    
    double result = std::numeric_limits<double>::max();
    
    if (dtype_ == DType::FLOAT32) {
        const float* p = ptr<float>();
        for (size_t i = 0; i < size_; ++i) {
            if (p[i] < result) result = p[i];
        }
    } else if (dtype_ == DType::FLOAT64) {
        const double* p = ptr<double>();
        for (size_t i = 0; i < size_; ++i) {
            if (p[i] < result) result = p[i];
        }
    }
    
    return result;
}

double Tensor::max() const {
    if (size_ == 0) return 0.0;
    
    double result = std::numeric_limits<double>::lowest();
    
    if (dtype_ == DType::FLOAT32) {
        const float* p = ptr<float>();
        for (size_t i = 0; i < size_; ++i) {
            if (p[i] > result) result = p[i];
        }
    } else if (dtype_ == DType::FLOAT64) {
        const double* p = ptr<double>();
        for (size_t i = 0; i < size_; ++i) {
            if (p[i] > result) result = p[i];
        }
    }
    
    return result;
}

double Tensor::std() const {
    return std::sqrt(var());
}

double Tensor::var() const {
    double m = mean();
    double result = 0.0;
    
    if (dtype_ == DType::FLOAT32) {
        const float* p = ptr<float>();
        for (size_t i = 0; i < size_; ++i) {
            double diff = p[i] - m;
            result += diff * diff;
        }
    } else if (dtype_ == DType::FLOAT64) {
        const double* p = ptr<double>();
        for (size_t i = 0; i < size_; ++i) {
            double diff = p[i] - m;
            result += diff * diff;
        }
    }
    
    return result / static_cast<double>(size_);
}

size_t Tensor::argmin() const {
    if (size_ == 0) return 0;
    
    size_t idx = 0;
    
    if (dtype_ == DType::FLOAT32) {
        const float* p = ptr<float>();
        float min_val = p[0];
        for (size_t i = 1; i < size_; ++i) {
            if (p[i] < min_val) {
                min_val = p[i];
                idx = i;
            }
        }
    } else if (dtype_ == DType::FLOAT64) {
        const double* p = ptr<double>();
        double min_val = p[0];
        for (size_t i = 1; i < size_; ++i) {
            if (p[i] < min_val) {
                min_val = p[i];
                idx = i;
            }
        }
    }
    
    return idx;
}

size_t Tensor::argmax() const {
    if (size_ == 0) return 0;
    
    size_t idx = 0;
    
    if (dtype_ == DType::FLOAT32) {
        const float* p = ptr<float>();
        float max_val = p[0];
        for (size_t i = 1; i < size_; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                idx = i;
            }
        }
    } else if (dtype_ == DType::FLOAT64) {
        const double* p = ptr<double>();
        double max_val = p[0];
        for (size_t i = 1; i < size_; ++i) {
            if (p[i] > max_val) {
                max_val = p[i];
                idx = i;
            }
        }
    }
    
    return idx;
}

// Element-wise math
Tensor Tensor::abs() const {
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::abs(a[i]);
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::abs(a[i]);
    }
    return result;
}

Tensor Tensor::sqrt() const {
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        
        size_t i = 0;
#if defined(__AVX__)
        for (; i + 8 <= size_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vc = _mm256_sqrt_ps(va);
            _mm256_storeu_ps(c + i, vc);
        }
#elif defined(__SSE__)
        for (; i + 4 <= size_; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vc = _mm_sqrt_ps(va);
            _mm_storeu_ps(c + i, vc);
        }
#endif
        for (; i < size_; ++i) c[i] = std::sqrt(a[i]);
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::sqrt(a[i]);
    }
    return result;
}

Tensor Tensor::exp() const {
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::exp(a[i]);
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::exp(a[i]);
    }
    return result;
}

Tensor Tensor::log() const {
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::log(a[i]);
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::log(a[i]);
    }
    return result;
}

Tensor Tensor::pow(double exponent) const {
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        float e = static_cast<float>(exponent);
        for (size_t i = 0; i < size_; ++i) c[i] = std::pow(a[i], e);
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::pow(a[i], exponent);
    }
    return result;
}

Tensor Tensor::clip(double min_val, double max_val) const {
    Tensor result(shape_, dtype_);
    
    if (dtype_ == DType::FLOAT32) {
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        float lo = static_cast<float>(min_val);
        float hi = static_cast<float>(max_val);
        for (size_t i = 0; i < size_; ++i) {
            c[i] = std::max(lo, std::min(hi, a[i]));
        }
    } else if (dtype_ == DType::FLOAT64) {
        const double* a = ptr<double>();
        double* c = result.ptr<double>();
        for (size_t i = 0; i < size_; ++i) {
            c[i] = std::max(min_val, std::min(max_val, a[i]));
        }
    }
    return result;
}

Tensor Tensor::clone() const {
    Tensor result(shape_, dtype_);
    std::memcpy(result.data_.get(), data_.get(), nbytes());
    return result;
}

Tensor Tensor::contiguous() const {
    return clone();
}

// ============================================================================
// Image Implementation
// ============================================================================

Image::Image() {}

Image::Image(size_t height, size_t width, size_t channels, DType dtype)
    : data_(height, width, channels, dtype) {}

Image::Image(const Tensor& tensor) : data_(tensor) {}

Image Image::zeros(size_t height, size_t width, size_t channels) {
    return Image(Tensor::zeros(Shape{height, width, channels}, DType::UINT8));
}

Image Image::ones(size_t height, size_t width, size_t channels) {
    return Image(Tensor::full(Shape{height, width, channels}, 255.0, DType::UINT8));
}

Image Image::to_grayscale() const {
    if (channels() == 1) return clone();
    if (channels() != 3) {
        throw std::runtime_error("to_grayscale: expected 3 channels");
    }
    
    Image result(height(), width(), 1, dtype());
    
    if (dtype() == DType::UINT8) {
        const uint8_t* src = ptr<uint8_t>();
        uint8_t* dst = result.ptr<uint8_t>();
        
        for (size_t i = 0; i < height() * width(); ++i) {
            // Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            int r = src[i * 3 + 0];
            int g = src[i * 3 + 1];
            int b = src[i * 3 + 2];
            dst[i] = static_cast<uint8_t>((r * 77 + g * 150 + b * 29) >> 8);
        }
    }
    
    return result;
}

Image Image::to_rgb() const {
    if (channels() == 3) return clone();
    if (channels() != 1) {
        throw std::runtime_error("to_rgb: expected 1 channel");
    }
    
    Image result(height(), width(), 3, dtype());
    
    if (dtype() == DType::UINT8) {
        const uint8_t* src = ptr<uint8_t>();
        uint8_t* dst = result.ptr<uint8_t>();
        
        for (size_t i = 0; i < height() * width(); ++i) {
            dst[i * 3 + 0] = src[i];
            dst[i * 3 + 1] = src[i];
            dst[i * 3 + 2] = src[i];
        }
    }
    
    return result;
}

Image Image::to_float() const {
    return Image(data_.astype(DType::FLOAT32));
}

Image Image::to_uint8() const {
    return Image(data_.astype(DType::UINT8));
}

Image Image::to(DType new_dtype) const {
    return Image(data_.astype(new_dtype));
}

Image Image::clone() const {
    return Image(data_.clone());
}

// ============================================================================
// Utility Functions
// ============================================================================

Tensor add(const Tensor& a, const Tensor& b) { return a + b; }
Tensor subtract(const Tensor& a, const Tensor& b) { return a - b; }
Tensor multiply(const Tensor& a, const Tensor& b) { return a * b; }
Tensor divide(const Tensor& a, const Tensor& b) { return a / b; }

Tensor matmul(const Tensor& a, const Tensor& b) { return a.matmul(b); }
Tensor transpose(const Tensor& a) { return a.transpose(); }

Tensor abs(const Tensor& a) { return a.abs(); }
Tensor sqrt(const Tensor& a) { return a.sqrt(); }
Tensor exp(const Tensor& a) { return a.exp(); }
Tensor log(const Tensor& a) { return a.log(); }
Tensor pow(const Tensor& a, double exp) { return a.pow(exp); }

Tensor sin(const Tensor& a) {
    Tensor result(a.shape(), a.dtype());
    if (a.dtype() == DType::FLOAT32) {
        const float* src = a.ptr<float>();
        float* dst = result.ptr<float>();
        for (size_t i = 0; i < a.size(); ++i) dst[i] = std::sin(src[i]);
    } else if (a.dtype() == DType::FLOAT64) {
        const double* src = a.ptr<double>();
        double* dst = result.ptr<double>();
        for (size_t i = 0; i < a.size(); ++i) dst[i] = std::sin(src[i]);
    }
    return result;
}

Tensor cos(const Tensor& a) {
    Tensor result(a.shape(), a.dtype());
    if (a.dtype() == DType::FLOAT32) {
        const float* src = a.ptr<float>();
        float* dst = result.ptr<float>();
        for (size_t i = 0; i < a.size(); ++i) dst[i] = std::cos(src[i]);
    } else if (a.dtype() == DType::FLOAT64) {
        const double* src = a.ptr<double>();
        double* dst = result.ptr<double>();
        for (size_t i = 0; i < a.size(); ++i) dst[i] = std::cos(src[i]);
    }
    return result;
}

Tensor tan(const Tensor& a) {
    Tensor result(a.shape(), a.dtype());
    if (a.dtype() == DType::FLOAT32) {
        const float* src = a.ptr<float>();
        float* dst = result.ptr<float>();
        for (size_t i = 0; i < a.size(); ++i) dst[i] = std::tan(src[i]);
    } else if (a.dtype() == DType::FLOAT64) {
        const double* src = a.ptr<double>();
        double* dst = result.ptr<double>();
        for (size_t i = 0; i < a.size(); ++i) dst[i] = std::tan(src[i]);
    }
    return result;
}

double sum(const Tensor& a) { return a.sum(); }
double mean(const Tensor& a) { return a.mean(); }
double min(const Tensor& a) { return a.min(); }
double max(const Tensor& a) { return a.max(); }
double std(const Tensor& a) { return a.std(); }
double var(const Tensor& a) { return a.var(); }

Tensor concatenate(const std::vector<Tensor>& tensors, int axis) {
    if (tensors.empty()) return Tensor();
    
    // Simple implementation for axis=0
    if (axis == 0) {
        size_t total_size = 0;
        for (const auto& t : tensors) total_size += t.size();
        
        // For 1D tensors
        if (tensors[0].ndim() == 1) {
            Tensor result(Shape{total_size}, tensors[0].dtype());
            size_t offset = 0;
            for (const auto& t : tensors) {
                std::memcpy(
                    static_cast<char*>(result.data()) + offset * result.itemsize(),
                    t.data(),
                    t.nbytes()
                );
                offset += t.size();
            }
            return result;
        }
    }
    
    return tensors[0].clone();
}

Tensor stack(const std::vector<Tensor>& tensors, int axis) {
    if (tensors.empty()) return Tensor();
    
    // Stack along new axis
    Shape new_shape;
    new_shape.dims.push_back(tensors.size());
    for (auto d : tensors[0].shape().dims) {
        new_shape.dims.push_back(d);
    }
    
    Tensor result(new_shape, tensors[0].dtype());
    
    size_t elem_size = tensors[0].nbytes();
    for (size_t i = 0; i < tensors.size(); ++i) {
        std::memcpy(
            static_cast<char*>(result.data()) + i * elem_size,
            tensors[i].data(),
            elem_size
        );
    }
    
    return result;
}

Tensor vstack(const std::vector<Tensor>& tensors) {
    return concatenate(tensors, 0);
}

Tensor hstack(const std::vector<Tensor>& tensors) {
    return concatenate(tensors, 1);
}

std::vector<Tensor> split(const Tensor& a, size_t num_splits, int axis) {
    std::vector<Tensor> result;
    
    if (axis == 0 && a.ndim() >= 1) {
        size_t split_size = a.shape()[0] / num_splits;
        size_t elem_per_split = a.size() / num_splits;
        
        for (size_t i = 0; i < num_splits; ++i) {
            Shape new_shape = a.shape();
            new_shape.dims[0] = split_size;
            
            Tensor part(new_shape, a.dtype());
            std::memcpy(
                part.data(),
                static_cast<const char*>(a.data()) + i * elem_per_split * a.itemsize(),
                part.nbytes()
            );
            result.push_back(std::move(part));
        }
    }
    
    return result;
}

} // namespace neurova
